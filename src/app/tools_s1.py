"""The Session 1 tools: weather, web_fetch, sql_query, make_chart, openml_*, pandas_query.

TODO(task 2): the API is async (FastAPI + an async agent). A BLOCKING call inside a tool freezes every other request
while it runs. Convert these four tools to `async def` and keep the event loop free:
  get_weather            plain function -> async def (no I/O, but keep the whole tool set async)
  web_fetch              trafilatura.fetch_url blocks: download with httpx.AsyncClient (timeout=15, follow_redirects=True),
                         then run the CPU-heavy trafilatura.extract in asyncio.to_thread
  sql_query              sqlite3 blocks: run the query in asyncio.to_thread (or use aiosqlite, already installed)
  search_knowledge_base  in kb/tools.py: embedding + Qdrant block -> asyncio.to_thread(kb.search, ...)
The output of every tool must stay exactly the same. The autograder checks /debug/tools (async=true) and scans these
functions for blocking calls (time.sleep, httpx.get/Client, requests, sqlite3.connect, trafilatura.fetch_url).

Only the paths moved (data/ in this project, cache under 'session5-agent'). Charts are written to CHARTS_DIR
and served by the API at /charts so the Gradio UI can show them.
"""
import json
import os
import re
import sqlite3
import uuid
from pathlib import Path

import altair as alt
import openml
import pandas as pd
import platformdirs
from altair.utils.core import parse_shorthand
import sqlglot
import trafilatura
from langchain_core.tools import tool
from sqlglot import exp

BASE_DIR = Path(__file__).resolve().parents[2]  # repo root (data/ lives here)
CACHE_DIR = Path(platformdirs.user_cache_dir("session5-agent"))

# Generated at runtime, safe to wipe — goes in the OS cache dir, not the repo.
CHARTS_DIR = CACHE_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

DATABASES = {
    "chinook": BASE_DIR / "data" / "chinook.sqlite",  # music store: Artist/Album/Track/Invoice
    "northwind": BASE_DIR / "data" / "northwind.sqlite",  # trading co: Customers/Orders/Products/Suppliers
}



# ---------------------------------------------------------------------------
# In-process cache: lets sql_query/pandas_query hand make_chart a short
# reference instead of making the model retype a whole dataset as a JSON
# string argument. Retyping large/precise data as tool-call arguments is a
# common LLM failure mode ("Failed to parse tool call arguments as JSON") —
# this makes it structurally impossible by removing the need to do it at all.
# ---------------------------------------------------------------------------
_DATA_CACHE: dict[str, list[dict]] = {}


def _cache_rows(rows: list[dict]) -> str:
    ref = uuid.uuid4().hex[:8]
    _DATA_CACHE[ref] = rows
    return ref


def reject_unless_select_only(query: str) -> str | None:
    """Parse `query` with sqlglot and return an error string unless it is exactly
    one read-only SELECT/UNION statement. Returns None when the query is safe.

    A parser-based check (vs. a naive string prefix check) catches things a prefix
    check misses: a hidden second statement after ';', a CTE ("WITH ... SELECT"),
    or SQL comments masking a DROP/DELETE.
    """
    try:
        statements = sqlglot.parse(query)
    except sqlglot.errors.ParseError as e:
        return f"SQL parse error: {e}"

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return "Rejected: exactly one SQL statement is allowed."
    if not isinstance(statements[0], (exp.Select, exp.Union)):
        return f"Rejected: only SELECT statements are allowed, got {type(statements[0]).__name__}."
    return None


# ---------------------------------------------------------------------------
# Tool 1 — weather (toy, no real API — mirrors the slide's exact example)
# ---------------------------------------------------------------------------
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location.

    Args:
        location: City name
    """
    return f"Weather in {location}: Sunny, 72F"


# ---------------------------------------------------------------------------
# Tool 2 — web fetch -> clean markdown
# ---------------------------------------------------------------------------
@tool
def web_fetch(url: str) -> str:
    """Fetch a web page and return its main content as clean Markdown,
    with navigation, ads, and boilerplate stripped out.

    Args:
        url: Full URL of the page to fetch (e.g. https://example.com/article)
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return f"Could not fetch {url}"
    markdown = trafilatura.extract(downloaded, output_format="markdown", with_metadata=True)
    if not markdown:
        return f"Fetched {url} but found no extractable article content."
    return markdown[:8000]  # keep the demo output readable


# ---------------------------------------------------------------------------
# Tool 3 — SQL query against a toy database (read-only)
# ---------------------------------------------------------------------------
@tool
def sql_query(database: str, query: str) -> str:
    """Run a read-only SQL SELECT query against one of the toy databases.

    Returns a JSON object: {"data_ref": ..., "row_count": ..., "preview": [...]}.
    "preview" is just the first rows for you to read/summarize — to chart the
    result, pass "data_ref" straight into make_chart's data_ref argument. Never
    retype/regenerate the full row data yourself as a make_chart argument; for
    anything more than a couple of rows that reliably produces a malformed tool
    call (the JSON-in-JSON escaping is exactly the kind of thing LLMs get wrong).

    Available databases:
      - "chinook": a music store — Artist, Album, Track, Customer, Invoice, InvoiceLine, Genre, ...
      - "northwind": a trading company — Customers, Orders, "Order Details", Products,
        Suppliers, Employees, Categories, Shippers, ... (note: quote "Order Details",
        it has a space in the table name; also group by CustomerID, not just
        CompanyName — a couple of distinct customers share the placeholder name "IT")

    Args:
        database: one of "chinook", "northwind"
        query: A single SQL SELECT statement. No INSERT/UPDATE/DELETE/DROP allowed.
    """
    db_path = DATABASES.get(database)
    if db_path is None:
        return f"Unknown database '{database}'. Choose one of: {', '.join(DATABASES)}"

    rejection = reject_unless_select_only(query)
    if rejection:
        return rejection
    stripped = query.strip().rstrip(";")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(stripped).fetchall()
    except sqlite3.Error as e:
        return f"SQL error: {e}"
    finally:
        con.close()
    result = [dict(row) for row in rows[:200]]  # cap rows for the demo
    ref = _cache_rows(result)
    return json.dumps({"data_ref": ref, "row_count": len(result), "preview": result[:20]}, default=str)


# ---------------------------------------------------------------------------
# Tool 4 — Altair chart from tabular data (generic encoding API)
# ---------------------------------------------------------------------------
ALLOWED_MARKS = {"bar", "line", "point", "area", "circle", "square", "tick", "rect", "arc"}
ALLOWED_CHANNELS = {"x", "y", "color", "size", "shape", "opacity", "column", "row", "tooltip", "text", "order", "detail"}


@tool
def make_chart(
    mark: str,
    encoding: dict[str, str | list[str]],
    data_ref: str = "",
    data_json: str = "",
    title: str = "",
) -> str:
    """Create a chart using Altair's own encoding shorthand — the same API a data
    scientist would write directly:
        alt.Chart(df).mark_bar().encode(x="Department:N", y="mean(Sales):Q")

    Always prefer data_ref: pass the "data_ref" value returned by sql_query or
    pandas_query directly. Only use data_json for a tiny, hand-built dataset — never
    retype a query's actual rows into data_json yourself, that fails easily for
    anything more than a couple of rows.

    Args:
        mark: one of bar, line, point, area, circle, square, tick, rect, arc
        encoding: {channel: shorthand} using Altair's "field:type" or
            "aggregate(field):type" shorthand syntax — Altair computes the
            aggregate itself, you never need to pre-aggregate in SQL first.
            Allowed channels: x, y, color, size, shape, opacity, column, row,
            tooltip (may be a list of shorthand strings), text, order, detail.
            Type codes: Q=quantitative, N=nominal, O=ordinal, T=temporal.
            Aggregates: mean(), sum(), count(), median(), min(), max(), etc.
            Examples:
              {"x": "Department:N", "y": "mean(Sales):Q"}
                -> bar chart of average Sales per Department, computed automatically
              {"x": "sepallength:Q", "y": "petallength:Q", "color": "class:N"}
                -> scatter plot colored by class
        data_ref: the "data_ref" from a prior sql_query/pandas_query call (preferred)
        data_json: JSON array of row objects — only for tiny, hand-built data
        title: optional chart title
    """
    if mark not in ALLOWED_MARKS:
        return f"Rejected: unknown mark '{mark}'. Choose one of: {', '.join(sorted(ALLOWED_MARKS))}"

    bad_channels = set(encoding) - ALLOWED_CHANNELS
    if bad_channels:
        return f"Rejected: unknown encoding channel(s) {sorted(bad_channels)}. Allowed: {', '.join(sorted(ALLOWED_CHANNELS))}"

    if data_ref:
        rows = _DATA_CACHE.get(data_ref)
        if rows is None:
            return f"Unknown data_ref '{data_ref}' (it may be from a different session) — re-run the query first."
    elif data_json:
        rows = json.loads(data_json)
    else:
        return "Rejected: provide either data_ref (preferred) or data_json."

    df = pd.DataFrame(rows)

    # Altair's own shorthand parser doesn't check the field actually exists in the
    # data — it just silently renders an empty chart — so check that ourselves.
    for channel, shorthand in encoding.items():
        for sh in (shorthand if isinstance(shorthand, list) else [shorthand]):
            field = parse_shorthand(sh, df).get("field")
            if field is not None and field not in df.columns:
                return f"Rejected: '{channel}' references unknown field '{field}'. Available columns: {list(df.columns)}"

    try:
        chart = getattr(alt.Chart(df), f"mark_{mark}")().encode(**encoding).properties(title=title or "")
    except Exception as e:
        return f"Chart error: {e}"

    out_path = CHARTS_DIR / f"chart_{uuid.uuid4().hex[:10]}.html"
    chart.save(out_path)
    return f"Chart saved as {out_path.name} (full path: {out_path})"


# ---------------------------------------------------------------------------
# Tool 5 — OpenML dataset search & download
# ---------------------------------------------------------------------------
@tool
def openml_search(keyword: str, limit: int = 5) -> str:
    """Search openml.org for open datasets matching a keyword.
    Returns a JSON list of {id, name, version, num_instances, num_features}.

    Args:
        keyword: search term, e.g. "iris" or "titanic"
        limit: max number of results to return
    """
    datasets = openml.datasets.list_datasets(data_name=keyword, output_format="dataframe")
    matches = datasets.head(limit)
    cols = [c for c in ["did", "name", "version", "NumberOfInstances", "NumberOfFeatures"] if c in matches.columns]
    return matches[cols].to_json(orient="records")


# Re-downloadable dataset cache — same OS cache dir as CHARTS_DIR, not the repo.
OPENML_CACHE_DIR = CACHE_DIR / "openml_cache"
OPENML_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@tool
def openml_download(dataset_id: int) -> str:
    """Download an OpenML dataset by its numeric id, cache it locally as CSV, and
    return a JSON summary (shape, column names, first rows, and the cache_path
    needed by pandas_query to actually query the data afterward).

    Args:
        dataset_id: the OpenML dataset id (the "did" from openml_search)
    """
    dataset = openml.datasets.get_dataset(dataset_id)
    df, *_ = dataset.get_data()
    cache_path = OPENML_CACHE_DIR / f"{dataset_id}.csv"
    df.to_csv(cache_path, index=False)
    summary = {
        "name": dataset.name,
        "dataset_id": dataset_id,
        "cache_path": str(cache_path),
        "shape": list(df.shape),
        "columns": list(df.columns)[:30],
        "head": json.loads(df.head(5).to_json(orient="records")),
    }
    return json.dumps(summary, default=str)


@tool
def pandas_query(cache_path: str, query: str) -> str:
    """Load a dataset previously downloaded by openml_download and run a real SQL
    SELECT query against it, using pandas' own SQL support (DataFrame.to_sql +
    pd.read_sql_query over an in-memory SQLite connection). The table is always
    called "df".

    Returns a JSON object: {"data_ref": ..., "row_count": ..., "preview": [...]}.
    "preview" is just the first rows for you to read/summarize — to chart the
    result, pass "data_ref" straight into make_chart's data_ref argument. Never
    retype/regenerate the full row data yourself as a make_chart argument.

    Args:
        cache_path: local CSV path from openml_download's "cache_path" field
        query: a single SQL SELECT statement against a table named "df",
               e.g. "SELECT class, AVG(petallength) AS avg_petal FROM df GROUP BY class ORDER BY avg_petal DESC"
    """
    path = Path(cache_path).resolve()
    if OPENML_CACHE_DIR.resolve() not in path.parents or not path.exists():
        return f"Rejected: {cache_path} is not a known dataset cache file."

    rejection = reject_unless_select_only(query)
    if rejection:
        return rejection
    stripped = query.strip().rstrip(";")

    df = pd.read_csv(path)
    con = sqlite3.connect(":memory:")
    try:
        df.to_sql("df", con, index=False)
        result_df = pd.read_sql_query(stripped, con)
    except (sqlite3.Error, pd.errors.DatabaseError) as e:
        return f"SQL error: {e}"
    finally:
        con.close()

    result = json.loads(result_df.head(200).to_json(orient="records"))
    ref = _cache_rows(result)
    return json.dumps({"data_ref": ref, "row_count": len(result), "preview": result[:20]})


S1_TOOLS = [get_weather, web_fetch, sql_query, make_chart, openml_search, openml_download, pandas_query]
