
"""The Session 1 tools: weather, web_fetch, sql_query, make_chart, openml_*, pandas_query.

TODO(task 2): the API is async (FastAPI + an async agent). A BLOCKING call inside a tool freezes every other request
while it runs. Convert these four tools to `async def` and keep the event loop free:
  get_weather            plain function -> async def (no I/O, but keep the whole tool set async)
  web_fetch              download with httpx.AsyncClient, then run trafilatura.extract in asyncio.to_thread
  sql_query              run the SQLite query in asyncio.to_thread
  search_knowledge_base  in kb/tools.py: embedding + Qdrant block -> asyncio.to_thread(kb.search, ...)
"""

import asyncio
import json
import os
import re
import sqlite3
import uuid
from pathlib import Path

import altair as alt
import httpx
import openml
import pandas as pd
import platformdirs
import sqlglot
import trafilatura
from altair.utils.core import parse_shorthand
from langchain_core.tools import tool
from sqlglot import exp


BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(platformdirs.user_cache_dir("session5-agent"))

# Generated at runtime, safe to wipe — goes in the OS cache dir, not the repo.
CHARTS_DIR = CACHE_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

DATABASES = {
    "chinook": BASE_DIR / "data" / "chinook.sqlite",
    "northwind": BASE_DIR / "data" / "northwind.sqlite",
}


# ---------------------------------------------------------------------------
# In-process cache: lets sql_query/pandas_query hand make_chart a short
# reference instead of making the model retype a whole dataset as JSON.
# ---------------------------------------------------------------------------

_DATA_CACHE: dict[str, list[dict]] = {}


def _cache_rows(rows: list[dict]) -> str:
    ref = uuid.uuid4().hex[:8]
    _DATA_CACHE[ref] = rows
    return ref


def reject_unless_select_only(query: str) -> str | None:
    """Return an error unless query is exactly one read-only SELECT/UNION."""
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
# Tool 1 — weather
# ---------------------------------------------------------------------------

@tool
async def get_weather(location: str) -> str:
    """Get current weather for a location.

    Args:
        location: City name
    """
    return f"Weather in {location}: Sunny, 72F"


# ---------------------------------------------------------------------------
# Tool 2 — web fetch -> clean markdown
# ---------------------------------------------------------------------------

@tool
async def web_fetch(url: str) -> str:
    """Fetch a web page and return its main content as clean Markdown,
    with navigation, ads, and boilerplate stripped out.

    Args:
        url: Full URL of the page to fetch (e.g. https://example.com/article)
    """
    try:
        async with httpx.AsyncClient(
            timeout=15,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            downloaded = response.text

    except httpx.HTTPError:
        return f"Could not fetch {url}"

    if not downloaded:
        return f"Could not fetch {url}"

    markdown = await asyncio.to_thread(
        trafilatura.extract,
        downloaded,
        output_format="markdown",
        with_metadata=True,
    )

    if not markdown:
        return f"Fetched {url} but found no extractable article content."

    return markdown[:8000]


# ---------------------------------------------------------------------------
# Tool 3 — SQL query against a toy database (read-only)
# ---------------------------------------------------------------------------

def _run_sql_query(db_path: Path, stripped: str) -> str:
    """Run SQLite work outside the asyncio event loop."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    try:
        rows = con.execute(stripped).fetchall()

    except sqlite3.Error as e:
        return f"SQL error: {e}"

    finally:
        con.close()

    result = [dict(row) for row in rows[:200]]
    ref = _cache_rows(result)

    return json.dumps(
        {
            "data_ref": ref,
            "row_count": len(result),
            "preview": result[:20],
        },
        default=str,
    )


@tool
async def sql_query(database: str, query: str) -> str:
    """Run a read-only SQL SELECT query against one of the toy databases.

    Returns a JSON object: {"data_ref": ..., "row_count": ..., "preview": [...]}.
    "preview" is just the first rows for you to read/summarize — to chart the
    result, pass "data_ref" straight into make_chart's data_ref argument.

    Available databases:
      - "chinook": a music store — Artist, Album, Track, Customer, Invoice, ...
      - "northwind": a trading company — Customers, Orders, Products, Suppliers, ...

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

    return await asyncio.to_thread(
        _run_sql_query,
        db_path,
        stripped,
    )


# ---------------------------------------------------------------------------
# Tool 4 — Altair chart from tabular data
# ---------------------------------------------------------------------------

ALLOWED_MARKS = {
    "bar", "line", "point", "area", "circle",
    "square", "tick", "rect", "arc",
}

ALLOWED_CHANNELS = {
    "x", "y", "color", "size", "shape", "opacity",
    "column", "row", "tooltip", "text", "order", "detail",
}


@tool
def make_chart(
    mark: str,
    encoding: dict[str, str | list[str]],
    data_ref: str = "",
    data_json: str = "",
    title: str = "",
) -> str:
    """Create a chart using Altair's own encoding shorthand.

    Args:
        mark: one of bar, line, point, area, circle, square, tick, rect, arc
        encoding: channel and Altair shorthand mapping
        data_ref: the data_ref from a prior sql_query/pandas_query call
        data_json: JSON array of row objects for a tiny hand-built dataset
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

    # Check that referenced fields exist in the data.
    for channel, shorthand in encoding.items():
        for sh in (shorthand if isinstance(shorthand, list) else [shorthand]):
            field = parse_shorthand(sh, df).get("field")

            if field is not None and field not in df.columns:
                return f"Rejected: '{channel}' references unknown field '{field}'. Available columns: {list(df.columns)}"

    try:
        chart = (
            getattr(alt.Chart(df), f"mark_{mark}")()
            .encode(**encoding)
            .properties(title=title or "")
        )

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

    Args:
        keyword: search term, e.g. "iris" or "titanic"
        limit: max number of results to return
    """
    datasets = openml.datasets.list_datasets(
        data_name=keyword,
        output_format="dataframe",
    )

    matches = datasets.head(limit)

    cols = [
        c for c in [
            "did",
            "name",
            "version",
            "NumberOfInstances",
            "NumberOfFeatures",
        ]
        if c in matches.columns
    ]

    return matches[cols].to_json(orient="records")


# Re-downloadable dataset cache.
OPENML_CACHE_DIR = CACHE_DIR / "openml_cache"
OPENML_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@tool
def openml_download(dataset_id: int) -> str:
    """Download an OpenML dataset by its numeric id and cache it locally.

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
    """Load a downloaded dataset and run a SQL SELECT query against table df.

    Args:
        cache_path: local CSV path from openml_download
        query: a single SQL SELECT statement against a table named df
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

    return json.dumps({
        "data_ref": ref,
        "row_count": len(result),
        "preview": result[:20],
    })


S1_TOOLS = [
    get_weather,
    web_fetch,
    sql_query,
    make_chart,
    openml_search,
    openml_download,
    pandas_query,
]
