import asyncio
import os
import re

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, dynamic_prompt
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from app.context import Ctx
from app.custom_tools import build_custom_tools
from app.kb.store import KnowledgeBase
from app.kb.tools import build_kb_tools
from app.llm import agent_llm
from app.tools_s1 import S1_TOOLS

SYSTEM_PROMPT = """You are a friendly, concise assistant for backend developers. Answer directly; don't narrate your process or name tools unless asked.

Tools: weather, web_fetch, SQL over "chinook" (music store) and "northwind" (trading company), Altair charts, OpenML + pandas, tasks and notes (kept per user), and the user's own documents.

KNOWLEDGE BASE
- The KNOWLEDGE BASE block below lists the user's own documents (overview, keywords, one line per file).
- If a question could be answered from them, call search_knowledge_base FIRST. If the user names a document ("in the policy pdf"), pass its exact file name from the list as doc_name. Don't search for unrelated questions (weather, SQL, OpenML) or when it is empty.
- When the user uploads a file, the message gives a file_id: call ingest_document with it, then say in one sentence what the document contains.
- Cite each claim from the documents by copying the bracketed source of the passage you used, exactly, e.g. [Handbook.pdf › 2.1 Refund policy, p.3-4]. Never cite by passage number or invent a source.
- Never say a document doesn't mention something without searching it first. If results say LOW CONFIDENCE or are empty, say the documents don't cover it; you may add a general-knowledge answer, labelled as such.

ERROR HANDLING
- A tool result starting with ERROR[...] means the step failed: tell the user plainly what happened and the next step, in your own words. Never claim success; never show codes, ids, paths or traces. Retry at most once, and only if the next step says it can help.

DATA AND CHARTS
- Prefer tools over guessing. Pick chinook or northwind by topic. For OpenML: openml_download, then pandas_query on table "df".
- sql_query/pandas_query return a data_ref: pass it to make_chart, never retype rows. Let make_chart aggregate (Altair shorthand such as "mean(Sales):Q").
- Charts appear in the UI automatically: don't mention file paths or links."""

CHART_RE = re.compile(r"(chart_[0-9a-f]+\.html)")


OLD_RESULT_CHARS = 300


def shrink_old_tool_results(messages: list) -> list:
    """Context hygiene: tool results from EARLIER turns shrink to a stub. The assistant's answer already used them,
    and a thread that keeps every passage and query result grows past the model's per-request limit within a few turns."""
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    return [
        m.model_copy(update={"content": f"{m.content[:OLD_RESULT_CHARS]} ...[older tool result shortened]"})
        if isinstance(m, ToolMessage) and i < last_human and isinstance(m.content, str) and len(m.content) > OLD_RESULT_CHARS else m
        for i, m in enumerate(messages)
    ]


class ShrinkOldToolResults(AgentMiddleware):
    """Applies shrink_old_tool_results to what is SENT to the model; the stored thread is unchanged."""

    def wrap_model_call(self, request, handler):
        return handler(request.override(messages=shrink_old_tool_results(request.messages)))

    async def awrap_model_call(self, request, handler):
        return await handler(request.override(messages=shrink_old_tool_results(request.messages)))


async def mcp_tools():
    """MCP tools become ordinary LangChain tools. The notes and tasks server (src/app/mcp_server/server.py) is reached over HTTP at
    MCP_URL (docker compose and CI start it). Retries a few times: the server may still be starting.

    TODO(task 9): make every MCP call run AS THE LOGGED-IN USER. The server reads `Authorization: Bearer <token>`. Write a tool
    interceptor (langchain-mcp-adapters: `async def interceptor(request, handler)`; `request.runtime.context` is this request's Ctx,
    `request.override(headers={...})` returns a request with extra HTTP headers) and pass it as tool_interceptors=[...] below.
    The token comes from Ctx (task 9: add `token` to Ctx, /agent fills it from the Authorization header). The model never sees it.
    """
    url = os.getenv("MCP_URL", "http://localhost:9000/mcp")
    client = MultiServerMCPClient({"tasks": {"url": url, "transport": "streamable_http"}})  # TODO(task 9): tool_interceptors=[...]
    for attempt in range(15):
        try:
            return await client.get_tools()
        except Exception:
            if attempt == 14:
                raise
            await asyncio.sleep(2)



async def build_agent(kb: KnowledgeBase, model=agent_llm, checkpointer=None):
    """Build the Document Agent with tools, dynamic prompt, middleware and memory."""

    tools = [
        *S1_TOOLS,
        *build_kb_tools(kb),
        *await mcp_tools(),
        *build_custom_tools(kb),
    ]

    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        session_id = request.runtime.context.session_id

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"{kb.prompt_block(session_id)}"
        )

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[
            prompt,
            ShrinkOldToolResults(),
        ],
        context_schema=Ctx,
        checkpointer=checkpointer,
    )

    return agent, tools

def clean_answer(text: str) -> str:
    """Models sometimes swap '-' for typographic hyphens (U+2010/U+2011) inside file names, or wrap a citation in CJK
    brackets 【…】, so it would no longer match the stored document name / citation format. Normalise them back."""
    return text.replace("\u2010", "-").replace("\u2011", "-").replace("\u3010", "[").replace("\u3011", "]")


def steps_from(messages: list) -> list[dict]:
    """Pair each tool call with its result so the UI can show the reasoning trail."""
    results = {m.tool_call_id: str(m.content) for m in messages if isinstance(m, ToolMessage)}
    return [
        {"tool": c["name"], "args": c["args"], "result": results.get(c["id"], "")[:700]}
        for m in messages
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    ]


def charts_from(messages: list) -> list[str]:
    """make_chart writes an HTML file; expose it under the API's /charts mount."""
    return [
        f"/charts/{m.group(1)}"
        for t in messages
        if isinstance(t, ToolMessage) and t.name == "make_chart" and (m := CHART_RE.search(str(t.content)))
    ]
