"""Tasks 7 and 8: four tools. Names, arguments, outputs, errors, metric and log names are FIXED (the autograder checks them).

The heavy parts are PROVIDED (kb/store.py: stats / outline / delete_document, kb/pipeline.py: ingest_file).
You write the tool wrappers: signature + docstring (the docstring is the prompt: say WHEN to use it, describe each argument),
validation, ToolException with `ERROR[CODE]: what happened. Next step: what to do.`, the metric, the log line, `handle_tool_error = True`.
All four are `async def` (blocking calls go through asyncio.to_thread). They act on the CALLING USER's knowledge base:
`runtime.context.user_id` (task 6), never on another user's. build_custom_tools(kb) returns all four.
Look at kb/tools.py for the pattern (ToolRuntime, ToolException, handle_tool_error).

---------------------------------------------------------------------------------- task 7.1  kb_stats(top_keywords: int = 5) -> str
  output  "<D> documents, <S> sections, <C> chunks. Top keywords: k1, k2, ..."   from kb.stats(user_id); at most `top_keywords` keywords
          empty knowledge base: "The knowledge base is empty."
  error   top_keywords outside 1..20  ->  ERROR[INVALID_INPUT]

---------------------------------------------------------------------------------- task 7.2  document_outline(doc_name: str) -> str
  The structure of one document: use it when the user asks what a document contains, how it is organised, or which section covers
  something (then search that section). Resolve doc_name with kb.resolve_doc_name(user_id, doc_name), find the doc_id in
  kb.documents(user_id), read the sections with kb.outline(user_id, doc_id).
  output  first line   "Outline of '<exact file name>' (<N> sections):"          (always the word "sections")
          then one line per section, in reading order, 1-based position:
              "<i>. [L<level>] <heading> (p.<a>-<b>): <summary>"
          the page part is " (p.<a>-<b>)" when page_start != page_end, " (p.<a>)" when they are equal, and left out when the
          document has no pages (page_start is None); then ": <summary>"
  error   no such document in THIS user's knowledge base  ->  ERROR[UNKNOWN_DOCUMENT]  (list the valid names in the message)

---------------------------------------------------------------------------------- task 7.3  observability, every call, ok AND error
  metric  kb_stats: counter "kbstats.calls"    document_outline: counter "outline.calls"
          record with attributes {"status": "ok" | "error"}  (Prometheus: agent_outline_calls_total, ...)
          metrics.get_meter("agent").create_counter(...)   (see how app/telemetry.py creates TOOL_CALLS)
  log     log.info("kb_stats_done", tool="kb_stats", status=..., duration_ms=<int>)
          log.info("document_outline_done", tool="document_outline", status=..., duration_ms=<int>, sections=<int, 0 on error>)
          (`from app.telemetry import log`; never log document text)
  Grafana (recommended, not graded): src/app/dashboards/outline.json with a panel for outline.calls by status and a logs panel
          ({service_name="extraction-agent"} | json | event="document_outline_done").

---------------------------------------------------------------------------------- task 8.1  delete_document(doc_name: str) -> str
  Removes one document from the user's knowledge base. doc_name is a file name from the KNOWLEDGE BASE list.
  Resolve it with kb.resolve_doc_name(user_id, doc_name); find the document's doc_id in kb.documents(user_id);
  delete with kb.delete_document(user_id, doc_id).
  output  "Deleted '<exact file name>'. <N> documents left."
  error   no such document in THIS user's knowledge base  ->  ERROR[UNKNOWN_DOCUMENT]  (list the valid names in the message)

---------------------------------------------------------------------------------- task 8.2  update_document(doc_name: str, source: str) -> str
  Replaces a document with a new version the user uploaded. source = the file_id of the NEW upload (like ingest_document).
  Order matters: ingest the new file FIRST under the OLD file name (kb/pipeline.ingest_file, files.resolve), and only when that
  succeeded delete the old version by its doc_id: a failed update must leave the old document untouched.
  If the new file is identical to a document already in the knowledge base ingest_file returns it as `existing`: change nothing.
  output  "Updated '<exact file name>': <S> sections, <C> chunks."      identical content: "'<name>' is already up to date."
  errors  unknown doc_name -> ERROR[UNKNOWN_DOCUMENT];  bad/unknown file_id -> use app.errors.to_tool_exception on the DocError
          (it yields ERROR[FILE_NOT_FOUND]).
"""
from langchain_core.tools import tool  # noqa: F401  (you will need it)

from app.kb.store import KnowledgeBase


def build_custom_tools(kb: KnowledgeBase) -> list:
    # TODO(tasks 7 and 8): define kb_stats, document_outline, delete_document and update_document here and return all four
    return []
