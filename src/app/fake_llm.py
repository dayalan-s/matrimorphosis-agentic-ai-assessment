"""A scripted chat model for automated grading (LLM_MODE=fake). Deterministic, no network.

It decides what the agent does from the last user message, so a test can exercise the real agent loop, tools,
checkpointer and knowledge base without an API key:

    ingest <file_id>          -> calls ingest_document(source=<file_id>)
    search <query>            -> calls search_knowledge_base(query=<query>)
    call <tool> <json args>   -> calls any registered tool with those arguments
    recall                    -> answers with every earlier user message in this thread (proves chat memory)
    showprompt                -> answers with the system prompt the model was given (proves the dynamic prompt)
    anything else             -> "ok: <text>"
After a tool runs, the model answers "Tool result: <tool output>".
"""
import json
import re
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda


def _tool_call(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": uuid.uuid4().hex[:12], "type": "tool_call"}])


class ScriptedChatModel(BaseChatModel):
    model_name: str = "scripted-fake"

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    def _reply(self, messages: list) -> AIMessage:
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"Tool result: {str(messages[-1].content)[:600]}")
        humans = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
        text = humans[-1].strip() if humans else ""
        word, _, rest = text.partition(" ")
        if word == "ingest":
            return _tool_call("ingest_document", {"source": rest.strip()})
        if word == "search":
            return _tool_call("search_knowledge_base", {"query": rest.strip()})
        if word == "call":
            name, _, raw = rest.partition(" ")
            try:
                args = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                return AIMessage(content="Could not parse the tool arguments as JSON.")
            return _tool_call(name.strip(), args)
        if word == "showprompt":
            return AIMessage(content="SYSTEM PROMPT: " + "\n".join(str(m.content) for m in messages if isinstance(m, SystemMessage)))
        if word == "recall":
            return AIMessage(content="You said: " + " | ".join(humans[:-1]))
        return AIMessage(content=f"ok: {text[:200]}")

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda prompt: _structured(schema, str(prompt)))


def _structured(schema, prompt: str) -> Any:
    """Plausible, deterministic answers for the ingestion pipeline's three structured calls."""
    from app.kb.models import DocProfile, KBProfile, SectionSpan, WindowSections

    if schema is WindowSections:
        blocks = [(int(i), t) for i, t in re.findall(r"^\[(\d+)\] (.*)$", prompt, re.M)] or [(0, "")]
        starts = [i for i, t in blocks if t.lstrip().startswith("#")]
        if not starts or starts[0] != blocks[0][0]:
            starts = [blocks[0][0], *starts]  # text before the first heading is its own section
        text = dict(blocks)
        ends = [*[s - 1 for s in starts[1:]], blocks[-1][0]]
        spans = []
        for start, end in zip(starts, ends):
            head = text[start].lstrip()
            level = len(head) - len(head.lstrip("#"))
            spans.append(SectionSpan(heading=head.lstrip("# ").strip()[:60] or "Content", level=max(level, 1), start_block=start, end_block=end,
                                     keywords=["demo", "document"], summary="Scripted summary of this part."))
        return WindowSections(sections=spans)
    if schema is DocProfile:
        name = (re.search(r"File: (.+)", prompt) or [None, "document"])[1].strip().rsplit(".", 1)[0]
        return DocProfile(title=name, doc_type="document", language="English", summary=f"Scripted profile of {name}.", keywords=["demo", "document"])
    return KBProfile(overview="Scripted knowledge base overview.", keywords=["demo", "document"])
