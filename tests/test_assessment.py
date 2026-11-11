"""Black-box tests: they only talk HTTP to your running API (a scripted model replaces the LLM, see src/app/fake_llm.py).

phase1 runs first; then the CI restarts your container; then phase2 checks that state survived.
Run them yourself against `docker compose up`:   BASE_URL=http://localhost:8000 uv run pytest tests/ -m phase1
(the restart-dependent tests need the restart too: `bash scripts/local_eval.sh` does the whole thing exactly like CI)

Until you implement task 5, POST /users does not exist: the other tests then run as an anonymous user, so each task is graded on its own.
"""
import ast
import json
import os
import re
import uuid
from pathlib import Path

import httpx
import pytest

RESULTS = Path("results")
STATE = RESULTS / "state.json"

REQUIRED_TOOLS = {"get_weather", "web_fetch", "sql_query", "make_chart", "openml_search", "openml_download", "pandas_query",
                  "ingest_document", "search_knowledge_base", "add_note", "list_notes",
                  "add_task", "list_tasks", "set_task_status", "delete_task", "task_summary", "delete_note",
                  "kb_stats", "document_outline", "delete_document", "update_document"}
MCP_URL = os.getenv("MCP_URL", "http://localhost:9000/mcp")
BUILT_IN_TOOLS = {"get_weather", "web_fetch", "sql_query", "make_chart", "openml_search", "openml_download", "pandas_query",
                  "ingest_document", "search_knowledge_base"}
ASYNC_TOOLS = ["get_weather", "web_fetch", "sql_query", "search_knowledge_base"]
PASSWORD = "correct-horse-battery"


def rand() -> str:
    return uuid.uuid4().hex[:8]


# ------------------------------------------------------------------ users and requests
def new_user(client) -> dict:
    """Registers and logs in a fresh user. If the API has no /users yet (task 5 not done) the user is anonymous (no headers)."""
    name = f"user_{rand()}"
    r = client.post("/users", json={"username": name, "password": PASSWORD})
    if r.status_code in (404, 405):
        return {"name": name, "headers": {}, "anonymous": True}
    assert r.status_code == 201, f"POST /users returned {r.status_code}: {r.text[:200]}"
    login = client.post("/login", json={"username": name, "password": PASSWORD})
    assert login.status_code == 200 and login.json().get("token"), f"POST /login returned {login.status_code}: {login.text[:200]}"
    return {"name": name, "user_id": r.json()["user_id"], "token": login.json()["token"],
            "headers": {"Authorization": f"Bearer {login.json()['token']}"}, "anonymous": False}


def need_users(user: dict) -> None:
    assert not user["anonymous"], "task 5 is not done yet: POST /users does not exist"


def ask(client, user: dict, thread: str, message: str, expect: int = 200) -> dict:
    r = client.post("/agent", json={"message": message, "thread_id": thread}, headers=user["headers"])
    assert r.status_code == expect, f"/agent returned {r.status_code} (expected {expect}): {r.text[:300]}"
    return r.json() if expect == 200 else {}


def upload(client, user: dict, session: str, name: str, text: str, expect: int = 200) -> str:
    r = client.post("/upload", data={"session_id": session}, files={"file": (name, text.encode())}, headers=user["headers"])
    assert r.status_code == expect, f"/upload returned {r.status_code} (expected {expect}): {r.text[:300]}"
    return r.json()["file_id"] if expect == 200 else ""


def ingest(client, user: dict, session: str, name: str, text: str) -> dict:
    out = ask(client, user, session, f"ingest {upload(client, user, session, name, text)}")
    assert "Ingested" in out["answer"] or "already in the knowledge base" in out["answer"], out["answer"]
    return out


def tool_steps(resp: dict, tool: str) -> list[dict]:
    return [s for s in resp["steps"] if s["tool"] == tool]


def call(client, user: dict, tool: str, args: dict, thread: str | None = None) -> str:
    """Runs one tool through the real agent loop and returns its raw result text."""
    out = ask(client, user, thread or f"call-{rand()}", f"call {tool} {json.dumps(args)}")
    steps = tool_steps(out, tool)
    assert steps, f"the agent did not run '{tool}': is it registered? Answer: {out['answer'][:200]}"
    return steps[0]["result"]


def search(client, user: dict, thread: str, text: str) -> str:
    out = ask(client, user, thread, f"search {text}")
    steps = tool_steps(out, "search_knowledge_base")
    assert steps, f"the agent did not run search_knowledge_base: {out['answer'][:200]}"
    return steps[0]["result"]


def kb_docs(client, user: dict, thread: str = "any") -> list[str]:
    r = client.get(f"/kb/{thread}", headers=user["headers"])
    assert r.status_code == 200, f"GET /kb/{thread} returned {r.status_code}: {r.text[:200]}"
    return [d["filename"] for d in r.json()["documents"]]


ERROR_RE = r"ERROR\[{code}\]: .+Next step: .+"


def metric_total(client, name: str, **attrs) -> float:
    metrics = client.get("/debug/telemetry").json()["metrics"]
    return sum(p.get("value", p.get("count", 0)) for m in metrics if m["name"] == name for p in m["points"]
               if all(p["attributes"].get(k) == v for k, v in attrs.items()))


def log_lines(client, event: str) -> list[dict]:
    return [line for line in client.get("/debug/telemetry").json()["logs"] if line.get("event") == event]


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(**kw) -> None:
    RESULTS.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({**load_state(), **kw}))


# ================================================================== task 1: the agent definition
@pytest.mark.phase1
def test_agent_tools(client):
    tools = set(client.get("/health").json()["tools"])
    assert BUILT_IN_TOOLS <= tools, f"missing built-in tools: {sorted(BUILT_IN_TOOLS - tools)} (S1_TOOLS + build_kb_tools(kb), and also add await mcp_tools() and build_custom_tools(kb))"


@pytest.mark.phase1
def test_agent_prompt(client):
    """The model must see the CURRENT knowledge base on every call: a @dynamic_prompt that appends kb.prompt_block(...)."""
    user, thread, name = new_user(client), f"prompt-{rand()}", f"handbook-{rand()}.md"
    ingest(client, user, thread, name, f"# Handbook\n\nVacation policy {rand()}.\n")
    shown = ask(client, user, thread, "showprompt")["answer"]
    assert shown.startswith("SYSTEM PROMPT: ") and "friendly, concise assistant" in shown, "the SYSTEM_PROMPT must still be the base of the prompt"
    assert name in shown, "the system prompt does not list the uploaded document: add kb.prompt_block(...) with @dynamic_prompt"


@pytest.mark.phase1
def test_baseline_intact(client):
    user, session, token = new_user(client), f"base-{rand()}", f"quokka{rand()}"
    out = ingest(client, user, session, f"notes-{rand()}.md", f"# Quokka facts\n\nThe {token} is a small marsupial found in Australia.\n")
    assert tool_steps(out, "ingest_document")
    result = search(client, user, session, token)
    assert re.search(r"Passage 1 \[[^\]]+\.md › [^\]]+\]", result), f"citation format changed: {result[:300]}"
    assert token in result
    sql = ask(client, user, session, 'call sql_query {"database": "chinook", "query": "SELECT COUNT(*) AS n FROM Artist"}')
    assert "row_count" in tool_steps(sql, "sql_query")[0]["result"], "the SQL tool is broken (are the bundled databases in data/ still there?)"
    r = client.post("/upload", data={"session_id": session}, files={"file": ("malware.exe", b"MZ")}, headers=user["headers"])
    assert r.status_code == 422, "unsupported files must be rejected with 422"
    kb_docs(client, user, session)


# ================================================================== task 2: async
@pytest.mark.phase1
def test_async_runtime(client):
    r = client.get("/debug/tools")
    assert r.status_code == 200, "GET /debug/tools is missing"
    by_name = {t["name"]: t["async"] for t in r.json()}
    sync = [n for n in ASYNC_TOOLS if not by_name.get(n)]
    assert not sync, f"still sync (or missing): {sync}"


@pytest.mark.phase1
def test_async_static(client):
    static = json.loads((RESULTS / "static.json").read_text()) if (RESULTS / "static.json").exists() else None
    assert static is not None, "results/static.json is missing: CI runs scripts/check_async.py before the tests (local_eval.sh does too)"
    assert static["async_static"], "blocking calls inside async tools: " + "; ".join(static["problems"])


# ================================================================== task 3: chat memory
@pytest.mark.phase1
def test_memory_live(client):
    user, marker, thread = new_user(client), f"marker{rand()}", f"mem-{rand()}"
    ask(client, user, thread, f"my marker is {marker}")
    assert marker in ask(client, user, thread, "recall")["answer"], "the thread forgot what was said in it: is the checkpointer passed to create_agent?"
    assert marker not in ask(client, user, f"other-{rand()}", "recall")["answer"], "another thread must not see this thread's messages"
    save_state(marker=marker, mem_thread=thread, mem_user=user)  # phase 2 asks again after the restart


# ================================================================== tasks 4, 5, 6: migration, accounts, per-user knowledge base
@pytest.mark.phase1
def test_migration_roundtrip(client):
    path = RESULTS / "migration.json"
    assert path.exists() and path.read_text().strip(), "results/migration.json is missing: CI runs scripts/check_migration.py first"
    m = json.loads(path.read_text())
    assert m.get("revisions") == 1, f"write exactly ONE migration in src/app/migrations (heads found: {m.get('revisions')}). {m.get('log', '')}"
    assert m.get("roundtrip"), f"alembic upgrade head / downgrade base / upgrade head must create, drop and re-create users, tokens, threads: {m}"


@pytest.mark.phase1
def test_migration_schema(client):
    r = client.get("/debug/schema")
    assert r.status_code == 200, "GET /debug/schema failed"
    need = {"users": {"id", "username", "password_hash", "created"}, "tokens": {"token", "user_id", "created"}, "threads": {"thread_id", "user_id"}}
    got = r.json()
    missing = {t: sorted(cols - set(got["tables"].get(t, []))) for t, cols in need.items() if not cols <= set(got["tables"].get(t, []))}
    assert not missing, f"tables/columns missing in the running database: {missing}"
    assert got["alembic_revision"], "alembic_version is empty: the tables must be created by `alembic upgrade head` (the migrate service), not by app code"


@pytest.mark.phase1
def test_users_register_login(client):
    name = f"reg_{rand()}"
    r = client.post("/users", json={"username": name, "password": PASSWORD})
    assert r.status_code == 201, f"POST /users: expected 201, got {r.status_code} {r.text[:200]}"
    body = r.json()
    assert body.get("username") == name and body.get("user_id"), f"response needs user_id and username: {body}"
    assert "password" not in json.dumps(body).lower(), "never return the password or its hash"
    assert client.post("/users", json={"username": name, "password": "another-password"}).status_code == 409, "a taken username must give 409"
    for bad in ({"username": "Ab", "password": PASSWORD}, {"username": "has space", "password": PASSWORD}, {"username": f"ok_{rand()}", "password": "short"}):
        assert client.post("/users", json=bad).status_code == 422, f"invalid input must give 422: {bad}"
    ok = client.post("/login", json={"username": name, "password": PASSWORD})
    assert ok.status_code == 200 and ok.json().get("token") and ok.json().get("user_id") == body["user_id"], f"POST /login: {ok.status_code} {ok.text[:200]}"
    assert client.post("/login", json={"username": name, "password": "wrong-password"}).status_code == 401, "wrong password must give 401"
    assert client.post("/login", json={"username": f"nobody_{rand()}", "password": PASSWORD}).status_code == 401, "unknown user must give 401"


@pytest.mark.phase1
def test_users_manage(client):
    user = new_user(client)
    need_users(user)
    me = client.get("/me", headers=user["headers"])
    assert me.status_code == 200 and me.json() == {"user_id": user["user_id"], "username": user["name"]}, f"GET /me: {me.status_code} {me.text[:200]}"
    second = client.post("/login", json={"username": user["name"], "password": PASSWORD}).json()["token"]
    out = client.post("/logout", headers=user["headers"])
    assert out.status_code == 204, f"POST /logout must give 204, got {out.status_code}"
    assert client.get("/me", headers=user["headers"]).status_code == 401, "a logged-out token must give 401"
    assert client.get("/me", headers={"Authorization": f"Bearer {second}"}).status_code == 200, "logout revokes only the token used"


@pytest.mark.phase1
def test_users_delete(client):
    user, thread, token = new_user(client), f"del-{rand()}", f"gone{rand()}"
    need_users(user)
    ingest(client, user, thread, f"doomed-{rand()}.md", f"# Doomed\n\nThe word is {token}.\n")
    assert client.delete("/users/me", headers=user["headers"]).status_code == 204, "DELETE /users/me must give 204"
    assert client.get("/me", headers=user["headers"]).status_code == 401, "the deleted user's token must stop working"
    assert client.post("/login", json={"username": user["name"], "password": PASSWORD}).status_code == 401, "the deleted user cannot log in"
    again = client.post("/users", json={"username": user["name"], "password": PASSWORD})
    assert again.status_code == 201, "the username must be free again after deletion"
    fresh = {"headers": {"Authorization": "Bearer " + client.post("/login", json={"username": user["name"], "password": PASSWORD}).json()["token"]}}
    assert kb_docs(client, fresh) == [], "the old knowledge base must be gone: the re-registered user starts empty (kb.reset(user_id))"


@pytest.mark.phase1
def test_users_auth(client):
    user = new_user(client)
    need_users(user)
    routes = [("POST", "/agent", {"json": {"message": "hi", "thread_id": "t1"}}),
              ("POST", "/upload", {"data": {"session_id": "t1"}, "files": {"file": ("a.txt", b"hello")}}),
              ("GET", "/kb/t1", {}), ("DELETE", "/kb/t1", {}), ("GET", "/me", {}), ("POST", "/logout", {}), ("DELETE", "/users/me", {})]
    bad = {}
    for method, path, kw in routes[:5]:  # logout/delete-me are checked without a token only (they would act on a real one)
        for label, headers in (("no token", {}), ("bogus token", {"Authorization": "Bearer nope"}), ("not Bearer", {"Authorization": "Basic abc"})):
            if client.request(method, path, headers=headers, **kw).status_code != 401:
                bad[f"{method} {path} ({label})"] = "not 401"
    for method, path, kw in routes[5:]:
        if client.request(method, path, **kw).status_code != 401:
            bad[f"{method} {path} (no token)"] = "not 401"
    assert not bad, f"protected routes must answer 401: {sorted(bad)}"
    assert client.get("/kb/t1", headers=user["headers"]).status_code == 200, "a valid token must work"


@pytest.mark.phase1
def test_user_kb_shared_across_chats(client):
    """Same user, different chats: one knowledge base. (The documents also feed the phase-2 restart check.)"""
    user, token = new_user(client), f"zebracorn{rand()}"
    need_users(user)
    name = f"zebra-{rand()}.md"
    ingest(client, user, f"chat-a-{rand()}", name, f"# Zebra notes\n\nThe secret word is {token} and nothing else.\n")
    other = f"chat-b-{rand()}"
    assert name in kb_docs(client, user, other), "a second chat of the SAME user sees no documents: the knowledge base is still keyed by session"
    assert token in search(client, user, other, token), "chat B cannot find chat A's document"
    prompt = ask(client, user, other, "showprompt")["answer"]
    assert name in prompt, "the dynamic prompt of chat B must list the user's documents"
    save_state(kb_user=user, kb_token=token, kb_name=name)


@pytest.mark.phase1
def test_user_isolation(client):
    alice, bob = new_user(client), new_user(client)
    need_users(alice)
    token, thread = f"secret{rand()}", f"alice-{rand()}"
    name = f"private-{rand()}.md"
    ingest(client, alice, thread, name, f"# Private\n\nThe launch code is {token}.\n")
    bobs = f"bob-{rand()}"
    assert kb_docs(client, bob, bobs) == [], "Bob sees Alice's documents"
    assert token not in search(client, bob, bobs, token), "Bob's search found Alice's document"
    assert name not in ask(client, bob, bobs, "showprompt")["answer"], "Bob's prompt lists Alice's document"
    ask(client, bob, thread, "recall", expect=403)  # Alice's thread
    upload(client, bob, thread, "x.md", "# x\n\nhello\n", expect=403)
    client.delete(f"/kb/{bobs}", headers=bob["headers"])  # Bob clearing HIS knowledge base...
    assert token in search(client, alice, f"alice2-{rand()}", token), "...must not touch Alice's"
    save_state(iso_alice=alice, iso_bob=bob, iso_thread=thread, iso_token=token)


# ================================================================== task 7: kb_stats, document_outline
STATS_RE = re.compile(r"^(\d+) documents, (\d+) sections, (\d+) chunks\. Top keywords: (.+)\.$", re.S)


def stats(client, user, **args) -> tuple[int, int, int, list[str]]:
    text = call(client, user, "kb_stats", args)
    m = STATS_RE.match(text.strip())
    assert m, f"kb_stats output format is '<D> documents, <S> sections, <C> chunks. Top keywords: a, b.' got: {text[:300]}"
    return int(m[1]), int(m[2]), int(m[3]), [k.strip() for k in m[4].split(",")]


@pytest.mark.phase1
def test_kb_stats(client):
    user = new_user(client)
    if not user["anonymous"]:  # a fresh user has an empty knowledge base
        assert call(client, user, "kb_stats", {}).strip() == "The knowledge base is empty.", "empty knowledge base text"
    ingest(client, user, f"stats-{rand()}", f"first-{rand()}.md", f"# One\n\nAlpha {rand()}.\n")
    d0, s0, c0, _ = stats(client, user)
    ingest(client, user, f"stats-{rand()}", f"second-{rand()}.md", f"# Two\n\nBeta {rand()}.\n")
    d1, s1, c1, _ = stats(client, user)
    assert (d1, s1 > s0, c1 > c0) == (d0 + 1, True, True), f"one more document should add 1 document, sections and chunks: {(d0, s0, c0)} -> {(d1, s1, c1)}"
    assert len(stats(client, user, top_keywords=1)[3]) == 1, "top_keywords=1 must list exactly one keyword"
    for bad in (0, 21):
        assert re.match(ERROR_RE.format(code="INVALID_INPUT"), call(client, user, "kb_stats", {"top_keywords": bad}), re.S), \
            f"top_keywords={bad} must fail with ERROR[INVALID_INPUT]: ... Next step: ..."
    before = metric_total(client, "kbstats.calls")
    stats(client, user)
    assert before > 0 and metric_total(client, "kbstats.calls") > before, "record the kbstats.calls counter on every call"
    lines = log_lines(client, "kb_stats_done")
    assert lines and all(l.get("tool") == "kb_stats" and isinstance(l.get("duration_ms"), (int, float)) for l in lines), \
        "log kb_stats_done with tool='kb_stats', status and a numeric duration_ms"
    assert {l.get("status") for l in lines} >= {"ok", "error"}, "log status='ok' and status='error' (rejected input)"


OUTLINE_MD = ("# Introduction\n\nWelcome.\n\n## Scope\n\nWhat is covered.\n\n# Rules\n\nThe rules.\n\n## Refunds\n\nMoney back.\n")
SUMMARY = "Scripted summary of this part."


@pytest.mark.phase1
def test_outline_output(client):
    user, thread, name = new_user(client), f"outl-{rand()}", f"policy-{rand()}.md"
    ingest(client, user, thread, name, OUTLINE_MD)
    out = call(client, user, "document_outline", {"doc_name": name}, thread)
    expected = "\n".join([f"Outline of '{name}' (4 sections):", f"1. [L1] Introduction: {SUMMARY}", f"2. [L2] Scope: {SUMMARY}",
                          f"3. [L1] Rules: {SUMMARY}", f"4. [L2] Refunds: {SUMMARY}"])
    assert out.strip() == expected, f"unexpected outline.\n--- got ---\n{out}\n--- expected ---\n{expected}"
    assert call(client, user, "document_outline", {"doc_name": name.upper()}, thread).startswith(f"Outline of '{name}'"), \
        "doc_name must be resolved like the search tool does (kb.resolve_doc_name), the output uses the exact stored name"


@pytest.mark.phase1
def test_outline_errors(client):
    owner, thread, name = new_user(client), f"outerr-{rand()}", f"mine-{rand()}.md"
    ingest(client, owner, thread, name, OUTLINE_MD)
    unknown = call(client, owner, "document_outline", {"doc_name": f"nothing-{rand()}.pdf"})
    assert re.match(ERROR_RE.format(code="UNKNOWN_DOCUMENT"), unknown, re.S), f"expected ERROR[UNKNOWN_DOCUMENT]: ... Next step: ..., got {unknown[:200]!r}"
    thief = new_user(client)
    if not thief["anonymous"]:
        stolen = call(client, thief, "document_outline", {"doc_name": name})
        assert re.match(ERROR_RE.format(code="UNKNOWN_DOCUMENT"), stolen, re.S), "another user must not be able to outline this document"


@pytest.mark.phase1
def test_outline_metric(client):
    user, thread, name = new_user(client), f"outm-{rand()}", f"metric-{rand()}.md"
    ingest(client, user, thread, name, OUTLINE_MD)
    ok0, err0 = metric_total(client, "outline.calls", status="ok"), metric_total(client, "outline.calls", status="error")
    call(client, user, "document_outline", {"doc_name": name}, thread)
    call(client, user, "document_outline", {"doc_name": f"nothing-{rand()}.md"}, thread)
    assert metric_total(client, "outline.calls", status="ok") > ok0, "outline.calls must grow with attributes {'status': 'ok'} on a good call"
    assert metric_total(client, "outline.calls", status="error") > err0, "outline.calls must grow with {'status': 'error'} on a rejected call"


@pytest.mark.phase1
def test_outline_log(client):
    user, thread, name = new_user(client), f"outl-{rand()}", f"log-{rand()}.md"
    ingest(client, user, thread, name, OUTLINE_MD)
    call(client, user, "document_outline", {"doc_name": name}, thread)
    call(client, user, "document_outline", {"doc_name": f"nothing-{rand()}.md"}, thread)
    lines = log_lines(client, "document_outline_done")
    assert lines, "no log line with event 'document_outline_done'"
    assert all(l.get("tool") == "document_outline" and isinstance(l.get("duration_ms"), (int, float)) for l in lines), \
        "every line needs tool='document_outline' and a numeric duration_ms"
    oks, errs = [l for l in lines if l.get("status") == "ok"], [l for l in lines if l.get("status") == "error"]
    assert oks and errs, "log status='ok' for good calls and status='error' for rejected ones"
    assert oks[-1].get("sections") == 4 and all(l.get("sections") == 0 for l in errs), "log sections=<count> on ok and sections=0 on error"
    before = len(lines)
    call(client, user, "document_outline", {"doc_name": name}, thread)
    assert len(log_lines(client, "document_outline_done")) > before, "one log line per call, every call"


# ================================================================== task 8: delete_document, update_document
@pytest.mark.phase1
def test_delete_works(client):
    user, thread = new_user(client), f"del-{rand()}"
    t1, t2 = f"alphatoken{rand()}", f"betatoken{rand()}"
    n1, n2 = f"first-{rand()}.md", f"second-{rand()}.md"
    ingest(client, user, thread, n1, f"# First\n\nThe word is {t1}.\n")
    ingest(client, user, thread, n2, f"# Second\n\nThe word is {t2}.\n")
    other = None
    if not user["anonymous"]:  # someone else with a document of the SAME file name
        other = new_user(client)
        ingest(client, other, f"o-{rand()}", n1, f"# First\n\nOther content {t1}.\n")
    out = call(client, user, "delete_document", {"doc_name": n1}, thread)
    assert out.strip() == f"Deleted '{n1}'. 1 documents left.", f"unexpected delete_document output: {out[:200]!r}"
    assert t1 not in search(client, user, thread, t1), "the deleted document is still searchable"
    assert t2 in search(client, user, thread, t2), "the other document must stay searchable"
    assert kb_docs(client, user, thread) == [n2], "the knowledge base list must show only the remaining document"
    assert n1 not in ask(client, user, thread, "showprompt")["answer"], "the prompt must stop listing the deleted document"
    if other:
        assert t1 in search(client, other, f"o2-{rand()}", t1), "deleting must not touch another user's document with the same name"
    call(client, user, "delete_document", {"doc_name": n2}, thread)
    assert "empty" in ask(client, user, thread, "showprompt")["answer"].lower().split("knowledge base:")[-1][:40], \
        "with no documents left the prompt must say the knowledge base is empty"


@pytest.mark.phase1
def test_delete_errors(client):
    owner, thread = new_user(client), f"delerr-{rand()}"
    token, name = f"keepme{rand()}", f"mine-{rand()}.md"
    ingest(client, owner, thread, name, f"# Mine\n\nThe word is {token}.\n")
    unknown = call(client, owner, "delete_document", {"doc_name": f"nothing-{rand()}.pdf"})
    assert re.match(ERROR_RE.format(code="UNKNOWN_DOCUMENT"), unknown, re.S), f"expected ERROR[UNKNOWN_DOCUMENT]: ... Next step: ..., got {unknown[:200]!r}"
    thief = new_user(client)
    if not thief["anonymous"]:
        stolen = call(client, thief, "delete_document", {"doc_name": name})
        assert re.match(ERROR_RE.format(code="UNKNOWN_DOCUMENT"), stolen, re.S), "another user must not be able to delete (or see) this document"
        assert token in search(client, owner, thread, token), "the document must survive the attempt"


@pytest.mark.phase1
def test_update_works(client):
    user, thread, name = new_user(client), f"upd-{rand()}", f"report-{rand()}.md"
    old, new = f"oldtoken{rand()}", f"newtoken{rand()}"
    ingest(client, user, thread, name, f"# Report\n\nVersion one: {old}.\n")
    ingest(client, user, thread, f"bystander-{rand()}.md", f"# Other\n\nUnrelated {rand()}.\n")
    docs_before = kb_docs(client, user, thread)
    file_id = upload(client, user, thread, f"renamed-{rand()}.md", f"# Report\n\nVersion two: {new}.\n")
    out = call(client, user, "update_document", {"doc_name": name, "source": file_id}, thread)
    assert re.fullmatch(rf"Updated '{re.escape(name)}': \d+ sections?, \d+ chunks?\.", out.strip()), f"unexpected update_document output: {out[:200]!r}"
    assert old not in search(client, user, thread, old), "the old version is still searchable"
    found = search(client, user, thread, new)
    assert new in found and f"[{name} ›" in found, f"the new content must be searchable under the OLD file name: {found[:300]}"
    assert sorted(kb_docs(client, user, thread)) == sorted(docs_before), "an update keeps the same documents (names and count)"
    same = upload(client, user, thread, f"again-{rand()}.md", f"# Report\n\nVersion two: {new}.\n")
    out = call(client, user, "update_document", {"doc_name": name, "source": same}, thread)
    assert out.strip() == f"'{name}' is already up to date.", f"identical content: {out[:200]!r}"


@pytest.mark.phase1
def test_update_errors(client):
    user, thread, name, token = new_user(client), f"upderr-{rand()}", f"stable-{rand()}.md", f"stable{rand()}"
    ingest(client, user, thread, name, f"# Stable\n\nThe word is {token}.\n")
    file_id = upload(client, user, thread, f"v2-{rand()}.md", f"# Stable\n\nChanged {rand()}.\n")
    unknown = call(client, user, "update_document", {"doc_name": f"nothing-{rand()}.md", "source": file_id}, thread)
    assert re.match(ERROR_RE.format(code="UNKNOWN_DOCUMENT"), unknown, re.S), f"expected ERROR[UNKNOWN_DOCUMENT]: ... got {unknown[:200]!r}"
    missing = call(client, user, "update_document", {"doc_name": name, "source": "deadbeef00"}, thread)
    assert re.match(ERROR_RE.format(code="FILE_NOT_FOUND"), missing, re.S), f"expected ERROR[FILE_NOT_FOUND]: ... got {missing[:200]!r}"
    assert token in search(client, user, thread, token), "a failed update must leave the old document untouched"
    assert kb_docs(client, user, thread) == [name]


# ================================================================== task 9: notes and tasks MCP server
def mcp_call(tool: str, args: dict, authorization: str | None = None) -> tuple[bool, str]:
    """Speaks MCP (streamable HTTP + JSON-RPC) straight to the server, like any MCP client. -> (is_error, text)."""
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    if authorization is not None:
        headers["Authorization"] = authorization

    def rpc(c, body, extra=None):
        r = c.post(MCP_URL, headers={**headers, **(extra or {})}, json=body)
        assert r.status_code in (200, 202), f"MCP server returned {r.status_code}: {r.text[:200]}"
        if r.status_code == 202 or not r.text.strip():
            return r, None
        text = r.text
        if "data:" in text and r.headers.get("content-type", "").startswith("text/event-stream"):
            text = "".join(l[5:].strip() for l in text.splitlines() if l.startswith("data:"))
        return r, json.loads(text)

    with httpx.Client(timeout=30) as c:
        r, _ = rpc(c, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "grader", "version": "1"}}})
        sid = {"mcp-session-id": r.headers["mcp-session-id"]} if "mcp-session-id" in r.headers else {}
        rpc(c, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
        _, out = rpc(c, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool, "arguments": args}}, sid)
    result = out["result"]
    return bool(result.get("isError")), "".join(b.get("text", "") for b in result.get("content", []))


def bearer(user: dict) -> str:
    return user["headers"]["Authorization"]


def mcp_value(result: str):
    """The JSON a tool returned. Through the agent the text may arrive as a JSON string or as a stringified content list."""
    try:
        return json.loads(result)
    except ValueError:
        blocks = ast.literal_eval(result)
        return json.loads("".join(b["text"] for b in blocks))


def mcp(client, user: dict, tool: str, args: dict | None = None, thread: str | None = None):
    return mcp_value(call(client, user, tool, args or {}, thread))


def is_error(text: str, code: str) -> bool:
    return bool(re.search(ERROR_RE.format(code=code), text, re.S))


@pytest.mark.phase1
def test_mcp_tools(client):
    user = new_user(client)
    need_users(user)
    thread = f"mcp-{rand()}"
    task = mcp(client, user, "add_task", {"title": "write report", "details": "q3"}, thread)
    assert task == {"id": task["id"], "title": "write report", "status": "todo"}, f"add_task returns {{id, title, status}}: {task}"
    assert [t["title"] for t in mcp(client, user, "list_tasks", {}, thread)] == ["write report"], "list_tasks: this user's tasks only"
    assert mcp(client, user, "set_task_status", {"task_id": task["id"], "status": "in_progress"}, thread) == {"id": task["id"], "status": "in_progress"}
    assert mcp(client, user, "task_summary", {}, thread) == {"todo": 0, "in_progress": 1, "done": 0}
    note = mcp(client, user, "add_note", {"title": "idea", "body": "try graphs"}, thread)
    assert note["title"] == "idea" and mcp(client, user, "list_notes", {}, thread)[0]["body"] == "try graphs"
    assert mcp(client, user, "delete_note", {"note_id": note["id"]}, thread) == {"deleted": note["id"]}
    assert mcp(client, user, "delete_task", {"task_id": task["id"]}, thread) == {"deleted": task["id"]}


@pytest.mark.phase1
def test_mcp_auth(client):
    user = new_user(client)
    need_users(user)
    calls = {"add_note": {"title": "t", "body": "b"}, "list_notes": {}, "delete_note": {"note_id": 1}, "add_task": {"title": "t"},
             "list_tasks": {}, "set_task_status": {"task_id": 1, "status": "done"}, "delete_task": {"task_id": 1}, "task_summary": {}}
    bad = {}
    for tool, args in calls.items():
        for label, header in (("no token", None), ("not Bearer", "Basic abc"), ("unknown token", "Bearer nope-" + rand())):
            err, text = mcp_call(tool, args, header)
            if not (err and is_error(text, "UNAUTHENTICATED")):
                bad[f"{tool} ({label})"] = text[:80]
    assert not bad, f"every tool must answer ERROR[UNAUTHENTICATED]: ... Next step: ... without a valid token: {bad}"
    err, text = mcp_call("list_tasks", {}, bearer(user))
    assert not err and json.loads(text) == [], f"a valid token must work: {text[:200]}"


@pytest.mark.phase1
def test_mcp_isolation(client):
    alice, bob = new_user(client), new_user(client)
    need_users(alice)
    title, note_title = f"alice-task-{rand()}", f"alice-note-{rand()}"
    task = mcp(client, alice, "add_task", {"title": title})
    note = mcp(client, alice, "add_note", {"title": note_title, "body": "private"})
    assert mcp(client, bob, "list_tasks") == [] and mcp(client, bob, "list_notes") == [], "Bob sees Alice's tasks or notes"
    assert mcp(client, bob, "task_summary") == {"todo": 0, "in_progress": 0, "done": 0}
    assert is_error(call(client, bob, "set_task_status", {"task_id": task["id"], "status": "in_progress"}), "TASK_NOT_FOUND"), "Bob must not touch Alice's task"
    assert is_error(call(client, bob, "delete_task", {"task_id": task["id"]}), "TASK_NOT_FOUND")
    assert is_error(call(client, bob, "delete_note", {"note_id": note["id"]}), "NOTE_NOT_FOUND")
    err, text = mcp_call("list_tasks", {}, bearer(bob))
    assert title not in text, "Bob's own token must not show Alice's tasks over MCP either"
    assert [t["title"] for t in mcp(client, alice, "list_tasks")] == [title], "Alice's data must be untouched"
    save_state(mcp_alice=alice, mcp_bob=bob, mcp_title=title)


@pytest.mark.phase1
def test_mcp_task_lifecycle(client):
    user = new_user(client)
    need_users(user)
    ids = [mcp(client, user, "add_task", {"title": f"task {i}"})["id"] for i in range(3)]
    assert len(mcp(client, user, "list_tasks", {"status": "todo"})) == 3
    mcp(client, user, "set_task_status", {"task_id": ids[0], "status": "in_progress"})
    assert [t["id"] for t in mcp(client, user, "list_tasks", {"status": "in_progress"})] == [ids[0]], "list_tasks must filter by status"
    mcp(client, user, "set_task_status", {"task_id": ids[0], "status": "done"})
    assert mcp(client, user, "task_summary") == {"todo": 2, "in_progress": 0, "done": 1}
    errors = {
        "INVALID_TRANSITION": call(client, user, "set_task_status", {"task_id": ids[1], "status": "done"}),   # todo -> done is not allowed
        "INVALID_STATUS": call(client, user, "set_task_status", {"task_id": ids[1], "status": "blocked"}),
        "TASK_NOT_FOUND": call(client, user, "set_task_status", {"task_id": 999999, "status": "in_progress"}),
        "INVALID_INPUT": call(client, user, "add_task", {"title": "   "}),
    }
    wrong = {code: text[:120] for code, text in errors.items() if not is_error(text, code)}
    assert not wrong, f"expected 'ERROR[CODE]: ... Next step: ...' for: {wrong}"
    assert is_error(call(client, user, "list_tasks", {"status": "later"}), "INVALID_STATUS")
    mcp(client, user, "delete_task", {"task_id": ids[2]})
    assert len(mcp(client, user, "list_tasks")) == 2


@pytest.mark.phase1
def test_mcp_notes(client):
    user = new_user(client)
    need_users(user)
    a = mcp(client, user, "add_note", {"title": "first", "body": "one"})
    mcp(client, user, "add_note", {"title": "second", "body": "two"})
    assert [(n["title"], n["body"]) for n in mcp(client, user, "list_notes")] == [("first", "one"), ("second", "two")]
    mcp(client, user, "delete_note", {"note_id": a["id"]})
    assert [n["title"] for n in mcp(client, user, "list_notes")] == ["second"]
    assert is_error(call(client, user, "delete_note", {"note_id": a["id"]}), "NOTE_NOT_FOUND")
    assert is_error(call(client, user, "add_note", {"title": " ", "body": "x"}), "INVALID_INPUT")


@pytest.mark.phase1
def test_mcp_token_hygiene(client):
    user = new_user(client)
    need_users(user)
    token, thread = user["token"], f"hyg-{rand()}"
    outs = [ask(client, user, thread, f"call {t} {json.dumps(a)}") for t, a in
            (("add_task", {"title": "x"}), ("list_tasks", {}), ("add_note", {"title": "n", "body": "b"}))]
    outs += [ask(client, user, thread, "showprompt"), ask(client, user, thread, "recall")]
    assert token not in json.dumps(outs), "the bearer token leaked into a step, answer or prompt: forward it in a header only"
    assert token not in json.dumps(client.get("/debug/telemetry").json()), "the bearer token leaked into a log line or metric attribute"


# ================================================================== phase 2 (after `docker restart`)
@pytest.mark.phase2
def test_memory_persists_after_restart(client):
    state = load_state()
    out = ask(client, state["mem_user"], state["mem_thread"], "recall")
    assert state["marker"] in out["answer"], f"the chat forgot its history after a restart. Answer: {out['answer'][:200]}"


@pytest.mark.phase2
def test_user_state_persists_after_restart(client):
    state = load_state()
    alice, bob = state["iso_alice"], state["iso_bob"]
    assert client.get("/me", headers=alice["headers"]).status_code == 200, "Alice's token stopped working after the restart: tokens must live in SQLite"
    again = client.post("/login", json={"username": alice["name"], "password": PASSWORD})
    assert again.status_code == 200, "login must still work after the restart (users live in SQLite on the volume)"
    kb_user = state["kb_user"]
    assert state["kb_name"] in kb_docs(client, kb_user), "the user's knowledge base lost its documents after a restart"
    assert state["kb_token"] in search(client, kb_user, f"after-{rand()}", state["kb_token"]), "search finds nothing after the restart"
    ask(client, bob, state["iso_thread"], "recall", expect=403)  # thread ownership must persist too
    assert state["iso_token"] not in search(client, bob, f"bob-after-{rand()}", state["iso_token"]), "isolation broke after the restart"


@pytest.mark.phase2
def test_mcp_persists_after_restart(client):
    state = load_state()
    alice, bob = state["mcp_alice"], state["mcp_bob"]
    assert [t["title"] for t in mcp(client, alice, "list_tasks")] == [state["mcp_title"]], "Alice's tasks are gone after the restart (MCP_DB must live on the volume)"
    assert mcp(client, bob, "list_tasks") == [], "isolation broke after the restart"
