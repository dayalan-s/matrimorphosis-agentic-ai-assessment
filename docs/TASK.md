# The assessment: nine tasks, 34 points

You get a working Document Agent (FastAPI + LangChain agent + Gradio UI + knowledge base + observability).
You **integrate**: wiring, small wrappers, one migration. The hard parts are provided, and you only edit `src/app/`.

## What you have to do (in short)
1. **Set up:** your own private repo from the template, your Groq key, `bash scripts/check_setup.sh`, `docker compose up --build -d` (see `README.md`).
2. **Do the nine tasks below, in order.** Each task has its own checks, so you can see your progress after every task.
3. **Check yourself:** `bash scripts/local_eval.sh` runs the grader on your machine and tells you why each failed check fails.
4. **Push to `main`:** the GitHub run **evaluate** grades you. It is red until you have all 34 points; open its **Summary** tab for the score and what to fix.
5. **Submit the run URL** and a read-only token through the form before **12 Nov 2026, 23:59 IST** (`docs/SUBMIT.md`). Partial points count and your best attempt is kept.

## The nine tasks (do them in this order)
| # | Task | Pts | You edit | Done when these checks pass |
|---|---|---|---|---|
| 1 | Define the agent | 3 | `agent.py` | `agent-tools` `agent-prompt` `baseline` |
| 2 | Make the tools async | 2 | `tools_s1.py`, `kb/tools.py` | `async-tools` `async-blocking` |
| 3 | Chat memory in SQLite | 2 | `persistence.py`, `main.py` | `memory-live` `sqlite-memory` |
| 4 | Alembic migration | 2 | `migrations/` | `migration-roundtrip` `migration-schema` |
| 5 | User accounts and auth | 4 | `main.py` | `users-login` `users-manage` `users-delete` `users-auth` |
| 6 | One knowledge base per user | 3 | `context.py`, `agent.py`, `kb/tools.py`, `main.py` | `user-kb` `user-isolation` `user-persists` |
| 7 | Two tools with observability | 6 | `custom_tools.py` | `kb-stats` `outline-output` (2) `outline-errors` `outline-metric` `outline-log` |
| 8 | Manage the knowledge base with tools | 4 | `custom_tools.py` | `delete-works` `delete-errors` `update-works` `update-errors` |
| 9 | Notes and tasks MCP server with per-user auth | 8 | `mcp_server/server.py`, `agent.py`, `context.py`, `main.py` | `mcp-tools` `mcp-auth` (2) `mcp-isolation` `mcp-lifecycle` `mcp-notes` `mcp-token-hygiene` `mcp-persists` |

The starter scores 0. About 320 lines in 9 files in total, mostly wiring and docstrings. The exact output formats, error codes and
names are in the docstrings of the files you edit (read them first); this page says what to do and what is graded.
All paths are under `src/app/`. These are the only files you edit.

### Provided building blocks (read, do not rewrite)
| File | What it gives you |
|---|---|
| `users.py` | `UserStore`: create, login, token lookup, revoke, delete user, claim thread (SQLite, hashed passwords) |
| `mcp_server/taskstore.py` | `TaskStore`: notes and tasks (`todo` / `in_progress` / `done`) per user, with the error codes |
| `kb/store.py` | `kb.stats`, `kb.outline`, `kb.delete_document` |
| `kb/pipeline.py`, `kb/files.py`, `errors.py` | `ingest_file`, upload resolution, `to_tool_exception` |
| `alembic.ini`, `alembic/`, `docker-compose.yml` (outside `src/app/`) | the Alembic setup and the `migrate` service |

### 1. Define the agent · `agent.py`
`build_agent` is a bare agent. Write the real one (steps in its docstring):
- a list of tools: `S1_TOOLS` + `build_kb_tools(kb)` + `await mcp_tools()` + `build_custom_tools(kb)` (the last two stay small until tasks 7 to 9)
- a `@dynamic_prompt` returning `SYSTEM_PROMPT` + `kb.prompt_block(...)` on every model call (use the chat's `session_id` for now; task 6 switches to the user)
- `create_agent(..., middleware=[...], context_schema=Ctx, checkpointer=checkpointer)`, and return `agent, tools`

### 2. Make the tools async · `tools_s1.py`, `kb/tools.py`
A blocking call inside a tool freezes every other request. Make `get_weather`, `web_fetch`, `sql_query`, `search_knowledge_base` `async def` with the same output:
`httpx.AsyncClient` for the download, `asyncio.to_thread` for sqlite and `kb.search`. Checked at runtime (`GET /debug/tools`) and by a code scan (`scripts/check_async.py`).

### 3. Chat memory in SQLite · `persistence.py`, `main.py`
`open_checkpointer()` yields an `AsyncSqliteSaver` on `CHECKPOINT_DB` (create the folder). The `lifespan` opens it, passes it to `build_agent`, and keeps it open until shutdown.
Graded: a thread remembers, another thread does not, and the memory survives a container restart.

### 4. Alembic migration · `migrations/`
Write ONE migration (`alembic revision -m "create users"`): `upgrade` creates `users`, `tokens`, `threads` (columns in `users.py`), `downgrade` drops them.
`docker compose` runs it in the `migrate` service; CI runs upgrade, downgrade, upgrade and checks the tables each time. Only the migration creates the tables, never app code (`GET /debug/schema`).

### 5. User accounts and auth · `main.py`
- `POST /users` (201, 409 if taken, 422 if invalid) and `POST /login` (200 with a token, 401): use `UserStore` through `asyncio.to_thread`
- `GET /me`, `POST /logout` (204, revokes only that token), `DELETE /users/me` (204)
- a FastAPI dependency that turns `Authorization: Bearer <token>` into the current user, **401** otherwise, on `/agent`, `/upload`, `GET`/`DELETE /kb/{session_id}`, `/me`, `/logout`, `/users/me`

### 6. One knowledge base per user · `context.py`, `agent.py`, `kb/tools.py`, `main.py`
Add `user_id` to `Ctx`; `/agent` passes `Ctx(session_id=..., user_id=...)`. Ingest, search, the dynamic prompt and `/kb/...` use the **user id** as the knowledge-base id, so all chats of a user share one and nobody else sees it
(uploaded files stay in the chat folder). `DELETE /users/me` also calls `kb.reset(user_id)`.
A chat `thread_id` belongs to the first user who uses it (`claim_thread`): `/agent` and `/upload` on someone else's thread answer **403**. Users, tokens, threads and knowledge bases must survive a restart.

### 7. Two tools with observability · `custom_tools.py`
- `kb_stats(top_keywords=5)`: counts and top keywords from `kb.stats`; `ERROR[INVALID_INPUT]` outside 1 to 20
- `document_outline(doc_name)`: the sections of one document from `kb.outline`; `ERROR[UNKNOWN_DOCUMENT]` (also for another user's document)
- each tool: counter (`kbstats.calls`, `outline.calls`, with `{"status": "ok"|"error"}`) and a structured log line (`kb_stats_done`, `document_outline_done`) on **every** call
- Grafana (recommended, not graded): `dashboards/outline.json` with a panel for the counter and one for the logs

Try the real agent: *"How is the handbook organised?"*, *"Which section covers refunds?"*

### 8. Manage the knowledge base with tools · `custom_tools.py`
- `delete_document(doc_name)`: removes one document of the calling user (`kb.delete_document`); search, the list and the prompt forget it
- `update_document(doc_name, source)`: replaces a document with a newly uploaded version under the same name. Ingest the new file first, delete the old one only afterwards, so a failed update leaves the old document untouched
- errors: `ERROR[UNKNOWN_DOCUMENT]`, `ERROR[FILE_NOT_FOUND]`; never another user's document

### 9. Notes and tasks MCP server with per-user auth · `mcp_server/server.py`, `agent.py`, `context.py`, `main.py`
The storage is provided; you integrate it:
- **tools:** register `add_note`, `list_notes`, `delete_note`, `add_task`, `list_tasks`, `set_task_status`, `delete_task`, `task_summary`; each returns `json.dumps(...)` of the `TaskStore` result, and `TaskError` becomes `ToolError(str(error))`
- **auth in the server:** every tool runs as the calling user. Read `Authorization: Bearer <token>` from the request (`ctx: Context`, `ctx.request_context.request.headers`), look it up with `UserStore().user_for_token`, otherwise answer `ERROR[UNAUTHENTICATED]`. Never take a user id from a tool argument
- **forward the token from the agent:** add `token` to `Ctx` (`/agent` fills it from the header) and write a `langchain-mcp-adapters` tool interceptor in `agent.py` that sets `Authorization: Bearer <runtime.context.token>` on every MCP call. The token must never appear in a tool argument, step, answer, prompt or log line
- **persistence:** `MCP_DB` is on the data volume; `docker compose` runs the server as `notes-mcp`, CI starts it as a second container and restarts both

## How the grader drives your app
CI runs the migration round trip, starts the MCP server and the API (`LLM_MODE=fake`, a scripted model in `src/app/fake_llm.py`), runs phase 1, **restarts both containers with the same data volume**, runs phase 2.
Until `POST /users` exists (task 5) the other checks run as an anonymous user, so each task is graded on its own. By hand:
```bash
docker compose up -d --build        # api, ui, notes-mcp, migrate, observability
TOKEN=$(curl -s localhost:8000/login -H 'content-type: application/json' -d '{"username":"me","password":"correct-horse"}' | jq -r .token)
curl -s localhost:8000/agent -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"message":"add a task: write the report","thread_id":"t1"}'
```
With `LLM_MODE=fake` you can drive any tool: `call <tool> <json args>`, `ingest <file_id>`, `search <text>`, `recall`, `showprompt`.
Your data must live under `/data` (`CHECKPOINT_DB`, `KB_DB`, `MCP_DB`, `QDRANT_URL=path:/data/qdrant` in the Dockerfile `runner` target; the users database sits next to `KB_DB`).

## Rules
- Edit only the files in the table above (plus new files in `migrations/` and `dashboards/`, `README.md`, `.gitignore`). Everything else, including the rest of `src/app/`, is locked and compared byte for byte.
- No new dependencies (`pyproject.toml` is locked); `alembic`, `langgraph-checkpoint-sqlite`, `aiosqlite`, `mcp`, OpenTelemetry are installed.
- Your own work: submissions are compared automatically.
- Your code must not detect or special-case the grader (`LLM_MODE`, `EMBEDDER`, `ASSESSMENT_DEBUG`, the fake model, the debug internals): `scripts/check_integrity.py` flags it and the score becomes 0 until you remove it.
- **There is no manual review.** Your grade is the score of the CI run.
