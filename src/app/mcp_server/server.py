"""Task 9: the notes and tasks MCP server. Run by docker compose / CI as:  python src/app/mcp_server/server.py --http  (port 9000, path /mcp)

The storage is PROVIDED (taskstore.py: TaskStore), the user lookup is PROVIDED (app/users.py: UserStore.user_for_token).
You write the integration: register the eight tools below on `mcp`, and make every call run AS THE CALLING USER.

How the server knows the user (task 9)
  The agent forwards the logged-in user's token on every MCP call as the HTTP header  Authorization: Bearer <token>  (you write that
  half in agent.py, task 9). A tool declares a `ctx: Context` parameter; FastMCP injects it and the HTTP request is
  ctx.request_context.request (Starlette: .headers). Write ONE helper `current_user_id(ctx) -> str` that
    - reads the bearer token, looks it up with UserStore().user_for_token(token)
    - raises ToolError("ERROR[UNAUTHENTICATED]: <what>. Next step: log in and retry.") when the header is missing, malformed or unknown
  and call it first in every tool. Never take a user id from a tool argument: the model must not be able to name another user.

The eight tools (names and arguments are fixed; each returns json.dumps(<what the TaskStore method returned>), a string)
  add_note(title: str, body: str)               list_notes()                          delete_note(note_id: int)
  add_task(title: str, details: str = "")       list_tasks(status: str = "")          set_task_status(task_id: int, status: str)
  delete_task(task_id: int)                     task_summary()
  status is "todo" | "in_progress" | "done". TaskStore raises TaskError for bad input: re-raise it as ToolError(str(error)).
  The docstring of each tool is its prompt for the model: say WHEN to use it and describe the arguments.
"""
import json  # noqa: F401  (you will need it)
import os
import sys

from mcp.server.fastmcp import Context, FastMCP  # noqa: F401
from mcp.server.fastmcp.exceptions import ToolError  # noqa: F401

from app.mcp_server.taskstore import TaskError, TaskStore  # noqa: F401
from app.users import UserStore  # noqa: F401

mcp = FastMCP("tasks", host="0.0.0.0", port=int(os.getenv("MCP_PORT", "9000")))
store = TaskStore()

# TODO(task 9): the eight tools (@mcp.tool())
# TODO(task 9): current_user_id(ctx), used by every tool


if __name__ == "__main__":
    # stdio cannot carry the Authorization header: the agent talks to this server over HTTP (MCP_URL)
    mcp.run(transport="streamable-http" if "--http" in sys.argv else "stdio")
