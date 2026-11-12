
"""Task 9: the notes and tasks MCP server."""

import json
import os
import sys

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from app.mcp_server.taskstore import TaskError, TaskStore
from app.users import UserStore


mcp = FastMCP("tasks", host="0.0.0.0", port=int(os.getenv("MCP_PORT", "9000")))
store = TaskStore()


def current_user_id(ctx: Context) -> str:
    """Get the authenticated user from the MCP HTTP request."""
    request = ctx.request_context.request
    authorization = request.headers.get("Authorization", "")
    parts = authorization.split()

    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise ToolError(
            "ERROR[UNAUTHENTICATED]: Missing or malformed authentication token. "
            "Next step: log in and retry."
        )

    token = parts[1]
    user = UserStore().user_for_token(token)

    if user is None:
        raise ToolError(
            "ERROR[UNAUTHENTICATED]: Unknown or expired authentication token. "
            "Next step: log in and retry."
        )

    return user["user_id"]


def run_store(method, *args):
    """Run a TaskStore method and convert its errors to MCP tool errors."""
    try:
        return json.dumps(method(*args))
    except TaskError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
def add_note(title: str, body: str, ctx: Context) -> str:
    """Save a note for the logged-in user. Use when the user wants to remember information. Arguments: title and body."""
    user_id = current_user_id(ctx)
    return run_store(store.add_note, user_id, title, body)


@mcp.tool()
def list_notes(ctx: Context) -> str:
    """List the logged-in user's saved notes. Use when the user wants to recall their notes."""
    user_id = current_user_id(ctx)
    return run_store(store.list_notes, user_id)


@mcp.tool()
def delete_note(note_id: int, ctx: Context) -> str:
    """Delete one of the logged-in user's notes. Use the note ID returned by list_notes."""
    user_id = current_user_id(ctx)
    return run_store(store.delete_note, user_id, note_id)


@mcp.tool()
def add_task(title: str, details: str = "", ctx: Context = None) -> str:
    """Create a task for the logged-in user. Use when the user wants to track work. Arguments: title and optional details."""
    user_id = current_user_id(ctx)
    return run_store(store.add_task, user_id, title, details)


@mcp.tool()
def list_tasks(status: str = "", ctx: Context = None) -> str:
    """List the logged-in user's tasks. Optionally filter by todo, in_progress, or done."""
    user_id = current_user_id(ctx)
    return run_store(store.list_tasks, user_id, status)


@mcp.tool()
def set_task_status(task_id: int, status: str, ctx: Context = None) -> str:
    """Change a task's status. Use todo, in_progress, or done and follow the allowed task lifecycle."""
    user_id = current_user_id(ctx)
    return run_store(store.set_task_status, user_id, task_id, status)


@mcp.tool()
def delete_task(task_id: int, ctx: Context = None) -> str:
    """Delete one of the logged-in user's tasks. Use the task ID returned by list_tasks."""
    user_id = current_user_id(ctx)
    return run_store(store.delete_task, user_id, task_id)


@mcp.tool()
def task_summary(ctx: Context) -> str:
    """Summarise the logged-in user's tasks by status."""
    user_id = current_user_id(ctx)
    return run_store(store.summary, user_id)


if __name__ == "__main__":
    # The agent connects over HTTP so it can forward the Authorization header.
    mcp.run(transport="streamable-http" if "--http" in sys.argv else "stdio")
