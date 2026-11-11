"""PROVIDED (task 9): notes and tasks for the MCP server, one private list per user, in SQLite (MCP_DB, default next to KB_DB).

Plain Python, no MCP, no auth: the server (server.py) works out WHO is calling and passes `user_id`; every method only ever
touches that user's rows (another user's id behaves like 'not found', so nothing leaks). Blocking sqlite3: the MCP tools are
small, so calling it directly is fine there. Errors are TaskError; str(error) is the full 'ERROR[CODE]: ... Next step: ...' text.

Task lifecycle:  todo -> in_progress -> done   (in_progress -> todo and done -> in_progress are allowed; todo -> done is not,
nor "to the same status")
"""
import os
import sqlite3
import time
from pathlib import Path

STATUSES = ("todo", "in_progress", "done")
ALLOWED = {("todo", "in_progress"), ("in_progress", "done"), ("in_progress", "todo"), ("done", "in_progress")}


def mcp_db_path() -> Path:
    return Path(os.getenv("MCP_DB") or Path(os.getenv("KB_DB", "kb.sqlite")).resolve().with_name("mcp.sqlite"))


class TaskError(Exception):
    def __init__(self, code: str, what: str, next_step: str):
        super().__init__(f"ERROR[{code}]: {what} Next step: {next_step}")
        self.code = code


class TaskStore:
    def __init__(self, path: str | Path | None = None):
        self.path = str(path or mcp_db_path())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, title TEXT NOT NULL, details TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'todo', created REAL NOT NULL, updated REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS notes_user ON notes(user_id);
                CREATE INDEX IF NOT EXISTS tasks_user ON tasks(user_id);
            """)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    # ---- notes ----
    def add_note(self, user_id: str, title: str, body: str) -> dict:
        if not title.strip():
            raise TaskError("INVALID_INPUT", "a note needs a title.", "retry with a short title.")
        with self._db() as db:
            cur = db.execute("INSERT INTO notes(user_id, title, body, created) VALUES (?,?,?,?)", (user_id, title.strip()[:120], body[:2000], time.time()))
        return {"id": cur.lastrowid, "title": title.strip()[:120]}

    def list_notes(self, user_id: str) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT id, title, body FROM notes WHERE user_id=? ORDER BY id", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_note(self, user_id: str, note_id: int) -> dict:
        with self._db() as db:
            if not db.execute("DELETE FROM notes WHERE id=? AND user_id=?", (note_id, user_id)).rowcount:
                raise TaskError("NOTE_NOT_FOUND", f"you have no note #{note_id}.", "call list_notes and use one of its ids.")
        return {"deleted": note_id}

    # ---- tasks ----
    def add_task(self, user_id: str, title: str, details: str = "") -> dict:
        if not title.strip():
            raise TaskError("INVALID_INPUT", "a task needs a title.", "retry with a short title.")
        now = time.time()
        with self._db() as db:
            cur = db.execute("INSERT INTO tasks(user_id, title, details, status, created, updated) VALUES (?,?,?,?,?,?)",
                             (user_id, title.strip()[:120], details[:2000], "todo", now, now))
        return {"id": cur.lastrowid, "title": title.strip()[:120], "status": "todo"}

    def list_tasks(self, user_id: str, status: str = "") -> list[dict]:
        if status and status not in STATUSES:
            raise TaskError("INVALID_STATUS", f"'{status}' is not a status.", f"use one of: {', '.join(STATUSES)} (or leave it empty for all).")
        sql, args = "SELECT id, title, details, status FROM tasks WHERE user_id=?", [user_id]
        if status:
            sql, args = sql + " AND status=?", [*args, status]
        with self._db() as db:
            rows = db.execute(sql + " ORDER BY id", args).fetchall()
        return [dict(r) for r in rows]

    def set_task_status(self, user_id: str, task_id: int, status: str) -> dict:
        if status not in STATUSES:
            raise TaskError("INVALID_STATUS", f"'{status}' is not a status.", f"use one of: {', '.join(STATUSES)}.")
        with self._db() as db:
            row = db.execute("SELECT status FROM tasks WHERE id=? AND user_id=?", (task_id, user_id)).fetchone()
            if not row:
                raise TaskError("TASK_NOT_FOUND", f"you have no task #{task_id}.", "call list_tasks and use one of its ids.")
            if (row["status"], status) not in ALLOWED:
                raise TaskError("INVALID_TRANSITION", f"a task cannot go from {row['status']} to {status}.",
                                "move it one step at a time: todo, in_progress, done (todo cannot jump to done).")
            db.execute("UPDATE tasks SET status=?, updated=? WHERE id=? AND user_id=?", (status, time.time(), task_id, user_id))
        return {"id": task_id, "status": status}

    def delete_task(self, user_id: str, task_id: int) -> dict:
        with self._db() as db:
            if not db.execute("DELETE FROM tasks WHERE id=? AND user_id=?", (task_id, user_id)).rowcount:
                raise TaskError("TASK_NOT_FOUND", f"you have no task #{task_id}.", "call list_tasks and use one of its ids.")
        return {"deleted": task_id}

    def summary(self, user_id: str) -> dict:
        with self._db() as db:
            rows = db.execute("SELECT status, COUNT(*) AS n FROM tasks WHERE user_id=? GROUP BY status", (user_id,)).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        return {s: counts.get(s, 0) for s in STATUSES}
