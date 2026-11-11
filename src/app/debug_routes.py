"""PROVIDED and LOCKED: /health and the /debug/* endpoints the autograder reads. main.py calls install_routes(app); do not change this file
(it is not one of the files you edit, so changing it makes the submission invalid)."""
import sqlite3

from fastapi import FastAPI, Request

from app.telemetry import DEBUG, LOG_BUFFER, metrics_snapshot
from app.users import users_db_path


def install_routes(app: FastAPI) -> None:
    @app.get("/health")
    async def health(request: Request):
        return {"status": "ok", "tools": [t.name for t in request.app.state.tools]}

    if not DEBUG:  # ASSESSMENT_DEBUG=1: what the autograder (and you) can inspect
        return

    @app.get("/debug/telemetry")
    async def debug_telemetry():
        """Metrics recorded so far (in memory) and the most recent structured log events."""
        return {"metrics": metrics_snapshot(), "logs": list(LOG_BUFFER)[-300:]}

    @app.get("/debug/tools")
    async def debug_tools(request: Request):
        """Every registered tool and whether it is async (has a coroutine and no sync function)."""
        return [{"name": t.name, "async": getattr(t, "coroutine", None) is not None and getattr(t, "func", None) is None}
                for t in request.app.state.tools]

    @app.get("/debug/schema")
    async def debug_schema():
        """Tables and columns of the users database and the Alembic revision it is at (the autograder checks your migration here)."""
        con = sqlite3.connect(users_db_path())
        try:
            tables = {t: [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
                      for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            revision = [r[0] for r in con.execute("SELECT version_num FROM alembic_version")] if "alembic_version" in tables else []
        finally:
            con.close()
        return {"tables": tables, "alembic_revision": revision}
