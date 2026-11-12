
"""Chat memory: the LangGraph SQLite checkpointer.

The SQLite-backed checkpointer preserves conversations across API container restarts.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")


@asynccontextmanager
async def open_checkpointer():
    """Open the SQLite checkpointer and keep it open until application shutdown."""

    db_path = Path(CHECKPOINT_DB)

    # Create the parent directory if it does not exist.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        yield checkpointer
