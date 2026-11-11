"""Alembic environment (locked: do not edit). Target: <folder of KB_DB>/users.sqlite, the same file the app opens."""
import os
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine

db_path = Path(os.getenv("KB_DB", "kb.sqlite")).resolve().with_name("users.sqlite")
engine = create_engine(f"sqlite:///{db_path}")

with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=None, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()
