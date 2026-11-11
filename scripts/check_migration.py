#!/usr/bin/env python3
"""Runs INSIDE the app image, on the CI data volume, before the API starts (task 4). Prints one JSON object:
upgrade -> tables exist, downgrade -> tables gone, upgrade again -> tables back. Needs only alembic + the stdlib."""
import json
import os
import sqlite3
import subprocess
from pathlib import Path

NEED = {"users": {"id", "username", "password_hash", "created"}, "tokens": {"token", "user_id", "created"}, "threads": {"thread_id", "user_id"}}
DB = Path(os.getenv("KB_DB", "kb.sqlite")).resolve().with_name("users.sqlite")


def alembic(*args: str) -> tuple[bool, str]:
    p = subprocess.run(["alembic", *args], capture_output=True, text=True)
    return p.returncode == 0, (p.stdout + p.stderr)[-400:]


def schema() -> dict[str, set[str]]:
    if not DB.exists():
        return {}
    con = sqlite3.connect(DB)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {t: {r[1] for r in con.execute(f'PRAGMA table_info("{t}")')} for t in tables}
    finally:
        con.close()


def complete(s: dict) -> bool:
    return all(cols <= s.get(t, set()) for t, cols in NEED.items())


out: dict = {}
ok, msg = alembic("heads")
out["revisions"] = len([l for l in msg.splitlines() if "(head)" in l]) if ok else 0
ok, msg = alembic("upgrade", "head")
out["upgrade_ok"], out["tables_after_upgrade"] = ok, complete(schema())
ok, msg2 = alembic("downgrade", "base")
out["downgrade_ok"], out["tables_gone_after_downgrade"] = ok, not any(t in schema() for t in NEED)
ok, msg3 = alembic("upgrade", "head")
out["reupgrade_ok"], out["tables_after_reupgrade"] = ok, complete(schema())
out["roundtrip"] = out["revisions"] == 1 and all(out[k] for k in ("upgrade_ok", "tables_after_upgrade", "downgrade_ok",
                                                                     "tables_gone_after_downgrade", "reupgrade_ok", "tables_after_reupgrade"))
out["log"] = msg if not out["upgrade_ok"] else (msg2 if not out["downgrade_ok"] else msg3 if not out["reupgrade_ok"] else "")
print(json.dumps(out))
