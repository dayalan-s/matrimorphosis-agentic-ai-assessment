"""PROVIDED (tasks 5 and 6): the user store. You do not write this file; you wire it into the app (see main.py) and write the migration.

Every user owns ONE knowledge base that all of their chats share; nobody else can see it. The HTTP contract (the autograder checks it):

  POST /users   {"username": "...", "password": "..."}  -> 201 {"user_id": "...", "username": "..."}
                  invalid username/password -> 422        username taken -> 409
  POST /login   {"username": "...", "password": "..."}  -> 200 {"token": "...", "user_id": "..."}      wrong user or password -> 401
  GET /me       -> 200 {"user_id": "...", "username": "..."}   the authenticated user
  POST /logout  -> 204, and that token is revoked (it answers 401 from then on); other tokens of the user stay valid
  DELETE /users/me  -> 204: deletes the account (user, tokens, thread claims) AND the user's whole knowledge base (kb.reset(user_id));
                  the username can be registered again afterwards; the old password / tokens stop working
  Authorization: Bearer <token>   required on POST /agent, POST /upload, GET /kb/{session_id}, DELETE /kb/{session_id}, GET /me,
                  POST /logout, DELETE /users/me      missing, malformed or unknown token -> 401
  GET/DELETE /kb/{session_id}     the session id is irrelevant now: they show / clear the KB of the AUTHENTICATED user
  thread_id     a chat thread belongs to the first user who uses it; another user using that thread_id on /agent or /upload -> 403

The store reads and writes SQLite tables that YOUR Alembic migration creates (src/app/migrations): it never creates tables itself.
    users(id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created REAL NOT NULL)
    tokens(token TEXT PRIMARY KEY, user_id TEXT NOT NULL, created REAL NOT NULL)
    threads(thread_id TEXT PRIMARY KEY, user_id TEXT NOT NULL)
All methods are synchronous (blocking sqlite3): call them with `await asyncio.to_thread(...)` from async code.
"""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
import uuid
from pathlib import Path

USERNAME_RE = re.compile(r"^[a-z0-9_]{3,32}$")


def users_db_path() -> Path:
    """<folder of KB_DB>/users.sqlite: the same volume as the knowledge base, so it survives restarts."""
    return Path(os.getenv("KB_DB", "kb.sqlite")).resolve().with_name("users.sqlite")


class UsernameTaken(Exception):
    pass


class InvalidCredentials(ValueError):
    """Username or password does not meet the rules (-> HTTP 422)."""


def _hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def _verify(password: str, stored: str) -> bool:
    _, salt, digest = stored.split("$")
    return hmac.compare_digest(_hash(password, bytes.fromhex(salt)).split("$")[2], digest)


class UserStore:
    def __init__(self, path: str | Path | None = None):
        self.path = str(path or users_db_path())

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def create(self, username: str, password: str) -> dict:
        if not USERNAME_RE.match(username) or len(password) < 8:
            raise InvalidCredentials("username: 3-32 chars of a-z 0-9 _; password: at least 8 characters")
        user_id = uuid.uuid4().hex[:12]
        try:
            with self._db() as db:
                db.execute("INSERT INTO users(id, username, password_hash, created) VALUES (?,?,?,?)",
                           (user_id, username, _hash(password), time.time()))
        except sqlite3.IntegrityError as exc:
            raise UsernameTaken(username) from exc
        return {"user_id": user_id, "username": username}

    def login(self, username: str, password: str) -> dict | None:
        """-> {"token", "user_id"} or None. Always spends the hashing time, so unknown users are not detectable by timing."""
        with self._db() as db:
            row = db.execute("SELECT id, password_hash FROM users WHERE username=?", (username,)).fetchone()
            ok = _verify(password, row["password_hash"] if row else _hash("x" * 8))
            if not (row and ok):
                return None
            token = secrets.token_urlsafe(32)
            db.execute("INSERT INTO tokens(token, user_id, created) VALUES (?,?,?)", (token, row["id"], time.time()))
        return {"token": token, "user_id": row["id"]}

    def user_for_token(self, token: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT u.id, u.username FROM tokens t JOIN users u ON u.id = t.user_id WHERE t.token=?", (token,)).fetchone()
        return {"user_id": row["id"], "username": row["username"]} if row else None

    def claim_thread(self, thread_id: str, user_id: str) -> bool:
        """The first user to use a thread owns it. True if `user_id` may use the thread."""
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO threads(thread_id, user_id) VALUES (?,?)", (thread_id, user_id))
            return db.execute("SELECT user_id FROM threads WHERE thread_id=?", (thread_id,)).fetchone()["user_id"] == user_id

    def revoke(self, token: str) -> None:
        with self._db() as db:
            db.execute("DELETE FROM tokens WHERE token=?", (token,))

    def delete_user(self, user_id: str) -> None:
        """Removes the user, their tokens and thread claims. The knowledge base is separate: the endpoint calls kb.reset(user_id)."""
        with self._db() as db:
            for table, col in (("tokens", "user_id"), ("threads", "user_id"), ("users", "id")):
                db.execute(f"DELETE FROM {table} WHERE {col}=?", (user_id,))
