
from dataclasses import dataclass


@dataclass
class Ctx:
    """Per-request runtime context for user-specific knowledge and MCP access."""

    session_id: str
    user_id: str | None = None
    token: str | None = None
