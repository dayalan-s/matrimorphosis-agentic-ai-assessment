from dataclasses import dataclass


@dataclass
class Ctx:
    """Per-request runtime context: tools and the dynamic prompt read the session from here."""

    session_id: str
