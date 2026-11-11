"""Uploaded files live under UPLOAD_DIR/<session>/<file_id>__<name>. The agent only ever passes a file_id."""
import ipaddress
import os
import re
import socket
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
import platformdirs

from app.errors import DocError
from app.kb.extract import SUPPORTED

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR") or platformdirs.user_data_dir("session5-agent")) / "uploads"
MAX_BYTES = 20 * 1024 * 1024
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def safe_session(session_id: str) -> str:
    if not SESSION_RE.match(session_id):
        raise DocError("INVALID_INPUT", "Invalid session id.", "Start a new session.")
    return session_id


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name).name)[:120] or "file"


def save_upload(session_id: str, filename: str, data: bytes) -> tuple[str, Path]:
    name = safe_name(filename)
    if Path(name).suffix.lower() not in SUPPORTED:
        raise DocError("UNSUPPORTED_FILE", f"Unsupported file type (supported: {', '.join(sorted(SUPPORTED))}).", "Upload a PDF, DOCX, TXT, MD or image.")
    if len(data) > MAX_BYTES:
        raise DocError("TOO_LARGE", f"File too large (limit {MAX_BYTES // 1024 // 1024} MB).", "Upload a smaller file.")
    folder = UPLOAD_DIR / safe_session(session_id)
    folder.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex[:10]
    path = folder / f"{file_id}__{name}"
    path.write_bytes(data)
    return file_id, path


def resolve(session_id: str, file_id: str) -> tuple[Path, str]:
    """file_id -> (path, original filename). Only files this session uploaded are reachable."""
    if not re.fullmatch(r"[0-9a-f]{10}", file_id):
        raise DocError("FILE_NOT_FOUND", "That is not a valid file_id.", "Use the file_id from the upload message, or ask the user to upload again.")
    for p in (UPLOAD_DIR / safe_session(session_id)).glob(f"{file_id}__*"):
        return p, p.name.split("__", 1)[1]
    raise DocError("FILE_NOT_FOUND", "No uploaded file with that file_id exists in this session.", "Ask the user to upload the file again.")


def _public_host(host: str) -> bool:
    try:
        return all(ipaddress.ip_address(i[4][0]).is_global for i in socket.getaddrinfo(host, None))
    except (socket.gaierror, ValueError):
        return False


async def download(session_id: str, url: str) -> tuple[str, Path]:
    """Fetch a document by URL (public hosts only: no localhost / private ranges)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or not _public_host(parsed.hostname):
        raise DocError("URL_NOT_ALLOWED", "Only public http(s) URLs can be fetched.", "Ask the user to upload the file instead.")
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as c:
        r = await c.get(url)
    if r.status_code >= 300:
        raise DocError("DOWNLOAD_FAILED", f"The link returned HTTP {r.status_code}.", "Check the link, or ask the user to upload the file.")
    name = unquote(Path(parsed.path).name) or "download"
    if not Path(name).suffix:
        ctype = r.headers.get("content-type", "")
        name += ".pdf" if "pdf" in ctype else ".txt"
    return save_upload(session_id, name, r.content)
