"""Errors the agent can explain. Every failure becomes `ERROR[CODE]: what happened. Next step: what to do`.

Tools raise ToolException with that text (handle_tool_error=True), so the model reads it as a tool result
with status=error, tells the user in plain words, and the metrics count it by code. No stack traces, ids or
internal paths ever reach the model or the user.
"""
import re

from langchain_core.tools import ToolException


class DocError(Exception):
    """A document problem we can describe precisely (bad type, too big, empty, encrypted ...)."""

    def __init__(self, code: str, message: str, next_step: str):
        super().__init__(message)
        self.code, self.message, self.next_step = code, message, next_step


def to_tool_exception(exc: BaseException) -> ToolException:
    """Map any exception raised inside a tool to a clear, safe ToolException."""
    if isinstance(exc, ToolException):
        return exc
    code, message, nxt = classify(exc)
    return ToolException(f"ERROR[{code}]: {message} Next step: {nxt}")


def classify(exc: BaseException) -> tuple[str, str, str]:
    if isinstance(exc, DocError):
        return exc.code, exc.message, exc.next_step
    if isinstance(exc, FileNotFoundError):
        return "FILE_NOT_FOUND", "No uploaded file with that id exists in this session.", "Ask the user to upload the file again."
    name, text = type(exc).__name__, str(exc)
    if name == "RateLimitError":
        wait = re.search(r"try again in ([\d.]+\s*[a-z]+)", text, re.I)
        return ("RATE_LIMITED", "The AI model is temporarily rate limited.",
                f"Ask the user to retry in about {wait.group(1) if wait else 'a minute'}.")
    if name in {"APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "ConnectTimeout", "TimeoutException"}:
        return "SERVICE_UNAVAILABLE", "A service this step needs did not respond in time.", "Ask the user to retry shortly."
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return "SERVER_MISCONFIGURED", "The server's model credentials are not working.", "Tell the user an administrator must fix the configuration."
    if "Request too large" in text or "413" in text[:40] or "context_length" in text:
        return ("CONTEXT_TOO_LARGE", "This conversation has grown too large for the model's per-request limit.",
                "Ask the user to start a new session or ask a shorter question.")
    if name == "NotFoundError":
        return "MODEL_UNAVAILABLE", "The configured AI model is not available.", "Tell the user an administrator must change the model setting."
    if name in {"UnexpectedResponse", "ResponseHandlingException"} or "qdrant" in type(exc).__module__:
        return "STORAGE_ERROR", "The knowledge base storage is not responding.", "Ask the user to retry shortly."
    if isinstance(exc, ValueError):  # our own validation messages are written for people
        return "INVALID_INPUT", text[:200], "Fix the input and try again."
    return "INTERNAL_ERROR", "Something unexpected went wrong on the server.", "Ask the user to retry; if it repeats, report it."


def http_status(exc: BaseException) -> tuple[int, str, dict]:
    """(status code, safe message, headers) for an exception that escaped the agent."""
    code, message, nxt = classify(exc)
    status = {"RATE_LIMITED": 429, "SERVICE_UNAVAILABLE": 503, "STORAGE_ERROR": 503, "INVALID_INPUT": 422,
              "FILE_NOT_FOUND": 404, "CONTEXT_TOO_LARGE": 413, "SERVER_MISCONFIGURED": 500, "MODEL_UNAVAILABLE": 500}.get(code, 500)
    if type(exc).__name__ == "GraphRecursionError":
        return 422, "The agent needed too many steps for this request. Try a narrower question.", {}
    if isinstance(exc, DocError):
        status = 422
    headers = {"Retry-After": "30"} if status == 429 else {}
    return status, f"{message} {nxt}", headers


def fallback_ok(exc: BaseException) -> bool:
    """True for 'the model answered badly' (bad JSON, schema mismatch): safe to degrade gracefully.
    False for outages, rate limits, auth and missing models: those must fail loudly, not be papered over."""
    return classify(exc)[0] in {"INTERNAL_ERROR", "INVALID_INPUT"}
