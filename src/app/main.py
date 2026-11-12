
import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.staticfiles import StaticFiles
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from pydantic import BaseModel

from app.telemetry import (
    RUN_STEPS,
    HTTP_DURATION,
    HTTP_REQUESTS,
    configure_logging,
    log,
    run_config,
    setup_telemetry,
    shutdown_telemetry,
)

setup_telemetry()
configure_logging()
HTTPXClientInstrumentor().instrument()

from app.agent import build_agent, charts_from, clean_answer, steps_from
from app.context import Ctx
from app.debug_routes import install_routes
from app.errors import DocError, http_status
from app.kb import files
from app.kb.store import KnowledgeBase
from app.persistence import open_checkpointer
from app.schemas import AgentRequest, AgentResponse, KBResponse, UploadResponse
from app.tools_s1 import CHARTS_DIR
from app.users import InvalidCredentials, UserStore, UsernameTaken


class UserCredentials(BaseModel):
    username: str
    password: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.kb = KnowledgeBase()
    app.state.users = UserStore()

    async with open_checkpointer() as saver:
        agent, tools = await build_agent(
            app.state.kb,
            checkpointer=saver,
        )

        app.state.agent, app.state.tools = agent, tools
        app.state.tool_names = [t.name for t in tools]

        log.info("startup_complete", tools=app.state.tool_names)

        yield

    shutdown_telemetry()


app = FastAPI(
    title="Agentic AI assessment - Document Agent (tools + knowledge base + MCP)",
    lifespan=lifespan,
)

FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics")
app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")
install_routes(app)


@app.middleware("http")
async def observe(request: Request, call_next):
    """Per-request: request_id in every log line, API-hit metrics by route/method/status."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=rid)

    t0, status = time.perf_counter(), 500

    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response

    finally:
        route = getattr(
            request.scope.get("route"),
            "path",
            request.url.path,
        )
        elapsed = time.perf_counter() - t0

        attrs = {
            "http.route": route,
            "http.method": request.method,
            "http.status_code": str(status),
        }

        if route != "/health":
            HTTP_REQUESTS.add(1, attrs)
            HTTP_DURATION.record(elapsed, attrs)

            log.info(
                "http_request",
                route=route,
                method=request.method,
                status=status,
                duration_ms=round(elapsed * 1000),
            )


async def get_current_user(request: Request) -> dict:
    """Validate the Bearer token and return the authenticated user."""
    authorization = request.headers.get("Authorization", "")
    parts = authorization.split()

    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]

    user = await asyncio.to_thread(
        request.app.state.users.user_for_token,
        token,
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {**user, "_token": token}


@app.post("/users", status_code=201)
async def register_user(data: UserCredentials, request: Request):
    try:
        return await asyncio.to_thread(
            request.app.state.users.create,
            data.username,
            data.password,
        )

    except UsernameTaken as exc:
        raise HTTPException(
            status_code=409,
            detail="Username already exists.",
        ) from exc

    except (InvalidCredentials, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.post("/login")
async def login_user(data: UserCredentials, request: Request):
    result = await asyncio.to_thread(
        request.app.state.users.login,
        data.username,
        data.password,
    )

    if result is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    return result


@app.get("/me")
async def current_user(current_user: dict = Depends(get_current_user)):
    return {
        "user_id": current_user["user_id"],
        "username": current_user["username"],
    }


@app.post("/logout", status_code=204)
async def logout_user(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    await asyncio.to_thread(
        request.app.state.users.revoke,
        current_user["_token"],
    )
    return Response(status_code=204)


@app.delete("/users/me", status_code=204)
async def delete_account(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    await asyncio.to_thread(
        request.app.state.users.delete_user,
        user_id,
    )

    request.app.state.kb.reset(user_id)

    return Response(status_code=204)


@app.post("/agent", response_model=AgentResponse)
async def agent(
    req: AgentRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    structlog.contextvars.bind_contextvars(thread_id=req.thread_id)

    owns_thread = await asyncio.to_thread(
        request.app.state.users.claim_thread,
        req.thread_id,
        user_id,
    )

    if not owns_thread:
        raise HTTPException(
            status_code=403,
            detail="This chat thread belongs to another user.",
        )

    result = await _guard(
        request.app.state.agent.ainvoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config=run_config(thread_id=req.thread_id),
            context=Ctx(
                session_id=user_id,
                user_id=user_id,
                token=current_user["_token"],
            ),
        )
    )

    msgs = result["messages"]

    turn = msgs[max(i for i, m in enumerate(msgs) if m.type == "human"):]

    steps = steps_from(turn)
    RUN_STEPS.record(len(steps))

    log.info(
        "agent_done",
        steps=len(steps),
        tools=[s["tool"] for s in steps],
        answer_chars=len(str(msgs[-1].content)),
    )

    return AgentResponse(
        answer=clean_answer(str(msgs[-1].content)),
        steps=steps,
        charts=charts_from(turn),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    owns_thread = await asyncio.to_thread(
        request.app.state.users.claim_thread,
        session_id,
        user_id,
    )

    if not owns_thread:
        raise HTTPException(
            status_code=403,
            detail="This chat thread belongs to another user.",
        )

    data = await file.read(files.MAX_BYTES + 1)

    try:
        file_id, path = files.save_upload(
            user_id,
            file.filename or "file",
            data,
        )
    except DocError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{exc.message} {exc.next_step}",
        ) from exc

    log.info(
        "file_uploaded",
        filename=path.name.split("__", 1)[1],
        size=len(data),
    )

    return UploadResponse(
        file_id=file_id,
        filename=path.name.split("__", 1)[1],
        size=len(data),
    )


@app.get("/kb/{session_id}", response_model=KBResponse)
async def kb_state(
    session_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Return the authenticated user's knowledge base, regardless of the URL session ID."""
    user_id = current_user["user_id"]
    kb: KnowledgeBase = request.app.state.kb

    prof = kb.get_profile(user_id) or {
        "overview": None,
        "keywords": [],
    }

    return KBResponse(
        **prof,
        documents=kb.documents(user_id),
        prompt_block=kb.prompt_block(user_id),
    )


@app.delete("/kb/{session_id}")
async def kb_reset(
    session_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Clear only the authenticated user's knowledge base."""
    user_id = current_user["user_id"]
    request.app.state.kb.reset(user_id)
    return {"status": "cleared"}


async def _guard(awaitable):
    try:
        return await awaitable

    except Exception as exc:
        log.error(
            "request_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:1000],
        )

        status, message, headers = http_status(exc)

        raise HTTPException(
            status_code=status,
            detail=message,
            headers=headers,
        ) from exc
