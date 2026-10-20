from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.runner import Runner
from api.routes import conversations, tickets
from api.schemas.api import ErrorDetail, ErrorResponse
from config import settings
from logging_conf import configure_logging
from storage import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    db.init_db(settings.DATABASE_PATH)
    conn = db.get_connection(settings.DATABASE_PATH)
    # AsyncSqliteSaver is itself an async context manager -- open ONE connection for the
    # app's whole lifetime, not a new one per request. agents/runner.py reads this off
    # app.state instead of opening its own.
    async with AsyncSqliteSaver.from_conn_string(settings.DATABASE_PATH) as checkpointer:
        app.state.conn = conn
        app.state.runner = Runner(conn, checkpointer)
        yield
    conn.close()


app = FastAPI(
    title="Support Ticket Triage Service",
    description=(
        "Wraps an LLM-based triage agent (LangGraph) behind a REST API. Classifies incoming "
        "support tickets, searches a mock knowledge base, and decides the next action -- "
        "auto-responding, routing to a specialist, or escalating to a human. Side-effecting "
        "actions (refunds, incident paging) always require explicit human approval via "
        "/tool-calls/{id}/approve before they execute."
    ),
    version="1.0.0",
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = f"req_{uuid4().hex[:8]}"
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(error=ErrorDetail(**detail)).model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error=ErrorDetail(code="validation_error", message=str(exc.errors()))).model_dump(),
    )


@app.get("/health", tags=["meta"], summary="Liveness/readiness check")
async def health():
    db_ok = db.ping(settings.DATABASE_PATH)
    checks = {"database": db_ok, "openai_api_key_set": bool(settings.OPENAI_API_KEY)}
    healthy = all(checks.values())
    return JSONResponse(status_code=200 if healthy else 503, content={"status": "ok" if healthy else "degraded", "checks": checks})


app.include_router(tickets.router, prefix="/api/v1", tags=["tickets"])
app.include_router(conversations.router, prefix="/api/v1", tags=["conversations"])
