from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.runner import Runner
from api.routes import conversations, metrics, tickets
from api.schemas.api import ErrorDetail, ErrorResponse
from api.security import verify_api_key
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


# /health stays open (infra/monitoring checks shouldn't need a key); every real endpoint
# under /api/v1 requires X-API-Key, enforced once here rather than per-route.
app.include_router(tickets.router, prefix="/api/v1", tags=["tickets"], dependencies=[Depends(verify_api_key)])
app.include_router(conversations.router, prefix="/api/v1", tags=["conversations"], dependencies=[Depends(verify_api_key)])
# No router-level dependency here: /metrics needs the same X-API-Key guard as every other
# endpoint, but /dashboard is a plain HTML page a browser navigates to directly (no way
# to attach a custom header to that), so it carries no auth dependency of its own -- see
# api/routes/metrics.py's own per-route `dependencies=` on get_metrics for where the guard
# actually lives.
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])
