from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from api.schemas.api import ErrorResponse, MetricsResponse
from api.security import verify_api_key
from storage import db

router = APIRouter()

_DASHBOARD_HTML = (Path(__file__).resolve().parent.parent / "static" / "dashboard.html").read_text(encoding="utf-8")


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Service-wide usage metrics",
    description=(
        "Requests/tokens/cost/latency rolled up across EVERY conversation (not just one -- "
        "see `GET /conversations/{id}` for a single conversation's usage), plus "
        "action/urgency/tool-call breakdowns and an hourly timeseries. This is the data "
        "`GET /dashboard` renders; call it directly for your own tooling/CI checks instead "
        "of scraping the HTML page.\n\n"
        "All totals are per **run** (one graph invocation), not per conversation -- "
        "approving a tool call resumes the graph, which is a second run with the same "
        "`final_action`, so it's counted again here.\n\n"
        "`timeseries` is bucketed per minute in Asia/Bangkok local time (UTC+7), not UTC -- "
        "so the \"requests over time\" graph shows real movement during a normal testing "
        "session instead of one flat hourly bar."
    ),
    responses={
        401: {"model": ErrorResponse},
        200: {"content": {"application/json": {"example": {
            "total_conversations": 8, "total_runs": 11, "total_tokens": 61603,
            "total_cost_usd": 0.01043, "avg_latency_ms": 3844.9,
            "action_breakdown": {"auto_respond": 4, "escalate_to_human": 6, "route_to_specialist": 1},
            "urgency_breakdown": {"critical": 4, "medium": 5, "low": 2},
            "tool_call_status_breakdown": {"executed": 2, "rejected": 1},
            "timeseries": [{"bucket": "2026-10-23T16:22:00", "runs": 11, "tokens": 61603}],
        }}}},
    },
    dependencies=[Depends(verify_api_key)],
)
async def get_metrics(request: Request) -> MetricsResponse:
    return MetricsResponse(**db.get_global_metrics(request.app.state.conn))


@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Metrics dashboard (HTML)",
    description="A small charts-and-stat-tiles page that fetches GET /metrics client-side. "
                "Deliberately NOT behind X-API-Key auth -- a plain browser navigation can't "
                "send a custom header, so the page itself stays reachable and asks for the "
                "key in a field instead, same as Swagger's own 'Authorize' button; every "
                "/metrics call it then makes IS still gated by the key, same as any other client.",
)
async def get_dashboard() -> HTMLResponse:
    return HTMLResponse(_DASHBOARD_HTML)
