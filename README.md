# Support Ticket Triage Service

LLM-based agent (LangGraph) behind a FastAPI REST API. Classifies incoming support
tickets, searches a mock knowledge base, and decides the next action — auto-respond,
route to a specialist, or escalate to a human. Side-effecting actions (`issue_refund`,
`open_incident_or_page_oncall`) always pause for explicit human approval before they
execute.

See `WRITEUP.md` for architecture, design decisions, and trade-offs.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

cp .env.example .env        # then fill in OPENAI_API_KEY
```

## Run

```bash
uvicorn api.main:app --port 8000
```

- Swagger UI: http://127.0.0.1:8000/api/v1/docs
- Health check (no auth): http://127.0.0.1:8000/health

If `API_KEY` is set in `.env`, every `/api/v1/*` request needs an `X-API-Key` header
matching it — click "Authorize" in Swagger, or pass `-H "X-API-Key: ..."` with curl.
Leave it empty to disable auth (dev default).

## Try it

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "customer", "content": "My payment failed, now I see THREE charges of $29.99 and still no Pro access."}
    ],
    "customer_metadata": {"plan": "free", "tenure_months": 4, "prior_tickets": 0}
  }'
```

The response's `decision.tool_calls` will show `issue_refund` as `pending_approval`.
Approve it:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/conversations/{conversation_id}/tool-calls/{tool_call_id}/approve \
  -H "Content-Type: application/json" \
  -d '{"reason": "verified in Stripe"}'
```

Then pull the full audit trail:

```bash
curl http://127.0.0.1:8000/api/v1/conversations/{conversation_id}
```

## Test

```bash
pytest          # 77 tests, all LLM calls mocked at the brain-function boundary -- no API key needed, no network
```

## Eval

```bash
python eval/run_eval.py    # runs the real agent (real OpenAI calls) over eval/labeled_tickets.json
```

Prints per-ticket pass/fail on urgency, language, action, tool selection, and
KB-routing against hand-labeled expectations, plus an LLM-as-judge pass on any
reply that's actually grounded in cited KB docs. Costs a handful of real API calls
(gpt-4o-mini) — needs `OPENAI_API_KEY` set.

## Project layout

```
agents/           LangGraph orchestrator, prompts, tools, knowledge-base search
api/               FastAPI routes, request/response + error schemas
storage/           SQLite schema + data access
data/faq/          Mock knowledge-base docs
eval/              Labeled tickets + offline eval harness
tests/             pytest suite (mocked LLM boundary)
```
