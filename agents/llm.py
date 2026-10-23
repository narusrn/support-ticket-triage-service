"""Prompt registry (load_agents/get_agents) and the LLM client factory with
retry/backoff. No live OpenAI call happens in this module's own logic --
it only constructs clients and wraps calls; agents/orchestrator.py (built
later) is what actually invokes the model."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypeVar

import yaml
from langchain_openai import ChatOpenAI

from config import MODEL_PRICES_USD_PER_1K, settings

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class AgentSpec:
    name: str
    description: str
    version: str
    prompt: str
    path: Path


def _parse_prompt_file(path: Path) -> Optional[AgentSpec]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None  # no frontmatter -- e.g. _supervisor.md, not a routable specialist

    _, frontmatter_raw, body = text.split("---", 2)
    frontmatter = yaml.safe_load(frontmatter_raw) or {}

    return AgentSpec(
        name=frontmatter["name"],
        description=frontmatter["description"],
        version=str(frontmatter.get("version", "0.0.0")),
        prompt=body.strip(),
        path=path,
    )


def load_agents(prompts_dir: Path = PROMPTS_DIR) -> dict[str, AgentSpec]:
    """Globs prompts_dir for *.md files. Files starting with '_' (e.g.
    _supervisor.md) are skipped -- reserved for the supervisor's own prompt,
    never registered as a specialist a handoff can target."""
    agents: dict[str, AgentSpec] = {}
    for path in sorted(prompts_dir.glob("*.md")):
        if path.stem.startswith("_"):
            continue
        spec = _parse_prompt_file(path)
        if spec is not None:
            agents[spec.name] = spec
    return agents


_agents_cache: Optional[dict[str, AgentSpec]] = None
_agents_cache_at: float = 0.0


def get_agents(prompts_dir: Path = PROMPTS_DIR) -> dict[str, AgentSpec]:
    """Same as load_agents(), but cached for AGENTS_CACHE_TTL_SECONDS so editing
    a specialist's .md doesn't require a process restart to take effect, while
    avoiding a filesystem glob + YAML parse on every single call."""
    global _agents_cache, _agents_cache_at
    now = time.monotonic()
    if _agents_cache is None or (now - _agents_cache_at) > settings.AGENTS_CACHE_TTL_SECONDS:
        _agents_cache = load_agents(prompts_dir)
        _agents_cache_at = now
    return _agents_cache


def get_supervisor_prompt(prompts_dir: Path = PROMPTS_DIR) -> str:
    return (prompts_dir / "_supervisor.md").read_text(encoding="utf-8").strip()


def make_llm_client(model: str) -> ChatOpenAI:
    return ChatOpenAI(model=model, api_key=settings.OPENAI_API_KEY)


def extract_usage(response, model: str) -> dict:
    """Pulls real token usage off a raw AIMessage (LangChain's standard
    `usage_metadata` attribute) and prices it against MODEL_PRICES_USD_PER_1K --
    the one thing that table was added for but, until now, nothing ever read.
    A fake test response with no `usage_metadata` attribute just yields zeros,
    not a crash, via getattr's default."""
    meta = getattr(response, "usage_metadata", None) or {}
    prompt = meta.get("input_tokens", 0)
    completion = meta.get("output_tokens", 0)
    total = meta.get("total_tokens", prompt + completion)
    prices = MODEL_PRICES_USD_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    cost_usd = (prompt / 1000) * prices["input"] + (completion / 1000) * prices["output"]
    return {"model": model, "prompt": prompt, "completion": completion, "total": total, "cost_usd": cost_usd}


class LLMUnavailableError(Exception):
    """Raised when an LLM call exhausts its retry budget. agents/runner.py
    (built later) catches exactly this error type and ends the run with
    stopped_reason="llm_unavailable" instead of crashing."""


T = TypeVar("T")


def call_with_retry(fn: Callable[[], T], *, max_retries: int = 2, backoff_seconds: float = 0.5) -> T:
    """Wraps a single LLM call with retry/backoff on transient failure
    (rate-limit/timeout/5xx in real use). After max_retries+1 total attempts,
    raises LLMUnavailableError wrapping the last exception, instead of letting
    the original exception propagate."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- intentionally broad: any failure counts as transient here
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_seconds * (2 ** attempt))
    raise LLMUnavailableError(str(last_exc)) from last_exc
