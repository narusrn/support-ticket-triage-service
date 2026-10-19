import pytest

from agents.llm import (
    PROMPTS_DIR,
    LLMUnavailableError,
    call_with_retry,
    get_supervisor_prompt,
    load_agents,
)


def test_load_agents_parses_real_prompt_files():
    agents = load_agents(PROMPTS_DIR)
    assert set(agents) == {"triage_classifier", "kb_retriever", "action_decider"}

    classifier = agents["triage_classifier"]
    assert classifier.description.startswith("classifies urgency")
    assert classifier.version == "1.0.0"
    assert "## Changelog" in classifier.prompt


def test_underscore_prefixed_file_is_excluded_from_registry():
    agents = load_agents(PROMPTS_DIR)
    assert "_supervisor" not in agents
    assert "supervisor" not in agents  # it has no frontmatter name at all, shouldn't appear under any key


def test_supervisor_prompt_loads_as_plain_text_not_a_registered_agent():
    text = get_supervisor_prompt(PROMPTS_DIR)
    assert "handoff_to_agent" in text
    assert len(text) > 0


def test_load_agents_parses_frontmatter_from_a_fixture_dir(tmp_path):
    (tmp_path / "widget_helper.md").write_text(
        "---\nname: widget_helper\ndescription: helps with widgets\nversion: 2.3.1\n---\n\nBody text here.\n",
        encoding="utf-8",
    )
    (tmp_path / "_internal.md").write_text("no frontmatter, should be skipped\n", encoding="utf-8")

    agents = load_agents(tmp_path)
    assert set(agents) == {"widget_helper"}
    assert agents["widget_helper"].version == "2.3.1"
    assert agents["widget_helper"].prompt == "Body text here."


def test_call_with_retry_succeeds_after_transient_failures():
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("simulated transient error")
        return "ok"

    result = call_with_retry(flaky, max_retries=2, backoff_seconds=0)
    assert result == "ok"
    assert attempts["count"] == 3


def test_call_with_retry_gives_up_after_max_retries():
    attempts = {"count": 0}

    def always_fails():
        attempts["count"] += 1
        raise TimeoutError("simulated permanent error")

    with pytest.raises(LLMUnavailableError):
        call_with_retry(always_fails, max_retries=2, backoff_seconds=0)

    assert attempts["count"] == 3  # 1 initial try + 2 retries, no more
