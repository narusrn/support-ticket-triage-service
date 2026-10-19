from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    SUPERVISOR_MODEL: str = "gpt-4o-mini"

    # Persistence
    DATABASE_PATH: str = "./data/app.db"

    # Agent context engineering
    CONTEXT_TOKEN_THRESHOLD: int = 2000
    CONTEXT_KEEP_TOKENS: int = 800
    AGENTS_CACHE_TTL_SECONDS: int = 10800
    MEMORY_DIR: str = "./memory"
    MAX_STEPS: int = 12
    AUTO_RESPOND_CONFIDENCE_THRESHOLD: float = 0.75

    # Logging / agent trace
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"
    LOG_FILE: str = "app.log"
    TRACE_LOG_FILE: str = "./logs/agent_trace.jsonl"
    RUN_LOG_FILE: str = "./logs/runs.jsonl"

    ENV: str = "development"


settings = Settings()

# Static USD / 1K-token price table (input, output) -- only used to compute cost_usd
# in the run-summary log (logs/runs.jsonl). Not a live pricing lookup; update by hand
# if the models in use change.
MODEL_PRICES_USD_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}
