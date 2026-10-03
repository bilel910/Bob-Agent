from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_file_encoding="utf-8")

    GROQ_API_KEY: str
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str
    TOGETHER_API_KEY: str
    OPENAI_API_KEY: str
    QDRANT_API_KEY: str | None
    QDRANT_URL: str
    QDRANT_PORT: str = "6333"
    QDRANT_HOST: str | None = None

    # The llama-3.x models this project shipped with have been decommissioned by
    # Groq. qwen3.8-27b handles every structured-output schema here and produces
    # natural German, so it stays the model for the user-facing reply.
    TEXT_MODEL_NAME: str = "qwen/qwen3.8-27b"
    # Groq rate limits are per model, so the helper calls (router, memory analysis,
    # image scenario) run on a DIFFERENT model than the user-facing answer. That
    # gives them their own 8k token/min budget instead of eating the one the reply
    # needs, which is what used to throttle every 4th message into a 429 + backoff.
    # gpt-oss-20b handles all three structured-output schemas here; it is gpt-oss-120b
    # that fails MemoryAnalysis with "model did not call a tool".
    SMALL_TEXT_MODEL_NAME: str = "openai/gpt-oss-20b"
    # Groq refuses a request outright ("Request too large ... output tokens per
    # minute") when no max_tokens is set, because the default expected output
    # exceeds the 1000 OTPM limit of TEXT_MODEL_NAME on the free tier.
    TEXT_MAX_TOKENS: int = 600
    STT_MODEL_NAME: str = "whisper-large-v3-turbo"
    TTS_MODEL_NAME: str = "eleven_flash_v2_5"
    # FLUX.1-schnell-Free was retired by Together ("non-serverless model"), and the
    # plain FLUX.1-schnell is gone too, so there is no 4-step distilled model left.
    # NOTE: every Together image model additionally needs "third-party data sharing"
    # enabled for the org, otherwise it 403s with third_party_data_sharing_blocked.
    # "auto"    - try Together, fall back to Pollinations (no API key) if it fails
    # "together" - Together only, surface the error
    # "pollinations" - skip Together entirely
    TTI_PROVIDER: str = "auto"
    TTI_MODEL_NAME: str = "black-forest-labs/FLUX.2-dev"
    # steps=4 only made sense for the distilled "schnell" model. Leaving this None
    # omits the parameter entirely so each model uses its own sane default.
    TTI_STEPS: int | None = None
    # NOTE: no vision-capable model is currently available on Groq for this key, so
    # image understanding is effectively disabled. Both interfaces already catch the
    # resulting error and continue without the image description.
    ITT_MODEL_NAME: str = "llama-3.2-90b-vision-preview"

    MEMORY_TOP_K: int = 3
    ROUTER_MESSAGES_TO_ANALYZE: int = 3
    # The router only ever picks a non-text workflow when the user explicitly asks
    # for one, so when the last message contains no image/audio cue we can skip the
    # router LLM call entirely and go straight to 'conversation'. Set to False to
    # always ask the model.
    ROUTER_FAST_PATH: bool = True
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 20
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 5

    GOOGLE_TOKEN_PATH: str = "secrets/google_token.json"
    TIMEZONE: str = "Europe/Berlin"   # change to your own, e.g. "Europe/Paris"

    SHORT_TERM_MEMORY_DB_PATH: str = "/app/data/memory.db"


settings = Settings()
