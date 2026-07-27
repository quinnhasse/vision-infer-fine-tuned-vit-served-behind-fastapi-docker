"""Runtime configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Server settings.

    All values can be overridden via environment variables with the same name.
    """

    model_name: str = "google/vit-base-patch16-224"
    model_checkpoint: str = ""  # path to fine-tuned weights; uses hub model if empty
    top_k: int = 5
    batch_timeout_ms: int = 50  # max wait before flushing the batch queue
    max_batch_size: int = 32
    api_key: str = "changeme"  # set this via API_KEY env var in production
    device: str = "cpu"  # "cuda" when a GPU is available

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
