"""
Configuration settings for PaperAgent
"""

from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # Application
    app_name: str = "PaperAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql://paperagent:paperagent_password@localhost:5432/paperagent"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # LLM Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"

    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    default_llm_provider: str = "ollama"

    # LLM Parameters
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9

    # Literature Collection
    use_proxy: bool = False
    proxy_url: Optional[str] = None
    arxiv_max_results: int = 50

    # File Storage
    data_dir: str = "./data"
    papers_dir: str = "./data/papers"
    experiments_dir: str = "./data/experiments"
    literature_dir: str = "./data/literature"
    outputs_dir: str = "./data/outputs"

    # Security
    secret_key: str = "change-this-secret-key-in-production"
    allowed_hosts: List[str] = ["localhost", "127.0.0.1"]

    # Task Queue
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60

    # Journal Templates
    journal_templates_dir: str = "./paperagent/templates/journals"

    # Concurrent Tasks
    max_concurrent_tasks: int = 5

    # Session
    session_timeout: int = 3600

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.data_dir,
            self.papers_dir,
            self.experiments_dir,
            self.literature_dir,
            self.outputs_dir,
            self.journal_templates_dir,
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Global settings instance
settings = Settings()
