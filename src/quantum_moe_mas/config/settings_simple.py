"""
Simplified configuration management for validation.

This is a temporary simplified version for testing purposes.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    """Simplified application settings."""
    
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # Database
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    # AI APIs
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Security
    jwt_secret_key: str = "development-secret-key-min-32-chars-long"
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            database_url=os.getenv("DATABASE_URL"),
            redis_url=os.getenv("REDIS_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            jwt_secret_key=os.getenv("JWT_SECRET_KEY", "development-secret-key-min-32-chars-long"),
        )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
