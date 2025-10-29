"""
Enterprise-grade configuration management with environment validation.

This module provides type-safe configuration management using Pydantic,
with comprehensive validation and environment-specific settings.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    Field,
    validator,
    SecretStr,
    AnyHttpUrl,
    PostgresDsn,
    RedisDsn,
)
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # Supabase Configuration
    supabase_url: Optional[AnyHttpUrl] = Field(None, env="SUPABASE_URL")
    supabase_key: Optional[SecretStr] = Field(None, env="SUPABASE_KEY")
    supabase_service_key: Optional[SecretStr] = Field(None, env="SUPABASE_SERVICE_KEY")
    
    # PostgreSQL Configuration
    database_url: Optional[PostgresDsn] = Field(None, env="DATABASE_URL")
    db_host: str = Field("localhost", env="DB_HOST")
    db_port: int = Field(5432, env="DB_PORT")
    db_name: str = Field("quantum_moe_mas", env="DB_NAME")
    db_user: str = Field("postgres", env="DB_USER")
    db_password: SecretStr = Field(SecretStr(""), env="DB_PASSWORD")
    db_ssl_mode: str = Field("prefer", env="DB_SSL_MODE")
    
    # Redis Configuration
    redis_url: Optional[RedisDsn] = Field(None, env="REDIS_URL")
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_password: Optional[SecretStr] = Field(None, env="REDIS_PASSWORD")
    
    @validator("database_url", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        """Assemble database URL from components if not provided."""
        if isinstance(v, str):
            return v
        
        password = values.get("db_password")
        password_str = password.get_secret_value() if password else ""
        
        return PostgresDsn.build(
            scheme="postgresql",
            user=values.get("db_user"),
            password=password_str,
            host=values.get("db_host"),
            port=str(values.get("db_port")),
            path=f"/{values.get('db_name') or ''}",
        )
    
    @validator("redis_url", pre=True)
    def assemble_redis_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        """Assemble Redis URL from components if not provided."""
        if isinstance(v, str):
            return v
        
        password = values.get("redis_password")
        password_str = password.get_secret_value() if password else None
        
        return RedisDsn.build(
            scheme="redis",
            password=password_str,
            host=values.get("redis_host"),
            port=str(values.get("redis_port")),
            path=f"/{values.get('redis_db') or 0}",
        )


class AIAPISettings(BaseSettings):
    """AI/ML API configuration settings."""
    
    # OpenAI Configuration
    openai_api_key: Optional[SecretStr] = Field(None, env="OPENAI_API_KEY")
    openai_org_id: Optional[str] = Field(None, env="OPENAI_ORG_ID")
    
    # Anthropic Configuration
    anthropic_api_key: Optional[SecretStr] = Field(None, env="ANTHROPIC_API_KEY")
    
    # Hugging Face Configuration
    huggingface_api_key: Optional[SecretStr] = Field(None, env="HUGGINGFACE_API_KEY")
    
    # Google AI Studio
    google_ai_api_key: Optional[SecretStr] = Field(None, env="GOOGLE_AI_API_KEY")
    
    # Additional AI APIs
    groq_api_key: Optional[SecretStr] = Field(None, env="GROQ_API_KEY")
    cerebras_api_key: Optional[SecretStr] = Field(None, env="CEREBRAS_API_KEY")
    deepseek_api_key: Optional[SecretStr] = Field(None, env="DEEPSEEK_API_KEY")
    cohere_api_key: Optional[SecretStr] = Field(None, env="COHERE_API_KEY")
    replicate_api_token: Optional[SecretStr] = Field(None, env="REPLICATE_API_TOKEN")
    stability_api_key: Optional[SecretStr] = Field(None, env="STABILITY_API_KEY")


class QuantumSettings(BaseSettings):
    """Quantum computing configuration settings."""
    
    ibm_quantum_token: Optional[SecretStr] = Field(None, env="IBM_QUANTUM_TOKEN")
    ibm_quantum_instance: str = Field("ibm_quantum", env="IBM_QUANTUM_INSTANCE")
    ibm_quantum_backend: str = Field("simulator_statevector", env="IBM_QUANTUM_BACKEND")


class CloudSettings(BaseSettings):
    """Cloud services configuration settings."""
    
    # AWS Configuration
    aws_access_key_id: Optional[SecretStr] = Field(None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[SecretStr] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field("us-east-1", env="AWS_DEFAULT_REGION")
    aws_s3_bucket: Optional[str] = Field(None, env="AWS_S3_BUCKET")
    
    # Google Cloud Configuration
    google_cloud_project_id: Optional[str] = Field(None, env="GOOGLE_CLOUD_PROJECT_ID")
    google_application_credentials: Optional[Path] = Field(None, env="GOOGLE_APPLICATION_CREDENTIALS")
    
    # Azure Configuration
    azure_subscription_id: Optional[SecretStr] = Field(None, env="AZURE_SUBSCRIPTION_ID")
    azure_client_id: Optional[SecretStr] = Field(None, env="AZURE_CLIENT_ID")
    azure_client_secret: Optional[SecretStr] = Field(None, env="AZURE_CLIENT_SECRET")
    azure_tenant_id: Optional[SecretStr] = Field(None, env="AZURE_TENANT_ID")


class MarketingSettings(BaseSettings):
    """Marketing and CRM configuration settings."""
    
    # HubSpot Configuration
    hubspot_api_key: Optional[SecretStr] = Field(None, env="HUBSPOT_API_KEY")
    hubspot_client_id: Optional[str] = Field(None, env="HUBSPOT_CLIENT_ID")
    hubspot_client_secret: Optional[SecretStr] = Field(None, env="HUBSPOT_CLIENT_SECRET")


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    # JWT Configuration
    jwt_secret_key: SecretStr = Field(..., env="JWT_SECRET_KEY", min_length=32)
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Encryption Keys
    encryption_key: Optional[SecretStr] = Field(None, env="ENCRYPTION_KEY")
    fernet_key: Optional[SecretStr] = Field(None, env="FERNET_KEY")
    
    # API Security
    api_rate_limit_per_minute: int = Field(100, env="API_RATE_LIMIT_PER_MINUTE")
    api_rate_limit_per_hour: int = Field(1000, env="API_RATE_LIMIT_PER_HOUR")


class WebSettings(BaseSettings):
    """Web application configuration settings."""
    
    # Streamlit Configuration
    streamlit_server_port: int = Field(8501, env="STREAMLIT_SERVER_PORT")
    streamlit_server_address: str = Field("localhost", env="STREAMLIT_SERVER_ADDRESS")
    streamlit_theme_base: str = Field("light", env="STREAMLIT_THEME_BASE")
    
    # FastAPI Configuration
    fastapi_host: str = Field("0.0.0.0", env="FASTAPI_HOST")
    fastapi_port: int = Field(8000, env="FASTAPI_PORT")
    fastapi_reload: bool = Field(True, env="FASTAPI_RELOAD")
    fastapi_workers: int = Field(1, env="FASTAPI_WORKERS")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration settings."""
    
    # Sentry Configuration
    sentry_dsn: Optional[AnyHttpUrl] = Field(None, env="SENTRY_DSN")
    sentry_environment: str = Field("development", env="SENTRY_ENVIRONMENT")
    
    # Prometheus Configuration
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    metrics_enabled: bool = Field(True, env="METRICS_ENABLED")
    
    # New Relic Configuration
    new_relic_license_key: Optional[SecretStr] = Field(None, env="NEW_RELIC_LICENSE_KEY")
    new_relic_app_name: str = Field("quantum-moe-mas", env="NEW_RELIC_APP_NAME")


class PerformanceSettings(BaseSettings):
    """Performance and caching configuration settings."""
    
    # Cache Configuration
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL_SECONDS")
    cache_max_size: int = Field(1000, env="CACHE_MAX_SIZE")
    enable_query_cache: bool = Field(True, env="ENABLE_QUERY_CACHE")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")


class BusinessSettings(BaseSettings):
    """Business intelligence and ROI configuration settings."""
    
    # ROI Tracking
    roi_target_icm_per_session: float = Field(0.50, env="ROI_TARGET_ICM_PER_SESSION")
    efficiency_gain_target_percent: float = Field(25.0, env="EFFICIENCY_GAIN_TARGET_PERCENT")
    cost_optimization_target_percent: float = Field(40.0, env="COST_OPTIMIZATION_TARGET_PERCENT")
    
    # Analytics
    analytics_enabled: bool = Field(True, env="ANALYTICS_ENABLED")
    analytics_retention_days: int = Field(90, env="ANALYTICS_RETENTION_DAYS")


class Settings(BaseSettings):
    """Main application settings with comprehensive configuration management."""
    
    # Environment Configuration
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(True, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Nested Settings
    database: DatabaseSettings = DatabaseSettings()
    ai_apis: AIAPISettings = AIAPISettings()
    quantum: QuantumSettings = QuantumSettings()
    cloud: CloudSettings = CloudSettings()
    marketing: MarketingSettings = MarketingSettings()
    security: SecuritySettings = SecuritySettings()
    web: WebSettings = WebSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    performance: PerformanceSettings = PerformanceSettings()
    business: BusinessSettings = BusinessSettings()
    
    # Testing Configuration
    test_database_url: Optional[PostgresDsn] = Field(None, env="TEST_DATABASE_URL")
    pytest_timeout: int = Field(300, env="PYTEST_TIMEOUT")
    
    # Development Tools
    enable_debug_toolbar: bool = Field(True, env="ENABLE_DEBUG_TOOLBAR")
    enable_profiling: bool = Field(False, env="ENABLE_PROFILING")
    mock_external_apis: bool = Field(False, env="MOCK_EXTERNAL_APIS")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
        extra = "forbid"  # Prevent extra fields
    
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        allowed_environments = {"development", "staging", "production"}
        if v.lower() not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v.lower()
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level value."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()
    
    def validate_production_settings(self) -> None:
        """Validate production-specific settings."""
        if self.environment == "production":
            # Ensure critical production settings are configured
            if not self.security.jwt_secret_key:
                raise ValueError("JWT secret key is required in production")
            
            # Ensure debug is disabled in production
            if self.debug:
                raise ValueError("Debug mode must be disabled in production")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    @property
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.environment == "staging"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def get_database_url(self, for_testing: bool = False) -> str:
        """Get the appropriate database URL."""
        if for_testing and self.test_database_url:
            return str(self.test_database_url)
        return str(self.database.database_url)
    
    def get_redis_url(self) -> str:
        """Get the Redis connection URL."""
        return str(self.database.redis_url)
    
    def get_secret_value(self, secret: SecretStr) -> str:
        """Safely get secret value."""
        return secret.get_secret_value() if secret else ""


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    This function uses LRU cache to ensure settings are loaded only once
    and reused throughout the application lifecycle.
    
    Returns:
        Settings: The application settings instance.
    """
    return Settings()


def validate_required_settings() -> None:
    """
    Validate that all required settings are properly configured.
    
    Raises:
        ValueError: If required settings are missing or invalid.
    """
    settings = get_settings()
    
    # Check critical settings based on environment
    if settings.is_production:
        required_production_settings = [
            (settings.security.jwt_secret_key, "JWT_SECRET_KEY"),
            (settings.database.database_url, "DATABASE_URL"),
        ]
        
        for setting_value, setting_name in required_production_settings:
            if not setting_value:
                raise ValueError(f"{setting_name} is required in production environment")
    
    # Validate API keys if AI features are enabled
    ai_apis = settings.ai_apis
    if not any([
        ai_apis.openai_api_key,
        ai_apis.anthropic_api_key,
        ai_apis.huggingface_api_key,
    ]):
        raise ValueError("At least one AI API key must be configured")


# Export settings validation function
__all__ = ["Settings", "get_settings", "validate_required_settings"]