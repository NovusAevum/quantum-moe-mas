"""
Unit tests for configuration management.

This module tests the configuration system including settings validation,
environment variable loading, and configuration validation.
"""

import os
import pytest
from pydantic import ValidationError

from quantum_moe_mas.config.settings import (
    Settings,
    DatabaseSettings,
    SecuritySettings,
    validate_required_settings,
)


class TestDatabaseSettings:
    """Test database configuration settings."""
    
    def test_database_url_assembly(self):
        """Test database URL assembly from components."""
        settings = DatabaseSettings(
            db_host="localhost",
            db_port=5432,
            db_name="test_db",
            db_user="test_user",
            db_password="test_password"
        )
        
        assert "postgresql://" in str(settings.database_url)
        assert "localhost:5432" in str(settings.database_url)
        assert "test_db" in str(settings.database_url)
    
    def test_redis_url_assembly(self):
        """Test Redis URL assembly from components."""
        settings = DatabaseSettings(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0
        )
        
        assert "redis://" in str(settings.redis_url)
        assert "localhost:6379" in str(settings.redis_url)


class TestSecuritySettings:
    """Test security configuration settings."""
    
    def test_jwt_secret_key_validation(self):
        """Test JWT secret key length validation."""
        # Valid key (32+ characters)
        valid_key = "a" * 32
        settings = SecuritySettings(jwt_secret_key=valid_key)
        assert settings.jwt_secret_key.get_secret_value() == valid_key
        
        # Invalid key (too short)
        with pytest.raises(ValidationError):
            SecuritySettings(jwt_secret_key="short")
    
    def test_default_security_settings(self):
        """Test default security settings."""
        settings = SecuritySettings(jwt_secret_key="a" * 32)
        
        assert settings.jwt_algorithm == "HS256"
        assert settings.jwt_access_token_expire_minutes == 30
        assert settings.api_rate_limit_per_minute == 100


class TestMainSettings:
    """Test main application settings."""
    
    def test_environment_validation(self):
        """Test environment validation."""
        # Valid environments
        for env in ["development", "staging", "production"]:
            settings = Settings(
                environment=env,
                jwt_secret_key="a" * 32
            )
            assert settings.environment == env
        
        # Invalid environment
        with pytest.raises(ValidationError):
            Settings(
                environment="invalid",
                jwt_secret_key="a" * 32
            )
    
    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(
                log_level=level,
                jwt_secret_key="a" * 32
            )
            assert settings.log_level == level
        
        # Invalid log level
        with pytest.raises(ValidationError):
            Settings(
                log_level="INVALID",
                jwt_secret_key="a" * 32
            )
    
    def test_production_validation(self):
        """Test production-specific validation."""
        # Production with debug enabled should fail
        with pytest.raises(ValidationError):
            Settings(
                environment="production",
                debug=True,
                jwt_secret_key="a" * 32
            )
        
        # Valid production settings
        settings = Settings(
            environment="production",
            debug=False,
            jwt_secret_key="a" * 32
        )
        assert settings.is_production
        assert not settings.debug
    
    def test_environment_properties(self):
        """Test environment property methods."""
        dev_settings = Settings(
            environment="development",
            jwt_secret_key="a" * 32
        )
        assert dev_settings.is_development
        assert not dev_settings.is_staging
        assert not dev_settings.is_production
        
        prod_settings = Settings(
            environment="production",
            debug=False,
            jwt_secret_key="a" * 32
        )
        assert not prod_settings.is_development
        assert not prod_settings.is_staging
        assert prod_settings.is_production
    
    def test_database_url_methods(self):
        """Test database URL helper methods."""
        settings = Settings(
            jwt_secret_key="a" * 32,
            test_database_url="postgresql://test:test@localhost/test_db"
        )
        
        # Regular database URL
        db_url = settings.get_database_url()
        assert "postgresql://" in db_url
        
        # Test database URL
        test_db_url = settings.get_database_url(for_testing=True)
        assert "test_db" in test_db_url
    
    def test_secret_value_helper(self):
        """Test secret value helper method."""
        settings = Settings(jwt_secret_key="secret_key_123")
        
        secret_value = settings.get_secret_value(settings.security.jwt_secret_key)
        assert secret_value == "secret_key_123"


class TestSettingsValidation:
    """Test settings validation functions."""
    
    def test_validate_required_settings_development(self, monkeypatch):
        """Test validation in development environment."""
        # Set up development environment
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("JWT_SECRET_KEY", "a" * 32)
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        # Should not raise any exceptions
        validate_required_settings()
    
    def test_validate_required_settings_production_missing_jwt(self, monkeypatch):
        """Test validation in production with missing JWT key."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DEBUG", "false")
        # Don't set JWT_SECRET_KEY
        
        with pytest.raises(ValueError, match="JWT_SECRET_KEY is required"):
            validate_required_settings()
    
    def test_validate_required_settings_no_ai_apis(self, monkeypatch):
        """Test validation with no AI API keys configured."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("JWT_SECRET_KEY", "a" * 32)
        # Don't set any AI API keys
        
        with pytest.raises(ValueError, match="At least one AI API key must be configured"):
            validate_required_settings()
    
    def test_validate_required_settings_with_ai_apis(self, monkeypatch):
        """Test validation with AI API keys configured."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("JWT_SECRET_KEY", "a" * 32)
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
        
        # Should not raise any exceptions
        validate_required_settings()