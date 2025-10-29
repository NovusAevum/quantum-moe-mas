"""
Pytest configuration and fixtures for Quantum MoE MAS tests.

This module provides common test fixtures and configuration
for the entire test suite.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from quantum_moe_mas.config.settings import Settings
from quantum_moe_mas.core.logging import setup_logging


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Set up logging for tests."""
    setup_logging(log_level="DEBUG", environment="testing")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with safe defaults."""
    # Set test environment variables
    test_env = {
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "JWT_SECRET_KEY": "test_secret_key_that_is_long_enough_for_testing_purposes",
        "DATABASE_URL": "sqlite:///test.db",
        "REDIS_URL": "redis://localhost:6379/15",  # Use different DB for tests
    }
    
    # Temporarily set environment variables
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        # Clear the settings cache and create new settings
        from quantum_moe_mas.config.settings import get_settings
        get_settings.cache_clear()
        settings = get_settings()
        yield settings
    finally:
        # Restore original environment variables
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        
        # Clear the cache again
        get_settings.cache_clear()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response from OpenAI"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Test response from Anthropic"
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing."""
    mock_client = Mock()
    mock_client.table.return_value = mock_client
    mock_client.select.return_value = mock_client
    mock_client.insert.return_value = mock_client
    mock_client.update.return_value = mock_client
    mock_client.delete.return_value = mock_client
    mock_client.execute.return_value = Mock(data=[])
    return mock_client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    mock_client = Mock()
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1
    mock_client.exists.return_value = False
    return mock_client


@pytest.fixture
def sample_query_data():
    """Sample query data for testing."""
    return {
        "text": "Analyze this cybersecurity threat",
        "domain": "cyber",
        "modalities": ["text"],
        "user_id": "test_user_123",
        "context": {
            "session_id": "test_session_456",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    }


@pytest.fixture
def sample_expert_data():
    """Sample expert data for testing."""
    return {
        "id": "claude_sonnet_4",
        "name": "Claude Sonnet 4",
        "type": "language_model",
        "api_endpoint": "https://api.anthropic.com/v1/messages",
        "capabilities": ["reasoning", "analysis", "coding"],
        "cost_per_token": 0.00001,
        "max_tokens": 4096,
        "confidence_score": 0.95,
        "load_factor": 0.3
    }


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "id": "doc_123",
        "content": "This is a sample document for testing purposes.",
        "modality": "text",
        "metadata": {
            "source": "test",
            "category": "sample",
            "tags": ["test", "document"]
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add security marker to tests in security/ directory
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)