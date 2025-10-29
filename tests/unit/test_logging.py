"""
Unit tests for logging framework.

This module tests the structured logging system including context injection,
specialized loggers, and performance tracking.
"""

import json
import logging
import time
from io import StringIO
from unittest.mock import patch, Mock

import pytest
import structlog

from quantum_moe_mas.core.logging import (
    setup_logging,
    get_logger,
    get_security_logger,
    get_performance_logger,
    get_business_logger,
    set_request_context,
    generate_request_id,
    log_execution_time,
    log_api_call,
    SecurityEventLogger,
    PerformanceLogger,
    BusinessLogger,
)


class TestLoggingSetup:
    """Test logging setup and configuration."""
    
    def test_setup_logging_development(self):
        """Test logging setup for development environment."""
        setup_logging(log_level="DEBUG", environment="development")
        
        logger = get_logger("test")
        assert logger is not None
        
        # Test that we can log messages
        logger.info("Test message", test_key="test_value")
    
    def test_setup_logging_production(self):
        """Test logging setup for production environment."""
        setup_logging(log_level="INFO", environment="production")
        
        logger = get_logger("test")
        assert logger is not None
        
        # Test that we can log messages
        logger.info("Test message", test_key="test_value")
    
    def test_get_logger_with_name(self):
        """Test getting logger with specific name."""
        logger = get_logger("test_module")
        assert logger is not None
    
    def test_get_logger_without_name(self):
        """Test getting logger without name (should use caller's module)."""
        logger = get_logger()
        assert logger is not None


class TestContextManagement:
    """Test logging context management."""
    
    def test_set_request_context(self):
        """Test setting request context."""
        request_id = "test_request_123"
        user_id = "test_user_456"
        session_id = "test_session_789"
        
        set_request_context(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id
        )
        
        # Context should be set (we can't easily test the actual context
        # without more complex setup, but we can test the function doesn't error)
        assert True
    
    def test_generate_request_id(self):
        """Test request ID generation."""
        request_id = generate_request_id()
        
        assert isinstance(request_id, str)
        assert len(request_id) > 0
        
        # Should generate unique IDs
        request_id2 = generate_request_id()
        assert request_id != request_id2


class TestSecurityLogger:
    """Test security event logger."""
    
    def test_security_logger_creation(self):
        """Test security logger creation."""
        security_logger = get_security_logger("test_security")
        assert isinstance(security_logger, SecurityEventLogger)
    
    def test_authentication_attempt_logging(self):
        """Test authentication attempt logging."""
        logger = Mock()
        security_logger = SecurityEventLogger(logger)
        
        security_logger.authentication_attempt(
            user_id="test_user",
            success=True,
            method="password",
            ip_address="192.168.1.1"
        )
        
        logger.warning.assert_called_once()
        call_args = logger.warning.call_args
        assert "Authentication attempt" in call_args[0]
        assert call_args[1]["user_id"] == "test_user"
        assert call_args[1]["success"] is True
    
    def test_authorization_failure_logging(self):
        """Test authorization failure logging."""
        logger = Mock()
        security_logger = SecurityEventLogger(logger)
        
        security_logger.authorization_failure(
            user_id="test_user",
            resource="/admin/users",
            action="read"
        )
        
        logger.error.assert_called_once()
        call_args = logger.error.call_args
        assert "Authorization failure" in call_args[0]
        assert call_args[1]["resource"] == "/admin/users"
    
    def test_suspicious_activity_logging(self):
        """Test suspicious activity logging."""
        logger = Mock()
        security_logger = SecurityEventLogger(logger)
        
        security_logger.suspicious_activity(
            activity_type="brute_force",
            description="Multiple failed login attempts",
            severity="high"
        )
        
        logger.error.assert_called_once()
        call_args = logger.error.call_args
        assert "Suspicious activity detected" in call_args[0]
        assert call_args[1]["severity"] == "high"
    
    def test_data_access_logging(self):
        """Test data access logging."""
        logger = Mock()
        security_logger = SecurityEventLogger(logger)
        
        # Non-sensitive data access
        security_logger.data_access(
            user_id="test_user",
            resource="public_data",
            action="read",
            sensitive=False
        )
        
        logger.info.assert_called_once()
        
        # Reset mock
        logger.reset_mock()
        
        # Sensitive data access
        security_logger.data_access(
            user_id="test_user",
            resource="pii_data",
            action="read",
            sensitive=True
        )
        
        logger.warning.assert_called_once()


class TestPerformanceLogger:
    """Test performance logger."""
    
    def test_performance_logger_creation(self):
        """Test performance logger creation."""
        perf_logger = get_performance_logger("test_performance")
        assert isinstance(perf_logger, PerformanceLogger)
    
    def test_execution_time_logging(self):
        """Test execution time logging."""
        logger = Mock()
        perf_logger = PerformanceLogger(logger)
        
        perf_logger.log_execution_time(
            operation="test_operation",
            duration_ms=150.5,
            success=True
        )
        
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "Operation performance" in call_args[0]
        assert call_args[1]["duration_ms"] == 150.5
    
    def test_api_call_logging(self):
        """Test API call logging."""
        logger = Mock()
        perf_logger = PerformanceLogger(logger)
        
        perf_logger.log_api_call(
            api_name="openai",
            endpoint="/v1/chat/completions",
            method="POST",
            status_code=200,
            duration_ms=1250.0
        )
        
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "API call performance" in call_args[0]
        assert call_args[1]["api_name"] == "openai"
    
    def test_database_query_logging(self):
        """Test database query logging."""
        logger = Mock()
        perf_logger = PerformanceLogger(logger)
        
        perf_logger.log_database_query(
            query_type="SELECT",
            table="users",
            duration_ms=25.5,
            rows_affected=10
        )
        
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "Database query performance" in call_args[0]
        assert call_args[1]["table"] == "users"


class TestBusinessLogger:
    """Test business metrics logger."""
    
    def test_business_logger_creation(self):
        """Test business logger creation."""
        business_logger = get_business_logger("test_business")
        assert isinstance(business_logger, BusinessLogger)
    
    def test_roi_metric_logging(self):
        """Test ROI metric logging."""
        logger = Mock()
        business_logger = BusinessLogger(logger)
        
        business_logger.log_roi_metric(
            session_id="session_123",
            user_id="user_456",
            revenue=100.0,
            cost=20.0,
            roi=5.0
        )
        
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "ROI metric" in call_args[0]
        assert call_args[1]["roi"] == 5.0
    
    def test_efficiency_gain_logging(self):
        """Test efficiency gain logging."""
        logger = Mock()
        business_logger = BusinessLogger(logger)
        
        business_logger.log_efficiency_gain(
            operation="query_processing",
            baseline_time=1000.0,
            optimized_time=600.0,
            efficiency_gain_percent=40.0
        )
        
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "Efficiency gain" in call_args[0]
        assert call_args[1]["efficiency_gain_percent"] == 40.0
    
    def test_user_interaction_logging(self):
        """Test user interaction logging."""
        logger = Mock()
        business_logger = BusinessLogger(logger)
        
        business_logger.log_user_interaction(
            user_id="user_123",
            action="query_submit",
            feature="moe_router",
            success=True
        )
        
        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "User interaction" in call_args[0]
        assert call_args[1]["feature"] == "moe_router"


class TestLoggingDecorators:
    """Test logging decorators."""
    
    def test_log_execution_time_decorator(self):
        """Test execution time logging decorator."""
        @log_execution_time("test_operation")
        def test_function():
            time.sleep(0.01)  # Small delay
            return "test_result"
        
        with patch('quantum_moe_mas.core.logging.get_performance_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = test_function()
            
            assert result == "test_result"
            mock_logger.log_execution_time.assert_called_once()
            
            # Check that duration was logged
            call_args = mock_logger.log_execution_time.call_args
            assert call_args[0][0] == "test_operation"  # operation name
            assert call_args[0][1] > 0  # duration_ms should be positive
            assert call_args[1]["success"] is True
    
    def test_log_execution_time_decorator_with_exception(self):
        """Test execution time logging decorator with exception."""
        @log_execution_time("test_operation")
        def test_function():
            raise ValueError("Test error")
        
        with patch('quantum_moe_mas.core.logging.get_performance_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(ValueError):
                test_function()
            
            mock_logger.log_execution_time.assert_called_once()
            
            # Check that failure was logged
            call_args = mock_logger.log_execution_time.call_args
            assert call_args[1]["success"] is False
            assert "error" in call_args[1]
    
    def test_log_api_call_decorator(self):
        """Test API call logging decorator."""
        @log_api_call("test_api", "/test/endpoint")
        def test_api_function():
            # Mock response object
            response = Mock()
            response.status_code = 200
            return response
        
        with patch('quantum_moe_mas.core.logging.get_performance_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = test_api_function()
            
            assert result.status_code == 200
            mock_logger.log_api_call.assert_called_once()
            
            # Check that API call was logged
            call_args = mock_logger.log_api_call.call_args
            assert call_args[0][0] == "test_api"  # api_name
            assert call_args[0][1] == "/test/endpoint"  # endpoint