"""
Enterprise-grade structured logging framework.

This module provides comprehensive logging capabilities with structured JSON output,
context injection, performance metrics, and security event logging for enterprise
monitoring and observability.
"""

import json
import logging
import logging.config
import sys
import time
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from uuid import uuid4

import structlog
from structlog.types import FilteringBoundLogger

# Context variables for request tracking
request_id_context: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
session_id_context: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


class SecurityEventLogger:
    """Specialized logger for security events."""
    
    def __init__(self, logger: FilteringBoundLogger):
        self.logger = logger
    
    def authentication_attempt(
        self,
        user_id: Optional[str] = None,
        success: bool = False,
        method: str = "unknown",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Log authentication attempts."""
        self.logger.warning(
            "Authentication attempt",
            event_type="authentication",
            user_id=user_id,
            success=success,
            method=method,
            ip_address=ip_address,
            user_agent=user_agent,
            **kwargs
        )
    
    def authorization_failure(
        self,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Log authorization failures."""
        self.logger.error(
            "Authorization failure",
            event_type="authorization_failure",
            user_id=user_id,
            resource=resource,
            action=action,
            **kwargs
        )
    
    def suspicious_activity(
        self,
        activity_type: str,
        description: str,
        severity: str = "medium",
        **kwargs: Any
    ) -> None:
        """Log suspicious activities."""
        self.logger.error(
            "Suspicious activity detected",
            event_type="suspicious_activity",
            activity_type=activity_type,
            description=description,
            severity=severity,
            **kwargs
        )
    
    def data_access(
        self,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: str = "read",
        sensitive: bool = False,
        **kwargs: Any
    ) -> None:
        """Log data access events."""
        log_level = "warning" if sensitive else "info"
        getattr(self.logger, log_level)(
            "Data access",
            event_type="data_access",
            user_id=user_id,
            resource=resource,
            action=action,
            sensitive=sensitive,
            **kwargs
        )


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self, logger: FilteringBoundLogger):
        self.logger = logger
    
    def log_execution_time(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **kwargs: Any
    ) -> None:
        """Log operation execution time."""
        self.logger.info(
            "Operation performance",
            event_type="performance",
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            **kwargs
        )
    
    def log_api_call(
        self,
        api_name: str,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        **kwargs: Any
    ) -> None:
        """Log API call performance."""
        self.logger.info(
            "API call performance",
            event_type="api_performance",
            api_name=api_name,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_database_query(
        self,
        query_type: str,
        table: Optional[str] = None,
        duration_ms: float = 0.0,
        rows_affected: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Log database query performance."""
        self.logger.info(
            "Database query performance",
            event_type="db_performance",
            query_type=query_type,
            table=table,
            duration_ms=duration_ms,
            rows_affected=rows_affected,
            **kwargs
        )


class BusinessLogger:
    """Specialized logger for business events and metrics."""
    
    def __init__(self, logger: FilteringBoundLogger):
        self.logger = logger
    
    def log_roi_metric(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        revenue: float = 0.0,
        cost: float = 0.0,
        roi: float = 0.0,
        **kwargs: Any
    ) -> None:
        """Log ROI metrics."""
        self.logger.info(
            "ROI metric",
            event_type="business_roi",
            session_id=session_id,
            user_id=user_id,
            revenue=revenue,
            cost=cost,
            roi=roi,
            **kwargs
        )
    
    def log_efficiency_gain(
        self,
        operation: str,
        baseline_time: float,
        optimized_time: float,
        efficiency_gain_percent: float,
        **kwargs: Any
    ) -> None:
        """Log efficiency gains."""
        self.logger.info(
            "Efficiency gain",
            event_type="business_efficiency",
            operation=operation,
            baseline_time=baseline_time,
            optimized_time=optimized_time,
            efficiency_gain_percent=efficiency_gain_percent,
            **kwargs
        )
    
    def log_user_interaction(
        self,
        user_id: Optional[str] = None,
        action: str = "unknown",
        feature: Optional[str] = None,
        success: bool = True,
        **kwargs: Any
    ) -> None:
        """Log user interactions."""
        self.logger.info(
            "User interaction",
            event_type="user_interaction",
            user_id=user_id,
            action=action,
            feature=feature,
            success=success,
            **kwargs
        )


def add_context_processor(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add context information to log events."""
    # Add request context
    request_id = request_id_context.get()
    if request_id:
        event_dict["request_id"] = request_id
    
    user_id = user_id_context.get()
    if user_id:
        event_dict["user_id"] = user_id
    
    session_id = session_id_context.get()
    if session_id:
        event_dict["session_id"] = session_id
    
    # Add timestamp
    event_dict["timestamp"] = time.time()
    
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    environment: str = "development",
    log_file: Optional[Path] = None
) -> None:
    """
    Set up structured logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        environment: Environment name (development, staging, production)
        log_file: Optional log file path
    """
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        add_context_processor,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    
    if environment == "production":
        # JSON output for production
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ])
    else:
        # Human-readable output for development
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLogger().getEffectiveLevel()
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "json" if environment == "production" else "standard",
                "stream": sys.stdout
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console"]
        }
    }
    
    # Add file handler if log file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "json",
            "filename": str(log_file),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        logging_config["root"]["handlers"].append("file")
    
    logging.config.dictConfig(logging_config)


def get_logger(name: Optional[str] = None) -> FilteringBoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (defaults to caller's module name)
    
    Returns:
        FilteringBoundLogger: Configured logger instance
    """
    if name is None:
        # Get caller's module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "quantum_moe_mas")
    
    return structlog.get_logger(name)


def get_security_logger(name: Optional[str] = None) -> SecurityEventLogger:
    """Get a security event logger."""
    logger = get_logger(name)
    return SecurityEventLogger(logger)


def get_performance_logger(name: Optional[str] = None) -> PerformanceLogger:
    """Get a performance logger."""
    logger = get_logger(name)
    return PerformanceLogger(logger)


def get_business_logger(name: Optional[str] = None) -> BusinessLogger:
    """Get a business metrics logger."""
    logger = get_logger(name)
    return BusinessLogger(logger)


def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> None:
    """Set request context for logging."""
    if request_id:
        request_id_context.set(request_id)
    if user_id:
        user_id_context.set(user_id)
    if session_id:
        session_id_context.set(session_id)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid4())


def log_execution_time(operation_name: Optional[str] = None):
    """
    Decorator to log function execution time.
    
    Args:
        operation_name: Optional operation name (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            logger = get_performance_logger()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.log_execution_time(operation, duration_ms, success=True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.log_execution_time(operation, duration_ms, success=False, error=str(e))
                raise
        
        return wrapper
    return decorator


def log_api_call(api_name: str, endpoint: str):
    """
    Decorator to log API call performance.
    
    Args:
        api_name: Name of the API service
        endpoint: API endpoint being called
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = get_performance_logger()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Try to extract status code from result
                status_code = getattr(result, 'status_code', 200)
                method = kwargs.get('method', 'GET')
                
                logger.log_api_call(api_name, endpoint, method, status_code, duration_ms)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.log_api_call(api_name, endpoint, 'GET', 500, duration_ms, error=str(e))
                raise
        
        return wrapper
    return decorator


# Export all public functions and classes
__all__ = [
    "setup_logging",
    "get_logger",
    "get_security_logger",
    "get_performance_logger",
    "get_business_logger",
    "set_request_context",
    "generate_request_id",
    "log_execution_time",
    "log_api_call",
    "SecurityEventLogger",
    "PerformanceLogger",
    "BusinessLogger",
]