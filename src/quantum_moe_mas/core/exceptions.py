"""
Core exception classes for the Quantum MoE MAS system.

This module defines custom exceptions used throughout the system
for better error handling and debugging.
"""


class QuantumMoEMASError(Exception):
    """Base exception for all Quantum MoE MAS errors."""
    pass


class ConfigurationError(QuantumMoEMASError):
    """Raised when there's a configuration error."""
    pass


class DatabaseError(QuantumMoEMASError):
    """Raised when there's a database operation error."""
    pass


class ValidationError(QuantumMoEMASError):
    """Raised when data validation fails."""
    pass


class ProcessingError(QuantumMoEMASError):
    """Raised when document processing fails."""
    pass


class EmbeddingError(QuantumMoEMASError):
    """Raised when embedding generation fails."""
    pass


class SearchError(QuantumMoEMASError):
    """Raised when search operations fail."""
    pass


class AuthenticationError(QuantumMoEMASError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(QuantumMoEMASError):
    """Raised when authorization fails."""
    pass


class RateLimitError(QuantumMoEMASError):
    """Raised when rate limits are exceeded."""
    pass


class ExternalAPIError(QuantumMoEMASError):
    """Raised when external API calls fail."""
    pass


class QuantumComputingError(QuantumMoEMASError):
    """Raised when quantum computing operations fail."""
    pass


class MoERoutingError(QuantumMoEMASError):
    """Raised when MoE routing fails."""
    pass


class AgentError(QuantumMoEMASError):
    """Raised when agent operations fail."""
    pass