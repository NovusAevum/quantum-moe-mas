"""
Security Framework Module.

This module provides comprehensive security features including:
- OWASP Top 10 compliance
- JWT authentication and authorization
- Input validation and sanitization
- Data encryption and protection
- Security monitoring and audit logging
"""

from .auth import (
    JWTManager,
    AuthenticationService,
    AuthorizationService,
    SecurityContext,
    create_access_token,
    verify_token,
    get_current_user,
    require_permissions,
)
from .encryption import (
    EncryptionService,
    DataProtectionService,
    SecureStorage,
)
from .validation import (
    InputValidator,
    SecurityValidator,
    sanitize_input,
    validate_input,
)
from .owasp import (
    OWASPSecurityFramework,
    SecurityScanner,
    VulnerabilityAssessment,
)
from .monitoring import (
    SecurityMonitor,
    ThreatDetector,
    SecurityMetrics,
)

__all__ = [
    # Authentication & Authorization
    "JWTManager",
    "AuthenticationService", 
    "AuthorizationService",
    "SecurityContext",
    "create_access_token",
    "verify_token",
    "get_current_user",
    "require_permissions",
    
    # Encryption & Data Protection
    "EncryptionService",
    "DataProtectionService",
    "SecureStorage",
    
    # Input Validation
    "InputValidator",
    "SecurityValidator",
    "sanitize_input",
    "validate_input",
    
    # OWASP Compliance
    "OWASPSecurityFramework",
    "SecurityScanner",
    "VulnerabilityAssessment",
    
    # Security Monitoring
    "SecurityMonitor",
    "ThreatDetector",
    "SecurityMetrics",
]