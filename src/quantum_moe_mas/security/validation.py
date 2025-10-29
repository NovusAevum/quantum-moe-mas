"""
Input Validation and Sanitization Framework.

This module provides comprehensive input validation and sanitization to prevent
injection attacks, XSS, and other security vulnerabilities. Implements OWASP
input validation best practices.
"""

import html
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Callable, Pattern
from dataclasses import dataclass, field
from enum import Enum
import bleach
from pydantic import BaseModel, validator, ValidationError as PydanticValidationError

from quantum_moe_mas.core.logging import get_logger, get_security_logger
from quantum_moe_mas.core.exceptions import ValidationError
from quantum_moe_mas.config.settings import get_settings

logger = get_logger(__name__)
security_logger = get_security_logger(__name__)
settings = get_settings()


class ValidationLevel(Enum):
    """Input validation levels."""
    
    STRICT = "strict"      # Strict validation with whitelist approach
    MODERATE = "moderate"  # Balanced validation
    PERMISSIVE = "permissive"  # Minimal validation for trusted sources


class InputType(Enum):
    """Types of input data."""
    
    TEXT = "text"
    HTML = "html"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    NUMERIC = "numeric"
    ALPHANUMERIC = "alphanumeric"
    JSON = "json"
    SQL_IDENTIFIER = "sql_identifier"
    FILE_PATH = "file_path"
    API_KEY = "api_key"
    PASSWORD = "password"


@dataclass
class ValidationRule:
    """Input validation rule."""
    
    name: str
    pattern: Optional[Pattern] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_chars: Optional[str] = None
    forbidden_chars: Optional[str] = None
    custom_validator: Optional[Callable[[str], bool]] = None
    sanitizer: Optional[Callable[[str], str]] = None
    error_message: str = "Invalid input"


@dataclass
class ValidationResult:
    """Result of input validation."""
    
    is_valid: bool
    sanitized_value: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    original_value: Optional[str] = None
    applied_rules: List[str] = field(default_factory=list)


class SecurityValidator:
    """Security-focused input validator."""
    
    def __init__(self):
        # Common attack patterns
        self.sql_injection_patterns = [
            re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)", re.IGNORECASE),
            re.compile(r"(\b(OR|AND)\s+\d+\s*=\s*\d+)", re.IGNORECASE),
            re.compile(r"(--|#|/\*|\*/)", re.IGNORECASE),
            re.compile(r"(\bxp_\w+)", re.IGNORECASE),
            re.compile(r"(\bsp_\w+)", re.IGNORECASE),
        ]
        
        self.xss_patterns = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            re.compile(r"<iframe[^>]*>", re.IGNORECASE),
            re.compile(r"<object[^>]*>", re.IGNORECASE),
            re.compile(r"<embed[^>]*>", re.IGNORECASE),
        ]
        
        self.command_injection_patterns = [
            re.compile(r"[;&|`$(){}[\]<>]"),
            re.compile(r"\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig)\b", re.IGNORECASE),
            re.compile(r"(\.\.\/|\.\.\\\\)"),
        ]
        
        self.ldap_injection_patterns = [
            re.compile(r"[()&|!*]"),
            re.compile(r"\\[0-9a-fA-F]{2}"),
        ]
    
    def check_sql_injection(self, value: str) -> List[str]:
        """Check for SQL injection patterns."""
        threats = []
        for pattern in self.sql_injection_patterns:
            if pattern.search(value):
                threats.append(f"Potential SQL injection detected: {pattern.pattern}")
        return threats
    
    def check_xss(self, value: str) -> List[str]:
        """Check for XSS patterns."""
        threats = []
        for pattern in self.xss_patterns:
            if pattern.search(value):
                threats.append(f"Potential XSS detected: {pattern.pattern}")
        return threats
    
    def check_command_injection(self, value: str) -> List[str]:
        """Check for command injection patterns."""
        threats = []
        for pattern in self.command_injection_patterns:
            if pattern.search(value):
                threats.append(f"Potential command injection detected: {pattern.pattern}")
        return threats
    
    def check_ldap_injection(self, value: str) -> List[str]:
        """Check for LDAP injection patterns."""
        threats = []
        for pattern in self.ldap_injection_patterns:
            if pattern.search(value):
                threats.append(f"Potential LDAP injection detected: {pattern.pattern}")
        return threats
    
    def scan_for_threats(self, value: str) -> List[str]:
        """Comprehensive threat scanning."""
        all_threats = []
        all_threats.extend(self.check_sql_injection(value))
        all_threats.extend(self.check_xss(value))
        all_threats.extend(self.check_command_injection(value))
        all_threats.extend(self.check_ldap_injection(value))
        
        if all_threats:
            security_logger.suspicious_activity(
                activity_type="malicious_input",
                description=f"Detected potential threats in input: {', '.join(all_threats)}",
                severity="high",
                input_value=value[:100]  # Log first 100 chars only
            )
        
        return all_threats


class InputSanitizer:
    """Input sanitization service."""
    
    def __init__(self):
        # HTML sanitization configuration
        self.allowed_html_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'
        ]
        
        self.allowed_html_attributes = {
            '*': ['class', 'id'],
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'width', 'height'],
        }
    
    def sanitize_html(self, value: str, strict: bool = True) -> str:
        """Sanitize HTML content."""
        if strict:
            # Strip all HTML tags
            return bleach.clean(value, tags=[], strip=True)
        else:
            # Allow safe HTML tags
            return bleach.clean(
                value,
                tags=self.allowed_html_tags,
                attributes=self.allowed_html_attributes,
                strip=True
            )
    
    def sanitize_sql_identifier(self, value: str) -> str:
        """Sanitize SQL identifier (table/column names)."""
        # Only allow alphanumeric characters and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', value)
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        
        return sanitized
    
    def sanitize_file_path(self, value: str) -> str:
        """Sanitize file path to prevent directory traversal."""
        # Remove directory traversal patterns
        sanitized = re.sub(r'\.\.[\\/]', '', value)
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>:"|?*]', '', sanitized)
        
        # Normalize path separators
        sanitized = sanitized.replace('\\', '/')
        
        # Remove leading/trailing slashes and spaces
        sanitized = sanitized.strip('/ ')
        
        return sanitized
    
    def sanitize_url(self, value: str) -> str:
        """Sanitize URL."""
        # Parse and reconstruct URL to remove malicious components
        try:
            parsed = urllib.parse.urlparse(value)
            
            # Only allow safe schemes
            if parsed.scheme not in ['http', 'https', 'ftp', 'ftps']:
                return ''
            
            # Reconstruct clean URL
            clean_url = urllib.parse.urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                ''  # Remove fragment for security
            ))
            
            return clean_url
        except Exception:
            return ''
    
    def sanitize_email(self, value: str) -> str:
        """Sanitize email address."""
        # Basic email sanitization
        sanitized = value.strip().lower()
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>()[\]\\,;:\s@"]+', '', sanitized)
        
        return sanitized
    
    def sanitize_phone(self, value: str) -> str:
        """Sanitize phone number."""
        # Keep only digits, spaces, hyphens, parentheses, and plus sign
        sanitized = re.sub(r'[^0-9\s\-\(\)\+]', '', value)
        return sanitized.strip()
    
    def sanitize_alphanumeric(self, value: str) -> str:
        """Sanitize to alphanumeric only."""
        return re.sub(r'[^a-zA-Z0-9]', '', value)
    
    def sanitize_numeric(self, value: str) -> str:
        """Sanitize to numeric only."""
        return re.sub(r'[^0-9.\-]', '', value)


class InputValidator:
    """Comprehensive input validation service."""
    
    def __init__(self):
        self.security_validator = SecurityValidator()
        self.sanitizer = InputSanitizer()
        
        # Predefined validation rules
        self.validation_rules = {
            InputType.EMAIL: ValidationRule(
                name="email",
                pattern=re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
                max_length=254,
                sanitizer=self.sanitizer.sanitize_email,
                error_message="Invalid email format"
            ),
            InputType.URL: ValidationRule(
                name="url",
                pattern=re.compile(r'^https?://[^\s/$.?#].[^\s]*$', re.IGNORECASE),
                max_length=2048,
                sanitizer=self.sanitizer.sanitize_url,
                error_message="Invalid URL format"
            ),
            InputType.PHONE: ValidationRule(
                name="phone",
                pattern=re.compile(r'^[\+]?[1-9][\d]{0,15}$'),
                min_length=10,
                max_length=17,
                sanitizer=self.sanitizer.sanitize_phone,
                error_message="Invalid phone number format"
            ),
            InputType.ALPHANUMERIC: ValidationRule(
                name="alphanumeric",
                pattern=re.compile(r'^[a-zA-Z0-9]+$'),
                sanitizer=self.sanitizer.sanitize_alphanumeric,
                error_message="Only alphanumeric characters allowed"
            ),
            InputType.NUMERIC: ValidationRule(
                name="numeric",
                pattern=re.compile(r'^-?\d+(\.\d+)?$'),
                sanitizer=self.sanitizer.sanitize_numeric,
                error_message="Only numeric values allowed"
            ),
            InputType.SQL_IDENTIFIER: ValidationRule(
                name="sql_identifier",
                pattern=re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$'),
                max_length=64,
                sanitizer=self.sanitizer.sanitize_sql_identifier,
                error_message="Invalid SQL identifier"
            ),
            InputType.FILE_PATH: ValidationRule(
                name="file_path",
                pattern=re.compile(r'^[^<>:"|?*]+$'),
                max_length=260,
                sanitizer=self.sanitizer.sanitize_file_path,
                error_message="Invalid file path"
            ),
            InputType.API_KEY: ValidationRule(
                name="api_key",
                pattern=re.compile(r'^[a-zA-Z0-9\-_\.]+$'),
                min_length=16,
                max_length=256,
                error_message="Invalid API key format"
            ),
        }
    
    def validate(
        self,
        value: Any,
        input_type: InputType,
        validation_level: ValidationLevel = ValidationLevel.STRICT,
        custom_rules: Optional[List[ValidationRule]] = None
    ) -> ValidationResult:
        """Validate input value."""
        if value is None:
            return ValidationResult(
                is_valid=False,
                errors=["Value cannot be None"],
                original_value=None
            )
        
        # Convert to string
        str_value = str(value)
        original_value = str_value
        
        result = ValidationResult(
            is_valid=True,
            sanitized_value=str_value,
            original_value=original_value
        )
        
        # Security threat scanning (always performed)
        threats = self.security_validator.scan_for_threats(str_value)
        if threats:
            result.is_valid = False
            result.errors.extend(threats)
            return result
        
        # Apply validation rules
        rules_to_apply = []
        
        # Add predefined rule for input type
        if input_type in self.validation_rules:
            rules_to_apply.append(self.validation_rules[input_type])
        
        # Add custom rules
        if custom_rules:
            rules_to_apply.extend(custom_rules)
        
        # Apply each rule
        for rule in rules_to_apply:
            rule_result = self._apply_rule(str_value, rule, validation_level)
            
            if not rule_result.is_valid:
                result.is_valid = False
                result.errors.extend(rule_result.errors)
            
            result.warnings.extend(rule_result.warnings)
            result.applied_rules.append(rule.name)
            
            # Update sanitized value
            if rule_result.sanitized_value is not None:
                result.sanitized_value = rule_result.sanitized_value
                str_value = rule_result.sanitized_value
        
        # Log validation result
        if not result.is_valid:
            logger.warning(
                "Input validation failed",
                input_type=input_type.value,
                validation_level=validation_level.value,
                errors=result.errors,
                original_length=len(original_value)
            )
        
        return result
    
    def _apply_rule(
        self,
        value: str,
        rule: ValidationRule,
        validation_level: ValidationLevel
    ) -> ValidationResult:
        """Apply single validation rule."""
        result = ValidationResult(is_valid=True, sanitized_value=value)
        
        # Apply sanitizer first
        if rule.sanitizer:
            result.sanitized_value = rule.sanitizer(value)
            value = result.sanitized_value
        
        # Length validation
        if rule.min_length is not None and len(value) < rule.min_length:
            result.is_valid = False
            result.errors.append(f"Minimum length is {rule.min_length}")
        
        if rule.max_length is not None and len(value) > rule.max_length:
            if validation_level == ValidationLevel.STRICT:
                result.is_valid = False
                result.errors.append(f"Maximum length is {rule.max_length}")
            else:
                result.warnings.append(f"Value exceeds recommended length of {rule.max_length}")
                # Truncate if not strict
                result.sanitized_value = value[:rule.max_length]
        
        # Pattern validation
        if rule.pattern and not rule.pattern.match(value):
            result.is_valid = False
            result.errors.append(rule.error_message)
        
        # Character validation
        if rule.allowed_chars:
            invalid_chars = set(value) - set(rule.allowed_chars)
            if invalid_chars:
                result.is_valid = False
                result.errors.append(f"Invalid characters: {', '.join(invalid_chars)}")
        
        if rule.forbidden_chars:
            forbidden_found = set(value) & set(rule.forbidden_chars)
            if forbidden_found:
                result.is_valid = False
                result.errors.append(f"Forbidden characters: {', '.join(forbidden_found)}")
        
        # Custom validator
        if rule.custom_validator and not rule.custom_validator(value):
            result.is_valid = False
            result.errors.append(rule.error_message)
        
        return result
    
    def validate_dict(
        self,
        data: Dict[str, Any],
        field_types: Dict[str, InputType],
        validation_level: ValidationLevel = ValidationLevel.STRICT
    ) -> Dict[str, ValidationResult]:
        """Validate dictionary of values."""
        results = {}
        
        for field_name, value in data.items():
            input_type = field_types.get(field_name, InputType.TEXT)
            results[field_name] = self.validate(value, input_type, validation_level)
        
        return results
    
    def is_safe_input(self, value: str) -> bool:
        """Quick check if input is safe (no security threats)."""
        threats = self.security_validator.scan_for_threats(value)
        return len(threats) == 0


# Pydantic models for structured validation
class SecureBaseModel(BaseModel):
    """Base model with security validation."""
    
    class Config:
        validate_assignment = True
        str_strip_whitespace = True
        max_anystr_length = 10000  # Prevent DoS attacks
    
    @validator('*', pre=True)
    def validate_security(cls, v):
        """Apply security validation to all fields."""
        if isinstance(v, str):
            validator_instance = InputValidator()
            if not validator_instance.is_safe_input(v):
                raise ValueError("Input contains potential security threats")
        return v


class UserInput(SecureBaseModel):
    """Model for user input validation."""
    
    username: str = Field(..., min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_]+$')
    email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    password: str = Field(..., min_length=8, max_length=128)
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password strength."""
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v


class QueryInput(SecureBaseModel):
    """Model for query input validation."""
    
    query: str = Field(..., min_length=1, max_length=5000)
    domain: Optional[str] = Field(None, regex=r'^[a-zA-Z0-9_]+$')
    modalities: Optional[List[str]] = Field(default_factory=list)
    
    @validator('modalities')
    def validate_modalities(cls, v):
        """Validate modality values."""
        allowed_modalities = ['text', 'image', 'pdf', 'video', 'audio']
        for modality in v:
            if modality not in allowed_modalities:
                raise ValueError(f'Invalid modality: {modality}')
        return v


# Global instances
input_validator = InputValidator()
security_validator = SecurityValidator()
input_sanitizer = InputSanitizer()


# Convenience functions
def validate_input(
    value: Any,
    input_type: InputType,
    validation_level: ValidationLevel = ValidationLevel.STRICT
) -> ValidationResult:
    """Validate input value."""
    return input_validator.validate(value, input_type, validation_level)


def sanitize_input(value: str, input_type: InputType) -> str:
    """Sanitize input value."""
    result = input_validator.validate(value, input_type)
    return result.sanitized_value or value


def is_safe_input(value: str) -> bool:
    """Check if input is safe from security threats."""
    return input_validator.is_safe_input(value)