"""
JWT-based Authentication and Role-Based Authorization System.

This module provides comprehensive authentication and authorization services
with JWT tokens, role-based access control (RBAC), and security context management.
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from functools import wraps

import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt

from quantum_moe_mas.core.logging import get_logger, get_security_logger
from quantum_moe_mas.core.exceptions import AuthenticationError, AuthorizationError
from quantum_moe_mas.config.settings import get_settings

logger = get_logger(__name__)
security_logger = get_security_logger(__name__)
settings = get_settings()


class UserRole(Enum):
    """User roles for RBAC system."""
    
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    USER = "user"
    GUEST = "guest"


class Permission(Enum):
    """System permissions."""
    
    # System Administration
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    
    # User Management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Agent Operations
    AGENT_CREATE = "agent:create"
    AGENT_EXECUTE = "agent:execute"
    AGENT_CONFIGURE = "agent:configure"
    AGENT_MONITOR = "agent:monitor"
    
    # Data Access
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"
    
    # API Access
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"
    
    # Analytics & Reporting
    ANALYTICS_VIEW = "analytics:view"
    ANALYTICS_EXPORT = "analytics:export"
    ANALYTICS_ADMIN = "analytics:admin"


# Role-Permission Mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: {
        # Full system access
        Permission.SYSTEM_ADMIN,
        Permission.SYSTEM_CONFIG,
        Permission.SYSTEM_MONITOR,
        Permission.USER_CREATE,
        Permission.USER_READ,
        Permission.USER_UPDATE,
        Permission.USER_DELETE,
        Permission.AGENT_CREATE,
        Permission.AGENT_EXECUTE,
        Permission.AGENT_CONFIGURE,
        Permission.AGENT_MONITOR,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.DATA_DELETE,
        Permission.DATA_EXPORT,
        Permission.API_READ,
        Permission.API_WRITE,
        Permission.API_ADMIN,
        Permission.ANALYTICS_VIEW,
        Permission.ANALYTICS_EXPORT,
        Permission.ANALYTICS_ADMIN,
    },
    UserRole.MANAGER: {
        # Management and configuration access
        Permission.SYSTEM_MONITOR,
        Permission.USER_READ,
        Permission.USER_UPDATE,
        Permission.AGENT_CREATE,
        Permission.AGENT_EXECUTE,
        Permission.AGENT_CONFIGURE,
        Permission.AGENT_MONITOR,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.DATA_EXPORT,
        Permission.API_READ,
        Permission.API_WRITE,
        Permission.ANALYTICS_VIEW,
        Permission.ANALYTICS_EXPORT,
        Permission.ANALYTICS_ADMIN,
    },
    UserRole.ANALYST: {
        # Analysis and reporting access
        Permission.USER_READ,
        Permission.AGENT_EXECUTE,
        Permission.AGENT_MONITOR,
        Permission.DATA_READ,
        Permission.DATA_EXPORT,
        Permission.API_READ,
        Permission.ANALYTICS_VIEW,
        Permission.ANALYTICS_EXPORT,
    },
    UserRole.USER: {
        # Basic user access
        Permission.AGENT_EXECUTE,
        Permission.DATA_READ,
        Permission.API_READ,
        Permission.ANALYTICS_VIEW,
    },
    UserRole.GUEST: {
        # Limited read-only access
        Permission.DATA_READ,
        Permission.ANALYTICS_VIEW,
    },
}


@dataclass
class User:
    """User model with authentication and authorization data."""
    
    id: str
    username: str
    email: str
    hashed_password: str
    role: UserRole
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in ROLE_PERMISSIONS.get(self.role, set())
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        user_permissions = ROLE_PERMISSIONS.get(self.role, set())
        return any(perm in user_permissions for perm in permissions)
    
    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all specified permissions."""
        user_permissions = ROLE_PERMISSIONS.get(self.role, set())
        return all(perm in user_permissions for perm in permissions)
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    def can_login(self) -> bool:
        """Check if user can login."""
        return self.is_active and self.is_verified and not self.is_locked()


@dataclass
class SecurityContext:
    """Security context for request processing."""
    
    user: Optional[User] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_authenticated(self) -> bool:
        """Check if context has authenticated user."""
        return self.user is not None and self.user.can_login()
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if context has specific permission."""
        return permission in self.permissions
    
    def require_permission(self, permission: Permission) -> None:
        """Require specific permission or raise AuthorizationError."""
        if not self.has_permission(permission):
            raise AuthorizationError(f"Permission required: {permission.value}")


class JWTManager:
    """JWT token management service."""
    
    def __init__(self):
        self.secret_key = settings.security.jwt_secret_key.get_secret_value()
        self.algorithm = settings.security.jwt_algorithm
        self.access_token_expire_minutes = settings.security.jwt_access_token_expire_minutes
        self.refresh_token_expire_days = settings.security.jwt_refresh_token_expire_days
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access",
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        logger.info(
            "Access token created",
            user_id=user.id,
            username=user.username,
            expires_at=expire.isoformat()
        )
        
        return token
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "sub": user.id,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32),  # Unique token ID
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        logger.info(
            "Refresh token created",
            user_id=user.id,
            expires_at=expire.isoformat()
        )
        
        return token
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Verify token type
            if payload.get("type") != token_type:
                raise AuthenticationError(f"Invalid token type: expected {token_type}")
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, timezone.utc) < datetime.now(timezone.utc):
                raise AuthenticationError("Token has expired")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)


class AuthenticationService:
    """Authentication service with security features."""
    
    def __init__(self, jwt_manager: Optional[JWTManager] = None):
        self.jwt_manager = jwt_manager or JWTManager()
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 30
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[User]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username or email
            password: Plain text password
            ip_address: Client IP address
            user_agent: Client user agent
        
        Returns:
            User object if authentication successful, None otherwise
        """
        # Log authentication attempt
        security_logger.authentication_attempt(
            user_id=username,
            success=False,  # Will update if successful
            method="password",
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # TODO: Implement user lookup from database
        # For now, return None to indicate authentication failure
        user = await self._get_user_by_username(username)
        
        if not user:
            logger.warning(f"Authentication failed: user not found", username=username)
            return None
        
        # Check if account is locked
        if user.is_locked():
            logger.warning(f"Authentication failed: account locked", user_id=user.id)
            security_logger.authentication_attempt(
                user_id=user.id,
                success=False,
                method="password",
                ip_address=ip_address,
                user_agent=user_agent,
                reason="account_locked"
            )
            return None
        
        # Verify password
        if not self.jwt_manager.verify_password(password, user.hashed_password):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.locked_until = datetime.now(timezone.utc) + timedelta(
                    minutes=self.lockout_duration_minutes
                )
                logger.warning(f"Account locked due to failed attempts", user_id=user.id)
            
            await self._update_user(user)
            
            logger.warning(f"Authentication failed: invalid password", user_id=user.id)
            return None
        
        # Reset failed attempts on successful authentication
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now(timezone.utc)
        await self._update_user(user)
        
        # Log successful authentication
        security_logger.authentication_attempt(
            user_id=user.id,
            success=True,
            method="password",
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        logger.info(f"User authenticated successfully", user_id=user.id)
        return user
    
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER
    ) -> User:
        """Create new user account."""
        # Hash password
        hashed_password = self.jwt_manager.hash_password(password)
        
        # Generate user ID
        user_id = hashlib.sha256(f"{username}{email}".encode()).hexdigest()[:16]
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            hashed_password=hashed_password,
            role=role
        )
        
        # TODO: Save user to database
        await self._save_user(user)
        
        logger.info(f"User created", user_id=user.id, username=username, role=role.value)
        return user
    
    async def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username or email (placeholder for database lookup)."""
        # TODO: Implement database lookup
        return None
    
    async def _save_user(self, user: User) -> None:
        """Save user to database (placeholder)."""
        # TODO: Implement database save
        pass
    
    async def _update_user(self, user: User) -> None:
        """Update user in database (placeholder)."""
        # TODO: Implement database update
        pass


class AuthorizationService:
    """Authorization service with RBAC support."""
    
    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS
    
    def check_permission(
        self,
        user: User,
        permission: Permission,
        resource: Optional[str] = None
    ) -> bool:
        """Check if user has specific permission."""
        if not user.can_login():
            return False
        
        has_permission = user.has_permission(permission)
        
        if not has_permission:
            security_logger.authorization_failure(
                user_id=user.id,
                resource=resource,
                action=permission.value
            )
        
        return has_permission
    
    def require_permission(
        self,
        user: User,
        permission: Permission,
        resource: Optional[str] = None
    ) -> None:
        """Require permission or raise AuthorizationError."""
        if not self.check_permission(user, permission, resource):
            raise AuthorizationError(
                f"User {user.id} does not have permission {permission.value}"
            )
    
    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all permissions for user."""
        return self.role_permissions.get(user.role, set())
    
    def can_access_resource(
        self,
        user: User,
        resource_type: str,
        action: str,
        resource_id: Optional[str] = None
    ) -> bool:
        """Check if user can access specific resource."""
        # Map resource type and action to permission
        permission_map = {
            ("user", "read"): Permission.USER_READ,
            ("user", "create"): Permission.USER_CREATE,
            ("user", "update"): Permission.USER_UPDATE,
            ("user", "delete"): Permission.USER_DELETE,
            ("agent", "execute"): Permission.AGENT_EXECUTE,
            ("agent", "configure"): Permission.AGENT_CONFIGURE,
            ("data", "read"): Permission.DATA_READ,
            ("data", "write"): Permission.DATA_WRITE,
            ("data", "delete"): Permission.DATA_DELETE,
            ("api", "read"): Permission.API_READ,
            ("api", "write"): Permission.API_WRITE,
        }
        
        permission = permission_map.get((resource_type, action))
        if not permission:
            logger.warning(f"Unknown resource permission", resource_type=resource_type, action=action)
            return False
        
        return self.check_permission(user, permission, resource_id)


# Global instances
jwt_manager = JWTManager()
auth_service = AuthenticationService(jwt_manager)
authz_service = AuthorizationService()


# Convenience functions
def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create access token for user."""
    return jwt_manager.create_access_token(user, expires_delta)


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """Verify JWT token."""
    return jwt_manager.verify_token(token, token_type)


async def get_current_user(token: str) -> User:
    """Get current user from JWT token."""
    payload = verify_token(token)
    user_id = payload.get("sub")
    
    if not user_id:
        raise AuthenticationError("Invalid token: missing user ID")
    
    # TODO: Get user from database
    user = await auth_service._get_user_by_username(user_id)
    if not user:
        raise AuthenticationError("User not found")
    
    return user


def require_permissions(*permissions: Permission):
    """Decorator to require specific permissions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from context or arguments
            # This is a simplified implementation
            # In practice, you'd extract from request context
            user = kwargs.get('user') or getattr(args[0], 'user', None)
            
            if not user:
                raise AuthenticationError("Authentication required")
            
            for permission in permissions:
                authz_service.require_permission(user, permission)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator