"""
Role-Based Access Control (RBAC) and Session Management System.

This module provides comprehensive access control with granular permissions,
session management with configurable timeouts, and real-time security monitoring.
"""

import asyncio
import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict

from quantum_moe_mas.core.logging import get_logger, get_security_logger
from quantum_moe_mas.core.exceptions import AuthenticationError, AuthorizationError
from quantum_moe_mas.security.auth import User, UserRole, Permission, SecurityContext
from quantum_moe_mas.security.audit import audit_logger, AuditEventType
from quantum_moe_mas.config.settings import get_settings

logger = get_logger(__name__)
security_logger = get_security_logger(__name__)
settings = get_settings()


class SessionStatus(Enum):
    """Session status."""
    
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    LOCKED = "locked"


class AccessDecision(Enum):
    """Access control decision."""
    
    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"


@dataclass
class Session:
    """User session data."""
    
    id: str
    user_id: str
    user: User
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    status: SessionStatus = SessionStatus.ACTIVE
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if session is valid."""
        if self.status != SessionStatus.ACTIVE:
            return False
        
        if datetime.now(timezone.utc) > self.expires_at:
            return False
        
        return True
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)
    
    def extend_session(self, extension_minutes: int = 30) -> None:
        """Extend session expiration."""
        self.expires_at = datetime.now(timezone.utc) + timedelta(minutes=extension_minutes)
        self.update_activity()


@dataclass
class AccessRequest:
    """Access control request."""
    
    user: User
    resource: str
    action: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AccessPolicy:
    """Access control policy."""
    
    id: str
    name: str
    description: str
    resource_pattern: str
    actions: Set[str]
    roles: Set[UserRole]
    permissions: Set[Permission]
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionManager:
    """Session management service."""
    
    def __init__(self, default_timeout_minutes: int = 30):
        self.default_timeout_minutes = default_timeout_minutes
        self.sessions: Dict[str, Session] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> session_ids
        self.cleanup_task: Optional[asyncio.Task] = None
        self.max_sessions_per_user = 5
        
        # Start cleanup task
        asyncio.create_task(self._start_cleanup_task())
    
    async def _start_cleanup_task(self) -> None:
        """Start session cleanup task."""
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
    
    async def create_session(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout_minutes: Optional[int] = None
    ) -> Session:
        """Create new user session."""
        # Check session limits
        user_session_count = len(self.user_sessions.get(user.id, set()))
        if user_session_count >= self.max_sessions_per_user:
            # Terminate oldest session
            await self._terminate_oldest_session(user.id)
        
        # Generate session ID
        session_id = self._generate_session_id()
        
        # Calculate expiration
        timeout = timeout_minutes or self.default_timeout_minutes
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=timeout)
        
        # Create session
        session = Session(
            id=session_id,
            user_id=user.id,
            user=user,
            created_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Store session
        self.sessions[session_id] = session
        self.user_sessions[user.id].add(session_id)
        
        # Log session creation
        await audit_logger.log_event(
            AuditEventType.USER_LOGIN,
            user_id=user.id,
            session_id=session_id,
            source_ip=ip_address,
            user_agent=user_agent,
            details={
                "session_timeout_minutes": timeout,
                "user_role": user.role.value
            }
        )
        
        logger.info(f"Created session for user {user.id}", session_id=session_id)
        return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        
        if not session:
            return None
        
        if not session.is_valid():
            await self.terminate_session(session_id, reason="expired")
            return None
        
        # Update activity
        session.update_activity()
        return session
    
    async def terminate_session(self, session_id: str, reason: str = "manual") -> bool:
        """Terminate session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # Update session status
        session.status = SessionStatus.TERMINATED
        
        # Remove from active sessions
        self.sessions.pop(session_id, None)
        self.user_sessions[session.user_id].discard(session_id)
        
        # Log session termination
        await audit_logger.log_event(
            AuditEventType.USER_LOGOUT,
            user_id=session.user_id,
            session_id=session_id,
            source_ip=session.ip_address,
            details={"termination_reason": reason}
        )
        
        logger.info(f"Terminated session {session_id}, reason: {reason}")
        return True
    
    async def terminate_user_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """Terminate all sessions for a user."""
        user_session_ids = self.user_sessions.get(user_id, set()).copy()
        
        if except_session:
            user_session_ids.discard(except_session)
        
        terminated_count = 0
        for session_id in user_session_ids:
            if await self.terminate_session(session_id, reason="user_logout"):
                terminated_count += 1
        
        return terminated_count
    
    async def extend_session(self, session_id: str, extension_minutes: int = 30) -> bool:
        """Extend session expiration."""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.extend_session(extension_minutes)
        
        logger.info(f"Extended session {session_id} by {extension_minutes} minutes")
        return True
    
    async def _terminate_oldest_session(self, user_id: str) -> None:
        """Terminate oldest session for user."""
        user_session_ids = self.user_sessions.get(user_id, set())
        
        if not user_session_ids:
            return
        
        # Find oldest session
        oldest_session_id = None
        oldest_time = datetime.now(timezone.utc)
        
        for session_id in user_session_ids:
            session = self.sessions.get(session_id)
            if session and session.created_at < oldest_time:
                oldest_time = session.created_at
                oldest_session_id = session_id
        
        if oldest_session_id:
            await self.terminate_session(oldest_session_id, reason="session_limit")
    
    async def _cleanup_expired_sessions(self) -> None:
        """Cleanup expired sessions periodically."""
        while True:
            try:
                expired_sessions = []
                current_time = datetime.now(timezone.utc)
                
                for session_id, session in self.sessions.items():
                    if current_time > session.expires_at:
                        expired_sessions.append(session_id)
                
                # Terminate expired sessions
                for session_id in expired_sessions:
                    await self.terminate_session(session_id, reason="expired")
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                # Sleep for cleanup interval
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)  # Brief pause before retrying
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID."""
        random_data = secrets.token_bytes(32)
        timestamp = str(int(time.time() * 1000000))
        combined = random_data + timestamp.encode()
        return hashlib.sha256(combined).hexdigest()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        active_sessions = len(self.sessions)
        users_with_sessions = len([user_id for user_id, sessions in self.user_sessions.items() if sessions])
        
        # Session age distribution
        current_time = datetime.now(timezone.utc)
        age_distribution = {"<5min": 0, "5-30min": 0, "30min-2h": 0, ">2h": 0}
        
        for session in self.sessions.values():
            age_minutes = (current_time - session.created_at).total_seconds() / 60
            
            if age_minutes < 5:
                age_distribution["<5min"] += 1
            elif age_minutes < 30:
                age_distribution["5-30min"] += 1
            elif age_minutes < 120:
                age_distribution["30min-2h"] += 1
            else:
                age_distribution[">2h"] += 1
        
        return {
            "active_sessions": active_sessions,
            "users_with_sessions": users_with_sessions,
            "max_sessions_per_user": self.max_sessions_per_user,
            "default_timeout_minutes": self.default_timeout_minutes,
            "age_distribution": age_distribution,
        }


class AccessControlEngine:
    """Role-based access control engine."""
    
    def __init__(self):
        self.policies: Dict[str, AccessPolicy] = {}
        self.access_cache: Dict[str, Tuple[AccessDecision, datetime]] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self) -> None:
        """Initialize default access control policies."""
        default_policies = [
            AccessPolicy(
                id="admin_full_access",
                name="Administrator Full Access",
                description="Full system access for administrators",
                resource_pattern="*",
                actions={"*"},
                roles={UserRole.ADMIN},
                permissions=set(Permission),
                priority=100
            ),
            AccessPolicy(
                id="manager_system_access",
                name="Manager System Access",
                description="System management access for managers",
                resource_pattern="system/*",
                actions={"read", "write", "configure"},
                roles={UserRole.MANAGER},
                permissions={
                    Permission.SYSTEM_MONITOR,
                    Permission.AGENT_CONFIGURE,
                    Permission.DATA_READ,
                    Permission.DATA_WRITE,
                    Permission.ANALYTICS_VIEW
                },
                priority=80
            ),
            AccessPolicy(
                id="analyst_data_access",
                name="Analyst Data Access",
                description="Data analysis access for analysts",
                resource_pattern="data/*",
                actions={"read", "export"},
                roles={UserRole.ANALYST},
                permissions={
                    Permission.DATA_READ,
                    Permission.DATA_EXPORT,
                    Permission.ANALYTICS_VIEW,
                    Permission.ANALYTICS_EXPORT
                },
                priority=60
            ),
            AccessPolicy(
                id="user_basic_access",
                name="User Basic Access",
                description="Basic access for regular users",
                resource_pattern="user/*",
                actions={"read", "execute"},
                roles={UserRole.USER},
                permissions={
                    Permission.AGENT_EXECUTE,
                    Permission.DATA_READ,
                    Permission.ANALYTICS_VIEW
                },
                priority=40
            ),
            AccessPolicy(
                id="guest_read_only",
                name="Guest Read-Only Access",
                description="Read-only access for guests",
                resource_pattern="public/*",
                actions={"read"},
                roles={UserRole.GUEST},
                permissions={
                    Permission.DATA_READ,
                    Permission.ANALYTICS_VIEW
                },
                priority=20
            ),
        ]
        
        for policy in default_policies:
            self.policies[policy.id] = policy
    
    def add_policy(self, policy: AccessPolicy) -> None:
        """Add access control policy."""
        self.policies[policy.id] = policy
        
        # Clear cache since policies changed
        self.access_cache.clear()
        
        logger.info(f"Added access policy: {policy.name}")
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove access control policy."""
        if policy_id in self.policies:
            del self.policies[policy_id]
            self.access_cache.clear()
            logger.info(f"Removed access policy: {policy_id}")
            return True
        return False
    
    async def check_access(self, request: AccessRequest) -> AccessDecision:
        """Check access for request."""
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Check cache
        cached_result = self._get_cached_decision(cache_key)
        if cached_result:
            return cached_result
        
        # Evaluate policies
        decision = await self._evaluate_policies(request)
        
        # Cache decision
        self._cache_decision(cache_key, decision)
        
        # Log access decision
        await audit_logger.log_event(
            AuditEventType.PERMISSION_GRANTED if decision == AccessDecision.ALLOW else AuditEventType.PERMISSION_DENIED,
            user_id=request.user.id,
            resource=request.resource,
            action=request.action,
            result="success" if decision == AccessDecision.ALLOW else "failure",
            details={
                "decision": decision.value,
                "user_role": request.user.role.value,
                "context": request.context
            }
        )
        
        return decision
    
    async def _evaluate_policies(self, request: AccessRequest) -> AccessDecision:
        """Evaluate access policies for request."""
        applicable_policies = []
        
        # Find applicable policies
        for policy in self.policies.values():
            if not policy.active:
                continue
            
            if self._policy_matches_request(policy, request):
                applicable_policies.append(policy)
        
        if not applicable_policies:
            return AccessDecision.DENY
        
        # Sort by priority (higher priority first)
        applicable_policies.sort(key=lambda p: p.priority, reverse=True)
        
        # Evaluate policies in priority order
        for policy in applicable_policies:
            decision = await self._evaluate_policy(policy, request)
            
            if decision != AccessDecision.ABSTAIN:
                return decision
        
        # Default deny
        return AccessDecision.DENY
    
    def _policy_matches_request(self, policy: AccessPolicy, request: AccessRequest) -> bool:
        """Check if policy matches request."""
        # Check resource pattern
        if not self._matches_pattern(policy.resource_pattern, request.resource):
            return False
        
        # Check actions
        if "*" not in policy.actions and request.action not in policy.actions:
            return False
        
        # Check user role
        if policy.roles and request.user.role not in policy.roles:
            return False
        
        # Check user permissions
        if policy.permissions:
            user_permissions = set()
            # Get user permissions based on role (simplified)
            from quantum_moe_mas.security.auth import ROLE_PERMISSIONS
            user_permissions = ROLE_PERMISSIONS.get(request.user.role, set())
            
            if not policy.permissions.intersection(user_permissions):
                return False
        
        return True
    
    async def _evaluate_policy(self, policy: AccessPolicy, request: AccessRequest) -> AccessDecision:
        """Evaluate single policy."""
        # Check conditions
        if policy.conditions:
            if not await self._evaluate_conditions(policy.conditions, request):
                return AccessDecision.ABSTAIN
        
        # Policy matches and conditions are met
        return AccessDecision.ALLOW
    
    async def _evaluate_conditions(self, conditions: Dict[str, Any], request: AccessRequest) -> bool:
        """Evaluate policy conditions."""
        for condition_type, condition_value in conditions.items():
            if condition_type == "time_range":
                if not self._check_time_range(condition_value):
                    return False
            elif condition_type == "ip_range":
                if not self._check_ip_range(condition_value, request.context.get("ip_address")):
                    return False
            elif condition_type == "max_requests_per_hour":
                if not await self._check_rate_limit(request.user.id, condition_value):
                    return False
        
        return True
    
    def _matches_pattern(self, pattern: str, resource: str) -> bool:
        """Check if resource matches pattern."""
        if pattern == "*":
            return True
        
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return resource.startswith(prefix)
        
        return pattern == resource
    
    def _check_time_range(self, time_range: Dict[str, str]) -> bool:
        """Check if current time is within allowed range."""
        # Simplified time range check
        current_hour = datetime.now().hour
        start_hour = int(time_range.get("start", "0"))
        end_hour = int(time_range.get("end", "23"))
        
        return start_hour <= current_hour <= end_hour
    
    def _check_ip_range(self, ip_range: str, client_ip: Optional[str]) -> bool:
        """Check if client IP is within allowed range."""
        if not client_ip:
            return False
        
        # Simplified IP range check
        # In production, use proper IP network checking
        return client_ip.startswith(ip_range.split("/")[0].rsplit(".", 1)[0])
    
    async def _check_rate_limit(self, user_id: str, max_requests: int) -> bool:
        """Check rate limit for user."""
        # Simplified rate limiting
        # In production, use proper rate limiting with Redis or similar
        return True
    
    def _generate_cache_key(self, request: AccessRequest) -> str:
        """Generate cache key for request."""
        key_data = f"{request.user.id}:{request.resource}:{request.action}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _get_cached_decision(self, cache_key: str) -> Optional[AccessDecision]:
        """Get cached access decision."""
        if cache_key in self.access_cache:
            decision, cached_at = self.access_cache[cache_key]
            
            # Check if cache is still valid
            if datetime.now(timezone.utc) - cached_at < timedelta(seconds=self.cache_ttl_seconds):
                return decision
            else:
                # Remove expired cache entry
                del self.access_cache[cache_key]
        
        return None
    
    def _cache_decision(self, cache_key: str, decision: AccessDecision) -> None:
        """Cache access decision."""
        self.access_cache[cache_key] = (decision, datetime.now(timezone.utc))
    
    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get effective permissions for user."""
        permissions = set()
        
        # Get permissions from role
        from quantum_moe_mas.security.auth import ROLE_PERMISSIONS
        permissions.update(ROLE_PERMISSIONS.get(user.role, set()))
        
        # Get permissions from applicable policies
        for policy in self.policies.values():
            if policy.active and user.role in policy.roles:
                permissions.update(policy.permissions)
        
        return permissions
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get access control statistics."""
        return {
            "total_policies": len(self.policies),
            "active_policies": len([p for p in self.policies.values() if p.active]),
            "cache_entries": len(self.access_cache),
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }


class SecurityIncidentManager:
    """Security incident response and automation."""
    
    def __init__(self, session_manager: SessionManager, access_control: AccessControlEngine):
        self.session_manager = session_manager
        self.access_control = access_control
        self.incident_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.automated_responses = True
        
        # Initialize default incident handlers
        self._initialize_incident_handlers()
    
    def _initialize_incident_handlers(self) -> None:
        """Initialize default incident handlers."""
        self.register_incident_handler("brute_force_attack", self._handle_brute_force)
        self.register_incident_handler("privilege_escalation", self._handle_privilege_escalation)
        self.register_incident_handler("suspicious_activity", self._handle_suspicious_activity)
        self.register_incident_handler("unauthorized_access", self._handle_unauthorized_access)
    
    def register_incident_handler(self, incident_type: str, handler: Callable) -> None:
        """Register incident response handler."""
        self.incident_handlers[incident_type].append(handler)
        logger.info(f"Registered incident handler for {incident_type}")
    
    async def handle_security_incident(
        self,
        incident_type: str,
        details: Dict[str, Any],
        severity: str = "medium"
    ) -> None:
        """Handle security incident."""
        logger.warning(f"Security incident detected: {incident_type}", details=details, severity=severity)
        
        # Log incident
        await audit_logger.log_event(
            AuditEventType.INCIDENT_CREATED,
            details={
                "incident_type": incident_type,
                "severity": severity,
                **details
            }
        )
        
        # Execute incident handlers
        handlers = self.incident_handlers.get(incident_type, [])
        for handler in handlers:
            try:
                await handler(details)
            except Exception as e:
                logger.error(f"Error in incident handler: {e}")
    
    async def _handle_brute_force(self, details: Dict[str, Any]) -> None:
        """Handle brute force attack."""
        user_id = details.get("user_id")
        ip_address = details.get("ip_address")
        
        if user_id:
            # Terminate all user sessions
            terminated_count = await self.session_manager.terminate_user_sessions(user_id)
            logger.info(f"Terminated {terminated_count} sessions for user {user_id} due to brute force")
        
        if ip_address:
            # Block IP address (would integrate with firewall/WAF)
            logger.info(f"Would block IP address {ip_address} due to brute force attack")
    
    async def _handle_privilege_escalation(self, details: Dict[str, Any]) -> None:
        """Handle privilege escalation attempt."""
        user_id = details.get("user_id")
        
        if user_id:
            # Terminate user sessions and require re-authentication
            await self.session_manager.terminate_user_sessions(user_id)
            logger.warning(f"Terminated sessions for user {user_id} due to privilege escalation attempt")
    
    async def _handle_suspicious_activity(self, details: Dict[str, Any]) -> None:
        """Handle suspicious activity."""
        user_id = details.get("user_id")
        
        if user_id:
            # Increase monitoring for user
            logger.info(f"Increased monitoring for user {user_id} due to suspicious activity")
    
    async def _handle_unauthorized_access(self, details: Dict[str, Any]) -> None:
        """Handle unauthorized access attempt."""
        user_id = details.get("user_id")
        resource = details.get("resource")
        
        if user_id:
            # Log unauthorized access
            logger.warning(f"Unauthorized access attempt by user {user_id} to resource {resource}")


class SecurityDashboard:
    """Security monitoring dashboard data provider."""
    
    def __init__(
        self,
        session_manager: SessionManager,
        access_control: AccessControlEngine,
        incident_manager: SecurityIncidentManager
    ):
        self.session_manager = session_manager
        self.access_control = access_control
        self.incident_manager = incident_manager
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        # Session statistics
        session_stats = self.session_manager.get_session_stats()
        
        # Access control statistics
        policy_stats = self.access_control.get_policy_stats()
        
        # Recent audit events
        recent_events = await audit_logger.search_events(
            start_time=datetime.now(timezone.utc) - timedelta(hours=24),
            limit=100
        )
        
        # Event type distribution
        event_type_counts = defaultdict(int)
        for event in recent_events:
            event_type_counts[event.event_type.value] += 1
        
        # Failed access attempts
        failed_access_events = [
            e for e in recent_events 
            if e.event_type == AuditEventType.PERMISSION_DENIED
        ]
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sessions": session_stats,
            "access_control": policy_stats,
            "audit_events": {
                "total_24h": len(recent_events),
                "by_type": dict(event_type_counts),
                "failed_access_attempts": len(failed_access_events),
            },
            "security_status": {
                "active_sessions": session_stats["active_sessions"],
                "failed_access_24h": len(failed_access_events),
                "policies_active": policy_stats["active_policies"],
                "automated_responses": self.incident_manager.automated_responses,
            }
        }
    
    async def get_user_activity_report(self, user_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get user activity report."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        user_events = await audit_logger.search_events(
            start_time=start_time,
            user_id=user_id
        )
        
        # Analyze user activity
        activity_by_hour = defaultdict(int)
        resources_accessed = set()
        actions_performed = defaultdict(int)
        
        for event in user_events:
            hour = event.timestamp.hour
            activity_by_hour[hour] += 1
            
            if event.resource:
                resources_accessed.add(event.resource)
            
            if event.action:
                actions_performed[event.action] += 1
        
        return {
            "user_id": user_id,
            "period_hours": hours,
            "total_events": len(user_events),
            "activity_by_hour": dict(activity_by_hour),
            "resources_accessed": list(resources_accessed),
            "actions_performed": dict(actions_performed),
            "events": [event.to_dict() for event in user_events[-50:]]  # Last 50 events
        }


# Global instances
session_manager = SessionManager()
access_control_engine = AccessControlEngine()
security_incident_manager = SecurityIncidentManager(session_manager, access_control_engine)
security_dashboard = SecurityDashboard(session_manager, access_control_engine, security_incident_manager)