"""
Security Monitoring and Threat Detection System.

This module provides real-time security monitoring, threat detection,
and incident response capabilities for the Quantum MoE MAS system.
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Deque
from dataclasses import dataclass, field
from ipaddress import ip_address, ip_network

from quantum_moe_mas.core.logging import get_logger, get_security_logger
from quantum_moe_mas.core.exceptions import QuantumMoEMASError
from quantum_moe_mas.config.settings import get_settings

logger = get_logger(__name__)
security_logger = get_security_logger(__name__)
settings = get_settings()


class ThreatLevel(Enum):
    """Threat severity levels."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EventType(Enum):
    """Security event types."""
    
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHORIZATION_FAILURE = "authz_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE_ATTEMPT = "brute_force"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MALICIOUS_INPUT = "malicious_input"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    SYSTEM_COMPROMISE = "system_compromise"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


class IncidentStatus(Enum):
    """Security incident status."""
    
    OPEN = "open"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    
    id: str
    event_type: EventType
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    user_agent: Optional[str] = None
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "threat_level": self.threat_level.value,
            "timestamp": self.timestamp.isoformat(),
            "source_ip": self.source_ip,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "user_agent": self.user_agent,
            "description": self.description,
            "details": self.details,
            "metadata": self.metadata,
        }


@dataclass
class SecurityIncident:
    """Security incident data structure."""
    
    id: str
    title: str
    description: str
    threat_level: ThreatLevel
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    events: List[SecurityEvent] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    response_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_event(self, event: SecurityEvent) -> None:
        """Add security event to incident."""
        self.events.append(event)
        self.updated_at = datetime.now(timezone.utc)
    
    def update_status(self, status: IncidentStatus, notes: Optional[str] = None) -> None:
        """Update incident status."""
        self.status = status
        self.updated_at = datetime.now(timezone.utc)
        
        if notes:
            self.response_actions.append(f"{status.value}: {notes}")


@dataclass
class ThreatIndicator:
    """Threat indicator for pattern matching."""
    
    name: str
    pattern: str
    threat_level: ThreatLevel
    event_type: EventType
    description: str
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RateLimiter:
    """Rate limiting for security monitoring."""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, Deque[float]] = defaultdict(deque)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed within rate limit."""
        now = time.time()
        request_times = self.requests[identifier]
        
        # Remove old requests outside time window
        while request_times and request_times[0] <= now - self.time_window:
            request_times.popleft()
        
        # Check if under limit
        if len(request_times) < self.max_requests:
            request_times.append(now)
            return True
        
        return False
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        request_times = self.requests[identifier]
        
        # Remove old requests
        while request_times and request_times[0] <= now - self.time_window:
            request_times.popleft()
        
        return max(0, self.max_requests - len(request_times))


class AnomalyDetector:
    """Behavioral anomaly detection."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.user_baselines: Dict[str, Dict[str, Any]] = {}
        self.system_baseline: Dict[str, Any] = {}
    
    def update_user_baseline(self, user_id: str, activity_data: Dict[str, Any]) -> None:
        """Update user behavioral baseline."""
        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = {
                "login_times": deque(maxlen=self.window_size),
                "ip_addresses": set(),
                "user_agents": set(),
                "activity_patterns": defaultdict(int),
            }
        
        baseline = self.user_baselines[user_id]
        
        # Update login times
        if "login_time" in activity_data:
            baseline["login_times"].append(activity_data["login_time"])
        
        # Update IP addresses
        if "ip_address" in activity_data:
            baseline["ip_addresses"].add(activity_data["ip_address"])
        
        # Update user agents
        if "user_agent" in activity_data:
            baseline["user_agents"].add(activity_data["user_agent"])
        
        # Update activity patterns
        if "activity_type" in activity_data:
            baseline["activity_patterns"][activity_data["activity_type"]] += 1
    
    def detect_anomalies(self, user_id: str, current_activity: Dict[str, Any]) -> List[str]:
        """Detect anomalies in user behavior."""
        anomalies = []
        
        if user_id not in self.user_baselines:
            return anomalies
        
        baseline = self.user_baselines[user_id]
        
        # Check for unusual login time
        if "login_time" in current_activity and baseline["login_times"]:
            current_hour = datetime.fromtimestamp(current_activity["login_time"]).hour
            typical_hours = [datetime.fromtimestamp(t).hour for t in baseline["login_times"]]
            
            if typical_hours and current_hour not in typical_hours:
                anomalies.append("Unusual login time detected")
        
        # Check for new IP address
        if "ip_address" in current_activity:
            if current_activity["ip_address"] not in baseline["ip_addresses"]:
                anomalies.append("Login from new IP address")
        
        # Check for new user agent
        if "user_agent" in current_activity:
            if current_activity["user_agent"] not in baseline["user_agents"]:
                anomalies.append("Login from new device/browser")
        
        return anomalies


class ThreatDetector:
    """Advanced threat detection engine."""
    
    def __init__(self):
        self.indicators: List[ThreatIndicator] = []
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.anomaly_detector = AnomalyDetector()
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        
        # Initialize default threat indicators
        self._initialize_threat_indicators()
        
        # Initialize rate limiters
        self.rate_limiters["login"] = RateLimiter(max_requests=5, time_window=300)  # 5 attempts per 5 minutes
        self.rate_limiters["api"] = RateLimiter(max_requests=100, time_window=60)   # 100 requests per minute
    
    def _initialize_threat_indicators(self) -> None:
        """Initialize default threat indicators."""
        default_indicators = [
            ThreatIndicator(
                name="SQL Injection Pattern",
                pattern=r"(union|select|insert|update|delete|drop|create|alter|exec)",
                threat_level=ThreatLevel.HIGH,
                event_type=EventType.MALICIOUS_INPUT,
                description="Potential SQL injection attempt detected"
            ),
            ThreatIndicator(
                name="XSS Pattern",
                pattern=r"(<script|javascript:|on\w+\s*=)",
                threat_level=ThreatLevel.HIGH,
                event_type=EventType.MALICIOUS_INPUT,
                description="Potential XSS attempt detected"
            ),
            ThreatIndicator(
                name="Command Injection Pattern",
                pattern=r"(;|\||&|`|\$\(|\${)",
                threat_level=ThreatLevel.HIGH,
                event_type=EventType.MALICIOUS_INPUT,
                description="Potential command injection attempt detected"
            ),
            ThreatIndicator(
                name="Path Traversal Pattern",
                pattern=r"(\.\./|\.\.\\)",
                threat_level=ThreatLevel.MEDIUM,
                event_type=EventType.MALICIOUS_INPUT,
                description="Potential path traversal attempt detected"
            ),
        ]
        
        self.indicators.extend(default_indicators)
    
    def add_threat_indicator(self, indicator: ThreatIndicator) -> None:
        """Add custom threat indicator."""
        self.indicators.append(indicator)
        logger.info(f"Added threat indicator: {indicator.name}")
    
    def detect_threats(self, event_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect threats in event data."""
        threats = []
        
        # Check against threat indicators
        for indicator in self.indicators:
            if not indicator.active:
                continue
            
            # Check if pattern matches any field in event data
            for field_name, field_value in event_data.items():
                if isinstance(field_value, str):
                    import re
                    if re.search(indicator.pattern, field_value, re.IGNORECASE):
                        event = SecurityEvent(
                            id=self._generate_event_id(),
                            event_type=indicator.event_type,
                            threat_level=indicator.threat_level,
                            timestamp=datetime.now(timezone.utc),
                            source_ip=event_data.get("source_ip"),
                            user_id=event_data.get("user_id"),
                            session_id=event_data.get("session_id"),
                            user_agent=event_data.get("user_agent"),
                            description=indicator.description,
                            details={
                                "matched_pattern": indicator.pattern,
                                "matched_field": field_name,
                                "matched_value": field_value[:100],  # Truncate for logging
                            }
                        )
                        threats.append(event)
        
        # Check rate limits
        if "source_ip" in event_data and "event_type" in event_data:
            ip = event_data["source_ip"]
            event_type = event_data["event_type"]
            
            if event_type in self.rate_limiters:
                if not self.rate_limiters[event_type].is_allowed(ip):
                    event = SecurityEvent(
                        id=self._generate_event_id(),
                        event_type=EventType.RATE_LIMIT_EXCEEDED,
                        threat_level=ThreatLevel.MEDIUM,
                        timestamp=datetime.now(timezone.utc),
                        source_ip=ip,
                        description=f"Rate limit exceeded for {event_type}",
                        details={"event_type": event_type}
                    )
                    threats.append(event)
        
        # Check for behavioral anomalies
        if "user_id" in event_data:
            anomalies = self.anomaly_detector.detect_anomalies(
                event_data["user_id"],
                event_data
            )
            
            for anomaly in anomalies:
                event = SecurityEvent(
                    id=self._generate_event_id(),
                    event_type=EventType.ANOMALOUS_BEHAVIOR,
                    threat_level=ThreatLevel.MEDIUM,
                    timestamp=datetime.now(timezone.utc),
                    source_ip=event_data.get("source_ip"),
                    user_id=event_data["user_id"],
                    description=anomaly,
                    details=event_data
                )
                threats.append(event)
        
        return threats
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = str(int(time.time() * 1000000))
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]
    
    def block_ip(self, ip_address: str, reason: str) -> None:
        """Block IP address."""
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP address: {ip_address}, Reason: {reason}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self.blocked_ips
    
    def unblock_ip(self, ip_address: str) -> None:
        """Unblock IP address."""
        self.blocked_ips.discard(ip_address)
        logger.info(f"Unblocked IP address: {ip_address}")


class SecurityMonitor:
    """Main security monitoring service."""
    
    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.events: Deque[SecurityEvent] = deque(maxlen=10000)  # Keep last 10k events
        self.incidents: Dict[str, SecurityIncident] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Initialize default event handlers
        self._initialize_event_handlers()
    
    def _initialize_event_handlers(self) -> None:
        """Initialize default event handlers."""
        self.register_event_handler(
            EventType.BRUTE_FORCE_ATTEMPT,
            self._handle_brute_force
        )
        self.register_event_handler(
            EventType.MALICIOUS_INPUT,
            self._handle_malicious_input
        )
        self.register_event_handler(
            EventType.RATE_LIMIT_EXCEEDED,
            self._handle_rate_limit_exceeded
        )
    
    def register_event_handler(
        self,
        event_type: EventType,
        handler: Callable[[SecurityEvent], None]
    ) -> None:
        """Register event handler for specific event type."""
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")
    
    async def start_monitoring(self) -> None:
        """Start security monitoring."""
        if self.monitoring_active:
            logger.warning("Security monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Security monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop security monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Security monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Perform periodic security checks
                await self._perform_security_checks()
                
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retrying
    
    async def _perform_security_checks(self) -> None:
        """Perform periodic security checks."""
        # Check for patterns in recent events
        recent_events = list(self.events)[-100:]  # Last 100 events
        
        # Detect brute force attempts
        await self._detect_brute_force_attempts(recent_events)
        
        # Check for suspicious IP activity
        await self._check_suspicious_ip_activity(recent_events)
        
        # Update threat intelligence
        await self._update_threat_intelligence()
    
    async def _detect_brute_force_attempts(self, events: List[SecurityEvent]) -> None:
        """Detect brute force attempts."""
        # Group failed login attempts by IP
        failed_logins = defaultdict(list)
        
        for event in events:
            if (event.event_type == EventType.AUTHENTICATION_FAILURE and 
                event.source_ip and 
                event.timestamp > datetime.now(timezone.utc) - timedelta(minutes=15)):
                failed_logins[event.source_ip].append(event)
        
        # Check for brute force patterns
        for ip, login_events in failed_logins.items():
            if len(login_events) >= 5:  # 5 or more failed attempts
                # Create brute force event
                brute_force_event = SecurityEvent(
                    id=self.threat_detector._generate_event_id(),
                    event_type=EventType.BRUTE_FORCE_ATTEMPT,
                    threat_level=ThreatLevel.HIGH,
                    timestamp=datetime.now(timezone.utc),
                    source_ip=ip,
                    description=f"Brute force attempt detected from {ip}",
                    details={
                        "failed_attempts": len(login_events),
                        "time_window": "15 minutes"
                    }
                )
                
                await self.process_security_event(brute_force_event)
    
    async def _check_suspicious_ip_activity(self, events: List[SecurityEvent]) -> None:
        """Check for suspicious IP activity patterns."""
        # Group events by IP
        ip_activity = defaultdict(list)
        
        for event in events:
            if event.source_ip:
                ip_activity[event.source_ip].append(event)
        
        # Check for suspicious patterns
        for ip, ip_events in ip_activity.items():
            # Check for multiple event types from same IP
            event_types = set(event.event_type for event in ip_events)
            
            if len(event_types) >= 3 and len(ip_events) >= 10:
                suspicious_event = SecurityEvent(
                    id=self.threat_detector._generate_event_id(),
                    event_type=EventType.SUSPICIOUS_ACTIVITY,
                    threat_level=ThreatLevel.MEDIUM,
                    timestamp=datetime.now(timezone.utc),
                    source_ip=ip,
                    description=f"Suspicious activity pattern from {ip}",
                    details={
                        "event_count": len(ip_events),
                        "event_types": [et.value for et in event_types]
                    }
                )
                
                await self.process_security_event(suspicious_event)
    
    async def _update_threat_intelligence(self) -> None:
        """Update threat intelligence data."""
        # This would typically fetch from external threat intelligence feeds
        # For now, we'll just log that we're updating
        logger.debug("Updating threat intelligence data")
    
    async def process_security_event(self, event: SecurityEvent) -> None:
        """Process security event."""
        # Add to event history
        self.events.append(event)
        
        # Log security event
        security_logger.suspicious_activity(
            activity_type=event.event_type.value,
            description=event.description,
            severity=event.threat_level.value,
            source_ip=event.source_ip,
            user_id=event.user_id
        )
        
        # Execute event handlers
        handlers = self.event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await asyncio.get_event_loop().run_in_executor(None, handler, event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
        
        # Check if event should trigger incident
        if event.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            await self._create_or_update_incident(event)
    
    async def _create_or_update_incident(self, event: SecurityEvent) -> None:
        """Create or update security incident."""
        # Check for existing incident with same source IP
        existing_incident = None
        for incident in self.incidents.values():
            if (incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED] and
                any(e.source_ip == event.source_ip for e in incident.events)):
                existing_incident = incident
                break
        
        if existing_incident:
            # Add event to existing incident
            existing_incident.add_event(event)
            logger.info(f"Added event to existing incident: {existing_incident.id}")
        else:
            # Create new incident
            incident_id = f"INC_{int(time.time())}"
            incident = SecurityIncident(
                id=incident_id,
                title=f"Security Incident: {event.event_type.value}",
                description=event.description,
                threat_level=event.threat_level,
                status=IncidentStatus.OPEN,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                events=[event]
            )
            
            self.incidents[incident_id] = incident
            logger.warning(f"Created new security incident: {incident_id}")
    
    async def _handle_brute_force(self, event: SecurityEvent) -> None:
        """Handle brute force attempt."""
        if event.source_ip:
            # Block IP address
            self.threat_detector.block_ip(
                event.source_ip,
                "Brute force attempt detected"
            )
            
            logger.warning(f"Blocked IP {event.source_ip} due to brute force attempt")
    
    async def _handle_malicious_input(self, event: SecurityEvent) -> None:
        """Handle malicious input detection."""
        if event.source_ip:
            # Increase suspicion level for IP
            self.threat_detector.suspicious_patterns[event.source_ip] += 1
            
            # Block IP if too many malicious inputs
            if self.threat_detector.suspicious_patterns[event.source_ip] >= 3:
                self.threat_detector.block_ip(
                    event.source_ip,
                    "Multiple malicious input attempts"
                )
    
    async def _handle_rate_limit_exceeded(self, event: SecurityEvent) -> None:
        """Handle rate limit exceeded."""
        if event.source_ip:
            # Temporarily block IP for rate limiting
            self.threat_detector.block_ip(
                event.source_ip,
                "Rate limit exceeded"
            )
            
            # Schedule unblock after cooldown period
            asyncio.create_task(self._schedule_ip_unblock(event.source_ip, 300))  # 5 minutes
    
    async def _schedule_ip_unblock(self, ip_address: str, delay_seconds: int) -> None:
        """Schedule IP unblock after delay."""
        await asyncio.sleep(delay_seconds)
        self.threat_detector.unblock_ip(ip_address)
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security monitoring metrics."""
        recent_events = [e for e in self.events if 
                        e.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)]
        
        # Count events by type
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event.event_type.value] += 1
        
        # Count events by threat level
        threat_counts = defaultdict(int)
        for event in recent_events:
            threat_counts[event.threat_level.value] += 1
        
        # Count incidents by status
        incident_counts = defaultdict(int)
        for incident in self.incidents.values():
            incident_counts[incident.status.value] += 1
        
        return {
            "monitoring_active": self.monitoring_active,
            "total_events_24h": len(recent_events),
            "event_types": dict(event_counts),
            "threat_levels": dict(threat_counts),
            "incidents": dict(incident_counts),
            "blocked_ips": len(self.threat_detector.blocked_ips),
            "threat_indicators": len(self.threat_detector.indicators),
        }


class SecurityMetrics:
    """Security metrics collection and analysis."""
    
    def __init__(self, monitor: SecurityMonitor):
        self.monitor = monitor
    
    def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get data for security dashboard."""
        metrics = self.monitor.get_security_metrics()
        
        # Add additional analysis
        recent_events = [e for e in self.monitor.events if 
                        e.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)]
        
        # Top source IPs
        ip_counts = defaultdict(int)
        for event in recent_events:
            if event.source_ip:
                ip_counts[event.source_ip] += 1
        
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Recent critical incidents
        critical_incidents = [
            incident for incident in self.monitor.incidents.values()
            if incident.threat_level == ThreatLevel.CRITICAL
        ]
        
        return {
            **metrics,
            "top_source_ips": top_ips,
            "critical_incidents": len(critical_incidents),
            "open_incidents": len([i for i in self.monitor.incidents.values() 
                                 if i.status == IncidentStatus.OPEN]),
        }
    
    def generate_security_report(self, time_period: timedelta) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        cutoff_time = datetime.now(timezone.utc) - time_period
        
        # Filter events by time period
        period_events = [e for e in self.monitor.events if e.timestamp > cutoff_time]
        
        # Analyze events
        event_analysis = self._analyze_events(period_events)
        
        # Analyze incidents
        period_incidents = [i for i in self.monitor.incidents.values() 
                           if i.created_at > cutoff_time]
        incident_analysis = self._analyze_incidents(period_incidents)
        
        return {
            "report_period": {
                "start": cutoff_time.isoformat(),
                "end": datetime.now(timezone.utc).isoformat(),
                "duration_hours": time_period.total_seconds() / 3600,
            },
            "event_analysis": event_analysis,
            "incident_analysis": incident_analysis,
            "threat_summary": self._generate_threat_summary(period_events),
            "recommendations": self._generate_security_recommendations(period_events),
        }
    
    def _analyze_events(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze security events."""
        if not events:
            return {"total": 0}
        
        # Count by type and severity
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        hourly_distribution = defaultdict(int)
        
        for event in events:
            type_counts[event.event_type.value] += 1
            severity_counts[event.threat_level.value] += 1
            hour = event.timestamp.hour
            hourly_distribution[hour] += 1
        
        return {
            "total": len(events),
            "by_type": dict(type_counts),
            "by_severity": dict(severity_counts),
            "hourly_distribution": dict(hourly_distribution),
        }
    
    def _analyze_incidents(self, incidents: List[SecurityIncident]) -> Dict[str, Any]:
        """Analyze security incidents."""
        if not incidents:
            return {"total": 0}
        
        status_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for incident in incidents:
            status_counts[incident.status.value] += 1
            severity_counts[incident.threat_level.value] += 1
        
        return {
            "total": len(incidents),
            "by_status": dict(status_counts),
            "by_severity": dict(severity_counts),
        }
    
    def _generate_threat_summary(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Generate threat summary."""
        high_severity_events = [e for e in events 
                               if e.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]]
        
        unique_ips = set(e.source_ip for e in events if e.source_ip)
        
        return {
            "high_severity_events": len(high_severity_events),
            "unique_source_ips": len(unique_ips),
            "blocked_ips": len(self.monitor.threat_detector.blocked_ips),
            "threat_indicators_triggered": len(set(e.details.get("matched_pattern") 
                                                  for e in events 
                                                  if e.details.get("matched_pattern"))),
        }
    
    def _generate_security_recommendations(self, events: List[SecurityEvent]) -> List[str]:
        """Generate security recommendations based on events."""
        recommendations = []
        
        # Check for common patterns
        malicious_input_count = len([e for e in events 
                                   if e.event_type == EventType.MALICIOUS_INPUT])
        
        if malicious_input_count > 10:
            recommendations.append(
                "High number of malicious input attempts detected. "
                "Consider implementing stricter input validation."
            )
        
        brute_force_count = len([e for e in events 
                               if e.event_type == EventType.BRUTE_FORCE_ATTEMPT])
        
        if brute_force_count > 0:
            recommendations.append(
                "Brute force attempts detected. "
                "Consider implementing account lockout policies and CAPTCHA."
            )
        
        rate_limit_count = len([e for e in events 
                              if e.event_type == EventType.RATE_LIMIT_EXCEEDED])
        
        if rate_limit_count > 20:
            recommendations.append(
                "High number of rate limit violations. "
                "Consider adjusting rate limits or implementing progressive delays."
            )
        
        return recommendations


# Global instances
security_monitor = SecurityMonitor()
threat_detector = ThreatDetector()
security_metrics = SecurityMetrics(security_monitor)