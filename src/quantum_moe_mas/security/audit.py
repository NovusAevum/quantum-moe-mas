"""
Comprehensive Audit Logging and Compliance System.

This module provides enterprise-grade audit logging, PII detection and anonymization,
compliance reporting for SOC 2, GDPR, HIPAA standards, and automated security scanning.
"""

import asyncio
import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from quantum_moe_mas.core.logging import get_logger, get_security_logger
from quantum_moe_mas.core.exceptions import QuantumMoEMASError
from quantum_moe_mas.config.settings import get_settings

logger = get_logger(__name__)
security_logger = get_security_logger(__name__)
settings = get_settings()


class AuditEventType(Enum):
    """Types of audit events."""
    
    # Authentication & Authorization
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGED = "password_changed"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    
    # Data Access
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    
    # System Operations
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    SERVICE_START = "service_start"
    SERVICE_STOP = "service_stop"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    
    # Security Events
    SECURITY_VIOLATION = "security_violation"
    MALICIOUS_ACTIVITY = "malicious_activity"
    VULNERABILITY_DETECTED = "vulnerability_detected"
    INCIDENT_CREATED = "incident_created"
    INCIDENT_RESOLVED = "incident_resolved"
    
    # Compliance Events
    PII_ACCESS = "pii_access"
    PII_ANONYMIZED = "pii_anonymized"
    DATA_RETENTION_APPLIED = "data_retention_applied"
    CONSENT_GRANTED = "consent_granted"
    CONSENT_REVOKED = "consent_revoked"


class ComplianceStandard(Enum):
    """Compliance standards."""
    
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"


class PIIType(Enum):
    """Types of Personally Identifiable Information."""
    
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"


@dataclass
class AuditEvent:
    """Audit event record."""
    
    id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"  # success, failure, error
    details: Dict[str, Any] = field(default_factory=dict)
    compliance_tags: Set[ComplianceStandard] = field(default_factory=set)
    pii_detected: Set[PIIType] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "compliance_tags": [tag.value for tag in self.compliance_tags],
            "pii_detected": [pii.value for pii in self.pii_detected],
            "metadata": self.metadata,
        }


@dataclass
class PIIDetectionResult:
    """Result of PII detection."""
    
    found_pii: Dict[PIIType, List[str]]
    anonymized_text: str
    confidence_scores: Dict[PIIType, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PIIDetector:
    """PII detection and anonymization service."""
    
    def __init__(self):
        # PII detection patterns
        self.pii_patterns = {
            PIIType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            PIIType.PHONE: re.compile(
                r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
            ),
            PIIType.SSN: re.compile(
                r'\b\d{3}-?\d{2}-?\d{4}\b'
            ),
            PIIType.CREDIT_CARD: re.compile(
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
            ),
            PIIType.IP_ADDRESS: re.compile(
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ),
            PIIType.NAME: re.compile(
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'  # Simple name pattern
            ),
            PIIType.DATE_OF_BIRTH: re.compile(
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
            ),
        }
        
        # Anonymization replacements
        self.anonymization_map = {
            PIIType.EMAIL: "[EMAIL]",
            PIIType.PHONE: "[PHONE]",
            PIIType.SSN: "[SSN]",
            PIIType.CREDIT_CARD: "[CREDIT_CARD]",
            PIIType.IP_ADDRESS: "[IP_ADDRESS]",
            PIIType.NAME: "[NAME]",
            PIIType.ADDRESS: "[ADDRESS]",
            PIIType.DATE_OF_BIRTH: "[DOB]",
            PIIType.PASSPORT: "[PASSPORT]",
            PIIType.DRIVER_LICENSE: "[DRIVER_LICENSE]",
        }
    
    def detect_pii(self, text: str) -> PIIDetectionResult:
        """Detect PII in text."""
        found_pii: Dict[PIIType, List[str]] = {}
        confidence_scores: Dict[PIIType, float] = {}
        anonymized_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            
            if matches:
                # Handle different match formats
                if isinstance(matches[0], tuple):
                    # For patterns with groups (like phone numbers)
                    match_strings = [''.join(match) for match in matches]
                else:
                    match_strings = matches
                
                found_pii[pii_type] = match_strings
                confidence_scores[pii_type] = self._calculate_confidence(pii_type, match_strings)
                
                # Anonymize matches
                replacement = self.anonymization_map.get(pii_type, "[REDACTED]")
                for match in match_strings:
                    anonymized_text = anonymized_text.replace(match, replacement)
        
        return PIIDetectionResult(
            found_pii=found_pii,
            anonymized_text=anonymized_text,
            confidence_scores=confidence_scores
        )
    
    def _calculate_confidence(self, pii_type: PIIType, matches: List[str]) -> float:
        """Calculate confidence score for PII detection."""
        # Simple confidence calculation based on pattern strength
        base_confidence = {
            PIIType.EMAIL: 0.95,
            PIIType.PHONE: 0.85,
            PIIType.SSN: 0.90,
            PIIType.CREDIT_CARD: 0.80,
            PIIType.IP_ADDRESS: 0.70,
            PIIType.NAME: 0.60,  # Lower confidence for name patterns
            PIIType.DATE_OF_BIRTH: 0.75,
        }
        
        return base_confidence.get(pii_type, 0.50)
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize PII in structured data."""
        anonymized_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                pii_result = self.detect_pii(value)
                anonymized_data[key] = pii_result.anonymized_text
            elif isinstance(value, dict):
                anonymized_data[key] = self.anonymize_data(value)
            elif isinstance(value, list):
                anonymized_data[key] = [
                    self.anonymize_data(item) if isinstance(item, dict)
                    else self.detect_pii(item).anonymized_text if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                anonymized_data[key] = value
        
        return anonymized_data


class AuditLogger:
    """Comprehensive audit logging service."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "~/.quantum_moe_mas/audit_logs").expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.pii_detector = PIIDetector()
        self.audit_events: List[AuditEvent] = []
        
        # Compliance mappings
        self.compliance_mappings = {
            ComplianceStandard.SOC2: {
                AuditEventType.USER_LOGIN,
                AuditEventType.USER_LOGOUT,
                AuditEventType.LOGIN_FAILED,
                AuditEventType.PERMISSION_DENIED,
                AuditEventType.DATA_READ,
                AuditEventType.DATA_WRITE,
                AuditEventType.DATA_DELETE,
                AuditEventType.SYSTEM_CONFIG_CHANGE,
                AuditEventType.SECURITY_VIOLATION,
            },
            ComplianceStandard.GDPR: {
                AuditEventType.PII_ACCESS,
                AuditEventType.PII_ANONYMIZED,
                AuditEventType.DATA_EXPORT,
                AuditEventType.DATA_DELETE,
                AuditEventType.CONSENT_GRANTED,
                AuditEventType.CONSENT_REVOKED,
                AuditEventType.DATA_RETENTION_APPLIED,
            },
            ComplianceStandard.HIPAA: {
                AuditEventType.PII_ACCESS,
                AuditEventType.DATA_READ,
                AuditEventType.DATA_WRITE,
                AuditEventType.DATA_EXPORT,
                AuditEventType.USER_LOGIN,
                AuditEventType.USER_LOGOUT,
                AuditEventType.PERMISSION_DENIED,
            },
        }
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AuditEvent:
        """Log audit event."""
        event_id = str(uuid.uuid4())
        
        # Detect PII in details
        pii_detected = set()
        anonymized_details = details or {}
        
        if details:
            for key, value in details.items():
                if isinstance(value, str):
                    pii_result = self.pii_detector.detect_pii(value)
                    if pii_result.found_pii:
                        pii_detected.update(pii_result.found_pii.keys())
                        # Store anonymized version
                        anonymized_details[key] = pii_result.anonymized_text
        
        # Determine compliance tags
        compliance_tags = set()
        for standard, event_types in self.compliance_mappings.items():
            if event_type in event_types:
                compliance_tags.add(standard)
        
        # Add GDPR tag if PII detected
        if pii_detected:
            compliance_tags.add(ComplianceStandard.GDPR)
        
        audit_event = AuditEvent(
            id=event_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            session_id=session_id,
            source_ip=source_ip,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            details=anonymized_details,
            compliance_tags=compliance_tags,
            pii_detected=pii_detected,
            metadata=kwargs
        )
        
        # Store event
        self.audit_events.append(audit_event)
        await self._persist_event(audit_event)
        
        # Log to security logger
        security_logger.data_access(
            user_id=user_id,
            resource=resource,
            action=action,
            sensitive=bool(pii_detected),
            event_type=event_type.value,
            result=result
        )
        
        logger.info(
            "Audit event logged",
            event_id=event_id,
            event_type=event_type.value,
            user_id=user_id,
            resource=resource,
            compliance_tags=[tag.value for tag in compliance_tags]
        )
        
        return audit_event
    
    async def _persist_event(self, event: AuditEvent) -> None:
        """Persist audit event to storage."""
        # Create daily log file
        date_str = event.timestamp.strftime("%Y-%m-%d")
        log_file = self.storage_path / f"audit_{date_str}.jsonl"
        
        # Append event to log file
        with open(log_file, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')
    
    async def search_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        compliance_standard: Optional[ComplianceStandard] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Search audit events with filters."""
        filtered_events = []
        
        for event in self.audit_events:
            # Apply filters
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if event_types and event.event_type not in event_types:
                continue
            if user_id and event.user_id != user_id:
                continue
            if resource and event.resource != resource:
                continue
            if compliance_standard and compliance_standard not in event.compliance_tags:
                continue
            
            filtered_events.append(event)
            
            if len(filtered_events) >= limit:
                break
        
        return filtered_events
    
    async def get_compliance_events(
        self,
        standard: ComplianceStandard,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Get events relevant to specific compliance standard."""
        return await self.search_events(
            start_time=start_time,
            end_time=end_time,
            compliance_standard=standard
        )


class ComplianceReporter:
    """Compliance reporting service."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
    
    async def generate_soc2_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate SOC 2 compliance report."""
        events = await self.audit_logger.get_compliance_events(
            ComplianceStandard.SOC2,
            start_date,
            end_date
        )
        
        # SOC 2 Trust Service Criteria analysis
        security_events = [e for e in events if e.event_type in {
            AuditEventType.SECURITY_VIOLATION,
            AuditEventType.MALICIOUS_ACTIVITY,
            AuditEventType.LOGIN_FAILED,
            AuditEventType.PERMISSION_DENIED
        }]
        
        availability_events = [e for e in events if e.event_type in {
            AuditEventType.SERVICE_START,
            AuditEventType.SERVICE_STOP,
            AuditEventType.SYSTEM_CONFIG_CHANGE
        }]
        
        processing_integrity_events = [e for e in events if e.event_type in {
            AuditEventType.DATA_WRITE,
            AuditEventType.DATA_DELETE,
            AuditEventType.BACKUP_CREATED,
            AuditEventType.BACKUP_RESTORED
        }]
        
        confidentiality_events = [e for e in events if e.event_type in {
            AuditEventType.DATA_READ,
            AuditEventType.DATA_EXPORT,
            AuditEventType.PII_ACCESS
        }]
        
        return {
            "report_type": "SOC 2 Type II",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "trust_service_criteria": {
                "security": {
                    "total_events": len(security_events),
                    "violations": len([e for e in security_events if e.result == "failure"]),
                    "controls_effective": len([e for e in security_events if e.result == "success"]) > 0
                },
                "availability": {
                    "total_events": len(availability_events),
                    "uptime_events": len([e for e in availability_events if e.event_type == AuditEventType.SERVICE_START]),
                    "downtime_events": len([e for e in availability_events if e.event_type == AuditEventType.SERVICE_STOP])
                },
                "processing_integrity": {
                    "total_events": len(processing_integrity_events),
                    "successful_operations": len([e for e in processing_integrity_events if e.result == "success"]),
                    "failed_operations": len([e for e in processing_integrity_events if e.result == "failure"])
                },
                "confidentiality": {
                    "total_events": len(confidentiality_events),
                    "authorized_access": len([e for e in confidentiality_events if e.result == "success"]),
                    "unauthorized_attempts": len([e for e in confidentiality_events if e.result == "failure"])
                }
            },
            "summary": {
                "total_audit_events": len(events),
                "compliance_score": self._calculate_soc2_score(events),
                "recommendations": self._generate_soc2_recommendations(events)
            }
        }
    
    async def generate_gdpr_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate GDPR compliance report."""
        events = await self.audit_logger.get_compliance_events(
            ComplianceStandard.GDPR,
            start_date,
            end_date
        )
        
        # GDPR-specific analysis
        pii_access_events = [e for e in events if PIIType.EMAIL in e.pii_detected or 
                           PIIType.NAME in e.pii_detected or 
                           len(e.pii_detected) > 0]
        
        consent_events = [e for e in events if e.event_type in {
            AuditEventType.CONSENT_GRANTED,
            AuditEventType.CONSENT_REVOKED
        }]
        
        data_subject_rights_events = [e for e in events if e.event_type in {
            AuditEventType.DATA_EXPORT,
            AuditEventType.DATA_DELETE,
            AuditEventType.PII_ANONYMIZED
        }]
        
        return {
            "report_type": "GDPR Compliance Report",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "gdpr_principles": {
                "lawfulness_fairness_transparency": {
                    "consent_events": len(consent_events),
                    "consent_granted": len([e for e in consent_events if e.event_type == AuditEventType.CONSENT_GRANTED]),
                    "consent_revoked": len([e for e in consent_events if e.event_type == AuditEventType.CONSENT_REVOKED])
                },
                "purpose_limitation": {
                    "pii_access_events": len(pii_access_events),
                    "authorized_access": len([e for e in pii_access_events if e.result == "success"]),
                    "unauthorized_attempts": len([e for e in pii_access_events if e.result == "failure"])
                },
                "data_minimization": {
                    "anonymization_events": len([e for e in events if e.event_type == AuditEventType.PII_ANONYMIZED]),
                    "retention_events": len([e for e in events if e.event_type == AuditEventType.DATA_RETENTION_APPLIED])
                },
                "accuracy": {
                    "data_updates": len([e for e in events if e.event_type == AuditEventType.DATA_WRITE]),
                    "data_corrections": len([e for e in events if "correction" in e.details.get("action", "")])
                },
                "storage_limitation": {
                    "retention_applied": len([e for e in events if e.event_type == AuditEventType.DATA_RETENTION_APPLIED]),
                    "data_deleted": len([e for e in events if e.event_type == AuditEventType.DATA_DELETE])
                },
                "integrity_confidentiality": {
                    "security_events": len([e for e in events if e.event_type == AuditEventType.SECURITY_VIOLATION]),
                    "encryption_events": len([e for e in events if "encrypt" in str(e.details).lower()])
                }
            },
            "data_subject_rights": {
                "right_of_access": len([e for e in data_subject_rights_events if e.event_type == AuditEventType.DATA_EXPORT]),
                "right_to_rectification": len([e for e in events if "rectification" in e.details.get("action", "")]),
                "right_to_erasure": len([e for e in data_subject_rights_events if e.event_type == AuditEventType.DATA_DELETE]),
                "right_to_portability": len([e for e in data_subject_rights_events if e.event_type == AuditEventType.DATA_EXPORT]),
                "right_to_object": len([e for e in consent_events if e.event_type == AuditEventType.CONSENT_REVOKED])
            },
            "summary": {
                "total_audit_events": len(events),
                "pii_events": len(pii_access_events),
                "compliance_score": self._calculate_gdpr_score(events),
                "recommendations": self._generate_gdpr_recommendations(events)
            }
        }
    
    async def generate_hipaa_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate HIPAA compliance report."""
        events = await self.audit_logger.get_compliance_events(
            ComplianceStandard.HIPAA,
            start_date,
            end_date
        )
        
        # HIPAA-specific analysis
        phi_access_events = [e for e in events if e.pii_detected or 
                           e.event_type in {AuditEventType.PII_ACCESS, AuditEventType.DATA_READ}]
        
        administrative_events = [e for e in events if e.event_type in {
            AuditEventType.USER_LOGIN,
            AuditEventType.USER_LOGOUT,
            AuditEventType.PERMISSION_GRANTED,
            AuditEventType.PERMISSION_DENIED
        }]
        
        physical_events = [e for e in events if e.event_type in {
            AuditEventType.SYSTEM_CONFIG_CHANGE,
            AuditEventType.SERVICE_START,
            AuditEventType.SERVICE_STOP
        }]
        
        technical_events = [e for e in events if e.event_type in {
            AuditEventType.DATA_READ,
            AuditEventType.DATA_WRITE,
            AuditEventType.DATA_EXPORT,
            AuditEventType.SECURITY_VIOLATION
        }]
        
        return {
            "report_type": "HIPAA Compliance Report",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "hipaa_safeguards": {
                "administrative": {
                    "total_events": len(administrative_events),
                    "successful_logins": len([e for e in administrative_events if e.event_type == AuditEventType.USER_LOGIN and e.result == "success"]),
                    "failed_logins": len([e for e in administrative_events if e.event_type == AuditEventType.LOGIN_FAILED]),
                    "access_granted": len([e for e in administrative_events if e.event_type == AuditEventType.PERMISSION_GRANTED]),
                    "access_denied": len([e for e in administrative_events if e.event_type == AuditEventType.PERMISSION_DENIED])
                },
                "physical": {
                    "total_events": len(physical_events),
                    "system_changes": len([e for e in physical_events if e.event_type == AuditEventType.SYSTEM_CONFIG_CHANGE]),
                    "service_events": len([e for e in physical_events if e.event_type in {AuditEventType.SERVICE_START, AuditEventType.SERVICE_STOP}])
                },
                "technical": {
                    "total_events": len(technical_events),
                    "phi_access": len(phi_access_events),
                    "data_operations": len([e for e in technical_events if e.event_type in {AuditEventType.DATA_READ, AuditEventType.DATA_WRITE}]),
                    "security_incidents": len([e for e in technical_events if e.event_type == AuditEventType.SECURITY_VIOLATION])
                }
            },
            "phi_access_summary": {
                "total_phi_access_events": len(phi_access_events),
                "authorized_access": len([e for e in phi_access_events if e.result == "success"]),
                "unauthorized_attempts": len([e for e in phi_access_events if e.result == "failure"]),
                "unique_users": len(set(e.user_id for e in phi_access_events if e.user_id))
            },
            "summary": {
                "total_audit_events": len(events),
                "compliance_score": self._calculate_hipaa_score(events),
                "recommendations": self._generate_hipaa_recommendations(events)
            }
        }
    
    def _calculate_soc2_score(self, events: List[AuditEvent]) -> float:
        """Calculate SOC 2 compliance score."""
        if not events:
            return 0.0
        
        total_events = len(events)
        successful_events = len([e for e in events if e.result == "success"])
        
        # Basic compliance score based on success rate
        base_score = (successful_events / total_events) * 100
        
        # Deduct points for security violations
        security_violations = len([e for e in events if e.event_type == AuditEventType.SECURITY_VIOLATION])
        violation_penalty = min(security_violations * 5, 30)  # Max 30 point penalty
        
        return max(0, base_score - violation_penalty)
    
    def _calculate_gdpr_score(self, events: List[AuditEvent]) -> float:
        """Calculate GDPR compliance score."""
        if not events:
            return 0.0
        
        # Score based on proper handling of PII and consent
        pii_events = [e for e in events if e.pii_detected]
        consent_events = [e for e in events if e.event_type in {AuditEventType.CONSENT_GRANTED, AuditEventType.CONSENT_REVOKED}]
        
        # Base score
        base_score = 70.0
        
        # Add points for proper PII handling
        if pii_events:
            anonymized_events = len([e for e in events if e.event_type == AuditEventType.PII_ANONYMIZED])
            pii_score = (anonymized_events / len(pii_events)) * 20
            base_score += pii_score
        
        # Add points for consent management
        if consent_events:
            base_score += 10
        
        return min(100, base_score)
    
    def _calculate_hipaa_score(self, events: List[AuditEvent]) -> float:
        """Calculate HIPAA compliance score."""
        if not events:
            return 0.0
        
        # Score based on proper PHI access controls
        phi_events = [e for e in events if e.pii_detected or e.event_type == AuditEventType.PII_ACCESS]
        
        if not phi_events:
            return 100.0  # No PHI access, perfect score
        
        authorized_access = len([e for e in phi_events if e.result == "success"])
        unauthorized_attempts = len([e for e in phi_events if e.result == "failure"])
        
        # Base score from authorized access ratio
        if phi_events:
            base_score = (authorized_access / len(phi_events)) * 80
        else:
            base_score = 80
        
        # Add points for audit completeness
        audit_completeness = min(len(events) / 100, 1.0) * 20  # Up to 20 points for comprehensive auditing
        
        # Deduct points for unauthorized attempts
        unauthorized_penalty = min(unauthorized_attempts * 2, 20)  # Max 20 point penalty
        
        return max(0, base_score + audit_completeness - unauthorized_penalty)
    
    def _generate_soc2_recommendations(self, events: List[AuditEvent]) -> List[str]:
        """Generate SOC 2 recommendations."""
        recommendations = []
        
        security_violations = len([e for e in events if e.event_type == AuditEventType.SECURITY_VIOLATION])
        if security_violations > 0:
            recommendations.append("Implement additional security controls to reduce security violations")
        
        failed_logins = len([e for e in events if e.event_type == AuditEventType.LOGIN_FAILED])
        if failed_logins > 10:
            recommendations.append("Consider implementing account lockout policies to prevent brute force attacks")
        
        config_changes = len([e for e in events if e.event_type == AuditEventType.SYSTEM_CONFIG_CHANGE])
        if config_changes > 5:
            recommendations.append("Implement change management controls for system configuration changes")
        
        return recommendations
    
    def _generate_gdpr_recommendations(self, events: List[AuditEvent]) -> List[str]:
        """Generate GDPR recommendations."""
        recommendations = []
        
        pii_events = [e for e in events if e.pii_detected]
        anonymized_events = len([e for e in events if e.event_type == AuditEventType.PII_ANONYMIZED])
        
        if pii_events and anonymized_events == 0:
            recommendations.append("Implement PII anonymization procedures to comply with data minimization principle")
        
        consent_events = [e for e in events if e.event_type in {AuditEventType.CONSENT_GRANTED, AuditEventType.CONSENT_REVOKED}]
        if not consent_events:
            recommendations.append("Implement consent management system to track user consent")
        
        retention_events = len([e for e in events if e.event_type == AuditEventType.DATA_RETENTION_APPLIED])
        if retention_events == 0:
            recommendations.append("Implement data retention policies and automated deletion procedures")
        
        return recommendations
    
    def _generate_hipaa_recommendations(self, events: List[AuditEvent]) -> List[str]:
        """Generate HIPAA recommendations."""
        recommendations = []
        
        phi_events = [e for e in events if e.pii_detected or e.event_type == AuditEventType.PII_ACCESS]
        unauthorized_attempts = len([e for e in phi_events if e.result == "failure"])
        
        if unauthorized_attempts > 0:
            recommendations.append("Strengthen access controls to prevent unauthorized PHI access")
        
        unique_users = len(set(e.user_id for e in phi_events if e.user_id))
        if unique_users > 10:
            recommendations.append("Review user access permissions to ensure minimum necessary access to PHI")
        
        backup_events = len([e for e in events if e.event_type in {AuditEventType.BACKUP_CREATED, AuditEventType.BACKUP_RESTORED}])
        if backup_events == 0:
            recommendations.append("Implement regular backup procedures for PHI data protection")
        
        return recommendations


class DataRetentionManager:
    """Data retention and secure deletion manager."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.retention_policies: Dict[str, timedelta] = {
            "audit_logs": timedelta(days=2555),  # 7 years for audit logs
            "user_data": timedelta(days=1095),   # 3 years for user data
            "session_data": timedelta(days=30),  # 30 days for session data
            "temp_data": timedelta(days=1),      # 1 day for temporary data
        }
    
    async def apply_retention_policy(self, data_type: str) -> int:
        """Apply retention policy and delete expired data."""
        if data_type not in self.retention_policies:
            logger.warning(f"No retention policy defined for {data_type}")
            return 0
        
        retention_period = self.retention_policies[data_type]
        cutoff_date = datetime.now(timezone.utc) - retention_period
        
        # This would typically interact with the database
        # For now, we'll just log the retention action
        deleted_count = 0  # Placeholder
        
        await self.audit_logger.log_event(
            AuditEventType.DATA_RETENTION_APPLIED,
            resource=data_type,
            action="delete_expired",
            details={
                "retention_period_days": retention_period.days,
                "cutoff_date": cutoff_date.isoformat(),
                "deleted_records": deleted_count
            }
        )
        
        logger.info(f"Applied retention policy for {data_type}, deleted {deleted_count} records")
        return deleted_count
    
    def set_retention_policy(self, data_type: str, retention_period: timedelta) -> None:
        """Set retention policy for data type."""
        self.retention_policies[data_type] = retention_period
        logger.info(f"Set retention policy for {data_type}: {retention_period.days} days")


# Global instances
pii_detector = PIIDetector()
audit_logger = AuditLogger()
compliance_reporter = ComplianceReporter(audit_logger)
data_retention_manager = DataRetentionManager(audit_logger)