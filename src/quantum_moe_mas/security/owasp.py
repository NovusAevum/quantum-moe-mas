"""
OWASP Top 10 Compliance Framework.

This module implements comprehensive security controls based on the OWASP Top 10
security risks, providing automated vulnerability assessment and security scanning.
"""

import asyncio
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from quantum_moe_mas.core.logging import get_logger, get_security_logger
from quantum_moe_mas.core.exceptions import QuantumMoEMASError
from quantum_moe_mas.config.settings import get_settings

logger = get_logger(__name__)
security_logger = get_security_logger(__name__)
settings = get_settings()


class OWASPRisk(Enum):
    """OWASP Top 10 security risks."""
    
    A01_BROKEN_ACCESS_CONTROL = "A01:2021 – Broken Access Control"
    A02_CRYPTOGRAPHIC_FAILURES = "A02:2021 – Cryptographic Failures"
    A03_INJECTION = "A03:2021 – Injection"
    A04_INSECURE_DESIGN = "A04:2021 – Insecure Design"
    A05_SECURITY_MISCONFIGURATION = "A05:2021 – Security Misconfiguration"
    A06_VULNERABLE_COMPONENTS = "A06:2021 – Vulnerable and Outdated Components"
    A07_IDENTIFICATION_FAILURES = "A07:2021 – Identification and Authentication Failures"
    A08_SOFTWARE_INTEGRITY_FAILURES = "A08:2021 – Software and Data Integrity Failures"
    A09_LOGGING_FAILURES = "A09:2021 – Security Logging and Monitoring Failures"
    A10_SSRF = "A10:2021 – Server-Side Request Forgery (SSRF)"


class SeverityLevel(Enum):
    """Vulnerability severity levels."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityStatus(Enum):
    """Vulnerability status."""
    
    OPEN = "open"
    FIXED = "fixed"
    ACCEPTED = "accepted"
    FALSE_POSITIVE = "false_positive"


@dataclass
class Vulnerability:
    """Security vulnerability finding."""
    
    id: str
    title: str
    description: str
    owasp_category: OWASPRisk
    severity: SeverityLevel
    status: VulnerabilityStatus = VulnerabilityStatus.OPEN
    location: Optional[str] = None
    line_number: Optional[int] = None
    evidence: Optional[str] = None
    recommendation: Optional[str] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "owasp_category": self.owasp_category.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "location": self.location,
            "line_number": self.line_number,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "cwe_id": self.cwe_id,
            "cvss_score": self.cvss_score,
            "discovered_at": self.discovered_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SecurityScanResult:
    """Result of security scan."""
    
    scan_id: str
    scan_type: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_vulnerability(self, vulnerability: Vulnerability) -> None:
        """Add vulnerability to scan result."""
        self.vulnerabilities.append(vulnerability)
        
        # Update summary
        severity = vulnerability.severity.value
        self.summary[severity] = self.summary.get(severity, 0) + 1
    
    def get_critical_vulnerabilities(self) -> List[Vulnerability]:
        """Get critical severity vulnerabilities."""
        return [v for v in self.vulnerabilities if v.severity == SeverityLevel.CRITICAL]
    
    def get_high_vulnerabilities(self) -> List[Vulnerability]:
        """Get high severity vulnerabilities."""
        return [v for v in self.vulnerabilities if v.severity == SeverityLevel.HIGH]
    
    def has_critical_issues(self) -> bool:
        """Check if scan found critical issues."""
        return len(self.get_critical_vulnerabilities()) > 0


class AccessControlScanner:
    """Scanner for broken access control vulnerabilities (A01)."""
    
    def __init__(self):
        self.patterns = {
            "missing_auth": [
                r"@app\.route\([^)]*\)\s*\ndef\s+\w+\([^)]*\):",  # Flask routes without auth
                r"@router\.(get|post|put|delete)\([^)]*\)\s*\nasync\s+def\s+\w+\([^)]*\):",  # FastAPI routes
            ],
            "hardcoded_permissions": [
                r"if\s+user\.role\s*==\s*['\"]admin['\"]",
                r"if\s+user_type\s*==\s*['\"]superuser['\"]",
            ],
            "path_traversal": [
                r"open\([^)]*\.\./",
                r"file_path\s*=.*\.\./",
            ],
        }
    
    async def scan(self, code_content: str, file_path: str) -> List[Vulnerability]:
        """Scan for access control vulnerabilities."""
        vulnerabilities = []
        
        for vuln_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code_content, re.MULTILINE | re.IGNORECASE)
                
                for match in matches:
                    line_number = code_content[:match.start()].count('\n') + 1
                    
                    vulnerability = Vulnerability(
                        id=f"AC_{vuln_type}_{file_path}_{line_number}",
                        title=f"Potential Access Control Issue: {vuln_type}",
                        description=f"Detected potential access control vulnerability in {file_path}",
                        owasp_category=OWASPRisk.A01_BROKEN_ACCESS_CONTROL,
                        severity=SeverityLevel.HIGH,
                        location=file_path,
                        line_number=line_number,
                        evidence=match.group(0),
                        recommendation="Implement proper access control checks and authorization mechanisms",
                        cwe_id="CWE-284"
                    )
                    vulnerabilities.append(vulnerability)
        
        return vulnerabilities


class CryptographicScanner:
    """Scanner for cryptographic failures (A02)."""
    
    def __init__(self):
        self.patterns = {
            "weak_crypto": [
                r"hashlib\.md5\(",
                r"hashlib\.sha1\(",
                r"DES\(",
                r"RC4\(",
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*['\"][^'\"]{8,}['\"]",
                r"api_key\s*=\s*['\"][^'\"]{16,}['\"]",
                r"secret\s*=\s*['\"][^'\"]{8,}['\"]",
            ],
            "insecure_random": [
                r"random\.random\(",
                r"random\.randint\(",
            ],
        }
    
    async def scan(self, code_content: str, file_path: str) -> List[Vulnerability]:
        """Scan for cryptographic vulnerabilities."""
        vulnerabilities = []
        
        for vuln_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code_content, re.MULTILINE | re.IGNORECASE)
                
                for match in matches:
                    line_number = code_content[:match.start()].count('\n') + 1
                    
                    severity = SeverityLevel.CRITICAL if vuln_type == "hardcoded_secrets" else SeverityLevel.HIGH
                    
                    vulnerability = Vulnerability(
                        id=f"CRYPTO_{vuln_type}_{file_path}_{line_number}",
                        title=f"Cryptographic Issue: {vuln_type}",
                        description=f"Detected cryptographic vulnerability in {file_path}",
                        owasp_category=OWASPRisk.A02_CRYPTOGRAPHIC_FAILURES,
                        severity=severity,
                        location=file_path,
                        line_number=line_number,
                        evidence=match.group(0),
                        recommendation="Use strong cryptographic algorithms and secure key management",
                        cwe_id="CWE-327"
                    )
                    vulnerabilities.append(vulnerability)
        
        return vulnerabilities


class InjectionScanner:
    """Scanner for injection vulnerabilities (A03)."""
    
    def __init__(self):
        self.patterns = {
            "sql_injection": [
                r"execute\([^)]*%[^)]*\)",
                r"query\([^)]*\+[^)]*\)",
                r"cursor\.execute\([^)]*%[^)]*\)",
            ],
            "command_injection": [
                r"os\.system\([^)]*\+[^)]*\)",
                r"subprocess\.(call|run|Popen)\([^)]*\+[^)]*\)",
                r"eval\([^)]*input[^)]*\)",
            ],
            "ldap_injection": [
                r"ldap\.search\([^)]*\+[^)]*\)",
            ],
        }
    
    async def scan(self, code_content: str, file_path: str) -> List[Vulnerability]:
        """Scan for injection vulnerabilities."""
        vulnerabilities = []
        
        for vuln_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code_content, re.MULTILINE | re.IGNORECASE)
                
                for match in matches:
                    line_number = code_content[:match.start()].count('\n') + 1
                    
                    vulnerability = Vulnerability(
                        id=f"INJ_{vuln_type}_{file_path}_{line_number}",
                        title=f"Injection Vulnerability: {vuln_type}",
                        description=f"Detected potential injection vulnerability in {file_path}",
                        owasp_category=OWASPRisk.A03_INJECTION,
                        severity=SeverityLevel.CRITICAL,
                        location=file_path,
                        line_number=line_number,
                        evidence=match.group(0),
                        recommendation="Use parameterized queries and input validation",
                        cwe_id="CWE-89" if vuln_type == "sql_injection" else "CWE-78"
                    )
                    vulnerabilities.append(vulnerability)
        
        return vulnerabilities


class SecurityConfigurationScanner:
    """Scanner for security misconfiguration (A05)."""
    
    def __init__(self):
        self.config_checks = {
            "debug_enabled": [
                r"DEBUG\s*=\s*True",
                r"debug\s*=\s*True",
            ],
            "default_passwords": [
                r"password\s*=\s*['\"]admin['\"]",
                r"password\s*=\s*['\"]password['\"]",
                r"password\s*=\s*['\"]123456['\"]",
            ],
            "insecure_headers": [
                r"X-Frame-Options.*ALLOWALL",
                r"Access-Control-Allow-Origin.*\*",
            ],
        }
    
    async def scan(self, code_content: str, file_path: str) -> List[Vulnerability]:
        """Scan for security misconfigurations."""
        vulnerabilities = []
        
        for vuln_type, patterns in self.config_checks.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code_content, re.MULTILINE | re.IGNORECASE)
                
                for match in matches:
                    line_number = code_content[:match.start()].count('\n') + 1
                    
                    vulnerability = Vulnerability(
                        id=f"CONFIG_{vuln_type}_{file_path}_{line_number}",
                        title=f"Security Misconfiguration: {vuln_type}",
                        description=f"Detected security misconfiguration in {file_path}",
                        owasp_category=OWASPRisk.A05_SECURITY_MISCONFIGURATION,
                        severity=SeverityLevel.MEDIUM,
                        location=file_path,
                        line_number=line_number,
                        evidence=match.group(0),
                        recommendation="Review and harden security configuration",
                        cwe_id="CWE-16"
                    )
                    vulnerabilities.append(vulnerability)
        
        return vulnerabilities


class VulnerableComponentsScanner:
    """Scanner for vulnerable and outdated components (A06)."""
    
    async def scan_dependencies(self, requirements_file: str) -> List[Vulnerability]:
        """Scan dependencies for known vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Use safety to check for known vulnerabilities
            result = subprocess.run(
                ["safety", "check", "-r", requirements_file, "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # No vulnerabilities found
                return vulnerabilities
            
            # Parse safety output
            try:
                safety_data = json.loads(result.stdout)
                
                for vuln_data in safety_data:
                    vulnerability = Vulnerability(
                        id=f"VULN_COMP_{vuln_data.get('package')}_{vuln_data.get('id')}",
                        title=f"Vulnerable Component: {vuln_data.get('package')}",
                        description=vuln_data.get('advisory', 'Known vulnerability in dependency'),
                        owasp_category=OWASPRisk.A06_VULNERABLE_COMPONENTS,
                        severity=SeverityLevel.HIGH,
                        location=requirements_file,
                        evidence=f"Package: {vuln_data.get('package')} {vuln_data.get('installed_version')}",
                        recommendation=f"Update to version {vuln_data.get('safe_versions', 'latest')}",
                        cve_id=vuln_data.get('cve'),
                        metadata=vuln_data
                    )
                    vulnerabilities.append(vulnerability)
            
            except json.JSONDecodeError:
                logger.warning("Failed to parse safety output")
        
        except subprocess.TimeoutExpired:
            logger.warning("Safety scan timed out")
        except FileNotFoundError:
            logger.warning("Safety tool not found - install with 'pip install safety'")
        except Exception as e:
            logger.error(f"Error running safety scan: {e}")
        
        return vulnerabilities


class SecurityScanner:
    """Comprehensive security scanner implementing OWASP Top 10 checks."""
    
    def __init__(self):
        self.access_control_scanner = AccessControlScanner()
        self.crypto_scanner = CryptographicScanner()
        self.injection_scanner = InjectionScanner()
        self.config_scanner = SecurityConfigurationScanner()
        self.component_scanner = VulnerableComponentsScanner()
    
    async def scan_file(self, file_path: str) -> List[Vulnerability]:
        """Scan single file for vulnerabilities."""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Run all scanners
            scanners = [
                self.access_control_scanner.scan(content, file_path),
                self.crypto_scanner.scan(content, file_path),
                self.injection_scanner.scan(content, file_path),
                self.config_scanner.scan(content, file_path),
            ]
            
            results = await asyncio.gather(*scanners)
            
            for scanner_results in results:
                vulnerabilities.extend(scanner_results)
        
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
        
        return vulnerabilities
    
    async def scan_directory(self, directory_path: str) -> SecurityScanResult:
        """Scan directory for vulnerabilities."""
        scan_id = f"scan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        scan_result = SecurityScanResult(
            scan_id=scan_id,
            scan_type="directory_scan",
            started_at=datetime.now(timezone.utc)
        )
        
        # Find Python files to scan
        python_files = []
        directory = Path(directory_path)
        
        for file_path in directory.rglob("*.py"):
            if not any(part.startswith('.') for part in file_path.parts):
                python_files.append(str(file_path))
        
        logger.info(f"Scanning {len(python_files)} Python files")
        
        # Scan files concurrently
        scan_tasks = [self.scan_file(file_path) for file_path in python_files]
        results = await asyncio.gather(*scan_tasks)
        
        # Collect all vulnerabilities
        for file_vulnerabilities in results:
            for vulnerability in file_vulnerabilities:
                scan_result.add_vulnerability(vulnerability)
        
        # Scan dependencies if requirements file exists
        requirements_files = ["requirements.txt", "pyproject.toml", "Pipfile"]
        for req_file in requirements_files:
            req_path = directory / req_file
            if req_path.exists():
                dep_vulnerabilities = await self.component_scanner.scan_dependencies(str(req_path))
                for vulnerability in dep_vulnerabilities:
                    scan_result.add_vulnerability(vulnerability)
                break
        
        scan_result.completed_at = datetime.now(timezone.utc)
        
        # Log scan summary
        logger.info(
            "Security scan completed",
            scan_id=scan_id,
            total_vulnerabilities=len(scan_result.vulnerabilities),
            critical=scan_result.summary.get('critical', 0),
            high=scan_result.summary.get('high', 0),
            medium=scan_result.summary.get('medium', 0),
            low=scan_result.summary.get('low', 0)
        )
        
        return scan_result


class VulnerabilityAssessment:
    """Vulnerability assessment and management service."""
    
    def __init__(self):
        self.scanner = SecurityScanner()
        self.vulnerabilities: Dict[str, Vulnerability] = {}
    
    async def assess_project(self, project_path: str) -> SecurityScanResult:
        """Perform comprehensive vulnerability assessment."""
        logger.info(f"Starting vulnerability assessment for {project_path}")
        
        scan_result = await self.scanner.scan_directory(project_path)
        
        # Store vulnerabilities
        for vulnerability in scan_result.vulnerabilities:
            self.vulnerabilities[vulnerability.id] = vulnerability
        
        # Generate security report
        await self._generate_security_report(scan_result)
        
        return scan_result
    
    async def _generate_security_report(self, scan_result: SecurityScanResult) -> None:
        """Generate detailed security report."""
        report_data = {
            "scan_summary": {
                "scan_id": scan_result.scan_id,
                "started_at": scan_result.started_at.isoformat(),
                "completed_at": scan_result.completed_at.isoformat() if scan_result.completed_at else None,
                "total_vulnerabilities": len(scan_result.vulnerabilities),
                "severity_breakdown": scan_result.summary,
            },
            "owasp_coverage": self._analyze_owasp_coverage(scan_result.vulnerabilities),
            "critical_issues": [v.to_dict() for v in scan_result.get_critical_vulnerabilities()],
            "high_issues": [v.to_dict() for v in scan_result.get_high_vulnerabilities()],
            "recommendations": self._generate_recommendations(scan_result.vulnerabilities),
        }
        
        # Save report
        report_path = f"security_report_{scan_result.scan_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Security report saved to {report_path}")
    
    def _analyze_owasp_coverage(self, vulnerabilities: List[Vulnerability]) -> Dict[str, int]:
        """Analyze OWASP Top 10 coverage."""
        owasp_counts = {}
        
        for vulnerability in vulnerabilities:
            category = vulnerability.owasp_category.value
            owasp_counts[category] = owasp_counts.get(category, 0) + 1
        
        return owasp_counts
    
    def _generate_recommendations(self, vulnerabilities: List[Vulnerability]) -> List[str]:
        """Generate security recommendations."""
        recommendations = set()
        
        for vulnerability in vulnerabilities:
            if vulnerability.recommendation:
                recommendations.add(vulnerability.recommendation)
        
        return list(recommendations)
    
    def get_vulnerability_by_id(self, vulnerability_id: str) -> Optional[Vulnerability]:
        """Get vulnerability by ID."""
        return self.vulnerabilities.get(vulnerability_id)
    
    def update_vulnerability_status(
        self,
        vulnerability_id: str,
        status: VulnerabilityStatus,
        notes: Optional[str] = None
    ) -> bool:
        """Update vulnerability status."""
        vulnerability = self.vulnerabilities.get(vulnerability_id)
        if not vulnerability:
            return False
        
        old_status = vulnerability.status
        vulnerability.status = status
        
        if notes:
            vulnerability.metadata["status_notes"] = notes
        
        logger.info(
            f"Vulnerability status updated",
            vulnerability_id=vulnerability_id,
            old_status=old_status.value,
            new_status=status.value
        )
        
        return True


class OWASPSecurityFramework:
    """Main OWASP security framework orchestrator."""
    
    def __init__(self):
        self.vulnerability_assessment = VulnerabilityAssessment()
        self.security_controls: Dict[OWASPRisk, List[str]] = {
            OWASPRisk.A01_BROKEN_ACCESS_CONTROL: [
                "Implement proper authentication and authorization",
                "Use role-based access control (RBAC)",
                "Validate permissions on server-side",
                "Implement principle of least privilege",
            ],
            OWASPRisk.A02_CRYPTOGRAPHIC_FAILURES: [
                "Use strong encryption algorithms",
                "Implement proper key management",
                "Encrypt data at rest and in transit",
                "Use secure random number generators",
            ],
            OWASPRisk.A03_INJECTION: [
                "Use parameterized queries",
                "Implement input validation",
                "Use ORM frameworks",
                "Apply principle of least privilege for database access",
            ],
            OWASPRisk.A04_INSECURE_DESIGN: [
                "Implement secure design patterns",
                "Use threat modeling",
                "Apply defense in depth",
                "Implement secure coding practices",
            ],
            OWASPRisk.A05_SECURITY_MISCONFIGURATION: [
                "Harden security configurations",
                "Remove default accounts and passwords",
                "Keep software updated",
                "Implement security headers",
            ],
        }
    
    async def run_security_assessment(self, project_path: str) -> SecurityScanResult:
        """Run comprehensive security assessment."""
        return await self.vulnerability_assessment.assess_project(project_path)
    
    def get_security_controls(self, owasp_risk: OWASPRisk) -> List[str]:
        """Get security controls for specific OWASP risk."""
        return self.security_controls.get(owasp_risk, [])
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get OWASP compliance status."""
        total_vulnerabilities = len(self.vulnerability_assessment.vulnerabilities)
        
        # Count vulnerabilities by OWASP category
        owasp_breakdown = {}
        for vulnerability in self.vulnerability_assessment.vulnerabilities.values():
            category = vulnerability.owasp_category.value
            owasp_breakdown[category] = owasp_breakdown.get(category, 0) + 1
        
        return {
            "total_vulnerabilities": total_vulnerabilities,
            "owasp_breakdown": owasp_breakdown,
            "compliance_score": max(0, 100 - (total_vulnerabilities * 5)),  # Simple scoring
            "last_assessment": datetime.now(timezone.utc).isoformat(),
        }


# Global instances
owasp_framework = OWASPSecurityFramework()
security_scanner = SecurityScanner()
vulnerability_assessment = VulnerabilityAssessment()