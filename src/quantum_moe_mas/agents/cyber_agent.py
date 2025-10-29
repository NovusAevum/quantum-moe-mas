"""
Cyber Agent - CEH v12 Compliant Security Analysis and OSINT

This module implements a specialized cybersecurity agent that provides
threat analysis, vulnerability scanning, OSINT capabilities, and security
recommendations following CEH v12 standards.

Author: Wan Mohamad Hanis bin Wan Hassan
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Tuple

from pydantic import BaseModel, Field, ConfigDict

from quantum_moe_mas.agents.base_agent import BaseAgent, AgentCapability, AgentMessage, MessageType
from quantum_moe_mas.core.logging_simple import get_logger


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityType(Enum):
    """Types of vulnerabilities."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INFORMATION_DISCLOSURE = "information_disclosure"
    DENIAL_OF_SERVICE = "denial_of_service"
    BUFFER_OVERFLOW = "buffer_overflow"
    WEAK_ENCRYPTION = "weak_encryption"
    MISCONFIGURATION = "misconfiguration"


class OSINTSource(Enum):
    """OSINT data sources."""
    WHOIS = "whois"
    DNS = "dns"
    SUBDOMAIN_ENUM = "subdomain_enum"
    PORT_SCAN = "port_scan"
    SSL_CERT = "ssl_cert"
    SOCIAL_MEDIA = "social_media"
    SEARCH_ENGINE = "search_engine"
    THREAT_INTEL = "threat_intel"


@dataclass
class ThreatData:
    """Threat data structure."""
    target: str
    threat_type: str
    indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Vulnerability:
    """Vulnerability information."""
    id: str
    type: VulnerabilityType
    severity: ThreatLevel
    title: str
    description: str
    affected_component: str
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    remediation: Optional[str] = None
    references: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OSINTResult:
    """OSINT collection result."""
    source: OSINTSource
    target: str
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ThreatAnalysis(BaseModel):
    """Threat analysis result."""
    target: str
    threat_level: ThreatLevel
    vulnerabilities: List[Vulnerability]
    osint_data: List[OSINTResult]
    risk_score: float = Field(ge=0.0, le=10.0)
    analysis_summary: str
    recommendations: List[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VulnerabilityReport(BaseModel):
    """Vulnerability scan report."""
    target: str
    scan_type: str
    vulnerabilities: List[Vulnerability]
    total_vulnerabilities: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    scan_duration: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OSINTReport(BaseModel):
    """OSINT collection report."""
    target: str
    sources_used: List[OSINTSource]
    results: List[OSINTResult]
    summary: Dict[str, Any]
    collection_time: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SecurityRecommendation(BaseModel):
    """Security recommendation."""
    id: str
    title: str
    description: str
    priority: ThreatLevel
    category: str
    implementation_steps: List[str]
    estimated_effort: str
    business_impact: str


class CyberAgent(BaseAgent):
    """
    Cybersecurity agent with CEH v12 compliance.
    
    Provides comprehensive security analysis including:
    - Threat intelligence and analysis
    - Vulnerability scanning and assessment
    - OSINT (Open Source Intelligence) collection
    - Security recommendations and remediation guidance
    - Incident response support
    """
    
    def __init__(self, agent_id: str = "cyber_agent", config: Optional[Dict[str, Any]] = None):
        """Initialize the Cyber Agent."""
        capabilities = [
            AgentCapability(
                name="threat_analysis",
                description="Analyze threats and assess risk levels",
                version="1.0.0"
            ),
            AgentCapability(
                name="vulnerability_scanning",
                description="Scan for security vulnerabilities",
                version="1.0.0"
            ),
            AgentCapability(
                name="osint_collection",
                description="Collect open source intelligence",
                version="1.0.0"
            ),
            AgentCapability(
                name="security_recommendations",
                description="Generate security recommendations",
                version="1.0.0"
            ),
            AgentCapability(
                name="incident_response",
                description="Support incident response activities",
                version="1.0.0"
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            name="Cyber Security Agent",
            description="CEH v12 compliant cybersecurity agent for threat analysis and OSINT",
            capabilities=capabilities,
            config=config or {}
        )
        
        # Security configuration
        self.max_scan_targets = config.get("max_scan_targets", 100) if config else 100
        self.scan_timeout = config.get("scan_timeout", 300) if config else 300  # 5 minutes
        self.osint_timeout = config.get("osint_timeout", 180) if config else 180  # 3 minutes
        
        # Threat intelligence feeds (would be configured with real feeds)
        self.threat_feeds = config.get("threat_feeds", []) if config else []
        
        # Security tools configuration
        self.tools_config = config.get("tools", {}) if config else {}
        
        # Known vulnerability patterns
        self.vuln_patterns = self._load_vulnerability_patterns()
        
        # OSINT sources configuration
        self.osint_sources = self._configure_osint_sources()
        
        self._logger = get_logger(f"agent.{agent_id}")
    
    async def _initialize_agent(self) -> None:
        """Initialize cyber agent specific components."""
        self._logger.info("Initializing Cyber Agent")
        
        # Initialize security tools
        await self._initialize_security_tools()
        
        # Load threat intelligence
        await self._load_threat_intelligence()
        
        # Setup message handlers
        self.register_message_handler(MessageType.TASK_REQUEST, self._handle_security_task)
        
        self._logger.info("Cyber Agent initialized successfully")
    
    async def _cleanup_agent(self) -> None:
        """Cleanup cyber agent resources."""
        self._logger.info("Cleaning up Cyber Agent resources")
        # Cleanup any open connections, temporary files, etc.    
async def _initialize_security_tools(self) -> None:
        """Initialize security scanning tools."""
        # This would initialize actual security tools like nmap, nikto, etc.
        # For now, we'll simulate tool availability
        self.available_tools = {
            "nmap": True,
            "nikto": True,
            "sqlmap": True,
            "dirb": True,
            "whois": True,
            "dig": True
        }
        
        self._logger.info("Security tools initialized", tools=list(self.available_tools.keys()))
    
    async def _load_threat_intelligence(self) -> None:
        """Load threat intelligence feeds."""
        # This would load from actual threat intelligence sources
        self.threat_indicators = {
            "malicious_ips": set(),
            "malicious_domains": set(),
            "malware_hashes": set(),
            "suspicious_patterns": []
        }
        
        self._logger.info("Threat intelligence loaded")
    
    def _load_vulnerability_patterns(self) -> Dict[str, List[str]]:
        """Load vulnerability detection patterns."""
        return {
            "sql_injection": [
                r"error in your SQL syntax",
                r"mysql_fetch_array",
                r"ORA-\d{5}",
                r"Microsoft OLE DB Provider",
                r"SQLServer JDBC Driver"
            ],
            "xss": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>"
            ],
            "directory_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c"
            ],
            "command_injection": [
                r";\s*(cat|ls|dir|type)\s",
                r"\|\s*(cat|ls|dir|type)\s",
                r"&&\s*(cat|ls|dir|type)\s"
            ]
        }
    
    def _configure_osint_sources(self) -> Dict[str, Dict[str, Any]]:
        """Configure OSINT data sources."""
        return {
            "whois": {
                "enabled": True,
                "timeout": 30,
                "rate_limit": 10  # requests per minute
            },
            "dns": {
                "enabled": True,
                "timeout": 10,
                "resolvers": ["8.8.8.8", "1.1.1.1"]
            },
            "subdomain_enum": {
                "enabled": True,
                "wordlist_size": 1000,
                "timeout": 300
            },
            "port_scan": {
                "enabled": True,
                "common_ports": [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995],
                "timeout": 60
            }
        }
    
    async def _process_task_impl(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process cybersecurity tasks."""
        task_type = task.get("type", "")
        
        if task_type == "threat_analysis":
            return await self._handle_threat_analysis(task, context)
        elif task_type == "vulnerability_scan":
            return await self._handle_vulnerability_scan(task, context)
        elif task_type == "osint_collection":
            return await self._handle_osint_collection(task, context)
        elif task_type == "security_assessment":
            return await self._handle_security_assessment(task, context)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    async def _handle_security_task(self, message: AgentMessage) -> None:
        """Handle security-related task messages."""
        try:
            task = message.payload.get("task", {})
            result = await self._process_task_impl(task)
            
            # Send response
            response = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                payload={"result": result},
                correlation_id=message.correlation_id
            )
            
            await self.send_message(response)
            
        except Exception as e:
            self._logger.error("Error handling security task", error=str(e))
            
            # Send error response
            error_response = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.ERROR_REPORT,
                payload={"error": str(e)},
                correlation_id=message.correlation_id
            )
            
            await self.send_message(error_response)
    
    async def analyze_threat(self, data: ThreatData) -> ThreatAnalysis:
        """
        Analyze threat data and assess risk level.
        
        Args:
            data: Threat data to analyze
            
        Returns:
            Comprehensive threat analysis
        """
        self._logger.info("Starting threat analysis", target=data.target)
        
        start_time = time.time()
        
        try:
            # Collect OSINT data
            osint_results = await self._collect_osint_data(data.target)
            
            # Perform vulnerability assessment
            vulnerabilities = await self._assess_vulnerabilities(data.target, data.indicators)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(vulnerabilities, osint_results)
            
            # Determine threat level
            threat_level = self._determine_threat_level(risk_score)
            
            # Generate analysis summary
            summary = self._generate_analysis_summary(
                data.target, threat_level, vulnerabilities, osint_results
            )
            
            # Generate recommendations
            recommendations = await self._generate_security_recommendations(
                vulnerabilities, osint_results
            )
            
            analysis = ThreatAnalysis(
                target=data.target,
                threat_level=threat_level,
                vulnerabilities=vulnerabilities,
                osint_data=osint_results,
                risk_score=risk_score,
                analysis_summary=summary,
                recommendations=recommendations
            )
            
            analysis_time = time.time() - start_time
            self._logger.info(
                "Threat analysis completed",
                target=data.target,
                threat_level=threat_level.value,
                risk_score=risk_score,
                analysis_time=analysis_time
            )
            
            return analysis
            
        except Exception as e:
            self._logger.error("Threat analysis failed", target=data.target, error=str(e))
            raise
    
    async def perform_osint(self, target: str) -> OSINTReport:
        """
        Perform OSINT collection on target.
        
        Args:
            target: Target for OSINT collection
            
        Returns:
            OSINT collection report
        """
        self._logger.info("Starting OSINT collection", target=target)
        
        start_time = time.time()
        sources_used = []
        results = []
        
        try:
            # WHOIS lookup
            if self.osint_sources["whois"]["enabled"]:
                whois_result = await self._perform_whois_lookup(target)
                if whois_result:
                    results.append(whois_result)
                    sources_used.append(OSINTSource.WHOIS)
            
            # DNS enumeration
            if self.osint_sources["dns"]["enabled"]:
                dns_results = await self._perform_dns_enumeration(target)
                results.extend(dns_results)
                if dns_results:
                    sources_used.append(OSINTSource.DNS)
            
            # Subdomain enumeration
            if self.osint_sources["subdomain_enum"]["enabled"]:
                subdomain_results = await self._perform_subdomain_enumeration(target)
                results.extend(subdomain_results)
                if subdomain_results:
                    sources_used.append(OSINTSource.SUBDOMAIN_ENUM)
            
            # Port scanning
            if self.osint_sources["port_scan"]["enabled"]:
                port_results = await self._perform_port_scan(target)
                if port_results:
                    results.append(port_results)
                    sources_used.append(OSINTSource.PORT_SCAN)
            
            # SSL certificate analysis
            ssl_result = await self._analyze_ssl_certificate(target)
            if ssl_result:
                results.append(ssl_result)
                sources_used.append(OSINTSource.SSL_CERT)
            
            # Generate summary
            summary = self._generate_osint_summary(results)
            
            collection_time = time.time() - start_time
            
            report = OSINTReport(
                target=target,
                sources_used=sources_used,
                results=results,
                summary=summary,
                collection_time=collection_time
            )
            
            self._logger.info(
                "OSINT collection completed",
                target=target,
                sources_count=len(sources_used),
                results_count=len(results),
                collection_time=collection_time
            )
            
            return report
            
        except Exception as e:
            self._logger.error("OSINT collection failed", target=target, error=str(e))
            raise
    
    async def scan_vulnerabilities(self, target: str, scan_type: str = "comprehensive") -> VulnerabilityReport:
        """
        Perform vulnerability scanning on target.
        
        Args:
            target: Target to scan
            scan_type: Type of scan (quick, comprehensive, targeted)
            
        Returns:
            Vulnerability scan report
        """
        self._logger.info("Starting vulnerability scan", target=target, scan_type=scan_type)
        
        start_time = time.time()
        vulnerabilities = []
        
        try:
            # Web application vulnerabilities
            web_vulns = await self._scan_web_vulnerabilities(target)
            vulnerabilities.extend(web_vulns)
            
            # Network vulnerabilities
            if scan_type in ["comprehensive", "network"]:
                network_vulns = await self._scan_network_vulnerabilities(target)
                vulnerabilities.extend(network_vulns)
            
            # SSL/TLS vulnerabilities
            ssl_vulns = await self._scan_ssl_vulnerabilities(target)
            vulnerabilities.extend(ssl_vulns)
            
            # Configuration vulnerabilities
            if scan_type == "comprehensive":
                config_vulns = await self._scan_configuration_vulnerabilities(target)
                vulnerabilities.extend(config_vulns)
            
            # Count vulnerabilities by severity
            severity_counts = self._count_vulnerabilities_by_severity(vulnerabilities)
            
            scan_duration = time.time() - start_time
            
            report = VulnerabilityReport(
                target=target,
                scan_type=scan_type,
                vulnerabilities=vulnerabilities,
                total_vulnerabilities=len(vulnerabilities),
                critical_count=severity_counts[ThreatLevel.CRITICAL],
                high_count=severity_counts[ThreatLevel.HIGH],
                medium_count=severity_counts[ThreatLevel.MEDIUM],
                low_count=severity_counts[ThreatLevel.LOW],
                scan_duration=scan_duration
            )
            
            self._logger.info(
                "Vulnerability scan completed",
                target=target,
                total_vulnerabilities=len(vulnerabilities),
                critical=severity_counts[ThreatLevel.CRITICAL],
                high=severity_counts[ThreatLevel.HIGH],
                scan_duration=scan_duration
            )
            
            return report
            
        except Exception as e:
            self._logger.error("Vulnerability scan failed", target=target, error=str(e))
            raise
    
    async def generate_security_recommendations(
        self,
        analysis: ThreatAnalysis
    ) -> List[SecurityRecommendation]:
        """
        Generate security recommendations based on threat analysis.
        
        Args:
            analysis: Threat analysis results
            
        Returns:
            List of security recommendations
        """
        recommendations = []
        
        # Vulnerability-based recommendations
        for vuln in analysis.vulnerabilities:
            rec = self._create_vulnerability_recommendation(vuln)
            if rec:
                recommendations.append(rec)
        
        # OSINT-based recommendations
        for osint_result in analysis.osint_data:
            rec = self._create_osint_recommendation(osint_result)
            if rec:
                recommendations.append(rec)
        
        # General security recommendations based on threat level
        general_recs = self._create_general_recommendations(analysis.threat_level)
        recommendations.extend(general_recs)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority.value, reverse=True)
        
        return recommendations 
   
    # OSINT Collection Methods
    
    async def _collect_osint_data(self, target: str) -> List[OSINTResult]:
        """Collect OSINT data from multiple sources."""
        results = []
        
        # Perform OSINT collection
        osint_report = await self.perform_osint(target)
        results.extend(osint_report.results)
        
        return results
    
    async def _perform_whois_lookup(self, target: str) -> Optional[OSINTResult]:
        """Perform WHOIS lookup."""
        try:
            # Simulate WHOIS lookup (would use actual WHOIS service)
            whois_data = {
                "domain": target,
                "registrar": "Example Registrar",
                "creation_date": "2020-01-01",
                "expiration_date": "2025-01-01",
                "name_servers": ["ns1.example.com", "ns2.example.com"],
                "registrant_org": "Example Organization"
            }
            
            return OSINTResult(
                source=OSINTSource.WHOIS,
                target=target,
                data=whois_data,
                confidence=0.9
            )
            
        except Exception as e:
            self._logger.error("WHOIS lookup failed", target=target, error=str(e))
            return None
    
    async def _perform_dns_enumeration(self, target: str) -> List[OSINTResult]:
        """Perform DNS enumeration."""
        results = []
        
        try:
            # DNS record types to query
            record_types = ["A", "AAAA", "MX", "NS", "TXT", "CNAME"]
            
            for record_type in record_types:
                try:
                    # Simulate DNS query (would use actual DNS resolver)
                    dns_data = {
                        "record_type": record_type,
                        "records": [f"{record_type.lower()}.{target}"]
                    }
                    
                    result = OSINTResult(
                        source=OSINTSource.DNS,
                        target=target,
                        data=dns_data,
                        confidence=0.95
                    )
                    results.append(result)
                    
                except Exception:
                    continue
            
        except Exception as e:
            self._logger.error("DNS enumeration failed", target=target, error=str(e))
        
        return results
    
    async def _perform_subdomain_enumeration(self, target: str) -> List[OSINTResult]:
        """Perform subdomain enumeration."""
        results = []
        
        try:
            # Common subdomain prefixes
            subdomains = ["www", "mail", "ftp", "admin", "api", "dev", "test", "staging"]
            
            for subdomain in subdomains:
                full_domain = f"{subdomain}.{target}"
                
                # Simulate subdomain check (would perform actual DNS resolution)
                if hash(full_domain) % 3 == 0:  # Simulate some subdomains existing
                    subdomain_data = {
                        "subdomain": full_domain,
                        "ip_address": "192.168.1.1",
                        "status": "active"
                    }
                    
                    result = OSINTResult(
                        source=OSINTSource.SUBDOMAIN_ENUM,
                        target=full_domain,
                        data=subdomain_data,
                        confidence=0.8
                    )
                    results.append(result)
            
        except Exception as e:
            self._logger.error("Subdomain enumeration failed", target=target, error=str(e))
        
        return results
    
    async def _perform_port_scan(self, target: str) -> Optional[OSINTResult]:
        """Perform port scanning."""
        try:
            common_ports = self.osint_sources["port_scan"]["common_ports"]
            open_ports = []
            
            # Simulate port scanning (would use actual port scanning)
            for port in common_ports:
                if hash(f"{target}:{port}") % 4 == 0:  # Simulate some ports being open
                    open_ports.append({
                        "port": port,
                        "service": self._get_service_name(port),
                        "state": "open"
                    })
            
            port_data = {
                "target": target,
                "open_ports": open_ports,
                "total_scanned": len(common_ports)
            }
            
            return OSINTResult(
                source=OSINTSource.PORT_SCAN,
                target=target,
                data=port_data,
                confidence=0.9
            )
            
        except Exception as e:
            self._logger.error("Port scan failed", target=target, error=str(e))
            return None
    
    async def _analyze_ssl_certificate(self, target: str) -> Optional[OSINTResult]:
        """Analyze SSL certificate."""
        try:
            # Simulate SSL certificate analysis
            ssl_data = {
                "subject": f"CN={target}",
                "issuer": "Let's Encrypt Authority X3",
                "valid_from": "2024-01-01",
                "valid_to": "2024-12-31",
                "signature_algorithm": "SHA256withRSA",
                "key_size": 2048,
                "san_domains": [target, f"www.{target}"]
            }
            
            return OSINTResult(
                source=OSINTSource.SSL_CERT,
                target=target,
                data=ssl_data,
                confidence=0.95
            )
            
        except Exception as e:
            self._logger.error("SSL certificate analysis failed", target=target, error=str(e))
            return None
    
    # Vulnerability Scanning Methods
    
    async def _assess_vulnerabilities(self, target: str, indicators: List[str]) -> List[Vulnerability]:
        """Assess vulnerabilities based on target and indicators."""
        vulnerabilities = []
        
        # Perform comprehensive vulnerability scan
        vuln_report = await self.scan_vulnerabilities(target)
        vulnerabilities.extend(vuln_report.vulnerabilities)
        
        return vulnerabilities
    
    async def _scan_web_vulnerabilities(self, target: str) -> List[Vulnerability]:
        """Scan for web application vulnerabilities."""
        vulnerabilities = []
        
        # SQL Injection testing
        sql_vulns = await self._test_sql_injection(target)
        vulnerabilities.extend(sql_vulns)
        
        # XSS testing
        xss_vulns = await self._test_xss(target)
        vulnerabilities.extend(xss_vulns)
        
        # Directory traversal testing
        dir_vulns = await self._test_directory_traversal(target)
        vulnerabilities.extend(dir_vulns)
        
        return vulnerabilities
    
    async def _scan_network_vulnerabilities(self, target: str) -> List[Vulnerability]:
        """Scan for network vulnerabilities."""
        vulnerabilities = []
        
        # Simulate network vulnerability scanning
        # This would integrate with actual network scanning tools
        
        return vulnerabilities
    
    async def _scan_ssl_vulnerabilities(self, target: str) -> List[Vulnerability]:
        """Scan for SSL/TLS vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Simulate SSL vulnerability testing
            ssl_issues = [
                {
                    "type": VulnerabilityType.WEAK_ENCRYPTION,
                    "severity": ThreatLevel.MEDIUM,
                    "title": "Weak SSL/TLS Configuration",
                    "description": "Server supports weak cipher suites"
                }
            ]
            
            for issue in ssl_issues:
                vuln = Vulnerability(
                    id=f"ssl_{hash(target)}_{issue['type'].value}",
                    type=issue["type"],
                    severity=issue["severity"],
                    title=issue["title"],
                    description=issue["description"],
                    affected_component=f"SSL/TLS on {target}",
                    remediation="Update SSL/TLS configuration to use strong cipher suites"
                )
                vulnerabilities.append(vuln)
            
        except Exception as e:
            self._logger.error("SSL vulnerability scan failed", target=target, error=str(e))
        
        return vulnerabilities
    
    async def _scan_configuration_vulnerabilities(self, target: str) -> List[Vulnerability]:
        """Scan for configuration vulnerabilities."""
        vulnerabilities = []
        
        # Simulate configuration vulnerability scanning
        # This would check for misconfigurations, default credentials, etc.
        
        return vulnerabilities
    
    async def _test_sql_injection(self, target: str) -> List[Vulnerability]:
        """Test for SQL injection vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Simulate SQL injection testing
            # This would perform actual SQL injection tests
            
            # Example vulnerability found
            if hash(target) % 5 == 0:  # Simulate finding vulnerability sometimes
                vuln = Vulnerability(
                    id=f"sqli_{hash(target)}",
                    type=VulnerabilityType.SQL_INJECTION,
                    severity=ThreatLevel.HIGH,
                    title="SQL Injection Vulnerability",
                    description="Application is vulnerable to SQL injection attacks",
                    affected_component=f"Web application on {target}",
                    cvss_score=8.5,
                    remediation="Use parameterized queries and input validation"
                )
                vulnerabilities.append(vuln)
            
        except Exception as e:
            self._logger.error("SQL injection test failed", target=target, error=str(e))
        
        return vulnerabilities
    
    async def _test_xss(self, target: str) -> List[Vulnerability]:
        """Test for XSS vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Simulate XSS testing
            if hash(target) % 7 == 0:  # Simulate finding vulnerability sometimes
                vuln = Vulnerability(
                    id=f"xss_{hash(target)}",
                    type=VulnerabilityType.XSS,
                    severity=ThreatLevel.MEDIUM,
                    title="Cross-Site Scripting (XSS) Vulnerability",
                    description="Application is vulnerable to XSS attacks",
                    affected_component=f"Web application on {target}",
                    cvss_score=6.1,
                    remediation="Implement proper input validation and output encoding"
                )
                vulnerabilities.append(vuln)
            
        except Exception as e:
            self._logger.error("XSS test failed", target=target, error=str(e))
        
        return vulnerabilities
    
    async def _test_directory_traversal(self, target: str) -> List[Vulnerability]:
        """Test for directory traversal vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Simulate directory traversal testing
            if hash(target) % 11 == 0:  # Simulate finding vulnerability sometimes
                vuln = Vulnerability(
                    id=f"dirtraversal_{hash(target)}",
                    type=VulnerabilityType.INFORMATION_DISCLOSURE,
                    severity=ThreatLevel.MEDIUM,
                    title="Directory Traversal Vulnerability",
                    description="Application allows directory traversal attacks",
                    affected_component=f"Web application on {target}",
                    cvss_score=5.3,
                    remediation="Implement proper path validation and access controls"
                )
                vulnerabilities.append(vuln)
            
        except Exception as e:
            self._logger.error("Directory traversal test failed", target=target, error=str(e))
        
        return vulnerabilities 
   
    # Analysis and Recommendation Methods
    
    def _calculate_risk_score(
        self,
        vulnerabilities: List[Vulnerability],
        osint_results: List[OSINTResult]
    ) -> float:
        """Calculate overall risk score."""
        if not vulnerabilities:
            return 0.0
        
        # Base score from vulnerabilities
        vuln_score = 0.0
        for vuln in vulnerabilities:
            if vuln.cvss_score:
                vuln_score += vuln.cvss_score
            else:
                # Default scores by severity
                severity_scores = {
                    ThreatLevel.CRITICAL: 9.0,
                    ThreatLevel.HIGH: 7.0,
                    ThreatLevel.MEDIUM: 5.0,
                    ThreatLevel.LOW: 2.0
                }
                vuln_score += severity_scores.get(vuln.severity, 0.0)
        
        # Average vulnerability score
        avg_vuln_score = vuln_score / len(vulnerabilities)
        
        # Adjust based on OSINT findings
        osint_multiplier = 1.0
        for result in osint_results:
            if result.source == OSINTSource.PORT_SCAN:
                open_ports = len(result.data.get("open_ports", []))
                if open_ports > 5:
                    osint_multiplier += 0.1
            elif result.source == OSINTSource.SUBDOMAIN_ENUM:
                if result.data.get("status") == "active":
                    osint_multiplier += 0.05
        
        # Calculate final risk score (0-10 scale)
        risk_score = min(avg_vuln_score * osint_multiplier, 10.0)
        
        return round(risk_score, 2)
    
    def _determine_threat_level(self, risk_score: float) -> ThreatLevel:
        """Determine threat level based on risk score."""
        if risk_score >= 8.0:
            return ThreatLevel.CRITICAL
        elif risk_score >= 6.0:
            return ThreatLevel.HIGH
        elif risk_score >= 3.0:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _generate_analysis_summary(
        self,
        target: str,
        threat_level: ThreatLevel,
        vulnerabilities: List[Vulnerability],
        osint_results: List[OSINTResult]
    ) -> str:
        """Generate threat analysis summary."""
        vuln_count = len(vulnerabilities)
        osint_count = len(osint_results)
        
        critical_vulns = sum(1 for v in vulnerabilities if v.severity == ThreatLevel.CRITICAL)
        high_vulns = sum(1 for v in vulnerabilities if v.severity == ThreatLevel.HIGH)
        
        summary = f"""
        Threat Analysis Summary for {target}:
        
        Overall Threat Level: {threat_level.value.upper()}
        
        Vulnerabilities Found: {vuln_count}
        - Critical: {critical_vulns}
        - High: {high_vulns}
        - Medium: {vuln_count - critical_vulns - high_vulns}
        
        OSINT Data Points: {osint_count}
        
        Key Findings:
        """
        
        # Add key vulnerability findings
        if critical_vulns > 0:
            summary += f"\n- {critical_vulns} critical vulnerabilities require immediate attention"
        
        if high_vulns > 0:
            summary += f"\n- {high_vulns} high-severity vulnerabilities found"
        
        # Add OSINT findings
        for result in osint_results[:3]:  # Top 3 findings
            if result.source == OSINTSource.PORT_SCAN:
                open_ports = len(result.data.get("open_ports", []))
                summary += f"\n- {open_ports} open ports detected"
            elif result.source == OSINTSource.SUBDOMAIN_ENUM:
                summary += f"\n- Active subdomain found: {result.target}"
        
        return summary.strip()
    
    async def _generate_security_recommendations(
        self,
        vulnerabilities: List[Vulnerability],
        osint_results: List[OSINTResult]
    ) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        # Vulnerability-based recommendations
        critical_vulns = [v for v in vulnerabilities if v.severity == ThreatLevel.CRITICAL]
        if critical_vulns:
            recommendations.append(
                f"URGENT: Address {len(critical_vulns)} critical vulnerabilities immediately"
            )
        
        high_vulns = [v for v in vulnerabilities if v.severity == ThreatLevel.HIGH]
        if high_vulns:
            recommendations.append(
                f"High Priority: Remediate {len(high_vulns)} high-severity vulnerabilities"
            )
        
        # OSINT-based recommendations
        for result in osint_results:
            if result.source == OSINTSource.PORT_SCAN:
                open_ports = result.data.get("open_ports", [])
                if len(open_ports) > 10:
                    recommendations.append(
                        "Consider reducing attack surface by closing unnecessary ports"
                    )
            elif result.source == OSINTSource.SSL_CERT:
                cert_data = result.data
                if "valid_to" in cert_data:
                    # Check if certificate expires soon (simplified)
                    recommendations.append("Monitor SSL certificate expiration dates")
        
        # General recommendations
        recommendations.extend([
            "Implement regular security assessments and penetration testing",
            "Establish incident response procedures",
            "Deploy web application firewall (WAF) for additional protection",
            "Implement security monitoring and logging",
            "Conduct security awareness training for staff"
        ])
        
        return recommendations
    
    def _create_vulnerability_recommendation(self, vuln: Vulnerability) -> Optional[SecurityRecommendation]:
        """Create recommendation based on vulnerability."""
        if not vuln.remediation:
            return None
        
        return SecurityRecommendation(
            id=f"rec_{vuln.id}",
            title=f"Fix {vuln.title}",
            description=vuln.remediation,
            priority=vuln.severity,
            category="Vulnerability Remediation",
            implementation_steps=[
                "Analyze the vulnerability details",
                "Test the fix in a development environment",
                "Apply the remediation",
                "Verify the fix effectiveness"
            ],
            estimated_effort="2-8 hours depending on complexity",
            business_impact="Reduces security risk and potential data breach"
        )
    
    def _create_osint_recommendation(self, osint_result: OSINTResult) -> Optional[SecurityRecommendation]:
        """Create recommendation based on OSINT findings."""
        if osint_result.source == OSINTSource.PORT_SCAN:
            open_ports = osint_result.data.get("open_ports", [])
            if len(open_ports) > 5:
                return SecurityRecommendation(
                    id=f"rec_ports_{hash(osint_result.target)}",
                    title="Reduce Network Attack Surface",
                    description="Multiple open ports detected. Consider closing unnecessary services.",
                    priority=ThreatLevel.MEDIUM,
                    category="Network Security",
                    implementation_steps=[
                        "Audit all running services",
                        "Identify unnecessary services",
                        "Close unused ports",
                        "Implement firewall rules"
                    ],
                    estimated_effort="4-6 hours",
                    business_impact="Reduces potential attack vectors"
                )
        
        return None
    
    def _create_general_recommendations(self, threat_level: ThreatLevel) -> List[SecurityRecommendation]:
        """Create general security recommendations based on threat level."""
        recommendations = []
        
        if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            recommendations.append(
                SecurityRecommendation(
                    id="rec_incident_response",
                    title="Activate Incident Response Plan",
                    description="High threat level detected. Activate incident response procedures.",
                    priority=ThreatLevel.CRITICAL,
                    category="Incident Response",
                    implementation_steps=[
                        "Notify security team",
                        "Assess immediate risks",
                        "Implement containment measures",
                        "Begin remediation process"
                    ],
                    estimated_effort="Immediate action required",
                    business_impact="Prevents potential security incidents"
                )
            )
        
        return recommendations
    
    # Utility Methods
    
    def _get_service_name(self, port: int) -> str:
        """Get service name for port number."""
        service_map = {
            21: "FTP",
            22: "SSH",
            23: "Telnet",
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            110: "POP3",
            143: "IMAP",
            443: "HTTPS",
            993: "IMAPS",
            995: "POP3S"
        }
        return service_map.get(port, "Unknown")
    
    def _count_vulnerabilities_by_severity(self, vulnerabilities: List[Vulnerability]) -> Dict[ThreatLevel, int]:
        """Count vulnerabilities by severity level."""
        counts = {level: 0 for level in ThreatLevel}
        
        for vuln in vulnerabilities:
            counts[vuln.severity] += 1
        
        return counts
    
    def _generate_osint_summary(self, results: List[OSINTResult]) -> Dict[str, Any]:
        """Generate OSINT collection summary."""
        summary = {
            "total_results": len(results),
            "sources_used": list(set(result.source.value for result in results)),
            "confidence_avg": sum(result.confidence for result in results) / len(results) if results else 0,
            "key_findings": []
        }
        
        # Extract key findings
        for result in results:
            if result.source == OSINTSource.WHOIS:
                summary["key_findings"].append(f"Domain registered by {result.data.get('registrant_org', 'Unknown')}")
            elif result.source == OSINTSource.PORT_SCAN:
                open_ports = len(result.data.get("open_ports", []))
                summary["key_findings"].append(f"{open_ports} open ports detected")
        
        return summary
    
    # Task Handler Methods
    
    async def _handle_threat_analysis(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle threat analysis task."""
        target = task.get("target", "")
        threat_type = task.get("threat_type", "general")
        indicators = task.get("indicators", [])
        
        if not target:
            raise ValueError("Target is required for threat analysis")
        
        threat_data = ThreatData(
            target=target,
            threat_type=threat_type,
            indicators=indicators,
            metadata=task.get("metadata", {})
        )
        
        analysis = await self.analyze_threat(threat_data)
        
        return {
            "analysis": analysis.model_dump(),
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _handle_vulnerability_scan(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle vulnerability scan task."""
        target = task.get("target", "")
        scan_type = task.get("scan_type", "comprehensive")
        
        if not target:
            raise ValueError("Target is required for vulnerability scan")
        
        report = await self.scan_vulnerabilities(target, scan_type)
        
        return {
            "report": report.model_dump(),
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _handle_osint_collection(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle OSINT collection task."""
        target = task.get("target", "")
        
        if not target:
            raise ValueError("Target is required for OSINT collection")
        
        report = await self.perform_osint(target)
        
        return {
            "report": report.model_dump(),
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _handle_security_assessment(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle comprehensive security assessment task."""
        target = task.get("target", "")
        
        if not target:
            raise ValueError("Target is required for security assessment")
        
        # Perform comprehensive assessment
        threat_data = ThreatData(target=target, threat_type="comprehensive")
        analysis = await self.analyze_threat(threat_data)
        
        # Generate recommendations
        recommendations = await self.generate_security_recommendations(analysis)
        
        return {
            "analysis": analysis.model_dump(),
            "recommendations": [rec.model_dump() for rec in recommendations],
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }