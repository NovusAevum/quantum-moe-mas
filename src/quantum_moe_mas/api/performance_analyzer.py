"""
API Performance Analysis and Optimization System.

This module provides comprehensive performance analysis, comparison,
and optimization recommendations for all integrated API providers.
"""

import asyncio
import json
import os
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3

from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


class PerformanceMetric(Enum):
    """Types of performance metrics."""
    
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    COST_EFFICIENCY = "cost_efficiency"
    TOKEN_EFFICIENCY = "token_efficiency"
    AVAILABILITY = "availability"


class OptimizationPriority(Enum):
    """Optimization priority levels."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PerformanceSnapshot:
    """Performance snapshot for a provider at a point in time."""
    
    provider: str
    timestamp: datetime
    
    # Core metrics
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    success_rate: float
    error_rate: float
    
    # Efficiency metrics
    cost_per_request: float
    cost_per_token: float
    tokens_per_request: float
    
    # Quality metrics
    availability: float
    reliability_score: float
    
    # Sample size
    total_requests: int
    time_window_minutes: int


@dataclass
class PerformanceComparison:
    """Comparison between multiple providers."""
    
    comparison_id: str
    providers: List[str]
    metric: PerformanceMetric
    time_period: timedelta
    
    # Results
    rankings: List[Tuple[str, float]]  # (provider, score)
    best_provider: str
    worst_provider: str
    
    # Statistical analysis
    mean_value: float
    median_value: float
    std_deviation: float
    
    # Recommendations
    recommendations: List[str]
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    
    recommendation_id: str
    provider: str
    priority: OptimizationPriority
    
    # Issue identification
    issue_type: str
    issue_description: str
    impact_assessment: str
    
    # Solution
    recommendation_title: str
    recommendation_description: str
    implementation_steps: List[str]
    
    # Expected outcomes
    expected_improvement: Dict[str, float]  # metric -> improvement %
    implementation_effort: str  # "low", "medium", "high"
    risk_level: str  # "low", "medium", "high"
    
    # Supporting data
    supporting_metrics: Dict[str, Any]
    confidence_score: float  # 0-100
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"  # "pending", "implemented", "dismissed"


@dataclass
class PerformanceTrend:
    """Performance trend analysis."""
    
    provider: str
    metric: PerformanceMetric
    time_period: timedelta
    
    # Trend data
    data_points: List[Tuple[datetime, float]]
    trend_direction: str  # "improving", "degrading", "stable"
    trend_strength: float  # 0-1
    
    # Statistical analysis
    slope: float
    r_squared: float
    
    # Predictions
    predicted_next_value: Optional[float]
    prediction_confidence: float
    
    # Anomalies
    anomalies: List[Tuple[datetime, float, str]]  # (timestamp, value, reason)


class PerformanceAnalyzer:
    """
    Comprehensive API performance analysis and optimization system.
    
    Provides:
    - Real-time performance monitoring and analysis
    - Cross-provider performance comparisons
    - Trend analysis and anomaly detection
    - Optimization recommendations
    - Performance forecasting
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        analysis_interval_minutes: int = 15,
        trend_analysis_days: int = 7
    ) -> None:
        """
        Initialize performance analyzer.
        
        Args:
            storage_path: Path to SQLite database
            analysis_interval_minutes: Interval for performance analysis
            trend_analysis_days: Days of data for trend analysis
        """
        self.storage_path = storage_path or os.path.expanduser("~/.quantum_moe_mas/performance_analysis.db")
        self.analysis_interval = analysis_interval_minutes
        self.trend_analysis_days = trend_analysis_days
        
        # Performance data cache
        self.performance_cache: Dict[str, List[PerformanceSnapshot]] = {}
        self.comparison_cache: Dict[str, PerformanceComparison] = {}
        
        # Background tasks
        self.is_running = False
        self._analysis_task: Optional[asyncio.Task] = None
        
        logger.info(
            "Initialized PerformanceAnalyzer",
            storage_path=self.storage_path,
            analysis_interval=analysis_interval_minutes
        )
    
    async def initialize(self) -> None:
        """Initialize the performance analyzer and database."""
        try:
            # Create storage directory
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Initialize database
            await self._initialize_database()
            
            # Load recent performance data
            await self._load_recent_data()
            
            # Start background analysis
            self._analysis_task = asyncio.create_task(self._analysis_loop())
            
            self.is_running = True
            logger.info("PerformanceAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PerformanceAnalyzer: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the performance analyzer."""
        self.is_running = False
        
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        
        logger.info("PerformanceAnalyzer shutdown complete")
    
    async def _initialize_database(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            # Performance snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    avg_latency_ms REAL NOT NULL,
                    p95_latency_ms REAL NOT NULL,
                    p99_latency_ms REAL NOT NULL,
                    throughput_rps REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    error_rate REAL NOT NULL,
                    cost_per_request REAL NOT NULL,
                    cost_per_token REAL NOT NULL,
                    tokens_per_request REAL NOT NULL,
                    availability REAL NOT NULL,
                    reliability_score REAL NOT NULL,
                    total_requests INTEGER NOT NULL,
                    time_window_minutes INTEGER NOT NULL
                )
            """)
            
            # Performance comparisons table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    providers TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    time_period_hours INTEGER NOT NULL,
                    rankings TEXT NOT NULL,
                    best_provider TEXT NOT NULL,
                    worst_provider TEXT NOT NULL,
                    mean_value REAL NOT NULL,
                    median_value REAL NOT NULL,
                    std_deviation REAL NOT NULL,
                    recommendations TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Optimization recommendations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_recommendations (
                    recommendation_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    issue_type TEXT NOT NULL,
                    issue_description TEXT NOT NULL,
                    impact_assessment TEXT NOT NULL,
                    recommendation_title TEXT NOT NULL,
                    recommendation_description TEXT NOT NULL,
                    implementation_steps TEXT NOT NULL,
                    expected_improvement TEXT NOT NULL,
                    implementation_effort TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    supporting_metrics TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL
                )
            """)
            
            # Performance trends table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    time_period_hours INTEGER NOT NULL,
                    trend_direction TEXT NOT NULL,
                    trend_strength REAL NOT NULL,
                    slope REAL NOT NULL,
                    r_squared REAL NOT NULL,
                    predicted_next_value REAL,
                    prediction_confidence REAL NOT NULL,
                    anomalies TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_provider_time ON performance_snapshots(provider, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_comparisons_metric ON performance_comparisons(metric)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_provider ON optimization_recommendations(provider)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trends_provider_metric ON performance_trends(provider, metric)")
            
            conn.commit()
            logger.info("Performance analysis database schema initialized")
            
        finally:
            conn.close()
    
    async def record_performance_snapshot(
        self,
        provider: str,
        usage_data: Dict[str, Any],
        time_window_minutes: int = 15
    ) -> PerformanceSnapshot:
        """
        Record a performance snapshot for a provider.
        
        Args:
            provider: API provider name
            usage_data: Usage data from usage tracker
            time_window_minutes: Time window for the snapshot
        
        Returns:
            PerformanceSnapshot instance
        """
        # Extract metrics from usage data
        total_requests = usage_data.get("total_requests", 0)
        successful_requests = usage_data.get("successful_requests", 0)
        failed_requests = usage_data.get("failed_requests", 0)
        total_latency = usage_data.get("total_latency_ms", 0.0)
        total_cost = usage_data.get("total_cost", 0.0)
        total_tokens = usage_data.get("total_tokens", 0)
        
        # Calculate metrics
        if total_requests > 0:
            avg_latency_ms = total_latency / total_requests
            success_rate = (successful_requests / total_requests) * 100
            error_rate = (failed_requests / total_requests) * 100
            cost_per_request = total_cost / total_requests
            tokens_per_request = total_tokens / total_requests
            cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0.0
        else:
            avg_latency_ms = 0.0
            success_rate = 0.0
            error_rate = 0.0
            cost_per_request = 0.0
            tokens_per_request = 0.0
            cost_per_token = 0.0
        
        # Calculate throughput (requests per second)
        throughput_rps = total_requests / (time_window_minutes * 60) if time_window_minutes > 0 else 0.0
        
        # Calculate percentiles (simplified - would need actual latency distribution)
        p95_latency_ms = avg_latency_ms * 1.5  # Approximation
        p99_latency_ms = avg_latency_ms * 2.0  # Approximation
        
        # Calculate availability and reliability
        availability = success_rate  # Simplified
        reliability_score = min(100, success_rate + (100 - avg_latency_ms / 10))  # Simplified scoring
        
        snapshot = PerformanceSnapshot(
            provider=provider,
            timestamp=datetime.utcnow(),
            avg_latency_ms=avg_latency_ms,
            p95_latency_ms=p95_latency_ms,
            p99_latency_ms=p99_latency_ms,
            throughput_rps=throughput_rps,
            success_rate=success_rate,
            error_rate=error_rate,
            cost_per_request=cost_per_request,
            cost_per_token=cost_per_token,
            tokens_per_request=tokens_per_request,
            availability=availability,
            reliability_score=reliability_score,
            total_requests=total_requests,
            time_window_minutes=time_window_minutes
        )
        
        # Save to database
        await self._save_performance_snapshot(snapshot)
        
        # Add to cache
        if provider not in self.performance_cache:
            self.performance_cache[provider] = []
        
        self.performance_cache[provider].append(snapshot)
        
        # Keep only recent snapshots in cache
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.performance_cache[provider] = [
            s for s in self.performance_cache[provider]
            if s.timestamp > cutoff_time
        ]
        
        logger.debug(f"Recorded performance snapshot for {provider}")
        return snapshot
    
    async def _save_performance_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Save performance snapshot to database."""
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO performance_snapshots (
                    provider, timestamp, avg_latency_ms, p95_latency_ms, p99_latency_ms,
                    throughput_rps, success_rate, error_rate, cost_per_request,
                    cost_per_token, tokens_per_request, availability, reliability_score,
                    total_requests, time_window_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.provider, snapshot.timestamp.isoformat(),
                snapshot.avg_latency_ms, snapshot.p95_latency_ms, snapshot.p99_latency_ms,
                snapshot.throughput_rps, snapshot.success_rate, snapshot.error_rate,
                snapshot.cost_per_request, snapshot.cost_per_token, snapshot.tokens_per_request,
                snapshot.availability, snapshot.reliability_score,
                snapshot.total_requests, snapshot.time_window_minutes
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def compare_providers(
        self,
        providers: List[str],
        metric: PerformanceMetric,
        time_period: timedelta = timedelta(hours=24)
    ) -> PerformanceComparison:
        """
        Compare performance between multiple providers.
        
        Args:
            providers: List of provider names to compare
            metric: Performance metric to compare
            time_period: Time period for comparison
        
        Returns:
            PerformanceComparison instance
        """
        comparison_id = f"{'-'.join(providers)}_{metric.value}_{int(time_period.total_seconds())}"
        
        # Get performance data for each provider
        provider_values = {}
        cutoff_time = datetime.utcnow() - time_period
        
        for provider in providers:
            snapshots = await self._get_provider_snapshots(provider, cutoff_time)
            
            if not snapshots:
                provider_values[provider] = 0.0
                continue
            
            # Extract metric values
            values = []
            for snapshot in snapshots:
                if metric == PerformanceMetric.LATENCY:
                    values.append(snapshot.avg_latency_ms)
                elif metric == PerformanceMetric.THROUGHPUT:
                    values.append(snapshot.throughput_rps)
                elif metric == PerformanceMetric.SUCCESS_RATE:
                    values.append(snapshot.success_rate)
                elif metric == PerformanceMetric.ERROR_RATE:
                    values.append(snapshot.error_rate)
                elif metric == PerformanceMetric.COST_EFFICIENCY:
                    values.append(1.0 / snapshot.cost_per_request if snapshot.cost_per_request > 0 else 0.0)
                elif metric == PerformanceMetric.TOKEN_EFFICIENCY:
                    values.append(1.0 / snapshot.cost_per_token if snapshot.cost_per_token > 0 else 0.0)
                elif metric == PerformanceMetric.AVAILABILITY:
                    values.append(snapshot.availability)
            
            # Calculate average for this provider
            provider_values[provider] = statistics.mean(values) if values else 0.0
        
        # Rank providers (higher is better for most metrics, except latency and error rate)
        reverse_sort = metric not in [PerformanceMetric.LATENCY, PerformanceMetric.ERROR_RATE]
        rankings = sorted(provider_values.items(), key=lambda x: x[1], reverse=reverse_sort)
        
        # Statistical analysis
        all_values = list(provider_values.values())
        mean_value = statistics.mean(all_values) if all_values else 0.0
        median_value = statistics.median(all_values) if all_values else 0.0
        std_deviation = statistics.stdev(all_values) if len(all_values) > 1 else 0.0
        
        # Generate recommendations
        recommendations = self._generate_comparison_recommendations(rankings, metric)
        
        comparison = PerformanceComparison(
            comparison_id=comparison_id,
            providers=providers,
            metric=metric,
            time_period=time_period,
            rankings=rankings,
            best_provider=rankings[0][0] if rankings else "",
            worst_provider=rankings[-1][0] if rankings else "",
            mean_value=mean_value,
            median_value=median_value,
            std_deviation=std_deviation,
            recommendations=recommendations
        )
        
        # Save to database
        await self._save_performance_comparison(comparison)
        
        # Cache the comparison
        self.comparison_cache[comparison_id] = comparison
        
        logger.info(f"Compared {len(providers)} providers on {metric.value}: best={comparison.best_provider}")
        return comparison
    
    async def _get_provider_snapshots(
        self,
        provider: str,
        since_time: datetime
    ) -> List[PerformanceSnapshot]:
        """Get performance snapshots for a provider since a given time."""
        # First check cache
        if provider in self.performance_cache:
            cached_snapshots = [
                s for s in self.performance_cache[provider]
                if s.timestamp >= since_time
            ]
            if cached_snapshots:
                return cached_snapshots
        
        # Load from database
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM performance_snapshots
                WHERE provider = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (provider, since_time.isoformat()))
            
            rows = cursor.fetchall()
            snapshots = []
            
            for row in rows:
                snapshot = PerformanceSnapshot(
                    provider=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    avg_latency_ms=row[3],
                    p95_latency_ms=row[4],
                    p99_latency_ms=row[5],
                    throughput_rps=row[6],
                    success_rate=row[7],
                    error_rate=row[8],
                    cost_per_request=row[9],
                    cost_per_token=row[10],
                    tokens_per_request=row[11],
                    availability=row[12],
                    reliability_score=row[13],
                    total_requests=row[14],
                    time_window_minutes=row[15]
                )
                snapshots.append(snapshot)
            
            return snapshots
            
        finally:
            conn.close()
    
    def _generate_comparison_recommendations(
        self,
        rankings: List[Tuple[str, float]],
        metric: PerformanceMetric
    ) -> List[str]:
        """Generate recommendations based on provider comparison."""
        recommendations = []
        
        if not rankings:
            return recommendations
        
        best_provider, best_value = rankings[0]
        worst_provider, worst_value = rankings[-1]
        
        # Performance gap analysis
        if len(rankings) > 1:
            performance_gap = abs(best_value - worst_value)
            avg_value = sum(value for _, value in rankings) / len(rankings)
            gap_percentage = (performance_gap / avg_value) * 100 if avg_value > 0 else 0
            
            if gap_percentage > 50:
                recommendations.append(f"Significant performance gap detected: {best_provider} outperforms {worst_provider} by {gap_percentage:.1f}%")
                recommendations.append(f"Consider migrating traffic from {worst_provider} to {best_provider}")
            
            if metric == PerformanceMetric.COST_EFFICIENCY:
                recommendations.append(f"Use {best_provider} for cost-sensitive operations")
            elif metric == PerformanceMetric.LATENCY:
                recommendations.append(f"Use {best_provider} for latency-sensitive operations")
            elif metric == PerformanceMetric.SUCCESS_RATE:
                recommendations.append(f"Use {best_provider} for reliability-critical operations")
        
        return recommendations
    
    async def _save_performance_comparison(self, comparison: PerformanceComparison) -> None:
        """Save performance comparison to database."""
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO performance_comparisons (
                    comparison_id, providers, metric, time_period_hours,
                    rankings, best_provider, worst_provider,
                    mean_value, median_value, std_deviation,
                    recommendations, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                comparison.comparison_id,
                json.dumps(comparison.providers),
                comparison.metric.value,
                int(comparison.time_period.total_seconds() / 3600),
                json.dumps(comparison.rankings),
                comparison.best_provider,
                comparison.worst_provider,
                comparison.mean_value,
                comparison.median_value,
                comparison.std_deviation,
                json.dumps(comparison.recommendations),
                comparison.created_at.isoformat()
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def generate_optimization_recommendations(
        self,
        provider: str,
        analysis_period: timedelta = timedelta(days=7)
    ) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations for a provider.
        
        Args:
            provider: API provider name
            analysis_period: Period to analyze for recommendations
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        cutoff_time = datetime.utcnow() - analysis_period
        
        # Get recent performance data
        snapshots = await self._get_provider_snapshots(provider, cutoff_time)
        
        if not snapshots:
            return recommendations
        
        # Analyze different performance aspects
        recommendations.extend(await self._analyze_latency_performance(provider, snapshots))
        recommendations.extend(await self._analyze_cost_efficiency(provider, snapshots))
        recommendations.extend(await self._analyze_reliability(provider, snapshots))
        recommendations.extend(await self._analyze_throughput(provider, snapshots))
        
        # Save recommendations to database
        for rec in recommendations:
            await self._save_optimization_recommendation(rec)
        
        logger.info(f"Generated {len(recommendations)} optimization recommendations for {provider}")
        return recommendations
    
    async def _analyze_latency_performance(
        self,
        provider: str,
        snapshots: List[PerformanceSnapshot]
    ) -> List[OptimizationRecommendation]:
        """Analyze latency performance and generate recommendations."""
        recommendations = []
        
        if not snapshots:
            return recommendations
        
        # Calculate latency statistics
        latencies = [s.avg_latency_ms for s in snapshots]
        avg_latency = statistics.mean(latencies)
        p95_latencies = [s.p95_latency_ms for s in snapshots]
        avg_p95_latency = statistics.mean(p95_latencies)
        
        # High latency detection
        if avg_latency > 2000:  # > 2 seconds
            rec = OptimizationRecommendation(
                recommendation_id=f"{provider}_high_latency_{int(datetime.utcnow().timestamp())}",
                provider=provider,
                priority=OptimizationPriority.HIGH,
                issue_type="high_latency",
                issue_description=f"Average latency is {avg_latency:.0f}ms, which is above acceptable thresholds",
                impact_assessment="High latency impacts user experience and may indicate performance bottlenecks",
                recommendation_title="Optimize Request Latency",
                recommendation_description="Implement latency optimization strategies to improve response times",
                implementation_steps=[
                    "Analyze request patterns to identify bottlenecks",
                    "Implement request caching for frequently accessed data",
                    "Consider using faster endpoints or models if available",
                    "Optimize request payload size and structure"
                ],
                expected_improvement={"latency": -30.0},  # 30% reduction
                implementation_effort="medium",
                risk_level="low",
                supporting_metrics={
                    "avg_latency_ms": avg_latency,
                    "p95_latency_ms": avg_p95_latency,
                    "sample_size": len(snapshots)
                },
                confidence_score=85.0
            )
            recommendations.append(rec)
        
        return recommendations
    
    async def _analyze_cost_efficiency(
        self,
        provider: str,
        snapshots: List[PerformanceSnapshot]
    ) -> List[OptimizationRecommendation]:
        """Analyze cost efficiency and generate recommendations."""
        recommendations = []
        
        if not snapshots:
            return recommendations
        
        # Calculate cost statistics
        costs_per_request = [s.cost_per_request for s in snapshots if s.cost_per_request > 0]
        costs_per_token = [s.cost_per_token for s in snapshots if s.cost_per_token > 0]
        
        if costs_per_request:
            avg_cost_per_request = statistics.mean(costs_per_request)
            
            # High cost per request detection
            if avg_cost_per_request > 0.01:  # > $0.01 per request
                rec = OptimizationRecommendation(
                    recommendation_id=f"{provider}_high_cost_{int(datetime.utcnow().timestamp())}",
                    provider=provider,
                    priority=OptimizationPriority.MEDIUM,
                    issue_type="high_cost",
                    issue_description=f"Average cost per request is ${avg_cost_per_request:.4f}, which may be optimizable",
                    impact_assessment="High per-request costs can significantly impact budget efficiency",
                    recommendation_title="Optimize Cost Efficiency",
                    recommendation_description="Implement cost optimization strategies to reduce per-request expenses",
                    implementation_steps=[
                        "Analyze token usage patterns to identify optimization opportunities",
                        "Consider using smaller models for simpler tasks",
                        "Implement request batching where possible",
                        "Optimize prompt engineering to reduce token usage"
                    ],
                    expected_improvement={"cost_efficiency": 25.0},  # 25% improvement
                    implementation_effort="medium",
                    risk_level="low",
                    supporting_metrics={
                        "avg_cost_per_request": avg_cost_per_request,
                        "sample_size": len(costs_per_request)
                    },
                    confidence_score=75.0
                )
                recommendations.append(rec)
        
        return recommendations
    
    async def _analyze_reliability(
        self,
        provider: str,
        snapshots: List[PerformanceSnapshot]
    ) -> List[OptimizationRecommendation]:
        """Analyze reliability and generate recommendations."""
        recommendations = []
        
        if not snapshots:
            return recommendations
        
        # Calculate reliability statistics
        success_rates = [s.success_rate for s in snapshots]
        avg_success_rate = statistics.mean(success_rates)
        error_rates = [s.error_rate for s in snapshots]
        avg_error_rate = statistics.mean(error_rates)
        
        # Low reliability detection
        if avg_success_rate < 95.0:  # < 95% success rate
            rec = OptimizationRecommendation(
                recommendation_id=f"{provider}_low_reliability_{int(datetime.utcnow().timestamp())}",
                provider=provider,
                priority=OptimizationPriority.HIGH,
                issue_type="low_reliability",
                issue_description=f"Success rate is {avg_success_rate:.1f}%, below acceptable threshold of 95%",
                impact_assessment="Low reliability affects system stability and user experience",
                recommendation_title="Improve System Reliability",
                recommendation_description="Implement reliability improvements to reduce error rates",
                implementation_steps=[
                    "Implement robust error handling and retry logic",
                    "Add circuit breaker patterns for fault tolerance",
                    "Monitor and analyze error patterns to identify root causes",
                    "Consider implementing failover to alternative providers"
                ],
                expected_improvement={"success_rate": 5.0, "error_rate": -50.0},
                implementation_effort="high",
                risk_level="medium",
                supporting_metrics={
                    "avg_success_rate": avg_success_rate,
                    "avg_error_rate": avg_error_rate,
                    "sample_size": len(snapshots)
                },
                confidence_score=90.0
            )
            recommendations.append(rec)
        
        return recommendations
    
    async def _analyze_throughput(
        self,
        provider: str,
        snapshots: List[PerformanceSnapshot]
    ) -> List[OptimizationRecommendation]:
        """Analyze throughput and generate recommendations."""
        recommendations = []
        
        if not snapshots:
            return recommendations
        
        # Calculate throughput statistics
        throughputs = [s.throughput_rps for s in snapshots if s.throughput_rps > 0]
        
        if throughputs:
            avg_throughput = statistics.mean(throughputs)
            
            # Low throughput detection
            if avg_throughput < 1.0:  # < 1 request per second
                rec = OptimizationRecommendation(
                    recommendation_id=f"{provider}_low_throughput_{int(datetime.utcnow().timestamp())}",
                    provider=provider,
                    priority=OptimizationPriority.MEDIUM,
                    issue_type="low_throughput",
                    issue_description=f"Average throughput is {avg_throughput:.2f} RPS, which may indicate underutilization",
                    impact_assessment="Low throughput may indicate inefficient resource utilization",
                    recommendation_title="Optimize Request Throughput",
                    recommendation_description="Implement throughput optimization strategies",
                    implementation_steps=[
                        "Analyze request patterns and identify batching opportunities",
                        "Implement connection pooling and keep-alive connections",
                        "Consider parallel request processing where appropriate",
                        "Optimize request scheduling and load balancing"
                    ],
                    expected_improvement={"throughput": 100.0},  # 100% improvement
                    implementation_effort="medium",
                    risk_level="low",
                    supporting_metrics={
                        "avg_throughput_rps": avg_throughput,
                        "sample_size": len(throughputs)
                    },
                    confidence_score=70.0
                )
                recommendations.append(rec)
        
        return recommendations
    
    async def _save_optimization_recommendation(self, recommendation: OptimizationRecommendation) -> None:
        """Save optimization recommendation to database."""
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO optimization_recommendations (
                    recommendation_id, provider, priority, issue_type,
                    issue_description, impact_assessment, recommendation_title,
                    recommendation_description, implementation_steps,
                    expected_improvement, implementation_effort, risk_level,
                    supporting_metrics, confidence_score, created_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recommendation.recommendation_id,
                recommendation.provider,
                recommendation.priority.value,
                recommendation.issue_type,
                recommendation.issue_description,
                recommendation.impact_assessment,
                recommendation.recommendation_title,
                recommendation.recommendation_description,
                json.dumps(recommendation.implementation_steps),
                json.dumps(recommendation.expected_improvement),
                recommendation.implementation_effort,
                recommendation.risk_level,
                json.dumps(recommendation.supporting_metrics),
                recommendation.confidence_score,
                recommendation.created_at.isoformat(),
                recommendation.status
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def _load_recent_data(self) -> None:
        """Load recent performance data into cache."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT provider FROM performance_snapshots
                WHERE timestamp >= ?
            """, (cutoff_time.isoformat(),))
            
            providers = [row[0] for row in cursor.fetchall()]
            
            for provider in providers:
                snapshots = await self._get_provider_snapshots(provider, cutoff_time)
                self.performance_cache[provider] = snapshots
            
            logger.info(f"Loaded performance data for {len(providers)} providers")
            
        finally:
            conn.close()
    
    async def _analysis_loop(self) -> None:
        """Background loop for performance analysis."""
        while self.is_running:
            try:
                # Perform periodic analysis
                await self._perform_periodic_analysis()
                
                await asyncio.sleep(self.analysis_interval * 60)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_periodic_analysis(self) -> None:
        """Perform periodic performance analysis."""
        # Get all providers with recent data
        providers = list(self.performance_cache.keys())
        
        if len(providers) < 2:
            return
        
        # Perform cross-provider comparisons
        for metric in PerformanceMetric:
            try:
                await self.compare_providers(providers, metric, timedelta(hours=1))
            except Exception as e:
                logger.error(f"Error comparing providers on {metric.value}: {e}")
        
        # Generate optimization recommendations
        for provider in providers:
            try:
                await self.generate_optimization_recommendations(provider, timedelta(hours=6))
            except Exception as e:
                logger.error(f"Error generating recommendations for {provider}: {e}")
    
    def get_analyzer_status(self) -> Dict[str, Any]:
        """
        Get comprehensive analyzer status.
        
        Returns:
            Dictionary with analyzer status
        """
        return {
            "is_running": self.is_running,
            "cached_providers": len(self.performance_cache),
            "cached_comparisons": len(self.comparison_cache),
            "storage_path": self.storage_path,
            "analysis_interval_minutes": self.analysis_interval,
            "trend_analysis_days": self.trend_analysis_days,
            "recent_snapshots": {
                provider: len(snapshots)
                for provider, snapshots in self.performance_cache.items()
            }
        }