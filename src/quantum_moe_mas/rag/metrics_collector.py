"""
Performance metrics collection and monitoring for retrieval optimization.

This module implements comprehensive metrics collection to track retrieval
performance, confidence scores, and system efficiency as specified in requirements.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import statistics

import numpy as np

from quantum_moe_mas.core.logging_simple import get_logger
from quantum_moe_mas.rag.retrieval import SearchStrategy, RetrievalMetrics

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected."""
    
    RETRIEVAL_LATENCY = "retrieval_latency"
    CONFIDENCE_SCORE = "confidence_score"
    FALLBACK_TRIGGERED = "fallback_triggered"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUERY_COMPLEXITY = "query_complexity"
    RESULT_RELEVANCE = "result_relevance"
    SYSTEM_THROUGHPUT = "system_throughput"
    ERROR_RATE = "error_rate"
    STRATEGY_PERFORMANCE = "strategy_performance"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric statistics."""
    
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    p95: float
    p99: float
    std_dev: float
    
    @classmethod
    def from_values(cls, values: List[float]) -> AggregatedMetric:
        """Create aggregated metric from list of values."""
        if not values:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        sorted_values = sorted(values)
        count = len(values)
        
        return cls(
            count=count,
            sum=sum(values),
            min=min(values),
            max=max(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            p95=sorted_values[int(0.95 * count)] if count > 0 else 0.0,
            p99=sorted_values[int(0.99 * count)] if count > 0 else 0.0,
            std_dev=statistics.stdev(values) if count > 1 else 0.0,
        )


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    
    time_range: Tuple[datetime, datetime]
    total_queries: int
    avg_latency: float
    p95_latency: float
    confidence_distribution: Dict[str, int]
    fallback_rate: float
    cache_hit_rate: float
    strategy_performance: Dict[SearchStrategy, AggregatedMetric]
    error_rate: float
    throughput_qps: float
    efficiency_gains: Dict[str, float]
    recommendations: List[str]


class MetricsCollector:
    """Comprehensive metrics collection and analysis system."""
    
    def __init__(
        self,
        max_history_hours: int = 24,
        aggregation_window_minutes: int = 5,
        enable_real_time_alerts: bool = True,
    ):
        self.max_history_hours = max_history_hours
        self.aggregation_window_minutes = aggregation_window_minutes
        self.enable_real_time_alerts = enable_real_time_alerts
        
        # Metric storage
        self.metrics: Dict[MetricType, deque] = defaultdict(
            lambda: deque(maxlen=int(max_history_hours * 60 / aggregation_window_minutes))
        )
        
        # Real-time tracking
        self.current_window_start = time.time()
        self.current_window_metrics: Dict[MetricType, List[MetricPoint]] = defaultdict(list)
        
        # Performance baselines
        self.baselines = {
            MetricType.RETRIEVAL_LATENCY: 5.0,  # 5 seconds target
            MetricType.CONFIDENCE_SCORE: 0.7,   # 70% confidence threshold
            MetricType.CACHE_HIT_RATE: 0.4,     # 40% cache hit rate target
            MetricType.ERROR_RATE: 0.01,        # 1% error rate threshold
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            MetricType.RETRIEVAL_LATENCY: 10.0,  # Alert if > 10 seconds
            MetricType.ERROR_RATE: 0.05,         # Alert if > 5% error rate
            MetricType.CONFIDENCE_SCORE: 0.5,    # Alert if avg confidence < 50%
        }
        
        # Background tasks
        self._aggregation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("Metrics collector initialized")
    
    async def start(self) -> None:
        """Start background metric aggregation and cleanup tasks."""
        if self._aggregation_task is None:
            self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Metrics collector started")
    
    async def stop(self) -> None:
        """Stop background tasks."""
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
            self._aggregation_task = None
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        logger.info("Metrics collector stopped")
    
    def record_retrieval_metrics(
        self,
        query: str,
        strategy: SearchStrategy,
        metrics: RetrievalMetrics,
        cache_hit: bool = False,
        fallback_triggered: bool = False,
        user_feedback: Optional[float] = None,
    ) -> None:
        """Record comprehensive retrieval metrics."""
        timestamp = time.time()
        
        # Basic retrieval metrics
        self._record_metric(
            MetricType.RETRIEVAL_LATENCY,
            metrics.retrieval_time,
            timestamp,
            {"query_length": len(query), "strategy": strategy.value}
        )
        
        self._record_metric(
            MetricType.CONFIDENCE_SCORE,
            metrics.avg_score,
            timestamp,
            {"strategy": strategy.value, "result_count": metrics.retrieved_count}
        )
        
        # Cache performance
        self._record_metric(
            MetricType.CACHE_HIT_RATE,
            1.0 if cache_hit else 0.0,
            timestamp,
            {"strategy": strategy.value}
        )
        
        # Fallback tracking
        if fallback_triggered:
            self._record_metric(
                MetricType.FALLBACK_TRIGGERED,
                1.0,
                timestamp,
                {"original_strategy": strategy.value, "confidence": metrics.avg_score}
            )
        
        # Strategy performance
        self._record_metric(
            MetricType.STRATEGY_PERFORMANCE,
            metrics.avg_score,
            timestamp,
            {"strategy": strategy.value, "latency": metrics.retrieval_time}
        )
        
        # Query complexity analysis
        complexity_score = self._calculate_query_complexity(query)
        self._record_metric(
            MetricType.QUERY_COMPLEXITY,
            complexity_score,
            timestamp,
            {"query_length": len(query), "word_count": len(query.split())}
        )
        
        # User feedback if provided
        if user_feedback is not None:
            self._record_metric(
                MetricType.USER_SATISFACTION,
                user_feedback,
                timestamp,
                {"strategy": strategy.value, "confidence": metrics.avg_score}
            )
        
        # Check for real-time alerts
        if self.enable_real_time_alerts:
            self._check_alerts(metrics, strategy)
    
    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record error occurrence."""
        self._record_metric(
            MetricType.ERROR_RATE,
            1.0,
            time.time(),
            {
                "error_type": error_type,
                "error_message": error_message[:100],  # Truncate long messages
                **(context or {})
            }
        )
    
    def record_throughput(self, queries_per_second: float) -> None:
        """Record system throughput."""
        self._record_metric(
            MetricType.SYSTEM_THROUGHPUT,
            queries_per_second,
            time.time(),
            {}
        )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics."""
        current_time = time.time()
        window_duration = current_time - self.current_window_start
        
        metrics_summary = {}
        
        for metric_type, points in self.current_window_metrics.items():
            if not points:
                continue
            
            values = [p.value for p in points]
            
            if metric_type == MetricType.CACHE_HIT_RATE:
                # Special handling for hit rate
                metrics_summary[metric_type.value] = {
                    "current_rate": sum(values) / len(values) if values else 0.0,
                    "total_requests": len(values),
                }
            elif metric_type == MetricType.ERROR_RATE:
                # Error rate calculation
                metrics_summary[metric_type.value] = {
                    "error_count": sum(values),
                    "error_rate": sum(values) / max(1, window_duration / 60),  # errors per minute
                }
            else:
                # Standard aggregation
                metrics_summary[metric_type.value] = {
                    "current": values[-1] if values else 0.0,
                    "avg": sum(values) / len(values) if values else 0.0,
                    "min": min(values) if values else 0.0,
                    "max": max(values) if values else 0.0,
                    "count": len(values),
                }
        
        return {
            "window_start": datetime.fromtimestamp(self.current_window_start).isoformat(),
            "window_duration_seconds": window_duration,
            "metrics": metrics_summary,
        }
    
    def get_historical_metrics(
        self,
        metric_type: MetricType,
        hours_back: int = 1,
    ) -> List[Tuple[datetime, AggregatedMetric]]:
        """Get historical aggregated metrics."""
        if metric_type not in self.metrics:
            return []
        
        cutoff_time = time.time() - (hours_back * 3600)
        historical_data = []
        
        for window_data in self.metrics[metric_type]:
            if window_data["timestamp"] >= cutoff_time:
                timestamp = datetime.fromtimestamp(window_data["timestamp"])
                aggregated = AggregatedMetric.from_values(window_data["values"])
                historical_data.append((timestamp, aggregated))
        
        return historical_data
    
    def generate_performance_report(
        self,
        hours_back: int = 1,
    ) -> PerformanceReport:
        """Generate comprehensive performance report."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        cutoff_timestamp = start_time.timestamp()
        
        # Collect all metrics within time range
        all_latencies = []
        all_confidences = []
        fallback_count = 0
        cache_hits = 0
        cache_requests = 0
        error_count = 0
        total_queries = 0
        strategy_metrics = defaultdict(list)
        
        for metric_type, windows in self.metrics.items():
            for window_data in windows:
                if window_data["timestamp"] >= cutoff_timestamp:
                    values = window_data["values"]
                    
                    if metric_type == MetricType.RETRIEVAL_LATENCY:
                        all_latencies.extend(values)
                        total_queries += len(values)
                    elif metric_type == MetricType.CONFIDENCE_SCORE:
                        all_confidences.extend(values)
                    elif metric_type == MetricType.FALLBACK_TRIGGERED:
                        fallback_count += sum(values)
                    elif metric_type == MetricType.CACHE_HIT_RATE:
                        cache_hits += sum(values)
                        cache_requests += len(values)
                    elif metric_type == MetricType.ERROR_RATE:
                        error_count += sum(values)
                    elif metric_type == MetricType.STRATEGY_PERFORMANCE:
                        # Group by strategy from metadata
                        for i, value in enumerate(values):
                            metadata = window_data.get("metadata", [{}])
                            if i < len(metadata) and "strategy" in metadata[i]:
                                strategy_name = metadata[i]["strategy"]
                                try:
                                    strategy = SearchStrategy(strategy_name)
                                    strategy_metrics[strategy].append(value)
                                except ValueError:
                                    pass  # Skip invalid strategy names
        
        # Calculate aggregated statistics
        avg_latency = statistics.mean(all_latencies) if all_latencies else 0.0
        p95_latency = sorted(all_latencies)[int(0.95 * len(all_latencies))] if all_latencies else 0.0
        
        # Confidence distribution
        confidence_distribution = {
            "high (>0.8)": sum(1 for c in all_confidences if c > 0.8),
            "medium (0.5-0.8)": sum(1 for c in all_confidences if 0.5 <= c <= 0.8),
            "low (<0.5)": sum(1 for c in all_confidences if c < 0.5),
        }
        
        # Calculate rates
        fallback_rate = fallback_count / max(1, total_queries)
        cache_hit_rate = cache_hits / max(1, cache_requests)
        error_rate = error_count / max(1, total_queries)
        
        # Throughput calculation
        duration_hours = hours_back
        throughput_qps = total_queries / (duration_hours * 3600) if duration_hours > 0 else 0.0
        
        # Strategy performance aggregation
        strategy_performance = {}
        for strategy, values in strategy_metrics.items():
            if values:
                strategy_performance[strategy] = AggregatedMetric.from_values(values)
        
        # Calculate efficiency gains
        efficiency_gains = self._calculate_efficiency_gains(
            avg_latency, cache_hit_rate, fallback_rate
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_latency, p95_latency, confidence_distribution,
            fallback_rate, cache_hit_rate, error_rate
        )
        
        return PerformanceReport(
            time_range=(start_time, end_time),
            total_queries=total_queries,
            avg_latency=avg_latency,
            p95_latency=p95_latency,
            confidence_distribution=confidence_distribution,
            fallback_rate=fallback_rate,
            cache_hit_rate=cache_hit_rate,
            strategy_performance=strategy_performance,
            error_rate=error_rate,
            throughput_qps=throughput_qps,
            efficiency_gains=efficiency_gains,
            recommendations=recommendations,
        )
    
    def _record_metric(
        self,
        metric_type: MetricType,
        value: float,
        timestamp: float,
        metadata: Dict[str, Any],
    ) -> None:
        """Record a metric point."""
        point = MetricPoint(
            timestamp=timestamp,
            value=value,
            metadata=metadata,
        )
        
        self.current_window_metrics[metric_type].append(point)
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)."""
        # Simple complexity scoring based on various factors
        word_count = len(query.split())
        char_count = len(query)
        
        # Factors that increase complexity
        complexity_score = 0.0
        
        # Length factor
        complexity_score += min(word_count / 20, 0.3)  # Max 0.3 for length
        
        # Special characters and operators
        special_chars = sum(1 for c in query if c in "()[]{}\"'*?+-")
        complexity_score += min(special_chars / 10, 0.2)  # Max 0.2 for special chars
        
        # Question words (increase complexity)
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        question_count = sum(1 for word in query.lower().split() if word in question_words)
        complexity_score += min(question_count / 3, 0.2)  # Max 0.2 for questions
        
        # Technical terms (simplified detection)
        technical_indicators = ["algorithm", "implementation", "architecture", "optimization"]
        tech_count = sum(1 for term in technical_indicators if term in query.lower())
        complexity_score += min(tech_count / 2, 0.3)  # Max 0.3 for technical terms
        
        return min(complexity_score, 1.0)
    
    def _check_alerts(self, metrics: RetrievalMetrics, strategy: SearchStrategy) -> None:
        """Check for real-time alert conditions."""
        alerts = []
        
        # Latency alert
        if metrics.retrieval_time > self.alert_thresholds[MetricType.RETRIEVAL_LATENCY]:
            alerts.append(f"High latency detected: {metrics.retrieval_time:.2f}s (threshold: {self.alert_thresholds[MetricType.RETRIEVAL_LATENCY]}s)")
        
        # Confidence alert
        if metrics.avg_score < self.alert_thresholds[MetricType.CONFIDENCE_SCORE]:
            alerts.append(f"Low confidence detected: {metrics.avg_score:.2f} (threshold: {self.alert_thresholds[MetricType.CONFIDENCE_SCORE]})")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")
    
    def _calculate_efficiency_gains(
        self,
        avg_latency: float,
        cache_hit_rate: float,
        fallback_rate: float,
    ) -> Dict[str, float]:
        """Calculate efficiency gains compared to baselines."""
        gains = {}
        
        # Latency improvement
        baseline_latency = self.baselines[MetricType.RETRIEVAL_LATENCY]
        if baseline_latency > 0:
            latency_gain = max(0, (baseline_latency - avg_latency) / baseline_latency)
            gains["latency_improvement"] = latency_gain
        
        # Cache efficiency
        baseline_cache_rate = self.baselines[MetricType.CACHE_HIT_RATE]
        cache_efficiency = cache_hit_rate / baseline_cache_rate if baseline_cache_rate > 0 else 1.0
        gains["cache_efficiency"] = cache_efficiency
        
        # Overall efficiency (composite score)
        overall_efficiency = (
            (1 - fallback_rate) * 0.3 +  # Lower fallback rate is better
            cache_efficiency * 0.4 +      # Higher cache hit rate is better
            (1 - min(avg_latency / baseline_latency, 1.0)) * 0.3  # Lower latency is better
        )
        gains["overall_efficiency"] = overall_efficiency
        
        return gains
    
    def _generate_recommendations(
        self,
        avg_latency: float,
        p95_latency: float,
        confidence_distribution: Dict[str, int],
        fallback_rate: float,
        cache_hit_rate: float,
        error_rate: float,
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Latency recommendations
        if avg_latency > self.baselines[MetricType.RETRIEVAL_LATENCY]:
            recommendations.append(
                f"Consider optimizing retrieval latency (current: {avg_latency:.2f}s, target: {self.baselines[MetricType.RETRIEVAL_LATENCY]}s)"
            )
        
        if p95_latency > avg_latency * 2:
            recommendations.append("High latency variance detected - investigate outlier queries")
        
        # Confidence recommendations
        low_confidence_pct = confidence_distribution.get("low (<0.5)", 0)
        total_queries = sum(confidence_distribution.values())
        if total_queries > 0 and low_confidence_pct / total_queries > 0.2:
            recommendations.append("High percentage of low-confidence results - consider improving embedding quality or search strategies")
        
        # Fallback recommendations
        if fallback_rate > 0.3:
            recommendations.append(f"High fallback rate ({fallback_rate:.1%}) - consider adjusting confidence thresholds or improving primary search strategies")
        
        # Cache recommendations
        if cache_hit_rate < self.baselines[MetricType.CACHE_HIT_RATE]:
            recommendations.append(f"Cache hit rate below target ({cache_hit_rate:.1%} vs {self.baselines[MetricType.CACHE_HIT_RATE]:.1%}) - consider increasing cache size or improving cache strategies")
        
        # Error rate recommendations
        if error_rate > self.baselines[MetricType.ERROR_RATE]:
            recommendations.append(f"Error rate above threshold ({error_rate:.1%}) - investigate and fix recurring errors")
        
        return recommendations
    
    async def _aggregation_loop(self) -> None:
        """Background task to aggregate metrics periodically."""
        while True:
            try:
                await asyncio.sleep(self.aggregation_window_minutes * 60)
                await self._aggregate_current_window()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics aggregation: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up old metrics."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
    
    async def _aggregate_current_window(self) -> None:
        """Aggregate current window metrics."""
        current_time = time.time()
        
        for metric_type, points in self.current_window_metrics.items():
            if points:
                values = [p.value for p in points]
                metadata = [p.metadata for p in points]
                
                # Store aggregated window
                window_data = {
                    "timestamp": self.current_window_start,
                    "values": values,
                    "metadata": metadata,
                }
                
                self.metrics[metric_type].append(window_data)
        
        # Reset current window
        self.current_window_start = current_time
        self.current_window_metrics.clear()
        
        logger.debug("Metrics window aggregated")
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up metrics older than max_history_hours."""
        cutoff_time = time.time() - (self.max_history_hours * 3600)
        
        for metric_type, windows in self.metrics.items():
            # Remove old windows
            while windows and windows[0]["timestamp"] < cutoff_time:
                windows.popleft()
        
        logger.debug("Old metrics cleaned up")