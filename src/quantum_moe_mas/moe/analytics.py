"""
Advanced analytics and visualization for MoE routing.

This module provides comprehensive analytics, efficiency calculations,
and data preparation for UI dashboards.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

from quantum_moe_mas.moe.metrics import RoutingMetrics, RoutingDecision
from quantum_moe_mas.moe.expert import Expert
from quantum_moe_mas.core.logging_simple import get_logger


logger = get_logger(__name__)


@dataclass
class EfficiencyReport:
    """
    Comprehensive efficiency report for MoE system.
    
    Tracks latency reduction, cost savings, and performance improvements.
    """
    
    # Latency metrics
    baseline_latency_ms: float
    current_latency_ms: float
    latency_reduction_ms: float
    latency_reduction_percentage: float
    
    # Cost metrics
    baseline_cost: float
    current_cost: float
    cost_savings: float
    cost_savings_percentage: float
    
    # Token metrics
    baseline_tokens: int
    current_tokens: int
    tokens_saved: int
    tokens_saved_percentage: float
    
    # Performance metrics
    throughput_improvement: float
    accuracy_maintained: bool
    
    # Time period
    period_start: datetime
    period_end: datetime
    total_requests: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "latency": {
                "baseline_ms": self.baseline_latency_ms,
                "current_ms": self.current_latency_ms,
                "reduction_ms": self.latency_reduction_ms,
                "reduction_percentage": self.latency_reduction_percentage,
            },
            "cost": {
                "baseline": self.baseline_cost,
                "current": self.current_cost,
                "savings": self.cost_savings,
                "savings_percentage": self.cost_savings_percentage,
            },
            "tokens": {
                "baseline": self.baseline_tokens,
                "current": self.current_tokens,
                "saved": self.tokens_saved,
                "saved_percentage": self.tokens_saved_percentage,
            },
            "performance": {
                "throughput_improvement": self.throughput_improvement,
                "accuracy_maintained": self.accuracy_maintained,
            },
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
                "total_requests": self.total_requests,
            }
        }


@dataclass
class VisualizationData:
    """
    Data prepared for UI visualization.
    
    Provides structured data for charts, graphs, and dashboards.
    """
    
    # Time series data
    latency_over_time: List[Dict[str, Any]] = field(default_factory=list)
    confidence_over_time: List[Dict[str, Any]] = field(default_factory=list)
    throughput_over_time: List[Dict[str, Any]] = field(default_factory=list)
    
    # Distribution data
    expert_utilization: Dict[str, float] = field(default_factory=dict)
    domain_distribution: Dict[str, float] = field(default_factory=dict)
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Performance data
    top_experts: List[Dict[str, Any]] = field(default_factory=list)
    routing_heatmap: List[List[float]] = field(default_factory=list)
    
    # Real-time metrics
    current_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert to JSON string for API responses."""
        return json.dumps({
            "time_series": {
                "latency": self.latency_over_time,
                "confidence": self.confidence_over_time,
                "throughput": self.throughput_over_time,
            },
            "distributions": {
                "expert_utilization": self.expert_utilization,
                "domain_distribution": self.domain_distribution,
                "confidence_distribution": self.confidence_distribution,
            },
            "performance": {
                "top_experts": self.top_experts,
                "routing_heatmap": self.routing_heatmap,
            },
            "current_metrics": self.current_metrics,
        }, indent=2)


class RoutingAnalytics:
    """
    Advanced analytics engine for MoE routing system.
    
    Provides comprehensive analysis, efficiency calculations, and
    visualization data preparation.
    """
    
    def __init__(self) -> None:
        """Initialize routing analytics."""
        self.decision_history: List[RoutingDecision] = []
        self.max_history_size = 10000
        
        logger.info("Initialized RoutingAnalytics")
    
    def record_decision(self, decision: RoutingDecision) -> None:
        """
        Record a routing decision for analysis.
        
        Args:
            decision: Routing decision to record
        """
        self.decision_history.append(decision)
        
        # Maintain history size limit
        if len(self.decision_history) > self.max_history_size:
            self.decision_history = self.decision_history[-self.max_history_size:]
    
    def calculate_efficiency_report(
        self,
        baseline_latency_ms: float = 1000.0,
        baseline_cost_per_request: float = 0.01,
        baseline_tokens_per_request: int = 1000,
        time_window_hours: Optional[int] = None
    ) -> EfficiencyReport:
        """
        Calculate comprehensive efficiency report.
        
        Args:
            baseline_latency_ms: Baseline latency for comparison
            baseline_cost_per_request: Baseline cost per request
            baseline_tokens_per_request: Baseline tokens per request
            time_window_hours: Time window for analysis (None = all time)
        
        Returns:
            EfficiencyReport instance
        """
        # Filter decisions by time window
        decisions = self.decision_history
        if time_window_hours:
            cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
            decisions = [d for d in decisions if d.timestamp >= cutoff]
        
        if not decisions:
            # Return empty report
            return EfficiencyReport(
                baseline_latency_ms=baseline_latency_ms,
                current_latency_ms=0.0,
                latency_reduction_ms=0.0,
                latency_reduction_percentage=0.0,
                baseline_cost=0.0,
                current_cost=0.0,
                cost_savings=0.0,
                cost_savings_percentage=0.0,
                baseline_tokens=0,
                current_tokens=0,
                tokens_saved=0,
                tokens_saved_percentage=0.0,
                throughput_improvement=0.0,
                accuracy_maintained=True,
                period_start=datetime.utcnow(),
                period_end=datetime.utcnow(),
                total_requests=0
            )
        
        # Calculate current metrics
        total_requests = len(decisions)
        current_latency_ms = sum(d.latency_ms for d in decisions) / total_requests
        
        # Estimate current cost and tokens (using sparse activation ratio)
        sparse_ratio = 0.055  # 37B/671B
        current_cost = baseline_cost_per_request * sparse_ratio * total_requests
        current_tokens = int(baseline_tokens_per_request * sparse_ratio * total_requests)
        
        # Calculate baseline totals
        baseline_cost_total = baseline_cost_per_request * total_requests
        baseline_tokens_total = baseline_tokens_per_request * total_requests
        
        # Calculate reductions
        latency_reduction_ms = baseline_latency_ms - current_latency_ms
        latency_reduction_pct = (latency_reduction_ms / baseline_latency_ms) * 100.0
        
        cost_savings = baseline_cost_total - current_cost
        cost_savings_pct = (cost_savings / baseline_cost_total) * 100.0 if baseline_cost_total > 0 else 0.0
        
        tokens_saved = baseline_tokens_total - current_tokens
        tokens_saved_pct = (tokens_saved / baseline_tokens_total) * 100.0 if baseline_tokens_total > 0 else 0.0
        
        # Calculate throughput improvement
        baseline_throughput = 1000.0 / baseline_latency_ms  # requests per second
        current_throughput = 1000.0 / current_latency_ms if current_latency_ms > 0 else 0.0
        throughput_improvement = ((current_throughput - baseline_throughput) / baseline_throughput) * 100.0
        
        # Check if accuracy is maintained (confidence >= 80%)
        avg_confidence = sum(d.routing_confidence for d in decisions) / total_requests
        accuracy_maintained = avg_confidence >= 80.0
        
        return EfficiencyReport(
            baseline_latency_ms=baseline_latency_ms,
            current_latency_ms=current_latency_ms,
            latency_reduction_ms=latency_reduction_ms,
            latency_reduction_percentage=latency_reduction_pct,
            baseline_cost=baseline_cost_total,
            current_cost=current_cost,
            cost_savings=cost_savings,
            cost_savings_percentage=cost_savings_pct,
            baseline_tokens=baseline_tokens_total,
            current_tokens=current_tokens,
            tokens_saved=tokens_saved,
            tokens_saved_percentage=tokens_saved_pct,
            throughput_improvement=throughput_improvement,
            accuracy_maintained=accuracy_maintained,
            period_start=decisions[0].timestamp,
            period_end=decisions[-1].timestamp,
            total_requests=total_requests
        )
    
    def generate_visualization_data(
        self,
        metrics: RoutingMetrics,
        experts: Dict[str, Expert],
        time_window_hours: int = 24
    ) -> VisualizationData:
        """
        Generate data for UI visualization.
        
        Args:
            metrics: Current routing metrics
            experts: Dictionary of experts
            time_window_hours: Time window for time series data
        
        Returns:
            VisualizationData instance
        """
        viz_data = VisualizationData()
        
        # Filter recent decisions
        cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_decisions = [
            d for d in self.decision_history
            if d.timestamp >= cutoff
        ]
        
        # Generate time series data
        viz_data.latency_over_time = self._generate_latency_series(recent_decisions)
        viz_data.confidence_over_time = self._generate_confidence_series(recent_decisions)
        viz_data.throughput_over_time = self._generate_throughput_series(recent_decisions)
        
        # Generate distribution data
        viz_data.expert_utilization = self._calculate_expert_utilization(
            recent_decisions,
            experts
        )
        viz_data.domain_distribution = metrics.get_domain_distribution_percentages()
        viz_data.confidence_distribution = self._calculate_confidence_distribution(
            recent_decisions
        )
        
        # Generate performance data
        viz_data.top_experts = self._get_top_performing_experts(experts)
        viz_data.routing_heatmap = self._generate_routing_heatmap(
            recent_decisions,
            experts
        )
        
        # Current metrics
        viz_data.current_metrics = metrics.to_dict()
        
        return viz_data
    
    def _generate_latency_series(
        self,
        decisions: List[RoutingDecision]
    ) -> List[Dict[str, Any]]:
        """Generate latency time series data."""
        if not decisions:
            return []
        
        # Group by hour
        hourly_data: Dict[datetime, List[float]] = {}
        for decision in decisions:
            hour = decision.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour not in hourly_data:
                hourly_data[hour] = []
            hourly_data[hour].append(decision.latency_ms)
        
        # Calculate averages
        series = []
        for hour in sorted(hourly_data.keys()):
            latencies = hourly_data[hour]
            series.append({
                "timestamp": hour.isoformat(),
                "average_latency_ms": sum(latencies) / len(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "count": len(latencies)
            })
        
        return series
    
    def _generate_confidence_series(
        self,
        decisions: List[RoutingDecision]
    ) -> List[Dict[str, Any]]:
        """Generate confidence time series data."""
        if not decisions:
            return []
        
        # Group by hour
        hourly_data: Dict[datetime, List[float]] = {}
        for decision in decisions:
            hour = decision.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour not in hourly_data:
                hourly_data[hour] = []
            hourly_data[hour].append(decision.routing_confidence)
        
        # Calculate averages
        series = []
        for hour in sorted(hourly_data.keys()):
            confidences = hourly_data[hour]
            series.append({
                "timestamp": hour.isoformat(),
                "average_confidence": sum(confidences) / len(confidences),
                "min_confidence": min(confidences),
                "max_confidence": max(confidences),
                "count": len(confidences)
            })
        
        return series
    
    def _generate_throughput_series(
        self,
        decisions: List[RoutingDecision]
    ) -> List[Dict[str, Any]]:
        """Generate throughput time series data."""
        if not decisions:
            return []
        
        # Group by hour
        hourly_counts: Dict[datetime, int] = {}
        for decision in decisions:
            hour = decision.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        # Calculate throughput (requests per hour)
        series = []
        for hour in sorted(hourly_counts.keys()):
            series.append({
                "timestamp": hour.isoformat(),
                "requests_per_hour": hourly_counts[hour],
                "requests_per_second": hourly_counts[hour] / 3600.0
            })
        
        return series
    
    def _calculate_expert_utilization(
        self,
        decisions: List[RoutingDecision],
        experts: Dict[str, Expert]
    ) -> Dict[str, float]:
        """Calculate expert utilization percentages."""
        if not decisions:
            return {}
        
        selection_counts: Dict[str, int] = {}
        for decision in decisions:
            for expert_id in decision.selected_expert_ids:
                selection_counts[expert_id] = selection_counts.get(expert_id, 0) + 1
        
        total_selections = sum(selection_counts.values())
        if total_selections == 0:
            return {}
        
        # Convert to percentages with expert names
        utilization = {}
        for expert_id, count in selection_counts.items():
            expert = experts.get(expert_id)
            name = expert.name if expert else expert_id
            utilization[name] = (count / total_selections) * 100.0
        
        return utilization
    
    def _calculate_confidence_distribution(
        self,
        decisions: List[RoutingDecision]
    ) -> Dict[str, int]:
        """Calculate confidence score distribution."""
        if not decisions:
            return {}
        
        # Bin confidence scores
        bins = {
            "0-20": 0,
            "20-40": 0,
            "40-60": 0,
            "60-80": 0,
            "80-100": 0
        }
        
        for decision in decisions:
            confidence = decision.routing_confidence
            if confidence < 20:
                bins["0-20"] += 1
            elif confidence < 40:
                bins["20-40"] += 1
            elif confidence < 60:
                bins["40-60"] += 1
            elif confidence < 80:
                bins["60-80"] += 1
            else:
                bins["80-100"] += 1
        
        return bins
    
    def _get_top_performing_experts(
        self,
        experts: Dict[str, Expert],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top performing experts."""
        expert_list = list(experts.values())
        
        # Sort by success rate and confidence
        expert_list.sort(
            key=lambda e: (e.performance.success_rate, e.confidence_score),
            reverse=True
        )
        
        top_experts = []
        for expert in expert_list[:top_n]:
            top_experts.append({
                "name": expert.name,
                "id": expert.id,
                "success_rate": expert.performance.success_rate,
                "confidence_score": expert.confidence_score,
                "total_requests": expert.performance.total_requests,
                "average_latency_ms": expert.performance.average_latency_ms,
            })
        
        return top_experts
    
    def _generate_routing_heatmap(
        self,
        decisions: List[RoutingDecision],
        experts: Dict[str, Expert]
    ) -> List[List[float]]:
        """Generate routing heatmap data (domain x expert)."""
        if not decisions or not experts:
            return []
        
        # Get unique domains and experts
        domains = sorted(set(d.domain for d in decisions))
        expert_ids = sorted(experts.keys())
        
        # Initialize heatmap matrix
        heatmap = [[0.0 for _ in expert_ids] for _ in domains]
        
        # Count routing decisions
        for decision in decisions:
            domain_idx = domains.index(decision.domain)
            for expert_id in decision.selected_expert_ids:
                if expert_id in expert_ids:
                    expert_idx = expert_ids.index(expert_id)
                    heatmap[domain_idx][expert_idx] += 1.0
        
        # Normalize by row (domain)
        for i, row in enumerate(heatmap):
            row_sum = sum(row)
            if row_sum > 0:
                heatmap[i] = [val / row_sum for val in row]
        
        return heatmap
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics summary.
        
        Returns:
            Dictionary with analytics summary
        """
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "history_size": 0,
                "oldest_decision": None,
                "newest_decision": None,
            }
        
        return {
            "total_decisions": len(self.decision_history),
            "history_size": self.max_history_size,
            "oldest_decision": self.decision_history[0].timestamp.isoformat(),
            "newest_decision": self.decision_history[-1].timestamp.isoformat(),
            "average_latency_ms": sum(d.latency_ms for d in self.decision_history) / len(self.decision_history),
            "average_confidence": sum(d.routing_confidence for d in self.decision_history) / len(self.decision_history),
        }
