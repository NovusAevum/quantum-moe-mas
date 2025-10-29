"""
ROI Calculation Engine

Comprehensive ROI calculation engine with ICM/u target of $0.50+ per session.
Provides efficiency gain measurement, cost-benefit analysis, and financial impact tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import json

logger = logging.getLogger(__name__)

@dataclass
class SessionMetrics:
    """Metrics for a single user session"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    queries_processed: int = 0
    tokens_used: int = 0
    api_calls_made: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    response_time_avg: float = 0.0
    user_satisfaction: Optional[float] = None
    conversion_events: List[str] = field(default_factory=list)
    revenue_attributed: Decimal = field(default_factory=lambda: Decimal('0.00'))

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for analysis"""
    api_costs: Decimal = field(default_factory=lambda: Decimal('0.00'))
    infrastructure_costs: Decimal = field(default_factory=lambda: Decimal('0.00'))
    compute_costs: Decimal = field(default_factory=lambda: Decimal('0.00'))
    storage_costs: Decimal = field(default_factory=lambda: Decimal('0.00'))
    bandwidth_costs: Decimal = field(default_factory=lambda: Decimal('0.00'))
    total_costs: Decimal = field(default_factory=lambda: Decimal('0.00'))

@dataclass
class ROIMetrics:
    """Comprehensive ROI metrics"""
    session_id: str
    icm_per_unit: Decimal  # Incremental Contribution Margin per Unit
    total_revenue: Decimal
    total_costs: Decimal
    gross_profit: Decimal
    roi_percentage: Decimal
    efficiency_gain: Decimal
    cost_savings: Decimal
    time_savings_hours: Decimal
    user_satisfaction_score: Optional[float]
    conversion_rate: Optional[float]
    cost_breakdown: CostBreakdown
    calculated_at: datetime = field(default_factory=datetime.utcnow)

class ROICalculationEngine:
    """
    Advanced ROI calculation engine with comprehensive business metrics tracking.
    
    Features:
    - Real-time ROI calculation with $0.50+ ICM/u target
    - Efficiency gain measurement with baseline comparisons
    - Cost-benefit analysis across all operations
    - Revenue attribution and conversion tracking
    - Financial impact dashboards with real-time cost tracking
    """
    
    def __init__(self):
        self.sessions: Dict[str, SessionMetrics] = {}
        self.roi_history: List[ROIMetrics] = []
        self.baseline_metrics: Dict[str, float] = {
            'avg_response_time': 5.0,  # seconds
            'avg_api_cost_per_query': 0.02,  # USD
            'avg_user_satisfaction': 3.5,  # out of 5
            'baseline_conversion_rate': 0.05  # 5%
        }
        self.cost_rates = {
            'api_cost_per_token': Decimal('0.000002'),  # $0.000002 per token
            'compute_cost_per_second': Decimal('0.0001'),  # $0.0001 per second
            'storage_cost_per_gb_hour': Decimal('0.00005'),  # $0.00005 per GB-hour
            'bandwidth_cost_per_gb': Decimal('0.01')  # $0.01 per GB
        }
        
    async def start_session(self, session_id: str, user_id: str) -> SessionMetrics:
        """Start tracking a new user session"""
        session = SessionMetrics(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.utcnow()
        )
        self.sessions[session_id] = session
        logger.info(f"Started ROI tracking for session {session_id}")
        return session
    
    async def update_session_metrics(
        self,
        session_id: str,
        queries_processed: int = 0,
        tokens_used: int = 0,
        api_calls_made: int = 0,
        cache_hits: int = 0,
        cache_misses: int = 0,
        response_time: float = 0.0,
        conversion_event: Optional[str] = None,
        revenue_attributed: Optional[Decimal] = None
    ) -> None:
        """Update session metrics with new data"""
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return
            
        session = self.sessions[session_id]
        session.queries_processed += queries_processed
        session.tokens_used += tokens_used
        session.api_calls_made += api_calls_made
        session.cache_hits += cache_hits
        session.cache_misses += cache_misses
        
        # Update average response time
        if response_time > 0:
            total_queries = session.queries_processed
            if total_queries > 0:
                session.response_time_avg = (
                    (session.response_time_avg * (total_queries - 1) + response_time) / total_queries
                )
        
        # Track conversion events
        if conversion_event:
            session.conversion_events.append(conversion_event)
            
        # Add attributed revenue
        if revenue_attributed:
            session.revenue_attributed += revenue_attributed
            
        logger.debug(f"Updated metrics for session {session_id}")
    
    async def end_session(self, session_id: str, user_satisfaction: Optional[float] = None) -> ROIMetrics:
        """End session tracking and calculate final ROI metrics"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.sessions[session_id]
        session.end_time = datetime.utcnow()
        session.user_satisfaction = user_satisfaction
        
        # Calculate ROI metrics
        roi_metrics = await self._calculate_roi_metrics(session)
        self.roi_history.append(roi_metrics)
        
        # Clean up session
        del self.sessions[session_id]
        
        logger.info(f"Completed ROI calculation for session {session_id}: ICM/u = ${roi_metrics.icm_per_unit}")
        return roi_metrics
    
    async def _calculate_roi_metrics(self, session: SessionMetrics) -> ROIMetrics:
        """Calculate comprehensive ROI metrics for a session"""
        # Calculate session duration
        duration = (session.end_time - session.start_time).total_seconds()
        
        # Calculate costs
        cost_breakdown = await self._calculate_cost_breakdown(session, duration)
        
        # Calculate revenue (attributed + estimated value)
        total_revenue = session.revenue_attributed
        if not total_revenue:
            # Estimate revenue based on user engagement and conversion potential
            estimated_value = await self._estimate_session_value(session)
            total_revenue = estimated_value
        
        # Calculate gross profit
        gross_profit = total_revenue - cost_breakdown.total_costs
        
        # Calculate ROI percentage
        roi_percentage = Decimal('0')
        if cost_breakdown.total_costs > 0:
            roi_percentage = (gross_profit / cost_breakdown.total_costs * 100).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP
            )
        
        # Calculate ICM/u (Incremental Contribution Margin per Unit)
        icm_per_unit = gross_profit  # Per session in this case
        
        # Calculate efficiency gains
        efficiency_gain = await self._calculate_efficiency_gain(session)
        
        # Calculate cost savings
        cost_savings = await self._calculate_cost_savings(session)
        
        # Calculate time savings
        time_savings_hours = await self._calculate_time_savings(session, duration)
        
        # Calculate conversion rate
        conversion_rate = None
        if session.queries_processed > 0:
            conversion_rate = len(session.conversion_events) / session.queries_processed
        
        return ROIMetrics(
            session_id=session.session_id,
            icm_per_unit=icm_per_unit,
            total_revenue=total_revenue,
            total_costs=cost_breakdown.total_costs,
            gross_profit=gross_profit,
            roi_percentage=roi_percentage,
            efficiency_gain=efficiency_gain,
            cost_savings=cost_savings,
            time_savings_hours=time_savings_hours,
            user_satisfaction_score=session.user_satisfaction,
            conversion_rate=conversion_rate,
            cost_breakdown=cost_breakdown
        )
    
    async def _calculate_cost_breakdown(self, session: SessionMetrics, duration: float) -> CostBreakdown:
        """Calculate detailed cost breakdown for a session"""
        # API costs based on tokens used
        api_costs = Decimal(str(session.tokens_used)) * self.cost_rates['api_cost_per_token']
        
        # Compute costs based on processing time
        compute_costs = Decimal(str(duration)) * self.cost_rates['compute_cost_per_second']
        
        # Storage costs (estimated based on session data)
        estimated_storage_gb = Decimal('0.001')  # 1MB per session
        storage_costs = estimated_storage_gb * self.cost_rates['storage_cost_per_gb_hour'] * Decimal(str(duration / 3600))
        
        # Bandwidth costs (estimated)
        estimated_bandwidth_gb = Decimal('0.01')  # 10MB per session
        bandwidth_costs = estimated_bandwidth_gb * self.cost_rates['bandwidth_cost_per_gb']
        
        # Infrastructure costs (allocated portion)
        infrastructure_costs = Decimal('0.05')  # Fixed $0.05 per session
        
        total_costs = api_costs + compute_costs + storage_costs + bandwidth_costs + infrastructure_costs
        
        return CostBreakdown(
            api_costs=api_costs,
            infrastructure_costs=infrastructure_costs,
            compute_costs=compute_costs,
            storage_costs=storage_costs,
            bandwidth_costs=bandwidth_costs,
            total_costs=total_costs
        )
    
    async def _estimate_session_value(self, session: SessionMetrics) -> Decimal:
        """Estimate the business value of a session based on engagement metrics"""
        base_value = Decimal('1.00')  # Base $1.00 per session
        
        # Value multipliers based on engagement
        query_multiplier = min(Decimal(str(session.queries_processed)) * Decimal('0.10'), Decimal('2.00'))
        
        # Satisfaction multiplier
        satisfaction_multiplier = Decimal('1.00')
        if session.user_satisfaction:
            satisfaction_multiplier = Decimal(str(session.user_satisfaction / 5.0))
        
        # Cache efficiency multiplier (rewards efficient usage)
        cache_efficiency = Decimal('1.00')
        total_requests = session.cache_hits + session.cache_misses
        if total_requests > 0:
            cache_hit_rate = session.cache_hits / total_requests
            cache_efficiency = Decimal('1.00') + Decimal(str(cache_hit_rate * 0.5))
        
        # Response time multiplier (rewards fast responses)
        response_time_multiplier = Decimal('1.00')
        if session.response_time_avg > 0:
            baseline_time = self.baseline_metrics['avg_response_time']
            if session.response_time_avg < baseline_time:
                improvement = (baseline_time - session.response_time_avg) / baseline_time
                response_time_multiplier = Decimal('1.00') + Decimal(str(improvement * 0.3))
        
        estimated_value = (
            base_value * query_multiplier * satisfaction_multiplier * 
            cache_efficiency * response_time_multiplier
        ).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        return estimated_value
    
    async def _calculate_efficiency_gain(self, session: SessionMetrics) -> Decimal:
        """Calculate efficiency gain compared to baseline"""
        if session.queries_processed == 0:
            return Decimal('0.00')
        
        # Calculate cache efficiency gain
        total_requests = session.cache_hits + session.cache_misses
        cache_hit_rate = 0.0
        if total_requests > 0:
            cache_hit_rate = session.cache_hits / total_requests
        
        # Baseline assumes 20% cache hit rate
        baseline_cache_rate = 0.20
        cache_efficiency_gain = max(0, cache_hit_rate - baseline_cache_rate)
        
        # Calculate response time efficiency
        baseline_response_time = self.baseline_metrics['avg_response_time']
        response_time_gain = 0.0
        if session.response_time_avg > 0:
            response_time_gain = max(0, (baseline_response_time - session.response_time_avg) / baseline_response_time)
        
        # Combined efficiency gain as percentage
        total_efficiency_gain = (cache_efficiency_gain + response_time_gain) * 100
        
        return Decimal(str(total_efficiency_gain)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    async def _calculate_cost_savings(self, session: SessionMetrics) -> Decimal:
        """Calculate cost savings achieved through optimization"""
        if session.queries_processed == 0:
            return Decimal('0.00')
        
        # Calculate savings from cache hits (avoided API calls)
        api_cost_per_query = Decimal(str(self.baseline_metrics['avg_api_cost_per_query']))
        cache_savings = Decimal(str(session.cache_hits)) * api_cost_per_query
        
        # Calculate savings from efficient routing (MoE optimization)
        # Assume 30% cost reduction from optimal expert selection
        baseline_cost = Decimal(str(session.queries_processed)) * api_cost_per_query
        moe_savings = baseline_cost * Decimal('0.30')
        
        total_savings = cache_savings + moe_savings
        
        return total_savings.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    async def _calculate_time_savings(self, session: SessionMetrics, duration: float) -> Decimal:
        """Calculate time savings in hours"""
        if session.queries_processed == 0:
            return Decimal('0.00')
        
        # Baseline time per query (manual process)
        baseline_time_per_query = 300  # 5 minutes per query manually
        baseline_total_time = session.queries_processed * baseline_time_per_query
        
        # Actual time used
        actual_time = duration
        
        # Time savings in hours
        time_saved_seconds = max(0, baseline_total_time - actual_time)
        time_saved_hours = time_saved_seconds / 3600
        
        return Decimal(str(time_saved_hours)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    async def get_aggregate_roi_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, any]:
        """Get aggregated ROI metrics for a time period"""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Filter ROI history by date range
        filtered_metrics = [
            roi for roi in self.roi_history
            if start_date <= roi.calculated_at <= end_date
        ]
        
        if not filtered_metrics:
            return {
                'total_sessions': 0,
                'avg_icm_per_unit': Decimal('0.00'),
                'total_revenue': Decimal('0.00'),
                'total_costs': Decimal('0.00'),
                'avg_roi_percentage': Decimal('0.00'),
                'total_cost_savings': Decimal('0.00'),
                'total_time_savings_hours': Decimal('0.00'),
                'avg_user_satisfaction': 0.0,
                'avg_conversion_rate': 0.0
            }
        
        # Calculate aggregates
        total_sessions = len(filtered_metrics)
        total_revenue = sum(roi.total_revenue for roi in filtered_metrics)
        total_costs = sum(roi.total_costs for roi in filtered_metrics)
        avg_icm_per_unit = total_revenue / Decimal(str(total_sessions)) if total_sessions > 0 else Decimal('0.00')
        avg_roi_percentage = sum(roi.roi_percentage for roi in filtered_metrics) / Decimal(str(total_sessions))
        total_cost_savings = sum(roi.cost_savings for roi in filtered_metrics)
        total_time_savings = sum(roi.time_savings_hours for roi in filtered_metrics)
        
        # Calculate satisfaction and conversion averages
        satisfaction_scores = [roi.user_satisfaction_score for roi in filtered_metrics if roi.user_satisfaction_score is not None]
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0.0
        
        conversion_rates = [roi.conversion_rate for roi in filtered_metrics if roi.conversion_rate is not None]
        avg_conversion_rate = sum(conversion_rates) / len(conversion_rates) if conversion_rates else 0.0
        
        return {
            'total_sessions': total_sessions,
            'avg_icm_per_unit': avg_icm_per_unit.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            'total_revenue': total_revenue.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            'total_costs': total_costs.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            'avg_roi_percentage': avg_roi_percentage.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            'total_cost_savings': total_cost_savings.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            'total_time_savings_hours': total_time_savings.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            'avg_user_satisfaction': round(avg_satisfaction, 2),
            'avg_conversion_rate': round(avg_conversion_rate, 4),
            'target_icm_achieved': avg_icm_per_unit >= Decimal('0.50')
        }
    
    async def export_roi_data(self, format: str = 'json') -> str:
        """Export ROI data in specified format"""
        data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'total_sessions_tracked': len(self.roi_history),
            'roi_metrics': []
        }
        
        for roi in self.roi_history:
            roi_data = {
                'session_id': roi.session_id,
                'icm_per_unit': float(roi.icm_per_unit),
                'total_revenue': float(roi.total_revenue),
                'total_costs': float(roi.total_costs),
                'gross_profit': float(roi.gross_profit),
                'roi_percentage': float(roi.roi_percentage),
                'efficiency_gain': float(roi.efficiency_gain),
                'cost_savings': float(roi.cost_savings),
                'time_savings_hours': float(roi.time_savings_hours),
                'user_satisfaction_score': roi.user_satisfaction_score,
                'conversion_rate': roi.conversion_rate,
                'calculated_at': roi.calculated_at.isoformat(),
                'cost_breakdown': {
                    'api_costs': float(roi.cost_breakdown.api_costs),
                    'infrastructure_costs': float(roi.cost_breakdown.infrastructure_costs),
                    'compute_costs': float(roi.cost_breakdown.compute_costs),
                    'storage_costs': float(roi.cost_breakdown.storage_costs),
                    'bandwidth_costs': float(roi.cost_breakdown.bandwidth_costs),
                    'total_costs': float(roi.cost_breakdown.total_costs)
                }
            }
            data['roi_metrics'].append(roi_data)
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")