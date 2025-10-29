"""
Integration module for API usage tracking and cost management.

This module integrates all the usage tracking, cost management, performance analysis,
and signup management components with the existing API orchestrator.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from quantum_moe_mas.api.orchestrator import API_Orchestrator, APIProvider, APIResponse
from quantum_moe_mas.api.usage_tracker import UsageTracker, UsageMetricType, TimeWindow
from quantum_moe_mas.api.cost_manager import CostManager, BudgetPeriod, AlertType
from quantum_moe_mas.api.performance_analyzer import PerformanceAnalyzer, PerformanceMetric
from quantum_moe_mas.api.signup_manager import SignupManager, ProviderCategory
from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


@dataclass
class IntegratedAPIMetrics:
    """Comprehensive API metrics combining all tracking systems."""
    
    provider: str
    timestamp: datetime
    
    # Usage metrics
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    
    # Cost metrics
    total_cost: float
    cost_per_request: float
    cost_per_token: float
    
    # Performance metrics
    avg_latency_ms: float
    success_rate: float
    throughput_rps: float
    
    # Budget status
    budget_utilization: Dict[str, float]  # budget_name -> percentage
    active_alerts: List[str]
    
    # Optimization recommendations
    recommendations: List[str]


class IntegratedAPIManager:
    """
    Integrated API management system combining orchestration, tracking, and optimization.
    
    This class serves as the main interface for all API-related operations,
    integrating usage tracking, cost management, performance analysis, and signup management.
    """
    
    def __init__(
        self,
        orchestrator: Optional[API_Orchestrator] = None,
        usage_tracker: Optional[UsageTracker] = None,
        cost_manager: Optional[CostManager] = None,
        performance_analyzer: Optional[PerformanceAnalyzer] = None,
        signup_manager: Optional[SignupManager] = None
    ) -> None:
        """
        Initialize integrated API manager.
        
        Args:
            orchestrator: API orchestrator instance
            usage_tracker: Usage tracker instance
            cost_manager: Cost manager instance
            performance_analyzer: Performance analyzer instance
            signup_manager: Signup manager instance
        """
        # Initialize components
        self.orchestrator = orchestrator or API_Orchestrator()
        self.usage_tracker = usage_tracker or UsageTracker()
        self.cost_manager = cost_manager or CostManager()
        self.performance_analyzer = performance_analyzer or PerformanceAnalyzer()
        self.signup_manager = signup_manager or SignupManager()
        
        # Integration state
        self.is_running = False
        self._integration_task: Optional[asyncio.Task] = None
        
        # Setup cost alert callback
        self.cost_manager.add_alert_callback(self._handle_budget_alert)
        
        logger.info("Initialized IntegratedAPIManager")
    
    async def initialize(self) -> None:
        """Initialize all components and start integration."""
        try:
            # Initialize all components
            await self.orchestrator.initialize()
            await self.usage_tracker.initialize()
            await self.cost_manager.initialize()
            await self.performance_analyzer.initialize()
            await self.signup_manager.initialize()
            
            # Start integration loop
            self._integration_task = asyncio.create_task(self._integration_loop())
            
            self.is_running = True
            logger.info("IntegratedAPIManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize IntegratedAPIManager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown all components."""
        self.is_running = False
        
        if self._integration_task:
            self._integration_task.cancel()
            try:
                await self._integration_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown components
        await self.orchestrator.shutdown()
        await self.usage_tracker.shutdown()
        await self.cost_manager.shutdown()
        await self.performance_analyzer.shutdown()
        await self.signup_manager.shutdown()
        
        logger.info("IntegratedAPIManager shutdown complete")
    
    async def make_tracked_request(
        self,
        provider: APIProvider,
        request_data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> APIResponse:
        """
        Make an API request with comprehensive tracking.
        
        Args:
            provider: API provider
            request_data: Request payload
            user_id: Optional user identifier
            session_id: Optional session identifier
            **kwargs: Additional parameters for orchestrator
        
        Returns:
            APIResponse with tracking metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Make the request through orchestrator
            response = await self.orchestrator.make_request(provider, request_data, **kwargs)
            
            # Calculate metrics
            end_time = datetime.utcnow()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            # Extract usage information
            tokens_used = response.metadata.get("tokens_used", 0)
            cost = response.metadata.get("cost", 0.0)
            request_size = len(str(request_data).encode('utf-8'))
            response_size = len(str(response.data).encode('utf-8')) if response.data else 0
            
            # Track usage
            await self.usage_tracker.track_usage(
                provider=provider.value,
                endpoint=kwargs.get("endpoint", "/"),
                method=kwargs.get("method", "POST"),
                success=response.success,
                status_code=response.status_code,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                cost=cost,
                request_size_bytes=request_size,
                response_size_bytes=response_size,
                user_id=user_id,
                session_id=session_id,
                metadata=response.metadata
            )
            
            # Track cost
            await self.cost_manager.track_cost(
                provider=provider.value,
                cost=cost,
                input_tokens=tokens_used,  # Simplified
                output_tokens=0,  # Would need to be calculated properly
                requests=1,
                timestamp=start_time
            )
            
            # Record performance snapshot (periodically)
            if self._should_record_performance_snapshot(provider.value):
                usage_data = await self.usage_tracker.get_realtime_stats(provider.value)
                if usage_data and provider.value in usage_data:
                    await self.performance_analyzer.record_performance_snapshot(
                        provider.value,
                        usage_data[provider.value]
                    )
            
            return response
            
        except Exception as e:
            # Track failed request
            end_time = datetime.utcnow()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            await self.usage_tracker.track_usage(
                provider=provider.value,
                endpoint=kwargs.get("endpoint", "/"),
                method=kwargs.get("method", "POST"),
                success=False,
                status_code=None,
                latency_ms=latency_ms,
                tokens_used=0,
                cost=0.0,
                request_size_bytes=len(str(request_data).encode('utf-8')),
                response_size_bytes=0,
                user_id=user_id,
                session_id=session_id,
                metadata={"error": str(e)}
            )
            
            raise
    
    def _should_record_performance_snapshot(self, provider: str) -> bool:
        """Determine if we should record a performance snapshot."""
        # Record snapshot every 100 requests or every 15 minutes
        # This is a simplified implementation
        return hash(provider + str(datetime.utcnow().minute)) % 100 == 0
    
    async def get_comprehensive_metrics(
        self,
        provider: Optional[str] = None,
        time_window: TimeWindow = TimeWindow.HOUR
    ) -> Dict[str, IntegratedAPIMetrics]:
        """
        Get comprehensive metrics combining all tracking systems.
        
        Args:
            provider: Optional provider filter
            time_window: Time window for metrics
        
        Returns:
            Dictionary of integrated metrics by provider
        """
        metrics = {}
        
        # Get usage metrics
        usage_stats = await self.usage_tracker.get_realtime_stats(provider)
        
        # Get cost information
        current_spending = await self.cost_manager.get_current_spend(provider=provider)
        
        # Get budget alerts
        budget_alerts = await self.cost_manager.check_budget_alerts()
        
        # Process each provider
        providers_to_process = [provider] if provider else list(usage_stats.keys())
        
        for prov in providers_to_process:
            if prov not in usage_stats:
                continue
            
            usage_data = usage_stats[prov]
            
            # Calculate cost metrics
            total_cost = sum(
                cost for key, cost in current_spending.items()
                if key.startswith(prov)
            )
            
            cost_per_request = (
                total_cost / usage_data["total_requests"]
                if usage_data["total_requests"] > 0 else 0.0
            )
            
            cost_per_token = (
                total_cost / usage_data["total_tokens"]
                if usage_data.get("total_tokens", 0) > 0 else 0.0
            )
            
            # Get budget utilization
            budget_utilization = {}
            for budget_name, budget in self.cost_manager.budgets.items():
                if budget.provider == prov or budget.provider is None:
                    current_spend = current_spending.get(budget_name, 0.0)
                    utilization = (current_spend / budget.amount) * 100 if budget.amount > 0 else 0.0
                    budget_utilization[budget_name] = utilization
            
            # Get active alerts for this provider
            active_alerts = [
                alert.message for alert in budget_alerts
                if alert.provider == prov
            ]
            
            # Get optimization recommendations
            recommendations = await self.performance_analyzer.generate_optimization_recommendations(
                prov, timedelta(hours=6)
            )
            recommendation_titles = [rec.recommendation_title for rec in recommendations[:3]]
            
            # Create integrated metrics
            integrated_metrics = IntegratedAPIMetrics(
                provider=prov,
                timestamp=datetime.utcnow(),
                total_requests=usage_data["total_requests"],
                successful_requests=usage_data["successful_requests"],
                failed_requests=usage_data["failed_requests"],
                total_tokens=usage_data.get("total_tokens", 0),
                total_cost=total_cost,
                cost_per_request=cost_per_request,
                cost_per_token=cost_per_token,
                avg_latency_ms=usage_data.get("average_latency_ms", 0.0),
                success_rate=usage_data.get("success_rate", 0.0),
                throughput_rps=usage_data["total_requests"] / 3600,  # Simplified
                budget_utilization=budget_utilization,
                active_alerts=active_alerts,
                recommendations=recommendation_titles
            )
            
            metrics[prov] = integrated_metrics
        
        return metrics
    
    async def get_cost_optimization_report(
        self,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive cost optimization report.
        
        Args:
            provider: Optional provider filter
        
        Returns:
            Cost optimization report
        """
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "provider_filter": provider,
            "summary": {},
            "recommendations": [],
            "budget_status": {},
            "performance_insights": {}
        }
        
        # Get current spending
        current_spending = await self.cost_manager.get_current_spend(provider=provider)
        
        # Get cost breakdown
        cost_breakdown = await self.cost_manager.get_cost_breakdown(
            provider=provider,
            start_time=datetime.utcnow() - timedelta(days=7)
        )
        
        # Calculate summary
        total_cost = sum(current_spending.values())
        total_requests = 0
        
        usage_stats = await self.usage_tracker.get_realtime_stats(provider)
        for stats in usage_stats.values():
            total_requests += stats["total_requests"]
        
        report["summary"] = {
            "total_cost": total_cost,
            "total_requests": total_requests,
            "average_cost_per_request": total_cost / total_requests if total_requests > 0 else 0.0,
            "cost_breakdown_by_provider": {
                breakdown.provider: breakdown.total_cost
                for breakdown in cost_breakdown
            }
        }
        
        # Get optimization recommendations
        providers_to_analyze = [provider] if provider else list(usage_stats.keys())
        
        all_recommendations = []
        for prov in providers_to_analyze:
            prov_recommendations = await self.performance_analyzer.generate_optimization_recommendations(
                prov, timedelta(days=3)
            )
            all_recommendations.extend(prov_recommendations)
        
        # Sort by potential savings
        all_recommendations.sort(key=lambda r: r.potential_savings, reverse=True)
        
        report["recommendations"] = [
            {
                "provider": rec.provider,
                "title": rec.recommendation_title,
                "description": rec.recommendation_description,
                "potential_savings": rec.potential_savings,
                "confidence_score": rec.confidence_score,
                "implementation_effort": rec.implementation_effort,
                "priority": rec.priority.value
            }
            for rec in all_recommendations[:10]  # Top 10
        ]
        
        # Budget status
        budget_alerts = await self.cost_manager.check_budget_alerts()
        report["budget_status"] = {
            "active_alerts": len(budget_alerts),
            "budgets": {
                name: {
                    "amount": budget.amount,
                    "current_spend": current_spending.get(name, 0.0),
                    "utilization_percent": (current_spending.get(name, 0.0) / budget.amount) * 100 if budget.amount > 0 else 0.0
                }
                for name, budget in self.cost_manager.budgets.items()
            }
        }
        
        return report
    
    async def get_provider_comparison_report(
        self,
        providers: List[str],
        metrics: List[PerformanceMetric] = None
    ) -> Dict[str, Any]:
        """
        Generate provider comparison report.
        
        Args:
            providers: List of providers to compare
            metrics: List of metrics to compare
        
        Returns:
            Provider comparison report
        """
        if metrics is None:
            metrics = [
                PerformanceMetric.LATENCY,
                PerformanceMetric.SUCCESS_RATE,
                PerformanceMetric.COST_EFFICIENCY
            ]
        
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "providers": providers,
            "metrics": [m.value for m in metrics],
            "comparisons": {},
            "recommendations": []
        }
        
        # Perform comparisons for each metric
        for metric in metrics:
            try:
                comparison = await self.performance_analyzer.compare_providers(
                    providers, metric, timedelta(hours=24)
                )
                
                report["comparisons"][metric.value] = {
                    "rankings": comparison.rankings,
                    "best_provider": comparison.best_provider,
                    "worst_provider": comparison.worst_provider,
                    "mean_value": comparison.mean_value,
                    "recommendations": comparison.recommendations
                }
                
                # Add to overall recommendations
                report["recommendations"].extend(comparison.recommendations)
                
            except Exception as e:
                logger.error(f"Error comparing providers on {metric.value}: {e}")
                report["comparisons"][metric.value] = {"error": str(e)}
        
        return report
    
    def _handle_budget_alert(self, alert) -> None:
        """Handle budget alert callback."""
        logger.warning(f"Budget alert: {alert.message}")
        
        # Could implement additional actions here:
        # - Send notifications
        # - Auto-disable providers
        # - Switch to cheaper alternatives
        # - etc.
    
    async def _integration_loop(self) -> None:
        """Background integration loop for periodic tasks."""
        while self.is_running:
            try:
                # Perform periodic integration tasks
                await self._sync_performance_data()
                await self._cleanup_old_data()
                
                # Wait for next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in integration loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _sync_performance_data(self) -> None:
        """Sync performance data between components."""
        try:
            # Get usage stats for all providers
            usage_stats = await self.usage_tracker.get_realtime_stats()
            
            # Record performance snapshots for active providers
            for provider, stats in usage_stats.items():
                if stats["total_requests"] > 0:
                    await self.performance_analyzer.record_performance_snapshot(
                        provider, stats, time_window_minutes=15
                    )
            
        except Exception as e:
            logger.error(f"Error syncing performance data: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Cleanup old data from all components."""
        try:
            # This would implement cleanup logic for old data
            # to prevent database growth
            pass
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive integration status.
        
        Returns:
            Dictionary with integration status
        """
        return {
            "is_running": self.is_running,
            "components": {
                "orchestrator": self.orchestrator.get_orchestrator_status(),
                "usage_tracker": self.usage_tracker.get_tracker_status(),
                "cost_manager": self.cost_manager.get_cost_manager_status(),
                "performance_analyzer": self.performance_analyzer.get_analyzer_status(),
                "signup_manager": self.signup_manager.get_signup_manager_status()
            }
        }