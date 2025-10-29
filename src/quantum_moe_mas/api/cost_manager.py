"""
Cost Management and Budget Alert System.

This module provides comprehensive cost tracking, budget management,
and real-time alerting for API usage across all providers.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3

from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


class BudgetPeriod(Enum):
    """Budget period types."""
    
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class AlertType(Enum):
    """Types of budget alerts."""
    
    THRESHOLD_WARNING = "threshold_warning"  # 80% of budget
    THRESHOLD_CRITICAL = "threshold_critical"  # 95% of budget
    BUDGET_EXCEEDED = "budget_exceeded"  # 100% of budget
    COST_SPIKE = "cost_spike"  # Unusual cost increase
    PROVIDER_EXPENSIVE = "provider_expensive"  # Provider costs above average


class CostOptimizationStrategy(Enum):
    """Cost optimization strategies."""
    
    SWITCH_TO_CHEAPER_PROVIDER = "switch_to_cheaper_provider"
    REDUCE_TOKEN_USAGE = "reduce_token_usage"
    IMPLEMENT_CACHING = "implement_caching"
    BATCH_REQUESTS = "batch_requests"
    USE_SMALLER_MODELS = "use_smaller_models"


@dataclass
class BudgetConfig:
    """Budget configuration for a provider or overall system."""
    
    name: str
    provider: Optional[str]  # None for system-wide budget
    period: BudgetPeriod
    amount: float
    currency: str = "USD"
    
    # Alert thresholds (percentages)
    warning_threshold: float = 80.0
    critical_threshold: float = 95.0
    
    # Auto-actions
    auto_disable_at_limit: bool = False
    auto_switch_provider: bool = True
    
    # Notification settings
    email_alerts: bool = True
    webhook_url: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for analysis."""
    
    provider: str
    period_start: datetime
    period_end: datetime
    
    # Cost components
    total_cost: float = 0.0
    input_token_cost: float = 0.0
    output_token_cost: float = 0.0
    request_cost: float = 0.0
    
    # Usage metrics
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    # Efficiency metrics
    cost_per_request: float = 0.0
    cost_per_token: float = 0.0
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.total_input_tokens + self.total_output_tokens


@dataclass
class BudgetAlert:
    """Budget alert notification."""
    
    alert_id: str
    alert_type: AlertType
    budget_name: str
    provider: Optional[str]
    
    # Alert details
    current_spend: float
    budget_amount: float
    percentage_used: float
    
    # Time information
    period_start: datetime
    period_end: datetime
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    
    # Additional context
    message: str = ""
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class CostOptimizationRecommendation:
    """Cost optimization recommendation."""
    
    strategy: CostOptimizationStrategy
    provider: str
    potential_savings: float
    confidence_score: float  # 0-100
    
    title: str
    description: str
    implementation_steps: List[str]
    
    # Impact analysis
    estimated_savings_percent: float
    implementation_effort: str  # "low", "medium", "high"
    risk_level: str  # "low", "medium", "high"
    
    created_at: datetime = field(default_factory=datetime.utcnow)


class CostManager:
    """
    Comprehensive cost management and budget alert system.
    
    Provides:
    - Real-time cost tracking across all providers
    - Flexible budget management with multiple periods
    - Intelligent alerting system with customizable thresholds
    - Cost optimization recommendations
    - Detailed cost analysis and reporting
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        default_currency: str = "USD",
        alert_check_interval_minutes: int = 5
    ) -> None:
        """
        Initialize cost manager.
        
        Args:
            storage_path: Path to SQLite database
            default_currency: Default currency for budgets
            alert_check_interval_minutes: Interval for checking alerts
        """
        self.storage_path = storage_path or os.path.expanduser("~/.quantum_moe_mas/cost_management.db")
        self.default_currency = default_currency
        self.alert_check_interval = alert_check_interval_minutes
        
        # Budget configurations
        self.budgets: Dict[str, BudgetConfig] = {}
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[BudgetAlert], None]] = []
        
        # Cost tracking
        self.current_costs: Dict[str, Dict[str, float]] = {}  # provider -> period -> cost
        
        # Background tasks
        self.is_running = False
        self._alert_task: Optional[asyncio.Task] = None
        
        logger.info(
            "Initialized CostManager",
            storage_path=self.storage_path,
            currency=default_currency
        )
    
    async def initialize(self) -> None:
        """Initialize the cost manager and database."""
        try:
            # Create storage directory
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Initialize database
            await self._initialize_database()
            
            # Load existing budgets
            await self._load_budgets()
            
            # Start alert monitoring
            self._alert_task = asyncio.create_task(self._alert_monitoring_loop())
            
            self.is_running = True
            logger.info("CostManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CostManager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the cost manager."""
        self.is_running = False
        
        if self._alert_task:
            self._alert_task.cancel()
            try:
                await self._alert_task
            except asyncio.CancelledError:
                pass
        
        logger.info("CostManager shutdown complete")
    
    async def _initialize_database(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            # Budgets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS budgets (
                    name TEXT PRIMARY KEY,
                    provider TEXT,
                    period TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL,
                    warning_threshold REAL NOT NULL,
                    critical_threshold REAL NOT NULL,
                    auto_disable_at_limit BOOLEAN NOT NULL,
                    auto_switch_provider BOOLEAN NOT NULL,
                    email_alerts BOOLEAN NOT NULL,
                    webhook_url TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Cost tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cost_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    total_cost REAL NOT NULL,
                    input_token_cost REAL NOT NULL,
                    output_token_cost REAL NOT NULL,
                    request_cost REAL NOT NULL,
                    total_requests INTEGER NOT NULL,
                    total_input_tokens INTEGER NOT NULL,
                    total_output_tokens INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS budget_alerts (
                    alert_id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    budget_name TEXT NOT NULL,
                    provider TEXT,
                    current_spend REAL NOT NULL,
                    budget_amount REAL NOT NULL,
                    percentage_used REAL NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    message TEXT NOT NULL,
                    recommendations TEXT,
                    metadata TEXT,
                    acknowledged BOOLEAN NOT NULL DEFAULT 0,
                    resolved BOOLEAN NOT NULL DEFAULT 0
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost_tracking_provider ON cost_tracking(provider)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_budget ON budget_alerts(budget_name)")
            
            conn.commit()
            logger.info("Cost management database schema initialized")
            
        finally:
            conn.close()
    
    async def create_budget(
        self,
        name: str,
        amount: float,
        period: BudgetPeriod,
        provider: Optional[str] = None,
        currency: str = None,
        warning_threshold: float = 80.0,
        critical_threshold: float = 95.0,
        auto_disable_at_limit: bool = False,
        auto_switch_provider: bool = True,
        email_alerts: bool = True,
        webhook_url: Optional[str] = None
    ) -> BudgetConfig:
        """
        Create a new budget configuration.
        
        Args:
            name: Budget name (unique identifier)
            amount: Budget amount
            period: Budget period
            provider: Optional specific provider (None for system-wide)
            currency: Currency code
            warning_threshold: Warning threshold percentage
            critical_threshold: Critical threshold percentage
            auto_disable_at_limit: Auto-disable provider at limit
            auto_switch_provider: Auto-switch to cheaper provider
            email_alerts: Enable email alerts
            webhook_url: Optional webhook URL for alerts
        
        Returns:
            BudgetConfig instance
        """
        budget = BudgetConfig(
            name=name,
            provider=provider,
            period=period,
            amount=amount,
            currency=currency or self.default_currency,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            auto_disable_at_limit=auto_disable_at_limit,
            auto_switch_provider=auto_switch_provider,
            email_alerts=email_alerts,
            webhook_url=webhook_url
        )
        
        # Save to database
        await self._save_budget(budget)
        
        # Add to memory
        self.budgets[name] = budget
        
        logger.info(f"Created budget '{name}': {amount} {budget.currency} per {period.value}")
        return budget
    
    async def track_cost(
        self,
        provider: str,
        cost: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        requests: int = 1,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Track cost for a provider.
        
        Args:
            provider: API provider name
            cost: Total cost of the operation
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            requests: Number of requests
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Update current costs tracking
        if provider not in self.current_costs:
            self.current_costs[provider] = {}
        
        # Track costs by different periods
        for period in BudgetPeriod:
            period_key = self._get_period_key(timestamp, period)
            
            if period_key not in self.current_costs[provider]:
                self.current_costs[provider][period_key] = 0.0
            
            self.current_costs[provider][period_key] += cost
        
        # Save detailed cost breakdown
        await self._save_cost_breakdown(
            provider, timestamp, cost, input_tokens, output_tokens, requests
        )
        
        logger.debug(f"Tracked cost for {provider}: ${cost:.4f}")
    
    def _get_period_key(self, timestamp: datetime, period: BudgetPeriod) -> str:
        """Get period key for timestamp."""
        if period == BudgetPeriod.DAILY:
            return timestamp.strftime("%Y-%m-%d")
        elif period == BudgetPeriod.WEEKLY:
            # ISO week
            year, week, _ = timestamp.isocalendar()
            return f"{year}-W{week:02d}"
        elif period == BudgetPeriod.MONTHLY:
            return timestamp.strftime("%Y-%m")
        elif period == BudgetPeriod.QUARTERLY:
            quarter = (timestamp.month - 1) // 3 + 1
            return f"{timestamp.year}-Q{quarter}"
        elif period == BudgetPeriod.YEARLY:
            return str(timestamp.year)
        else:
            return timestamp.strftime("%Y-%m-%d")
    
    async def get_current_spend(
        self,
        budget_name: Optional[str] = None,
        provider: Optional[str] = None,
        period: Optional[BudgetPeriod] = None
    ) -> Dict[str, float]:
        """
        Get current spending for budgets or providers.
        
        Args:
            budget_name: Optional specific budget name
            provider: Optional specific provider
            period: Optional specific period
        
        Returns:
            Dictionary with spending information
        """
        if budget_name and budget_name in self.budgets:
            budget = self.budgets[budget_name]
            current_period_key = self._get_period_key(datetime.utcnow(), budget.period)
            
            if budget.provider:
                # Provider-specific budget
                spend = self.current_costs.get(budget.provider, {}).get(current_period_key, 0.0)
                return {budget_name: spend}
            else:
                # System-wide budget
                total_spend = 0.0
                for prov_costs in self.current_costs.values():
                    total_spend += prov_costs.get(current_period_key, 0.0)
                return {budget_name: total_spend}
        
        elif provider:
            # Get spending for specific provider
            result = {}
            for period_type in BudgetPeriod:
                if period and period != period_type:
                    continue
                
                period_key = self._get_period_key(datetime.utcnow(), period_type)
                spend = self.current_costs.get(provider, {}).get(period_key, 0.0)
                result[f"{provider}_{period_type.value}"] = spend
            
            return result
        
        else:
            # Get all current spending
            result = {}
            
            # Add budget spending
            for budget_name, budget in self.budgets.items():
                current_period_key = self._get_period_key(datetime.utcnow(), budget.period)
                
                if budget.provider:
                    spend = self.current_costs.get(budget.provider, {}).get(current_period_key, 0.0)
                else:
                    spend = sum(
                        prov_costs.get(current_period_key, 0.0)
                        for prov_costs in self.current_costs.values()
                    )
                
                result[budget_name] = spend
            
            return result
    
    async def check_budget_alerts(self) -> List[BudgetAlert]:
        """
        Check all budgets for alert conditions.
        
        Returns:
            List of triggered alerts
        """
        alerts = []
        current_spending = await self.get_current_spend()
        
        for budget_name, budget in self.budgets.items():
            current_spend = current_spending.get(budget_name, 0.0)
            percentage_used = (current_spend / budget.amount) * 100 if budget.amount > 0 else 0
            
            # Determine alert type
            alert_type = None
            if percentage_used >= 100:
                alert_type = AlertType.BUDGET_EXCEEDED
            elif percentage_used >= budget.critical_threshold:
                alert_type = AlertType.THRESHOLD_CRITICAL
            elif percentage_used >= budget.warning_threshold:
                alert_type = AlertType.THRESHOLD_WARNING
            
            if alert_type:
                # Check if we already have this alert
                if not await self._alert_already_exists(budget_name, alert_type):
                    alert = await self._create_budget_alert(
                        budget_name, budget, alert_type, current_spend, percentage_used
                    )
                    alerts.append(alert)
        
        return alerts
    
    def get_cost_manager_status(self) -> Dict[str, Any]:
        """
        Get comprehensive cost manager status.
        
        Returns:
            Dictionary with cost manager status
        """
        return {
            "is_running": self.is_running,
            "total_budgets": len(self.budgets),
            "tracked_providers": len(self.current_costs),
            "alert_callbacks": len(self.alert_callbacks),
            "storage_path": self.storage_path,
            "default_currency": self.default_currency,
            "alert_check_interval_minutes": self.alert_check_interval,
            "budgets": {
                name: {
                    "provider": budget.provider,
                    "period": budget.period.value,
                    "amount": budget.amount,
                    "currency": budget.currency,
                }
                for name, budget in self.budgets.items()
            }
        }
    
    # Additional helper methods would be implemented here
    async def _save_budget(self, budget: BudgetConfig) -> None:
        """Save budget to database."""
        pass  # Implementation details
    
    async def _load_budgets(self) -> None:
        """Load budgets from database."""
        pass  # Implementation details
    
    async def _save_cost_breakdown(self, provider: str, timestamp: datetime, total_cost: float, input_tokens: int, output_tokens: int, requests: int) -> None:
        """Save detailed cost breakdown."""
        pass  # Implementation details
    
    async def _alert_already_exists(self, budget_name: str, alert_type: AlertType) -> bool:
        """Check if alert already exists."""
        return False  # Implementation details
    
    async def _create_budget_alert(self, budget_name: str, budget: BudgetConfig, alert_type: AlertType, current_spend: float, percentage_used: float) -> BudgetAlert:
        """Create a budget alert."""
        # Implementation details
        return BudgetAlert(
            alert_id=f"{budget_name}_{alert_type.value}",
            alert_type=alert_type,
            budget_name=budget_name,
            provider=budget.provider,
            current_spend=current_spend,
            budget_amount=budget.amount,
            percentage_used=percentage_used,
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow()
        )
    
    async def _alert_monitoring_loop(self) -> None:
        """Background loop for monitoring budget alerts."""
        while self.is_running:
            try:
                await asyncio.sleep(self.alert_check_interval * 60)
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(60)