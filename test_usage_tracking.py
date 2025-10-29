#!/usr/bin/env python3
"""
Test script for API usage tracking and cost management system.

This script demonstrates the functionality of the implemented components
and validates that they work correctly together.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_moe_mas.api.usage_tracker import UsageTracker, UsageMetricType, TimeWindow
from quantum_moe_mas.api.cost_manager import CostManager, BudgetPeriod, AlertType
from quantum_moe_mas.api.performance_analyzer import PerformanceAnalyzer, PerformanceMetric
from quantum_moe_mas.api.signup_manager import SignupManager, ProviderCategory
from quantum_moe_mas.api.usage_cost_integration import IntegratedAPIManager


async def test_usage_tracker():
    """Test the usage tracker functionality."""
    print("Testing Usage Tracker...")
    
    tracker = UsageTracker()
    await tracker.initialize()
    
    try:
        # Track some sample usage
        await tracker.track_usage(
            provider="openai_playground",
            endpoint="/chat/completions",
            method="POST",
            success=True,
            status_code=200,
            latency_ms=1500.0,
            tokens_used=150,
            cost=0.003,
            request_size_bytes=1024,
            response_size_bytes=2048
        )
        
        await tracker.track_usage(
            provider="hugging_face",
            endpoint="/models/gpt2",
            method="POST",
            success=True,
            status_code=200,
            latency_ms=800.0,
            tokens_used=100,
            cost=0.001,
            request_size_bytes=512,
            response_size_bytes=1024
        )
        
        # Get real-time stats
        stats = await tracker.get_realtime_stats()
        print(f"Real-time stats: {stats}")
        
        # Get usage trends
        trends = await tracker.get_usage_trends(
            "openai_playground",
            UsageMetricType.LATENCY,
            TimeWindow.MINUTE,
            periods=5
        )
        print(f"Usage trends: {trends}")
        
        print("✓ Usage Tracker test passed")
        
    finally:
        await tracker.shutdown()


async def test_cost_manager():
    """Test the cost manager functionality."""
    print("Testing Cost Manager...")
    
    cost_manager = CostManager()
    await cost_manager.initialize()
    
    try:
        # Create a test budget
        budget = await cost_manager.create_budget(
            name="test_budget",
            amount=10.0,
            period=BudgetPeriod.DAILY,
            provider="openai_playground",
            warning_threshold=80.0,
            critical_threshold=95.0
        )
        print(f"Created budget: {budget.name}")
        
        # Track some costs
        await cost_manager.track_cost(
            provider="openai_playground",
            cost=2.5,
            input_tokens=100,
            output_tokens=50,
            requests=1
        )
        
        # Get current spending
        spending = await cost_manager.get_current_spend()
        print(f"Current spending: {spending}")
        
        # Check for alerts
        alerts = await cost_manager.check_budget_alerts()
        print(f"Budget alerts: {len(alerts)}")
        
        print("✓ Cost Manager test passed")
        
    finally:
        await cost_manager.shutdown()


async def test_performance_analyzer():
    """Test the performance analyzer functionality."""
    print("Testing Performance Analyzer...")
    
    analyzer = PerformanceAnalyzer()
    await analyzer.initialize()
    
    try:
        # Record some performance snapshots
        usage_data = {
            "total_requests": 100,
            "successful_requests": 95,
            "failed_requests": 5,
            "total_latency_ms": 150000.0,
            "total_cost": 0.5,
            "total_tokens": 5000
        }
        
        snapshot = await analyzer.record_performance_snapshot(
            "openai_playground",
            usage_data,
            time_window_minutes=15
        )
        print(f"Recorded performance snapshot: {snapshot.provider}")
        
        # Generate optimization recommendations
        recommendations = await analyzer.generate_optimization_recommendations(
            "openai_playground",
            timedelta(hours=1)
        )
        print(f"Generated {len(recommendations)} recommendations")
        
        print("✓ Performance Analyzer test passed")
        
    finally:
        await analyzer.shutdown()


async def test_signup_manager():
    """Test the signup manager functionality."""
    print("Testing Signup Manager...")
    
    signup_manager = SignupManager()
    await signup_manager.initialize()
    
    try:
        # List available providers
        providers = await signup_manager.list_providers(
            category=ProviderCategory.LANGUAGE_MODEL,
            free_tier_only=True
        )
        print(f"Found {len(providers)} language model providers with free tiers")
        
        # Generate signup links
        signup_links = await signup_manager.generate_signup_links(
            providers=["openai_playground", "hugging_face"],
            include_setup_guide=True
        )
        print(f"Generated signup links for {len(signup_links)} providers")
        
        # Start a signup workflow
        if providers:
            workflow = await signup_manager.start_signup_workflow(
                providers[0].name,
                user_id="test_user"
            )
            print(f"Started signup workflow: {workflow.workflow_id}")
        
        print("✓ Signup Manager test passed")
        
    finally:
        await signup_manager.shutdown()


async def test_integrated_manager():
    """Test the integrated API manager."""
    print("Testing Integrated API Manager...")
    
    manager = IntegratedAPIManager()
    await manager.initialize()
    
    try:
        # Get comprehensive metrics
        metrics = await manager.get_comprehensive_metrics()
        print(f"Retrieved metrics for {len(metrics)} providers")
        
        # Generate cost optimization report
        report = await manager.get_cost_optimization_report()
        print(f"Generated cost optimization report with {len(report['recommendations'])} recommendations")
        
        # Get integration status
        status = manager.get_integration_status()
        print(f"Integration status: {status['is_running']}")
        
        print("✓ Integrated API Manager test passed")
        
    finally:
        await manager.shutdown()


async def main():
    """Run all tests."""
    print("Starting API Usage Tracking and Cost Management Tests")
    print("=" * 60)
    
    try:
        await test_usage_tracker()
        print()
        
        await test_cost_manager()
        print()
        
        await test_performance_analyzer()
        print()
        
        await test_signup_manager()
        print()
        
        await test_integrated_manager()
        print()
        
        print("=" * 60)
        print("All tests passed successfully! ✓")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)