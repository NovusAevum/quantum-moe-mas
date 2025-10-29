"""
Business Intelligence and ROI Tracking Module

This module provides comprehensive business intelligence capabilities including:
- ROI calculation and tracking
- Performance analytics and reporting
- Cost-benefit analysis
- Revenue attribution
- Business optimization recommendations
"""

from .roi_engine import ROICalculationEngine
from .analytics import BusinessAnalytics
from .optimization import BusinessOptimizer

__all__ = [
    'ROICalculationEngine',
    'BusinessAnalytics', 
    'BusinessOptimizer'
]