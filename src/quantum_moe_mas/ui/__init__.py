"""
Streamlit UI Package for Quantum MoE MAS

This package provides the complete user interface for the Quantum-Infused
Mixture of Experts Multi-Agent System, including dashboards, analytics,
and interactive components.
"""

from quantum_moe_mas.ui.main import main as run_streamlit_app, QuantumMoEDashboard

__all__ = [
    "run_streamlit_app",
    "QuantumMoEDashboard"
]