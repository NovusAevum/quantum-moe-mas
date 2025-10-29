"""
Quantum MoE MAS - Streamlit Main Application

This is the main entry point for the Streamlit UI application.
It provides an interactive dashboard for the Quantum-Infused MoE Multi-Agent System.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_moe_mas.core.logging_simple import get_logger
from quantum_moe_mas.config.settings_simple import get_settings
from quantum_moe_mas.ui.components.interactive import (
    initialize_interactive_components,
    DragDropFileUploader,
    QueryRefinementSystem,
    UserPreferenceManager,
    ExportManager,
    NotificationSystem,
    render_interactive_sidebar
)
from quantum_moe_mas.ui.components.enhanced_query import EnhancedQueryInterface

logger = get_logger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Quantum MoE MAS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/wanhanis/quantum-moe-mas',
        'Report a bug': 'https://github.com/wanhanis/quantum-moe-mas/issues',
        'About': """
        # Quantum-Infused MoE Multi-Agent System
        
        A groundbreaking AI system combining quantum-inspired routing,
        multi-modal RAG, and specialized domain agents.
        
        **Version**: 0.1.0  
        **Author**: Wan Mohamad Hanis bin Wan Hassan
        """
    }
)
# Custom CSS for professional styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom theme */
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #2c5aa0);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .status-healthy { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    
    .expert-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .query-container {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class QuantumMoEDashboard:
    """Main dashboard controller for the Quantum MoE MAS system."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.settings = get_settings()
        self.initialize_session_state()
        
        # Initialize interactive components
        self.interactive_components = initialize_interactive_components()
        
        # Initialize enhanced query interface
        self.enhanced_query = EnhancedQueryInterface(self.interactive_components)
        
        # Mock system components for demonstration
        self.system_status = {
            "healthy": True,
            "components": {
                "moe_router": {"status": "healthy", "active_experts": 12},
                "rag_system": {"status": "healthy", "documents": 150},
                "api_orchestrator": {"status": "healthy", "apis": 30},
                "agents": {"status": "healthy", "active": 4}
            },
            "metrics": {
                "total_queries": 1247,
                "avg_response_time": 2.3,
                "success_rate": 98.5,
                "cost_savings": 45.2
            }
        }
    
    def initialize_session_state(self):
        """Initialize Streamlit session state."""
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        if 'current_query' not in st.session_state:
            st.session_state.current_query = ""
        
        if 'selected_experts' not in st.session_state:
            st.session_state.selected_experts = []
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        
        if 'analytics_data' not in st.session_state:
            st.session_state.analytics_data = self.generate_mock_analytics()
        
        if 'user_feedback' not in st.session_state:
            st.session_state.user_feedback = []
        
        if 'export_data' not in st.session_state:
            st.session_state.export_data = {
                "query_history": st.session_state.query_history,
                "analytics": st.session_state.analytics_data,
                "system_metrics": self.system_status["metrics"]
            }
    
    def generate_mock_analytics(self) -> Dict[str, Any]:
        """Generate mock analytics data for demonstration."""
        # Generate time series data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        
        return {
            "query_volume": {
                "dates": dates.tolist(),
                "values": [50 + i * 2 + (i % 7) * 10 for i in range(len(dates))]
            },
            "response_times": {
                "dates": dates.tolist(),
                "values": [2.5 - (i * 0.01) + (i % 5) * 0.2 for i in range(len(dates))]
            },
            "expert_utilization": {
                "Claude Sonnet": 35.2,
                "GPT-4": 28.7,
                "Qwen Coder": 18.9,
                "DeepSeek": 12.1,
                "Others": 5.1
            },
            "cost_savings": {
                "baseline_cost": 1250.00,
                "actual_cost": 685.50,
                "savings": 564.50,
                "savings_percentage": 45.2
            }
        }
    
    def render_header(self):
        """Render the main application header."""
        st.markdown("""
        <div class="main-header">
            <h1>🧠 Quantum MoE Multi-Agent System</h1>
            <p>Intelligent AI Orchestration with Quantum-Inspired Routing</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with navigation and system status."""
        with st.sidebar:
            st.markdown("## 🎛️ Control Panel")
            
            # Navigation
            page = st.selectbox(
                "Navigate to:",
                ["🏠 Dashboard", "🔍 Query Interface", "📊 Analytics", 
                 "🤖 Experts", "📚 Documents", "👥 Agents", "⚙️ Settings",
                 "🔔 Notifications", "📤 Export", "🎨 Preferences"]
            )
            
            st.markdown("---")
            
            # Interactive features sidebar
            render_interactive_sidebar()
            
            st.markdown("---")
            
            # System Status
            st.markdown("## 🔧 System Status")
            
            if self.system_status["healthy"]:
                st.success("🟢 System Healthy")
            else:
                st.error("🔴 System Issues")
            
            # Component status
            components = self.system_status["components"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MoE Router", "🟢 OK")
                st.metric("RAG System", "🟢 OK")
            
            with col2:
                st.metric("API Orchestrator", "🟢 OK")
                st.metric("Agents", "🟢 OK")
            
            st.markdown("---")
            
            # Quick metrics
            st.markdown("## 📈 Quick Metrics")
            metrics = self.system_status["metrics"]
            
            st.metric("Total Queries", f"{metrics['total_queries']:,}")
            st.metric("Avg Response", f"{metrics['avg_response_time']:.1f}s")
            st.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
            st.metric("Cost Savings", f"{metrics['cost_savings']:.1f}%")
            
            return page
    
    def render_dashboard_page(self):
        """Render the main dashboard page."""
        st.markdown("## 📊 System Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = self.system_status["metrics"]
        
        with col1:
            st.metric(
                "Total Queries Today",
                f"{metrics['total_queries']:,}",
                delta="↗️ +127 from yesterday"
            )
        
        with col2:
            st.metric(
                "Average Response Time",
                f"{metrics['avg_response_time']:.2f}s",
                delta="↘️ -0.3s improvement"
            )
        
        with col3:
            st.metric(
                "Success Rate",
                f"{metrics['success_rate']:.1f}%",
                delta="↗️ +2.1% improvement"
            )
        
        with col4:
            st.metric(
                "Cost Savings",
                f"{metrics['cost_savings']:.1f}%",
                delta="↗️ +5.2% this month"
            )
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Query Volume Trend")
            analytics = st.session_state.analytics_data
            
            fig = px.line(
                x=analytics["query_volume"]["dates"],
                y=analytics["query_volume"]["values"],
                title="Daily Query Volume",
                labels={"x": "Date", "y": "Queries"}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⚡ Response Time Trend")
            
            fig = px.line(
                x=analytics["response_times"]["dates"],
                y=analytics["response_times"]["values"],
                title="Average Response Time",
                labels={"x": "Date", "y": "Response Time (s)"}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Expert utilization
        st.markdown("### 🤖 Expert Utilization")
        
        utilization = analytics["expert_utilization"]
        
        fig = px.pie(
            values=list(utilization.values()),
            names=list(utilization.keys()),
            title="Expert Usage Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_query_page(self):
        """Render the query interface page."""
        st.markdown("## 🔍 Query Interface")
        
        # Query input section
        st.markdown("""
        <div class="query-container">
            <h3>💬 Ask the Quantum MoE System</h3>
            <p>Enter your query below and let our AI experts provide intelligent responses.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Query input
        query = st.text_area(
            "Enter your query:",
            value=st.session_state.current_query,
            height=100,
            placeholder="Ask anything about cybersecurity, cloud computing, marketing, or quantum computing..."
        )
        
        # File upload section
        st.markdown("### 📎 Upload Documents (Multi-Modal)")
        
        uploaded_files = st.file_uploader(
            "Upload files for context",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'mp4', 'mp3']
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size} bytes)")
        
        # Query options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            domain = st.selectbox(
                "Domain Focus:",
                ["Auto-detect", "Cybersecurity", "Cloud Computing", "Marketing", "Quantum Computing", "General"]
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Confidence Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.1
            )
        
        with col3:
            max_experts = st.selectbox(
                "Max Experts:",
                [1, 2, 3, 4, 5],
                index=1
            )
        
        # Submit button
        if st.button("🚀 Submit Query", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("🧠 Processing query with Quantum MoE routing..."):
                    # Simulate processing
                    time.sleep(2)
                    
                    # Mock response
                    response = self.generate_mock_response(query, domain)
                    
                    # Add to history
                    st.session_state.query_history.append({
                        "timestamp": datetime.now(),
                        "query": query,
                        "domain": domain,
                        "response": response,
                        "confidence": confidence_threshold
                    })
                    
                    # Display results
                    self.display_query_results(response)
            else:
                st.warning("⚠️ Please enter a query first.")
        
        # Query history
        if st.session_state.query_history:
            st.markdown("### 📚 Query History")
            
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Query {len(st.session_state.query_history) - i}: {item['query'][:50]}..."):
                    st.write(f"**Domain:** {item['domain']}")
                    st.write(f"**Timestamp:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Response:** {item['response']}")
    
    def generate_mock_response(self, query: str, domain: str) -> str:
        """Generate a mock response for demonstration."""
        responses = {
            "Cybersecurity": f"🔒 **Cybersecurity Analysis**: Based on the query '{query}', I've analyzed potential security implications using OSINT techniques and CEH v12 methodologies. Key findings include threat vectors, mitigation strategies, and compliance recommendations.",
            
            "Cloud Computing": f"☁️ **Cloud Architecture Response**: For '{query}', I recommend a multi-cloud approach using AWS, Google Cloud, and Azure. The solution includes auto-scaling, cost optimization, and infrastructure as code deployment strategies.",
            
            "Marketing": f"📈 **Marketing Intelligence**: Analyzing '{query}' through HubSpot integration reveals ROI optimization opportunities. Campaign performance metrics suggest 15-30% improvement potential with targeted audience segmentation.",
            
            "Quantum Computing": f"⚛️ **Quantum Analysis**: Using IBM Qiskit simulations for '{query}', the quantum algorithm demonstrates 2-7x efficiency gains over classical approaches. Quantum-classical hybrid optimization shows promising results.",
            
            "General": f"🧠 **Multi-Domain Analysis**: The query '{query}' has been processed through our quantum-inspired MoE routing system. Multiple expert agents have provided comprehensive insights across relevant domains."
        }
        
        return responses.get(domain, responses["General"])
    
    def display_query_results(self, response: str):
        """Display query results with expert routing visualization."""
        st.markdown("### 🎯 Query Results")
        
        # Expert routing visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 💬 Response")
            st.markdown(f"""
            <div class="result-item">
                {response}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🤖 Expert Routing")
            
            # Mock expert selection
            selected_experts = [
                {"name": "Claude Sonnet 4", "confidence": 0.95, "domain": "General"},
                {"name": "Qwen3 Coder Plus", "confidence": 0.87, "domain": "Technical"}
            ]
            
            for expert in selected_experts:
                st.markdown(f"""
                <div class="expert-card">
                    <strong>{expert['name']}</strong><br>
                    Confidence: {expert['confidence']:.1%}<br>
                    Domain: {expert['domain']}
                </div>
                """, unsafe_allow_html=True)
            
            # Routing metrics
            st.metric("Routing Time", "0.15s")
            st.metric("Total Cost", "$0.003")
    
    def run(self):
        """Run the main dashboard application."""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar and get selected page
            selected_page = self.render_sidebar()
            
            # Render selected page
            if selected_page == "🏠 Dashboard":
                self.render_dashboard_page()
            elif selected_page == "🔍 Query Interface":
                self.enhanced_query.render_enhanced_query_page(self.system_status, self.generate_mock_response)
            elif selected_page == "📊 Analytics":
                st.markdown("## 📊 Analytics Dashboard")
                st.info("Advanced analytics dashboard coming soon!")
            elif selected_page == "🤖 Experts":
                st.markdown("## 🤖 Expert Management")
                st.info("Expert pool management interface coming soon!")
            elif selected_page == "📚 Documents":
                st.markdown("## 📚 Document Management")
                st.info("Document library and RAG management coming soon!")
            elif selected_page == "👥 Agents":
                st.markdown("## 👥 Multi-Agent System")
                st.info("Agent coordination dashboard coming soon!")
            elif selected_page == "⚙️ Settings":
                st.markdown("## ⚙️ System Settings")
                st.info("System configuration interface coming soon!")
            elif selected_page == "🔔 Notifications":
                self.render_notifications_page()
            elif selected_page == "📤 Export":
                self.render_export_page()
            elif selected_page == "🎨 Preferences":
                self.render_preferences_page()
            
        except Exception as e:
            st.error(f"Application error: {e}")
            logger.error(f"Dashboard error: {e}")
    
    def render_notifications_page(self):
        """Render the notifications page."""
        st.markdown("## 🔔 Notification Center")
        
        notifications = self.interactive_components['notifications']
        notifications.render_notification_center()
        
        # Add some demo notifications
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➕ Add Success Notification"):
                notifications.add_notification("success", "Test Success", "This is a test success notification")
                st.rerun()
        
        with col2:
            if st.button("➕ Add Warning Notification"):
                notifications.add_notification("warning", "Test Warning", "This is a test warning notification")
                st.rerun()
        
        with col3:
            if st.button("➕ Add Error Notification"):
                notifications.add_notification("error", "Test Error", "This is a test error notification")
                st.rerun()
    
    def render_export_page(self):
        """Render the export page."""
        st.markdown("## 📤 Data Export Center")
        
        # Prepare export data
        export_data = {
            "Query History": st.session_state.get('query_history', []),
            "User Feedback": st.session_state.get('user_feedback', []),
            "Analytics Data": st.session_state.get('analytics_data', {}),
            "System Metrics": self.system_status.get('metrics', {})
        }
        
        export_manager = self.interactive_components['export_manager']
        export_manager.render_export_panel(export_data)
    
    def render_preferences_page(self):
        """Render the user preferences page."""
        st.markdown("## 🎨 User Preferences")
        
        preferences_manager = self.interactive_components['preferences']
        preferences_manager.render_preferences_panel()
    
    def display_enhanced_query_results(self, response: str, query_id: str):
        """Display enhanced query results with feedback system."""
        st.markdown("### 🎯 Query Results")
        
        # Expert routing visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 💬 Response")
            st.markdown(f"""
            <div class="result-item">
                {response}
            </div>
            """, unsafe_allow_html=True)
            
            # Feedback system
            query_refinement = self.interactive_components['query_refinement']
            feedback = query_refinement.render_feedback_system(query_id, response)
            
            if feedback:
                st.session_state.user_feedback.append(feedback.to_dict())
        
        with col2:
            st.markdown("#### 🤖 Expert Routing")
            
            # Mock expert selection with enhanced visualization
            selected_experts = [
                {"name": "Claude Sonnet 4", "confidence": 0.95, "domain": "General", "cost": "$0.002"},
                {"name": "Qwen3 Coder Plus", "confidence": 0.87, "domain": "Technical", "cost": "$0.001"}
            ]
            
            for expert in selected_experts:
                st.markdown(f"""
                <div class="expert-card">
                    <strong>{expert['name']}</strong><br>
                    Confidence: {expert['confidence']:.1%}<br>
                    Domain: {expert['domain']}<br>
                    Cost: {expert['cost']}
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced routing metrics
            st.metric("Routing Time", "0.15s")
            st.metric("Total Cost", "$0.003")
            st.metric("Efficiency Gain", "23%")
            
            # Export options for this query
            if st.button("📤 Export Result"):
                export_data = {
                    "query_result": {
                        "query_id": query_id,
                        "response": response,
                        "experts": selected_experts,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                export_manager = self.interactive_components['export_manager']
                export_manager.render_export_panel(export_data)


def main():
    """Main entry point for the Streamlit application."""
    try:
        dashboard = QuantumMoEDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Failed to initialize dashboard: {e}")
        logger.error(f"Dashboard initialization failed: {e}")


if __name__ == "__main__":
    main()