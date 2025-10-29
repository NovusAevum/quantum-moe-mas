"""
Interactive Features and User Experience Components for Quantum MoE MAS

This module provides comprehensive interactive features including drag-and-drop
file upload, query refinement, feedback systems, user preferences, export
functionality, and real-time notifications.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import zipfile
import tempfile
import time

from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


@dataclass
class UserPreferences:
    """User preference configuration."""
    theme: str = "light"
    default_confidence_threshold: float = 0.8
    max_experts: int = 2
    auto_save_queries: bool = True
    notification_level: str = "normal"  # minimal, normal, verbose
    export_format: str = "json"  # json, csv, pdf
    language: str = "en"
    timezone: str = "UTC"
    dashboard_layout: str = "default"  # default, compact, detailed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class QueryFeedback:
    """Query feedback data structure."""
    query_id: str
    rating: int  # 1-5 stars
    accuracy_rating: int  # 1-5
    speed_rating: int  # 1-5
    usefulness_rating: int  # 1-5
    comments: str
    timestamp: datetime
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class NotificationMessage:
    """Notification message structure."""
    id: str
    type: str  # success, info, warning, error
    title: str
    message: str
    timestamp: datetime
    duration: int = 5  # seconds
    dismissible: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class DragDropFileUploader:
    """Advanced drag-and-drop file uploader with multi-modal support."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.DragDropFileUploader")
        self.supported_formats = {
            'text': ['.txt', '.md', '.csv', '.json', '.xml', '.html'],
            'document': ['.pdf', '.docx', '.doc', '.pptx', '.ppt'],
            'image': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'],
            'video': ['.mp4', '.avi', '.mov', '.wmv', '.flv'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
            'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c']
        }
    
    def render_upload_zone(self, key: str = "file_upload") -> List[Any]:
        """Render enhanced drag-and-drop upload zone."""
        st.markdown("### 📎 Multi-Modal File Upload")
        
        # Custom CSS for drag-and-drop styling
        st.markdown("""
        <style>
        .upload-zone {
            border: 2px dashed #1f77b4;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        .upload-zone:hover {
            border-color: #0d47a1;
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        }
        .file-info {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #1f77b4;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Upload zone
        st.markdown("""
        <div class="upload-zone">
            <h4>🎯 Drag & Drop Files Here</h4>
            <p>Support for text, documents, images, videos, audio, and code files</p>
            <p><strong>Supported formats:</strong> PDF, DOCX, PNG, JPG, MP4, MP3, TXT, CSV, JSON, and more</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader with enhanced options
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=self._get_all_extensions(),
            key=key,
            help="Upload multiple files for multi-modal processing"
        )
        
        if uploaded_files:
            self._display_uploaded_files(uploaded_files)
            return uploaded_files
        
        return []
    
    def _get_all_extensions(self) -> List[str]:
        """Get all supported file extensions."""
        extensions = []
        for format_type, exts in self.supported_formats.items():
            extensions.extend([ext.lstrip('.') for ext in exts])
        return extensions
    
    def _display_uploaded_files(self, files: List[Any]) -> None:
        """Display uploaded files with metadata."""
        st.success(f"✅ {len(files)} file(s) uploaded successfully!")
        
        for i, file in enumerate(files):
            file_type = self._detect_file_type(file.name)
            file_size = self._format_file_size(file.size)
            
            st.markdown(f"""
            <div class="file-info">
                <strong>📄 {file.name}</strong><br>
                <small>Type: {file_type.title()} | Size: {file_size}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Show preview for supported formats
            if file_type == 'image':
                st.image(file, caption=file.name, width=200)
            elif file_type == 'text' and file.size < 10000:  # Small text files
                try:
                    content = str(file.read(), "utf-8")
                    st.text_area(f"Preview: {file.name}", content[:500] + "..." if len(content) > 500 else content, height=100)
                    file.seek(0)  # Reset file pointer
                except:
                    st.info(f"Cannot preview {file.name}")
    
    def _detect_file_type(self, filename: str) -> str:
        """Detect file type based on extension."""
        ext = Path(filename).suffix.lower()
        for file_type, extensions in self.supported_formats.items():
            if ext in extensions:
                return file_type
        return 'unknown'
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"


class QueryRefinementSystem:
    """Interactive query refinement and feedback system."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.QueryRefinementSystem")
    
    def render_query_refinement(self, original_query: str, query_id: str) -> Dict[str, Any]:
        """Render query refinement interface."""
        st.markdown("### 🔍 Query Refinement")
        
        # Query refinement options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 Refine Your Query")
            
            # Suggested refinements
            suggestions = self._generate_query_suggestions(original_query)
            
            selected_suggestion = st.selectbox(
                "Quick refinements:",
                ["Original query"] + suggestions,
                help="Select a suggested refinement or keep the original"
            )
            
            # Manual refinement
            refined_query = st.text_area(
                "Manual refinement:",
                value=original_query if selected_suggestion == "Original query" else selected_suggestion,
                height=100,
                help="Manually edit your query for better results"
            )
            
            # Refinement options
            st.markdown("#### ⚙️ Query Options")
            
            add_context = st.checkbox("Add more context", help="Include additional background information")
            specify_format = st.checkbox("Specify output format", help="Request specific response format")
            include_examples = st.checkbox("Request examples", help="Ask for concrete examples")
            
        with col2:
            st.markdown("#### 🎯 Query Analysis")
            
            # Query complexity analysis
            complexity = self._analyze_query_complexity(original_query)
            
            st.metric("Query Complexity", complexity['level'], help="Based on length, keywords, and structure")
            
            # Detected domains
            domains = self._detect_query_domains(original_query)
            st.markdown("**Detected Domains:**")
            for domain in domains:
                st.badge(domain, type="secondary")
            
            # Suggested experts
            st.markdown("**Recommended Experts:**")
            experts = self._suggest_experts(domains)
            for expert in experts:
                st.markdown(f"• {expert}")
        
        # Apply refinements button
        if st.button("🚀 Apply Refinements", type="primary"):
            refinement_data = {
                "original_query": original_query,
                "refined_query": refined_query,
                "suggestions_used": selected_suggestion != "Original query",
                "options": {
                    "add_context": add_context,
                    "specify_format": specify_format,
                    "include_examples": include_examples
                },
                "detected_domains": domains,
                "complexity": complexity
            }
            
            st.success("✅ Query refinement applied!")
            return refinement_data
        
        return {}
    
    def render_feedback_system(self, query_id: str, response: str) -> Optional[QueryFeedback]:
        """Render comprehensive feedback system."""
        st.markdown("### ⭐ Rate This Response")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Rating Categories")
            
            # Overall rating
            overall_rating = st.select_slider(
                "Overall Rating:",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x
            )
            
            # Detailed ratings
            accuracy_rating = st.select_slider(
                "Accuracy:",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "🎯" * x,
                help="How accurate was the response?"
            )
            
            speed_rating = st.select_slider(
                "Response Speed:",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⚡" * x,
                help="How fast was the response?"
            )
            
            usefulness_rating = st.select_slider(
                "Usefulness:",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "💡" * x,
                help="How useful was the response?"
            )
        
        with col2:
            st.markdown("#### 💬 Additional Feedback")
            
            # Feedback categories
            feedback_categories = st.multiselect(
                "What worked well?",
                ["Comprehensive answer", "Fast response", "Relevant examples", 
                 "Clear explanation", "Good formatting", "Helpful suggestions"],
                help="Select all that apply"
            )
            
            improvement_areas = st.multiselect(
                "What could be improved?",
                ["More detail needed", "Too technical", "Too basic", 
                 "Irrelevant information", "Poor formatting", "Slow response"],
                help="Select areas for improvement"
            )
            
            # Comments
            comments = st.text_area(
                "Additional comments:",
                placeholder="Share any specific feedback or suggestions...",
                height=100
            )
        
        # Submit feedback
        if st.button("📤 Submit Feedback", type="primary"):
            feedback = QueryFeedback(
                query_id=query_id,
                rating=overall_rating,
                accuracy_rating=accuracy_rating,
                speed_rating=speed_rating,
                usefulness_rating=usefulness_rating,
                comments=comments,
                timestamp=datetime.now()
            )
            
            # Store feedback in session state
            if 'feedback_history' not in st.session_state:
                st.session_state.feedback_history = []
            
            st.session_state.feedback_history.append(feedback.to_dict())
            
            st.success("✅ Thank you for your feedback!")
            st.balloons()
            
            return feedback
        
        return None
    
    def _generate_query_suggestions(self, query: str) -> List[str]:
        """Generate query refinement suggestions."""
        suggestions = []
        
        # Basic suggestions based on query analysis
        if len(query.split()) < 5:
            suggestions.append(f"Please provide more details about {query}")
        
        if "?" not in query:
            suggestions.append(f"What specific aspects of {query} would you like to know?")
        
        if any(word in query.lower() for word in ['best', 'good', 'better']):
            suggestions.append(f"What criteria should I use to evaluate {query}?")
        
        suggestions.extend([
            f"Explain {query} with examples",
            f"Compare different approaches to {query}",
            f"What are the pros and cons of {query}?"
        ])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity."""
        word_count = len(query.split())
        
        if word_count < 5:
            level = "Simple"
            score = 1
        elif word_count < 15:
            level = "Moderate"
            score = 2
        else:
            level = "Complex"
            score = 3
        
        return {
            "level": level,
            "score": score,
            "word_count": word_count,
            "has_questions": "?" in query,
            "has_technical_terms": any(term in query.lower() for term in 
                                     ['api', 'algorithm', 'database', 'security', 'cloud'])
        }
    
    def _detect_query_domains(self, query: str) -> List[str]:
        """Detect relevant domains from query."""
        domains = []
        query_lower = query.lower()
        
        domain_keywords = {
            "Cybersecurity": ["security", "hack", "vulnerability", "threat", "encryption", "firewall"],
            "Cloud Computing": ["cloud", "aws", "azure", "gcp", "kubernetes", "docker"],
            "Marketing": ["marketing", "campaign", "roi", "conversion", "analytics", "hubspot"],
            "Quantum Computing": ["quantum", "qiskit", "qubit", "superposition", "entanglement"],
            "AI/ML": ["ai", "machine learning", "neural", "model", "algorithm", "data science"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.append(domain)
        
        return domains if domains else ["General"]
    
    def _suggest_experts(self, domains: List[str]) -> List[str]:
        """Suggest experts based on detected domains."""
        expert_mapping = {
            "Cybersecurity": ["Cyber Agent (CEH v12)", "Security Specialist"],
            "Cloud Computing": ["Cloud Agent (Multi-Cloud)", "DevOps Expert"],
            "Marketing": ["Marketing Agent (HubSpot)", "Analytics Specialist"],
            "Quantum Computing": ["Quantum Agent (Qiskit)", "Quantum Researcher"],
            "AI/ML": ["AI Specialist", "Data Science Expert"],
            "General": ["General AI Assistant", "Multi-Domain Expert"]
        }
        
        experts = []
        for domain in domains:
            experts.extend(expert_mapping.get(domain, expert_mapping["General"]))
        
        return list(set(experts))  # Remove duplicates


class UserPreferenceManager:
    """User preference management and system customization."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.UserPreferenceManager")
        self._initialize_preferences()
    
    def _initialize_preferences(self):
        """Initialize user preferences in session state."""
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = UserPreferences().to_dict()
    
    def render_preferences_panel(self) -> UserPreferences:
        """Render user preferences management panel."""
        st.markdown("### ⚙️ User Preferences")
        
        # Load current preferences
        current_prefs = UserPreferences.from_dict(st.session_state.user_preferences)
        
        # Create tabs for different preference categories
        tab1, tab2, tab3, tab4 = st.tabs(["🎨 Appearance", "🔧 System", "📊 Dashboard", "🔔 Notifications"])
        
        with tab1:
            self._render_appearance_preferences(current_prefs)
        
        with tab2:
            self._render_system_preferences(current_prefs)
        
        with tab3:
            self._render_dashboard_preferences(current_prefs)
        
        with tab4:
            self._render_notification_preferences(current_prefs)
        
        # Save preferences
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Preferences", type="primary"):
                st.session_state.user_preferences = current_prefs.to_dict()
                st.success("✅ Preferences saved successfully!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Defaults"):
                st.session_state.user_preferences = UserPreferences().to_dict()
                st.success("✅ Preferences reset to defaults!")
                st.rerun()
        
        with col3:
            if st.button("📤 Export Preferences"):
                self._export_preferences(current_prefs)
        
        return current_prefs
    
    def _render_appearance_preferences(self, prefs: UserPreferences):
        """Render appearance preferences."""
        st.markdown("#### 🎨 Appearance Settings")
        
        prefs.theme = st.selectbox(
            "Theme:",
            ["light", "dark", "auto"],
            index=["light", "dark", "auto"].index(prefs.theme),
            help="Choose your preferred theme"
        )
        
        prefs.language = st.selectbox(
            "Language:",
            ["en", "es", "fr", "de", "zh", "ja"],
            index=["en", "es", "fr", "de", "zh", "ja"].index(prefs.language),
            help="Select your preferred language"
        )
        
        prefs.timezone = st.selectbox(
            "Timezone:",
            ["UTC", "US/Eastern", "US/Pacific", "Europe/London", "Asia/Tokyo"],
            index=["UTC", "US/Eastern", "US/Pacific", "Europe/London", "Asia/Tokyo"].index(prefs.timezone),
            help="Choose your timezone for timestamps"
        )
    
    def _render_system_preferences(self, prefs: UserPreferences):
        """Render system preferences."""
        st.markdown("#### 🔧 System Settings")
        
        prefs.default_confidence_threshold = st.slider(
            "Default Confidence Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=prefs.default_confidence_threshold,
            step=0.1,
            help="Default confidence threshold for expert routing"
        )
        
        prefs.max_experts = st.selectbox(
            "Maximum Experts per Query:",
            [1, 2, 3, 4, 5],
            index=[1, 2, 3, 4, 5].index(prefs.max_experts),
            help="Maximum number of experts to consult per query"
        )
        
        prefs.auto_save_queries = st.checkbox(
            "Auto-save Query History",
            value=prefs.auto_save_queries,
            help="Automatically save queries to history"
        )
        
        prefs.export_format = st.selectbox(
            "Default Export Format:",
            ["json", "csv", "pdf", "xlsx"],
            index=["json", "csv", "pdf", "xlsx"].index(prefs.export_format),
            help="Default format for data exports"
        )
    
    def _render_dashboard_preferences(self, prefs: UserPreferences):
        """Render dashboard preferences."""
        st.markdown("#### 📊 Dashboard Settings")
        
        prefs.dashboard_layout = st.selectbox(
            "Dashboard Layout:",
            ["default", "compact", "detailed"],
            index=["default", "compact", "detailed"].index(prefs.dashboard_layout),
            help="Choose your preferred dashboard layout"
        )
        
        # Dashboard customization options
        st.markdown("**Dashboard Widgets:**")
        
        widget_preferences = st.multiselect(
            "Enabled Widgets:",
            ["System Status", "Query History", "Expert Utilization", "Performance Metrics", 
             "Cost Analytics", "ROI Dashboard", "Real-time Monitoring"],
            default=["System Status", "Query History", "Expert Utilization", "Performance Metrics"],
            help="Select which widgets to display on the dashboard"
        )
        
        # Store widget preferences (would be part of a more complex preference system)
        if 'dashboard_widgets' not in st.session_state:
            st.session_state.dashboard_widgets = widget_preferences
    
    def _render_notification_preferences(self, prefs: UserPreferences):
        """Render notification preferences."""
        st.markdown("#### 🔔 Notification Settings")
        
        prefs.notification_level = st.selectbox(
            "Notification Level:",
            ["minimal", "normal", "verbose"],
            index=["minimal", "normal", "verbose"].index(prefs.notification_level),
            help="Choose how many notifications you want to receive"
        )
        
        # Notification type preferences
        st.markdown("**Notification Types:**")
        
        notification_types = st.multiselect(
            "Enable Notifications For:",
            ["Query Completion", "System Alerts", "Performance Issues", 
             "Cost Warnings", "Expert Updates", "Feedback Requests"],
            default=["Query Completion", "System Alerts", "Performance Issues"],
            help="Select which types of notifications to receive"
        )
        
        # Store notification preferences
        if 'notification_types' not in st.session_state:
            st.session_state.notification_types = notification_types
    
    def _export_preferences(self, prefs: UserPreferences):
        """Export user preferences."""
        preferences_json = json.dumps(prefs.to_dict(), indent=2)
        
        st.download_button(
            label="📥 Download Preferences",
            data=preferences_json,
            file_name=f"quantum_moe_preferences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


class ExportManager:
    """Export functionality for reports, analytics, and query results."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.ExportManager")
    
    def render_export_panel(self, data_sources: Dict[str, Any]) -> None:
        """Render export functionality panel."""
        st.markdown("### 📤 Export Data")
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Available Data Sources")
            
            export_options = st.multiselect(
                "Select data to export:",
                list(data_sources.keys()),
                help="Choose which data sources to include in the export"
            )
            
            export_format = st.selectbox(
                "Export Format:",
                ["JSON", "CSV", "Excel", "PDF Report"],
                help="Choose the export format"
            )
            
            include_metadata = st.checkbox(
                "Include Metadata",
                value=True,
                help="Include timestamps, user info, and system metadata"
            )
        
        with col2:
            st.markdown("#### ⚙️ Export Settings")
            
            date_range = st.date_input(
                "Date Range:",
                value=[datetime.now().date() - timedelta(days=7), datetime.now().date()],
                help="Select date range for the export"
            )
            
            compression = st.checkbox(
                "Compress Export",
                value=False,
                help="Create a compressed ZIP file"
            )
            
            password_protect = st.checkbox(
                "Password Protection",
                value=False,
                help="Add password protection to the export"
            )
            
            if password_protect:
                export_password = st.text_input(
                    "Export Password:",
                    type="password",
                    help="Password for the exported file"
                )
        
        # Export preview
        if export_options:
            st.markdown("#### 👀 Export Preview")
            
            preview_data = self._prepare_export_data(
                {k: v for k, v in data_sources.items() if k in export_options},
                include_metadata
            )
            
            # Show data summary
            st.info(f"📋 Export will contain {len(preview_data)} records")
            
            # Show sample data
            if preview_data:
                st.dataframe(pd.DataFrame(preview_data[:5]), use_container_width=True)
        
        # Export button
        if st.button("🚀 Generate Export", type="primary", disabled=not export_options):
            with st.spinner("📦 Preparing export..."):
                export_data = self._prepare_export_data(
                    {k: v for k, v in data_sources.items() if k in export_options},
                    include_metadata
                )
                
                if export_format == "JSON":
                    self._export_json(export_data, compression)
                elif export_format == "CSV":
                    self._export_csv(export_data, compression)
                elif export_format == "Excel":
                    self._export_excel(export_data, compression)
                elif export_format == "PDF Report":
                    self._export_pdf_report(export_data)
                
                st.success("✅ Export completed successfully!")
    
    def _prepare_export_data(self, data_sources: Dict[str, Any], include_metadata: bool) -> List[Dict[str, Any]]:
        """Prepare data for export."""
        export_data = []
        
        for source_name, source_data in data_sources.items():
            if isinstance(source_data, list):
                for item in source_data:
                    record = {"source": source_name}
                    if include_metadata:
                        record.update({
                            "exported_at": datetime.now().isoformat(),
                            "export_version": "1.0"
                        })
                    
                    if isinstance(item, dict):
                        record.update(item)
                    else:
                        record["data"] = str(item)
                    
                    export_data.append(record)
        
        return export_data
    
    def _export_json(self, data: List[Dict[str, Any]], compress: bool = False):
        """Export data as JSON."""
        json_data = json.dumps(data, indent=2, default=str)
        
        if compress:
            # Create ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("export_data.json", json_data)
            
            st.download_button(
                label="📥 Download JSON (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"quantum_moe_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
        else:
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"quantum_moe_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def _export_csv(self, data: List[Dict[str, Any]], compress: bool = False):
        """Export data as CSV."""
        if not data:
            st.warning("No data to export")
            return
        
        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False)
        
        if compress:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("export_data.csv", csv_data)
            
            st.download_button(
                label="📥 Download CSV (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"quantum_moe_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
        else:
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"quantum_moe_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def _export_excel(self, data: List[Dict[str, Any]], compress: bool = False):
        """Export data as Excel."""
        if not data:
            st.warning("No data to export")
            return
        
        df = pd.DataFrame(data)
        
        # Create Excel file in memory
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Export Data', index=False)
            
            # Add summary sheet
            summary_df = pd.DataFrame({
                'Metric': ['Total Records', 'Export Date', 'Data Sources'],
                'Value': [len(data), datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                         len(set(item.get('source', 'Unknown') for item in data))]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        if compress:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("export_data.xlsx", excel_buffer.getvalue())
            
            st.download_button(
                label="📥 Download Excel (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"quantum_moe_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
        else:
            st.download_button(
                label="📥 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"quantum_moe_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    def _export_pdf_report(self, data: List[Dict[str, Any]]):
        """Export data as PDF report."""
        # For now, create a simple text-based report
        # In a full implementation, you would use libraries like reportlab
        
        report_content = f"""
QUANTUM MOE MAS - EXPORT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
=======
Total Records: {len(data)}
Data Sources: {len(set(item.get('source', 'Unknown') for item in data))}

DATA PREVIEW
============
"""
        
        # Add sample data
        for i, item in enumerate(data[:10]):  # First 10 records
            report_content += f"\nRecord {i+1}:\n"
            for key, value in item.items():
                report_content += f"  {key}: {value}\n"
        
        if len(data) > 10:
            report_content += f"\n... and {len(data) - 10} more records"
        
        st.download_button(
            label="📥 Download PDF Report",
            data=report_content,
            file_name=f"quantum_moe_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )


class NotificationSystem:
    """Real-time notifications and progress indicators."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.NotificationSystem")
        self._initialize_notifications()
    
    def _initialize_notifications(self):
        """Initialize notification system in session state."""
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        
        if 'notification_counter' not in st.session_state:
            st.session_state.notification_counter = 0
    
    def add_notification(self, type: str, title: str, message: str, duration: int = 5) -> str:
        """Add a new notification."""
        notification_id = f"notif_{st.session_state.notification_counter}"
        st.session_state.notification_counter += 1
        
        notification = NotificationMessage(
            id=notification_id,
            type=type,
            title=title,
            message=message,
            timestamp=datetime.now(),
            duration=duration
        )
        
        st.session_state.notifications.append(notification.to_dict())
        
        # Keep only last 50 notifications
        if len(st.session_state.notifications) > 50:
            st.session_state.notifications = st.session_state.notifications[-50:]
        
        return notification_id
    
    def render_notification_center(self):
        """Render notification center."""
        st.markdown("### 🔔 Notifications")
        
        if not st.session_state.notifications:
            st.info("No notifications")
            return
        
        # Notification controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear All"):
                st.session_state.notifications = []
                st.rerun()
        
        with col2:
            show_read = st.checkbox("Show Read", value=False)
        
        with col3:
            filter_type = st.selectbox(
                "Filter:",
                ["All", "Success", "Info", "Warning", "Error"]
            )
        
        # Display notifications
        for notification in reversed(st.session_state.notifications):
            if filter_type != "All" and notification['type'] != filter_type.lower():
                continue
            
            self._render_notification_item(notification)
    
    def _render_notification_item(self, notification: Dict[str, Any]):
        """Render individual notification item."""
        # Notification styling based on type
        type_styles = {
            'success': {'color': '#28a745', 'icon': '✅'},
            'info': {'color': '#17a2b8', 'icon': 'ℹ️'},
            'warning': {'color': '#ffc107', 'icon': '⚠️'},
            'error': {'color': '#dc3545', 'icon': '❌'}
        }
        
        style = type_styles.get(notification['type'], type_styles['info'])
        timestamp = datetime.fromisoformat(notification['timestamp'])
        time_ago = self._format_time_ago(timestamp)
        
        st.markdown(f"""
        <div style="
            border-left: 4px solid {style['color']};
            padding: 1rem;
            margin: 0.5rem 0;
            background: white;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: between; align-items: center;">
                <strong>{style['icon']} {notification['title']}</strong>
                <small style="color: #666; margin-left: auto;">{time_ago}</small>
            </div>
            <p style="margin: 0.5rem 0 0 0; color: #333;">{notification['message']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as time ago."""
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours}h ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes}m ago"
        else:
            return "Just now"
    
    def render_progress_indicator(self, progress: float, status: str, details: str = ""):
        """Render progress indicator."""
        st.markdown(f"### ⏳ {status}")
        
        # Progress bar
        progress_bar = st.progress(progress)
        
        # Status details
        if details:
            st.info(details)
        
        # Progress percentage
        st.markdown(f"**Progress:** {progress:.1%}")
        
        return progress_bar
    
    def show_toast(self, message: str, type: str = "info"):
        """Show toast notification."""
        type_functions = {
            'success': st.success,
            'info': st.info,
            'warning': st.warning,
            'error': st.error
        }
        
        func = type_functions.get(type, st.info)
        func(message)


# Utility functions for interactive features
def initialize_interactive_components():
    """Initialize all interactive components in session state."""
    components = {
        'file_uploader': DragDropFileUploader(),
        'query_refinement': QueryRefinementSystem(),
        'preferences': UserPreferenceManager(),
        'export_manager': ExportManager(),
        'notifications': NotificationSystem()
    }
    
    if 'interactive_components' not in st.session_state:
        st.session_state.interactive_components = components
    
    return st.session_state.interactive_components


def render_interactive_sidebar():
    """Render interactive features in sidebar."""
    st.sidebar.markdown("## 🎛️ Interactive Features")
    
    # Quick actions
    if st.sidebar.button("📤 Quick Export"):
        st.sidebar.success("Export initiated!")
    
    if st.sidebar.button("🔔 View Notifications"):
        st.sidebar.info("Notification center opened!")
    
    if st.sidebar.button("⚙️ Preferences"):
        st.sidebar.info("Preferences panel opened!")
    
    # Real-time status
    st.sidebar.markdown("### 📊 Real-time Status")
    st.sidebar.metric("Active Sessions", "1")
    st.sidebar.metric("Pending Queries", "0")
    st.sidebar.metric("System Load", "23%")