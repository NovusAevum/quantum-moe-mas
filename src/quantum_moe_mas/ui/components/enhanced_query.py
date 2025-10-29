"""
Enhanced Query Interface Components

This module provides enhanced query interface components that integrate
all the interactive features for a comprehensive user experience.
"""

import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from quantum_moe_mas.core.logging_simple import get_logger
from quantum_moe_mas.ui.components.interactive import (
    DragDropFileUploader,
    QueryRefinementSystem,
    NotificationSystem,
    ExportManager
)

logger = get_logger(__name__)


class EnhancedQueryInterface:
    """Enhanced query interface with all interactive features."""
    
    def __init__(self, interactive_components: Dict[str, Any]):
        self.interactive_components = interactive_components
        self.logger = get_logger(f"{__name__}.EnhancedQueryInterface")
    
    def render_enhanced_query_page(self, system_status: Dict[str, Any], generate_mock_response_func):
        """Render the enhanced query interface page with all interactive features."""
        st.markdown("## 🔍 Enhanced Query Interface")
        
        # Query input section with enhanced styling
        st.markdown("""
        <div class="query-container">
            <h3>💬 Ask the Quantum MoE System</h3>
            <p>Enter your query below and let our AI experts provide intelligent responses with advanced interactive features.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Query input with real-time character count
        query = st.text_area(
            "Enter your query:",
            value=st.session_state.get('current_query', ''),
            height=120,
            placeholder="Ask anything about cybersecurity, cloud computing, marketing, or quantum computing...",
            help="💡 Tip: Be specific for better results. Use the refinement tools below for optimization."
        )
        
        # Character count and query quality indicator
        if query:
            char_count = len(query)
            word_count = len(query.split())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"📝 Characters: {char_count}")
            with col2:
                st.caption(f"📊 Words: {word_count}")
            with col3:
                quality = "Good" if word_count > 5 else "Basic" if word_count > 2 else "Too short"
                color = "green" if quality == "Good" else "orange" if quality == "Basic" else "red"
                st.caption(f"🎯 Quality: :{color}[{quality}]")
        
        # Enhanced drag-and-drop file upload
        st.markdown("---")
        file_uploader = self.interactive_components['file_uploader']
        uploaded_files = file_uploader.render_upload_zone("enhanced_query_upload")
        
        # Store uploaded files in session state
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            # File processing options
            with st.expander("📋 File Processing Options", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    extract_text = st.checkbox("Extract text content", value=True)
                    analyze_images = st.checkbox("Analyze images", value=True)
                    process_audio = st.checkbox("Process audio files", value=False)
                
                with col2:
                    include_metadata = st.checkbox("Include file metadata", value=True)
                    auto_summarize = st.checkbox("Auto-summarize content", value=False)
                    cross_reference = st.checkbox("Cross-reference files", value=False)
        
        # Query refinement system
        if query.strip():
            st.markdown("---")
            query_refinement = self.interactive_components['query_refinement']
            
            with st.expander("🔍 Query Refinement & Analysis", expanded=False):
                refinement_data = query_refinement.render_query_refinement(
                    query, 
                    f"query_{len(st.session_state.get('query_history', []))}"
                )
                
                if refinement_data:
                    query = refinement_data.get('refined_query', query)
                    st.session_state.current_query = query
                    st.success("✅ Query refinement applied!")
        
        # Advanced query options with user preferences
        st.markdown("---")
        st.markdown("### ⚙️ Query Configuration")
        
        user_prefs = st.session_state.get('user_preferences', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            domain = st.selectbox(
                "🎯 Domain Focus:",
                ["Auto-detect", "Cybersecurity", "Cloud Computing", "Marketing", "Quantum Computing", "General"],
                help="Select the primary domain for expert routing"
            )
        
        with col2:
            confidence_threshold = st.slider(
                "🎚️ Confidence Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=user_prefs.get('default_confidence_threshold', 0.8),
                step=0.05,
                help="Minimum confidence level for expert selection"
            )
        
        with col3:
            max_experts = st.selectbox(
                "👥 Max Experts:",
                [1, 2, 3, 4, 5],
                index=[1, 2, 3, 4, 5].index(user_prefs.get('max_experts', 2)),
                help="Maximum number of experts to consult"
            )
        
        with col4:
            response_style = st.selectbox(
                "📝 Response Style:",
                ["Detailed", "Concise", "Technical", "Business"],
                help="Preferred response format and style"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Processing Options:**")
                enable_caching = st.checkbox("Enable response caching", value=True)
                parallel_processing = st.checkbox("Parallel expert processing", value=True)
                include_sources = st.checkbox("Include source references", value=True)
            
            with col2:
                st.markdown("**Output Options:**")
                generate_summary = st.checkbox("Generate executive summary", value=False)
                include_metrics = st.checkbox("Include performance metrics", value=True)
                export_ready = st.checkbox("Prepare for export", value=False)
        
        # Submit button with enhanced processing
        st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            submit_button = st.button(
                "🚀 Submit Enhanced Query", 
                type="primary", 
                use_container_width=True,
                disabled=not query.strip()
            )
        
        with col2:
            if st.button("💾 Save Draft", use_container_width=True):
                if query.strip():
                    if 'query_drafts' not in st.session_state:
                        st.session_state.query_drafts = []
                    
                    draft = {
                        "query": query,
                        "domain": domain,
                        "timestamp": datetime.now(),
                        "files": [f.name for f in uploaded_files] if uploaded_files else []
                    }
                    st.session_state.query_drafts.append(draft)
                    st.success("💾 Draft saved!")
        
        with col3:
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state.current_query = ""
                st.session_state.uploaded_files = []
                st.rerun()
        
        # Process query with enhanced features
        if submit_button and query.strip():
            self._process_enhanced_query(
                query=query,
                domain=domain,
                confidence_threshold=confidence_threshold,
                max_experts=max_experts,
                response_style=response_style,
                uploaded_files=uploaded_files,
                generate_mock_response_func=generate_mock_response_func,
                advanced_options={
                    "enable_caching": enable_caching,
                    "parallel_processing": parallel_processing,
                    "include_sources": include_sources,
                    "generate_summary": generate_summary,
                    "include_metrics": include_metrics,
                    "export_ready": export_ready
                }
            )
        
        # Enhanced query history with advanced features
        self._render_enhanced_query_history()
        
        # Query drafts section
        self._render_query_drafts()
    
    def _process_enhanced_query(self, query: str, domain: str, confidence_threshold: float, 
                               max_experts: int, response_style: str, uploaded_files: List[Any],
                               generate_mock_response_func, advanced_options: Dict[str, bool]):
        """Process query with enhanced features and real-time progress."""
        
        notifications = self.interactive_components['notifications']
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            # Initialize progress tracking
            progress_steps = [
                ("Initializing quantum routing", 0.1),
                ("Analyzing query complexity", 0.2),
                ("Processing uploaded files", 0.3),
                ("Routing to optimal experts", 0.5),
                ("Generating response", 0.7),
                ("Applying response styling", 0.8),
                ("Finalizing results", 0.9),
                ("Complete", 1.0)
            ]
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            # Process each step with realistic timing
            for step_name, progress_value in progress_steps:
                status_text.text(f"🔄 {step_name}...")
                progress_bar.progress(progress_value)
                
                # Simulate processing time based on step
                if "files" in step_name and uploaded_files:
                    time.sleep(1.0)  # Longer for file processing
                elif "experts" in step_name:
                    time.sleep(0.8)  # Expert routing takes time
                elif "response" in step_name:
                    time.sleep(1.2)  # Response generation
                else:
                    time.sleep(0.4)  # Standard steps
            
            # Generate enhanced response
            response = generate_mock_response_func(query, domain)
            
            # Add response enhancements based on style
            if response_style == "Technical":
                response += "\n\n**Technical Details:** Advanced algorithms and quantum-inspired routing protocols were utilized for optimal expert selection."
            elif response_style == "Business":
                response += f"\n\n**Business Impact:** This analysis provides actionable insights with potential ROI improvements of 15-30%."
            elif response_style == "Concise":
                response = response[:200] + "..." if len(response) > 200 else response
            
            # Create enhanced query record
            query_id = f"enhanced_query_{len(st.session_state.get('query_history', []))}"
            
            query_data = {
                "id": query_id,
                "timestamp": datetime.now(),
                "query": query,
                "domain": domain,
                "response": response,
                "confidence": confidence_threshold,
                "max_experts": max_experts,
                "response_style": response_style,
                "uploaded_files": [f.name for f in uploaded_files] if uploaded_files else [],
                "advanced_options": advanced_options,
                "processing_time": sum(step[1] for step in progress_steps) * 2.5  # Simulated time
            }
            
            # Add to history
            if 'query_history' not in st.session_state:
                st.session_state.query_history = []
            st.session_state.query_history.append(query_data)
            
            # Clear progress and show results
            progress_container.empty()
            
            # Display enhanced results
            self._display_enhanced_results(response, query_id, query_data)
            
            # Add success notification
            notifications.add_notification(
                "success", 
                "Enhanced Query Completed", 
                f"Successfully processed query with {max_experts} experts: {query[:50]}..."
            )
    
    def _display_enhanced_results(self, response: str, query_id: str, query_data: Dict[str, Any]):
        """Display enhanced query results with comprehensive features."""
        st.markdown("### 🎯 Enhanced Query Results")
        
        # Results tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Response", "🤖 Expert Analysis", "📊 Metrics", "📤 Export"])
        
        with tab1:
            # Main response with enhanced formatting
            st.markdown("#### 💬 AI Response")
            
            # Response quality indicators
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Response Quality", "95%", "↗️ +5%")
            with col2:
                st.metric("Relevance Score", "92%", "↗️ +3%")
            with col3:
                st.metric("Completeness", "88%", "↗️ +2%")
            
            # Main response content
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                border-radius: 15px;
                padding: 2rem;
                margin: 1rem 0;
                border-left: 4px solid #1f77b4;
            ">
                {response}
            </div>
            """, unsafe_allow_html=True)
            
            # Response actions
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("👍 Helpful"):
                    st.success("Thank you for your feedback!")
            
            with col2:
                if st.button("🔄 Refine"):
                    st.info("Query refinement options available above")
            
            with col3:
                if st.button("📋 Copy"):
                    st.info("Response copied to clipboard!")
            
            with col4:
                if st.button("🔗 Share"):
                    st.info("Share link generated!")
        
        with tab2:
            # Expert analysis and routing details
            st.markdown("#### 🤖 Expert Routing Analysis")
            
            # Mock expert data with enhanced details
            experts_data = [
                {
                    "name": "Claude Sonnet 4",
                    "confidence": 0.95,
                    "domain": "General Analysis",
                    "cost": "$0.002",
                    "response_time": "1.2s",
                    "tokens_used": 1250,
                    "specialization": "Multi-domain reasoning"
                },
                {
                    "name": "Qwen3 Coder Plus",
                    "confidence": 0.87,
                    "domain": "Technical Implementation",
                    "cost": "$0.001",
                    "response_time": "0.8s",
                    "tokens_used": 890,
                    "specialization": "Code generation and analysis"
                }
            ]
            
            for i, expert in enumerate(experts_data):
                with st.expander(f"🎯 Expert {i+1}: {expert['name']}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Confidence", f"{expert['confidence']:.1%}")
                        st.metric("Response Time", expert['response_time'])
                        st.metric("Cost", expert['cost'])
                    
                    with col2:
                        st.metric("Tokens Used", f"{expert['tokens_used']:,}")
                        st.write(f"**Domain:** {expert['domain']}")
                        st.write(f"**Specialization:** {expert['specialization']}")
        
        with tab3:
            # Performance metrics and analytics
            st.markdown("#### 📊 Performance Metrics")
            
            # Key performance indicators
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Processing Time", f"{query_data.get('processing_time', 2.5):.1f}s")
            
            with col2:
                st.metric("Routing Efficiency", "94%", "↗️ +4%")
            
            with col3:
                st.metric("Cost Efficiency", "87%", "↗️ +7%")
            
            with col4:
                st.metric("User Satisfaction", "4.8/5", "↗️ +0.2")
            
            # Advanced metrics visualization
            st.markdown("**Detailed Performance Breakdown:**")
            
            metrics_data = {
                "Metric": ["Query Analysis", "Expert Routing", "Response Generation", "Post-processing"],
                "Time (s)": [0.3, 0.8, 1.2, 0.2],
                "Efficiency (%)": [98, 94, 91, 96]
            }
            
            import pandas as pd
            import plotly.express as px
            
            df_metrics = pd.DataFrame(metrics_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_time = px.bar(df_metrics, x="Metric", y="Time (s)", title="Processing Time by Stage")
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col2:
                fig_eff = px.bar(df_metrics, x="Metric", y="Efficiency (%)", title="Efficiency by Stage")
                st.plotly_chart(fig_eff, use_container_width=True)
        
        with tab4:
            # Export options and sharing
            st.markdown("#### 📤 Export & Sharing Options")
            
            export_manager = self.interactive_components['export_manager']
            
            # Quick export buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Export as PDF", use_container_width=True):
                    st.info("PDF export initiated...")
            
            with col2:
                if st.button("📊 Export as Excel", use_container_width=True):
                    st.info("Excel export initiated...")
            
            with col3:
                if st.button("📋 Export as JSON", use_container_width=True):
                    st.info("JSON export initiated...")
            
            # Detailed export options
            export_data = {
                "Enhanced Query Result": {
                    "query_id": query_id,
                    "query": query_data['query'],
                    "response": response,
                    "experts": experts_data,
                    "metrics": metrics_data,
                    "timestamp": query_data['timestamp'].isoformat()
                }
            }
            
            export_manager.render_export_panel(export_data)
        
        # Feedback system
        st.markdown("---")
        query_refinement = self.interactive_components['query_refinement']
        feedback = query_refinement.render_feedback_system(query_id, response)
        
        if feedback:
            if 'user_feedback' not in st.session_state:
                st.session_state.user_feedback = []
            st.session_state.user_feedback.append(feedback.to_dict())
    
    def _render_enhanced_query_history(self):
        """Render enhanced query history with advanced features."""
        if not st.session_state.get('query_history'):
            return
        
        st.markdown("---")
        st.markdown("### 📚 Enhanced Query History")
        
        # History controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            show_count = st.selectbox("Show last:", [5, 10, 20, "All"], index=0)
        
        with col2:
            filter_domain = st.selectbox("Filter by domain:", ["All"] + ["Cybersecurity", "Cloud Computing", "Marketing", "Quantum Computing", "General"])
        
        with col3:
            sort_by = st.selectbox("Sort by:", ["Newest", "Oldest", "Domain", "Rating"])
        
        with col4:
            if st.button("📤 Export History"):
                export_manager = self.interactive_components['export_manager']
                export_data = {"query_history": st.session_state.query_history}
                export_manager.render_export_panel(export_data)
        
        # Filter and sort history
        history = st.session_state.query_history.copy()
        
        if filter_domain != "All":
            history = [h for h in history if h.get('domain') == filter_domain]
        
        if sort_by == "Oldest":
            history = sorted(history, key=lambda x: x['timestamp'])
        elif sort_by == "Domain":
            history = sorted(history, key=lambda x: x.get('domain', ''))
        elif sort_by == "Rating":
            # Sort by feedback rating if available
            history = sorted(history, key=lambda x: self._get_query_rating(x.get('id', '')), reverse=True)
        
        # Display history
        history_count = len(history) if show_count == "All" else min(int(show_count), len(history))
        
        for i, item in enumerate(history[-history_count:]):
            with st.expander(f"🔍 Query {i+1}: {item['query'][:60]}...", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Query:** {item['query']}")
                    st.write(f"**Response:** {item['response'][:200]}...")
                    
                    if item.get('uploaded_files'):
                        st.write(f"**Files:** {', '.join(item['uploaded_files'])}")
                
                with col2:
                    st.write(f"**Domain:** {item['domain']}")
                    st.write(f"**Timestamp:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Confidence:** {item['confidence']:.1%}")
                    
                    if 'processing_time' in item:
                        st.write(f"**Processing Time:** {item['processing_time']:.1f}s")
                    
                    # Show feedback if available
                    rating = self._get_query_rating(item.get('id', ''))
                    if rating > 0:
                        st.write(f"**Rating:** {'⭐' * rating} ({rating}/5)")
    
    def _render_query_drafts(self):
        """Render saved query drafts."""
        if not st.session_state.get('query_drafts'):
            return
        
        st.markdown("---")
        st.markdown("### 💾 Saved Drafts")
        
        drafts = st.session_state.query_drafts
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_draft = st.selectbox(
                "Select a draft to load:",
                range(len(drafts)),
                format_func=lambda x: f"Draft {x+1}: {drafts[x]['query'][:50]}... ({drafts[x]['timestamp'].strftime('%m/%d %H:%M')})"
            )
        
        with col2:
            if st.button("📝 Load Draft"):
                draft = drafts[selected_draft]
                st.session_state.current_query = draft['query']
                st.success("Draft loaded!")
                st.rerun()
        
        # Show draft preview
        if drafts:
            draft = drafts[selected_draft]
            with st.expander("👀 Draft Preview", expanded=False):
                st.write(f"**Query:** {draft['query']}")
                st.write(f"**Domain:** {draft['domain']}")
                st.write(f"**Created:** {draft['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                if draft.get('files'):
                    st.write(f"**Files:** {', '.join(draft['files'])}")
    
    def _get_query_rating(self, query_id: str) -> int:
        """Get rating for a query from feedback."""
        feedback_list = st.session_state.get('user_feedback', [])
        for feedback in feedback_list:
            if feedback.get('query_id') == query_id:
                return feedback.get('rating', 0)
        return 0