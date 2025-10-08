"""
Vague Language Detection and Context Linking System
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import tempfile
import time

from utils.pdf_parser import extract_text_with_location, extract_text_simple
from utils.text_processor import segment_sentences, segment_paragraphs, map_sentences_to_blocks, map_paragraphs_to_blocks
from utils.vagueness_detector import analyze_sentence, analyze_paragraph

# Page configuration
st.set_page_config(
    page_title="Vague Language Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .vague-sentence {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .clear-sentence {
        background-color: #d4edda;
        padding: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .analyzing-badge {
        background-color: #17a2b8;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None
    if 'is_analyzing' not in st.session_state:
        st.session_state.is_analyzing = False
    if 'results_list' not in st.session_state:
        st.session_state.results_list = []
    if 'current_progress' not in st.session_state:
        st.session_state.current_progress = 0
    if 'total_sentences' not in st.session_state:
        st.session_state.total_sentences = 0
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""


def process_file(file, file_type):
    """Process uploaded file and extract text"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name
    
    if file_type == "pdf":
        # Extract text with location data
        blocks = extract_text_with_location(tmp_path)
        full_text = " ".join([block["text"] for block in blocks])
        return full_text, blocks, tmp_path
    else:  # txt file
        full_text = file.getvalue().decode("utf-8")
        # Create dummy blocks for txt files
        blocks = [{"text": full_text, "page": 1, "bbox": {"x0": 0, "y0": 0, "x1": 0, "y1": 0}}]
        return full_text, blocks, tmp_path


def main():
    """Main application function"""
    
    initialize_session_state()

    # Custom CSS for header styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .sub-header {
            font-size: 16px;
            color: #555;
            margin-top: 2px;
            margin-bottom: 2px;
            line-height: 1.2;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-header">🔍 Vague Language Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Automatically detects and explains vague language in construction contract documents using LLMs.</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Made as Proof of Concept by Prakhar Agrahari (24M0631) as Part of MTP Stage 1 under the guidance of Prof. Venkata Delhi</p>', unsafe_allow_html=True)

    # Sidebar (continue your code here)
    # Sidebar
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # API Key input
        api_key_input = st.text_input(
            "Google Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your Google Gemini API key. Get one at https://makersuite.google.com/app/apikey",
            placeholder="Enter your API key here..."
        )
        
        # Update session state when API key changes
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
        
        # Show API key status
        if st.session_state.api_key:
            st.success("✅ API Key Configured")
        else:
            st.error("❌ API Key Required")
        
        st.divider()
        
        st.header("⚙️ Configuration")
        
        # Analysis mode selection
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Paragraph-based (Recommended)", "Sentence-based"],
            help="Paragraph mode provides better context for accurate vagueness detection"
        )
        
        st.divider()
        
        # Model selection
        model_choice = st.selectbox(
            "Select Gemini Model",
            ["gemini-2.5-flash", "gemini-2.5-pro"],
            help="Flash is faster, Pro is more accurate"
        )
        
        st.divider()
        
        st.header("📋 About")
        st.info("""
        This tool helps you identify vague, ambiguous, or imprecise language in documents.
        
        **Features:**
        - AI-powered vagueness detection
        - Detailed explanations
        - Improvement suggestions
        - PDF location linking
        
        **Tip:** Use Paragraph mode for better accuracy!
        """)
        
        st.divider()
        
        # Clear cache button
        if st.button("🗑️ Clear Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != 'api_key':  # Keep API key
                    del st.session_state[key]
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results", "💡 Examples"])
    
    with tab1:
        st.header("Upload Your Document")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a PDF or TXT file",
                type=["pdf", "txt"],
                help="Upload the document you want to analyze"
            )
        
        with col2:
            st.info("""
            **Supported Formats:**
            - PDF documents
            - Plain text files
            
            **Best Results:**
            - Academic papers
            - Technical reports
            - Research proposals
            """)
        
        if uploaded_file:
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Process button
            if st.button("🚀 Analyze Document", type="primary", use_container_width=True):
                
                # Check if API key is provided
                if not st.session_state.api_key:
                    st.error("❌ Please enter your Google Gemini API key in the sidebar first!")
                    st.info("👉 Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
                    st.stop()
                
                # Reset session state for new analysis
                st.session_state.results_list = []
                st.session_state.current_progress = 0
                st.session_state.is_analyzing = True
                st.session_state.analysis_complete = False
                
                # Determine if using paragraph or sentence mode
                use_paragraphs = "Paragraph" in analysis_mode
                
                try:
                    # Process file
                    full_text, blocks, tmp_path = process_file(uploaded_file, file_type)
                    st.session_state.pdf_path = tmp_path if file_type == "pdf" else None
                    
                    # Segment into paragraphs or sentences
                    if use_paragraphs:
                        text_units = segment_paragraphs(full_text)
                        unit_name = "paragraph"
                        unit_name_plural = "paragraphs"
                    else:
                        text_units = segment_sentences(full_text)
                        unit_name = "sentence"
                        unit_name_plural = "sentences"
                    
                    st.session_state.total_sentences = len(text_units)
                    
                    st.success(f"📝 Found {len(text_units)} {unit_name_plural} to analyze")
                    
                    # Map text units to blocks
                    if use_paragraphs:
                        text_data = map_paragraphs_to_blocks(text_units, blocks)
                    else:
                        text_data = map_sentences_to_blocks(text_units, blocks)
                    
                    st.divider()
                    st.info(f"🔄 **Analysis in progress... Check results below!**")
                    
                    # Create live update containers
                    progress_container = st.container()
                    metrics_container = st.container()
                    table_container = st.container()
                    latest_container = st.container()
                    
                    # Progress bar and status
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                    # Metrics placeholders
                    with metrics_container:
                        col1, col2, col3, col4 = st.columns(4)
                        metric_analyzed = col1.empty()
                        metric_vague = col2.empty()
                        metric_clear = col3.empty()
                        metric_score = col4.empty()
                    
                    # Table placeholder
                    with table_container:
                        st.subheader("📊 Live Results")
                        table_placeholder = st.empty()
                    
                    # Latest unit placeholder
                    with latest_container:
                        st.subheader(f"🔍 Latest Analyzed {unit_name.title()}")
                        latest_placeholder = st.empty()
                    
                    # Analyze text units with live updates
                    for i, unit_info in enumerate(text_data):
                        # Get the text unit (paragraph or sentence)
                        text_unit = unit_info.get("paragraph") or unit_info.get("sentence")
                        
                        # Update progress
                        progress = (i + 1) / len(text_data)
                        progress_bar.progress(progress)
                        status_text.info(f"Analyzing {unit_name} {i + 1} of {len(text_data)}...")
                        
                        # Analyze using appropriate function with API key
                        if use_paragraphs:
                            analysis = analyze_paragraph(text_unit, api_key=st.session_state.api_key, model_name=model_choice)
                        else:
                            analysis = analyze_sentence(text_unit, api_key=st.session_state.api_key, model_name=model_choice)
                        
                        # Combine results
                        result = {
                            "Text": text_unit,
                            "Is Vague": "✅ Yes" if analysis["is_vague"] else "❌ No",
                            "Reason": analysis["reason"],
                            "Suggestion": analysis["suggestion"],
                            "Page": unit_info["page"],
                            "Location": f"Page {unit_info['page']}"
                        }
                        
                        # Add to results list
                        st.session_state.results_list.append(result)
                        st.session_state.current_progress = i + 1
                        
                        # Create DataFrame
                        current_df = pd.DataFrame(st.session_state.results_list)
                        st.session_state.results_df = current_df
                        
                        # Update metrics
                        total = len(current_df)
                        vague_count = len(current_df[current_df["Is Vague"] == "✅ Yes"])
                        clear_count = total - vague_count
                        vague_pct = (vague_count / total * 100) if total > 0 else 0
                        clarity_score = 100 - vague_pct
                        
                        metric_analyzed.metric("Analyzed", f"{total}/{len(text_data)}")
                        metric_vague.metric("Vague", vague_count, delta=f"{vague_pct:.1f}%")
                        metric_clear.metric("Clear", clear_count)
                        metric_score.metric("Clarity", f"{clarity_score:.1f}%")
                        
                        # Update table
                        table_placeholder.dataframe(
                            current_df,
                            use_container_width=True,
                            height=300,
                            column_config={
                                "Text": st.column_config.TextColumn(unit_name.title(), width="large"),
                                "Is Vague": st.column_config.TextColumn("Vague?", width="small"),
                                "Reason": st.column_config.TextColumn("Reason", width="medium"),
                                "Suggestion": st.column_config.TextColumn("Suggestion", width="medium"),
                                "Page": st.column_config.NumberColumn("Page", width="small"),
                            }
                        )
                        
                        # Update latest text unit - show more text for paragraphs
                        preview_length = 300 if use_paragraphs else 200
                        text_preview = text_unit[:preview_length]
                        if len(text_unit) > preview_length:
                            text_preview += "..."
                        
                        if result["Is Vague"] == "✅ Yes":
                            latest_placeholder.markdown(
                                f'<div class="vague-sentence"><strong>"{text_preview}"</strong><br>'
                                f'<em>❌ Vague - {result["Reason"]}</em><br>'
                                f'<small>Page {result["Page"]}</small></div>', 
                                unsafe_allow_html=True
                            )
                        else:
                            latest_placeholder.markdown(
                                f'<div class="clear-sentence"><strong>"{text_preview}"</strong><br>'
                                f'<em>✅ Clear and precise</em><br>'
                                f'<small>Page {result["Page"]}</small></div>', 
                                unsafe_allow_html=True
                            )
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Mark analysis as complete
                    st.session_state.analysis_complete = True
                    st.session_state.is_analyzing = False
                    
                    st.success("✅ Analysis complete! Check the 'Results' tab for full details and export options.")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
                    st.session_state.is_analyzing = False
    
    with tab2:
        st.header("Analysis Results")
        
        # Show results if available
        if st.session_state.results_df is not None and len(st.session_state.results_list) > 0:
            df = st.session_state.results_df
            
            # Show completion status
            if st.session_state.analysis_complete:
                st.success("✅ Analysis Complete!")
            else:
                st.info("ℹ️ No active analysis. Results from previous analysis shown below.")
            
            st.markdown("---")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_sentences = len(df)
            vague_count = len(df[df["Is Vague"] == "✅ Yes"])
            clear_count = total_sentences - vague_count
            vague_percentage = (vague_count / total_sentences * 100) if total_sentences > 0 else 0
            
            with col1:
                st.metric("Total Analyzed", total_sentences)
            with col2:
                st.metric("Vague Items", vague_count, delta=f"{vague_percentage:.1f}%")
            with col3:
                st.metric("Clear Items", clear_count)
            with col4:
                clarity_score = 100 - vague_percentage
                st.metric("Clarity Score", f"{clarity_score:.1f}%")
            
            st.divider()
            
            # Filter options
            col1, col2 = st.columns([1, 3])
            
            with col1:
                filter_option = st.selectbox(
                    "Filter Results",
                    ["All Items", "Vague Only", "Clear Only"],
                    key="filter_selector"
                )
            
            # Apply filter
            if filter_option == "Vague Only":
                filtered_df = df[df["Is Vague"] == "✅ Yes"]
            elif filter_option == "Clear Only":
                filtered_df = df[df["Is Vague"] == "❌ No"]
            else:
                filtered_df = df
            
            st.subheader(f"📋 {filter_option} ({len(filtered_df)} items)")
            
            # Display results table
            # Determine column name based on what's in the dataframe
            text_col_name = "Text" if "Text" in filtered_df.columns else "Sentence"
            
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=400,
                column_config={
                    text_col_name: st.column_config.TextColumn("Content", width="large"),
                    "Is Vague": st.column_config.TextColumn("Vague?", width="small"),
                    "Reason": st.column_config.TextColumn("Reason", width="medium"),
                    "Suggestion": st.column_config.TextColumn("Suggestion", width="medium"),
                    "Page": st.column_config.NumberColumn("Page", width="small"),
                    "Location": st.column_config.TextColumn("Location", width="small")
                }
            )
            
            # Export options
            st.divider()
            st.subheader("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name="vagueness_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_str = filtered_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📋 Download as JSON",
                    data=json_str,
                    file_name="vagueness_analysis.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                # Summary text report
                text_col_name = "Text" if "Text" in df.columns else "Sentence"
                unit_type = "Paragraphs" if "Text" in df.columns else "Sentences"
                
                summary = f"""# Vagueness Analysis Report

## Summary
- Total {unit_type}: {total_sentences}
- Vague {unit_type}: {vague_count} ({vague_percentage:.1f}%)
- Clear {unit_type}: {clear_count}
- Clarity Score: {clarity_score:.1f}%

## Vague {unit_type} Found:

"""
                for idx, row in df[df["Is Vague"] == "✅ Yes"].iterrows():
                    summary += f"\n### Item (Page {row['Page']})\n"
                    summary += f"**Text:** {row[text_col_name]}\n\n"
                    summary += f"**Reason:** {row['Reason']}\n\n"
                    summary += f"**Suggestion:** {row['Suggestion']}\n\n"
                    summary += "---\n"
                
                st.download_button(
                    label="📝 Download Report (MD)",
                    data=summary,
                    file_name="vagueness_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
        else:
            st.info("👈 Upload and analyze a document in the 'Upload & Analyze' tab")
            st.markdown("""
            ### What to expect:
            
            Once you start analyzing a document, you'll see **live updates** right in the Upload tab:
            
            #### During Analysis:
            - ⚡ **Real-time progress bar**
            - 📊 **Live metrics** (analyzed count, vague/clear counts, clarity score)
            - 📋 **Growing results table** 
            - 🔍 **Latest analyzed item** with color coding
            
            #### Analysis Modes:
            - **Paragraph-based (Recommended)**: Analyzes text in paragraph chunks for better context and more accurate results
            - **Sentence-based**: Analyzes individual sentences (may mark more items as vague due to lack of context)
            
            #### After Analysis:
            - Come to this tab for the **complete results**
            - **Filter** by vague or clear items
            - **Export** results in multiple formats (CSV, JSON, Markdown)
            - **Download** a detailed report
            
            The live updates happen **in the Upload & Analyze tab** so you can watch the progress!
            """)
    
    with tab3:
        st.header("Examples of Vague Language")
        
        st.info("💡 **Tip:** Using **Paragraph-based analysis** (recommended) provides better context and more accurate results!")
        
        st.divider()
        
        # Sentence-level examples
        st.subheader("📝 Sentence-Level Examples")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ❌ Vague Statements")
            
            examples_vague = [
                ("The system is very efficient.", "Subjective qualifier without metrics"),
                ("The experiment lasted for some time.", "Imprecise time reference"),
                ("Many users reported improvements.", "Ambiguous quantifier"),
                ("The method performs quite well.", "Hedging without specifics"),
                ("Results were somewhat better.", "Vague comparative")
            ]
            
            for sentence, reason in examples_vague:
                with st.container():
                    st.markdown(f'<div class="vague-sentence"><strong>{sentence}</strong><br><em>{reason}</em></div>', 
                              unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ✅ Clear Statements")
            
            examples_clear = [
                ("The system reduced processing time by 35%.", "Specific metric"),
                ("The experiment lasted for 10 hours.", "Precise time reference"),
                ("127 users (65% of participants) reported improvements.", "Exact numbers"),
                ("The method achieved 98% accuracy on the test set.", "Measurable outcome"),
                ("Results improved from 72% to 89% accuracy.", "Specific comparison")
            ]
            
            for sentence, reason in examples_clear:
                with st.container():
                    st.markdown(f'<div class="clear-sentence"><strong>{sentence}</strong><br><em>{reason}</em></div>', 
                              unsafe_allow_html=True)
        
        st.divider()
        
        # Paragraph-level examples
        st.subheader("📄 Paragraph-Level Examples")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ❌ Vague Paragraph")
            
            vague_para = """Our system performs very well in most scenarios. Users have reported 
            significant improvements compared to the old system. The processing time is much faster, 
            and the accuracy is quite good. We believe this approach could be beneficial for many 
            applications in the near future."""
            
            st.markdown(f'<div class="vague-sentence"><strong>{vague_para}</strong><br><br>'
                       '<em><strong>Issues:</strong> Contains subjective terms ("very well", "significant", '
                       '"much faster", "quite good"), vague quantifiers ("many applications"), and imprecise '
                       'timeframes ("near future").</em></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ✅ Clear Paragraph")
            
            clear_para = """Our system achieved 94% accuracy on the test dataset of 10,000 samples. 
            In user surveys, 87% of the 150 participants (n=130) reported time savings of 2-3 hours 
            per day compared to the legacy system. Processing time decreased from 45 seconds to 12 seconds 
            per request. Based on these results, we project deployment to 5 production environments by Q2 2025."""
            
            st.markdown(f'<div class="clear-sentence"><strong>{clear_para}</strong><br><br>'
                       '<em><strong>Strengths:</strong> Includes specific metrics (94% accuracy), exact numbers '
                       '(150 participants), concrete timeframes (Q2 2025), and measurable improvements '
                       '(45s to 12s).</em></div>', 
                       unsafe_allow_html=True)


if __name__ == "__main__":
    main()
