"""
Email Generation Assistant - Streamlit Web UI

A comprehensive web interface for:
- Interactive email generation
- Results visualization
- Model comparison
- Evaluation dashboards
"""
import os
import sys

# Suppress all warnings BEFORE any imports
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Redirect stderr temporarily to suppress import warnings
import io
old_stderr = sys.stderr
sys.stderr = io.StringIO()

import streamlit as st

# Restore stderr after streamlit import
sys.stderr = old_stderr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.email_generator import EmailGenerator
from src.evaluation.evaluator import EmailEvaluator
from src.models.groq_client import GroqClient
from src.config import (
    PRIMARY_MODEL,
    SECONDARY_MODEL,
    TEST_SCENARIOS_FILE,
    REFERENCE_EMAILS_FILE,
    MODEL_A_RESULTS_FILE,
    MODEL_B_RESULTS_FILE,
    COMPARISON_CSV_FILE
)
from src.utils.helpers import load_json

# Page configuration
st.set_page_config(
    page_title="Email Generation Assistant",
    page_icon="✉️",
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
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📧 Email Assistant")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "✍️ Generate Email",
        "📊 Evaluation Dashboard",
        "🔄 Model Comparison",
        "📁 Results Viewer",
        "⚙️ Settings"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Documentation")
st.sidebar.markdown("""
- [README.md](README.md)
- [Architecture](ARCHITECTURE.md)
- [Report Guide](REPORT_GUIDE.md)
""")

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'generated_email' not in st.session_state:
    st.session_state.generated_email = None
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None

# ===========================
# HOME PAGE
# ===========================
if page == "🏠 Home":
    st.markdown('<div class="main-header">✉️ Email Generation Assistant</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 AI-Powered Email Generation with Custom Evaluation Metrics

    Welcome to the **Email Generation Assistant** - a comprehensive system that generates professional
    business emails and evaluates them using custom metrics.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 Advanced AI")
        st.markdown("""
        Uses **Groq API** with advanced prompting:
        - Role-Playing
        - Few-Shot Learning
        - Chain-of-Thought
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Custom Metrics")
        st.markdown("""
        3 tailored evaluation metrics:
        - Fact Inclusion Score
        - Tone Alignment Score
        - Professional Quality Score
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 Model Comparison")
        st.markdown("""
        Compare different models:
        - llama-3.3-70b-versatile
        - openai/gpt-oss-120b
        - Side-by-side analysis
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🚀 Quick Start")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 1. Generate Email
        Go to **✍️ Generate Email** to:
        - Input intent, facts, and tone
        - Generate professional emails
        - See real-time results
        """)

        st.markdown("""
        #### 2. View Evaluation
        Check **📊 Evaluation Dashboard** to:
        - See evaluation metrics
        - Visualize performance
        - Analyze results
        """)

    with col2:
        st.markdown("""
        #### 3. Compare Models
        Visit **🔄 Model Comparison** to:
        - Compare llama vs mixtral
        - View metric breakdowns
        - See performance charts
        """)

        st.markdown("""
        #### 4. Browse Results
        Explore **📁 Results Viewer** to:
        - View all test scenarios
        - Check generated emails
        - Download results
        """)

    st.markdown("---")

    st.markdown("### 📈 Project Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Test Scenarios", "10", delta=None)

    with col2:
        st.metric("Custom Metrics", "3", delta=None)

    with col3:
        st.metric("Models Compared", "2", delta=None)

    with col4:
        try:
            if Path(MODEL_A_RESULTS_FILE).exists():
                data = load_json(MODEL_A_RESULTS_FILE)
                avg_score = data['summary']['overall_average']
                st.metric("Best Model Score", f"{avg_score:.1f}/100", delta=None)
            else:
                st.metric("Best Model Score", "N/A", delta="Run evaluation")
        except:
            st.metric("Best Model Score", "N/A", delta="Run evaluation")

# ===========================
# GENERATE EMAIL PAGE
# ===========================
elif page == "✍️ Generate Email":
    st.markdown('<div class="main-header">✍️ Generate Professional Email</div>', unsafe_allow_html=True)

    st.markdown("""
    Generate professional business emails using advanced AI. Input your requirements below and get
    a well-crafted email instantly.
    """)

    st.markdown("---")

    # Input form
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Email Details")

        intent = st.text_input(
            "Intent/Purpose",
            placeholder="e.g., Follow up after sales meeting",
            help="What is the main purpose of this email?"
        )

        tone = st.selectbox(
            "Tone",
            ["formal", "casual", "urgent", "empathetic"],
            help="Select the desired tone for the email"
        )

        st.markdown("#### Key Facts to Include")
        fact1 = st.text_input("Fact 1", placeholder="Enter first key fact")
        fact2 = st.text_input("Fact 2", placeholder="Enter second key fact")
        fact3 = st.text_input("Fact 3", placeholder="Enter third key fact")
        fact4 = st.text_input("Fact 4 (optional)", placeholder="Enter fourth key fact")

    with col2:
        st.markdown("### ⚙️ Generation Settings")

        model = st.selectbox(
            "Model",
            [PRIMARY_MODEL, SECONDARY_MODEL],
            help="Select which LLM model to use"
        )

        include_few_shot = st.checkbox(
            "Include Few-Shot Examples",
            value=True,
            help="Include example emails in the prompt"
        )

        evaluate_output = st.checkbox(
            "Evaluate Generated Email",
            value=True,
            help="Automatically evaluate the generated email"
        )

    st.markdown("---")

    # Generate button
    if st.button("🚀 Generate Email", type="primary", use_container_width=True):
        # Validate inputs
        key_facts = [f for f in [fact1, fact2, fact3, fact4] if f.strip()]

        if not intent or len(key_facts) < 2:
            st.error("❌ Please provide an intent and at least 2 key facts!")
        else:
            with st.spinner("🤖 Generating email..."):
                try:
                    # Initialize generator
                    if st.session_state.generator is None:
                        st.session_state.generator = EmailGenerator()

                    # Generate email
                    generated_email = st.session_state.generator.generate(
                        intent=intent,
                        key_facts=key_facts,
                        tone=tone,
                        model=model,
                        include_few_shot=include_few_shot
                    )

                    st.session_state.generated_email = generated_email

                    # Display result
                    st.markdown("### ✅ Generated Email")
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown(generated_email)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Download button
                    st.download_button(
                        label="📥 Download Email",
                        data=generated_email,
                        file_name="generated_email.txt",
                        mime="text/plain"
                    )

                    # Evaluate if requested
                    if evaluate_output:
                        st.markdown("---")
                        st.markdown("### 📊 Evaluation Results")

                        with st.spinner("Evaluating email quality..."):
                            evaluator = EmailEvaluator()

                            scenario = {
                                "intent": intent,
                                "key_facts": key_facts,
                                "tone": tone,
                                "scenario_id": "custom"
                            }

                            result = evaluator.evaluate_single(scenario, generated_email)
                            st.session_state.evaluation_result = result

                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric(
                                    "Overall Score",
                                    f"{result['average_score']:.1f}/100"
                                )

                            with col2:
                                fact_score = result['metrics']['fact_inclusion']['score']
                                st.metric(
                                    "Fact Inclusion",
                                    f"{fact_score:.1f}/100"
                                )

                            with col3:
                                tone_score = result['metrics']['tone_alignment']['score']
                                st.metric(
                                    "Tone Alignment",
                                    f"{tone_score:.1f}/100"
                                )

                            with col4:
                                quality_score = result['metrics']['professional_quality']['score']
                                st.metric(
                                    "Prof. Quality",
                                    f"{quality_score:.1f}/100"
                                )

                            # Detailed metrics
                            with st.expander("📋 Detailed Metrics"):
                                st.json(result['metrics'])

                except Exception as e:
                    st.error(f"❌ Error generating email: {str(e)}")
                    st.exception(e)

# ===========================
# EVALUATION DASHBOARD PAGE
# ===========================
elif page == "📊 Evaluation Dashboard":
    st.markdown('<div class="main-header">📊 Evaluation Dashboard</div>', unsafe_allow_html=True)

    st.markdown("Visualize evaluation results and analyze model performance.")

    st.markdown("---")

    # Load results
    try:
        if Path(MODEL_A_RESULTS_FILE).exists():
            model_a_data = load_json(MODEL_A_RESULTS_FILE)
            model_a_results = model_a_data['results']
            model_a_summary = model_a_data['summary']

            # Summary metrics
            st.markdown("### 📈 Overall Performance")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Overall Average",
                    f"{model_a_summary['overall_average']:.2f}/100"
                )

            with col2:
                st.metric(
                    "Fact Inclusion",
                    f"{model_a_summary['metric_averages']['fact_inclusion']:.2f}/100"
                )

            with col3:
                st.metric(
                    "Tone Alignment",
                    f"{model_a_summary['metric_averages']['tone_alignment']:.2f}/100"
                )

            with col4:
                st.metric(
                    "Professional Quality",
                    f"{model_a_summary['metric_averages']['professional_quality']:.2f}/100"
                )

            st.markdown("---")

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Metric Distribution")

                # Create bar chart
                metrics_data = {
                    'Metric': ['Fact Inclusion', 'Tone Alignment', 'Professional Quality'],
                    'Score': [
                        model_a_summary['metric_averages']['fact_inclusion'],
                        model_a_summary['metric_averages']['tone_alignment'],
                        model_a_summary['metric_averages']['professional_quality']
                    ]
                }

                fig = px.bar(
                    metrics_data,
                    x='Metric',
                    y='Score',
                    title=f'Average Scores - {PRIMARY_MODEL}',
                    color='Score',
                    color_continuous_scale='Blues',
                    range_y=[0, 100]
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 🎯 Score by Scenario")

                # Create line chart
                scenario_scores = []
                for result in model_a_results[:10]:  # First 10 scenarios
                    scenario_scores.append({
                        'Scenario': f"S{result['scenario_id']}",
                        'Score': result['average_score']
                    })

                fig = px.line(
                    scenario_scores,
                    x='Scenario',
                    y='Score',
                    title='Performance Across Scenarios',
                    markers=True,
                    range_y=[0, 100]
                )

                fig.add_hline(
                    y=model_a_summary['overall_average'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Average"
                )

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Detailed results table
            st.markdown("### 📋 Detailed Results")

            # Prepare data for table
            table_data = []
            for result in model_a_results[:10]:
                table_data.append({
                    'Scenario': result['scenario_id'],
                    'Intent': result['intent'][:30] + '...' if len(result['intent']) > 30 else result['intent'],
                    'Tone': result['tone'],
                    'Fact Score': f"{result['metrics']['fact_inclusion']['score']:.1f}",
                    'Tone Score': f"{result['metrics']['tone_alignment']['score']:.1f}",
                    'Quality Score': f"{result['metrics']['professional_quality']['score']:.1f}",
                    'Overall': f"{result['average_score']:.1f}"
                })

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)

        else:
            st.warning("⚠️ No evaluation results found. Please run the evaluation first.")
            st.markdown("Run: `python main.py run-all`")

    except Exception as e:
        st.error(f"Error loading evaluation results: {str(e)}")

# ===========================
# MODEL COMPARISON PAGE
# ===========================
elif page == "🔄 Model Comparison":
    st.markdown('<div class="main-header">🔄 Model Comparison</div>', unsafe_allow_html=True)

    st.markdown("""
    Compare performance between **llama-3.3-70b-versatile** and **openai/gpt-oss-120b** models.
    """)

    st.markdown("---")

    try:
        if Path(COMPARISON_CSV_FILE).exists():
            # Load comparison data
            comparison_df = pd.read_csv(COMPARISON_CSV_FILE)
            model_a_data = load_json(MODEL_A_RESULTS_FILE)
            model_b_data = load_json(MODEL_B_RESULTS_FILE)

            model_a_summary = model_a_data['summary']
            model_b_summary = model_b_data['summary']

            # Summary comparison
            st.markdown("### 📊 Overall Comparison")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Model A (llama-3.3-70b)")
                st.metric(
                    "Overall Score",
                    f"{model_a_summary['overall_average']:.2f}/100"
                )
                st.metric(
                    "Fact Inclusion",
                    f"{model_a_summary['metric_averages']['fact_inclusion']:.2f}/100"
                )
                st.metric(
                    "Tone Alignment",
                    f"{model_a_summary['metric_averages']['tone_alignment']:.2f}/100"
                )
                st.metric(
                    "Prof. Quality",
                    f"{model_a_summary['metric_averages']['professional_quality']:.2f}/100"
                )

            with col2:
                st.markdown("#### Model B (gpt-oss-120b)")
                st.metric(
                    "Overall Score",
                    f"{model_b_summary['overall_average']:.2f}/100"
                )
                st.metric(
                    "Fact Inclusion",
                    f"{model_b_summary['metric_averages']['fact_inclusion']:.2f}/100"
                )
                st.metric(
                    "Tone Alignment",
                    f"{model_b_summary['metric_averages']['tone_alignment']:.2f}/100"
                )
                st.metric(
                    "Prof. Quality",
                    f"{model_b_summary['metric_averages']['professional_quality']:.2f}/100"
                )

            with col3:
                st.markdown("#### Difference (A - B)")
                diff_overall = model_a_summary['overall_average'] - model_b_summary['overall_average']
                diff_fact = model_a_summary['metric_averages']['fact_inclusion'] - model_b_summary['metric_averages']['fact_inclusion']
                diff_tone = model_a_summary['metric_averages']['tone_alignment'] - model_b_summary['metric_averages']['tone_alignment']
                diff_quality = model_a_summary['metric_averages']['professional_quality'] - model_b_summary['metric_averages']['professional_quality']

                st.metric(
                    "Overall Score",
                    f"{abs(diff_overall):.2f}",
                    delta=f"{diff_overall:+.2f}"
                )
                st.metric(
                    "Fact Inclusion",
                    f"{abs(diff_fact):.2f}",
                    delta=f"{diff_fact:+.2f}"
                )
                st.metric(
                    "Tone Alignment",
                    f"{abs(diff_tone):.2f}",
                    delta=f"{diff_tone:+.2f}"
                )
                st.metric(
                    "Prof. Quality",
                    f"{abs(diff_quality):.2f}",
                    delta=f"{diff_quality:+.2f}"
                )

            st.markdown("---")

            # Comparison charts
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Metric Comparison")

                # Create grouped bar chart
                metrics_comparison = {
                    'Metric': ['Fact Inclusion', 'Tone Alignment', 'Prof. Quality'] * 2,
                    'Model': ['Model A'] * 3 + ['Model B'] * 3,
                    'Score': [
                        model_a_summary['metric_averages']['fact_inclusion'],
                        model_a_summary['metric_averages']['tone_alignment'],
                        model_a_summary['metric_averages']['professional_quality'],
                        model_b_summary['metric_averages']['fact_inclusion'],
                        model_b_summary['metric_averages']['tone_alignment'],
                        model_b_summary['metric_averages']['professional_quality']
                    ]
                }

                fig = px.bar(
                    metrics_comparison,
                    x='Metric',
                    y='Score',
                    color='Model',
                    barmode='group',
                    title='Side-by-Side Metric Comparison',
                    color_discrete_map={'Model A': '#1f77b4', 'Model B': '#ff7f0e'}
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 📈 Performance by Scenario")

                # Create line chart comparing both models
                scenario_comparison = []
                for _, row in comparison_df.iterrows():
                    scenario_comparison.append({
                        'Scenario': f"S{row['scenario_id']}",
                        'Model': 'Model A',
                        'Score': row['model_a_average']
                    })
                    scenario_comparison.append({
                        'Scenario': f"S{row['scenario_id']}",
                        'Model': 'Model B',
                        'Score': row['model_b_average']
                    })

                fig = px.line(
                    scenario_comparison,
                    x='Scenario',
                    y='Score',
                    color='Model',
                    title='Scenario-by-Scenario Comparison',
                    markers=True,
                    color_discrete_map={'Model A': '#1f77b4', 'Model B': '#ff7f0e'}
                )

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Detailed comparison table
            st.markdown("### 📋 Detailed Comparison")

            display_df = comparison_df[[
                'scenario_id', 'intent', 'tone',
                'model_a_average', 'model_b_average', 'diff_average'
            ]].copy()

            display_df.columns = [
                'Scenario', 'Intent', 'Tone',
                'Model A Score', 'Model B Score', 'Difference'
            ]

            # Add styling
            st.dataframe(
                display_df.style.background_gradient(
                    subset=['Model A Score', 'Model B Score'],
                    cmap='RdYlGn',
                    vmin=0,
                    vmax=100
                ),
                use_container_width=True
            )

            # Download button
            st.download_button(
                label="📥 Download Comparison CSV",
                data=comparison_df.to_csv(index=False),
                file_name="model_comparison.csv",
                mime="text/csv"
            )

        else:
            st.warning("⚠️ No comparison data found. Please run the complete pipeline first.")
            st.markdown("Run: `python main.py run-all`")

    except Exception as e:
        st.error(f"Error loading comparison data: {str(e)}")

# ===========================
# RESULTS VIEWER PAGE
# ===========================
elif page == "📁 Results Viewer":
    st.markdown('<div class="main-header">📁 Results Viewer</div>', unsafe_allow_html=True)

    st.markdown("Browse test scenarios, generated emails, and reference emails.")

    st.markdown("---")

    try:
        # Load test scenarios and reference emails
        scenarios = load_json(TEST_SCENARIOS_FILE)
        references = load_json(REFERENCE_EMAILS_FILE)

        # Create scenario selector
        scenario_options = [f"Scenario {s['scenario_id']}: {s['intent']}" for s in scenarios]
        selected_scenario = st.selectbox("Select Scenario", scenario_options)

        # Get selected scenario index
        scenario_idx = scenario_options.index(selected_scenario)
        scenario = scenarios[scenario_idx]
        reference = references[scenario_idx]

        st.markdown("---")

        # Display scenario details
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📝 Scenario Details")
            st.markdown(f"**Intent:** {scenario['intent']}")
            st.markdown(f"**Tone:** {scenario['tone']}")
            st.markdown("**Key Facts:**")
            for i, fact in enumerate(scenario['key_facts'], 1):
                st.markdown(f"{i}. {fact}")

        with col2:
            st.markdown("### 🎯 Reference Email")
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(reference['reference_email'])
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Load generated emails if available
        if Path(MODEL_A_RESULTS_FILE).exists():
            model_a_data = load_json(MODEL_A_RESULTS_FILE)

            # Find the result for this scenario
            result_a = next(
                (r for r in model_a_data['results'] if r['scenario_id'] == scenario['scenario_id']),
                None
            )

            if result_a:
                st.markdown("### 🤖 Generated Email (Model A)")
                st.markdown(result_a['generated_email'])

                st.markdown("#### Evaluation Scores")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Overall", f"{result_a['average_score']:.1f}/100")
                with col2:
                    st.metric("Fact", f"{result_a['metrics']['fact_inclusion']['score']:.1f}/100")
                with col3:
                    st.metric("Tone", f"{result_a['metrics']['tone_alignment']['score']:.1f}/100")
                with col4:
                    st.metric("Quality", f"{result_a['metrics']['professional_quality']['score']:.1f}/100")

    except Exception as e:
        st.error(f"Error loading results: {str(e)}")

# ===========================
# SETTINGS PAGE
# ===========================
elif page == "⚙️ Settings":
    st.markdown('<div class="main-header">⚙️ Settings</div>', unsafe_allow_html=True)

    st.markdown("Configure the Email Generation Assistant.")

    st.markdown("---")

    st.markdown("### 🔑 API Configuration")

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **Note:** API keys should be set in the `.env` file for security.

    To configure:
    1. Copy `.env.example` to `.env`
    2. Add your Groq API key
    3. Restart the application
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 Current Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Primary Model:**")
        st.code(PRIMARY_MODEL)

        st.markdown("**Secondary Model:**")
        st.code(SECONDARY_MODEL)

    with col2:
        st.markdown("**Test Scenarios:**")
        st.code(str(TEST_SCENARIOS_FILE))

        st.markdown("**Results Directory:**")
        st.code("data/results/")

    st.markdown("---")

    st.markdown("### 📚 Resources")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Documentation")
        st.markdown("- [README.md](README.md)")
        st.markdown("- [ARCHITECTURE.md](ARCHITECTURE.md)")
        st.markdown("- [REPORT_GUIDE.md](REPORT_GUIDE.md)")

    with col2:
        st.markdown("#### Data Files")
        st.markdown("- Test Scenarios")
        st.markdown("- Reference Emails")
        st.markdown("- Evaluation Results")

    with col3:
        st.markdown("#### External Links")
        st.markdown("- [Groq Console](https://console.groq.com)")
        st.markdown("- [Groq Docs](https://console.groq.com/docs)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Email Generation Assistant</strong> v1.0</p>
    <p>Built with Streamlit • Powered by Groq API</p>
    <p>AI Engineer Candidate Assessment Project</p>
</div>
""", unsafe_allow_html=True)
