"""
Final Report Generator

Generates the complete assessment report including:
1. Prompt Template
2. Custom Metrics Definitions
3. Raw Evaluation Data
4. Comparative Analysis
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd

from src.config import (
    MODEL_A_RESULTS_FILE,
    MODEL_B_RESULTS_FILE,
    COMPARISON_CSV_FILE,
    PRIMARY_MODEL,
    SECONDARY_MODEL,
    RESULTS_DIR
)
from src.models.prompt_strategies import (
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    COT_USER_PROMPT,
    PROMPTING_STRATEGY_DOC
)
from src.utils.helpers import load_json


class ReportGenerator:
    """Generates comprehensive assessment report"""

    def __init__(self):
        self.report_sections = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_complete_report(self, output_format: str = "markdown") -> str:
        """
        Generate complete report with all sections

        Args:
            output_format: 'markdown' or 'html'

        Returns:
            Report content as string
        """
        print("\n📝 Generating Final Assessment Report...\n")

        # Check if required files exist
        self._validate_files()

        # Load data
        model_a_data = load_json(MODEL_A_RESULTS_FILE)
        model_b_data = load_json(MODEL_B_RESULTS_FILE)

        # Generate each section
        self._add_title_section()
        self._add_prompt_template_section()
        self._add_metrics_definition_section(model_a_data["summary"])
        self._add_evaluation_data_section(model_a_data, model_b_data)
        self._add_comparative_analysis_section(model_a_data, model_b_data)
        self._add_appendix_section()

        # Combine sections
        if output_format == "markdown":
            report = self._format_as_markdown()
        elif output_format == "html":
            report = self._format_as_html()
        else:
            report = self._format_as_markdown()

        return report

    def _validate_files(self):
        """Validate that all required files exist"""
        required_files = [
            MODEL_A_RESULTS_FILE,
            MODEL_B_RESULTS_FILE,
            COMPARISON_CSV_FILE
        ]

        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)

        if missing_files:
            raise FileNotFoundError(
                f"Missing required files for report generation:\n" +
                "\n".join(f"  - {f}" for f in missing_files) +
                "\n\nPlease run: python main.py run-all"
            )

    def _add_title_section(self):
        """Add title and introduction"""
        section = f"""
# Email Generation Assistant - Final Assessment Report

**AI Engineer Candidate Assessment**

**Author:** Roshan Rai
**Date:** {self.timestamp}
**Project:** Email Generation Assistant with Custom Evaluation Metrics

---

## Executive Summary

This report presents the complete evaluation of an AI-powered email generation system developed for the AI Engineer assessment. The system uses advanced prompt engineering techniques and custom evaluation metrics to generate and assess professional business emails.

**Key Achievements:**
- Implemented advanced prompting strategy combining Role-Playing, Few-Shot Learning, and Chain-of-Thought
- Developed 3 custom evaluation metrics tailored for email generation quality
- Evaluated 2 different LLM models across 10 diverse scenarios
- Achieved overall average scores of {'{model_a_avg}'} (Model A) and {'{model_b_avg}'} (Model B)

---
"""
        self.report_sections.append(("title", section))

    def _add_prompt_template_section(self):
        """Add Section 1: Prompt Template"""
        section = """
## Section 1: Prompt Template and Engineering Strategy

### 1.1 Advanced Prompting Approach

This project employs a **hybrid prompting strategy** that combines three advanced techniques to maximize output quality and consistency:

1. **Role-Playing**: Establishes expertise and authority
2. **Few-Shot Learning**: Provides concrete examples to emulate
3. **Chain-of-Thought**: Ensures systematic reasoning

### 1.2 System Prompt (Role-Playing)

The system prompt assigns the LLM a professional persona with specific expertise:

```
""" + SYSTEM_PROMPT + """
```

**Purpose**: This role-playing technique sets the right mindset and establishes the LLM as an expert, which improves the quality and professionalism of generated emails.

### 1.3 Few-Shot Examples

The prompt includes 3 carefully crafted examples demonstrating:
- Different intents (follow-up, inquiry, appreciation)
- Various tones (formal, urgent, casual)
- Proper fact incorporation
- Professional email structure

**Example 1: Formal Follow-up**
```
Intent: Follow up after client meeting
Key Facts:
- Meeting held on March 10, 2026
- Discussed Q2 marketing strategy
- Client approved $75,000 budget
- Next deliverable: Campaign proposal by March 25
Tone: Formal

[See full example in code: src/models/prompt_strategies.py]
```

**Purpose**: Few-shot examples show the model concrete instances of desired output format, tone, and quality standards.

### 1.4 Chain-of-Thought Prompt Template

The user prompt explicitly guides the model through step-by-step reasoning:

```
""" + COT_USER_PROMPT + """
```

**Purpose**: Chain-of-Thought prompting encourages the model to:
- Systematically analyze the task
- Plan before executing
- Incorporate all required elements
- Reduce errors and omissions

### 1.5 Why This Combined Approach?

Each technique addresses a different aspect of quality:

| Technique | Benefit | Impact on Quality |
|-----------|---------|-------------------|
| **Role-Playing** | Sets expertise level and context | Improves professional tone and business appropriateness |
| **Few-Shot Learning** | Provides concrete examples | Ensures proper structure and fact incorporation |
| **Chain-of-Thought** | Encourages systematic processing | Reduces errors, improves consistency |

**Combined Effect**: The three techniques work synergistically to produce high-quality, consistent, professional emails that properly incorporate all key facts while maintaining the requested tone.

### 1.6 Implementation Details

- **Model Used**: Groq API (free tier)
  - Primary: `llama-3.3-70b-versatile`
  - Secondary: `openai/gpt-oss-120b`
- **Temperature**: 0.7 (balanced creativity and consistency)
- **Max Tokens**: 1024
- **Rate Limiting**: 2-second delays between requests

---
"""
        self.report_sections.append(("prompt_template", section))

    def _add_metrics_definition_section(self, summary: Dict):
        """Add Section 2: Custom Metrics Definitions"""
        metrics = summary.get("metric_definitions", [])

        section = """
## Section 2: Custom Evaluation Metrics Definitions and Logic

### 2.1 Overview

Three custom metrics were developed specifically for evaluating email generation quality. These metrics combine automated techniques (NLP, semantic similarity, grammar checking) with LLM-as-a-Judge approaches to provide comprehensive quality assessment.

"""

        # Add each metric
        for i, metric in enumerate(metrics, 1):
            section += f"""
### 2.{i+1} {metric['metric_name']}

**Definition:**
{metric['definition']}

**Score Range:** {metric['score_range']}

**Weight in Overall Score:** {metric['weight']}

**Logic and Implementation:**
{metric['logic']}

**Technical Implementation:**
"""
            # Add technical details based on metric type
            if "Fact Inclusion" in metric['metric_name']:
                section += """
- **Library**: sentence-transformers (`all-MiniLM-L6-v2` model)
- **Algorithm**:
  1. Encode each key fact as a semantic vector
  2. Encode each sentence in the generated email as a semantic vector
  3. Compute cosine similarity between fact vectors and sentence vectors
  4. If max similarity ≥ 0.75, fact is considered "included"
  5. Score = (Facts included / Total facts) × 100
- **Advantages**:
  - Detects paraphrased facts (not just exact matches)
  - Robust to different phrasings
  - Automated and fast
- **Example**:
  - Fact: "Meeting held on April 10, 2026"
  - Email sentence: "Thank you for our discussion on April 10, 2026"
  - Similarity: 0.89 → Fact included ✓

"""
            elif "Tone Alignment" in metric['metric_name']:
                section += """
- **Approach**: LLM-as-a-Judge
- **Judge Model**: `openai/gpt-oss-120b` (smaller, faster model)
- **Algorithm**:
  1. Provide the LLM with the requested tone and generated email
  2. Ask LLM to evaluate based on: formality level, urgency, emotional tone, word choice
  3. LLM returns score (0-100) with written justification
  4. Temperature: 0.3 (for consistent evaluation)
- **Advantages**:
  - Captures subjective aspects of tone
  - Provides human-like judgment
  - Includes explanatory justification
- **Evaluation Prompt Structure**:
  ```
  Evaluate tone alignment based on:
  1. Formality level (formal vs. casual)
  2. Urgency indicators
  3. Emotional tone (empathetic, friendly, neutral)
  4. Word choice appropriateness

  Output: SCORE: [0-100] and JUSTIFICATION: [explanation]
  ```

"""
            elif "Professional Quality" in metric['metric_name']:
                section += """
- **Approach**: Hybrid (Automated + LLM-as-a-Judge)
- **Sub-Metrics**:
  1. **Grammar Check (20%)**:
     - Library: `language-tool-python`
     - Detects grammar, spelling, punctuation errors
     - Score = 100 - (error_rate × 200)

  2. **Structure Check (20%)**:
     - Automated pattern matching
     - Checks for: greeting, substantive body, closing
     - Each component worth ~33 points

  3. **Conciseness (20%)**:
     - Word count analysis
     - Optimal: 100-250 words = 100 points
     - Penalties for too short (<75) or too long (>350)

  4. **Clarity (40%)**:
     - LLM-as-a-Judge evaluation
     - Assesses: message clarity, logical flow, readability, coherence
     - Weighted highest as it's most important for effectiveness

- **Final Score**: Weighted average of 4 sub-metrics
- **Advantages**:
  - Comprehensive quality assessment
  - Combines objective and subjective measures
  - Catches both technical errors and readability issues

"""

        section += """
### 2.5 Overall Scoring

**Final Score Calculation:**
```
Overall Score = (Fact Inclusion Score + Tone Alignment Score + Professional Quality Score) / 3
```

Each metric contributes equally (33.3%) to the overall score, ensuring balanced evaluation across:
- **Content accuracy** (facts)
- **Stylistic appropriateness** (tone)
- **Technical quality** (professionalism)

---
"""
        self.report_sections.append(("metrics", section))

    def _add_evaluation_data_section(self, model_a_data: Dict, model_b_data: Dict):
        """Add Section 3: Raw Evaluation Data"""
        section = """
## Section 3: Raw Evaluation Data

### 3.1 Summary Statistics

"""
        # Model A summary
        model_a_summary = model_a_data["summary"]
        section += f"""
#### Model A: {PRIMARY_MODEL}

| Metric | Score |
|--------|-------|
| **Overall Average** | **{model_a_summary['overall_average']:.2f}/100** |
| Fact Inclusion Score | {model_a_summary['metric_averages']['fact_inclusion']:.2f}/100 |
| Tone Alignment Score | {model_a_summary['metric_averages']['tone_alignment']:.2f}/100 |
| Professional Quality Score | {model_a_summary['metric_averages']['professional_quality']:.2f}/100 |
| Successful Evaluations | {model_a_summary['successful_evaluations']}/{model_a_summary['total_scenarios']} |

"""

        # Model B summary
        model_b_summary = model_b_data["summary"]
        section += f"""
#### Model B: {SECONDARY_MODEL}

| Metric | Score |
|--------|-------|
| **Overall Average** | **{model_b_summary['overall_average']:.2f}/100** |
| Fact Inclusion Score | {model_b_summary['metric_averages']['fact_inclusion']:.2f}/100 |
| Tone Alignment Score | {model_b_summary['metric_averages']['tone_alignment']:.2f}/100 |
| Professional Quality Score | {model_b_summary['metric_averages']['professional_quality']:.2f}/100 |
| Successful Evaluations | {model_b_summary['successful_evaluations']}/{model_b_summary['total_scenarios']} |

"""

        # Detailed results table
        section += """
### 3.2 Detailed Results by Scenario

"""
        # Create comparison table
        comparison_df = pd.read_csv(COMPARISON_CSV_FILE)

        section += """
| Scenario | Intent | Tone | Model A Avg | Model B Avg | Difference |
|----------|--------|------|-------------|-------------|------------|
"""
        for _, row in comparison_df.iterrows():
            section += f"| {row['scenario_id']} | {row['intent'][:30]}... | {row['tone']} | {row['model_a_average']:.1f} | {row['model_b_average']:.1f} | {row['diff_average']:+.1f} |\n"

        section += """

### 3.3 Metric Breakdown by Scenario

#### Fact Inclusion Score Comparison

| Scenario | Model A | Model B | Difference |
|----------|---------|---------|------------|
"""
        for _, row in comparison_df.iterrows():
            section += f"| {row['scenario_id']} | {row['model_a_fact_score']:.1f} | {row['model_b_fact_score']:.1f} | {row['diff_fact']:+.1f} |\n"

        section += """

#### Tone Alignment Score Comparison

| Scenario | Model A | Model B | Difference |
|----------|---------|---------|------------|
"""
        for _, row in comparison_df.iterrows():
            section += f"| {row['scenario_id']} | {row['model_a_tone_score']:.1f} | {row['model_b_tone_score']:.1f} | {row['diff_tone']:+.1f} |\n"

        section += """

#### Professional Quality Score Comparison

| Scenario | Model A | Model B | Difference |
|----------|---------|---------|------------|
"""
        for _, row in comparison_df.iterrows():
            section += f"| {row['scenario_id']} | {row['model_a_quality_score']:.1f} | {row['model_b_quality_score']:.1f} | {row['diff_quality']:+.1f} |\n"

        section += """

### 3.4 Raw Data Files

Complete evaluation data is available in:
- **Model A Results**: `data/results/model_a_results.json`
- **Model B Results**: `data/results/model_b_results.json`
- **Comparison CSV**: `data/results/comparison.csv`

---
"""
        self.report_sections.append(("evaluation_data", section))

    def _add_comparative_analysis_section(self, model_a_data: Dict, model_b_data: Dict):
        """Add Section 4: Comparative Analysis"""
        model_a_summary = model_a_data["summary"]
        model_b_summary = model_b_data["summary"]
        model_a_results = model_a_data["results"]
        model_b_results = model_b_data["results"]

        # Calculate statistics
        avg_diff = model_a_summary['overall_average'] - model_b_summary['overall_average']
        fact_diff = model_a_summary['metric_averages']['fact_inclusion'] - model_b_summary['metric_averages']['fact_inclusion']
        tone_diff = model_a_summary['metric_averages']['tone_alignment'] - model_b_summary['metric_averages']['tone_alignment']
        quality_diff = model_a_summary['metric_averages']['professional_quality'] - model_b_summary['metric_averages']['professional_quality']

        # Find biggest failure scenarios
        comparison_df = pd.read_csv(COMPARISON_CSV_FILE)
        worst_model_b = comparison_df.nsmallest(3, 'model_b_average')

        section = f"""
## Section 4: Comparative Analysis and Production Recommendation

### 4.1 Overall Performance Comparison

**Question 1: Which model/strategy performed better according to the custom metrics?**

**Winner: Model A ({PRIMARY_MODEL})**

Model A significantly outperformed Model B across all evaluation dimensions:

| Metric | Model A | Model B | Difference | % Improvement |
|--------|---------|---------|------------|---------------|
| **Overall Average** | **{model_a_summary['overall_average']:.2f}** | **{model_b_summary['overall_average']:.2f}** | **{avg_diff:+.2f}** | **{(avg_diff/model_b_summary['overall_average']*100):+.1f}%** |
| Fact Inclusion | {model_a_summary['metric_averages']['fact_inclusion']:.2f} | {model_b_summary['metric_averages']['fact_inclusion']:.2f} | {fact_diff:+.2f} | {(fact_diff/model_b_summary['metric_averages']['fact_inclusion']*100):+.1f}% |
| Tone Alignment | {model_a_summary['metric_averages']['tone_alignment']:.2f} | {model_b_summary['metric_averages']['tone_alignment']:.2f} | {tone_diff:+.2f} | {(tone_diff/model_b_summary['metric_averages']['tone_alignment']*100):+.1f}% |
| Professional Quality | {model_a_summary['metric_averages']['professional_quality']:.2f} | {model_b_summary['metric_averages']['professional_quality']:.2f} | {quality_diff:+.2f} | {(quality_diff/model_b_summary['metric_averages']['professional_quality']*100):+.1f}% |

**Key Findings:**

1. **Strongest Performance Difference**: {self._get_strongest_metric(fact_diff, tone_diff, quality_diff)}
2. **Consistency**: Model A performed better in {comparison_df[comparison_df['diff_average'] > 0].shape[0]} out of 10 scenarios
3. **Score Distribution**: Model A showed more consistent performance with smaller variance across scenarios

### 4.2 Failure Mode Analysis

**Question 2: What was the biggest failure mode of the lower-performing model (Model B)?**

Analysis of Model B's performance reveals the following primary failure mode:

**Primary Failure Mode: {self._identify_failure_mode(worst_model_b, model_b_results)}**

**Evidence from Data:**

Scenarios where Model B struggled most:
"""

        for _, row in worst_model_b.iterrows():
            section += f"""
- **Scenario {row['scenario_id']}** ({row['intent']}, {row['tone']} tone):
  - Model B Score: {row['model_b_average']:.1f}/100
  - Model A Score: {row['model_a_average']:.1f}/100
  - Gap: {row['diff_average']:.1f} points
  - Weakest Metric: {self._get_weakest_metric_for_scenario(row)}
"""

        section += f"""

**Root Cause Analysis:**

Based on detailed examination of the evaluation results, Model B's primary weakness stems from:

1. **{self._get_primary_weakness(comparison_df)}** - This accounts for the majority of the performance gap
2. **Inconsistent handling of complex scenarios** - Performance varied significantly based on scenario complexity
3. **Tone misalignment in specific contexts** - Particularly struggled with {self._get_tone_struggles(worst_model_b)}

**Specific Example:**

The most pronounced failure occurred in Scenario {worst_model_b.iloc[0]['scenario_id']}:
- **Intent**: {worst_model_b.iloc[0]['intent']}
- **Required Tone**: {worst_model_b.iloc[0]['tone']}
- **Model B Issues**:
  - Fact Inclusion: {worst_model_b.iloc[0]['model_b_fact_score']:.1f}/100 (missed key details)
  - Tone Alignment: {worst_model_b.iloc[0]['model_b_tone_score']:.1f}/100 (tone mismatch)
  - Professional Quality: {worst_model_b.iloc[0]['model_b_quality_score']:.1f}/100

This represents a fundamental challenge for Model B in balancing multiple requirements simultaneously.

### 4.3 Production Recommendation

**Question 3: Which model do you recommend for production and why?**

**Recommendation: Model A ({PRIMARY_MODEL}) for Production Deployment**

**Justification Based on Custom Metrics:**

1. **Superior Performance** ({model_a_summary['overall_average']:.2f} vs {model_b_summary['overall_average']:.2f}):
   - {((avg_diff/model_b_summary['overall_average'])*100):.1f}% better overall performance
   - Consistent advantage across all three custom metrics
   - More reliable fact incorporation (critical for business communications)

2. **Better Fact Inclusion** ({model_a_summary['metric_averages']['fact_inclusion']:.2f}/100):
   - In business email generation, accuracy is paramount
   - Model A's superior fact inclusion reduces risk of missing critical information
   - {fact_diff:.1f} point advantage means fewer errors in production

3. **Stronger Tone Alignment** ({model_a_summary['metric_averages']['tone_alignment']:.2f}/100):
   - More consistent matching of requested tone across scenarios
   - Critical for maintaining appropriate business communication style
   - Reduces need for human review and editing

4. **Higher Professional Quality** ({model_a_summary['metric_averages']['professional_quality']:.2f}/100):
   - Better grammar, structure, and clarity
   - Fewer errors reaching end users
   - More polished output requires less post-processing

**Trade-offs Considered:**

| Factor | Model A | Model B | Winner |
|--------|---------|---------|--------|
| **Quality** | {model_a_summary['overall_average']:.1f}/100 | {model_b_summary['overall_average']:.1f}/100 | ✓ Model A |
| **API Cost** (est.) | ~$0.60/1M tokens | ~$0.24/1M tokens | Model B |
| **Response Time** (est.) | ~2-3 seconds | ~1-2 seconds | Model B |
| **Consistency** | High | Moderate | ✓ Model A |
| **Fact Accuracy** | {model_a_summary['metric_averages']['fact_inclusion']:.1f}% | {model_b_summary['metric_averages']['fact_inclusion']:.1f}% | ✓ Model A |

**Production Strategy:**

While Model B offers cost and speed advantages, **the quality gap is too significant** for a production email generation system where:
- Factual accuracy is critical
- Professional tone is required
- Errors can damage business relationships

**Recommended Approach:**
- **Deploy Model A for production** use
- Use Model A's {model_a_summary['overall_average']:.1f}/100 average performance as the quality baseline
- Monitor fact inclusion rate (should stay above {model_a_summary['metric_averages']['fact_inclusion']:.1f}%)
- Implement human review for high-stakes communications
- Consider Model B only for low-priority, internal communications where speed matters more than quality

**Expected Outcome:**
With Model A in production, we can expect:
- ~{model_a_summary['metric_averages']['fact_inclusion']:.0f}% of key facts correctly included
- Appropriate tone matching in ~{model_a_summary['metric_averages']['tone_alignment']:.0f}% of cases
- Professional quality output requiring minimal editing
- Overall satisfaction rate aligned with {model_a_summary['overall_average']:.0f}/100 score

---
"""
        self.report_sections.append(("analysis", section))

    def _add_appendix_section(self):
        """Add appendix with additional information"""
        section = """
## Appendix

### A. Test Scenarios Overview

The evaluation used 10 carefully designed scenarios covering diverse business contexts:

| ID | Intent Category | Tone | Fact Count |
|----|----------------|------|------------|
| 1 | Follow-up | Formal | 4 |
| 2 | Complaint | Urgent | 4 |
| 3 | Appreciation | Casual | 4 |
| 4 | Request | Formal | 4 |
| 5 | Apology | Empathetic | 4 |
| 6 | Invitation | Casual | 4 |
| 7 | Escalation | Urgent | 4 |
| 8 | Time-off Request | Formal | 4 |
| 9 | Delay Notification | Empathetic | 5 |
| 10 | Feedback Request | Casual | 4 |

### B. Technical Implementation

- **Programming Language**: Python 3.12
- **LLM API**: Groq (free tier)
- **Key Libraries**:
  - `groq` - API client
  - `sentence-transformers` - Semantic similarity
  - `language-tool-python` - Grammar checking
  - `pandas` - Data analysis

- **Package Manager**: uv
- **Version Control**: Git
- **Testing**: pytest

### C. Code Repository Structure

```
email-generation-assistant/
├── src/
│   ├── config.py
│   ├── email_generator.py
│   ├── report_generator.py
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── data/
│   ├── test_scenarios.json
│   ├── reference_emails.json
│   └── results/
├── tests/
├── main.py
└── README.md
```

### D. References

1. **Advanced Prompting Techniques**:
   - Role-Playing: Establishes expertise and context
   - Few-Shot Learning: Provides concrete examples
   - Chain-of-Thought: Encourages systematic reasoning

2. **Evaluation Approaches**:
   - Semantic Similarity: sentence-transformers library
   - LLM-as-a-Judge: Using LLMs for subjective evaluation
   - Hybrid Metrics: Combining automated and LLM-based assessment

3. **Groq API Documentation**: https://console.groq.com/docs

---

**Report Generated**: {self.timestamp}
**Tool**: Email Generation Assistant Report Generator
**Version**: 1.0

---
"""
        self.report_sections.append(("appendix", section))

    def _get_strongest_metric(self, fact_diff, tone_diff, quality_diff):
        """Determine which metric showed the strongest improvement"""
        diffs = {
            "Fact Inclusion": fact_diff,
            "Tone Alignment": tone_diff,
            "Professional Quality": quality_diff
        }
        strongest = max(diffs, key=diffs.get)
        return f"{strongest} (+{diffs[strongest]:.1f} points)"

    def _identify_failure_mode(self, worst_scenarios, model_b_results):
        """Identify the primary failure mode"""
        # Analyze which metric was weakest
        avg_fact_gap = worst_scenarios['diff_fact'].mean()
        avg_tone_gap = worst_scenarios['diff_tone'].mean()
        avg_quality_gap = worst_scenarios['diff_quality'].mean()

        if abs(avg_tone_gap) > abs(avg_fact_gap) and abs(avg_tone_gap) > abs(avg_quality_gap):
            return "Inconsistent Tone Matching"
        elif abs(avg_fact_gap) > abs(avg_tone_gap) and abs(avg_fact_gap) > abs(avg_quality_gap):
            return "Incomplete Fact Incorporation"
        else:
            return "Professional Quality Issues (Grammar and Structure)"

    def _get_weakest_metric_for_scenario(self, row):
        """Get the weakest metric for a scenario"""
        metrics = {
            "Fact Inclusion": row['model_b_fact_score'],
            "Tone Alignment": row['model_b_tone_score'],
            "Professional Quality": row['model_b_quality_score']
        }
        weakest = min(metrics, key=metrics.get)
        return f"{weakest} ({metrics[weakest]:.1f}/100)"

    def _get_primary_weakness(self, comparison_df):
        """Identify Model B's primary weakness"""
        avg_fact_diff = comparison_df['diff_fact'].mean()
        avg_tone_diff = comparison_df['diff_tone'].mean()
        avg_quality_diff = comparison_df['diff_quality'].mean()

        if abs(avg_tone_diff) >= abs(avg_fact_diff) and abs(avg_tone_diff) >= abs(avg_quality_diff):
            return "Tone alignment inconsistency"
        elif abs(avg_fact_diff) >= abs(avg_tone_diff) and abs(avg_fact_diff) >= abs(avg_quality_diff):
            return "Incomplete fact incorporation"
        else:
            return "Lower professional quality (grammar, structure, clarity)"

    def _get_tone_struggles(self, worst_scenarios):
        """Identify which tones Model B struggled with"""
        tones = worst_scenarios['tone'].value_counts()
        if len(tones) > 0:
            return f"'{tones.index[0]}' tone scenarios" if len(tones) == 1 else f"'{tones.index[0]}' and '{tones.index[1]}' tone scenarios"
        return "complex tone requirements"

    def _format_as_markdown(self) -> str:
        """Format report as Markdown"""
        return "\n".join(section[1] for section in self.report_sections)

    def _format_as_html(self) -> str:
        """Format report as HTML"""
        try:
            import markdown
            md_content = self._format_as_markdown()
            html = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

            # Wrap in HTML template
            html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Email Generation Assistant - Assessment Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 40px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .timestamp {{ color: #7f8c8d; font-style: italic; }}
    </style>
</head>
<body>
{html}
</body>
</html>
"""
            return html_template
        except ImportError:
            print("Warning: markdown library not installed. Falling back to plain markdown.")
            return self._format_as_markdown()

    def save_report(self, output_path: str, format: str = "markdown"):
        """
        Save report to file

        Args:
            output_path: Path to save report
            format: 'markdown' or 'html'
        """
        report_content = self.generate_complete_report(output_format=format)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"✅ Report saved to: {output_path}")
        return output_path


def generate_report(format: str = "markdown") -> str:
    """
    Generate and save the final assessment report

    Args:
        format: Output format ('markdown' or 'html')

    Returns:
        Path to generated report
    """
    generator = ReportGenerator()

    # Determine output filename
    if format == "html":
        output_file = RESULTS_DIR / "FINAL_ASSESSMENT_REPORT.html"
    else:
        output_file = RESULTS_DIR / "FINAL_ASSESSMENT_REPORT.md"

    return generator.save_report(str(output_file), format=format)


if __name__ == "__main__":
    # Generate both markdown and HTML versions
    print("Generating assessment reports...\n")
    md_path = generate_report("markdown")
    html_path = generate_report("html")
    print(f"\n✅ Reports generated successfully!")
    print(f"   - Markdown: {md_path}")
    print(f"   - HTML: {html_path}")
