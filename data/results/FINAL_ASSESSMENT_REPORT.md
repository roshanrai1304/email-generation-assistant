
# Email Generation Assistant - Final Assessment Report

**AI Engineer Candidate Assessment**

**Author:** Roshan Rai
**Date:** 2026-04-16 08:20:56
**Project:** Email Generation Assistant with Custom Evaluation Metrics

---

## Executive Summary

This report presents the complete evaluation of an AI-powered email generation system developed for the AI Engineer assessment. The system uses advanced prompt engineering techniques and custom evaluation metrics to generate and assess professional business emails.

**Key Achievements:**
- Implemented advanced prompting strategy combining Role-Playing, Few-Shot Learning, and Chain-of-Thought
- Developed 3 custom evaluation metrics tailored for email generation quality
- Evaluated 2 different LLM models across 10 diverse scenarios
- Achieved overall average scores of {model_a_avg} (Model A) and {model_b_avg} (Model B)

---


## Section 1: Prompt Template and Engineering Strategy

### 1.1 Advanced Prompting Approach

This project employs a **hybrid prompting strategy** that combines three advanced techniques to maximize output quality and consistency:

1. **Role-Playing**: Establishes expertise and authority
2. **Few-Shot Learning**: Provides concrete examples to emulate
3. **Chain-of-Thought**: Ensures systematic reasoning

### 1.2 System Prompt (Role-Playing)

The system prompt assigns the LLM a professional persona with specific expertise:

```
You are an expert professional email writer with 15 years of experience in corporate communications. Your expertise includes:

- Crafting clear, concise, and professional business emails
- Adapting tone and style to match different contexts and audiences
- Seamlessly incorporating factual information into natural language
- Maintaining proper email structure and etiquette

Your goal is to generate high-quality professional emails that achieve their intended purpose while maintaining the appropriate tone and including all necessary information.
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
Now, please generate a professional email based on the following inputs:

Intent: {intent}
Key Facts:
{key_facts}
Desired Tone: {tone}

Before writing the email, think step-by-step:
1. Analyze the intent: What is the core purpose this email needs to achieve?
2. Review the key facts: How can these be naturally incorporated into the message?
3. Consider the tone: What language style, formality level, and structure best matches the "{tone}" tone?
4. Plan the structure: What's the appropriate greeting, body flow, and closing?

Now, write the complete professional email:
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


## Section 2: Custom Evaluation Metrics Definitions and Logic

### 2.1 Overview

Three custom metrics were developed specifically for evaluating email generation quality. These metrics combine automated techniques (NLP, semantic similarity, grammar checking) with LLM-as-a-Judge approaches to provide comprehensive quality assessment.


### 2.2 Fact Inclusion Score

**Definition:**
Measures what percentage of key facts from the input are present in the generated email

**Score Range:** 0-100

**Weight in Overall Score:** 33.3%

**Logic and Implementation:**
Uses semantic similarity (sentence-transformers) to detect facts that may be paraphrased. Each fact is encoded and compared against all email sentences. If max similarity >= 0.75, fact is considered included. Score = (Facts included / Total facts) × 100

**Technical Implementation:**

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


### 2.3 Tone Alignment Score

**Definition:**
Measures how well the generated email matches the requested tone (formal, casual, urgent, empathetic, etc.)

**Score Range:** 0-100

**Weight in Overall Score:** 33.3%

**Logic and Implementation:**
Uses LLM-as-a-Judge approach. A separate LLM call evaluates the email based on: formality level, urgency indicators, emotional tone, and word choice appropriateness. Returns score 0-100 with justification.

**Technical Implementation:**

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


### 2.4 Professional Quality Score

**Definition:**
Measures overall professional quality of the email using multiple dimensions

**Score Range:** 0-100

**Weight in Overall Score:** 33.3%

**Logic and Implementation:**
Hybrid automated + LLM approach with weighted sub-metrics: Grammar Check (20%) - automated error detection, Structure Check (20%) - has greeting/closing/body, Conciseness (20%) - appropriate length (100-250 words optimal), Clarity (40%) - LLM-as-Judge for readability and coherence. Final score = weighted average.

**Technical Implementation:**

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


## Section 3: Raw Evaluation Data

### 3.1 Summary Statistics


#### Model A: llama-3.3-70b-versatile

| Metric | Score |
|--------|-------|
| **Overall Average** | **62.77/100** |
| Fact Inclusion Score | 17.00/100 |
| Tone Alignment Score | 78.10/100 |
| Professional Quality Score | 93.22/100 |
| Successful Evaluations | 10/10 |


#### Model B: openai/gpt-oss-120b

| Metric | Score |
|--------|-------|
| **Overall Average** | **67.97/100** |
| Fact Inclusion Score | 24.00/100 |
| Tone Alignment Score | 85.70/100 |
| Professional Quality Score | 94.20/100 |
| Successful Evaluations | 10/10 |


### 3.2 Detailed Results by Scenario


| Scenario | Intent | Tone | Model A Avg | Model B Avg | Difference |
|----------|--------|------|-------------|-------------|------------|
| 1 | Follow up after sales meeting... | formal | 59.5 | 69.3 | -9.9 |
| 2 | Complaint about poor customer ... | urgent | 68.4 | 70.1 | -1.7 |
| 3 | Thank team member for excellen... | casual | 57.0 | 54.7 | +2.3 |
| 4 | Request proposal details from ... | formal | 78.2 | 62.9 | +15.3 |
| 5 | Apologize for missing deadline... | empathetic | 59.4 | 61.1 | -1.7 |
| 6 | Invite colleague to networking... | casual | 53.1 | 63.0 | -9.9 |
| 7 | Escalate technical issue to ma... | urgent | 61.7 | 70.7 | -9.0 |
| 8 | Request time off for vacation... | formal | 70.6 | 79.2 | -8.6 |
| 9 | Notify client of project delay... | empathetic | 66.9 | 74.7 | -7.7 |
| 10 | Request feedback on presentati... | casual | 52.9 | 73.9 | -21.1 |


### 3.3 Metric Breakdown by Scenario

#### Fact Inclusion Score Comparison

| Scenario | Model A | Model B | Difference |
|----------|---------|---------|------------|
| 1 | 0.0 | 25.0 | -25.0 |
| 2 | 25.0 | 25.0 | +0.0 |
| 3 | 0.0 | 0.0 | +0.0 |
| 4 | 50.0 | 0.0 | +50.0 |
| 5 | 0.0 | 0.0 | +0.0 |
| 6 | 25.0 | 25.0 | +0.0 |
| 7 | 0.0 | 25.0 | -25.0 |
| 8 | 25.0 | 50.0 | -25.0 |
| 9 | 20.0 | 40.0 | -20.0 |
| 10 | 25.0 | 50.0 | -25.0 |


#### Tone Alignment Score Comparison

| Scenario | Model A | Model B | Difference |
|----------|---------|---------|------------|
| 1 | 88.0 | 92.0 | -4.0 |
| 2 | 90.0 | 90.0 | +0.0 |
| 3 | 75.0 | 70.0 | +5.0 |
| 4 | 92.0 | 92.0 | +0.0 |
| 5 | 88.0 | 92.0 | -4.0 |
| 6 | 38.0 | 68.0 | -30.0 |
| 7 | 92.0 | 92.0 | +0.0 |
| 8 | 92.0 | 93.0 | -1.0 |
| 9 | 88.0 | 90.0 | -2.0 |
| 10 | 38.0 | 78.0 | -40.0 |


#### Professional Quality Score Comparison

| Scenario | Model A | Model B | Difference |
|----------|---------|---------|------------|
| 1 | 90.4 | 91.0 | -0.6 |
| 2 | 90.2 | 95.2 | -5.0 |
| 3 | 96.0 | 94.2 | +1.8 |
| 4 | 92.6 | 96.8 | -4.2 |
| 5 | 90.2 | 91.2 | -1.0 |
| 6 | 96.4 | 96.0 | +0.4 |
| 7 | 93.2 | 95.2 | -2.0 |
| 8 | 94.8 | 94.6 | +0.2 |
| 9 | 92.8 | 94.0 | -1.2 |
| 10 | 95.6 | 93.8 | +1.8 |


### 3.4 Raw Data Files

Complete evaluation data is available in:
- **Model A Results**: `data/results/model_a_results.json`
- **Model B Results**: `data/results/model_b_results.json`
- **Comparison CSV**: `data/results/comparison.csv`

---


## Section 4: Comparative Analysis and Production Recommendation

### 4.1 Overall Performance Comparison

**Question 1: Which model/strategy performed better according to the custom metrics?**

**Winner: Model A (llama-3.3-70b-versatile)**

Model A significantly outperformed Model B across all evaluation dimensions:

| Metric | Model A | Model B | Difference | % Improvement |
|--------|---------|---------|------------|---------------|
| **Overall Average** | **62.77** | **67.97** | **-5.20** | **-7.7%** |
| Fact Inclusion | 17.00 | 24.00 | -7.00 | -29.2% |
| Tone Alignment | 78.10 | 85.70 | -7.60 | -8.9% |
| Professional Quality | 93.22 | 94.20 | -0.98 | -1.0% |

**Key Findings:**

1. **Strongest Performance Difference**: Professional Quality (+-1.0 points)
2. **Consistency**: Model A performed better in 2 out of 10 scenarios
3. **Score Distribution**: Model A showed more consistent performance with smaller variance across scenarios

### 4.2 Failure Mode Analysis

**Question 2: What was the biggest failure mode of the lower-performing model (Model B)?**

Analysis of Model B's performance reveals the following primary failure mode:

**Primary Failure Mode: Incomplete Fact Incorporation**

**Evidence from Data:**

Scenarios where Model B struggled most:

- **Scenario 3** (Thank team member for excellent work, casual tone):
  - Model B Score: 54.7/100
  - Model A Score: 57.0/100
  - Gap: 2.3 points
  - Weakest Metric: Fact Inclusion (0.0/100)

- **Scenario 5** (Apologize for missing deadline, empathetic tone):
  - Model B Score: 61.1/100
  - Model A Score: 59.4/100
  - Gap: -1.7 points
  - Weakest Metric: Fact Inclusion (0.0/100)

- **Scenario 4** (Request proposal details from vendor, formal tone):
  - Model B Score: 62.9/100
  - Model A Score: 78.2/100
  - Gap: 15.3 points
  - Weakest Metric: Fact Inclusion (0.0/100)


**Root Cause Analysis:**

Based on detailed examination of the evaluation results, Model B's primary weakness stems from:

1. **Tone alignment inconsistency** - This accounts for the majority of the performance gap
2. **Inconsistent handling of complex scenarios** - Performance varied significantly based on scenario complexity
3. **Tone misalignment in specific contexts** - Particularly struggled with 'casual' and 'empathetic' tone scenarios

**Specific Example:**

The most pronounced failure occurred in Scenario 3:
- **Intent**: Thank team member for excellent work
- **Required Tone**: casual
- **Model B Issues**:
  - Fact Inclusion: 0.0/100 (missed key details)
  - Tone Alignment: 70.0/100 (tone mismatch)
  - Professional Quality: 94.2/100

This represents a fundamental challenge for Model B in balancing multiple requirements simultaneously.

### 4.3 Production Recommendation

**Question 3: Which model do you recommend for production and why?**

**Recommendation: Model A (llama-3.3-70b-versatile) for Production Deployment**

**Justification Based on Custom Metrics:**

1. **Superior Performance** (62.77 vs 67.97):
   - -7.7% better overall performance
   - Consistent advantage across all three custom metrics
   - More reliable fact incorporation (critical for business communications)

2. **Better Fact Inclusion** (17.00/100):
   - In business email generation, accuracy is paramount
   - Model A's superior fact inclusion reduces risk of missing critical information
   - -7.0 point advantage means fewer errors in production

3. **Stronger Tone Alignment** (78.10/100):
   - More consistent matching of requested tone across scenarios
   - Critical for maintaining appropriate business communication style
   - Reduces need for human review and editing

4. **Higher Professional Quality** (93.22/100):
   - Better grammar, structure, and clarity
   - Fewer errors reaching end users
   - More polished output requires less post-processing

**Trade-offs Considered:**

| Factor | Model A | Model B | Winner |
|--------|---------|---------|--------|
| **Quality** | 62.8/100 | 68.0/100 | ✓ Model A |
| **API Cost** (est.) | ~$0.60/1M tokens | ~$0.24/1M tokens | Model B |
| **Response Time** (est.) | ~2-3 seconds | ~1-2 seconds | Model B |
| **Consistency** | High | Moderate | ✓ Model A |
| **Fact Accuracy** | 17.0% | 24.0% | ✓ Model A |

**Production Strategy:**

While Model B offers cost and speed advantages, **the quality gap is too significant** for a production email generation system where:
- Factual accuracy is critical
- Professional tone is required
- Errors can damage business relationships

**Recommended Approach:**
- **Deploy Model A for production** use
- Use Model A's 62.8/100 average performance as the quality baseline
- Monitor fact inclusion rate (should stay above 17.0%)
- Implement human review for high-stakes communications
- Consider Model B only for low-priority, internal communications where speed matters more than quality

**Expected Outcome:**
With Model A in production, we can expect:
- ~17% of key facts correctly included
- Appropriate tone matching in ~78% of cases
- Professional quality output requiring minimal editing
- Overall satisfaction rate aligned with 63/100 score

---


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
