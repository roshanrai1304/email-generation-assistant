# Email Generation Assistant

AI-powered email generation system with custom evaluation metrics for the AI Engineer Candidate Assessment.

## 🎯 Project Overview

This project demonstrates:
- **Advanced Prompt Engineering**: Combines Role-Playing, Few-Shot Learning, and Chain-of-Thought techniques
- **Custom Evaluation Metrics**: Three tailored metrics for assessing email generation quality
- **Model Comparison**: Side-by-side evaluation of different LLMs using Groq API (free tier)
- **Production-Ready Code**: Clean architecture with comprehensive testing and documentation

## 📋 Features

### Email Generation
- Takes 3 inputs: **Intent**, **Key Facts**, **Tone**
- Generates professional emails using advanced prompting techniques
- Supports multiple LLM models via Groq API

### Custom Evaluation Metrics

#### 1. **Fact Inclusion Score** (0-100)
- **What**: Measures percentage of key facts present in generated email
- **How**: Semantic similarity using sentence-transformers
- **Logic**: Each fact is encoded and compared to email sentences; fact is "included" if max similarity ≥ 0.75

#### 2. **Tone Alignment Score** (0-100)
- **What**: Measures how well email matches requested tone (formal, casual, urgent, empathetic)
- **How**: LLM-as-a-Judge evaluates formality, urgency, emotion, word choice
- **Logic**: Separate LLM call scores tone alignment with justification

#### 3. **Professional Quality Score** (0-100)
- **What**: Measures overall email professionalism
- **How**: Hybrid approach with weighted sub-metrics:
  - Grammar Check (20%): Automated error detection
  - Structure Check (20%): Greeting, body, closing
  - Conciseness (20%): Optimal length (100-250 words)
  - Clarity (40%): LLM-as-Judge for readability

## 🏗️ Architecture

> **📘 For detailed architecture and implementation explanation, see [ARCHITECTURE.md](ARCHITECTURE.md)**

```
email-generation-assistant/
├── src/
│   ├── config.py                 # Configuration and settings
│   ├── email_generator.py        # Email generation orchestration
│   ├── report_generator.py       # Automated report generation
│   ├── models/
│   │   ├── groq_client.py        # Groq API wrapper
│   │   └── prompt_strategies.py  # Advanced prompting techniques
│   ├── evaluation/
│   │   ├── metrics.py            # Custom metric implementations
│   │   └── evaluator.py          # Evaluation orchestration
│   └── utils/
│       └── helpers.py            # Utility functions
├── data/
│   ├── test_scenarios.json       # 10 test scenarios
│   ├── reference_emails.json     # Human reference emails
│   └── results/                  # Evaluation outputs
├── main.py                       # CLI entry point
├── requirements.in               # Direct dependencies
└── requirements.txt              # Locked dependencies
```

**Key Architecture Documents**:
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete architecture and implementation guide
- **[REPORT_GUIDE.md](REPORT_GUIDE.md)** - Report generation and PDF conversion guide
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Implementation status and next steps

## 🚀 Setup

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Groq API key (free tier)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd "Email Generation Assistant"
```

2. **Install uv** (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Create virtual environment and install dependencies**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

4. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

5. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get your free Groq API key at: https://console.groq.com/

## 📖 Usage

### Option 1: Web UI (Streamlit) 🌟 **NEW!**

Launch the interactive web interface:

**Recommended (Clean - No Warnings):**
```bash
./run_streamlit.sh          # macOS/Linux
run_streamlit.bat           # Windows
```

**Alternative:**
```bash
streamlit run streamlit_app.py
```

> 💡 **Note:** Use the launch script for a clean experience without warnings. See [RUN_CLEAN.md](RUN_CLEAN.md) for details.

The web UI provides:
- 🏠 **Home** - Project overview and quick stats
- ✍️ **Generate Email** - Interactive email generation with real-time evaluation
- 📊 **Evaluation Dashboard** - Visualize results with charts and graphs
- 🔄 **Model Comparison** - Compare models side-by-side with visualizations
- 📁 **Results Viewer** - Browse scenarios and generated emails
- ⚙️ **Settings** - Configuration and resources

**Benefits of Web UI:**
- Interactive and user-friendly
- Real-time visualization
- No command-line needed
- Beautiful charts with Plotly
- Easy to demonstrate and present

### Option 2: Command Line Interface

Run the complete pipeline:
```bash
python main.py run-all
```

This will:
1. Generate emails with Model A (llama-3.3-70b-versatile)
2. Evaluate Model A results
3. Generate emails with Model B (openai/gpt-oss-120b)
4. Evaluate Model B results
5. Compare both models
6. **Generate final assessment report** (Markdown + HTML)

### Individual Commands

**Generate emails with a specific model:**
```bash
python main.py generate --model a  # Use primary model
python main.py generate --model b  # Use secondary model
```

**Evaluate generated emails:**
```bash
python main.py evaluate --model a  # Evaluate Model A
python main.py evaluate --model b  # Evaluate Model B
```

**Compare models:**
```bash
python main.py compare
```

**Generate final assessment report:**
```bash
python main.py report                    # Generate both formats
python main.py report --format markdown  # Markdown only
python main.py report --format html      # HTML only
```

### Output Files

- `data/results/model_a_generated.json` - Generated emails from Model A
- `data/results/model_a_results.json` - Evaluation results for Model A
- `data/results/model_b_generated.json` - Generated emails from Model B
- `data/results/model_b_results.json` - Evaluation results for Model B
- `data/results/comparison.csv` - Side-by-side comparison of both models
- `data/results/FINAL_ASSESSMENT_REPORT.md` - **Final report (Markdown)**
- `data/results/FINAL_ASSESSMENT_REPORT.html` - **Final report (HTML)**

## 🧪 Test Scenarios

The project includes 10 diverse test scenarios covering:
- **Business contexts**: Follow-ups, complaints, requests, apologies
- **Tones**: Formal, casual, urgent, empathetic
- **Fact complexity**: 3-5 facts per scenario

Example scenario:
```json
{
  "scenario_id": 1,
  "intent": "Follow up after sales meeting",
  "key_facts": [
    "Meeting held on April 10, 2026",
    "Discussed new CRM implementation",
    "Quoted $50,000 for the project",
    "Next step: Send detailed proposal by April 20"
  ],
  "tone": "formal"
}
```

## 📊 Automated Report Generation

This project includes a **comprehensive automated report generator** that creates the complete final assessment report required for submission.

### What's Generated

The automated report includes all 4 required deliverable sections:

1. **Prompt Template Section**
   - Complete system prompt with role-playing
   - All few-shot examples
   - Chain-of-thought template
   - Detailed explanation of the combined approach

2. **Custom Metrics Definitions**
   - Full definition and logic for each metric
   - Technical implementation details
   - Code examples and algorithms
   - Scoring methodology

3. **Raw Evaluation Data**
   - Summary statistics for both models
   - Detailed results for all 10 scenarios
   - Metric breakdown tables
   - Performance comparisons

4. **Comparative Analysis**
   - Which model performed better (with data)
   - Biggest failure mode analysis
   - Production recommendation with justification
   - Data-driven insights

### How to Generate

The report is automatically generated when you run the complete pipeline:

```bash
python main.py run-all  # Generates report at the end
```

Or generate it separately:

```bash
python main.py report  # Creates both Markdown and HTML versions
```

### Output Formats

- **Markdown** (`FINAL_ASSESSMENT_REPORT.md`) - GitHub-friendly, easy to edit
- **HTML** (`FINAL_ASSESSMENT_REPORT.html`) - Professional styling, print-ready

### Converting to PDF

See **[REPORT_GUIDE.md](REPORT_GUIDE.md)** for detailed instructions on:
- Converting HTML to PDF using Chrome (recommended)
- Using command-line tools (pandoc, wkhtmltopdf)
- Uploading to Google Docs

**Quick PDF Generation:**
1. Open `data/results/FINAL_ASSESSMENT_REPORT.html` in Chrome
2. Press `Ctrl+P` (or `Cmd+P` on Mac)
3. Select "Save as PDF"
4. Done! 🎉

## 🎓 Advanced Prompting Strategy

This project uses a **combined approach** of three advanced techniques:

### 1. Role-Playing
The system prompt assigns the LLM the role of an "expert professional email writer with 15 years of experience" to establish expertise and set expectations.

### 2. Few-Shot Learning
The prompt includes 3 diverse examples demonstrating:
- Different intents and tones
- Proper fact incorporation
- Professional email structure

### 3. Chain-of-Thought
The prompt explicitly asks the model to think step-by-step:
1. Analyze the intent
2. Review key facts
3. Consider the tone
4. Plan the structure
5. Write the email

**Why this approach?**
- **Role-Playing**: Sets the right expertise level
- **Few-Shot**: Provides concrete examples to emulate
- **Chain-of-Thought**: Ensures systematic processing and reduces errors

## 📊 Evaluation Results

After running the pipeline, you'll get:

### Summary Statistics
```
Total Scenarios: 10
Overall Average Score: 87.5/100

Metric Averages:
  • Fact Inclusion Score: 92.3/100
  • Tone Alignment Score: 85.1/100
  • Professional Quality Score: 85.2/100
```

### Detailed Results
- Per-scenario scores for all 3 metrics
- Fact inclusion details (which facts were included)
- Tone alignment justifications
- Professional quality sub-scores

### Model Comparison
- Side-by-side metric comparison
- Performance differences
- Failure mode analysis

## 🔧 Configuration

Edit `.env` to customize:
```bash
GROQ_API_KEY=your_api_key_here
PRIMARY_MODEL=llama-3.3-70b-versatile
SECONDARY_MODEL=openai/gpt-oss-120b
MAX_TOKENS=1024
TEMPERATURE=0.7
```

## 🧪 Testing

Run tests with pytest:
```bash
pytest tests/
```

Code formatting with black:
```bash
black src/ tests/
```

Code quality check:
```bash
pylint src/
```

## 📝 Deliverables

This project satisfies all assessment requirements:

✅ **Email Generation Assistant**
- Takes Intent, Key Facts, Tone as inputs
- Generates professional emails
- Uses advanced prompt engineering (documented)

✅ **Custom Evaluation Metrics**
- 3 custom metrics implemented
- Tailored specifically for email generation
- Combines automated + LLM-as-Judge approaches

✅ **Test Data**
- 10 unique scenarios
- 10 human reference emails

✅ **Model Comparison**
- Evaluates 2 different models
- Comparative analysis
- Production recommendation

✅ **Code Repository**
- Clean, well-documented code
- Easy setup and execution
- Comprehensive README

## 🤖 Groq API Free Tier

This project is optimized for Groq's free tier:
- **Rate limiting**: Built-in 2-second delays between requests
- **Model selection**: Uses smaller model (mixtral) for LLM-as-Judge tasks
- **Batch processing**: Sequential processing to avoid rate limits
- **Error handling**: Graceful handling of API errors

## 📚 Project Structure Details

### Core Modules

- **`src/config.py`**: Centralized configuration
- **`src/email_generator.py`**: Email generation orchestration
- **`src/models/groq_client.py`**: Groq API wrapper with rate limiting
- **`src/models/prompt_strategies.py`**: Advanced prompting implementation
- **`src/evaluation/metrics.py`**: Custom metric implementations
- **`src/evaluation/evaluator.py`**: Evaluation orchestration
- **`src/utils/helpers.py`**: Utility functions

### Data Files

- **`data/test_scenarios.json`**: 10 test scenarios
- **`data/reference_emails.json`**: 10 human-written reference emails
- **`data/results/`**: Generated emails and evaluation results

## 🎯 Next Steps

After running the evaluation:

1. **Review Results**: Check `data/results/comparison.csv`
2. **Write Analysis**: Create comparative analysis summary
3. **Generate Report**: Compile final PDF report with:
   - Prompt template
   - Metric definitions
   - Evaluation data
   - Comparative analysis

## 🙋 FAQ

**Q: How do I get a Groq API key?**
A: Visit https://console.groq.com/ and sign up for free.

**Q: Can I use different models?**
A: Yes! Edit `.env` to change `PRIMARY_MODEL` and `SECONDARY_MODEL` to any Groq-supported model.

**Q: How long does the full pipeline take?**
A: Approximately 10-15 minutes for both models (20 scenarios total with rate limiting).

**Q: Can I add my own test scenarios?**
A: Yes! Edit `data/test_scenarios.json` and `data/reference_emails.json`.

## 📄 License

This project is created for the AI Engineer Candidate Assessment.

## 👤 Author

Roshan Rai - AI Engineer Candidate

---

**Built with**: Python, Groq API, sentence-transformers, language-tool-python, pandas
