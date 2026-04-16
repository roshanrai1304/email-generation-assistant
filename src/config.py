"""
Configuration module for Email Generation Assistant
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file")

# Model Configuration
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "llama-3.3-70b-versatile")
SECONDARY_MODEL = os.getenv("SECONDARY_MODEL", "openai/gpt-oss-120b")

# Generation Settings
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Evaluation Settings
SEMANTIC_SIMILARITY_THRESHOLD = 0.75
BATCH_SIZE = 10

# Test Data Files
TEST_SCENARIOS_FILE = DATA_DIR / "test_scenarios.json"
REFERENCE_EMAILS_FILE = DATA_DIR / "reference_emails.json"

# Results Files
MODEL_A_RESULTS_FILE = RESULTS_DIR / "model_a_results.json"
MODEL_B_RESULTS_FILE = RESULTS_DIR / "model_b_results.json"
COMPARISON_CSV_FILE = RESULTS_DIR / "comparison.csv"
EVALUATION_REPORT_FILE = RESULTS_DIR / "evaluation_report.json"
