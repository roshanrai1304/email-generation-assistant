"""
Custom Evaluation Metrics for Email Generation

This module implements three custom metrics:
1. Fact Inclusion Score - Measures how well key facts are incorporated
2. Tone Alignment Score - Measures adherence to requested tone
3. Professional Quality Score - Measures overall email quality
"""
from typing import List, Dict, Tuple
import re
import warnings
import os

# Suppress transformers warnings for vision models we don't use
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', category=FutureWarning)

from sentence_transformers import SentenceTransformer, util
from src.models.groq_client import GroqClient
from src.config import SEMANTIC_SIMILARITY_THRESHOLD


class FactInclusionScore:
    """
    Metric 1: Fact Inclusion Score

    Measures what percentage of key facts from the input are present in the generated email.
    Uses semantic similarity to detect facts that are paraphrased or reworded.

    Score Range: 0-100
    - 100: All facts included
    - 0: No facts included

    Logic:
    1. For each key fact, compute semantic similarity with all sentences in the email
    2. If max similarity >= threshold (0.75), fact is considered included
    3. Score = (Facts included / Total facts) × 100
    """

    def __init__(self):
        """Initialize with sentence transformer model for semantic similarity"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = SEMANTIC_SIMILARITY_THRESHOLD

    def calculate(self, key_facts: List[str], generated_email: str) -> Dict:
        """
        Calculate fact inclusion score

        Args:
            key_facts: List of facts that should be in email
            generated_email: Generated email text

        Returns:
            Dictionary with score and details
        """
        if not key_facts:
            return {
                "score": 100,
                "facts_included": 0,
                "total_facts": 0,
                "details": []
            }

        # Split email into sentences
        sentences = self._split_into_sentences(generated_email)

        # Encode facts and sentences
        fact_embeddings = self.model.encode(key_facts, convert_to_tensor=True)
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)

        facts_included = 0
        details = []

        for i, fact in enumerate(key_facts):
            # Compute similarity between this fact and all sentences
            similarities = util.pytorch_cos_sim(fact_embeddings[i], sentence_embeddings)[0]
            max_similarity = similarities.max().item()

            is_included = max_similarity >= self.threshold
            if is_included:
                facts_included += 1

            details.append({
                "fact": fact,
                "included": is_included,
                "max_similarity": round(max_similarity, 3)
            })

        score = (facts_included / len(key_facts)) * 100

        return {
            "score": round(score, 2),
            "facts_included": facts_included,
            "total_facts": len(key_facts),
            "details": details
        }

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


class ToneAlignmentScore:
    """
    Metric 2: Tone Alignment Score

    Measures how well the generated email matches the requested tone.
    Uses LLM-as-a-Judge to evaluate tone alignment.

    Score Range: 0-100
    - 100: Perfect tone alignment
    - 0: Complete mismatch

    Logic:
    1. Provide LLM with the requested tone and generated email
    2. Ask LLM to rate tone alignment on 0-100 scale
    3. LLM evaluates: formality, urgency, emotional tone, word choice
    4. Return numerical score with justification
    """

    def __init__(self, groq_client: GroqClient):
        """
        Initialize with Groq client for LLM-as-a-Judge

        Args:
            groq_client: GroqClient instance
        """
        self.client = groq_client

    def calculate(self, tone: str, generated_email: str, model: str = "openai/gpt-oss-120b") -> Dict:
        """
        Calculate tone alignment score using LLM-as-a-Judge

        Args:
            tone: Requested tone (e.g., "formal", "casual", "urgent")
            generated_email: Generated email text
            model: Model to use for judging (using smaller/faster model)

        Returns:
            Dictionary with score and justification
        """
        judge_prompt = f"""You are an expert evaluator of professional email writing. Your task is to rate how well an email's tone matches the requested tone.

Requested Tone: {tone}

Email to Evaluate:
{generated_email}

Please evaluate the tone alignment based on:
1. Formality level (formal vs. casual language)
2. Urgency indicators (if applicable)
3. Emotional tone (empathetic, friendly, neutral, etc.)
4. Word choice and phrasing appropriateness

Provide your evaluation in this EXACT format:
SCORE: [number from 0-100]
JUSTIFICATION: [2-3 sentences explaining your rating]

Example:
SCORE: 85
JUSTIFICATION: The email demonstrates strong alignment with the formal tone through professional language and structured format. Minor improvement could be made in the closing to be slightly more formal.
"""

        try:
            response = self.client.generate_completion(
                prompt=judge_prompt,
                model=model,
                temperature=0.3  # Lower temperature for more consistent evaluation
            )

            content = response["content"]

            # Parse score and justification
            score_match = re.search(r'SCORE:\s*(\d+)', content)
            justification_match = re.search(r'JUSTIFICATION:\s*(.+)', content, re.DOTALL)

            if score_match:
                score = int(score_match.group(1))
                score = max(0, min(100, score))  # Clamp to 0-100
            else:
                score = 0

            justification = justification_match.group(1).strip() if justification_match else content

            return {
                "score": score,
                "justification": justification,
                "raw_response": content
            }

        except Exception as e:
            return {
                "score": 0,
                "justification": f"Error during evaluation: {str(e)}",
                "raw_response": ""
            }


class ProfessionalQualityScore:
    """
    Metric 3: Professional Quality Score

    Measures overall professional quality of the email using multiple sub-metrics.
    Hybrid approach combining automated checks and LLM evaluation.

    Score Range: 0-100
    - 100: Excellent professional quality
    - 0: Poor quality

    Logic:
    1. Grammar Check (20%): LLM-as-Judge for grammar and spelling
    2. Structure Check (20%): Automated check for greeting, body, closing
    3. Conciseness (20%): Automated length check (optimal 100-250 words)
    4. Clarity (40%): LLM-as-Judge for readability and coherence

    Final Score = Weighted average of sub-scores

    Note: Uses LLM-based grammar checking instead of rule-based tools,
    providing more intelligent and context-aware evaluation without requiring Java.
    """

    def __init__(self, groq_client: GroqClient):
        """
        Initialize with LLM client for all evaluations

        Args:
            groq_client: GroqClient instance for grammar, clarity evaluation
        """
        self.client = groq_client

    def calculate(self, generated_email: str, model: str = "openai/gpt-oss-120b") -> Dict:
        """
        Calculate professional quality score

        Args:
            generated_email: Generated email text
            model: Model to use for clarity evaluation

        Returns:
            Dictionary with overall score and sub-scores
        """
        # Sub-metric 1: Grammar (20%)
        grammar_score = self._check_grammar(generated_email)

        # Sub-metric 2: Structure (20%)
        structure_score = self._check_structure(generated_email)

        # Sub-metric 3: Conciseness (20%)
        conciseness_score = self._check_conciseness(generated_email)

        # Sub-metric 4: Clarity (40%) - LLM-as-Judge
        clarity_result = self._check_clarity(generated_email, model)

        # Calculate weighted average
        overall_score = (
            grammar_score * 0.20 +
            structure_score * 0.20 +
            conciseness_score * 0.20 +
            clarity_result["score"] * 0.40
        )

        return {
            "score": round(overall_score, 2),
            "sub_scores": {
                "grammar": round(grammar_score, 2),
                "structure": round(structure_score, 2),
                "conciseness": round(conciseness_score, 2),
                "clarity": round(clarity_result["score"], 2)
            },
            "details": {
                "grammar_errors": clarity_result.get("grammar_errors", 0),
                "has_greeting": clarity_result.get("has_greeting", False),
                "has_closing": clarity_result.get("has_closing", False),
                "word_count": len(generated_email.split()),
                "clarity_justification": clarity_result.get("justification", "")
            }
        }

    def _check_grammar(self, text: str, model: str = "openai/gpt-oss-120b") -> float:
        """
        Check grammar and spelling using LLM-as-Judge

        Uses AI to evaluate grammar quality instead of rule-based checker
        This is more flexible and doesn't require Java

        Returns score 0-100 (better grammar = higher score)
        """
        try:
            grammar_prompt = f"""You are an expert grammar and spelling checker. Evaluate this text for grammar, spelling, and punctuation errors.

Text to evaluate:
{text}

Evaluate based on:
1. Grammar correctness
2. Spelling accuracy
3. Punctuation usage
4. Sentence structure

Provide your evaluation in this EXACT format:
SCORE: [0-100, where 100 is perfect, 0 is many errors]
ERRORS: [brief list of any errors found, or "None"]

Example:
SCORE: 92
ERRORS: Minor: missing comma after introductory phrase
"""

            response = self.client.generate_completion(
                prompt=grammar_prompt,
                model=model,
                temperature=0.2  # Low temperature for consistent evaluation
            )

            content = response["content"]

            # Parse score
            score_match = re.search(r'SCORE:\s*(\d+)', content)
            if score_match:
                score = int(score_match.group(1))
                score = max(0, min(100, score))  # Clamp to 0-100
            else:
                score = 85  # Default if parsing fails

            return float(score)

        except Exception as e:
            print(f"Grammar check error: {e}")
            return 85.0  # Default score on error

    def _check_structure(self, text: str) -> float:
        """
        Check if email has proper structure (greeting, body, closing)

        Returns score 0-100
        """
        score = 0

        # Check for greeting (Dear, Hi, Hello, etc.)
        greeting_patterns = [
            r'\b(Dear|Hi|Hello|Hey|Greetings)\b',
            r'^Subject:',
        ]
        has_greeting = any(re.search(pattern, text, re.IGNORECASE) for pattern in greeting_patterns)
        if has_greeting:
            score += 33.33

        # Check for closing (Best regards, Sincerely, etc.)
        closing_patterns = [
            r'\b(Best regards|Sincerely|Best|Regards|Cheers|Thanks|Thank you)\b.*$',
        ]
        has_closing = any(re.search(pattern, text, re.IGNORECASE | re.MULTILINE) for pattern in closing_patterns)
        if has_closing:
            score += 33.33

        # Check for substantive body content
        sentences = text.split('.')
        if len(sentences) >= 3:  # At least 3 sentences
            score += 33.34

        return min(100, score)

    def _check_conciseness(self, text: str) -> float:
        """
        Check if email is appropriately concise

        Returns score 0-100 (optimal: 100-250 words)
        """
        word_count = len(text.split())

        if 100 <= word_count <= 250:
            return 100
        elif 75 <= word_count < 100:
            return 85
        elif 250 < word_count <= 350:
            return 85
        elif 50 <= word_count < 75:
            return 70
        elif 350 < word_count <= 500:
            return 70
        else:
            return 50

    def _check_clarity(self, text: str, model: str) -> Dict:
        """
        Use LLM to evaluate clarity and coherence

        Returns score 0-100 with justification
        """
        clarity_prompt = f"""You are an expert evaluator of professional email writing. Evaluate the clarity and coherence of this email.

Email:
{text}

Evaluate based on:
1. Message clarity - Is the purpose clear?
2. Logical flow - Do ideas flow logically?
3. Readability - Is it easy to understand?
4. Coherence - Are sentences well-connected?

Provide your evaluation in this EXACT format:
SCORE: [number from 0-100]
JUSTIFICATION: [1-2 sentences explaining your rating]
"""

        try:
            response = self.client.generate_completion(
                prompt=clarity_prompt,
                model=model,
                temperature=0.3
            )

            content = response["content"]
            score_match = re.search(r'SCORE:\s*(\d+)', content)
            justification_match = re.search(r'JUSTIFICATION:\s*(.+)', content, re.DOTALL)

            score = int(score_match.group(1)) if score_match else 75
            score = max(0, min(100, score))

            justification = justification_match.group(1).strip() if justification_match else content

            return {
                "score": score,
                "justification": justification
            }
        except:
            return {
                "score": 75,
                "justification": "Error during clarity evaluation"
            }
