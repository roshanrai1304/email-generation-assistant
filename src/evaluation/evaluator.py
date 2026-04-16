"""
Main Evaluation Module

This module orchestrates the evaluation of generated emails using custom metrics.
"""
import json
from typing import List, Dict
from tqdm import tqdm
from src.evaluation.metrics import (
    FactInclusionScore,
    ToneAlignmentScore,
    ProfessionalQualityScore
)
from src.models.groq_client import GroqClient


class EmailEvaluator:
    """Main evaluator class for email generation assessment"""

    def __init__(self, groq_client: GroqClient = None):
        """
        Initialize evaluator with custom metrics

        Args:
            groq_client: GroqClient instance (created if not provided)
        """
        self.groq_client = groq_client or GroqClient()

        # Initialize metrics
        self.fact_metric = FactInclusionScore()
        self.tone_metric = ToneAlignmentScore(self.groq_client)
        self.quality_metric = ProfessionalQualityScore(self.groq_client)

    def evaluate_single(
        self,
        scenario: Dict,
        generated_email: str,
        reference_email: str = None
    ) -> Dict:
        """
        Evaluate a single generated email

        Args:
            scenario: Input scenario with intent, key_facts, tone
            generated_email: Generated email text
            reference_email: Optional reference email for comparison

        Returns:
            Evaluation results dictionary
        """
        # Extract scenario data
        intent = scenario.get("intent", "")
        key_facts = scenario.get("key_facts", [])
        tone = scenario.get("tone", "")
        scenario_id = scenario.get("scenario_id", 0)

        # Calculate metrics
        fact_result = self.fact_metric.calculate(key_facts, generated_email)
        tone_result = self.tone_metric.calculate(tone, generated_email)
        quality_result = self.quality_metric.calculate(generated_email)

        # Calculate average score
        average_score = (
            fact_result["score"] +
            tone_result["score"] +
            quality_result["score"]
        ) / 3

        return {
            "scenario_id": scenario_id,
            "intent": intent,
            "tone": tone,
            "generated_email": generated_email,
            "reference_email": reference_email,
            "metrics": {
                "fact_inclusion": {
                    "score": fact_result["score"],
                    "facts_included": fact_result["facts_included"],
                    "total_facts": fact_result["total_facts"],
                    "details": fact_result["details"]
                },
                "tone_alignment": {
                    "score": tone_result["score"],
                    "justification": tone_result["justification"]
                },
                "professional_quality": {
                    "score": quality_result["score"],
                    "sub_scores": quality_result["sub_scores"],
                    "details": quality_result["details"]
                }
            },
            "average_score": round(average_score, 2)
        }

    def evaluate_batch(
        self,
        scenarios: List[Dict],
        generated_emails: List[str],
        reference_emails: List[str] = None
    ) -> List[Dict]:
        """
        Evaluate multiple emails

        Args:
            scenarios: List of input scenarios
            generated_emails: List of generated email texts
            reference_emails: Optional list of reference emails

        Returns:
            List of evaluation results
        """
        if reference_emails is None:
            reference_emails = [None] * len(scenarios)

        results = []

        print("Evaluating generated emails...")
        for i, (scenario, generated, reference) in enumerate(
            tqdm(zip(scenarios, generated_emails, reference_emails), total=len(scenarios))
        ):
            try:
                result = self.evaluate_single(scenario, generated, reference)
                results.append(result)
            except Exception as e:
                print(f"\nError evaluating scenario {i+1}: {str(e)}")
                results.append({
                    "scenario_id": scenario.get("scenario_id", i+1),
                    "error": str(e),
                    "average_score": 0
                })

        return results

    def generate_summary(self, evaluation_results: List[Dict]) -> Dict:
        """
        Generate summary statistics from evaluation results

        Args:
            evaluation_results: List of evaluation result dictionaries

        Returns:
            Summary statistics dictionary
        """
        if not evaluation_results:
            return {
                "total_scenarios": 0,
                "overall_average": 0,
                "metric_averages": {}
            }

        total_scenarios = len(evaluation_results)

        # Calculate average scores
        fact_scores = []
        tone_scores = []
        quality_scores = []
        overall_scores = []

        for result in evaluation_results:
            if "error" not in result:
                metrics = result.get("metrics", {})
                fact_scores.append(metrics.get("fact_inclusion", {}).get("score", 0))
                tone_scores.append(metrics.get("tone_alignment", {}).get("score", 0))
                quality_scores.append(metrics.get("professional_quality", {}).get("score", 0))
                overall_scores.append(result.get("average_score", 0))

        summary = {
            "total_scenarios": total_scenarios,
            "successful_evaluations": len(overall_scores),
            "overall_average": round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else 0,
            "metric_averages": {
                "fact_inclusion": round(sum(fact_scores) / len(fact_scores), 2) if fact_scores else 0,
                "tone_alignment": round(sum(tone_scores) / len(tone_scores), 2) if tone_scores else 0,
                "professional_quality": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0
            },
            "metric_definitions": self.get_metric_definitions()
        }

        return summary

    @staticmethod
    def get_metric_definitions() -> List[Dict]:
        """
        Get definitions of all custom metrics

        Returns:
            List of metric definition dictionaries
        """
        return [
            {
                "metric_name": "Fact Inclusion Score",
                "definition": "Measures what percentage of key facts from the input are present in the generated email",
                "logic": "Uses semantic similarity (sentence-transformers) to detect facts that may be paraphrased. Each fact is encoded and compared against all email sentences. If max similarity >= 0.75, fact is considered included. Score = (Facts included / Total facts) × 100",
                "score_range": "0-100",
                "max_score": 100,
                "weight": "33.3%"
            },
            {
                "metric_name": "Tone Alignment Score",
                "definition": "Measures how well the generated email matches the requested tone (formal, casual, urgent, empathetic, etc.)",
                "logic": "Uses LLM-as-a-Judge approach. A separate LLM call evaluates the email based on: formality level, urgency indicators, emotional tone, and word choice appropriateness. Returns score 0-100 with justification.",
                "score_range": "0-100",
                "max_score": 100,
                "weight": "33.3%"
            },
            {
                "metric_name": "Professional Quality Score",
                "definition": "Measures overall professional quality of the email using multiple dimensions",
                "logic": "Hybrid automated + LLM approach with weighted sub-metrics: Grammar Check (20%) - automated error detection, Structure Check (20%) - has greeting/closing/body, Conciseness (20%) - appropriate length (100-250 words optimal), Clarity (40%) - LLM-as-Judge for readability and coherence. Final score = weighted average.",
                "score_range": "0-100",
                "max_score": 100,
                "weight": "33.3%"
            }
        ]

    def save_results(self, results: List[Dict], summary: Dict, filepath: str):
        """
        Save evaluation results to JSON file

        Args:
            results: Evaluation results list
            summary: Summary statistics
            filepath: Output file path
        """
        output = {
            "summary": summary,
            "results": results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {filepath}")
