"""
Email Generation Module

This module orchestrates email generation using advanced prompt engineering
and the Groq API.
"""
from typing import Dict, List, Optional
from src.models.groq_client import GroqClient
from src.models.prompt_strategies import (
    get_system_prompt,
    get_cot_prompt,
    FEW_SHOT_EXAMPLES
)
from src.config import PRIMARY_MODEL, SECONDARY_MODEL


class EmailGenerator:
    """Main email generation class using advanced prompting"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize email generator

        Args:
            api_key: Groq API key (optional, uses config default)
        """
        self.client = GroqClient(api_key=api_key)
        self.system_prompt = get_system_prompt()
        self.user_prompt_template = get_cot_prompt()

    def generate(
        self,
        intent: str,
        key_facts: List[str],
        tone: str,
        model: str = PRIMARY_MODEL,
        include_few_shot: bool = True
    ) -> str:
        """
        Generate professional email based on inputs

        Args:
            intent: Purpose of the email
            key_facts: List of facts that must be included
            tone: Desired tone (formal, casual, urgent, empathetic, etc.)
            model: Model to use for generation
            include_few_shot: Whether to include few-shot examples

        Returns:
            Generated email text
        """
        # Format key facts as bullet points
        facts_text = "\n".join([f"- {fact}" for fact in key_facts])

        # Build user prompt
        user_prompt = self.user_prompt_template.format(
            intent=intent,
            key_facts=facts_text,
            tone=tone
        )

        # Add few-shot examples if requested
        if include_few_shot:
            user_prompt = FEW_SHOT_EXAMPLES + "\n\n" + user_prompt

        # Generate email
        response = self.client.generate_completion(
            prompt=user_prompt,
            model=model,
            system_prompt=self.system_prompt
        )

        return response["content"].strip()

    def generate_from_scenario(
        self,
        scenario: Dict,
        model: str = PRIMARY_MODEL,
        include_few_shot: bool = True
    ) -> str:
        """
        Generate email from a scenario dictionary

        Args:
            scenario: Dictionary with 'intent', 'key_facts', 'tone' keys
            model: Model to use for generation
            include_few_shot: Whether to include few-shot examples

        Returns:
            Generated email text
        """
        return self.generate(
            intent=scenario["intent"],
            key_facts=scenario["key_facts"],
            tone=scenario["tone"],
            model=model,
            include_few_shot=include_few_shot
        )

    def batch_generate(
        self,
        scenarios: List[Dict],
        model: str = PRIMARY_MODEL,
        include_few_shot: bool = True
    ) -> List[Dict]:
        """
        Generate emails for multiple scenarios

        Args:
            scenarios: List of scenario dictionaries
            model: Model to use for generation
            include_few_shot: Whether to include few-shot examples

        Returns:
            List of results with scenario data and generated emails
        """
        results = []

        for i, scenario in enumerate(scenarios, 1):
            print(f"Generating email {i}/{len(scenarios)}...")

            try:
                generated_email = self.generate_from_scenario(
                    scenario=scenario,
                    model=model,
                    include_few_shot=include_few_shot
                )

                results.append({
                    "scenario_id": scenario.get("scenario_id", i),
                    "intent": scenario["intent"],
                    "key_facts": scenario["key_facts"],
                    "tone": scenario["tone"],
                    "generated_email": generated_email,
                    "model": model,
                    "status": "success"
                })

            except Exception as e:
                print(f"Error generating email {i}: {str(e)}")
                results.append({
                    "scenario_id": scenario.get("scenario_id", i),
                    "intent": scenario["intent"],
                    "key_facts": scenario["key_facts"],
                    "tone": scenario["tone"],
                    "generated_email": None,
                    "model": model,
                    "status": "error",
                    "error": str(e)
                })

        return results


def create_email_generator(api_key: Optional[str] = None) -> EmailGenerator:
    """
    Factory function to create EmailGenerator instance

    Args:
        api_key: Optional Groq API key

    Returns:
        EmailGenerator instance
    """
    return EmailGenerator(api_key=api_key)
