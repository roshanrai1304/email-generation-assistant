"""
Tests for email generation module
"""
import pytest
from src.email_generator import EmailGenerator
from src.models.prompt_strategies import get_system_prompt, get_cot_prompt


def test_email_generator_initialization():
    """Test that EmailGenerator initializes correctly"""
    generator = EmailGenerator()
    assert generator is not None
    assert generator.client is not None
    assert generator.system_prompt is not None


def test_prompt_strategies():
    """Test that prompt strategies are properly defined"""
    system_prompt = get_system_prompt()
    assert len(system_prompt) > 0
    assert "professional email writer" in system_prompt.lower()

    cot_prompt = get_cot_prompt()
    assert len(cot_prompt) > 0
    assert "step-by-step" in cot_prompt.lower()


def test_scenario_structure():
    """Test that test scenarios have required fields"""
    from src.utils.helpers import load_json
    from src.config import TEST_SCENARIOS_FILE

    scenarios = load_json(TEST_SCENARIOS_FILE)

    assert len(scenarios) == 10, "Should have 10 test scenarios"

    for scenario in scenarios:
        assert "scenario_id" in scenario
        assert "intent" in scenario
        assert "key_facts" in scenario
        assert "tone" in scenario
        assert isinstance(scenario["key_facts"], list)
        assert len(scenario["key_facts"]) >= 3


def test_reference_emails_match_scenarios():
    """Test that reference emails match test scenarios"""
    from src.utils.helpers import load_json
    from src.config import TEST_SCENARIOS_FILE, REFERENCE_EMAILS_FILE

    scenarios = load_json(TEST_SCENARIOS_FILE)
    references = load_json(REFERENCE_EMAILS_FILE)

    assert len(scenarios) == len(references), "Scenarios and references should match"

    scenario_ids = {s["scenario_id"] for s in scenarios}
    reference_ids = {r["scenario_id"] for r in references}

    assert scenario_ids == reference_ids, "Scenario IDs should match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
