"""
Tests for custom evaluation metrics
"""
import pytest
from src.evaluation.metrics import FactInclusionScore


def test_fact_inclusion_score_initialization():
    """Test that FactInclusionScore initializes correctly"""
    metric = FactInclusionScore()
    assert metric is not None
    assert metric.model is not None
    assert metric.threshold == 0.75


def test_fact_inclusion_perfect_match():
    """Test fact inclusion with perfect matches"""
    metric = FactInclusionScore()

    key_facts = [
        "Meeting held on April 10, 2026",
        "Discussed CRM implementation"
    ]

    email = """
    Dear Client,

    Thank you for the meeting held on April 10, 2026.
    We discussed CRM implementation during our conversation.

    Best regards
    """

    result = metric.calculate(key_facts, email)

    assert result["score"] >= 80  # Should have high score for good matches
    assert result["total_facts"] == 2
    assert result["facts_included"] >= 1  # At least one fact should match


def test_fact_inclusion_no_match():
    """Test fact inclusion with no matches"""
    metric = FactInclusionScore()

    key_facts = [
        "Meeting held on April 10, 2026",
        "Quoted $50,000"
    ]

    email = """
    Dear Client,

    Thank you for your inquiry. We look forward to working with you.

    Best regards
    """

    result = metric.calculate(key_facts, email)

    assert result["score"] < 50  # Should have low score for poor matches
    assert result["total_facts"] == 2


def test_fact_inclusion_empty_facts():
    """Test fact inclusion with no facts"""
    metric = FactInclusionScore()

    result = metric.calculate([], "Some email text")

    assert result["score"] == 100  # No facts to include = perfect score
    assert result["total_facts"] == 0
    assert result["facts_included"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
