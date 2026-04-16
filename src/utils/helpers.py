"""
Utility helper functions
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


def load_json(filepath: str) -> any:
    """
    Load JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Parsed JSON data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: any, filepath: str, indent: int = 2):
    """
    Save data to JSON file

    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def create_comparison_dataframe(
    model_a_results: List[Dict],
    model_b_results: List[Dict]
) -> pd.DataFrame:
    """
    Create comparison DataFrame for two models

    Args:
        model_a_results: Results from model A
        model_b_results: Results from model B

    Returns:
        pandas DataFrame with side-by-side comparison
    """
    rows = []

    for res_a, res_b in zip(model_a_results, model_b_results):
        row = {
            "scenario_id": res_a.get("scenario_id", ""),
            "intent": res_a.get("intent", ""),
            "tone": res_a.get("tone", ""),

            # Model A scores
            "model_a_fact_score": res_a.get("metrics", {}).get("fact_inclusion", {}).get("score", 0),
            "model_a_tone_score": res_a.get("metrics", {}).get("tone_alignment", {}).get("score", 0),
            "model_a_quality_score": res_a.get("metrics", {}).get("professional_quality", {}).get("score", 0),
            "model_a_average": res_a.get("average_score", 0),

            # Model B scores
            "model_b_fact_score": res_b.get("metrics", {}).get("fact_inclusion", {}).get("score", 0),
            "model_b_tone_score": res_b.get("metrics", {}).get("tone_alignment", {}).get("score", 0),
            "model_b_quality_score": res_b.get("metrics", {}).get("professional_quality", {}).get("score", 0),
            "model_b_average": res_b.get("average_score", 0),
        }

        # Calculate differences
        row["diff_fact"] = row["model_a_fact_score"] - row["model_b_fact_score"]
        row["diff_tone"] = row["model_a_tone_score"] - row["model_b_tone_score"]
        row["diff_quality"] = row["model_a_quality_score"] - row["model_b_quality_score"]
        row["diff_average"] = row["model_a_average"] - row["model_b_average"]

        rows.append(row)

    return pd.DataFrame(rows)


def save_comparison_csv(
    model_a_results: List[Dict],
    model_b_results: List[Dict],
    filepath: str
):
    """
    Create and save comparison CSV

    Args:
        model_a_results: Results from model A
        model_b_results: Results from model B
        filepath: Output CSV file path
    """
    df = create_comparison_dataframe(model_a_results, model_b_results)
    df.to_csv(filepath, index=False)
    print(f"Comparison CSV saved to {filepath}")


def print_evaluation_summary(summary: Dict):
    """
    Print formatted evaluation summary

    Args:
        summary: Summary statistics dictionary
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Scenarios: {summary['total_scenarios']}")
    print(f"Successful Evaluations: {summary['successful_evaluations']}")
    print(f"\nOverall Average Score: {summary['overall_average']:.2f}/100")
    print("\nMetric Averages:")
    print(f"  • Fact Inclusion Score: {summary['metric_averages']['fact_inclusion']:.2f}/100")
    print(f"  • Tone Alignment Score: {summary['metric_averages']['tone_alignment']:.2f}/100")
    print(f"  • Professional Quality Score: {summary['metric_averages']['professional_quality']:.2f}/100")
    print("="*60 + "\n")


def print_comparison_summary(
    model_a_name: str,
    model_b_name: str,
    model_a_summary: Dict,
    model_b_summary: Dict
):
    """
    Print comparison summary between two models

    Args:
        model_a_name: Name of model A
        model_b_name: Name of model B
        model_a_summary: Summary stats for model A
        model_b_summary: Summary stats for model B
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"\nModel A: {model_a_name}")
    print(f"Model B: {model_b_name}")
    print("\n" + "-"*60)
    print(f"{'Metric':<30} {'Model A':<15} {'Model B':<15}")
    print("-"*60)

    metrics = [
        ("Overall Average", "overall_average"),
        ("Fact Inclusion", "fact_inclusion"),
        ("Tone Alignment", "tone_alignment"),
        ("Professional Quality", "professional_quality")
    ]

    for label, key in metrics:
        if key == "overall_average":
            val_a = model_a_summary.get(key, 0)
            val_b = model_b_summary.get(key, 0)
        else:
            val_a = model_a_summary.get("metric_averages", {}).get(key, 0)
            val_b = model_b_summary.get("metric_averages", {}).get(key, 0)

        diff = val_a - val_b
        winner = "✓" if diff > 0 else ""
        print(f"{label:<30} {val_a:>6.2f} {winner:<8} {val_b:>6.2f}")

    print("="*60 + "\n")
