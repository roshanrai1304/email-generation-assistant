"""
Email Generation Assistant - Main CLI Entry Point

Usage:
    python main.py generate --model <model_name>
    python main.py evaluate --model <model_name>
    python main.py compare
    python main.py report [--format markdown|html]
    python main.py run-all
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.email_generator import EmailGenerator
from src.evaluation.evaluator import EmailEvaluator
from src.models.groq_client import GroqClient
from src.report_generator import generate_report
from src.utils.helpers import (
    load_json,
    save_json,
    save_comparison_csv,
    print_evaluation_summary,
    print_comparison_summary
)
from src.config import (
    PRIMARY_MODEL,
    SECONDARY_MODEL,
    TEST_SCENARIOS_FILE,
    REFERENCE_EMAILS_FILE,
    MODEL_A_RESULTS_FILE,
    MODEL_B_RESULTS_FILE,
    COMPARISON_CSV_FILE,
    EVALUATION_REPORT_FILE
)


def generate_emails(model_name: str, output_file: str):
    """
    Generate emails for all test scenarios

    Args:
        model_name: Name of model to use
        output_file: Where to save generated emails
    """
    print(f"\n{'='*60}")
    print(f"GENERATING EMAILS WITH MODEL: {model_name}")
    print(f"{'='*60}\n")

    # Load test scenarios
    scenarios = load_json(TEST_SCENARIOS_FILE)
    print(f"Loaded {len(scenarios)} test scenarios")

    # Initialize generator
    generator = EmailGenerator()

    # Generate emails
    results = generator.batch_generate(scenarios, model=model_name)

    # Save results
    save_json(results, output_file)
    print(f"\n✓ Generated emails saved to {output_file}\n")

    return results


def evaluate_emails(model_name: str, generated_emails_file: str, output_file: str):
    """
    Evaluate generated emails using custom metrics

    Args:
        model_name: Name of model used
        generated_emails_file: File with generated emails
        output_file: Where to save evaluation results
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING EMAILS FROM: {model_name}")
    print(f"{'='*60}\n")

    # Load data
    scenarios = load_json(TEST_SCENARIOS_FILE)
    reference_emails = load_json(REFERENCE_EMAILS_FILE)
    generated_results = load_json(generated_emails_file)

    # Extract generated email texts
    generated_texts = [r["generated_email"] for r in generated_results]
    reference_texts = [r["reference_email"] for r in reference_emails]

    # Initialize evaluator
    evaluator = EmailEvaluator()

    # Evaluate
    evaluation_results = evaluator.evaluate_batch(
        scenarios=scenarios,
        generated_emails=generated_texts,
        reference_emails=reference_texts
    )

    # Generate summary
    summary = evaluator.generate_summary(evaluation_results)

    # Save results
    evaluator.save_results(evaluation_results, summary, output_file)

    # Print summary
    print_evaluation_summary(summary)

    return evaluation_results, summary


def compare_models():
    """
    Compare two models and generate comparison report
    """
    print(f"\n{'='*60}")
    print("COMPARING MODELS")
    print(f"{'='*60}\n")

    # Check if both model results exist
    if not Path(MODEL_A_RESULTS_FILE).exists():
        print(f"❌ Model A results not found: {MODEL_A_RESULTS_FILE}")
        print("Run: python main.py evaluate --model a")
        return

    if not Path(MODEL_B_RESULTS_FILE).exists():
        print(f"❌ Model B results not found: {MODEL_B_RESULTS_FILE}")
        print("Run: python main.py evaluate --model b")
        return

    # Load results
    with open(MODEL_A_RESULTS_FILE) as f:
        model_a_data = load_json(MODEL_A_RESULTS_FILE)
        model_a_results = model_a_data["results"]
        model_a_summary = model_a_data["summary"]

    with open(MODEL_B_RESULTS_FILE) as f:
        model_b_data = load_json(MODEL_B_RESULTS_FILE)
        model_b_results = model_b_data["results"]
        model_b_summary = model_b_data["summary"]

    # Create comparison CSV
    save_comparison_csv(model_a_results, model_b_results, COMPARISON_CSV_FILE)

    # Print comparison
    print_comparison_summary(
        PRIMARY_MODEL,
        SECONDARY_MODEL,
        model_a_summary,
        model_b_summary
    )

    # Generate analysis
    print("\n📊 Detailed comparison saved to:", COMPARISON_CSV_FILE)
    print("\nNext step: Write comparative analysis summary")
    print("See COMPARISON_CSV_FILE for detailed metrics\n")


def run_all_pipeline():
    """
    Run complete pipeline: generate -> evaluate for both models -> compare
    """
    print("\n🚀 Running complete evaluation pipeline...\n")

    # Step 1: Generate with Model A
    print("Step 1/5: Generating emails with Model A...")
    generate_emails(PRIMARY_MODEL, str(MODEL_A_RESULTS_FILE).replace('_results.json', '_generated.json'))

    # Step 2: Evaluate Model A
    print("\nStep 2/5: Evaluating Model A...")
    evaluate_emails(
        PRIMARY_MODEL,
        str(MODEL_A_RESULTS_FILE).replace('_results.json', '_generated.json'),
        MODEL_A_RESULTS_FILE
    )

    # Step 3: Generate with Model B
    print("\nStep 3/5: Generating emails with Model B...")
    generate_emails(SECONDARY_MODEL, str(MODEL_B_RESULTS_FILE).replace('_results.json', '_generated.json'))

    # Step 4: Evaluate Model B
    print("\nStep 4/5: Evaluating Model B...")
    evaluate_emails(
        SECONDARY_MODEL,
        str(MODEL_B_RESULTS_FILE).replace('_results.json', '_generated.json'),
        MODEL_B_RESULTS_FILE
    )

    # Step 5: Compare
    print("\nStep 5/6: Comparing models...")
    compare_models()

    # Step 6: Generate Report
    print("\nStep 6/6: Generating final assessment report...")
    generate_report("markdown")
    generate_report("html")

    print("\n✅ Complete pipeline finished!")
    print("\n📄 Final Report:")
    print("   - Markdown: data/results/FINAL_ASSESSMENT_REPORT.md")
    print("   - HTML: data/results/FINAL_ASSESSMENT_REPORT.html")
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Email Generation Assistant - AI Engineer Assessment"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate emails")
    gen_parser.add_argument(
        "--model",
        choices=["a", "b", "primary", "secondary"],
        default="a",
        help="Model to use (a=primary, b=secondary)"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate generated emails")
    eval_parser.add_argument(
        "--model",
        choices=["a", "b"],
        required=True,
        help="Which model's output to evaluate"
    )

    # Compare command
    subparsers.add_parser("compare", help="Compare two models")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate final assessment report")
    report_parser.add_argument(
        "--format",
        choices=["markdown", "html", "both"],
        default="both",
        help="Report format (default: both)"
    )

    # Run all command
    subparsers.add_parser("run-all", help="Run complete pipeline")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "generate":
            model = PRIMARY_MODEL if args.model in ["a", "primary"] else SECONDARY_MODEL
            output = MODEL_A_RESULTS_FILE if args.model in ["a", "primary"] else MODEL_B_RESULTS_FILE
            output = str(output).replace('_results.json', '_generated.json')
            generate_emails(model, output)

        elif args.command == "evaluate":
            if args.model == "a":
                model = PRIMARY_MODEL
                gen_file = str(MODEL_A_RESULTS_FILE).replace('_results.json', '_generated.json')
                out_file = MODEL_A_RESULTS_FILE
            else:
                model = SECONDARY_MODEL
                gen_file = str(MODEL_B_RESULTS_FILE).replace('_results.json', '_generated.json')
                out_file = MODEL_B_RESULTS_FILE

            evaluate_emails(model, gen_file, out_file)

        elif args.command == "compare":
            compare_models()

        elif args.command == "report":
            if args.format == "both":
                generate_report("markdown")
                generate_report("html")
            else:
                generate_report(args.format)

        elif args.command == "run-all":
            run_all_pipeline()

    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
