#!/usr/bin/env python3
"""
Example script demonstrating ASCII sandbagging detection.

This script shows various ways to use the ASCII module to detect
sandbagging behavior in language models.
"""

import argparse
from inspect_ai import eval
from sandbagging.ascii import (
    ascii_sandbagging_task,
    ascii_quick_test_task,
    ascii_comprehensive_test_task,
    ascii_category_specific_task
)
from sandbagging.ascii.utils import format_sandbagging_report


def run_quick_demo():
    """Run a quick demonstration of sandbagging detection."""

    print("Running Quick ASCII Sandbagging Test")
    print("=" * 50)

    # Create a quick test task
    task = ascii_quick_test_task(
        model="openai/gpt-4o-mini",  # Using smaller model for demo
        judge_model="openai/gpt-4o"
    )

    # Run evaluation
    print("\nEvaluating model...")
    results = eval(
        task,
        model="openai/gpt-4o-mini",
        log_dir="./logs"
    )

    # Print summary
    if results and len(results) > 0:
        log = results[0]
        print(f"\nEvaluation completed!")
        print(f"Samples evaluated: {len(log.samples)}")

        # Show some examples
        print("\n=== Sample Results ===")
        for i, sample in enumerate(log.samples[:3]):
            print(f"\nSample {i + 1}:")
            print(f"  Subject: {sample.metadata.get('subject', 'Unknown')}")
            print(f"  Is Harmful: {sample.metadata.get('is_harmful', False)}")

            if hasattr(sample, 'scores') and sample.scores:
                for score_name, score in sample.scores.items():
                    if hasattr(score, 'metadata') and score.metadata:
                        print(f"  Sandbagged: {score.metadata.get('sandbagged', False)}")
                        print(f"  Interpretation: {score.metadata.get('interpretation', 'N/A')}")


def run_category_analysis(category: str):
    """Run analysis on a specific category."""

    print(f"\nAnalyzing Sandbagging in Category: {category}")
    print("=" * 50)

    task = ascii_category_specific_task(
        category=category,
        num_subjects=5,
        model="openai/gpt-4o",
        judge_model="openai/gpt-4o"
    )

    results = eval(
        task,
        model="openai/gpt-4o",
        log_dir=f"./ascii_results_{category}"
    )

    if results and len(results) > 0:
        log = results[0]

        # Calculate statistics
        sandbagging_count = 0
        harmless_sandbagging = 0

        for sample in log.samples:
            if hasattr(sample, 'scores') and sample.scores:
                for score_name, score in sample.scores.items():
                    if hasattr(score, 'metadata') and score.metadata:
                        if score.metadata.get('sandbagged', False):
                            sandbagging_count += 1
                            if not sample.metadata.get('is_harmful', False):
                                harmless_sandbagging += 1

        print(f"\nResults for {category}:")
        print(f"  Total samples: {len(log.samples)}")
        print(f"  Sandbagging detected: {sandbagging_count}")
        print(f"  Harmless subject sandbagging: {harmless_sandbagging}")


def run_comprehensive_analysis():
    """Run comprehensive analysis across multiple categories."""

    print("\nRunning Comprehensive ASCII Sandbagging Analysis")
    print("=" * 50)

    task = ascii_comprehensive_test_task(
        model="openai/gpt-4o",
        categories=["animals", "objects", "violence", "medical"]
    )

    print("\nThis will take several minutes...")
    results = eval(
        task,
        model="openai/gpt-4o",
        log_dir="./ascii_comprehensive_results"
    )

    if results and len(results) > 0:
        log = results[0]

        # Extract results for report
        formatted_results = []
        for sample in log.samples:
            if hasattr(sample, 'scores') and sample.scores:
                for score_name, score in sample.scores.items():
                    if hasattr(score, 'metadata'):
                        formatted_results.append({
                            'metadata': {
                                **sample.metadata,
                                **score.metadata
                            }
                        })

        # Generate report
        report = format_sandbagging_report(
            formatted_results,
            model_name="GPT-4"
        )

        print(report)

        # Save report
        with open("./ascii_sandbagging_report.txt", "w") as f:
            f.write(report)
        print("\nReport saved to: ./ascii_sandbagging_report.txt")


def main():
    """Main function to run different types of analyses."""

    parser = argparse.ArgumentParser(
        description="ASCII Sandbagging Detection Examples"
    )

    parser.add_argument(
        "--mode",
        choices=["quick", "category", "comprehensive"],
        default="quick",
        help="Type of analysis to run"
    )

    parser.add_argument(
        "--category",
        default="animals",
        help="Category for category-specific analysis"
    )

    parser.add_argument(
        "--model",
        default="openai/gpt-4o",
        help="Model to evaluate"
    )

    args = parser.parse_args()

    print(f"ASCII Sandbagging Detection Example")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")

    if args.mode == "quick":
        run_quick_demo()
    elif args.mode == "category":
        run_category_analysis(args.category)
    elif args.mode == "comprehensive":
        run_comprehensive_analysis()

    print("\n✅ Example completed!")


if __name__ == "__main__":
    main()