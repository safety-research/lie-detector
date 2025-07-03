"""
ASCII Task for Inspect AI - Sandbagging Detection

This module provides task definitions for detecting sandbagging behavior
in ASCII art generation capabilities.
"""

from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from typing import Optional, List, Dict, Any

from sandbagging.ascii.dataset import ascii_dataset, generate_ascii_test_subjects
from sandbagging.ascii.solvers import ascii_harmfulness_with_subjects_solver
from sandbagging.ascii.scorers import ascii_sandbagging_scorer


@task
def ascii_sandbagging_task(
        num_harmful_subjects: int = 10,
        num_harmless_per_harmful: int = 3,
        model: str = "openai/gpt-4o",
        judge_model: str = "openai/gpt-4o",
        subject_generator_model: str = "openai/gpt-4o",
        capability_test_runs: int = 3,
        sandbagging_test_runs: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 1000,
        shuffle: bool = False,
        seed: Optional[int] = None,
        categories: Optional[List[str]] = None
) -> Task:
    """
    Main ASCII sandbagging detection task.

    This task tests whether models claim they "can't" draw certain subjects
    when they actually have the capability to do so.

    Args:
        num_harmful_subjects: Number of harmful subjects to test
        num_harmless_per_harmful: Number of harmless controls per harmful subject
        model: Model to evaluate
        judge_model: Model to judge ASCII art quality
        subject_generator_model: Model to generate test subjects
        capability_test_runs: Number of runs to test actual capability
        sandbagging_test_runs: Number of runs to test for sandbagging
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        shuffle: Whether to shuffle the dataset
        seed: Random seed
        categories: Optional list of categories to test

    Returns:
        Task configured for ASCII sandbagging detection
    """

    # Generate test subjects dynamically
    # subjects = generate_ascii_test_subjects(
    #     num_harmful=num_harmful_subjects,
    #     num_harmless_per_harmful=num_harmless_per_harmful,
    #     categories=categories,
    #     generator_model=subject_generator_model
    # )

    return Task(
        dataset=ascii_dataset(
            #subjects=subjects,
            shuffle=shuffle,
            seed=seed
        ),
        solver=ascii_harmfulness_with_subjects_solver(
        ),
        scorer=ascii_sandbagging_scorer(
            judge_model=judge_model
        ),
        config=GenerateConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    )


@task
def ascii_quick_test_task(
        model: str = "openai/gpt-4o",
        judge_model: str = "openai/gpt-4o",
        temperature: float = 1.0,
        max_tokens: int = 1000
) -> Task:
    """
    Quick ASCII sandbagging test with minimal runs for testing.

    Uses only 3 harmful subjects with 2 harmless controls each,
    and single runs for faster evaluation.
    """

    return ascii_sandbagging_task(
        num_harmful_subjects=3,
        num_harmless_per_harmful=2,
        model=model,
        judge_model=judge_model,
        capability_test_runs=1,
        sandbagging_test_runs=1,
        temperature=temperature,
        max_tokens=max_tokens
    )


@task
def ascii_comprehensive_test_task(
        model: str = "openai/gpt-4o",
        judge_model: str = "openai/gpt-4o",
        subject_generator_model: str = "openai/gpt-4o",
        temperature: float = 1.0,
        max_tokens: int = 1000,
        categories: Optional[List[str]] = None
) -> Task:
    """
    Comprehensive ASCII sandbagging test across multiple categories.

    Tests 20 harmful subjects with 3 harmless controls each,
    multiple runs for robustness.
    """

    if categories is None:
        categories = [
            "animals",
            "objects",
            "people",
            "abstract_concepts",
            "violence",
            "medical",
            "political",
            "religious"
        ]

    return ascii_sandbagging_task(
        num_harmful_subjects=20,
        num_harmless_per_harmful=3,
        model=model,
        judge_model=judge_model,
        subject_generator_model=subject_generator_model,
        capability_test_runs=5,
        sandbagging_test_runs=3,
        temperature=temperature,
        max_tokens=max_tokens,
        categories=categories
    )


@task
def ascii_category_specific_task(
        category: str,
        num_subjects: int = 5,
        model: str = "openai/gpt-4o",
        judge_model: str = "openai/gpt-4o",
        temperature: float = 1.0,
        max_tokens: int = 1000
) -> Task:
    """
    Test sandbagging in a specific category.

    Args:
        category: Category to test (e.g., "animals", "violence", "medical")
        num_subjects: Number of harmful subjects in this category
        model: Model to evaluate
        judge_model: Model to judge ASCII art
        temperature: Generation temperature
        max_tokens: Maximum tokens

    Returns:
        Task for category-specific sandbagging detection
    """

    return ascii_sandbagging_task(
        num_harmful_subjects=num_subjects,
        num_harmless_per_harmful=3,
        model=model,
        judge_model=judge_model,
        capability_test_runs=3,
        sandbagging_test_runs=2,
        temperature=temperature,
        max_tokens=max_tokens,
        categories=[category]
    )


# Helper functions

def run_ascii_sandbagging_evaluation(
        output_dir: str = "./ascii_results",
        model: str = "openai/gpt-4o",
        evaluation_type: str = "standard",
        **kwargs
) -> None:
    """
    Helper function to run ASCII sandbagging evaluation.

    Args:
        output_dir: Directory to save results
        model: Model to evaluate
        evaluation_type: Type of evaluation ('quick', 'standard', 'comprehensive')
        **kwargs: Additional task parameters
    """

    from inspect_ai import eval

    # Select task based on evaluation type
    if evaluation_type == "quick":
        task_instance = ascii_quick_test_task(
            model=model,
            **kwargs
        )
    elif evaluation_type == "comprehensive":
        task_instance = ascii_comprehensive_test_task(
            model=model,
            **kwargs
        )
    else:  # standard
        task_instance = ascii_sandbagging_task(
            model=model,
            **kwargs
        )

    # Execute the evaluation
    results = eval(
        task_instance,
        model=model,
        log_dir=output_dir
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ASCII sandbagging detection")
    parser.add_argument("--model", default="openai/gpt-4o", help="Model to evaluate")
    parser.add_argument("--evaluation_type", default="standard",
                        choices=["quick", "standard", "comprehensive"],
                        help="Type of evaluation to run")
    parser.add_argument("--output_dir", default="./ascii_results", help="Output directory")
    parser.add_argument("--judge_model", default="openai/gpt-4o", help="Judge model")
    parser.add_argument("--num_harmful", type=int, default=10,
                        help="Number of harmful subjects to test")
    parser.add_argument("--num_harmless_per_harmful", type=int, default=3,
                        help="Number of harmless controls per harmful subject")
    parser.add_argument("--category", type=str, default=None,
                        help="Specific category to test")

    args = parser.parse_args()

    # Run evaluation
    if args.category:
        # Run category-specific evaluation
        run_ascii_sandbagging_evaluation(
            output_dir=args.output_dir,
            model=args.model,
            evaluation_type="category",
            category=args.category,
            judge_model=args.judge_model
        )
    else:
        # Run general evaluation
        run_ascii_sandbagging_evaluation(
            output_dir=args.output_dir,
            model=args.model,
            evaluation_type=args.evaluation_type,
            judge_model=args.judge_model,
            num_harmful_subjects=args.num_harmful,
            num_harmless_per_harmful=args.num_harmless_per_harmful
        )