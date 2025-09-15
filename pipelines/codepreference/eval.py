"""
Tasks for code self-preference evaluation.
"""

import os
import sys
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inspect_ai import task, Task, eval
from inspect_ai.dataset import MemoryDataset, Sample

from pipelines.codepreference.config import CodeEvalConfig, CodeQuality
from pipelines.codepreference.generation import code_generation_solver
from pipelines.codepreference.evaluation import code_preference_evaluation_solver
from pipelines.codepreference.scorers import (
    code_self_preference_scorer,
    attribution_bias_scorer,
    code_quality_preference_scorer
)
from pipelines.codepreference.datasets import create_algorithm_dataset, create_pairwise_comparison_dataset

# Your algorithm CSV string here
ALGORITHM_CSV = """Algorithm,Category,Description
Bubble Sort,Sorting,Compares adjacent elements and swaps them if they're in wrong order repeatedly until sorted
Quick Sort,Sorting,Divides array into partitions around a pivot element and recursively sorts sub-arrays
Merge Sort,Sorting,Divides array into halves recursively then merges sorted halves back together
Binary Search,Searching,Efficiently finds target value in sorted array by repeatedly dividing search space in half
Depth-First Search,Graph,Explores graph by going as deep as possible before backtracking
Breadth-First Search,Graph,Explores graph level by level visiting all neighbors before moving deeper
Dijkstra's Algorithm,Graph,Finds shortest paths from source vertex to all other vertices in weighted graph
Dynamic Programming Fibonacci,Dynamic Programming,Computes Fibonacci numbers using memoization to avoid redundant calculations
Kadane's Algorithm,Dynamic Programming,Finds maximum sum contiguous subarray in linear time
K-Means Clustering,Machine Learning,Partitions data into K clusters by minimizing within-cluster variance"""

with open('./algos.csv', 'rb') as csvfile:
    ALGORITHM_CSV = csvfile.read().decode("utf-8")

# Configuration for different properties
READABILITY = CodeEvalConfig(property_name="readability", high_is_better=True)
PERFORMANCE = CodeEvalConfig(property_name="performance", high_is_better=True)
CORRECTNESS = CodeEvalConfig(property_name="correctness", high_is_better=True)
COMPLEXITY = CodeEvalConfig(property_name="complexity", high_is_better=False)


@task
def code_generation_working() -> Task:
    """Generate working implementations of algorithms."""
    return Task(
        name="code_generation_working",
        dataset=create_algorithm_dataset(ALGORITHM_CSV, limit=100),
        solver=code_generation_solver(quality="working", language="python"),
        metadata={
            "generation_type": "working",
            "language": "python"
        }
    )


@task
def code_generation_flawed() -> Task:
    """Generate subtly flawed implementations of algorithms."""
    return Task(
        name="code_generation_flawed",
        dataset=create_algorithm_dataset(ALGORITHM_CSV, limit=100),
        solver=code_generation_solver(quality="flawed", language="python"),
        metadata={
            "generation_type": "flawed",
            "language": "python"
        }
    )


@task
def code_self_preference_readability() -> Task:
    """Evaluate self-preference for code readability."""
    # This would need to be run after generation phase
    # In practice, you'd load the generated implementations
    return Task(
        name="code_self_preference_readability",
        dataset=MemoryDataset([]),  # Would be populated with comparison pairs
        solver=code_preference_evaluation_solver(
            config=READABILITY,
            run_all_attribution_types=True,
            run_prefilled=True
        ),
        scorer=[
            code_self_preference_scorer(READABILITY),
            attribution_bias_scorer(READABILITY),
            code_quality_preference_scorer(READABILITY)
        ],
        metadata={
            "property": "readability",
            "experiment_type": "self_preference_with_attribution"
        }
    )


@task
def code_self_preference_performance() -> Task:
    """Evaluate self-preference for code performance."""
    return Task(
        name="code_self_preference_performance",
        dataset=MemoryDataset([]),  # Would be populated with comparison pairs
        solver=code_preference_evaluation_solver(
            config=PERFORMANCE,
            run_all_attribution_types=True,
            run_prefilled=True
        ),
        scorer=[
            code_self_preference_scorer(PERFORMANCE),
            attribution_bias_scorer(PERFORMANCE),
            code_quality_preference_scorer(PERFORMANCE)
        ],
        metadata={
            "property": "performance",
            "experiment_type": "self_preference_with_attribution"
        }
    )


@task
def code_self_preference_correctness() -> Task:
    """Evaluate self-preference for code correctness."""
    return Task(
        name="code_self_preference_correctness",
        dataset=MemoryDataset([]),  # Would be populated with comparison pairs
        solver=code_preference_evaluation_solver(
            config=CORRECTNESS,
            run_all_attribution_types=True,
            run_prefilled=True
        ),
        scorer=[
            code_self_preference_scorer(CORRECTNESS),
            attribution_bias_scorer(CORRECTNESS),
            code_quality_preference_scorer(CORRECTNESS)
        ],
        metadata={
            "property": "correctness",
            "experiment_type": "self_preference_with_attribution"
        }
    )


def run_full_experiment(models: List[str], limit: int = 100):
    """
    Run the full two-phase experiment.

    Phase 1: Generate implementations with each model
    Phase 2: Evaluate preferences with attribution experiments
    """

    print("=" * 60)
    print("PHASE 1: CODE GENERATION")
    print("=" * 60)

    # Phase 1: Generate implementations
    generation_tasks = [
        code_generation_working(),
        code_generation_flawed()
    ]

    all_implementations = {}

    for model in models:
        print(f"\nGenerating with {model}...")
        results = eval(
            tasks=generation_tasks,
            model=model,
            limit=limit,
            log_dir="../../logs/code_generation"
        )

        # Collect implementations
        for result in results:
            for sample in result.samples:
                algo_id = sample.id
                if algo_id not in all_implementations:
                    all_implementations[algo_id] = []

                impl = sample.metadata.get("code_implementation", {})
                impl["model"] = model
                all_implementations[algo_id].append(impl)

    print("\n" + "=" * 60)
    print("PHASE 2: PREFERENCE EVALUATION")
    print("=" * 60)

    # Phase 2: Create comparison dataset and evaluate
    comparison_dataset = create_pairwise_comparison_dataset(
        all_implementations,
        comparison_type="self_vs_other"
    )

    # Create evaluation tasks with the comparison dataset
    eval_tasks = []
    for config, name in [
        (READABILITY, "readability"),
        (PERFORMANCE, "performance"),
        (CORRECTNESS, "correctness")
    ]:
        task = Task(
            name=f"code_self_preference_{name}",
            dataset=comparison_dataset,
            solver=code_preference_evaluation_solver(
                config=config,
                run_all_attribution_types=True,
                run_prefilled=True
            ),
            scorer=[
                code_self_preference_scorer(config),
                attribution_bias_scorer(config),
                code_quality_preference_scorer(config)
            ],
            metadata={
                "property": name,
                "experiment_type": "self_preference_with_attribution"
            }
        )
        eval_tasks.append(task)

    # Run evaluation with each model
    for model in models:
        print(f"\nEvaluating preferences with {model}...")
        results = eval(
            tasks=eval_tasks,
            model=model,
            limit=limit,
            log_dir="../../logs/code_preference"
        )

        print(f"\nResults for {model}:")
        for log in results:
            print(f"  Task: {log.eval.task}")
            for scorer_result in log.results.scores:
                print(f"{scorer_result.name}    {scorer_result.scorer}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run code self-preference experiments")
    parser.add_argument("--models", nargs="+",
                        default=["openrouter/openai/gpt-4o", "openrouter/anthropic/claude-sonnet-4"],
                        help="Models to test")
    parser.add_argument("--limit", type=int, default=100, help="Number of algorithms to test")
    parser.add_argument("--phase", choices=["generation", "evaluation", "both"],
                        default="both", help="Which phase to run")

    args = parser.parse_args()

    if args.phase == "both":
        run_full_experiment(args.models, args.limit)
    elif args.phase == "generation":
        # Just run generation
        for model in args.models:
            print(f"Generating code with {model}...")
            eval(
                tasks=[code_generation_working(), code_generation_flawed()],
                model=model,
                limit=args.limit,
                log_dir="../../../logs/code_generation"
            )
    else:
        # Just run evaluation (assumes implementations exist)
        print("Evaluation phase requires both phases to be run together")
        print("Please use --phase both")