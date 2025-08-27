"""
Dataset creation for code preference experiments.
"""

from typing import List, Dict, Optional
import csv
from io import StringIO

from inspect_ai.dataset import Dataset, Sample, MemoryDataset


def create_algorithm_dataset(algorithm_csv: str, limit: Optional[int] = None) -> Dataset:
    """
    Create a dataset from the algorithm CSV.

    Args:
        algorithm_csv: CSV string containing algorithms
        limit: Maximum number of algorithms to include
    """

    # Parse CSV
    reader = csv.DictReader(StringIO(algorithm_csv))
    algorithms = list(reader)

    if limit:
        algorithms = algorithms[:limit]

    samples = []
    for i, algo in enumerate(algorithms):
        sample = Sample(
            id=f"algo_{i}_{algo['Algorithm'].lower().replace(' ', '_')}",
            input=f"Implement {algo['Algorithm']}",
            metadata={
                "algorithm_name": algo['Algorithm'],
                "algorithm_description": algo['Description'],
                "category": algo['Category']
            }
        )
        samples.append(sample)

    return MemoryDataset(samples)


def create_pairwise_comparison_dataset(
        implementations: Dict[str, List[Dict]],
        comparison_type: str = "self_vs_other"
) -> Dataset:
    """
    Create dataset for pairwise comparisons of implementations.

    Args:
        implementations: Dict mapping algorithm_id to list of implementations
        comparison_type: Type of comparison (self_vs_other, working_vs_flawed, etc.)
    """

    samples = []

    for algo_id, impls in implementations.items():
        if len(impls) < 2:
            continue

        # Create different comparison types
        if comparison_type == "self_vs_other":
            # Group by model
            by_model = {}
            for impl in impls:
                model = impl.get("model", "unknown")
                if model not in by_model:
                    by_model[model] = []
                by_model[model].append(impl)

            # Create comparisons
            models = list(by_model.keys())
            for i, model1 in enumerate(models):
                for model2 in models[i + 1:]:
                    # Take first implementation from each model
                    impl1 = by_model[model1][0]
                    impl2 = by_model[model2][0]

                    sample = Sample(
                        id=f"{algo_id}_compare_{model1}_vs_{model2}",
                        input=f"Compare implementations of {impl1.get('algorithm_name')}",
                        metadata={
                            "algorithm_name": impl1.get("algorithm_name"),
                            "implementations": [impl1, impl2],
                            "comparison_type": comparison_type,
                            "models": [model1, model2]
                        }
                    )
                    samples.append(sample)

        elif comparison_type == "working_vs_flawed":
            # Group by quality
            working = [impl for impl in impls if impl.get("quality") == "working"]
            flawed = [impl for impl in impls if impl.get("quality") == "flawed"]

            for w_impl in working[:1]:  # Take first working
                for f_impl in flawed[:1]:  # Take first flawed
                    sample = Sample(
                        id=f"{algo_id}_compare_working_vs_flawed",
                        input=f"Compare implementations of {w_impl.get('algorithm_name')}",
                        metadata={
                            "algorithm_name": w_impl.get("algorithm_name"),
                            "implementations": [w_impl, f_impl],
                            "comparison_type": comparison_type,
                            "qualities": ["working", "flawed"]
                        }
                    )
                    samples.append(sample)

    return MemoryDataset(samples)