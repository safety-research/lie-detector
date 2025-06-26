from pathlib import Path

from inspect_ai import task, Task, eval

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ssyc.ethics_preference_dataset import ethics_preference_dataset
from ssyc.scorer import self_sycophancy_scorer, self_skepticism_scorer
from ssyc.ssyc_solvers import combined_sycophancy_solver


@task
def ssyc_common_sense() -> Task:
    """Main task that evaluates sycophancy in models"""
    return Task(
        dataset=ethics_preference_dataset(name="commonsense"),
        solver=combined_sycophancy_solver(),
        scorer=self_skepticism_scorer()
    )

@task
def ssyc_deontology() -> Task:
    """Main task that evaluates sycophancy in models"""
    return Task(
        dataset=ethics_preference_dataset(name="deontology"),
        solver=combined_sycophancy_solver(),
        scorer=self_skepticism_scorer()
    )

@task
def ssyc_justice() -> Task:
    """Main task that evaluates sycophancy in models"""
    return Task(
        dataset=ethics_preference_dataset(name="justice"),
        solver=combined_sycophancy_solver(),
        scorer=self_skepticism_scorer()
    )


if __name__ == "__main__":
    # Create the task
    task = ssyc_common_sense()

    # Run evaluation with a specific model
    results = eval(
        tasks=[task],
        model="openai/gpt-4o-mini",  # or any model you want to test
        limit=5,  # Limit samples for faster debugging
        log_dir="./logs",  # Optional: specify log directory
        # debug_errors=True,  # Optional: enable debug mode
    )