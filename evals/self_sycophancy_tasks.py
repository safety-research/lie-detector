import os
import sys

from inspect_ai import task, Task, eval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Harmfulness imports
from ssyc.harmfulness.dataset import ethics_preference_dataset
from ssyc.harmfulness.scorers import (
    self_sycophancy_scorer as harm_sycophancy_scorer,
    self_skepticism_scorer as harm_skepticism_scorer
)
from ssyc.harmfulness.solvers import combined_sycophancy_solver

# Correctness imports
from ssyc.correctness.dataset import science_math_dataset, arc_challenge_dataset
from ssyc.correctness.scorers import (
    self_sycophancy_scorer as corr_sycophancy_scorer,
    self_skepticism_scorer as corr_skepticism_scorer,
    answer_accuracy_scorer
)
from ssyc.correctness.solvers import science_math_sycophancy_solver


@task
def harmfulness_common_sense() -> Task:
    """Evaluate both self-sycophancy and self-skepticism on common sense ethics"""
    return Task(
        dataset=ethics_preference_dataset(name="commonsense"),
        solver=combined_sycophancy_solver(),
        scorer=[harm_sycophancy_scorer(), harm_skepticism_scorer()]
    )


@task
def harmfulness_deontology() -> Task:
    """Evaluate both self-sycophancy and self-skepticism on deontological ethics"""
    return Task(
        dataset=ethics_preference_dataset(name="deontology"),
        solver=combined_sycophancy_solver(),
        scorer=[harm_sycophancy_scorer(), harm_skepticism_scorer()]
    )


@task
def harmfulness_justice() -> Task:
    """Evaluate both self-sycophancy and self-skepticism on justice ethics"""
    return Task(
        dataset=ethics_preference_dataset(name="justice"),
        solver=combined_sycophancy_solver(),
        scorer=[harm_sycophancy_scorer(), harm_skepticism_scorer()]
    )


@task
def correctness_physics() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on physics questions"""
    return Task(
        dataset=science_math_dataset("physics"),
        solver=science_math_sycophancy_solver(),
        scorer=[corr_sycophancy_scorer(), corr_skepticism_scorer(), answer_accuracy_scorer()]
    )


@task
def correctness_mathematics() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on mathematics questions"""
    return Task(
        dataset=science_math_dataset("mathematics"),
        solver=science_math_sycophancy_solver(),
        scorer=[corr_sycophancy_scorer(), corr_skepticism_scorer(), answer_accuracy_scorer()]
    )


@task
def correctness_chemistry() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on chemistry questions"""
    return Task(
        dataset=science_math_dataset("chemistry"),
        solver=science_math_sycophancy_solver(),
        scorer=[corr_sycophancy_scorer(), corr_skepticism_scorer(), answer_accuracy_scorer()]
    )


@task
def correctness_biology() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on biology questions"""
    return Task(
        dataset=science_math_dataset("biology"),
        solver=science_math_sycophancy_solver(),
        scorer=[corr_sycophancy_scorer(), corr_skepticism_scorer(), answer_accuracy_scorer()]
    )


@task
def correctness_arc() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on ARC Challenge questions"""
    return Task(
        dataset=arc_challenge_dataset(),
        solver=science_math_sycophancy_solver(),
        scorer=[corr_sycophancy_scorer(), corr_skepticism_scorer(), answer_accuracy_scorer()]
    )


if __name__ == "__main__":
    # Create the task
    task = harmfulness_deontology()

    # Run evaluation with a specific model
    results = eval(
        tasks=[task],
        model="openai/gpt-4o-mini",  # or any model you want to test
        limit=5,  # Limit samples for faster debugging
        log_dir="./logs",  # Optional: specify log directory
        # debug_errors=True,  # Optional: enable debug mode
    )