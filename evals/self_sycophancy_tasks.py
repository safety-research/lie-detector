import os
import sys

from inspect_ai import task, Task, eval


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Harmfulness imports
from pipelines.selfsycophancy.harmfulness.dataset import ethics_preference_dataset
from pipelines.selfsycophancy.harmfulness.solvers import harmfulness_sycophancy_solver
from pipelines.selfsycophancy.harmfulness.scorers import harmfulness_distribution_sycophancy_scorer, \
    harmfulness_distribution_skepticism_scorer, risk_uncertainty_shift_scorer, significant_sycophancy_scorer

# Correctness imports
from pipelines.selfsycophancy.correctness.dataset import science_math_dataset, arc_challenge_dataset
from pipelines.selfsycophancy.correctness.scorers import (
    confidence_shift_scorer, distribution_based_sycophancy_scorer, significant_sycophancy_scorer
)
from pipelines.selfsycophancy.correctness.solvers import science_math_sycophancy_solver
from pipelines.selfsycophancy.hooks import SelfSycophancyHook

harmfulness_scorers =  [harmfulness_distribution_sycophancy_scorer(),
                        harmfulness_distribution_skepticism_scorer(),
                        risk_uncertainty_shift_scorer(),
                        significant_sycophancy_scorer()]

correctness_scorers = [distribution_based_sycophancy_scorer(),
                       confidence_shift_scorer(),
                       significant_sycophancy_scorer()]

@task
def self_sycophancy_harmfulness_common_sense() -> Task:
    """Evaluate both self-sycophancy and self-skepticism on common sense ethics"""
    return Task(
        dataset=ethics_preference_dataset(name="commonsense"),
        solver=harmfulness_sycophancy_solver(),
        scorer=harmfulness_scorers
    )


@task
def self_sycophancy_harmfulness_deontology() -> Task:
    """Evaluate both self-sycophancy and self-skepticism on deontological ethics"""
    return Task(
        dataset=ethics_preference_dataset(name="deontology"),
        solver=harmfulness_sycophancy_solver(),
        scorer=harmfulness_scorers
    )


@task
def self_sycophancy_harmfulness_justice() -> Task:
    """Evaluate both self-sycophancy and self-skepticism on justice ethics"""
    return Task(
        dataset=ethics_preference_dataset(name="justice"),
        solver=harmfulness_sycophancy_solver(),
        scorer=harmfulness_scorers
    )


@task
def self_sycophancy_correctness_physics() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on physics questions"""
    return Task(
        dataset=science_math_dataset("physics"),
        solver=science_math_sycophancy_solver(),
        scorer=correctness_scorers
    )


@task
def self_sycophancy_correctness_mathematics() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on mathematics questions"""
    return Task(
        dataset=science_math_dataset("mathematics"),
        solver=science_math_sycophancy_solver(),
        scorer=correctness_scorers
    )


@task
def self_sycophancy_correctness_chemistry() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on chemistry questions"""
    return Task(
        dataset=science_math_dataset("chemistry"),
        solver=science_math_sycophancy_solver(),
        scorer=correctness_scorers
    )


@task
def self_sycophancy_correctness_biology() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on biology questions"""
    return Task(
        dataset=science_math_dataset("biology"),
        solver=science_math_sycophancy_solver(),
        scorer=correctness_scorers
    )


@task
def self_sycophancy_correctness_arc() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on ARC Challenge questions"""
    return Task(
        dataset=arc_challenge_dataset(),
        solver=science_math_sycophancy_solver(),
        scorer=correctness_scorers
    )


if __name__ == "__main__":
    # Set up OpenRouter authentication
    # if "OPENROUTER_API_KEY" in os.environ:
    #     os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    #     os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    #     print("OpenRouter authentication configured")
    # else:
    #     print("Warning: OPENROUTER_API_KEY not found in environment variables")
    #
    # Create the task
    # task = correctness_physics()

    # Define all self-sycophancy tasks
    all_tasks = [
        self_sycophancy_harmfulness_common_sense(),
        self_sycophancy_harmfulness_deontology(),
        self_sycophancy_harmfulness_justice(),
        self_sycophancy_correctness_physics(),
        self_sycophancy_correctness_mathematics(),
        self_sycophancy_correctness_chemistry(),
        self_sycophancy_correctness_biology(),
        self_sycophancy_correctness_arc()
    ]

    print("Running all self-sycophancy tasks...")
    print(f"Number of tasks: {len(all_tasks)}")
    print(f"Limit: 100 samples per task")

    # Run evaluation with all tasks
    model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    # model = "openrouter/meta-llama/llama-3.3-70b-instruct"

    results = eval(
        tasks=all_tasks,
        model=model,  # Using OpenRouter Claude model
        limit=1,  # Limit samples for faster debugging
        log_dir="../logs",  # Optional: specify log directory
        # debug_errors=True,  # Optional: enable debug mode
    )