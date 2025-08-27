import os
import sys

from inspect_ai import task, Task, eval

from pipelines.selfsycophancy.multistep.datasets import ethics_mcq
from pipelines.selfsycophancy.singlestep.scorers import statistically_validated_sycophancy_scorer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.generalization_enums import MotivationFramework, KnowledgeFramework

# Harmfulness imports
from pipelines.selfsycophancy.singlestep.harmfulness.dataset import ethics_preference_dataset
from pipelines.selfsycophancy.singlestep.harmfulness.solvers import harmfulness_sycophancy_solver
from pipelines.selfsycophancy.singlestep.harmfulness.scorers import harmfulness_distribution_sycophancy_scorer, \
    harmfulness_distribution_skepticism_scorer, risk_uncertainty_shift_scorer, significant_sycophancy_scorer, \
    risk_shift_valence_scorer, risk_mass_migration_valence_scorer

# Correctness imports
from pipelines.selfsycophancy.singlestep.correctness.dataset import science_math_dataset, arc_challenge_dataset
from pipelines.selfsycophancy.singlestep.correctness.scorers import (
    confidence_shift_scorer, distribution_based_sycophancy_scorer, significant_sycophancy_scorer,
    distribution_divergence_valence_scorer, correctness_shift_valence_scorer, confidence_evolution_valence_scorer
)
from pipelines.selfsycophancy.singlestep.correctness.solvers import science_math_sycophancy_solver
from pipelines.selfsycophancy.hooks import SelfSycophancyHook

harmfulness_scorers =  [ statistically_validated_sycophancy_scorer(),
                        harmfulness_distribution_sycophancy_scorer(),
                        harmfulness_distribution_skepticism_scorer(),
                        risk_uncertainty_shift_scorer(),
                        significant_sycophancy_scorer(),

                        risk_shift_valence_scorer(),
                        risk_mass_migration_valence_scorer()]


correctness_scorers = [ statistically_validated_sycophancy_scorer(),
                        distribution_based_sycophancy_scorer(),
                        confidence_shift_scorer(),
                        significant_sycophancy_scorer(),
                        distribution_divergence_valence_scorer(),
                        correctness_shift_valence_scorer(),
                        confidence_evolution_valence_scorer()]
                       #statistically_validated_sycophancy_scorer()]

@task
def self_sycophancy_harmfulness_common_sense() -> Task:
    """Evaluate both self-sycophancy and self-skepticism on common sense ethics"""
    return Task(
        name="self-sycophancy_harmfulness_common_sense",
        #dataset=ethics_preference_dataset(name="commonsense"),
        dataset=ethics_mcq(
            scenario="commonsense",
            n_options=8,
            n_harmful_target=4,  # Balance harmful/harmless
            samples_per_question=300
        ),
        solver=harmfulness_sycophancy_solver(),
        scorer=harmfulness_scorers,
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.INTROSPECTIVE_KNOWLEDGE
        }

    )


@task
def self_sycophancy_harmfulness_deontology() -> Task:
    """Evaluate both self-sycophancy and self-skepticism on deontological ethics"""
    return Task(
        name="self-sycophancy_harmfulness_deontology",
        dataset=ethics_preference_dataset(name="deontology"),
        solver=harmfulness_sycophancy_solver(),
        scorer=harmfulness_scorers,
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.INTROSPECTIVE_KNOWLEDGE
        }
    )


@task
def self_sycophancy_harmfulness_justice() -> Task:
    """Evaluate both self-sycophancy and self-skepticism on justice ethics"""
    return Task(
        name="self-sycophancy_harmfulness_justice",
        dataset=ethics_mcq(
            scenario="justice",
            n_options=8,
            n_harmful_target=4,  # Balance harmful/harmless
            samples_per_question=300
        ),
        solver=harmfulness_sycophancy_solver(),
        scorer=harmfulness_scorers,
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.INTROSPECTIVE_KNOWLEDGE
        }
    )


@task
def self_sycophancy_correctness_physics() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on physics questions"""
    return Task(
        name="self-sycophancy_correctness_physics",
        dataset=science_math_dataset("physics"),
        solver=science_math_sycophancy_solver(),
        scorer=correctness_scorers,
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.INTROSPECTIVE_KNOWLEDGE
        }
    )


@task
def self_sycophancy_correctness_mathematics() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on mathematics questions"""
    return Task(
        name="self-sycophancy_correctness_mathematics",
        dataset=science_math_dataset("mathematics"),
        solver=science_math_sycophancy_solver(),
        scorer=correctness_scorers,
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.INTROSPECTIVE_KNOWLEDGE
        }
    )


@task
def self_sycophancy_correctness_chemistry() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on chemistry questions"""
    return Task(
        name="self-sycophancy_correctness_chemistry",
        dataset=science_math_dataset("chemistry"),
        solver=science_math_sycophancy_solver(),
        scorer=correctness_scorers,
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.INTROSPECTIVE_KNOWLEDGE
        }
    )


@task
def self_sycophancy_correctness_biology() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on biology questions"""
    return Task(
        name="self-sycophancy_correctness_biology",
        dataset=science_math_dataset("biology"),
        solver=science_math_sycophancy_solver(),
        scorer=correctness_scorers,
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.INTROSPECTIVE_KNOWLEDGE
        }
    )


@task
def self_sycophancy_correctness_arc() -> Task:
    """Evaluate self-sycophancy, self-skepticism, and accuracy on ARC Challenge questions"""
    return Task(
        name="self-sycophancy_correctness_arc",
        dataset=arc_challenge_dataset(),
        solver=science_math_sycophancy_solver(),
        scorer=correctness_scorers,
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.INTROSPECTIVE_KNOWLEDGE
        }
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
        # self_sycophancy_harmfulness_deontology(),
        self_sycophancy_harmfulness_common_sense(),
        self_sycophancy_harmfulness_justice(),
        self_sycophancy_correctness_physics(),
        self_sycophancy_correctness_mathematics(),
        self_sycophancy_correctness_chemistry(),
        # self_sycophancy_correctness_biology(),
        # self_sycophancy_correctness_arc()
    ]

    print("Running all self-sycophancy tasks...")
    print(f"Number of tasks: {len(all_tasks)}")
    print(f"Limit: 100 samples per task")

    # Run evaluation with all tasks
    # model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    # model = "openrouter/meta-llama/llama-3.3-70b-instruct"
    #model = "openrouter/google/gemma-3-12b-it"
    #model = "openrouter/openai/gpt-4o"

    #model = ["openrouter/openai/gpt-4o", "openrouter/openai/gpt-oss-120b", "openrouter/openai/gpt-oss-20b"] #"openrouter/google/gemma-3-12b-it"
    #model = []
    models = ["openrouter/openai/gpt-oss-120b",
              "openrouter/openai/gpt-oss-20b",
              "openrouter/openai/gpt-5-nano",
              "openrouter/openai/gpt-5-mini",
              "openrouter/openai/gpt-5",
              "openrouter/anthropic/claude-sonnet-4",
              "openrouter/anthropic/claude-opus-4.1",
              "openrouter/anthropic/claude-3.5-haiku",
              "openrouter/google/gemini-2.5-pro",
              "openrouter/google/gemma-3-27b-it"]

    results = eval(
        tasks=all_tasks,
        model=models,  # Using OpenRouter Claude model
        limit=300,  # Limit samples for faster debugging
        log_dir="../logs/self-sycophancy-single/all-1",  # Optional: specify log directory
        retry_on_error=5,
        fail_on_error=0.2
        # debug_errors=True,  # Optional: enable debug mode
    )