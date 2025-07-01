import inspect
import os
import sys
from typing import List

from dotenv import load_dotenv
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample, MemoryDataset

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.n_truths_and_a_lie.scenario import generate_themed_scenarios
from games.n_truths_and_a_lie import GameConfig
from games.n_truths_and_a_lie.scorer import lie_detection_scorer, interrogation_quality_scorer, combined_n_truths_scorer
from games.n_truths_and_a_lie.solvers import n_truths_solver

MAX_SCENARIOS = 1


def _configs_to_dataset(configs: List[GameConfig]) -> MemoryDataset:
    """
    Create dataset from configurations
    """
    samples = []
    for config in configs:
        samples.append(Sample(
            input=str([truth.content for truth in config.fact_set.get_truths()]),
            target=str([lie.content for lie in config.fact_set.get_lies()]),
            metadata={
                "game_config": config.model_dump(),
                "game_id": config.game_id,
                "theme": config.fact_set.theme,
                "num_truths": config.fact_set.num_truths,
                "num_lies": config.fact_set.num_lies
            }
        ))

    # Create dataset
    return MemoryDataset(samples)

@task
def n_truths_and_a_lie_professional_background() -> Task:
    """Create an N-truths-and-a-lie task from game configurations"""

    configs = generate_themed_scenarios("professional_background", num_scenarios=MAX_SCENARIOS)
    dataset = _configs_to_dataset(configs)

    return Task(
        name=inspect.currentframe().f_code.co_name,
        dataset=dataset,
        solver=n_truths_solver(),
        scorer=[
            lie_detection_scorer(),
            interrogation_quality_scorer(),
            combined_n_truths_scorer()
        ],
        sandbox="local"
    )

@task
def n_truths_and_a_lie_hidden_talents() -> Task:
    """Create an N-truths-and-a-lie task from game configurations"""

    configs = generate_themed_scenarios("hidden_talents", num_scenarios=MAX_SCENARIOS)
    dataset = _configs_to_dataset(configs)

    return Task(
        name=inspect.currentframe().f_code.co_name,
        dataset=dataset,
        solver=n_truths_solver(),
        scorer=[
            lie_detection_scorer(),
            interrogation_quality_scorer(),
            combined_n_truths_scorer()
        ],
        sandbox="local"
    )

@task
def n_truths_and_a_lie_family_history() -> Task:
    """Create an N-truths-and-a-lie task from game configurations"""

    configs = generate_themed_scenarios("family_history", num_scenarios=MAX_SCENARIOS)
    dataset = _configs_to_dataset(configs)

    return Task(
        name=inspect.currentframe().f_code.co_name,
        dataset=dataset,
        solver=n_truths_solver(),
        scorer=[
            lie_detection_scorer(),
            interrogation_quality_scorer(),
            combined_n_truths_scorer()
        ],
        sandbox="local"
    )

@task
def n_truths_and_a_lie_personal_quirks() -> Task:
    """Create an N-truths-and-a-lie task from game configurations"""

    configs = generate_themed_scenarios("personal_quirks", num_scenarios=MAX_SCENARIOS)
    dataset = _configs_to_dataset(configs)

    return Task(
        name=inspect.currentframe().f_code.co_name,
        dataset=dataset,
        solver=n_truths_solver(),
        scorer=[
            lie_detection_scorer(),
            interrogation_quality_scorer(),
            combined_n_truths_scorer()
        ],
        sandbox="local"
    )

@task
def n_truths_and_a_lie_travel_experiences() -> Task:
    """Create an N-truths-and-a-lie task from game configurations"""

    configs = generate_themed_scenarios("travel_experiences", num_scenarios=MAX_SCENARIOS)
    dataset = _configs_to_dataset(configs)

    return Task(
        name=inspect.currentframe().f_code.co_name,
        dataset=dataset,
        solver=n_truths_solver(),
        scorer=[
            lie_detection_scorer(),
            interrogation_quality_scorer(),
            combined_n_truths_scorer()
        ],
        sandbox="local"
    )


if __name__ == "__main__":
    # Example usage - run corporate scenario evaluation
    task = n_truths_and_a_lie_professional_background()

    # Run evaluation with a specific model
    results = eval(
        tasks=[task],
        model="openai/gpt-4o-mini",  # or any model you want to test
        limit=2,  # Limit samples for faster debugging
        log_dir="./logs",  # Specify log directory
        # debug_errors=True,  # Optional: enable debug mode
    )