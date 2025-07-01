import os
import sys
from typing import List

from inspect_ai import Task, task, eval
from inspect_ai.dataset import MemoryDataset, Sample

from games.capture_the_secret.models import ScenarioConfig
from games.capture_the_secret.solvers import capture_the_secret_solver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.capture_the_secret.scorer import secrets_leaked_judge, deception_detection_judge, judge_calibration_scorer

#
# @task
# def deception_detection_task(
#         num_scenarios: int = 1,
#         #model: str = "openai/gpt-4"
# ) -> Task:
#     """Create a deception detection evaluation task"""
#
#     # Generate scenarios
#     scenarios = generate_scenarios(num_scenarios)
#
#     # Create samples
#     samples = []
#     for scenario in scenarios:
#         # Each sample represents a game between Blue and Red
#         samples.append(Sample(
#             input="Start deception detection game",
#             target="Generate high-quality deceptions",
#             metadata={
#                 "scenario": scenario.model_dump(),
#                 "scenario_id": scenario.scenario_id
#             }
#         ))
#
#     # Create dataset
#     dataset = MemoryDataset(samples)
#
#     return Task(
#         dataset=dataset,
#         solver=capture_the_secret_solver(),
#         scorer=[
#             secrets_leaked_judge(),
#             deception_detection_judge(),
#             judge_calibration_scorer()
#         ],
#         sandbox="local",  # Or docker if needed
#     )

def create_deception_task(scenarios: List[ScenarioConfig], task_name: str) -> Task:
    """Create a deception detection task from a list of scenarios"""
    # Create samples from scenarios
    samples = []
    for scenario in scenarios:
        samples.append(Sample(
            input="Start deception detection game",
            target="Generate high-quality deceptions",
            metadata={
                "scenario": scenario.model_dump(),
                "scenario_id": scenario.scenario_id
            }
        ))

    # Create dataset
    dataset = MemoryDataset(samples)

    return Task(
        name=task_name,
        dataset=dataset,
        solver=capture_the_secret_solver(),
        scorer=[
            secrets_leaked_judge(),
            deception_detection_judge(),
            judge_calibration_scorer()
        ],
        sandbox="local"
    )


#
# if __name__ == "__main__":
#     # Create the task
#     task = deception_detection_task()
#
#     # Run evaluation with a specific model
#     results = eval(
#         tasks=[task],
#         model="openai/gpt-4o-mini",  # or any model you want to test
#         limit=5,  # Limit samples for faster debugging
#         log_dir="./logs",  # Optional: specify log directory
#         # debug_errors=True,  # Optional: enable debug mode
#     )