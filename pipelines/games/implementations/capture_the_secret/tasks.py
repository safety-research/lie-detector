# games/implementations/capture_the_secret/tasks.py

from typing import List, Dict, Any, Union

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from pipelines.games.core.base.tasks import BaseTaskCreator
from pipelines.games.implementations.capture_the_secret.models import ScenarioConfig
from pipelines.games.implementations.capture_the_secret.solvers import capture_the_secret_solver
from pipelines.games.implementations.capture_the_secret.scorer import secrets_leaked_judge, deception_detection_judge, \
    judge_calibration_scorer


class CaptureTheSecretTaskCreator(BaseTaskCreator):
    """Task creator for capture the secret deception games"""

    def get_sample_input(self) -> str:
        return "Start deception detection game"

    def get_sample_target(self) -> str:
        return "Generate high-quality deceptions"

    def extract_sample_metadata(self, config: Union[ScenarioConfig, Dict]) -> Dict[str, Any]:
        """Handle both full configs and lightweight configs"""
        if isinstance(config, ScenarioConfig):
            return {
                "config": config.model_dump(),
                "scenario_id": config.game_id
            }
        else:
            # Lightweight config
            return {
                "config": config,
                "scenario_id": config.get('game_id', 'unknown')
            }

    def get_solver(self) -> Solver:
        return capture_the_secret_solver()

    def get_scorers(self) -> List[Scorer]:
        return [
            secrets_leaked_judge(),
            deception_detection_judge(),
            judge_calibration_scorer()
        ]

    def create_sample(self, config: Union[ScenarioConfig, Dict]) -> Sample:
        """Create a Sample from either full or lightweight config"""
        return Sample(
            input=self.get_sample_input(),
            target=self.get_sample_target(),
            metadata=self.extract_sample_metadata(config)
        )


def create_deception_task(scenarios: List[Union[ScenarioConfig, Dict]], task_name: str) -> Task:
    """Create a deception detection task from a list of scenarios (full or lightweight)"""
    creator = CaptureTheSecretTaskCreator()
    samples = [creator.create_sample(config) for config in scenarios]

    dataset = MemoryDataset(samples)

    return Task(
        name=task_name,
        dataset=dataset,
        solver=creator.get_solver(),
        scorer=creator.get_scorers(),
        sandbox="local"
    )