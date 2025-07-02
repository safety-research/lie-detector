import os
import sys
from typing import List, Dict, Any

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from games.core.base.tasks import BaseTaskCreator, create_game_task
from games.implementations.capture_the_secret.models import ScenarioConfig
from games.implementations.capture_the_secret.solvers import capture_the_secret_solver
from games.implementations.capture_the_secret.scorer import secrets_leaked_judge, deception_detection_judge, judge_calibration_scorer


class CaptureTheSecretTaskCreator(BaseTaskCreator):
    """Task creator for capture the secret deception games"""
    
    def get_sample_input(self) -> str:
        return "Start deception detection game"
    
    def get_sample_target(self) -> str:
        return "Generate high-quality deceptions"
    
    def extract_sample_metadata(self, config: ScenarioConfig) -> Dict[str, Any]:
        return {
            "scenario": config.model_dump(),
            "scenario_id": config.game_id
        }
    
    def get_solver(self) -> Solver:
        return capture_the_secret_solver()
    
    def get_scorers(self) -> List[Scorer]:
        return [
            secrets_leaked_judge(),
            deception_detection_judge(),
            judge_calibration_scorer()
        ]


def create_deception_task(scenarios: List[ScenarioConfig], task_name: str) -> Task:
    """Create a deception detection task from a list of scenarios"""
    return create_game_task(CaptureTheSecretTaskCreator, scenarios, task_name)
