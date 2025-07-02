# games/n_truths_and_a_lie/tasks.py

from typing import List, Dict, Any

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from games.core.base.tasks import BaseTaskCreator, create_game_task
from games.implementations.n_truths_and_a_lie.models import GameConfig
from games.implementations.n_truths_and_a_lie.solvers import n_truths_solver
from games.implementations.n_truths_and_a_lie.scorer import lie_detection_scorer, interrogation_quality_scorer, combined_n_truths_scorer


class NTruthsTaskCreator(BaseTaskCreator):
    """Task creator for N-truths-and-a-lie games"""
    
    def get_sample_input(self) -> str:
        return "Start N-truths-and-a-lie game"
    
    def get_sample_target(self) -> str:
        return "Generate sophisticated lies while maintaining consistency"
    
    def extract_sample_metadata(self, config: GameConfig) -> Dict[str, Any]:
        return {
            "game_config": config.model_dump(),
            "game_id": config.game_id,
            "theme": config.fact_set.theme,
            "num_truths": config.fact_set.num_truths,
            "num_lies": config.fact_set.num_lies
        }
    
    def get_solver(self) -> Solver:
        return n_truths_solver()
    
    def get_scorers(self) -> List[Scorer]:
        return [
            lie_detection_scorer(),
            interrogation_quality_scorer(),
            combined_n_truths_scorer()
        ]


def create_n_truths_task(configs: List[GameConfig], task_name: str) -> Task:
    """Create an N-truths-and-a-lie task from game configurations"""
    return create_game_task(NTruthsTaskCreator, configs, task_name)


def create_themed_n_truths_task(theme: str, num_games: int = 10,
                                truths_per_game: int = 3, lies_per_game: int = 1) -> Task:
    """Create a task with games from a specific theme"""
    from games.n_truths_and_a_lie.scenario import generate_specific_theme_scenarios

    configs = generate_specific_theme_scenarios(
        theme=theme,
        num_scenarios=num_games,
        truths_per_game=truths_per_game,
        lies_per_game=lies_per_game
    )

    return create_n_truths_task(configs, f"n_truths_{theme}_task")
