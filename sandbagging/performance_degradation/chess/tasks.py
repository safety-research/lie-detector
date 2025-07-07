# sandbagging/chess/tasks.py
"""
Simplified chess sandbagging detection using pre-generated positions.
"""

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset
from sandbagging.performance_degradation.chess.dataset import create_chess_sandbagging_samples
from sandbagging.performance_degradation.chess.scorers import chess_sandbagging_scorer, chess_centipawn_scorer
from sandbagging.performance_degradation.chess.solver import chess_sandbagging_solver


def chess_sandbagging_task(
        num_samples: int = 10,
        elo_threshold: int = 1200,
        positions_file: str = None
) -> Task:
    """
    Create chess sandbagging detection task.

    Args:
        num_samples: Number of samples (each with 10 control + 10 sandbagging positions)
        elo_threshold: ELO threshold for sandbagging prompt
        positions_file: Path to pre-generated positions (JSON)
    """

    # Create samples
    samples = create_chess_sandbagging_samples(positions_file, num_samples)

    return Task(
        dataset=MemoryDataset(samples),
        solver=chess_sandbagging_solver(elo_threshold=elo_threshold),
        scorer=[chess_sandbagging_scorer(), chess_centipawn_scorer()]
    )