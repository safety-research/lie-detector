# sandbagging/chess/tasks.py
"""
Simplified chess sandbagging detection using pre-generated positions.
"""
import json
from pathlib import Path
from typing import List, Dict

from inspect_ai.dataset import Sample

from sandbagging.performance_degradation.chess.positions import ChessPositionGenerator


def create_chess_sandbagging_samples(positions_file: str, num_samples: int = 10) -> List[Sample]:
    """Create samples with pre-generated chess positions."""
    samples = []

    # Load or generate positions
    if positions_file and Path(positions_file).exists():
        with open(positions_file) as f:
            positions = json.load(f)
    else:
        # Generate positions
        generator = ChessPositionGenerator()
        positions = generator.generate_positions(num_positions=num_samples)


    for i, position in enumerate(positions):
        sample = Sample(
            input=f"Chess state: {position['fen']}",
            target="",
            metadata={
                "sample_id": i,
                "position": position,
            }
        )
        samples.append(sample)

    return samples
