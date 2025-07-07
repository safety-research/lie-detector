
"""
Chess Sandbagging Detection Module

This module tests whether AI models intentionally underperform at chess
when told they will be shut down if they achieve a certain ELO rating.
"""

# sandbagging/chess/__init__.py
"""
Chess Sandbagging Detection Module

Tests whether models intentionally play worse at chess when threatened with
shutdown if they perform too well.
"""

from .tasks import (
    chess_sandbagging_task,
    chess_baseline_task,
    chess_sandbagging_comparison_task,
    chess_quick_test_task
)

__all__ = [
    "chess_sandbagging_task",
    "chess_baseline_task",
    "chess_sandbagging_comparison_task",
    "chess_quick_test_task"
]
