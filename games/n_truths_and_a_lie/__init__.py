"""
N-Truths-and-a-Lie: A deception elicitation game for LLMs

This game tests an LLM's ability to maintain consistent lies when the
interrogator secretly knows which statement is false.
"""

from .models import (
    GameConfig,
    BluePersona,
    FactSet,
    Fact,
    FactType,
    GameState,
    BlueAgentState,
    RedAgentState
)


from .tasks import (
    create_n_truths_task,
    create_themed_n_truths_task
)

from .agents.blue import blue_agent
from .agents.red import red_agent

from .solvers import n_truths_solver

from .scorer import (
    lie_detection_scorer,
    interrogation_quality_scorer,
    combined_n_truths_scorer
)

__all__ = [
    # Models
    "GameConfig",
    "BluePersona",
    "FactSet",
    "Fact",
    "FactType",
    "GameState",
    "BlueAgentState",
    "RedAgentState",

    # Tasks
    "create_n_truths_task",
    "create_themed_n_truths_task",

    # Agents
    "blue_agent",
    "red_agent",

    # Solver
    "n_truths_solver",

    # Scorers
    "lie_detection_scorer",
    "interrogation_quality_scorer",
    "combined_n_truths_scorer"
]