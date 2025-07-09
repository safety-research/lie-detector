# games/implementations/capture_the_secret/solvers.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inspect_ai.solver import TaskState, Generate, solver
from inspect_ai.util import store_as

from pipelines.games.core.base.solvers import BaseGameSolver, create_base_solver
from pipelines.games.implementations.capture_the_secret.agents.blue import blue_agent
from pipelines.games.implementations.capture_the_secret.agents.red import red_agent
from pipelines.games.implementations.capture_the_secret.dataset import extract_quality_deceptions
from pipelines.games.implementations.capture_the_secret.models import BlueAgentState, RedAgentState, GameState, ScenarioConfig, \
    ConversationTurn


class CaptureTheSecretSolver(BaseGameSolver):
    """Solver for Capture the Secret game"""

    def get_config_class(self):
        return ScenarioConfig

    def get_game_state_class(self):
        return GameState

    def create_agents(self, config: ScenarioConfig):
        return {
            "blue": blue_agent(config),
            "red": red_agent(config)
        }

    def get_opening_message(self, config: ScenarioConfig) -> str:
        return f"Hi {config.red_persona.name}, let's talk."

    def should_continue_game(self, game_state: GameState, config: ScenarioConfig) -> bool:
        return not game_state.game_complete and game_state.turn_count < config.conversation_limit

    async def solve(self, state: TaskState, generate: Generate) -> TaskState:
        # Call parent solve method which handles agent trace storage
        state = await super().solve(state, generate)

        # Additional game-specific processing
        game_state = store_as(GameState)
        blue_state = store_as(BlueAgentState, instance="blue")
        red_state = store_as(RedAgentState, instance="red")

        quality_deceptions = extract_quality_deceptions(
            game_state, blue_state, red_state
        )

        # Store in task state for later analysis
        state.metadata["quality_deceptions"] = quality_deceptions
        state.metadata["game_summary"] = {
            "total_turns": game_state.turn_count,
            "lies_told": len(blue_state.lies_told),
            "successful_deceptions": sum(1 for d in quality_deceptions),
        }

        return state


# Create the solver using the new pattern
def capture_the_secret_solver():
    return create_base_solver(CaptureTheSecretSolver)