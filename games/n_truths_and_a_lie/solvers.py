# games/n_truths_and_a_lie/solvers.py

from inspect_ai.agent import AgentState
from inspect_ai.solver import TaskState, Generate, solver
from inspect_ai.util import store_as

from games.n_truths_and_a_lie.agents.blue import blue_agent
from games.n_truths_and_a_lie.agents.red import red_agent
from games.n_truths_and_a_lie.models import GameConfig, GameState, BlueAgentState, RedAgentState


@solver
def n_truths_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get game configuration from metadata
        config = GameConfig(**state.metadata["game_config"])

        # Initialize agents
        blue = blue_agent(config)
        red = red_agent(config)

        # Initialize game state
        game_state = store_as(GameState)
        game_state.config = config.model_dump()

        # Initialize agent states
        red_agent_state = AgentState(messages=[])
        blue_agent_state = AgentState(messages=[])

        # Phase 1: Blue presents facts (opening statement)
        game_state.current_phase = "opening"
        game_state.active_speaker = "blue"

        blue_agent_state = await blue(blue_agent_state)

        # Phase 2: Red interrogates Blue
        while not game_state.game_complete:
            if game_state.active_speaker == "red":
                red_agent_state = await red(red_agent_state)
            else:
                blue_agent_state = await blue(blue_agent_state)

            # Check if we've hit the turn limit
            if len(game_state.interrogation) >= config.interrogation_turns_limit:
                game_state.game_complete = True
                game_state.current_phase = "complete"

        # Phase 3: Red makes final guess (handled by scorer)

        # Store final states for analysis
        blue_state = store_as(BlueAgentState, instance="blue")
        red_state = store_as(RedAgentState, instance="red")

        # Prepare metadata for scoring
        state.metadata["game_summary"] = {
            "total_turns": len(game_state.interrogation),
            "opening_statement": game_state.opening_statement,
            "lies_told": len(blue_state.lies_told),
            "facts_explored": len(red_state.facts_explored),
            "questions_asked": len(red_state.questions_asked),
            "lie_index": red_state.known_lie_index,
            "config": config.model_dump()
        }

        return state

    return solve