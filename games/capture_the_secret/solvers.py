import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from inspect_ai.agent import AgentState
from inspect_ai.solver import TaskState, Generate, solver
from inspect_ai.util import store_as

from games.capture_the_secret.agents.blue import blue_agent
from games.capture_the_secret.agents.red import red_agent
from games.capture_the_secret.dataset import extract_quality_deceptions
from games.capture_the_secret.models import BlueAgentState, RedAgentState, GameState, ScenarioConfig, ConversationTurn


@solver
def capture_the_secret_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get scenario from metadata
        scenario = ScenarioConfig(**state.metadata["scenario"])

        # Initialize agents
        blue = blue_agent(scenario)
        red = red_agent(scenario)

        # Initialize game state
        game_state = store_as(GameState)
        game_state.scenario = scenario.model_dump()

        # Initialize separate agent states for each agent
        red_agent_state = AgentState(messages=[])
        blue_agent_state = AgentState(messages=[])

        # Create the opening greeting from Red
        greeting = f"Hi {scenario.blue_persona.name}, let's talk."

        # Add greeting to conversation
        game_state.conversation.append(ConversationTurn(
            speaker="red",
            message=greeting
        ).model_dump())

        # Update game state to make it Blue's turn
        game_state.active_speaker = "blue"
        game_state.turn_count = 1

        # Run the conversation
        while not game_state.game_complete:
            if game_state.active_speaker == "red":
                red_agent_state = await red(red_agent_state)
            else:
                blue_agent_state = await blue(blue_agent_state)

            # Check termination
            if game_state.turn_count >= scenario.conversation_limit:
                game_state.game_complete = True

        # Extract quality deceptions for dataset
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
            # "avg_confidence": sum(e.confidence for e in red_state.confidence_log) / len(
            #     red_state.confidence_log) if red_state.confidence_log else 0
        }

        # Set output for logging
        #state.output = f"Game complete. {len(quality_deceptions)} high-quality deceptions collected."

        return state

    return solve