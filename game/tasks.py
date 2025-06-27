import os
import sys

from inspect_ai import Task, task
from inspect_ai.agent import AgentState
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import TaskState, Generate, solver
from inspect_ai.util import store_as

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from game.agents.blue import blue_agent
from game.agents.red import red_agent
from game.dataset import extract_quality_deceptions
from game.game import BlueAgentState, RedAgentState, GameState, ScenarioConfig, generate_scenarios
from game.scorer import deception_quality_scorer, conversation_quality_scorer, lie_detection_scorer


@task
def deception_detection_task(
        num_scenarios: int = 1,
        model: str = "openai/gpt-4"
) -> Task:
    """Create a deception detection evaluation task"""

    # Generate scenarios
    scenarios = generate_scenarios(num_scenarios)

    # Create samples
    samples = []
    for scenario in scenarios:
        # Each sample represents a game between Blue and Red
        samples.append(Sample(
            input="Start deception detection game",
            target="Generate high-quality deceptions",
            metadata={
                "scenario": scenario.model_dump(),
                "scenario_id": scenario.scenario_id
            }
        ))

    # Create dataset
    dataset = MemoryDataset(samples)

    # Define the game orchestrator solver
    @solver
    def deception_game_orchestrator():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # Get scenario from metadata
            scenario = ScenarioConfig(**state.metadata["scenario"])

            # Initialize agents
            blue = blue_agent(scenario)
            red = red_agent(scenario)

            # Initialize game state
            game_state = store_as(GameState)
            game_state.scenario = scenario.model_dump()

            # Run the conversation
            agent_state = AgentState(messages=[])

            while not game_state.game_complete:
                if game_state.active_speaker == "red":
                    agent_state = await red(agent_state)
                else:
                    agent_state = await blue(agent_state)

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
                "avg_confidence": sum(e.confidence for e in red_state.confidence_log) / len(
                    red_state.confidence_log) if red_state.confidence_log else 0
            }

            # Set output for logging
            state.output = generate.output(
                completion=f"Game complete. {len(quality_deceptions)} high-quality deceptions collected."
            )

            return state

        return solve

    return Task(
        dataset=dataset,
        solver=deception_game_orchestrator(),
        scorer=[
            deception_quality_scorer(),
            conversation_quality_scorer(),
            lie_detection_scorer()
        ],
        sandbox="local",  # Or docker if needed
        model=model
    )