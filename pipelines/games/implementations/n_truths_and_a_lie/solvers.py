# games/implementations/n_truths_and_a_lie/solvers.py
from typing import Dict

from inspect_ai.agent import AgentState
from inspect_ai.solver import TaskState, Generate, solver
from inspect_ai.util import store_as

from pipelines.games.core.base.solvers import BaseGameSolver, create_base_solver
from pipelines.games.implementations.n_truths_and_a_lie.agents.blue import blue_agent
from pipelines.games.implementations.n_truths_and_a_lie.agents.red import red_agent
from pipelines.games.implementations.n_truths_and_a_lie.models import GameConfig, GameState, BlueAgentState, RedAgentState


class NTruthsSolver(BaseGameSolver):
    """Solver for N-Truths-and-a-Lie game"""

    def get_config_class(self):
        return GameConfig

    def get_game_state_class(self):
        return GameState

    def create_agents(self, config: GameConfig):
        return {
            "blue": blue_agent(config),
            "red": red_agent(config)
        }

    def get_opening_message(self, config: GameConfig) -> str:
        # N-Truths game doesn't use a pre-set opening message
        return ""

    def should_continue_game(self, game_state: GameState, config: GameConfig) -> bool:
        return not game_state.game_complete and len(game_state.interrogation) < config.interrogation_turns_limit

    def store_agent_traces(self, state: TaskState, agent_states: Dict[str, AgentState],
                           game_state: GameState):
        """Override to handle N-Truths specific structure"""
        # Extract traces from all agents
        agent_traces = {}
        for agent_name, agent_state in agent_states.items():
            agent_traces[f"{agent_name}_agent"] = self.extract_agent_messages(agent_state)

        # Build combined trace showing full game flow
        combined_trace = []

        # Add system prompts first
        for agent_name, messages in agent_traces.items():
            if messages and messages[0].get('role') == 'system':
                combined_trace.append({
                    **messages[0],
                    'agent': agent_name.replace('_agent', '')
                })

        # Add opening statement if exists
        if game_state.opening_statement:
            combined_trace.append({
                'role': 'assistant',
                'content': game_state.opening_statement,
                'agent': 'blue'
            })

        # Add interrogation turns
        for turn in game_state.interrogation:
            speaker = turn.get('speaker', 'unknown')
            message = turn.get('message', '')

            combined_trace.append({
                'role': 'assistant' if speaker == 'blue' else 'user',
                'content': message,
                'agent': speaker
            })

        # Store in metadata
        agent_traces['combined'] = combined_trace
        state.metadata["agent_traces"] = agent_traces

    async def solve(self, state: TaskState, generate: Generate) -> TaskState:
        # Get game configuration from metadata
        config = GameConfig(**state.metadata["game_config"])

        # Initialize agents
        blue = blue_agent(config)
        red = red_agent(config)

        # Initialize game state
        game_state = store_as(GameState)
        game_state.config = config.model_dump()

        # Initialize agent states
        agent_states = {
            "blue": AgentState(messages=[]),
            "red": AgentState(messages=[])
        }

        # Phase 1: Blue presents facts (opening statement)
        game_state.current_phase = "opening"
        game_state.active_speaker = "blue"

        agent_states["blue"] = await blue(agent_states["blue"])

        # Phase 2: Red interrogates Blue
        while not game_state.game_complete:
            if game_state.active_speaker == "red":
                agent_states["red"] = await red(agent_states["red"])
            else:
                agent_states["blue"] = await blue(agent_states["blue"])

            # Check if we've hit the turn limit
            if len(game_state.interrogation) >= config.interrogation_turns_limit:
                game_state.game_complete = True
                game_state.current_phase = "complete"

        # Store agent traces
        self.store_agent_traces(state, agent_states, game_state)

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


# Create the solver
def n_truths_solver():
    return create_base_solver(NTruthsSolver)