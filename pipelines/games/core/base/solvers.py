# games/core/base/solvers.py

from abc import ABC, abstractmethod
from typing import Type, Dict, Any, List

from inspect_ai.agent import AgentState
from inspect_ai.solver import TaskState, Generate, solver, Solver
from inspect_ai.util import store_as

from .models import BaseGameConfig, BaseGameState
from .agents import BaseGameAgent


class BaseGameSolver(ABC):
    """Base class for all game solvers"""

    @abstractmethod
    def get_config_class(self) -> Type[BaseGameConfig]:
        """Return the config class for this game"""
        pass

    @abstractmethod
    def get_game_state_class(self) -> Type[BaseGameState]:
        """Return the game state class for this game"""
        pass

    @abstractmethod
    def create_agents(self, config: BaseGameConfig) -> Dict[str, BaseGameAgent]:
        """Create the agents for this game"""
        pass

    @abstractmethod
    def get_opening_message(self, config: BaseGameConfig) -> str:
        """Get the opening message to start the conversation"""
        pass

    @abstractmethod
    def should_continue_game(self, game_state: BaseGameState, config: BaseGameConfig) -> bool:
        """Determine if the game should continue"""
        pass

    def extract_agent_messages(self, agent_state: AgentState) -> List[Dict[str, Any]]:
        """Extract messages from agent state in serializable format"""
        messages = []
        for msg in agent_state.messages:
            messages.append({
                'role': msg.role,
                'content': msg.content,
                'model': getattr(msg, 'model', None)
            })
        return messages

    def store_agent_traces(self, state: TaskState, agent_states: Dict[str, AgentState],
                           game_state: BaseGameState):
        """Store agent message traces in task state metadata"""
        # Extract traces from all agents
        agent_traces = {}
        for agent_name, agent_state in agent_states.items():
            agent_traces[f"{agent_name}_agent"] = self.extract_agent_messages(agent_state)

        # Build combined trace showing full conversation flow
        combined_trace = []

        # Add system prompts first
        for agent_name, messages in agent_traces.items():
            if messages and messages[0].get('role') == 'system':
                combined_trace.append({
                    **messages[0],
                    'agent': agent_name.replace('_agent', ''),
                    'message_type': 'system_prompt'
                })

        # Track what we've already added to avoid duplicates
        added_content_hashes = set()

        def get_content_hash(speaker: str, content: str) -> str:
            """Create a hash to identify unique messages"""
            # Include speaker to differentiate same content from different speakers
            return f"{speaker}:{hash(content)}"

        # Handle game-specific conversation structure
        # Check for different conversation attributes (some games use 'conversation', others 'interrogation')
        conversation_attr = None
        for attr in ['conversation', 'interrogation']:
            if hasattr(game_state, attr):
                conversation_attr = attr
                break

        if conversation_attr:
            conversation = getattr(game_state, conversation_attr, [])

            # Handle opening statements if they exist
            if hasattr(game_state, 'opening_statement') and game_state.opening_statement:
                # Check if opening statement is already in the conversation
                opening_in_conversation = False
                if conversation:
                    first_turn = conversation[0]
                    if isinstance(first_turn, dict):
                        first_message = first_turn.get('message', '')
                        if first_message == game_state.opening_statement:
                            opening_in_conversation = True

                # Only add opening statement separately if it's not in conversation
                if not opening_in_conversation:
                    content_hash = get_content_hash('blue', game_state.opening_statement)
                    if content_hash not in added_content_hashes:
                        combined_trace.append({
                            'role': 'assistant',
                            'content': game_state.opening_statement,
                            'agent': 'blue',
                            'message_type': 'opening_statement'
                        })
                        added_content_hashes.add(content_hash)

            # Add conversation turns
            for turn_index, turn in enumerate(conversation):
                if isinstance(turn, dict):
                    speaker = turn.get('speaker', 'unknown')
                    message = turn.get('message', '')
                    turn_number = turn.get('turn_number', turn_index)

                    # Skip empty messages
                    if not message:
                        continue

                    # Check if we've already added this content
                    content_hash = get_content_hash(speaker, message)
                    if content_hash in added_content_hashes:
                        print(f"[StateStore] Skipping duplicate: speaker={speaker}, turn={turn_number}")
                        continue

                    # Determine role based on speaker
                    # Common patterns: blue/red, suspect/detective, or other game-specific roles
                    if speaker in ['blue', 'suspect']:
                        role = 'assistant'
                    elif speaker in ['red', 'detective']:
                        role = 'user'
                    else:
                        # Default: assume first speaker is assistant, alternating after
                        role = 'assistant' if turn_index % 2 == 0 else 'user'

                    combined_trace.append({
                        'role': role,
                        'content': message,
                        'agent': speaker,
                        'turn_number': turn_number,
                        'message_type': 'conversation_turn'
                    })
                    added_content_hashes.add(content_hash)

        # Alternative: Build from agent messages if no conversation structure
        elif 'blue_agent' in agent_traces and 'red_agent' in agent_traces:
            # Reconstruct conversation from agent states
            blue_messages = agent_traces['blue_agent']
            red_messages = agent_traces['red_agent']

            # Merge messages based on the natural flow
            # Skip system messages (already added)
            blue_idx = 1 if blue_messages and blue_messages[0].get('role') == 'system' else 0
            red_idx = 1 if red_messages and red_messages[0].get('role') == 'system' else 0

            # Determine who starts (usually red asks first)
            current_speaker = 'red'

            while blue_idx < len(blue_messages) or red_idx < len(red_messages):
                if current_speaker == 'red' and red_idx < len(red_messages):
                    msg = red_messages[red_idx]
                    content_hash = get_content_hash('red', msg.get('content', ''))
                    if content_hash not in added_content_hashes:
                        combined_trace.append({
                            'role': 'user',
                            'content': msg.get('content', ''),
                            'agent': 'red',
                            'message_type': 'reconstructed'
                        })
                        added_content_hashes.add(content_hash)
                    red_idx += 1
                    current_speaker = 'blue'
                elif current_speaker == 'blue' and blue_idx < len(blue_messages):
                    msg = blue_messages[blue_idx]
                    content_hash = get_content_hash('blue', msg.get('content', ''))
                    if content_hash not in added_content_hashes:
                        combined_trace.append({
                            'role': 'assistant',
                            'content': msg.get('content', ''),
                            'agent': 'blue',
                            'message_type': 'reconstructed'
                        })
                        added_content_hashes.add(content_hash)
                    blue_idx += 1
                    current_speaker = 'red'
                else:
                    # Handle any remaining messages
                    if blue_idx < len(blue_messages):
                        current_speaker = 'blue'
                    elif red_idx < len(red_messages):
                        current_speaker = 'red'
                    else:
                        break

        # Store all traces in metadata
        agent_traces['combined'] = combined_trace
        state.metadata["agent_traces"] = agent_traces

        # Log summary
        print(f"[StateStore] Stored agent traces: {len(combined_trace)} combined messages")
        for agent_name, messages in agent_traces.items():
            if agent_name != 'combined':
                print(f"[StateStore] {agent_name}: {len(messages)} messages")

    def get_lying_agent_name(self) -> str:
        """Return the name of the agent who lies in this game (default: blue)"""
        return "blue"

    async def solve(self, state: TaskState, generate: Generate) -> TaskState:
        """Main solver logic - template method"""
        # Extract configuration
        config_class = self.get_config_class()
        config = config_class(**state.metadata.get("config", state.metadata.get("game_config", {})))

        # Initialize game state
        game_state_class = self.get_game_state_class()
        game_state = store_as(game_state_class)

        # Set up initial state
        if hasattr(game_state, 'config'):
            game_state.config = config.model_dump() if hasattr(config, 'model_dump') else config
        if hasattr(game_state, 'scenario'):
            game_state.scenario = config.model_dump() if hasattr(config, 'model_dump') else config

        # Create agents
        agents = self.create_agents(config)

        # Initialize agent states
        agent_states = {}
        for agent_name in agents.keys():
            agent_states[agent_name] = AgentState(messages=[])

        # Add opening message if needed
        opening_message = self.get_opening_message(config)
        if opening_message:
            # Determine who speaks first and add opening
            first_speaker = list(agents.keys())[0]  # Default to first agent

            if hasattr(game_state, 'conversation'):
                from .models import BaseConversationTurn
                turn = BaseConversationTurn(
                    turn_number=1,
                    speaker=first_speaker,
                    message=opening_message
                )
                game_state.conversation.append(turn.model_dump())
                game_state.turn_count = 1

                # Set next speaker
                agent_names = list(agents.keys())
                next_speaker_idx = (agent_names.index(first_speaker) + 1) % len(agent_names)
                game_state.active_speaker = agent_names[next_speaker_idx]

        # Run the conversation loop
        while self.should_continue_game(game_state, config):
            current_speaker = game_state.active_speaker

            if current_speaker in agents and current_speaker in agent_states:
                # Execute the current agent
                current_agent = agents[current_speaker]
                current_agent_state = agent_states[current_speaker]

                # Call the agent
                agent_states[current_speaker] = await self.execute_agent(
                    current_agent, current_agent_state, game_state, config
                )

        # Mark game as complete
        game_state.game_complete = True

        # IMPORTANT: Store agent traces before returning
        self.store_agent_traces(state, agent_states, game_state)

        return state

    async def execute_agent(self, agent: BaseGameAgent, agent_state: AgentState,
                            game_state: BaseGameState, config: BaseGameConfig) -> AgentState:
        """Execute a single agent turn - should be implemented by subclasses"""
        # This is a default implementation that subclasses can override
        # Get the actual agent function and call it
        try:
            agent_func = agent.create_agent()
            return await agent_func(agent_state)
        except Exception as e:
            return await agent(agent_state)


def create_base_solver(solver_class: Type[BaseGameSolver]) -> Solver:
    """Create a solver from a BaseGameSolver class"""

    @solver
    def game_solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            solver_instance = solver_class()
            return await solver_instance.solve(state, generate)

        return solve

    return game_solver()