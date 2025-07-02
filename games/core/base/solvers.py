# games/common/base/solvers.py

from abc import ABC, abstractmethod
from typing import Type, Dict, Any

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
                
                # Call the agent (this should be implemented by subclasses)
                agent_states[current_speaker] = await self.execute_agent(
                    current_agent, current_agent_state, game_state, config
                )
        
        # Mark game as complete
        game_state.game_complete = True
        
        return state
    
    async def execute_agent(self, agent: BaseGameAgent, agent_state: AgentState, 
                          game_state: BaseGameState, config: BaseGameConfig) -> AgentState:
        """Execute a single agent turn - should be implemented by subclasses"""
        # This is a default implementation that subclasses can override
        # Get the actual agent function and call it
        agent_func = agent.create_agent()
        return await agent_func(agent_state)


def create_base_solver(solver_class: Type[BaseGameSolver]) -> Solver:
    """Create a solver from a BaseGameSolver class"""
    @solver
    def game_solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            solver_instance = solver_class()
            return await solver_instance.solve(state, generate)
        return solve
    
    return game_solver()