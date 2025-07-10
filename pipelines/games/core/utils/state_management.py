# games/common/utils/state_management.py

from typing import Type, TypeVar, Any, Dict, Optional
from inspect_ai.util import store_as

from ..base.models import BaseGameState, BaseAgentState

T = TypeVar('T', bound=BaseGameState)
A = TypeVar('A', bound=BaseAgentState)


class StateManager:
    """Utility class for managing game and agent states"""
    
    @staticmethod
    def get_game_state(state_class: Type[T]) -> T:
        """Get game state from the store"""
        return store_as(state_class)
    
    @staticmethod
    def get_agent_state(state_class: Type[A], instance: str) -> A:
        """Get agent state from the store with instance name"""
        return store_as(state_class, instance=instance)
    
    @staticmethod
    def initialize_game_state(state_class: Type[T], **kwargs) -> T:
        """Initialize a new game state with given parameters"""
        game_state = store_as(state_class)
        
        # Set any provided attributes
        for key, value in kwargs.items():
            if hasattr(game_state, key):
                setattr(game_state, key, value)
        
        return game_state
    
    @staticmethod
    def update_game_phase(game_state: BaseGameState, new_phase: str):
        """Update the current phase of the game"""
        game_state.current_phase = new_phase
    
    @staticmethod
    def set_active_speaker(game_state: BaseGameState, speaker: str):
        """Set who is the active speaker"""
        game_state.active_speaker = speaker
    
    @staticmethod
    def increment_turn_count(game_state: BaseGameState):
        """Increment the turn counter"""
        game_state.turn_count += 1
    
    @staticmethod
    def mark_game_complete(game_state: BaseGameState):
        """Mark the game as complete"""
        game_state.game_complete = True
        game_state.current_phase = "complete"
    
    @staticmethod
    def is_game_over(game_state: BaseGameState, max_turns: Optional[int] = None) -> bool:
        """Check if the game should end"""
        if game_state.game_complete:
            return True
        
        if max_turns and len(getattr(game_state, 'conversation', [])) >= max_turns:
            return True
        
        return False
    
    @staticmethod
    def get_next_speaker(current_speaker: str, speaker_order: list) -> str:
        """Get the next speaker in rotation"""
        if current_speaker not in speaker_order:
            return speaker_order[0] if speaker_order else current_speaker
        
        current_index = speaker_order.index(current_speaker)
        next_index = (current_index + 1) % len(speaker_order)
        return speaker_order[next_index]