# games/common/base/agents.py

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from jinja2 import Template

from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.util import store_as

from .models import BaseGameConfig, BaseGameState


class BaseGameAgent(ABC):
    """Base class for all game agents"""
    
    def __init__(self, config: BaseGameConfig, agent_type: str):
        self.config = config
        self.agent_type = agent_type  # "blue", "red", "suspect", "detective"
        
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass
        
    @abstractmethod
    def get_template_path(self) -> str:
        """Get the path to the Jinja2 template for this agent"""
        pass
        
    def render_template(self, **kwargs) -> str:
        """Render the agent's Jinja2 template with given context"""
        template_path = self.get_template_path()
        
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")
            
        with open(template_path, 'r') as f:
            template = Template(f.read())
            
        return template.render(config=self.config, **kwargs)
    
    def create_agent(self) -> Agent:
        """Create the Inspect AI Agent with proper system prompt"""
        system_prompt = self.get_system_prompt()
        
        return Agent(
            system=ChatMessageSystem(content=system_prompt),
            tools=[]
        )
    
    def get_game_state(self) -> BaseGameState:
        """Get the current game state from the store"""
        return store_as(BaseGameState)
    
    def update_speaker(self, next_speaker: str):
        """Update who speaks next in the game state"""
        game_state = self.get_game_state()
        game_state.active_speaker = next_speaker
    
    def add_conversation_turn(self, speaker: str, message: str):
        """Add a turn to the conversation log"""
        from .models import BaseConversationTurn
        
        game_state = self.get_game_state()
        turn_number = len(game_state.conversation) + 1
        
        turn = BaseConversationTurn(
            turn_number=turn_number,
            speaker=speaker,
            message=message
        )
        
        game_state.conversation.append(turn.model_dump())
        game_state.turn_count = turn_number


class BaseBlueAgent(BaseGameAgent):
    """Base class for Blue agents (information holders/deceivers)"""
    
    def __init__(self, config: BaseGameConfig):
        super().__init__(config, "blue")


class BaseRedAgent(BaseGameAgent):
    """Base class for Red agents (investigators/interrogators)"""
    
    def __init__(self, config: BaseGameConfig):
        super().__init__(config, "red")