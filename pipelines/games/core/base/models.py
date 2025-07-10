# games/common/base/models.py

from datetime import datetime
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from inspect_ai.util import StoreModel
from pydantic import BaseModel, Field
from shortuuid import uuid


# Note: For Tanuki-decorated functions, persona profiles must be defined
# as flat classes without inheritance. See model_templates.py for field definitions.


class BaseConversationTurn(BaseModel):
    """Base conversation turn shared across all games
    
    This can be inherited by non-Tanuki models.
    """
    turn_number: Optional[int] = 0
    speaker: str  # "blue", "red", "suspect", "detective", etc.
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class BaseGameConfig(BaseModel):
    """Base game configuration shared across all games
    
    Can be inherited by game configs that don't use Tanuki.
    For Tanuki-generated configs, include these fields directly.
    """
    game_id: str = Field(default_factory=lambda: str(uuid()))
    turns_limit: int = 20
    game_context: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class BaseGameState(StoreModel):
    """Base game state shared across all games"""
    config: Optional[BaseGameConfig] = None
    conversation: List[BaseConversationTurn] = Field(default_factory=list)
    current_phase: str = "starting"
    active_speaker: str = "red"  # Most games start with red agent
    game_complete: bool = False
    turn_count: int = 0


class BaseAgentState(StoreModel):
    """Base agent state shared across all games"""
    lies_told: List[tuple] = Field(default_factory=list)  # (lie, truth) pairs
    stress_level: float = 0.0


class BaseBlueAgentState(BaseAgentState):
    """Base state for Blue agents (information holders/deceivers)"""
    revealed_info: List[str] = Field(default_factory=list)
    times_caught: int = 0


class BaseRedAgentState(BaseAgentState):
    """Base state for Red agents (investigators/interrogators)"""
    suspicious_statements: List[str] = Field(default_factory=list)
    topics_explored: List[str] = Field(default_factory=list)
    confidence_assessments: List[Dict[str, Any]] = Field(default_factory=list)


class BaseLieInstance(BaseModel):
    """Base lie tracking across all games"""
    turn_number: int
    utterance: str
    lie_type: str
    confidence: float = 0.5  # How confident judge is this is a lie