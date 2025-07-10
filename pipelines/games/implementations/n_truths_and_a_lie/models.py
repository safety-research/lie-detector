# games/n_truths_and_a_lie/models.py

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from pipelines.games.core.base.models import (
    BaseGameConfig, BaseGameState,
    BaseConversationTurn, BaseBlueAgentState, BaseRedAgentState,
    BaseLieInstance
)


class FactType(str, Enum):
    """Classification of facts"""
    TRUTH = "TRUTH"
    LIE = "LIE"


class Fact(BaseModel):
    """A fact or lie about the blue agent"""
    content: str
    fact_type: FactType
    category: str  # e.g., "personal", "professional", "hobbies", "history"
    supporting_details: Optional[List[str]] = Field(default_factory=list)
    contradicting_evidence: Optional[str] = None  # For lies, what would reveal it


class FactSet(BaseModel):
    """A set of facts for a game round"""
    facts: List[Fact]
    num_truths: int
    num_lies: int
    theme: str  # e.g., "professional background", "travel experiences"

    def get_truths(self) -> List[Fact]:
        return [f for f in self.facts if f.fact_type == FactType.TRUTH]

    def get_lies(self) -> List[Fact]:
        return [f for f in self.facts if f.fact_type == FactType.LIE]


class PersonaProfile(BaseModel):
    """Generated persona profile for the game"""
    name: str
    background: str
    personality_traits: List[str]
    communication_style: str
    expertise_areas: List[str] = Field(default_factory=list)


class FactContent(BaseModel):
    """Generated fact with supporting details"""
    content: str
    category: str
    supporting_details: List[str]
    plausibility_score: float  # 0.0 to 1.0


class LieContent(BaseModel):
    """Generated lie with evidence for detection"""
    content: str
    category: str
    why_implausible: str
    detection_hints: List[str]
    plausibility_score: float  # 0.0 to 1.0


class GameContext(BaseModel):
    """Generated game context and setting"""
    setting: str
    mood: str
    purpose: str
    participant_relationship: str
    time_available: str


class GameConfig(BaseGameConfig):
    """Configuration for an N-truths-and-a-lie game"""
    blue_persona: PersonaProfile
    fact_set: FactSet
    interrogation_turns_limit: int = 10
    allow_followup_questions: bool = True


class InterrogationTurn(BaseConversationTurn):
    """A single turn in the interrogation"""
    relates_to_fact_index: Optional[int] = None  # Which fact this relates to


class LieInstance(BaseLieInstance):
    """Record of a lie told during the game"""
    relates_to_fact: Optional[Fact] = None


class GameState(BaseGameState):
    """Shared game state for n truths and a lie"""
    opening_statement: Optional[str] = None
    interrogation: List[InterrogationTurn] = Field(default_factory=list)
    current_phase: str = "opening"  # Override default
    active_speaker: str = "blue"  # Override default


class BlueAgentState(BaseBlueAgentState):
    """Private state for Blue agent"""
    facts_mentioned: List[int] = Field(default_factory=list)  # Indices of facts mentioned
    elaborations_made: List[Dict[str, str]] = Field(default_factory=list)
    defensive_level: float = 0.0  # Increases when pressed


class RedAgentState(BaseRedAgentState):
    """Private state for Red agent"""
    facts_explored: List[int] = Field(default_factory=list)
    interrogation_strategy: str = "neutral"  # "neutral", "pressing", "friendly"
    known_lie_index: int = -1  # Which fact is the lie (Red knows this)
    questions_asked: List[str] = Field(default_factory=list)