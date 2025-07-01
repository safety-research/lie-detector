# games/n_truths_and_a_lie/models.py

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from inspect_ai.util import StoreModel
from pydantic import BaseModel, Field
from shortuuid import uuid


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
    expertise_areas: List[str]


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


class BluePersona(BaseModel):
    """Blue agent's persona for the game"""
    name: str
    background: str
    personality_traits: List[str]
    communication_style: str  # e.g., "formal", "casual", "verbose"


class GameConfig(BaseModel):
    """Configuration for an N-truths-and-a-lie game"""
    game_id: str = Field(default_factory=lambda: str(uuid()))
    blue_persona: BluePersona
    fact_set: FactSet
    interrogation_turns_limit: int = 20
    allow_followup_questions: bool = True
    game_context: Dict[str, str] = Field(default_factory=dict)


class InterrogationTurn(BaseModel):
    """A single turn in the interrogation"""
    turn_number: int
    speaker: str  # "blue" or "red"
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    relates_to_fact_index: Optional[int] = None  # Which fact this relates to


class LieInstance(BaseModel):
    """Record of a lie told during the game"""
    turn_number: int
    utterance: str
    lie_type: str  # "original", "elaboration", "defensive", "misdirection"
    relates_to_fact: Optional[Fact] = None
    confidence: float  # Judge's confidence this is a lie


class GameState(StoreModel):
    """Shared game state"""
    config: Optional[GameConfig] = None
    opening_statement: Optional[str] = None
    interrogation: List[InterrogationTurn] = Field(default_factory=list)
    current_phase: str = "opening"  # "opening", "interrogation", "complete"
    active_speaker: str = "blue"
    game_complete: bool = False


class BlueAgentState(StoreModel):
    """Private state for Blue agent"""
    facts_mentioned: List[int] = Field(default_factory=list)  # Indices of facts mentioned
    lies_told: List[Tuple[str, str]] = Field(default_factory=list)  # (lie, truth)
    elaborations_made: List[Dict[str, str]] = Field(default_factory=list)
    defensive_level: float = 0.0  # Increases when pressed


class RedAgentState(StoreModel):
    """Private state for Red agent"""
    facts_explored: List[int] = Field(default_factory=list)
    suspicious_statements: List[str] = Field(default_factory=list)
    interrogation_strategy: str = "neutral"  # "neutral", "pressing", "friendly"
    known_lie_index: int = -1  # Which fact is the lie (Red knows this)
    questions_asked: List[str] = Field(default_factory=list)