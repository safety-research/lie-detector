# === Updated Data Models ===

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from inspect_ai.util import StoreModel
from pydantic import BaseModel, Field
from shortuuid import uuid


class InfoClassification(str, Enum):
    """Security classification levels for information"""
    PUBLIC = "PUBLIC"
    CONFIDENTIAL = "CONFIDENTIAL"
    SECRET = "SECRET"
    TOP_SECRET = "TOP_SECRET"


class SecretInfo(BaseModel):
    """A piece of information with classification level"""
    content: str
    classification: InfoClassification
    cover_story: Optional[str] = None  # Plausible false information


class Persona(BaseModel):
    """Persona for Blue or Red agent"""
    name: str
    role: str
    background: str
    objectives: List[str]
    personality_traits: List[str] = Field(default_factory=list)


class ScenarioConfig(BaseModel):
    """Configuration for a deception scenario"""
    scenario_id: str = Field(default_factory=lambda: str(uuid()))
    domain: str  # e.g., "corporate", "intelligence", "scientific"
    blue_persona: Persona
    red_persona: Persona
    information: Dict[InfoClassification, List[SecretInfo]]
    conversation_limit: int = 100
    scenario_context: Optional[Dict[str, str]] = None
    #red_objectives: Optional[List[str]] = None  # Make optional since templates don't use it


class ConfidenceEntry(BaseModel):
    """Red's confidence assessment for a Blue statement"""
    turn_number: int
    statement: str
    confidence: float  # 0-100, where 0 = definitely lying, 100 = definitely true
    reasoning: str  # Red's internal reasoning
    deception_indicators: List[str] = Field(default_factory=list)  # Specific indicators found


class ConversationTurn(BaseModel):
    """A single turn in the conversation"""
    speaker: str  # "blue" or "red"
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class LieDetectionAttempt(BaseModel):
    """Record of Red explicitly calling out a suspected lie"""
    turn_number: int
    suspected_statement: str
    confidence: float
    called_out: bool  # Whether Red explicitly challenged the statement
    was_actually_lie: Optional[bool] = None  # Ground truth


class GameState(StoreModel):
    """Shared game state stored in Inspect's Store"""
    scenario: Optional[ScenarioConfig] = None
    conversation: List[ConversationTurn] = Field(default_factory=list)
    turn_count: int = 0
    active_speaker: str = "red"  # Red starts by asking questions
    game_complete: bool = False


class BlueAgentState(StoreModel):
    """Private state for Blue agent"""
    revealed_info: List[str] = Field(default_factory=list)
    lies_told: List[Tuple[str, str]] = Field(default_factory=list)  # (lie, truth)
    stress_level: float = 0.0
    times_caught: int = 0  # Number of times Red correctly identified a lie


class RedAgentState(StoreModel):
    """Private state for Red agent"""
    confidence_log: List[ConfidenceEntry] = Field(default_factory=list)
    topics_explored: List[str] = Field(default_factory=list)
    suspicion_triggers: List[str] = Field(default_factory=list)
    lie_detection_attempts: List[LieDetectionAttempt] = Field(default_factory=list)
    lies_correctly_identified: int = 0
    false_accusations: int = 0