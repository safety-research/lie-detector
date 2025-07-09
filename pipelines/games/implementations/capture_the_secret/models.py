# === Updated Data Models ===

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from pipelines.games.core.base.models import (
    BaseGameConfig, BaseGameState,
    BaseConversationTurn, BaseBlueAgentState, BaseRedAgentState,
    BaseLieInstance
)


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


class DomainContext(BaseModel):
    """Generated domain-specific context"""
    organization_name: str
    industry_background: str
    current_events: List[str]
    common_terminology: List[str]
    insider_knowledge: List[str]


class PersonaProfile(BaseModel):
    """Generated persona profile"""
    name: str
    role: str
    background: str
    personality_traits: List[str]
    communication_style: Optional[str] = None
    motivations: List[str]


class SecretInformation(BaseModel):
    """Generated classified information"""
    content: str
    classification_level: str
    cover_story: Optional[str] = None
    sensitivity_explanation: str


class MeetingScenario(BaseModel):
    """Generated meeting context"""
    location: str
    connection_method: str
    timing_context: str
    backstory: str
    conversation_motivation: str
    time_pressure: str
    atmosphere: str


class Persona(BaseModel):
    """Persona for Blue or Red agent"""
    role: str
    objectives: List[str]
    name: str
    background: str
    personality_traits: List[str] = Field(default_factory=list)


class ScenarioConfig(BaseGameConfig):
    """Configuration for a deception scenario"""
    domain: str  # e.g., "corporate", "intelligence", "scientific"
    blue_persona: Persona
    red_persona: Persona
    information: Dict[InfoClassification, List[SecretInfo]]
    conversation_limit: int = 10
    turns_limit: int = 10  # Also set the base class field
    scenario_context: Optional[Dict[str, str]] = None
    
    @property
    def scenario_id(self) -> str:
        """Alias for game_id for backward compatibility"""
        return self.game_id

    def set_scenario_id(self, scenario_id: str):
        self.game_id = scenario_id


class ConfidenceEntry(BaseModel):
    """Red's confidence assessment for a Blue statement"""
    turn_number: int
    statement: str
    confidence: float  # 0-100, where 0 = definitely lying, 100 = definitely true
    reasoning: str  # Red's internal reasoning
    deception_indicators: List[str] = Field(default_factory=list)  # Specific indicators found


class ConversationTurn(BaseConversationTurn):
    """A single turn in the conversation"""
    pass  # Inherits all fields from BaseConversationTurn


class LieDetectionAttempt(BaseLieInstance):
    """Record of Red explicitly calling out a suspected lie"""
    suspected_statement: str
    called_out: bool  # Whether Red explicitly challenged the statement
    was_actually_lie: Optional[bool] = None  # Ground truth


class GameState(BaseGameState):
    """Shared game state stored in Inspect's Store"""
    scenario: Optional[ScenarioConfig] = None
    active_speaker: str = "red"  # Red starts by asking questions


class BlueAgentState(BaseBlueAgentState):
    """Private state for Blue agent"""
    pass  # Inherits all fields from BaseBlueAgentState


class RedAgentState(BaseRedAgentState):
    """Private state for Red agent"""
    confidence_log: List[ConfidenceEntry] = Field(default_factory=list)
    suspicion_triggers: List[str] = Field(default_factory=list)
    lie_detection_attempts: List[LieDetectionAttempt] = Field(default_factory=list)
    lies_correctly_identified: int = 0
    false_accusations: int = 0