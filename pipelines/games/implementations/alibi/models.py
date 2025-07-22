# games/alibi/models.py

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, Field

from pipelines.games.core.base.models import (
    BaseGameConfig, BaseGameState,
    BaseConversationTurn, BaseBlueAgentState, BaseRedAgentState,
    BaseLieInstance
)


class IncidentDetails(BaseModel):
    """Generated incident details"""
    location: str
    description: str
    evidence_against_suspect: List[str]
    investigation_details: Dict[str, str]


class AlibiStory(BaseModel):
    """Generated alibi story"""
    location: str
    activities: List[str]
    witnesses: List[str]
    evidence: List[str]
    alibi_type: str#Literal['work', 'social', 'home', 'medical', 'shopping', 'travel']


class PersonaDetails(BaseModel):
    """Generated persona characteristics"""
    name: str
    background: str
    personality_traits: List[str]
    motivation: str


class TimeFrame(BaseModel):
    """Specific time period for the incident"""
    start_time: Optional[str] = None  # e.g., "2:30 PM"
    end_time: Optional[str] = None    # e.g., "4:15 PM"
    date: Optional[str] = None        # e.g., "Tuesday, March 15th"
    duration_minutes: Optional[str] = None


class AlibiElement(BaseModel):
    """A component of the suspect's alibi story"""
    time_period: str  # e.g., "2:00-3:30 PM"
    location: str
    activity: str
    witness: Optional[str] = None
    verifiable_evidence: Optional[str] = None  # receipts, timestamps, etc.
    is_truthful: bool
    cover_story: Optional[str] = None  # Alternative explanation if questioned


class AlibiType(str, Enum):
    """Types of alibis the suspect can claim"""
    WORK = "WORK"
    SOCIAL = "SOCIAL"
    HOME = "HOME"
    TRAVEL = "TRAVEL"
    MEDICAL = "MEDICAL"
    SHOPPING = "SHOPPING"


class IncidentType(str, Enum):
    """Types of incidents requiring an alibi"""
    THEFT = "THEFT"
    VANDALISM = "VANDALISM"
    FRAUD = "FRAUD"
    ASSAULT = "ASSAULT"
    BREAKING_ENTERING = "BREAKING_ENTERING"
    WORKPLACE_MISCONDUCT = "WORKPLACE_MISCONDUCT"

class Incident(BaseModel):
    """The incident that requires an alibi"""
    incident_type: IncidentType
    location: str
    description: str
    time_frame: TimeFrame
    evidence_against_suspect: List[str]  # What makes them a suspect
    investigation_details: Dict[str, str]


class SuspectProfile(BaseModel):
    """Blue agent's persona as the suspect"""
    relationship_to_incident: str  # How they're connected to the incident
    actual_whereabouts: List[AlibiElement]  # What they really did
    alibi_story: List[AlibiElement]  # What they claim they did
    motivation_to_lie: str
    name: str
    background: str
    personality_traits: List[str]


class DetectiveProfile(BaseModel):
    """Red agent's persona as the detective"""
    rank: str
    investigation_style: str  # "aggressive", "methodical", "friendly"
    name: str
    background: str
    personality_traits: List[str]


class GameConfig(BaseGameConfig):
    """Configuration for an alibi game"""
    incident: Incident
    suspect_profile: SuspectProfile
    detective_profile: DetectiveProfile
    interrogation_turns_limit: int = 10
    turns_limit: int = 10  # Also set the base class field


class InterrogationTurn(BaseConversationTurn):
    """A single turn in the interrogation"""
    focuses_on_time: Optional[str] = None  # Which time period this relates to
    challenges_alibi_element: Optional[int] = None  # Index of alibi element being questioned


class InconsistencyRecord(BaseModel):
    """Record of an inconsistency detected in the alibi"""
    turn_number: int
    inconsistency_type: str  # "timeline", "location", "witness", "evidence"
    original_statement: str
    contradicting_statement: str
    severity: str  # "minor", "major", "critical"


class LieInstance(BaseLieInstance):
    """Record of a lie told during interrogation"""
    relates_to_element: Optional[AlibiElement] = None
    detection_difficulty: float = 0.5  # How hard this lie is to catch (0-1)


class GameState(BaseGameState):
    """Shared game state for alibi game"""
    interrogation: List[InterrogationTurn] = Field(default_factory=list)
    pressure_level: float = 0.0  # Increases as detective finds inconsistencies
    current_phase: str = "briefing"  # Override default
    active_speaker: str = "detective"  # Override default


class SuspectAgentState(BaseBlueAgentState):
    """Private state for Suspect (Blue) agent"""
    alibi_elements_mentioned: List[int] = Field(default_factory=list)
    inconsistencies_created: List[InconsistencyRecord] = Field(default_factory=list)
    defensive_level: float = 0.0
    time_periods_questioned: List[str] = Field(default_factory=list)


class DetectiveAgentState(BaseRedAgentState):
    """Private state for Detective (Red) agent"""
    inconsistencies_noted: List[InconsistencyRecord] = Field(default_factory=list)
    questioning_strategy: str = "chronological"  # "chronological", "random", "pressure"
    time_periods_explored: List[str] = Field(default_factory=list)
    evidence_collected: List[str] = Field(default_factory=list)
    confidence_in_guilt: float = 0.5  # 0-1 scale