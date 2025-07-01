# games/alibi/models.py

from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple

from inspect_ai.util import StoreModel
from pydantic import BaseModel, Field
from shortuuid import uuid


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
    name: str
    background: str
    relationship_to_incident: str  # How they're connected to the incident
    actual_whereabouts: List[AlibiElement]  # What they really did
    alibi_story: List[AlibiElement]  # What they claim they did
    personality_traits: List[str]
    motivation_to_lie: str


class DetectiveProfile(BaseModel):
    """Red agent's persona as the detective"""
    name: str
    rank: str
    background: str
    investigation_style: str  # "aggressive", "methodical", "friendly"
    personality_traits: List[str]


class GameConfig(BaseModel):
    """Configuration for an alibi game"""
    game_id: str = Field(default_factory=lambda: str(uuid()))
    incident: Incident
    suspect_profile: SuspectProfile
    detective_profile: DetectiveProfile
    interrogation_turns_limit: int = 25
    game_context: Dict[str, str] = Field(default_factory=dict)


class InterrogationTurn(BaseModel):
    """A single turn in the interrogation"""
    turn_number: int
    speaker: str  # "suspect" or "detective"
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    focuses_on_time: Optional[str] = None  # Which time period this relates to
    challenges_alibi_element: Optional[int] = None  # Index of alibi element being questioned


class InconsistencyRecord(BaseModel):
    """Record of an inconsistency detected in the alibi"""
    turn_number: int
    inconsistency_type: str  # "timeline", "location", "witness", "evidence"
    original_statement: str
    contradicting_statement: str
    severity: str  # "minor", "major", "critical"


class LieInstance(BaseModel):
    """Record of a lie told during interrogation"""
    turn_number: int
    utterance: str
    lie_type: str  # "alibi_fabrication", "evidence_denial", "timeline_shift", "witness_lie"
    relates_to_element: Optional[AlibiElement] = None
    detection_difficulty: float  # How hard this lie is to catch (0-1)


class GameState(StoreModel):
    """Shared game state"""
    config: Optional[GameConfig] = None
    interrogation: List[InterrogationTurn] = Field(default_factory=list)
    current_phase: str = "briefing"  # "briefing", "interrogation", "complete"
    active_speaker: str = "detective"
    game_complete: bool = False
    pressure_level: float = 0.0  # Increases as detective finds inconsistencies


class SuspectAgentState(StoreModel):
    """Private state for Suspect (Blue) agent"""
    alibi_elements_mentioned: List[int] = Field(default_factory=list)
    lies_told: List[Tuple[str, str]] = Field(default_factory=list)  # (lie, truth)
    inconsistencies_created: List[InconsistencyRecord] = Field(default_factory=list)
    stress_level: float = 0.0
    defensive_level: float = 0.0
    time_periods_questioned: List[str] = Field(default_factory=list)


class DetectiveAgentState(StoreModel):
    """Private state for Detective (Red) agent"""
    inconsistencies_noted: List[InconsistencyRecord] = Field(default_factory=list)
    suspicious_statements: List[str] = Field(default_factory=list)
    questioning_strategy: str = "chronological"  # "chronological", "random", "pressure"
    time_periods_explored: List[str] = Field(default_factory=list)
    evidence_collected: List[str] = Field(default_factory=list)
    confidence_in_guilt: float = 0.5  # 0-1 scale