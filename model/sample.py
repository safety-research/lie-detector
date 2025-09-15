"""
Pydantic models for standardized sample data across all lie detection pipelines.

This module defines the common data structure for samples saved to S3 and disk
by all hook implementations in the lie-detector project.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """A single message in a conversation trace."""
    role: str = Field(..., description="Role of the speaker (user, assistant, system)")
    content: str = Field(..., description="Content of the message")


class ScoreData(BaseModel):
    """Score information from a scorer."""
    value: Union[float, str] = Field(..., description="Score value (numeric or CORRECT/INCORRECT/NOANSWER)")
    explanation: Optional[str] = Field(None, description="Explanation for the score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional scorer-specific metadata")


class LieDetectionSample(BaseModel):
    """
    Standardized model for lie detection samples across all pipelines.
    
    This model represents the common structure used by all hook implementations
    when saving samples to S3 and disk.
    """
    
    # Core identification fields
    sample_id: str = Field(..., description="Unique identifier for the sample")
    task: str = Field(..., description="Name of the evaluation task")
    task_id: str = Field(..., description="Unique identifier for the task run")
    timestamp: str = Field(..., description="ISO timestamp of when the evaluation was created")
    model: str = Field(..., description="Model name/identifier used for the evaluation")
    
    # Conversation data
    trace: List[ConversationMessage] = Field(
        ..., 
        description="Full conversation trace including all messages"
    )
    
    # Lie detection specific fields
    lie_detection_prompt: ConversationMessage = Field(
        ...,
        description="The elicitation prompt to use to detect if the model lied"
    )
    did_lie: bool = Field(
        ...,
        description="Whether the model was determined to have lied"
    )
    
    # Evidence and explanations
    evidence: Optional[str] = Field(
        None,
        description="Evidence or explanation for the lie detection decision"
    )
    explanation: Optional[str] = Field(
        None,
        description="Alternative field for explanation (used by some hooks)"
    )
    
    # Scoring data
    scores: Dict[str, Union[ScoreData, Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Scores from various scorers with their metadata"
    )
    
    # Pipeline-specific metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline-specific metadata and additional context"
    )
    
    class Config:
        extra = "forbid"  # Strict validation - no extra fields allowed
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Pipeline-specific metadata models for better type safety

class IntrospectiveLieMetrics(BaseModel):
    """Metrics specific to introspective lie detection."""
    t_pred_neutral: Optional[float] = None
    t_pred_threat: Optional[float] = None
    actual_consistency_neutral: Optional[float] = None
    actual_consistency_threat: Optional[float] = None
    b_star: Optional[str] = None
    b_star_confidence: Optional[float] = None
    baseline_responses: List[str] = Field(default_factory=list)
    responses_neutral: List[str] = Field(default_factory=list)
    responses_threat: List[str] = Field(default_factory=list)
    is_introspective_lie: Optional[bool] = None
    lie_confidence: Optional[str] = None
    truthfulness_gap: Optional[float] = None
    condition: str = "unknown"


class SycophancyMetadata(BaseModel):
    """Metadata specific to sycophancy evaluation."""
    prompt_type: Optional[str] = None
    influenced_response: Optional[str] = None
    control_response: Optional[str] = None
    has_control: bool = False
    first_answer: Optional[str] = None
    second_answer: Optional[str] = None
    answer_changed: Optional[bool] = None
    change_type: Optional[str] = None
    system_prompt: Optional[str] = None


class GameLieData(BaseModel):
    """Data structure for lies detected in game scenarios."""
    utterance: str = Field(..., description="The lying utterance")
    explanation: str = Field(..., description="Explanation of why it's a lie")
    turn: Optional[int] = Field(None, description="Turn number when the lie occurred")


# Export all models
__all__ = [
    'ConversationMessage',
    'ScoreData',
    'LieDetectionSample',
    'IntrospectiveLieMetrics',
    'SycophancyMetadata',
    'GameLieData'
]