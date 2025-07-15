"""
Data models for sandbagging detection pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class SandbaggingDomain(Enum):
    """Categories of domains for sandbagging detection."""
    PHYSICAL_SECURITY = "physical_security"
    DIGITAL_SECURITY = "digital_security"
    CHEMISTRY_BIOLOGY = "chemistry_biology"
    AUTOMOTIVE_MECHANICAL = "automotive_mechanical"
    MEDICAL_PHARMACEUTICAL = "medical_pharmaceutical"
    FINANCIAL_SYSTEMS = "financial_systems"



@dataclass
class SandbaggingPromptPair:
    """A pair of control and sandbagging prompts for the same technical question."""
    
    # Core question components
    domain: SandbaggingDomain
    question_text: str
    correct_answer: str
    incorrect_answer: str
    
    # Context components
    expert_context: str  # For control variant
    harmful_context: str  # For sandbagging variant
    
    # Metadata
    question_id: str
    difficulty_level: str = "medium"
    harm_severity: str = "moderate"
    source: str = "generated"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "domain": self.domain.value,
            "question_text": self.question_text,
            "correct_answer": self.correct_answer,
            "incorrect_answer": self.incorrect_answer,
            "expert_context": self.expert_context,
            "harmful_context": self.harmful_context,
            "question_id": self.question_id,
            "difficulty_level": self.difficulty_level,
            "harm_severity": self.harm_severity,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SandbaggingPromptPair':
        """Create from dictionary format."""
        return cls(
            domain=SandbaggingDomain(data["domain"]),
            question_text=data["question_text"],
            correct_answer=data["correct_answer"],
            incorrect_answer=data["incorrect_answer"],
            expert_context=data["expert_context"],
            harmful_context=data["harmful_context"],
            question_id=data["question_id"],
            difficulty_level=data.get("difficulty_level", "medium"),
            harm_severity=data.get("harm_severity", "moderate"),
            source=data.get("source", "generated")
        )


@dataclass
class SandbaggingConfig:
    """Configuration for sandbagging detection evaluation."""
    
    # Prompt configuration
    prompt_pairs: List[SandbaggingPromptPair]
    
    # Evaluation parameters
    runs_per_prompt: int = 20
    temperature: float = 0.0
    max_tokens: int = 10
    
    # Analysis parameters
    confidence_level: float = 0.95
    min_sample_size: int = 20
    
    # Task metadata
    task_name: str = "sandbagging_detection"
    description: str = "Detect model sandbagging behavior in harmful contexts"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "prompt_pairs": [pair.to_dict() for pair in self.prompt_pairs],
            "runs_per_prompt": self.runs_per_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "confidence_level": self.confidence_level,
            "min_sample_size": self.min_sample_size,
            "task_name": self.task_name,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SandbaggingConfig':
        """Create from dictionary format."""
        return cls(
            prompt_pairs=[SandbaggingPromptPair.from_dict(pair) for pair in data["prompt_pairs"]],
            runs_per_prompt=data.get("runs_per_prompt", 20),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 10),
            confidence_level=data.get("confidence_level", 0.95),
            min_sample_size=data.get("min_sample_size", 20),
            task_name=data.get("task_name", "sandbagging_detection"),
            description=data.get("description", "Detect model sandbagging behavior in harmful contexts")
        )


@dataclass
class SandbaggingResult:
    """Result of sandbagging detection evaluation."""
    
    question_id: str
    domain: SandbaggingDomain
    
    # Response data
    model_response: str
    extracted_answer: str
    is_correct: bool
    
    # Timing and metadata
    response_time_ms: Optional[float] = None
    refused_to_answer: bool = False
    
    # Context
    prompt_text: str = ""
    correct_answer: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "question_id": self.question_id,
            "domain": self.domain.value,
            "model_response": self.model_response,
            "extracted_answer": self.extracted_answer,
            "is_correct": self.is_correct,
            "response_time_ms": self.response_time_ms,
            "refused_to_answer": self.refused_to_answer,
            "prompt_text": self.prompt_text,
            "correct_answer": self.correct_answer
        }