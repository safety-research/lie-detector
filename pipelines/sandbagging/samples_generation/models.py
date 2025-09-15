"""
Data models for sandbagging samples generation pipeline.
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
class SandbaggingSample:
    """A single sandbagging sample with control and harmful variants."""
    
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
    source: str = "grok_generated"
    
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
    def from_dict(cls, data: Dict[str, Any]) -> 'SandbaggingSample':
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
            source=data.get("source", "grok_generated")
        )


@dataclass
class GenerationConfig:
    """Configuration for sample generation."""
    
    # Generation parameters
    num_samples: int = 100
    samples_per_batch: int = 12
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # Model configuration
    model_name: str = "x-ai/grok-4"
    
    # Task metadata
    task_name: str = "sandbagging_samples_generation"
    description: str = "Generate new sandbagging samples using Grok-4"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "num_samples": self.num_samples,
            "samples_per_batch": self.samples_per_batch,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_name": self.model_name,
            "task_name": self.task_name,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationConfig':
        """Create from dictionary format."""
        return cls(
            num_samples=data.get("num_samples", 100),
            samples_per_batch=data.get("samples_per_batch", 12),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 1000),
            model_name=data.get("model_name", "x-ai/grok-4"),
            task_name=data.get("task_name", "sandbagging_samples_generation"),
            description=data.get("description", "Generate new sandbagging samples using Grok-4")
        )


@dataclass
class GenerationResult:
    """Result of sample generation."""
    
    generated_samples: List[SandbaggingSample]
    total_generated: int
    successful_generations: int
    failed_generations: int
    
    # Metadata
    generation_time_seconds: Optional[float] = None
    model_used: str = "x-ai/grok-4"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "generated_samples": [sample.to_dict() for sample in self.generated_samples],
            "total_generated": self.total_generated,
            "successful_generations": self.successful_generations,
            "failed_generations": self.failed_generations,
            "generation_time_seconds": self.generation_time_seconds,
            "model_used": self.model_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationResult':
        """Create from dictionary format."""
        return cls(
            generated_samples=[SandbaggingSample.from_dict(sample) for sample in data["generated_samples"]],
            total_generated=data["total_generated"],
            successful_generations=data["successful_generations"],
            failed_generations=data["failed_generations"],
            generation_time_seconds=data.get("generation_time_seconds"),
            model_used=data.get("model_used", "x-ai/grok-4")
        ) 