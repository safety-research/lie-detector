from enum import Enum


class MotivationFramework(Enum):
    """Generalization framework for task motivation categories"""
    INSTRUCTION_FOLLOWING = "instruction_following"
    IN_CONTEXT_PRESSURE = "in_context_pressure"
    INTERNALIZED_DRIVE_TO_LIE = "internalized_drive_to_lie"


class KnowledgeFramework(Enum):
    """Generalization framework for knowledge access categories"""
    IN_CONTEXT_KNOWLEDGE = "in_context_knowledge"
    INTERNAL_KNOWLEDGE = "internal_knowledge"
    INTROSPECTIVE_KNOWLEDGE = "introspective_knowledge"