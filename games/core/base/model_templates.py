# games/common/base/model_templates.py
"""
Template definitions for common model fields.
Since Tanuki requires flat class definitions, these serve as templates
for what fields should be included in game-specific models.
"""

from typing import Dict, List

# Template field definitions for reference
PERSONA_FIELDS = {
    "name": "str",
    "background": "str", 
    "personality_traits": "List[str]"
}

GAME_CONFIG_FIELDS = {
    "game_id": "str = Field(default_factory=lambda: str(uuid()))",
    "turns_limit": "int = 20",
    "game_context": "Dict[str, str] = Field(default_factory=dict)"
}

CONVERSATION_TURN_FIELDS = {
    "turn_number": "int",
    "speaker": "str",
    "message": "str",
    "timestamp": "datetime = Field(default_factory=datetime.now)"
}

GAME_STATE_FIELDS = {
    "conversation": "List[Dict] = Field(default_factory=list)",
    "current_phase": "str = 'starting'",
    "active_speaker": "str = 'red'",
    "game_complete": "bool = False",
    "turn_count": "int = 0"
}

BLUE_AGENT_STATE_FIELDS = {
    "revealed_info": "List[str] = Field(default_factory=list)",
    "lies_told": "List[Tuple[str, str]] = Field(default_factory=list)",
    "stress_level": "float = 0.0",
    "times_caught": "int = 0"
}

RED_AGENT_STATE_FIELDS = {
    "suspicious_statements": "List[str] = Field(default_factory=list)",
    "topics_explored": "List[str] = Field(default_factory=list)",
    "confidence_assessments": "List[Dict[str, Any]] = Field(default_factory=list)"
}

LIE_INSTANCE_FIELDS = {
    "turn_number": "int",
    "utterance": "str",
    "lie_type": "str",
    "confidence": "float = 0.5"
}


def create_model_docstring(model_type: str, game_name: str) -> str:
    """Generate a docstring for a model class"""
    return f"{model_type} for {game_name} game"