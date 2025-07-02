# games/common/scenario/base_generator.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, Optional
import tanuki

from ..base.models import BasePersonaProfile, BaseGameConfig


class BaseScenarioGenerator(ABC):
    """Base class for scenario generation across all games"""
    
    @abstractmethod
    def get_config_class(self) -> Type[BaseGameConfig]:
        """Return the config class this generator creates"""
        pass
    
    @abstractmethod
    def generate_scenario_config(self, **kwargs) -> BaseGameConfig:
        """Generate a single scenario configuration"""
        pass
    
    def generate_multiple_scenarios(self, num_scenarios: int, **kwargs) -> List[BaseGameConfig]:
        """Generate multiple scenario configurations"""
        scenarios = []
        for _ in range(num_scenarios):
            scenario = self.generate_scenario_config(**kwargs)
            scenarios.append(scenario)
        return scenarios


class PersonaGenerator:
    """Common utilities for generating personas across games"""
    
    @staticmethod
    def generate_persona_fields(
        domain: str,
        role_type: str,  # "investigator", "suspect", "information_holder", etc.
        personality_style: str = "balanced"  # "aggressive", "passive", "balanced"
    ) -> Dict[str, Any]:
        """Generate persona field values that can be used to construct game-specific personas
        
        Returns a dict with standard persona fields that games can use.
        Since Tanuki requires flat classes, games must define their own persona classes
        but can use these generated values.
        """
        # This would typically call a Tanuki-decorated function specific to each game
        return {
            "name": "",
            "background": "",
            "personality_traits": []
        }
    
    @staticmethod
    @tanuki.patch  
    def generate_personality_traits(
        base_personality: str,
        role_requirements: List[str],
        num_traits: int = 3
    ) -> List[str]:
        """Generate personality traits for a persona"""
        pass
    
    @staticmethod
    @tanuki.patch
    def generate_background_story(
        role: str,
        domain: str,
        complexity_level: str = "medium"
    ) -> str:
        """Generate a background story for a persona"""
        pass


class DomainContextGenerator:
    """Common utilities for generating domain-specific contexts"""
    
    @staticmethod
    @tanuki.patch
    def generate_setting_context(
        domain: str,  # "corporate", "academic", "social", "investigative"
        formality_level: str = "medium",  # "formal", "medium", "casual"
        time_pressure: str = "medium"  # "high", "medium", "low"
    ) -> Dict[str, str]:
        """Generate context about the setting and atmosphere"""
        pass
    
    @staticmethod
    @tanuki.patch
    def generate_relationship_context(
        persona1_role: str,
        persona2_role: str,
        domain: str,
        relationship_type: str = "professional"  # "professional", "personal", "adversarial"
    ) -> Dict[str, str]:
        """Generate context about the relationship between personas"""
        pass


class ConversationContextGenerator:
    """Utilities for generating conversation contexts"""
    
    @staticmethod
    @tanuki.patch
    def generate_opening_context(
        initiator_role: str,
        responder_role: str,
        conversation_purpose: str,
        setting: Dict[str, str]
    ) -> str:
        """Generate context for how a conversation should begin"""
        pass
    
    @staticmethod
    @tanuki.patch
    def generate_conversation_constraints(
        domain: str,
        privacy_level: str = "medium",  # "public", "medium", "private"
        time_limit: str = "medium"  # "short", "medium", "long"
    ) -> Dict[str, Any]:
        """Generate constraints that should affect the conversation"""
        pass