# games/common/scenario/persona_utils.py

from typing import List, Dict, Any, Optional
from enum import Enum


class PersonalityArchetype(str, Enum):
    """Common personality archetypes across games"""
    ANALYTICAL = "analytical"
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    CHARMING = "charming"
    NERVOUS = "nervous"
    CONFIDENT = "confident"
    METHODICAL = "methodical"
    INTUITIVE = "intuitive"


class CommunicationStyle(str, Enum):
    """Common communication styles"""
    FORMAL = "formal"
    CASUAL = "casual"
    VERBOSE = "verbose"
    CONCISE = "concise"
    EVASIVE = "evasive"
    DIRECT = "direct"


class PersonaUtils:
    """Utility functions for working with personas"""
    
    # Common personality trait sets
    INVESTIGATOR_TRAITS = [
        "methodical", "persistent", "observant", "analytical", "patient",
        "intuitive", "thorough", "skeptical", "empathetic", "assertive"
    ]
    
    DECEPTIVE_TRAITS = [
        "charming", "quick-thinking", "adaptable", "confident", "creative",
        "persuasive", "calm under pressure", "detail-oriented", "empathetic", "strategic"
    ]
    
    NERVOUS_TRAITS = [
        "anxious", "fidgety", "talkative when nervous", "defensive", "hesitant",
        "overthinking", "self-doubting", "rambling", "apologetic", "evasive"
    ]
    
    CONFIDENT_TRAITS = [
        "assertive", "self-assured", "articulate", "composed", "charismatic",
        "decisive", "outgoing", "optimistic", "ambitious", "persuasive"
    ]
    
    @staticmethod
    def get_traits_for_archetype(archetype: PersonalityArchetype, count: int = 3) -> List[str]:
        """Get personality traits for a given archetype"""
        trait_mapping = {
            PersonalityArchetype.ANALYTICAL: PersonaUtils.INVESTIGATOR_TRAITS,
            PersonalityArchetype.AGGRESSIVE: ["assertive", "confrontational", "impatient", "direct", "persistent"],
            PersonalityArchetype.DEFENSIVE: ["cautious", "guarded", "suspicious", "protective", "reactive"],
            PersonalityArchetype.CHARMING: ["charismatic", "friendly", "persuasive", "engaging", "empathetic"],
            PersonalityArchetype.NERVOUS: PersonaUtils.NERVOUS_TRAITS,
            PersonalityArchetype.CONFIDENT: PersonaUtils.CONFIDENT_TRAITS,
            PersonalityArchetype.METHODICAL: ["systematic", "thorough", "patient", "organized", "careful"],
            PersonalityArchetype.INTUITIVE: ["perceptive", "instinctive", "empathetic", "creative", "spontaneous"]
        }
        
        traits = trait_mapping.get(archetype, PersonaUtils.CONFIDENT_TRAITS)
        return traits[:count]
    
    @staticmethod
    def get_communication_style_description(style: CommunicationStyle) -> str:
        """Get a description of how someone with this style communicates"""
        descriptions = {
            CommunicationStyle.FORMAL: "Uses proper grammar, avoids contractions, maintains professional tone",
            CommunicationStyle.CASUAL: "Uses informal language, contractions, and colloquialisms",
            CommunicationStyle.VERBOSE: "Provides extensive detail, uses long sentences, explains thoroughly",
            CommunicationStyle.CONCISE: "Gets straight to the point, uses short sentences, minimal elaboration",
            CommunicationStyle.EVASIVE: "Avoids direct answers, uses vague language, redirects questions",
            CommunicationStyle.DIRECT: "Speaks plainly, answers questions directly, minimal hedging"
        }
        return descriptions.get(style, "Standard communication approach")
    
    @staticmethod
    def generate_compatible_traits(primary_archetype: PersonalityArchetype, 
                                 secondary_archetype: Optional[PersonalityArchetype] = None,
                                 count: int = 3) -> List[str]:
        """Generate a mix of traits from compatible archetypes"""
        primary_traits = PersonaUtils.get_traits_for_archetype(primary_archetype, count // 2 + 1)
        
        if secondary_archetype:
            secondary_traits = PersonaUtils.get_traits_for_archetype(secondary_archetype, count // 2)
            combined = primary_traits + secondary_traits
            return combined[:count]
        
        return primary_traits[:count]
    
    @staticmethod
    def get_archetype_for_role(role: str) -> PersonalityArchetype:
        """Suggest an appropriate archetype for a given role"""
        role_mapping = {
            "detective": PersonalityArchetype.ANALYTICAL,
            "investigator": PersonalityArchetype.METHODICAL,
            "suspect": PersonalityArchetype.DEFENSIVE,
            "interrogator": PersonalityArchetype.AGGRESSIVE,
            "information_holder": PersonalityArchetype.CHARMING,
            "spy": PersonalityArchetype.CONFIDENT,
            "witness": PersonalityArchetype.NERVOUS
        }
        
        return role_mapping.get(role.lower(), PersonalityArchetype.CONFIDENT)
    
    @staticmethod
    def validate_persona_compatibility(traits: List[str], communication_style: CommunicationStyle) -> bool:
        """Check if personality traits are compatible with communication style"""
        # Define incompatible combinations
        incompatible = {
            CommunicationStyle.EVASIVE: ["direct", "assertive", "straightforward"],
            CommunicationStyle.DIRECT: ["evasive", "mysterious", "vague"],
            CommunicationStyle.FORMAL: ["casual", "relaxed", "informal"],
            CommunicationStyle.CASUAL: ["formal", "rigid", "proper"]
        }
        
        conflicting_traits = incompatible.get(communication_style, [])
        return not any(trait.lower() in [ct.lower() for ct in conflicting_traits] for trait in traits)