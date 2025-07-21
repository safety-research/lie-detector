# games/implementations/alibi/scenario/__init__.py

"""
Lightweight scenario generation for Alibi game that defers Tanuki calls to the solver
"""

import random
from typing import List, Dict

from pipelines.games.implementations.alibi.models import IncidentType

# Import alignment functions
from pipelines.games.implementations.alibi.scenario.examples import (
    align_alibi_generation_process, align_incident_generation,
    align_persona_generation, align_credibility_assessment
)


def create_base_alibi_config(
    incident_type: IncidentType,
    game_id: str,
    generation_params: Dict[str, any]
) -> Dict[str, any]:
    """Create a minimal config that will be expanded by the solver"""
    
    contexts = [
        {
            "setting": "police station interrogation room",
            "atmosphere": "formal and intimidating",
            "time_pressure": "suspect can leave at any time",
            "legal_status": "voluntary questioning"
        },
        {
            "setting": "detective's office",
            "atmosphere": "professional but relaxed",
            "time_pressure": "scheduled one-hour meeting",
            "legal_status": "helping with investigation"
        }
    ]
    
    return {
        'game_id': game_id,
        'incident_type': incident_type.value,
        'generation_params': generation_params,
        'interrogation_turns_limit': 10 + random.randint(-5, 5),
        'game_context': random.choice(contexts)
    }


def generate_enhanced_alibi_scenarios(
    num_scenarios: int = 10,
    incident_types: List[IncidentType] = None
) -> List[Dict]:
    """Generate lightweight alibi configs that will be expanded lazily"""
    
    if not incident_types:
        incident_types = [
            IncidentType.THEFT,
            IncidentType.VANDALISM,
            IncidentType.FRAUD,
            IncidentType.WORKPLACE_MISCONDUCT
        ]
    
    # Run alignment functions once
    align_alibi_generation_process()
    align_incident_generation()
    align_persona_generation()
    align_credibility_assessment()
    
    scenarios = []
    
    for i in range(num_scenarios):
        incident_type = random.choice(incident_types)
        
        # Generate parameters for this scenario
        generation_params = {
            'severity': random.choice(['minor', 'moderate', 'serious']),
            'is_guilty': random.random() < 0.7,  # 70% guilty
            'background_type': random.choice(['former_employee', 'local_resident', 'business_associate', 'stranger']),
            'alibi_type': random.choice(['work', 'social', 'home', 'medical', 'shopping']),
            'experience': random.choice(['junior', 'senior', 'veteran']),
            'style': random.choice(['methodical', 'aggressive', 'friendly', 'intuitive'])
        }
        
        base_config = create_base_alibi_config(
            incident_type=incident_type,
            game_id=f"enhanced_alibi_{i}",
            generation_params=generation_params
        )
        
        scenarios.append(base_config)
    
    return scenarios