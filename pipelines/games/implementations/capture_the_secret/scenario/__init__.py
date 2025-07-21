# games/implementations/capture_the_secret/scenario/__init__.py

"""
Lightweight scenario generation that defers Tanuki calls to the solver
"""

import random
from typing import List, Dict

from pipelines.games.implementations.capture_the_secret.models import ScenarioConfig

# Import alignment functions (these can still run upfront as they're one-time)
from pipelines.games.implementations.capture_the_secret.scenario.examples import (
    align_domain_context, align_persona_generation, align_information_generation,
    align_meeting_context, align_risk_assessment, align_elicitation_strategy
)


def create_base_scenario_config(
    domain: str,
    game_id: str,
    generation_params: Dict[str, str]
) -> Dict[str, any]:
    """Create a minimal config that will be expanded by the solver"""
    return {
        'game_id': game_id,
        'domain': domain,
        'generation_params': generation_params,
        'conversation_limit': 25 + random.randint(-5, 10)
    }


def generate_enhanced_capture_scenarios(num_scenarios: int = 10, domains: List[str] = None) -> List[Dict]:
    """Generate lightweight scenario configs that will be expanded lazily"""
    
    if not domains:
        domains = ['corporate', 'intelligence', 'scientific', 'relationship', 'sales']
    
    # Run alignment functions once
    align_domain_context()
    align_persona_generation()
    align_information_generation()
    align_meeting_context()
    align_risk_assessment()
    align_elicitation_strategy()
    
    scenarios = []
    
    for i in range(num_scenarios):
        domain = random.choice(domains)
        
        # Generate parameters for this scenario
        generation_params = {
            'seniority': random.choice(['junior', 'mid', 'senior', 'executive']),
            'access_level': random.choice(['limited', 'moderate', 'extensive']),
            'approach_style': random.choice(['investigative', 'casual', 'aggressive', 'friendly']),
            'insider_knowledge': random.choice(['none', 'some', 'significant']),
            'formality': random.choice(['formal', 'semi_formal', 'casual']),
            'risk_level': random.choice(['low', 'medium', 'high'])
        }
        
        # Create lightweight config
        base_config = create_base_scenario_config(
            domain=domain,
            game_id=f"enhanced_{domain}_{i}",
            generation_params=generation_params
        )
        
        scenarios.append(base_config)
    
    return scenarios


def generate_domain_specific_scenarios(domain: str, num_scenarios: int) -> List[Dict]:
    """Generate scenarios for a specific domain"""
    
    # Run alignment functions once
    align_domain_context()
    align_persona_generation()
    align_information_generation()
    align_meeting_context()
    
    scenarios = []
    
    for i in range(num_scenarios):
        generation_params = {
            'seniority': random.choice(['junior', 'mid', 'senior', 'executive']),
            'access_level': random.choice(['limited', 'moderate', 'extensive']),
            'approach_style': random.choice(['investigative', 'casual', 'aggressive', 'friendly']),
            'insider_knowledge': random.choice(['none', 'some', 'significant']),
            'formality': random.choice(['formal', 'semi_formal', 'casual']),
            'risk_level': random.choice(['low', 'medium', 'high'])
        }
        
        base_config = create_base_scenario_config(
            domain=domain,
            game_id=f"{domain}_scenario_{i}",
            generation_params=generation_params
        )
        
        scenarios.append(base_config)
    
    return scenarios