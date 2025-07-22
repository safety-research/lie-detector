# games/implementations/alibi/scenario/__init__.py

"""
Enhanced scenario generation for Alibi game with rich combinations
"""

import random
from typing import List, Dict, Optional
from itertools import product

from pipelines.games.implementations.alibi.models import IncidentType, GameConfig

# Import alignment functions
from pipelines.games.implementations.alibi.scenario.examples import (
    align_alibi_generation_process, align_incident_generation,
    align_persona_generation
)

# Expanded incident types with subtypes
INCIDENT_VARIATIONS = {
    IncidentType.THEFT: {
        'subtypes': ['corporate_espionage', 'petty_theft', 'identity_theft', 'intellectual_property',
                     'inventory_theft'],
        'locations': ['office', 'warehouse', 'retail_store', 'parking_garage', 'conference_room', 'server_room'],
        'times': ['early_morning', 'business_hours', 'lunch_break', 'after_hours', 'weekend', 'holiday'],
        'values': ['high_value', 'moderate_value', 'low_value', 'sentimental_value', 'strategic_value']
    },
    IncidentType.VANDALISM: {
        'subtypes': ['property_damage', 'graffiti', 'sabotage', 'digital_vandalism', 'vehicle_damage'],
        'locations': ['parking_lot', 'building_exterior', 'office_space', 'public_area', 'equipment_room'],
        'times': ['late_night', 'early_morning', 'business_hours', 'weekend'],
        'values': ['extensive_damage', 'moderate_damage', 'minor_damage', 'symbolic_damage']
    },
    IncidentType.FRAUD: {
        'subtypes': ['financial_fraud', 'insurance_fraud', 'identity_fraud', 'document_forgery', 'expense_fraud'],
        'locations': ['bank', 'insurance_office', 'corporate_office', 'online', 'government_building'],
        'times': ['business_hours', 'online_anytime', 'scheduled_appointment', 'drop_off'],
        'values': ['large_scale', 'moderate_scale', 'small_scale', 'attempted_only']
    },
    IncidentType.WORKPLACE_MISCONDUCT: {
        'subtypes': ['harassment', 'data_breach', 'policy_violation', 'conflict_of_interest', 'time_theft'],
        'locations': ['private_office', 'open_workspace', 'break_room', 'conference_room', 'remote_work'],
        'times': ['regular_hours', 'overtime', 'company_event', 'business_trip'],
        'values': ['severe_violation', 'moderate_violation', 'minor_violation', 'pattern_of_behavior']
    },
    IncidentType.BREAKING_ENTERING: {
        'subtypes': ['forced_entry', 'unauthorized_access', 'trespassing', 'burglary', 'attempted_entry'],
        'locations': ['main_entrance', 'emergency_exit', 'window', 'rooftop', 'loading_dock', 'restricted_area'],
        'times': ['late_night', 'early_morning', 'weekend', 'holiday', 'during_closure', 'shift_change'],
        'values': ['successful_entry', 'attempted_entry', 'with_theft', 'with_vandalism', 'no_loss']
    },
    IncidentType.ASSAULT: {
        'subtypes': ['physical_assault', 'verbal_assault', 'threatening_behavior', 'altercation', 'intimidation'],
        'locations': ['parking_lot', 'lobby', 'workplace', 'elevator', 'stairwell', 'cafeteria'],
        'times': ['arrival_time', 'departure_time', 'business_hours', 'break_time', 'after_hours'],
        'values': ['severe_injury', 'minor_injury', 'no_injury', 'threats_only', 'property_damage']
    }
}

# Expanded context variations
INTERROGATION_CONTEXTS = [
    {
        "setting": "police station interrogation room",
        "atmosphere": "formal and intimidating",
        "time_pressure": "suspect can leave at any time",
        "legal_status": "voluntary questioning",
        "recording": "video and audio recorded"
    },
    {
        "setting": "detective's office",
        "atmosphere": "professional but relaxed",
        "time_pressure": "scheduled one-hour meeting",
        "legal_status": "helping with investigation",
        "recording": "audio only"
    },
    {
        "setting": "neutral location cafe",
        "atmosphere": "casual and non-threatening",
        "time_pressure": "informal chat",
        "legal_status": "witness interview",
        "recording": "detective's notes only"
    },
    {
        "setting": "suspect's workplace",
        "atmosphere": "semi-formal, aware of colleagues",
        "time_pressure": "during work hours",
        "legal_status": "workplace investigation",
        "recording": "with HR present"
    },
    {
        "setting": "video conference call",
        "atmosphere": "distant but focused",
        "time_pressure": "scheduled call",
        "legal_status": "remote questioning",
        "recording": "screen recorded"
    }
]

# Expanded generation parameters
SUSPECT_BACKGROUNDS = [
    'former_employee', 'current_employee', 'contractor', 'client', 'vendor',
    'local_resident', 'visitor', 'business_partner', 'competitor', 'stranger',
    'family_connection', 'social_connection', 'service_provider'
]

ALIBI_TYPES = [
    'work', 'social', 'home', 'medical', 'shopping', 'travel', 'exercise',
    'family_obligation', 'volunteer_work', 'education', 'entertainment',
    'personal_errand', 'religious_activity', 'hobby_activity'
]

DETECTIVE_EXPERIENCE = ['rookie', 'junior', 'experienced', 'senior', 'veteran', 'specialized']

DETECTIVE_STYLES = [
    'methodical', 'aggressive', 'friendly', 'intuitive', 'analytical',
    'empathetic', 'confrontational', 'patient', 'rapid-fire', 'psychological'
]

SUSPECT_PERSONALITIES = [
    'nervous', 'confident', 'defensive', 'cooperative', 'evasive',
    'talkative', 'quiet', 'emotional', 'analytical', 'hostile',
    'charming', 'confused', 'indignant', 'resigned'
]

COMPLICATING_FACTORS = [
    'previous_record', 'no_record', 'influential_connections', 'media_attention',
    'multiple_suspects', 'circumstantial_evidence', 'conflicting_witnesses',
    'time_sensitive', 'internal_politics', 'personal_connection'
]

EVIDENCE_STRENGTH = ['weak', 'moderate', 'strong', 'overwhelming', 'conflicting']

SUSPECT_STRATEGIES = [
    'full_cooperation', 'selective_truth', 'complete_denial', 'redirect_blame',
    'emotional_appeal', 'legal_technicalities', 'memory_issues', 'alternative_theory'
]


def generate_contextual_variations(base_params: Dict[str, any]) -> List[Dict[str, any]]:
    """Generate variations of parameters based on logical connections"""
    variations = []

    # Create logical connections between parameters
    if base_params['is_guilty']:
        # Guilty suspects might have different strategies
        strategies = ['selective_truth', 'complete_denial', 'redirect_blame', 'memory_issues']
        personalities = ['nervous', 'defensive', 'evasive', 'hostile']
    else:
        # Innocent suspects might behave differently
        strategies = ['full_cooperation', 'emotional_appeal', 'alternative_theory']
        personalities = ['confident', 'cooperative', 'indignant', 'confused']

    # Time-based alibi correlations
    time_alibi_map = {
        'early_morning': ['exercise', 'travel', 'work'],
        'business_hours': ['work', 'medical', 'shopping', 'personal_errand'],
        'lunch_break': ['social', 'shopping', 'personal_errand'],
        'after_hours': ['social', 'family_obligation', 'entertainment', 'exercise'],
        'weekend': ['social', 'family_obligation', 'hobby_activity', 'religious_activity'],
        'late_night': ['home', 'entertainment', 'work']
    }

    # Generate 2-3 variations with logical consistency
    for i in range(random.randint(2, 3)):
        variation = base_params.copy()
        variation['suspect_strategy'] = random.choice(strategies)
        variation['suspect_personality'] = random.choice(personalities)

        # Adjust alibi based on time if available
        if 'incident_time' in variation and variation['incident_time'] in time_alibi_map:
            variation['alibi_type'] = random.choice(time_alibi_map[variation['incident_time']])

        variations.append(variation)

    return variations


def create_rich_alibi_config(
        incident_type: IncidentType,
        game_id: str,
        generation_params: Dict[str, any]
) -> Dict[str, any]:
    """Create a rich config with many interconnected parameters"""

    # Select incident-specific variations
    incident_vars = INCIDENT_VARIATIONS[incident_type]

    # Build comprehensive config
    config = {
        'game_id': game_id,
        'incident_type': incident_type.value,
        'incident_subtype': random.choice(incident_vars['subtypes']),
        'incident_location': random.choice(incident_vars['locations']),
        'incident_time': random.choice(incident_vars['times']),
        'incident_value': random.choice(incident_vars['values']),
        'generation_params': generation_params,
        'interrogation_turns_limit': random.randint(8, 20),
        'game_context': random.choice(INTERROGATION_CONTEXTS),
        'evidence_strength': random.choice(EVIDENCE_STRENGTH),
        'complicating_factor': random.choice(COMPLICATING_FACTORS),
        'suspect_strategy': generation_params.get('suspect_strategy', random.choice(SUSPECT_STRATEGIES)),
        'time_elapsed_since_incident': random.choice(['hours', 'days', 'weeks', 'months']),
        'suspect_preparation': random.choice(['unprepared', 'somewhat_prepared', 'well_prepared']),
        'detective_knowledge': random.choice(['minimal', 'moderate', 'extensive']),
        'stakes': random.choice(['low', 'moderate', 'high', 'career_ending'])
    }

    return config


def generate_scenario_batch(
        batch_size: int = 5,
        incident_type: Optional[IncidentType] = None,
        theme: Optional[str] = None
) -> List[Dict[str, any]]:
    """Generate a batch of related scenarios with variations"""

    scenarios = []

    # If no incident type specified, choose one
    if not incident_type:
        incident_type = random.choice(list(IncidentType))

    # Base parameters for the batch
    base_severity = random.choice(['minor', 'moderate', 'serious'])
    base_location_type = random.choice(INCIDENT_VARIATIONS[incident_type]['locations'])

    for i in range(batch_size):
        # Create variations around the base
        generation_params = {
            'severity': base_severity if i == 0 else random.choice(['minor', 'moderate', 'serious']),
            'is_guilty': random.random() < 0.7,  # 70% guilty
            'background_type': random.choice(SUSPECT_BACKGROUNDS),
            'alibi_type': random.choice(ALIBI_TYPES),
            'experience': random.choice(DETECTIVE_EXPERIENCE),
            'style': random.choice(DETECTIVE_STYLES),
            'suspect_personality': random.choice(SUSPECT_PERSONALITIES),
            'suspect_strategy': random.choice(SUSPECT_STRATEGIES),
            'incident_time': random.choice(INCIDENT_VARIATIONS[incident_type]['times'])
        }

        # Apply theme if specified
        if theme == 'corporate':
            generation_params['background_type'] = random.choice(
                ['former_employee', 'current_employee', 'contractor', 'competitor'])
            generation_params['alibi_type'] = random.choice(['work', 'travel', 'social', 'home'])
        elif theme == 'community':
            generation_params['background_type'] = random.choice(
                ['local_resident', 'visitor', 'family_connection', 'social_connection'])
            generation_params['alibi_type'] = random.choice(
                ['social', 'family_obligation', 'volunteer_work', 'religious_activity'])

        config = create_rich_alibi_config(
            incident_type=incident_type,
            game_id=f"batch_{theme or 'mixed'}_{incident_type.value}_{i}",
            generation_params=generation_params
        )

        scenarios.append(config)

    return scenarios


def generate_enhanced_alibi_scenarios(
        num_scenarios: int = 10,
        incident_types: List[IncidentType] = None,
        include_edge_cases: bool = True,
        themes: List[str] = None
) -> List[GameConfig]:
    """Generate enhanced alibi configs with rich variations"""

    if not incident_types:
        incident_types = list(IncidentType)

    if not themes:
        themes = ['corporate', 'community', 'mixed', None]  # None = no theme

    # Run alignment functions once
    align_alibi_generation_process()
    align_incident_generation()
    align_persona_generation()

    scenarios = []

    # Generate regular scenarios
    scenarios_per_type = max(1, num_scenarios // (len(incident_types) * len(themes)))

    for incident_type in incident_types:
        for theme in themes:
            batch = generate_scenario_batch(
                batch_size=scenarios_per_type,
                incident_type=incident_type,
                theme=theme
            )
            scenarios.extend(batch)

    # Add edge cases if requested
    if include_edge_cases:
        edge_cases = generate_edge_case_scenarios(min(10, num_scenarios // 5))
        scenarios.extend(edge_cases)

    # Ensure we have the right number of scenarios
    if len(scenarios) > num_scenarios:
        scenarios = random.sample(scenarios, num_scenarios)
    elif len(scenarios) < num_scenarios:
        # Generate additional random scenarios
        while len(scenarios) < num_scenarios:
            incident_type = random.choice(incident_types)
            generation_params = {
                'severity': random.choice(['minor', 'moderate', 'serious']),
                'is_guilty': random.random() < 0.7,
                'background_type': random.choice(SUSPECT_BACKGROUNDS),
                'alibi_type': random.choice(ALIBI_TYPES),
                'experience': random.choice(DETECTIVE_EXPERIENCE),
                'style': random.choice(DETECTIVE_STYLES),
                'suspect_personality': random.choice(SUSPECT_PERSONALITIES),
                'suspect_strategy': random.choice(SUSPECT_STRATEGIES)
            }

            config = create_rich_alibi_config(
                incident_type=incident_type,
                game_id=f"enhanced_alibi_{len(scenarios)}",
                generation_params=generation_params
            )
            scenarios.append(config)

    return scenarios


def generate_edge_case_scenarios(num_scenarios: int = 5) -> List[Dict[str, any]]:
    """Generate edge case scenarios for thorough testing"""

    edge_cases = []

    edge_case_templates = [
        {
            # The overly cooperative guilty suspect
            'is_guilty': True,
            'suspect_strategy': 'full_cooperation',
            'suspect_personality': 'confident',
            'complicating_factor': 'conflicting_witnesses',
            'description': 'guilty_but_cooperative'
        },
        {
            # The hostile innocent suspect
            'is_guilty': False,
            'suspect_strategy': 'complete_denial',
            'suspect_personality': 'hostile',
            'complicating_factor': 'previous_record',
            'description': 'innocent_but_hostile'
        },
        {
            # The confused elderly suspect
            'is_guilty': random.choice([True, False]),
            'suspect_strategy': 'memory_issues',
            'suspect_personality': 'confused',
            'background_type': 'local_resident',
            'description': 'elderly_confused'
        },
        {
            # The influential suspect with connections
            'is_guilty': True,
            'suspect_strategy': 'legal_technicalities',
            'suspect_personality': 'confident',
            'complicating_factor': 'influential_connections',
            'description': 'influential_suspect'
        },
        {
            # The suspect with perfect alibi but guilty
            'is_guilty': True,
            'alibi_type': 'medical',
            'evidence_strength': 'weak',
            'suspect_preparation': 'well_prepared',
            'description': 'perfect_alibi_guilty'
        }
    ]

    for i, template in enumerate(edge_case_templates[:num_scenarios]):
        incident_type = random.choice(list(IncidentType))

        generation_params = {
            'severity': random.choice(['moderate', 'serious']),
            'is_guilty': template.get('is_guilty', random.choice([True, False])),
            'background_type': template.get('background_type', random.choice(SUSPECT_BACKGROUNDS)),
            'alibi_type': template.get('alibi_type', random.choice(ALIBI_TYPES)),
            'experience': random.choice(['senior', 'veteran']),  # Experienced detectives for edge cases
            'style': random.choice(DETECTIVE_STYLES),
            'suspect_personality': template.get('suspect_personality', random.choice(SUSPECT_PERSONALITIES)),
            'suspect_strategy': template.get('suspect_strategy', random.choice(SUSPECT_STRATEGIES))
        }

        config = create_rich_alibi_config(
            incident_type=incident_type,
            game_id=f"edge_case_{template['description']}_{i}",
            generation_params=generation_params
        )

        # Override with edge case specific settings
        config['complicating_factor'] = template.get('complicating_factor', random.choice(COMPLICATING_FACTORS))
        config['evidence_strength'] = template.get('evidence_strength', random.choice(EVIDENCE_STRENGTH))
        config['suspect_preparation'] = template.get('suspect_preparation', random.choice(
            ['unprepared', 'somewhat_prepared', 'well_prepared']))

        edge_cases.append(config)

    return edge_cases


def generate_progressive_difficulty_scenarios(
        num_scenarios: int = 15,
        incident_types: List[IncidentType] = None
) -> List[GameConfig]:
    """Generate scenarios with progressively increasing difficulty"""

    if not incident_types:
        incident_types = list(IncidentType)

    scenarios = []

    # Define difficulty levels
    difficulty_levels = [
        {
            'name': 'beginner',
            'evidence_strength': ['strong', 'overwhelming'],
            'suspect_preparation': ['unprepared'],
            'complicating_factors': ['no_record', 'time_sensitive'],
            'detective_experience': ['veteran', 'specialized'],
            'guilty_rate': 0.8
        },
        {
            'name': 'intermediate',
            'evidence_strength': ['moderate', 'strong'],
            'suspect_preparation': ['somewhat_prepared'],
            'complicating_factors': ['multiple_suspects', 'conflicting_witnesses'],
            'detective_experience': ['experienced', 'senior'],
            'guilty_rate': 0.7
        },
        {
            'name': 'advanced',
            'evidence_strength': ['weak', 'moderate', 'conflicting'],
            'suspect_preparation': ['well_prepared'],
            'complicating_factors': ['influential_connections', 'media_attention', 'internal_politics'],
            'detective_experience': ['junior', 'experienced'],
            'guilty_rate': 0.6
        }
    ]

    scenarios_per_level = num_scenarios // len(difficulty_levels)

    for level_idx, level in enumerate(difficulty_levels):
        for i in range(scenarios_per_level):
            incident_type = random.choice(incident_types)

            generation_params = {
                'severity': random.choice(['minor', 'moderate', 'serious']),
                'is_guilty': random.random() < level['guilty_rate'],
                'background_type': random.choice(SUSPECT_BACKGROUNDS),
                'alibi_type': random.choice(ALIBI_TYPES),
                'experience': random.choice(level['detective_experience']),
                'style': random.choice(DETECTIVE_STYLES),
                'suspect_personality': random.choice(SUSPECT_PERSONALITIES),
                'suspect_strategy': random.choice(SUSPECT_STRATEGIES)
            }

            config = create_rich_alibi_config(
                incident_type=incident_type,
                game_id=f"progressive_{level['name']}_{i}",
                generation_params=generation_params
            )

            # Apply difficulty-specific settings
            config['evidence_strength'] = random.choice(level['evidence_strength'])
            config['suspect_preparation'] = random.choice(level['suspect_preparation'])
            config['complicating_factor'] = random.choice(level['complicating_factors'])
            config['difficulty_level'] = level['name']

            scenarios.append(config)

    return scenarios