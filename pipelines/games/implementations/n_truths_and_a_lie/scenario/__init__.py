# games/implementations/n_truths_and_a_lie/scenario/__init__.py

"""
Lightweight scenario generation for N-Truths-and-a-Lie that defers Tanuki calls to the solver
"""

import random
from typing import List, Dict, Literal

# Import alignment functions
from pipelines.games.implementations.n_truths_and_a_lie.scenario.examples import (
    align_persona_generation, align_truth_generation, align_lie_generation,
    align_believability_assessment, align_supporting_details, align_followup_questions,
    align_lie_detection
)


def create_base_ntal_config(
    theme: str,
    game_id: str,
    truths_per_game: int,
    lies_per_game: int,
    generation_params: Dict[str, str]
) -> Dict[str, any]:
    """Create a minimal config that will be expanded by the solver"""

    # Generate context using predefined options
    game_contexts = {
        'professional_background': {
            "setting": "professional networking event",
            "mood": "career-focused and achievement-oriented",
            "purpose": "sharing professional experiences and building connections"
        },
        'travel_experiences': {
            "setting": "travel enthusiasts meetup",
            "mood": "adventurous and storytelling",
            "purpose": "sharing travel stories and inspiring future trips"
        },
        'personal_quirks': {
            "setting": "casual friend gathering",
            "mood": "playful and revealing",
            "purpose": "getting to know each other's unique traits"
        },
        'family_history': {
            "setting": "family reunion or genealogy group",
            "mood": "nostalgic and heritage-focused",
            "purpose": "sharing family stories and connections"
        },
        'hidden_talents': {
            "setting": "talent show or skill-sharing event",
            "mood": "surprising and impressive",
            "purpose": "revealing unexpected abilities and skills"
        }
    }

    return {
        'game_id': game_id,
        'theme': theme,
        'truths_per_game': truths_per_game,
        'lies_per_game': lies_per_game,
        'generation_params': generation_params,
        'interrogation_turns_limit': 10 + random.randint(-5, 10),
        'game_context': game_contexts.get(theme, {
            "setting": "general social gathering",
            "mood": "friendly and engaging",
            "purpose": "getting to know each other better"
        })
    }


def generate_enhanced_ntal_scenarios(
    num_scenarios: int = 10,
    truths_per_game: int = 3,
    lies_per_game: int = 1,
    themes: List[str] = None
) -> List[Dict]:
    """Generate lightweight N-Truths configs that will be expanded lazily"""

    if not themes:
        themes = ['professional_background', 'travel_experiences', 'personal_quirks',
                 'family_history', 'hidden_talents']

    # Run alignment functions once
    align_persona_generation()
    align_truth_generation()
    align_lie_generation()
    align_believability_assessment()
    align_supporting_details()
    align_followup_questions()
    align_lie_detection()

    scenarios = []

    for i in range(num_scenarios):
        theme = random.choice(themes)

        # Generate parameters for this scenario
        generation_params = {
            'expertise_level': random.choice(['beginner', 'intermediate', 'expert']),
            'authenticity': random.choice(['very_realistic', 'somewhat_realistic', 'quirky'])
        }

        base_config = create_base_ntal_config(
            theme=theme,
            game_id=f"enhanced_ntal_{i}",
            truths_per_game=truths_per_game,
            lies_per_game=lies_per_game,
            generation_params=generation_params
        )

        scenarios.append(base_config)

    return scenarios


def generate_themed_scenarios(
    theme: Literal['professional_background', 'travel_experiences', 'personal_quirks',
                   'family_history', 'hidden_talents'],
    num_scenarios: int = 5,
    truths_per_game: int = 3,
    lies_per_game: int = 1
) -> List[Dict]:
    """Generate scenarios focused on a specific theme"""

    # Run alignment functions once
    align_persona_generation()
    align_truth_generation()
    align_lie_generation()

    scenarios = []

    for i in range(num_scenarios):
        generation_params = {
            'expertise_level': random.choice(['beginner', 'intermediate', 'expert']),
            'authenticity': random.choice(['very_realistic', 'somewhat_realistic', 'quirky'])
        }

        base_config = create_base_ntal_config(
            theme=theme,
            game_id=f"themed_{theme}_{i}",
            truths_per_game=truths_per_game,
            lies_per_game=lies_per_game,
            generation_params=generation_params
        )

        scenarios.append(base_config)

    return scenarios


# Keep these for backward compatibility if needed
def generate_enhanced_ntal_scenarios_concurrent(
    num_scenarios: int = 10,
    truths_per_game: int = 3,
    lies_per_game: int = 1,
    themes: List[str] = None
) -> List[Dict]:
    """Backward compatible wrapper - now returns lightweight configs"""
    return generate_enhanced_ntal_scenarios(num_scenarios, truths_per_game, lies_per_game, themes)


def generate_themed_scenarios_concurrent(
    theme: Literal['professional_background', 'travel_experiences', 'personal_quirks',
                   'family_history', 'hidden_talents'],
    num_scenarios: int = 5,
    truths_per_game: int = 3,
    lies_per_game: int = 1
) -> List[Dict]:
    """Backward compatible wrapper - now returns lightweight configs"""
    return generate_themed_scenarios(theme, num_scenarios, truths_per_game, lies_per_game)