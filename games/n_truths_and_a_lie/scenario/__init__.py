"""
Tanuki.py-powered scenario generation for N Truths and a Lie game
Enhanced scenario generation using LLM-powered functions with type safety
Refactored to use concurrent execution for better performance
"""

import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal

from games.n_truths_and_a_lie.models import GameConfig, BluePersona, FactSet, Fact, FactType

from games.n_truths_and_a_lie.scenario.examples import (
    align_persona_generation, align_truth_generation, align_lie_generation,
    align_believability_assessment, align_supporting_details, align_followup_questions,
    align_lie_detection
)

from games.n_truths_and_a_lie.scenario.examples import (
    generate_themed_persona, generate_truth_fact, generate_lie_fact,
    assess_fact_believability, generate_supporting_details, generate_followup_questions,
    detect_lie_indicators
)


def generate_ntal_scenario_concurrent(theme: str, truths_per_game: int = 3, lies_per_game: int = 1):
    """Concurrent version using only threading (no asyncio)"""

    # Pre-generate random values
    expertise_level = random.choice(['beginner', 'intermediate', 'expert'])
    authenticity = random.choice(['very_realistic', 'somewhat_realistic', 'quirky'])

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit persona generation task
        future_persona = executor.submit(generate_themed_persona, theme, expertise_level, authenticity)
        
        # Wait for persona result
        persona_data = future_persona.result()

        # Create persona object
        persona = BluePersona(
            name=persona_data.name,
            background=persona_data.background,
            personality_traits=persona_data.personality_traits,
            communication_style=persona_data.communication_style
        )

        # Generate facts concurrently
        categories = ["education", "work", "achievements", "travel", "family", "hobbies", "skills"]
        used_categories = set()

        # Submit truth generation tasks
        truth_futures = []
        for _ in range(truths_per_game):
            available_categories = [c for c in categories if c not in used_categories]
            if not available_categories:
                available_categories = categories  # Reset if we've used all
            
            category = random.choice(available_categories)
            used_categories.add(category)
            
            difficulty = random.choice(['easy', 'medium', 'hard'])
            future_truth = executor.submit(generate_truth_fact, theme, category, persona_data.background, difficulty)
            truth_futures.append(future_truth)

        # Submit lie generation tasks
        lie_futures = []
        for _ in range(lies_per_game):
            available_categories = [c for c in categories if c not in used_categories]
            if not available_categories:
                available_categories = categories
            
            category = random.choice(available_categories)
            used_categories.add(category)
            
            lie_type = random.choice(['impossible', 'exaggerated', 'timing_impossible', 'inconsistent'])
            future_lie = executor.submit(generate_lie_fact, theme, category, persona_data.background, lie_type)
            lie_futures.append(future_lie)

        # Collect truth results
        facts = []
        for future_truth in truth_futures:
            truth_data = future_truth.result()
            
            # Generate additional supporting details
            extra_details = generate_supporting_details(truth_data.content, "moderate")
            all_details = truth_data.supporting_details + extra_details[:2]  # Limit total details
            
            truth_fact = Fact(
                content=truth_data.content,
                fact_type=FactType.TRUTH,
                category=truth_data.category,
                supporting_details=all_details
            )
            facts.append(truth_fact)

        # Collect lie results
        for future_lie in lie_futures:
            lie_data = future_lie.result()
            
            lie_fact = Fact(
                content=lie_data.content,
                fact_type=FactType.LIE,
                category=lie_data.category,
                contradicting_evidence=lie_data.why_implausible
            )
            facts.append(lie_fact)

    # Shuffle facts so lies aren't always in predictable positions
    random.shuffle(facts)

    # Create fact set
    fact_set = FactSet(
        facts=facts,
        num_truths=truths_per_game,
        num_lies=lies_per_game,
        theme=theme
    )

    # Generate game context
    formality_levels = ['casual', 'semi_formal', 'formal']
    settings = [
        "dinner party ice breaker",
        "team building exercise",
        "virtual meeting warm-up",
        "networking event conversation",
        "workshop activity"
    ]

    context_setting = random.choice(settings)
    formality = random.choice(formality_levels)

    # Generate context using predefined options
    game_context = {
        "setting": context_setting,
        "mood": "engaging and curious",
        "purpose": "building connections through storytelling"
    }

    # Create game config
    config = GameConfig(
        game_id=f"enhanced_ntal",
        blue_persona=persona,
        fact_set=fact_set,
        interrogation_turns_limit=20 + random.randint(-5, 10),
        allow_followup_questions=True,
        game_context=game_context
    )

    return config


def generate_enhanced_ntal_scenarios_concurrent(
    num_scenarios: int = 10,
    truths_per_game: int = 3,
    lies_per_game: int = 1,
    themes: List[str] = []
) -> List[GameConfig]:
    """Generate scenarios using threading-based concurrency"""
    if not themes:
        themes = ['professional_background', 'travel_experiences', 'personal_quirks', 'family_history', 'hidden_talents']

    # Run alignment functions
    align_persona_generation()
    align_truth_generation()
    align_lie_generation()
    align_believability_assessment()
    align_supporting_details()
    align_followup_questions()
    align_lie_detection()

    scenarios = []

    with ThreadPoolExecutor(max_workers=min(num_scenarios, 10)) as executor:
        # Submit all tasks
        futures = []
        for i in range(num_scenarios):
            theme = random.choice(themes)
            future = executor.submit(generate_ntal_scenario_concurrent, theme, truths_per_game, lies_per_game)
            futures.append(future)

        # Collect results
        for i, future in enumerate(futures):
            scenario = future.result()
            scenario.game_id = f"enhanced_ntal_{i}"
            scenarios.append(scenario)

    return scenarios


def generate_themed_scenarios_concurrent(
    theme: Literal['professional_background', 'travel_experiences', 'personal_quirks', 'family_history', 'hidden_talents'],
    num_scenarios: int = 5,
    truths_per_game: int = 3,
    lies_per_game: int = 1
) -> List[GameConfig]:
    """Generate scenarios focused on a specific theme using concurrency"""
    
    # Run alignment functions
    align_persona_generation()
    align_truth_generation()
    align_lie_generation()

    scenarios = []

    with ThreadPoolExecutor(max_workers=min(num_scenarios, 10)) as executor:
        # Submit all tasks for the specific theme
        futures = []
        for i in range(num_scenarios):
            future = executor.submit(generate_ntal_scenario_concurrent, theme, truths_per_game, lies_per_game)
            futures.append(future)

        # Collect results
        for i, future in enumerate(futures):
            scenario = future.result()
            scenario.game_id = f"themed_{theme}_{i}"
            
            # Create themed game context
            themed_contexts = {
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
            
            scenario.game_context = themed_contexts.get(theme, {
                "setting": "general social gathering",
                "mood": "friendly and engaging",
                "purpose": "getting to know each other better"
            })
            
            scenarios.append(scenario)

    return scenarios


# Use the threading version as default
def generate_enhanced_ntal_scenarios(
    num_scenarios: int = 10,
    truths_per_game: int = 3,
    lies_per_game: int = 1
) -> List[GameConfig]:
    """Wrapper to use threading-based concurrent generation"""
    return generate_enhanced_ntal_scenarios_concurrent(num_scenarios, truths_per_game, lies_per_game)


def generate_themed_scenarios(
    theme: Literal['professional_background', 'travel_experiences', 'personal_quirks', 'family_history', 'hidden_talents'],
    num_scenarios: int = 5,
    truths_per_game: int = 3,
    lies_per_game: int = 1
) -> List[GameConfig]:
    """Wrapper to use threading-based themed generation"""
    return generate_themed_scenarios_concurrent(theme, num_scenarios, truths_per_game, lies_per_game)