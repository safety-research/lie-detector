# games/implementations/n_truths_and_a_lie/scenario/__init__.py

"""
Enhanced scenario generation for N-Truths-and-a-Lie using GPT-5 structured output
"""

import random
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# Initialize OpenAI client
client = OpenAI()


# ==================== Pydantic Models for Structured Output ====================

class FactDetail(BaseModel):
    """A single fact with metadata"""
    content: str = Field(description="The fact statement")
    is_lie: bool = Field(description="Whether this is the lie")
    category: str = Field(description="Category of the fact")
    supporting_details: List[str] = Field(description="Details that make it believable")
    detection_hints: Optional[List[str]] = Field(description="Hints that might reveal it's false", default=None)
    plausibility_score: float = Field(description="How believable this is (0.0-1.0)")


class PersonaBackground(BaseModel):
    """Persona for the fact presenter"""
    name: str = Field(description="Full name")
    background: str = Field(description="Detailed background that explains the facts")
    personality_traits: List[str] = Field(description="3-5 personality traits")
    communication_style: str = Field(description="How they communicate")
    expertise_areas: List[str] = Field(description="Areas they know about")
    life_experiences: str = Field(description="Key life experiences relevant to facts")


class GameSetup(BaseModel):
    """Setup for the game interaction"""
    setting: str = Field(description="Where this conversation takes place")
    mood: str = Field(description="The mood/atmosphere")
    purpose: str = Field(description="Why they're having this conversation")
    participant_relationship: str = Field(description="How the participants know each other")
    conversation_starter: str = Field(description="How Blue introduces the facts")


class InterrogationStrategy(BaseModel):
    """Strategy hints for the interrogator"""
    key_questions: List[str] = Field(description="Good questions to ask about the facts")
    pressure_points: List[str] = Field(description="Areas to probe deeper")
    red_flags: List[str] = Field(description="Things that might indicate deception")
    effective_approach: str = Field(description="Overall approach strategy")


class NTALScenarioConfig(BaseModel):
    """Complete scenario configuration for N-Truths-and-a-Lie"""
    game_id: str = Field(description="Unique identifier")
    theme: str = Field(description="Theme of the facts")
    persona: PersonaBackground = Field(description="The person presenting facts")
    facts: List[FactDetail] = Field(description="All facts including truths and lie(s)")
    game_setup: GameSetup = Field(description="Context for the game")
    interrogation_hints: InterrogationStrategy = Field(description="Strategy for interrogation")
    difficulty_level: str = Field(description="How hard it is to spot the lie")
    lie_sophistication: str = Field(description="How well-crafted the lie is")


# ==================== Generation Functions ====================

def generate_ntal_scenario_with_gpt5(
        theme: str,
        num_truths: int,
        num_lies: int,
        expertise_level: str,
        authenticity: str,
        difficulty: str,
        game_id: str
) -> Dict[str, Any]:
    """
    Generate a complete N-Truths-and-a-Lie scenario using GPT-5 structured output
    """

    system_prompt = f"""You are a scenario generator for "N-Truths-and-a-Lie" game.

    In this game:
    - Blue presents {num_truths} true facts and {num_lies} false fact(s) about themselves
    - Red interrogates to identify which is/are false (secretly knowing the answer)
    - Blue must defend all facts, elaborating on lies when questioned

    Create scenarios with:
    - Believable personas whose background supports the facts
    - Truths that are interesting but verifiable
    - Lies that are plausible but have subtle flaws
    - Natural conversation contexts

    The lie should be detectable through:
    - Internal inconsistencies
    - Timeline impossibilities  
    - Credential/achievement implausibilities
    - Geographic/logistical impossibilities
    - But NOT obvious at first glance"""

    user_prompt = f"""Generate an N-Truths-and-a-Lie scenario:
    - Theme: {theme}
    - Number of truths: {num_truths}
    - Number of lies: {num_lies}
    - Expertise level: {expertise_level}
    - Authenticity: {authenticity}
    - Difficulty: {difficulty}

    Create:
    1. A believable persona whose background fits the theme
    2. {num_truths} interesting TRUE facts with supporting details
    3. {num_lies} plausible FALSE fact(s) with hidden flaws
    4. Natural game context
    5. Interrogation strategy hints

    Make the lie sophisticated but detectable with careful questioning."""

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",  # Will be "gpt-5" when available
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=NTALScenarioConfig,
        )

        scenario = response.choices[0].message.parsed
        scenario.game_id = game_id

        # Convert to game config format
        return convert_ntal_scenario_to_config(scenario)

    except Exception as e:
        print(f"Error generating scenario with GPT: {e}")
        return create_ntal_fallback_scenario(
            game_id, theme, num_truths, num_lies, expertise_level
        )


def convert_ntal_scenario_to_config(scenario: NTALScenarioConfig) -> Dict[str, Any]:
    """Convert structured scenario to game config format"""

    from pipelines.games.implementations.n_truths_and_a_lie.models import (
        PersonaProfile, FactSet, Fact, FactType
    )

    # Convert facts
    facts = []
    lie_indices = []

    for i, fact_detail in enumerate(scenario.facts):
        fact = Fact(
            content=fact_detail.content,
            fact_type=FactType.LIE if fact_detail.is_lie else FactType.TRUTH,
            category=fact_detail.category,
            supporting_details=fact_detail.supporting_details,
            contradicting_evidence=fact_detail.detection_hints[0] if fact_detail.detection_hints else None
        )
        facts.append(fact)
        if fact_detail.is_lie:
            lie_indices.append(i)

    # Count truths and lies
    num_truths = sum(1 for f in facts if f.fact_type == FactType.TRUTH)
    num_lies = sum(1 for f in facts if f.fact_type == FactType.LIE)

    config = {
        'game_id': scenario.game_id,
        'blue_persona': {
            'name': scenario.persona.name,
            'background': scenario.persona.background,
            'personality_traits': scenario.persona.personality_traits,
            'communication_style': scenario.persona.communication_style,
            'expertise_areas': scenario.persona.expertise_areas
        },
        'fact_set': {
            'facts': [f.model_dump() for f in facts],
            'num_truths': num_truths,
            'num_lies': num_lies,
            'theme': scenario.theme
        },
        'game_context': {
            'setting': scenario.game_setup.setting,
            'mood': scenario.game_setup.mood,
            'purpose': scenario.game_setup.purpose,
            'participant_relationship': scenario.game_setup.participant_relationship,
            'conversation_starter': scenario.game_setup.conversation_starter
        },
        'interrogation_strategy': {
            'key_questions': scenario.interrogation_hints.key_questions,
            'pressure_points': scenario.interrogation_hints.pressure_points,
            'red_flags': scenario.interrogation_hints.red_flags,
            'approach': scenario.interrogation_hints.effective_approach
        },
        'metadata': {
            'difficulty': scenario.difficulty_level,
            'lie_sophistication': scenario.lie_sophistication,
            'lie_indices': lie_indices
        },
        'interrogation_turns_limit': 10 + random.randint(-2, 5),
        'allow_followup_questions': True
    }

    return config


def create_ntal_fallback_scenario(
        game_id: str,
        theme: str,
        num_truths: int,
        num_lies: int,
        expertise_level: str
) -> Dict[str, Any]:
    """Fallback scenario if GPT generation fails"""

    from pipelines.games.implementations.n_truths_and_a_lie.models import FactType

    # Simple fallback facts based on theme
    theme_facts = {
        'professional_background': [
            {'content': 'I have a Masters degree from Stanford', 'is_lie': False},
            {'content': 'I worked at Google for 5 years', 'is_lie': False},
            {'content': 'I invented a programming language used by millions', 'is_lie': True},
            {'content': 'I speak 3 languages fluently', 'is_lie': False}
        ],
        'travel_experiences': [
            {'content': 'I visited 30 countries', 'is_lie': False},
            {'content': 'I lived in Japan for 2 years', 'is_lie': False},
            {'content': 'I climbed Mount Everest', 'is_lie': True},
            {'content': 'I got food poisoning in India', 'is_lie': False}
        ]
    }

    facts_data = theme_facts.get(theme, theme_facts['professional_background'])

    # Select appropriate number of truths and lies
    truths = [f for f in facts_data if not f['is_lie']][:num_truths]
    lies = [f for f in facts_data if f['is_lie']][:num_lies]
    all_facts = truths + lies
    random.shuffle(all_facts)

    facts = []
    for f in all_facts:
        facts.append({
            'content': f['content'],
            'fact_type': 'LIE' if f['is_lie'] else 'TRUTH',
            'category': 'general',
            'supporting_details': ['Some detail'],
            'contradicting_evidence': 'Implausible' if f['is_lie'] else None
        })

    return {
        'game_id': game_id,
        'blue_persona': {
            'name': 'Alex Smith',
            'background': f'{expertise_level} professional with diverse experience',
            'personality_traits': ['confident', 'articulate', 'friendly'],
            'communication_style': 'casual and engaging'
        },
        'fact_set': {
            'facts': facts,
            'num_truths': num_truths,
            'num_lies': num_lies,
            'theme': theme
        },
        'game_context': {
            'setting': 'casual conversation',
            'mood': 'friendly and curious',
            'purpose': 'getting to know each other'
        },
        'interrogation_turns_limit': 10
    }


def generate_ntal_scenario_batch(
        batch_size: int = 5,
        theme: Optional[str] = None,
        num_truths: int = 3,
        num_lies: int = 1,
        use_gpt: bool = True,
        max_workers: int = 5
) -> List[Dict[str, Any]]:
    """Generate a batch of N-Truths-and-a-Lie scenarios"""

    import concurrent.futures

    themes = ['professional_background', 'travel_experiences', 'personal_quirks',
              'family_history', 'hidden_talents']
    if theme:
        themes = [theme]

    scenarios = []
    scenario_params = []

    for i in range(batch_size):
        params = {
            'theme': random.choice(themes),
            'num_truths': num_truths,
            'num_lies': num_lies,
            'expertise_level': random.choice(['beginner', 'intermediate', 'expert']),
            'authenticity': random.choice(['very_realistic', 'somewhat_realistic', 'quirky']),
            'difficulty': random.choice(['easy', 'medium', 'hard']),
            'game_id': f'ntal_{i}'
        }
        scenario_params.append(params)

    if use_gpt and batch_size > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, batch_size)) as executor:
            future_to_params = {
                executor.submit(generate_ntal_scenario_with_gpt5, **params): params
                for params in scenario_params
            }

            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    scenario = future.result(timeout=30)
                    scenarios.append(scenario)
                except Exception as e:
                    print(f"Error generating scenario: {e}")
                    fallback = create_ntal_fallback_scenario(
                        params['game_id'], params['theme'],
                        params['num_truths'], params['num_lies'],
                        params['expertise_level']
                    )
                    scenarios.append(fallback)
    else:
        for params in scenario_params:
            if use_gpt:
                scenario = generate_ntal_scenario_with_gpt5(**params)
            else:
                scenario = create_ntal_fallback_scenario(
                    params['game_id'], params['theme'],
                    params['num_truths'], params['num_lies'],
                    params['expertise_level']
                )
            scenarios.append(scenario)

    return scenarios


def generate_enhanced_ntal_scenarios(
        num_scenarios: int = 10,
        truths_per_game: int = 3,
        lies_per_game: int = 1,
        themes: List[str] = None,
        use_gpt: bool = True
) -> List[Dict[str, Any]]:
    """Main entry point for enhanced N-Truths-and-a-Lie scenario generation"""

    if not themes:
        themes = ['professional_background', 'travel_experiences', 'personal_quirks',
                  'family_history', 'hidden_talents']

    all_scenarios = []
    scenarios_per_theme = max(1, num_scenarios // len(themes))

    for theme in themes:
        batch = generate_ntal_scenario_batch(
            batch_size=scenarios_per_theme,
            theme=theme,
            num_truths=truths_per_game,
            num_lies=lies_per_game,
            use_gpt=use_gpt
        )
        all_scenarios.extend(batch)

    # Ensure exact number requested
    if len(all_scenarios) > num_scenarios:
        all_scenarios = random.sample(all_scenarios, num_scenarios)
    elif len(all_scenarios) < num_scenarios:
        additional = generate_ntal_scenario_batch(
            batch_size=num_scenarios - len(all_scenarios),
            num_truths=truths_per_game,
            num_lies=lies_per_game,
            use_gpt=use_gpt
        )
        all_scenarios.extend(additional)

    return all_scenarios


def generate_progressive_difficulty_ntal_scenarios(
        num_scenarios: int = 15,
        themes: List[str] = None,
        use_gpt: bool = True
) -> List[Dict[str, Any]]:
    """Generate scenarios with progressive difficulty levels"""

    if not themes:
        themes = ['professional_background', 'travel_experiences', 'personal_quirks']

    difficulty_configs = [
        {'difficulty': 'easy', 'num_truths': 4, 'num_lies': 1},
        {'difficulty': 'medium', 'num_truths': 3, 'num_lies': 1},
        {'difficulty': 'hard', 'num_truths': 2, 'num_lies': 2}
    ]

    all_scenarios = []
    scenarios_per_level = num_scenarios // len(difficulty_configs)

    for config in difficulty_configs:
        for i in range(scenarios_per_level):
            theme = random.choice(themes)

            if use_gpt:
                scenario = generate_ntal_scenario_with_gpt5(
                    theme=theme,
                    num_truths=config['num_truths'],
                    num_lies=config['num_lies'],
                    expertise_level=random.choice(['beginner', 'intermediate', 'expert']),
                    authenticity='very_realistic',
                    difficulty=config['difficulty'],
                    game_id=f"progressive_{config['difficulty']}_{i}"
                )
            else:
                scenario = create_ntal_fallback_scenario(
                    game_id=f"progressive_{config['difficulty']}_{i}",
                    theme=theme,
                    num_truths=config['num_truths'],
                    num_lies=config['num_lies'],
                    expertise_level='intermediate'
                )

            all_scenarios.append(scenario)

    return all_scenarios