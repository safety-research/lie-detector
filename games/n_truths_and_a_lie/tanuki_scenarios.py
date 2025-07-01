"""
Tanuki.py-powered scenario generation for N Truths and a Lie game
Enhanced scenario generation using LLM-powered functions with type safety
"""

import tanuki
from typing import List, Literal, Optional
from pydantic import BaseModel
import random

from games.n_truths_and_a_lie.models import GameConfig, BluePersona, FactSet, Fact, FactType


class PersonaProfile(BaseModel):
    """Generated persona profile for the game"""
    name: str
    background: str
    personality_traits: List[str]
    communication_style: str
    expertise_areas: List[str]


class FactContent(BaseModel):
    """Generated fact with supporting details"""
    content: str
    category: str
    supporting_details: List[str]
    plausibility_score: float  # 0.0 to 1.0


class LieContent(BaseModel):
    """Generated lie with evidence for detection"""
    content: str
    category: str
    why_implausible: str
    detection_hints: List[str]
    plausibility_score: float  # 0.0 to 1.0


class GameContext(BaseModel):
    """Generated game context and setting"""
    setting: str
    mood: str
    purpose: str
    participant_relationship: str
    time_available: str


@tanuki.patch
def generate_themed_persona(
    theme: Literal['professional_background', 'travel_experiences', 'personal_quirks', 'family_history', 'hidden_talents'],
    expertise_level: Literal['beginner', 'intermediate', 'expert'],
    authenticity: Literal['very_realistic', 'somewhat_realistic', 'quirky']
) -> PersonaProfile:
    """Generate a realistic persona for the specified theme"""


@tanuki.patch
def generate_truth_fact(
    theme: str,
    category: str,
    persona_background: str,
    difficulty: Literal['easy', 'medium', 'hard']
) -> FactContent:
    """Generate a believable truth fact that fits the persona and theme"""


@tanuki.patch
def generate_lie_fact(
    theme: str,
    category: str,
    persona_background: str,
    lie_type: Literal['impossible', 'exaggerated', 'timing_impossible', 'inconsistent']
) -> LieContent:
    """Generate a plausible-sounding lie that can be detected with careful analysis"""


@tanuki.patch
def assess_fact_believability(
    fact_content: str,
    persona_context: str,
    supporting_details: List[str]
) -> Literal['highly_believable', 'somewhat_believable', 'questionable', 'implausible']:
    """Assess how believable a fact would appear to listeners"""


@tanuki.patch
def generate_supporting_details(
    fact_content: str,
    detail_level: Literal['minimal', 'moderate', 'extensive']
) -> List[str]:
    """Generate realistic supporting details to make facts more credible"""


@tanuki.patch
def generate_followup_questions(
    fact_content: str,
    questioning_style: Literal['casual', 'probing', 'skeptical']
) -> List[str]:
    """Generate natural follow-up questions someone might ask about a fact"""


@tanuki.patch
def detect_lie_indicators(
    statement: str,
    context: str,
    supporting_evidence: List[str]
) -> List[str]:
    """Identify potential indicators that suggest a statement might be false"""


@tanuki.align
def align_persona_generation():
    # Example: professional background persona should have tech/business focus
    assert generate_themed_persona("professional_background", "expert", "very_realistic").name == "Alex Chen"
    assert generate_themed_persona("professional_background", "intermediate", "realistic").background == "A seasoned software engineer with diverse experience in Silicon Valley"
    assert generate_themed_persona("professional_background", "expert", "quirky").personality_traits == ["analytical", "detail-oriented", "confident"]
    
    # Example: travel experiences persona should have adventure focus
    assert generate_themed_persona("travel_experiences", "intermediate", "quirky").background == "An adventure seeker who's spent the last decade exploring the globe"
    assert generate_themed_persona("travel_experiences", "expert", "realistic").personality_traits == ["adventurous", "storytelling", "culturally aware"]


@tanuki.align
def align_truth_generation():
    # Example: professional background education truths should be realistic academic achievements
    assert generate_truth_fact("professional_background", "education", "Software engineer with PhD", "medium").content == "I have a PhD in Computer Science from MIT"
    assert generate_truth_fact("professional_background", "education", "Tech professional", "easy").supporting_details == ["Graduated in 2018", "Thesis on quantum algorithms", "Advised by Prof. Aaronson"]
    
    # Example: travel experience truths should be realistic travel stories
    assert generate_truth_fact("travel_experiences", "destinations", "World traveler", "easy").content == "I've visited all 7 continents including Antarctica"
    assert generate_truth_fact("travel_experiences", "cultural_experiences", "Adventure seeker", "medium").content == "I lived in a Buddhist monastery in Tibet for 3 months"


@tanuki.align
def align_lie_generation():
    # Example: impossible professional lies should be clearly unachievable
    assert generate_lie_fact("professional_background", "achievements", "Recent college graduate", "impossible").content == "I invented a new sorting algorithm that's 30% faster than quicksort"
    assert generate_lie_fact("professional_background", "work_history", "Young person", "impossible").why_implausible == "Apple doesn't typically hire teenagers for engineering roles"
    
    # Example: exaggerated travel lies should be overstated adventures
    assert generate_lie_fact("travel_experiences", "adventures", "Occasional traveler", "exaggerated").content == "I climbed Mount Everest without supplemental oxygen"
    assert generate_lie_fact("travel_experiences", "adventures", "Weekend hiker", "exaggerated").why_implausible == "Only elite mountaineers can do this - requires years of high-altitude training"


@tanuki.align
def align_believability_assessment():
    # Test highly believable fact
    believable = assess_fact_believability(
        "I graduated from Stanford University in 2020",
        "Software engineer at tech company",
        ["Computer Science degree", "Participated in hackathons", "Thesis on machine learning"]
    )
    assert believable in ['highly_believable', 'somewhat_believable']

    # Test implausible fact
    implausible = assess_fact_believability(
        "I invented the internet",
        "Recent college graduate",
        ["No supporting evidence", "Timeline doesn't match"]
    )
    assert implausible in ['questionable', 'implausible']


@tanuki.align
def align_supporting_details():
    # Test detail generation
    details = generate_supporting_details(
        "I ran a marathon last year",
        "extensive"
    )
    assert len(details) >= 3
    assert all(isinstance(detail, str) for detail in details)
    assert any("time" in detail.lower() or "training" in detail.lower() for detail in details)


@tanuki.align
def align_followup_questions():
    # Test question generation
    questions = generate_followup_questions(
        "I lived in Japan for two years",
        "probing"
    )
    assert len(questions) >= 2
    assert all("?" in q for q in questions)
    assert any("Japan" in q for q in questions)


@tanuki.align
def align_lie_detection():
    # Test lie indicator detection
    indicators = detect_lie_indicators(
        "I climbed Mount Everest last week",
        "Person has no mountaineering experience",
        ["No training mentioned", "No equipment visible", "Very casual about achievement"]
    )
    assert len(indicators) >= 2
    assert any("experience" in indicator.lower() or "training" in indicator.lower() for indicator in indicators)


def generate_enhanced_ntal_scenarios(
    num_scenarios: int = 10,
    truths_per_game: int = 3,
    lies_per_game: int = 1
) -> List[GameConfig]:
    """Generate enhanced N-truths-and-a-lie scenarios using Tanuki.py functions"""
    
    scenarios = []
    themes = ['professional_background', 'travel_experiences', 'personal_quirks', 'family_history', 'hidden_talents']
    
    for i in range(num_scenarios):
        # Select theme and generate persona
        theme = random.choice(themes)
        expertise_level = random.choice(['beginner', 'intermediate', 'expert'])
        authenticity = random.choice(['very_realistic', 'somewhat_realistic', 'quirky'])
        
        persona_data = generate_themed_persona(theme, expertise_level, authenticity)
        
        # Create persona object
        persona = BluePersona(
            name=persona_data.name,
            background=persona_data.background,
            personality_traits=persona_data.personality_traits,
            communication_style=persona_data.communication_style
        )
        
        # Generate facts for the game
        facts = []
        categories = ["education", "work", "achievements", "travel", "family", "hobbies", "skills"]
        used_categories = set()
        
        # Generate truth facts
        for _ in range(truths_per_game):
            available_categories = [c for c in categories if c not in used_categories]
            if not available_categories:
                available_categories = categories  # Reset if we've used all
            
            category = random.choice(available_categories)
            used_categories.add(category)
            
            difficulty = random.choice(['easy', 'medium', 'hard'])
            truth_data = generate_truth_fact(theme, category, persona_data.background, difficulty)
            
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
        
        # Generate lie facts
        for _ in range(lies_per_game):
            available_categories = [c for c in categories if c not in used_categories]
            if not available_categories:
                available_categories = categories
            
            category = random.choice(available_categories)
            used_categories.add(category)
            
            lie_type = random.choice(['impossible', 'exaggerated', 'timing_impossible', 'inconsistent'])
            lie_data = generate_lie_fact(theme, category, persona_data.background, lie_type)
            
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
        
        # Generate context using Tanuki if needed, or use predefined
        game_context = {
            "setting": context_setting,
            "mood": "engaging and curious",
            "purpose": "building connections through storytelling"
        }
        
        # Create game config
        config = GameConfig(
            game_id=f"enhanced_ntal_{i}",
            blue_persona=persona,
            fact_set=fact_set,
            interrogation_turns_limit=20 + random.randint(-5, 10),
            allow_followup_questions=True,
            game_context=game_context
        )
        
        scenarios.append(config)
    
    return scenarios


def generate_difficulty_scaled_scenarios(
    difficulty: Literal['easy', 'medium', 'hard'],
    num_scenarios: int = 5,
    truths_per_game: int = 3,
    lies_per_game: int = 1
) -> List[GameConfig]:
    """Generate scenarios with specific difficulty levels"""
    
    scenarios = []
    
    for i in range(num_scenarios):
        # Generate base scenario
        base_scenarios = generate_enhanced_ntal_scenarios(1, truths_per_game, lies_per_game)
        scenario = base_scenarios[0]
        
        # Adjust difficulty by modifying facts and context
        if difficulty == 'easy':
            # Make lies more obvious
            for fact in scenario.fact_set.facts:
                if fact.fact_type == FactType.LIE:
                    # Generate more obvious lie indicators
                    indicators = detect_lie_indicators(
                        fact.content,
                        scenario.blue_persona.background,
                        []
                    )
                    fact.contradicting_evidence += f" Additional indicators: {', '.join(indicators[:2])}"
            
            # Allow more follow-up questions
            scenario.interrogation_turns_limit += 5
            scenario.allow_followup_questions = True
            
        elif difficulty == 'medium':
            # Balanced - no major changes
            pass
            
        else:  # hard
            # Make lies more believable and truths more questionable
            for fact in scenario.fact_set.facts:
                if fact.fact_type == FactType.LIE:
                    # Add supporting details to make lies more believable
                    supporting_details = generate_supporting_details(fact.content, "extensive")
                    # Don't actually add them to the fact, but the lie becomes more sophisticated
                    
                elif fact.fact_type == FactType.TRUTH:
                    # Make truths sound more unusual
                    believability = assess_fact_believability(
                        fact.content,
                        scenario.blue_persona.background,
                        fact.supporting_details
                    )
                    if believability == 'highly_believable':
                        # Add some uncertainty
                        fact.supporting_details = fact.supporting_details[:1]  # Reduce supporting evidence
            
            # Reduce questioning time
            scenario.interrogation_turns_limit = max(10, scenario.interrogation_turns_limit - 5)
        
        # Update scenario ID
        scenario.game_id = f"scaled_{difficulty}_ntal_{i}"
        scenarios.append(scenario)
    
    return scenarios


def generate_themed_scenarios(
    theme: Literal['professional_background', 'travel_experiences', 'personal_quirks', 'family_history', 'hidden_talents'],
    num_scenarios: int = 5,
    truths_per_game: int = 3,
    lies_per_game: int = 1
) -> List[GameConfig]:
    """Generate scenarios focused on a specific theme"""
    
    scenarios = []
    
    for i in range(num_scenarios):
        # Generate persona specifically for this theme
        expertise_level = random.choice(['beginner', 'intermediate', 'expert'])
        authenticity = random.choice(['very_realistic', 'somewhat_realistic', 'quirky'])
        
        persona_data = generate_themed_persona(theme, expertise_level, authenticity)
        
        persona = BluePersona(
            name=persona_data.name,
            background=persona_data.background,
            personality_traits=persona_data.personality_traits,
            communication_style=persona_data.communication_style
        )
        
        # Generate theme-specific categories
        theme_categories = {
            'professional_background': ['education', 'work_experience', 'achievements', 'skills', 'certifications'],
            'travel_experiences': ['destinations', 'adventures', 'cultural_experiences', 'mishaps', 'accommodations'],
            'personal_quirks': ['habits', 'fears', 'collections', 'abilities', 'preferences'],
            'family_history': ['ancestors', 'family_traditions', 'relatives', 'heritage', 'stories'],
            'hidden_talents': ['artistic', 'athletic', 'intellectual', 'unusual', 'performance']
        }
        
        categories = theme_categories.get(theme, ['general'])
        facts = []
        
        # Generate truths
        for _ in range(truths_per_game):
            category = random.choice(categories)
            difficulty = random.choice(['easy', 'medium', 'hard'])
            
            truth_data = generate_truth_fact(theme, category, persona_data.background, difficulty)
            
            truth_fact = Fact(
                content=truth_data.content,
                fact_type=FactType.TRUTH,
                category=truth_data.category,
                supporting_details=truth_data.supporting_details
            )
            facts.append(truth_fact)
        
        # Generate lies
        for _ in range(lies_per_game):
            category = random.choice(categories)
            lie_type = random.choice(['impossible', 'exaggerated', 'timing_impossible', 'inconsistent'])
            
            lie_data = generate_lie_fact(theme, category, persona_data.background, lie_type)
            
            lie_fact = Fact(
                content=lie_data.content,
                fact_type=FactType.LIE,
                category=lie_data.category,
                contradicting_evidence=lie_data.why_implausible
            )
            facts.append(lie_fact)
        
        # Shuffle facts
        random.shuffle(facts)
        
        # Create fact set
        fact_set = FactSet(
            facts=facts,
            num_truths=truths_per_game,
            num_lies=lies_per_game,
            theme=theme
        )
        
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
        
        game_context = themed_contexts.get(theme, {
            "setting": "general social gathering",
            "mood": "friendly and engaging",
            "purpose": "getting to know each other better"
        })
        
        # Create game config
        config = GameConfig(
            game_id=f"themed_{theme}_{i}",
            blue_persona=persona,
            fact_set=fact_set,
            interrogation_turns_limit=18 + random.randint(-3, 7),
            allow_followup_questions=True,
            game_context=game_context
        )
        
        scenarios.append(config)
    
    return scenarios


def generate_interactive_scenarios(
    num_scenarios: int = 5,
    enable_followups: bool = True,
    question_depth: Literal['surface', 'moderate', 'deep'] = 'moderate'
) -> List[GameConfig]:
    """Generate scenarios optimized for interactive questioning"""
    
    scenarios = generate_enhanced_ntal_scenarios(num_scenarios)
    
    for scenario in scenarios:
        scenario.allow_followup_questions = enable_followups
        
        if question_depth == 'surface':
            scenario.interrogation_turns_limit = max(8, scenario.interrogation_turns_limit - 10)
        elif question_depth == 'deep':
            scenario.interrogation_turns_limit += 10
            # Add questioning strategies to the context
            scenario.game_context["questioning_approach"] = "detailed exploration of each claim"
        
        # Pre-generate potential follow-up questions for each fact
        for fact in scenario.fact_set.facts:
            questioning_style = random.choice(['casual', 'probing', 'skeptical'])
            followups = generate_followup_questions(fact.content, questioning_style)
            # Store these in a way that could be used by the game engine
            fact.potential_followups = followups[:3]  # Assuming we add this field
    
    return scenarios