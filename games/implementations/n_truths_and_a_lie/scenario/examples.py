from typing import List, Literal

import tanuki
from dotenv import load_dotenv

from games.implementations.n_truths_and_a_lie.models import (
    PersonaProfile, FactContent, LieContent, GameContext
)
load_dotenv()

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
    assert generate_themed_persona("professional_background", "expert", "very_realistic") == PersonaProfile(
        name="Alex Chen",
        background="A seasoned software engineer with diverse experience in Silicon Valley",
        personality_traits=["analytical", "detail-oriented", "confident"],
        communication_style="Technical but accessible, tends to use industry jargon",
        expertise_areas=["software engineering", "AI/ML", "startup management"]
    )

    assert generate_themed_persona("professional_background", "intermediate", "very_realistic") == PersonaProfile(
        name="Jordan Smith",
        background="A seasoned software engineer with diverse experience in Silicon Valley",
        personality_traits=["analytical", "detail-oriented", "confident"],
        communication_style="Professional and measured, shares technical insights",
        expertise_areas=["product management", "team leadership", "agile methodologies"]
    )

    assert generate_themed_persona("professional_background", "expert", "quirky") == PersonaProfile(
        name="Sam Taylor",
        background="Former tech executive turned entrepreneur, unconventional career path",
        personality_traits=["analytical", "detail-oriented", "confident"],
        communication_style="Enthusiastic and storytelling, mixes business and personal anecdotes",
        expertise_areas=["startup funding", "team building", "pivot strategies"]
    )

    # Example: travel experiences persona should have adventure focus
    assert generate_themed_persona("travel_experiences", "intermediate", "quirky") == PersonaProfile(
        name="Casey Johnson",
        background="An adventure seeker who's spent the last decade exploring the globe",
        personality_traits=["adventurous", "storytelling", "culturally aware"],
        communication_style="Animated and descriptive, loves sharing travel stories",
        expertise_areas=["backpacking", "cultural immersion", "budget travel"]
    )

    assert generate_themed_persona("travel_experiences", "expert", "very_realistic") == PersonaProfile(
        name="Morgan Davis",
        background="Professional travel blogger and photographer with global experience",
        personality_traits=["adventurous", "storytelling", "culturally aware"],
        communication_style="Engaging narrator, balances adventure with practical insights",
        expertise_areas=["photography", "cultural documentation", "sustainable tourism"]
    )


@tanuki.align
def align_truth_generation():
    # Example: professional background education truths should be realistic academic achievements
    assert generate_truth_fact("professional_background", "education", "Software engineer with PhD", "medium") == FactContent(
        content="I have a PhD in Computer Science from MIT",
        category="education",
        supporting_details=["Graduated in 2018", "Thesis on quantum algorithms", "Advised by Prof. Aaronson"],
        plausibility_score=0.9
    )

    assert generate_truth_fact("professional_background", "education", "Tech professional", "easy") == FactContent(
        content="I graduated from Stanford University in 2020",
        category="education",
        supporting_details=["Graduated in 2018", "Thesis on quantum algorithms", "Advised by Prof. Aaronson"],
        plausibility_score=0.95
    )

    # Example: travel experience truths should be realistic travel stories
    assert generate_truth_fact("travel_experiences", "destinations", "World traveler", "easy") == FactContent(
        content="I've visited all 7 continents including Antarctica",
        category="destinations",
        supporting_details=["Antarctica in 2019", "Research station visit", "Saw emperor penguins"],
        plausibility_score=0.8
    )

    assert generate_truth_fact("travel_experiences", "cultural_experiences", "Adventure seeker", "medium") == FactContent(
        content="I lived in a Buddhist monastery in Tibet for 3 months",
        category="cultural_experiences",
        supporting_details=["Learned meditation", "No speaking for first month", "Daily 4am wake up"],
        plausibility_score=0.75
    )


@tanuki.align
def align_lie_generation():
    # Example: impossible professional lies should be clearly unachievable
    assert generate_lie_fact("professional_background", "achievements", "Recent college graduate", "impossible") == LieContent(
        content="I invented a new sorting algorithm that's 30% faster than quicksort",
        category="achievements",
        why_implausible="No such algorithm exists - quicksort is already near-optimal for general cases",
        detection_hints=["Algorithm theory contradiction", "No publications", "Impossible performance claim"],
        plausibility_score=0.1
    )

    assert generate_lie_fact("professional_background", "work_history", "Young person", "impossible") == LieContent(
        content="I was the youngest person ever hired at Apple at age 19",
        category="work_history",
        why_implausible="Apple doesn't typically hire teenagers for engineering roles",
        detection_hints=["Age verification records", "Apple hiring policies", "Lack of necessary experience"],
        plausibility_score=0.2
    )

    # Example: exaggerated travel lies should be overstated adventures
    assert generate_lie_fact("travel_experiences", "adventures", "Occasional traveler", "exaggerated") == LieContent(
        content="I climbed Mount Everest without supplemental oxygen",
        category="adventures",
        why_implausible="Only elite mountaineers can do this - requires years of high-altitude training",
        detection_hints=["No mountaineering credentials", "Extreme rarity of feat", "Lack of supporting documentation"],
        plausibility_score=0.05
    )

    assert generate_lie_fact("travel_experiences", "adventures", "Weekend hiker", "exaggerated") == LieContent(
        content="I completed the Appalachian Trail in 2 months",
        category="adventures",
        why_implausible="Only elite mountaineers can do this - requires years of high-altitude training",
        detection_hints=["Impossible timeline", "No thru-hiker credentials", "Lack of preparation evidence"],
        plausibility_score=0.15
    )


@tanuki.align
def align_believability_assessment():
    # Test highly believable fact
    assert assess_fact_believability(
        "I graduated from Stanford University in 2020",
        "Software engineer at tech company",
        ["Computer Science degree", "Participated in hackathons", "Thesis on machine learning"]
    ) == "highly_believable"

    # Test somewhat believable fact
    assert assess_fact_believability(
        "I lived in Japan for two years teaching English",
        "Recent college graduate",
        ["TEFL certificate", "Japanese language basics", "Cultural exchange program"]
    ) == "somewhat_believable"

    # Test questionable fact
    assert assess_fact_believability(
        "I started three successful companies before age 25",
        "Young entrepreneur",
        ["Ambitious goals", "Some business experience", "Networking connections"]
    ) == "questionable"

    # Test implausible fact
    assert assess_fact_believability(
        "I invented the internet",
        "Recent college graduate",
        ["No supporting evidence", "Timeline doesn't match"]
    ) == "implausible"


@tanuki.align
def align_supporting_details():
    # Test detail generation for running accomplishment
    assert generate_supporting_details(
        "I ran a marathon last year",
        "extensive"
    ) == [
        "Boston Marathon in April 2023",
        "Finished in 3 hours 45 minutes",
        "Trained for 6 months beforehand",
        "Raised $2000 for charity",
        "Have the finisher's medal and certificate"
    ]

    # Test detail generation for education
    assert generate_supporting_details(
        "I have a Master's degree in Computer Science",
        "moderate"
    ) == [
        "Graduated from University of California, Berkeley",
        "Specialized in artificial intelligence",
        "Thesis on neural network optimization"
    ]

    # Test detail generation for travel experience
    assert generate_supporting_details(
        "I backpacked through Southeast Asia for 6 months",
        "minimal"
    ) == [
        "Visited Thailand, Vietnam, and Cambodia",
        "Stayed mostly in hostels"
    ]


@tanuki.align
def align_followup_questions():
    # Test question generation for travel experience
    assert generate_followup_questions(
        "I lived in Japan for two years",
        "probing"
    ) == [
        "What city did you live in?",
        "What was your job there?",
        "How well did you learn Japanese?",
        "What was the biggest cultural difference you noticed?",
        "Do you still keep in touch with people from there?"
    ]

    # Test question generation for professional achievement
    assert generate_followup_questions(
        "I published research in Nature magazine",
        "skeptical"
    ) == [
        "What was the title of your paper?",
        "When was it published?",
        "Who were your co-authors?",
        "What was the main finding?",
        "How long did the research take?"
    ]

    # Test question generation for casual fact
    assert generate_followup_questions(
        "I can play five musical instruments",
        "casual"
    ) == [
        "Which instruments do you play?",
        "How long have you been playing?",
        "Do you perform anywhere?",
        "Which one is your favorite?"
    ]


@tanuki.align
def align_lie_detection():
    # Test lie indicator detection for impossible claim
    assert detect_lie_indicators(
        "I climbed Mount Everest last week",
        "Person has no mountaineering experience",
        ["No training mentioned", "No equipment visible", "Very casual about achievement"]
    ) == [
        "Lack of mountaineering experience or credentials",
        "No mention of extensive training required",
        "Casual attitude about extraordinary achievement",
        "No supporting documentation or photos",
        "Timeline seems rushed for such expedition"
    ]

    # Test lie indicator detection for exaggerated professional claim
    assert detect_lie_indicators(
        "I was personally mentored by Steve Jobs",
        "Young professional in tech",
        ["No verifiable connection", "Steve Jobs died in 2011", "Age doesn't match timeline"]
    ) == [
        "Timeline inconsistency with Steve Jobs' death",
        "No verifiable professional connection",
        "Age and career timeline don't align",
        "No supporting evidence or witnesses",
        "Claim seems designed to impress"
    ]

    # Test lie indicator detection for inconsistent story
    assert detect_lie_indicators(
        "I lived in Paris for five years but don't speak French",
        "Claims international experience",
        ["Language inconsistency", "Cultural immersion contradiction", "Practical living concerns"]
    ) == [
        "Language acquisition inconsistency",
        "Contradiction with immersive living experience",
        "Practical difficulties of living without language",
        "Cultural engagement seems limited",
        "Story lacks logical coherence"
    ]