"""
Tanuki.py-powered scenario generation for Alibi game
Enhanced scenario generation using LLM-powered functions with type safety
Refactored to use concurrent execution for better performance
"""

import random
from concurrent.futures import ThreadPoolExecutor
from typing import List

from pipelines.games.implementations.alibi.models import (
    GameConfig, SuspectProfile, DetectiveProfile, IncidentType, Incident, AlibiElement
)

from pipelines.games.implementations.alibi.scenario.examples import align_alibi_generation_process, align_incident_generation, \
    align_persona_generation, \
    align_credibility_assessment

from pipelines.games.implementations.alibi.scenario.examples import generate_incident_scenario, \
    generate_suspect_alibi, generate_suspect_persona, generate_detective_persona, assess_alibi_credibility, \
    generate_time_frame


def generate_incident_alibi_scenario_concurrent(incident_type: IncidentType):
    """Concurrent version using only threading (no asyncio)"""

    # Pre-generate random values
    severity = random.choice(['minor', 'moderate', 'serious'])
    is_actually_guilty = random.random() < 0.7
    background_type = random.choice(['former_employee', 'local_resident', 'business_associate', 'stranger'])
    alibi_type = random.choice(['work', 'social', 'home', 'medical', 'shopping'])
    experience = random.choice(['junior', 'senior', 'veteran'])
    style = random.choice(['methodical', 'aggressive', 'friendly', 'intuitive'])

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all independent tasks
        future_incident = executor.submit(generate_incident_scenario, incident_type.value, severity)
        future_time_frame = executor.submit(generate_time_frame)
        future_suspect_persona = executor.submit(generate_suspect_persona, incident_type.value, background_type)
        future_detective_persona = executor.submit(generate_detective_persona, experience, style)

        # Wait for results
        incident_details = future_incident.result()
        time_frame = future_time_frame.result()
        suspect_persona_details = future_suspect_persona.result()
        detective_persona_details = future_detective_persona.result()

        # Create incident object
        incident = Incident(
            incident_type=incident_type,
            location=incident_details.location,
            description=incident_details.description,
            time_frame=time_frame,
            evidence_against_suspect=incident_details.evidence_against_suspect,
            investigation_details=incident_details.investigation_details
        )

        # Generate alibi story (depends on previous results)
        alibi_story = generate_suspect_alibi(
            incident.location,
            f"{time_frame.start_time}-{time_frame.end_time}",
            is_actually_guilty,
            alibi_type
        )

    # Rest of the function remains the same...
    # Create actual whereabouts and alibi story
    actual_whereabouts = []
    alibi_elements = []

    if is_actually_guilty:
        # Truth: they were actually at the incident location
        actual_whereabouts.append(AlibiElement(
            time_period=f"{time_frame.start_time}-{time_frame.end_time}",
            location=incident.location,
            activity=f"Committed {incident.incident_type.value.lower()}",
            witness=None,
            verifiable_evidence=None,
            is_truthful=True
        ))

        # Lie: they claim to have been somewhere else
        alibi_elements.append(AlibiElement(
            time_period=f"{time_frame.start_time}-{time_frame.end_time}",
            location=alibi_story.location,
            activity=random.choice(alibi_story.activities),
            witness=random.choice(alibi_story.witnesses) if alibi_story.witnesses else None,
            verifiable_evidence=random.choice(alibi_story.evidence) if alibi_story.evidence else None,
            is_truthful=False,
            cover_story="Stick to the story, provide details to make it believable"
        ))
    else:
        # Innocent: create truthful alibi
        innocent_activity = AlibiElement(
            time_period=f"{time_frame.start_time}-{time_frame.end_time}",
            location=alibi_story.location,
            activity=random.choice(alibi_story.activities),
            witness=random.choice(alibi_story.witnesses) if alibi_story.witnesses and random.random() < 0.6 else None,
            verifiable_evidence=random.choice(
                alibi_story.evidence) if alibi_story.evidence and random.random() < 0.5 else None,
            is_truthful=True
        )
        actual_whereabouts.append(innocent_activity)
        alibi_elements.append(innocent_activity)

    # Create profiles and game config
    suspect_profile = SuspectProfile(
        name=suspect_persona_details.name,
        background=suspect_persona_details.background,
        relationship_to_incident=f"{background_type.replace('_', ' ').title()} with potential access",
        actual_whereabouts=actual_whereabouts,
        alibi_story=alibi_elements,
        personality_traits=suspect_persona_details.personality_traits,
        motivation_to_lie=suspect_persona_details.motivation
    )

    detective_profile = DetectiveProfile(
        name=detective_persona_details.name,
        rank=f"{experience.title()} Detective",
        background=detective_persona_details.background,
        investigation_style=style,
        personality_traits=detective_persona_details.personality_traits
    )

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

    game_config = GameConfig(
        game_id=f"enhanced_alibi",
        incident=incident,
        suspect_profile=suspect_profile,
        detective_profile=detective_profile,
        interrogation_turns_limit=10 + random.randint(-5, 5),
        game_context=random.choice(contexts)
    )

    return game_config

def generate_enhanced_alibi_scenarios_concurrent(num_scenarios: int = 10, incident_types=[]) -> List[GameConfig]:
    """Generate scenarios using threading-based concurrency"""
    if not incident_types:
        incident_types = [
            IncidentType.THEFT,
            IncidentType.VANDALISM,
            IncidentType.FRAUD,
            IncidentType.WORKPLACE_MISCONDUCT
        ]

    align_alibi_generation_process()
    align_incident_generation()
    align_persona_generation()
    align_credibility_assessment()

    scenarios = []

    with ThreadPoolExecutor(max_workers=min(num_scenarios, 10)) as executor:
        # Submit all tasks
        futures = []
        for i in range(num_scenarios):
            incident_type = random.choice(incident_types)
            future = executor.submit(generate_incident_alibi_scenario_concurrent, incident_type)
            futures.append(future)

        # Collect results
        for i, future in enumerate(futures):
            scenario = future.result()
            scenario.game_id = f"enhanced_alibi_{i}"
            scenarios.append(scenario)

    return scenarios


# Use the threading version as default
def generate_enhanced_alibi_scenarios(num_scenarios: int = 10, incident_types=[]) -> List[GameConfig]:
    """Wrapper to use threading-based concurrent generation"""
    return generate_enhanced_alibi_scenarios_concurrent(num_scenarios, incident_types)