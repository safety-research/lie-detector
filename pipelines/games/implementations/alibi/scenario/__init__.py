# pipelines/games/implementations/alibi/scenario/__init__.py

"""
Enhanced scenario generation for Alibi game using GPT-4 structured output
"""

import random
import concurrent.futures
from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from pipelines.games.implementations.alibi.models import (
    IncidentType, GameConfig, Incident, TimeFrame,
    SuspectProfile, DetectiveProfile, AlibiElement
)

load_dotenv()
client = OpenAI()


# ==================== Pydantic Models for Structured Output ====================

class TimeSlot(BaseModel):
    """Represents a time period"""
    start_time: str = Field(description="Start time (e.g., '2:00 PM')")
    end_time: str = Field(description="End time (e.g., '4:30 PM')")
    date: str = Field(description="Date (e.g., 'Tuesday, March 15th')")
    duration_minutes: int = Field(description="Duration in minutes")


class InvestigationDetail(BaseModel):
    """Single investigation detail"""
    key: str = Field(description="Detail type (e.g., 'case_number', 'lead_detective')")
    value: str = Field(description="Detail value")


class IncidentInformation(BaseModel):
    """Details about the incident"""
    type: str = Field(description="Type of incident",
                      pattern="^(THEFT|FRAUD|VANDALISM|ASSAULT|BREAKING_ENTERING|WORKPLACE_MISCONDUCT)$")
    subtype: str = Field(description="Specific subtype")
    location: str = Field(description="Where it occurred")
    time_frame: TimeSlot = Field(description="When it occurred")
    severity: str = Field(description="Severity level", pattern="^(minor|moderate|serious)$")
    description: str = Field(description="Detailed description")
    evidence_against_suspect: List[str] = Field(description="Evidence pointing to suspect")
    investigation_details: List[InvestigationDetail] = Field(description="Investigation metadata")


class AlibiStory(BaseModel):
    """Suspect's alibi story"""
    time_period: str = Field(description="Time period covered")
    location: str = Field(description="Where they claim to have been")
    activity: str = Field(description="What they were doing")
    witness: Optional[str] = Field(description="Who can verify", default=None)
    verifiable_evidence: Optional[str] = Field(description="Physical evidence", default=None)
    is_truthful: bool = Field(description="Whether this is true")
    cover_story_if_lying: Optional[str] = Field(description="How to maintain lie if pressed", default=None)


class PersonaDetails(BaseModel):
    """Character details"""
    name: str = Field(description="Full name")
    background: str = Field(description="Professional/personal background")
    personality_traits: List[str] = Field(description="3-5 personality traits")
    role_description: str = Field(description="Their role in the scenario")
    motivation: str = Field(description="What drives them")


class SuspectInformation(BaseModel):
    """Complete suspect information"""
    persona: PersonaDetails = Field(description="Suspect's identity")
    relationship_to_incident: str = Field(description="How they're connected")
    actual_whereabouts: List[AlibiStory] = Field(description="What really happened")
    alibi_claims: List[AlibiStory] = Field(description="What they claim happened")
    motivation_to_lie: str = Field(description="Why they might lie")
    stress_triggers: List[str] = Field(description="What makes them nervous")
    defense_strategy: str = Field(description="How they'll defend themselves")


class DetectiveInformation(BaseModel):
    """Detective information"""
    persona: PersonaDetails = Field(description="Detective's identity")
    rank: str = Field(description="Official rank")
    investigation_style: str = Field(description="Approach to interrogation")
    experience_level: str = Field(description="Level of experience")
    specialization: str = Field(description="Area of expertise")
    questioning_strategy: str = Field(description="How they'll conduct interrogation")


class InterrogationContext(BaseModel):
    """Context for the interrogation"""
    setting: str = Field(description="Where interrogation takes place")
    atmosphere: str = Field(description="The mood/atmosphere")
    time_pressure: str = Field(description="Any time constraints")
    legal_status: str = Field(description="Legal status of questioning")
    recording_method: str = Field(description="How it's being recorded")
    witnesses_present: str = Field(description="Who else is present")


class ComplicatingFactors(BaseModel):
    """Additional complications"""
    primary_complication: str = Field(description="Main complicating factor")
    secondary_complications: List[str] = Field(description="Other complications")
    impact_on_investigation: str = Field(description="How these affect the case")


class AlibiScenarioConfig(BaseModel):
    """Complete Alibi scenario configuration"""
    game_id: str = Field(description="Unique identifier")
    incident: IncidentInformation = Field(description="The incident details")
    suspect: SuspectInformation = Field(description="Suspect information")
    detective: DetectiveInformation = Field(description="Detective information")
    context: InterrogationContext = Field(description="Interrogation context")
    complications: ComplicatingFactors = Field(description="Complicating factors")
    difficulty_level: str = Field(description="Overall difficulty")
    theme: str = Field(description="Scenario theme")


# ==================== Generation Functions ====================

def generate_alibi_scenario_with_gpt(
        incident_type: str,
        incident_subtype: str,
        is_guilty: bool,
        suspect_background: str,
        detective_style: str,
        evidence_strength: str,
        complication: str,
        game_id: str
) -> Dict[str, Any]:
    """Generate a complete Alibi scenario using GPT-4 structured output"""

    system_prompt = """You are a scenario generator for a detective interrogation game called "Alibi".

    In this game:
    - A suspect is questioned about their whereabouts during an incident
    - The suspect may be guilty (lying about whereabouts) or innocent (truthful)
    - The detective must determine truth through skilled interrogation

    Create realistic, engaging scenarios with:
    - Complex, multi-dimensional characters
    - Believable alibis (false if guilty, true if innocent)
    - Realistic evidence and investigation details
    - Natural interrogation dynamics
    - Subtle inconsistencies in lies that can be uncovered

    For investigation_details, create entries like:
    - key: "case_number", value: "CASE-2024-001"
    - key: "lead_detective", value: "Det. Martinez"
    - key: "witness_count", value: "3 witnesses"
    - key: "evidence_collected", value: "Security footage, fingerprints"
    """

    guilt_status = "GUILTY (was at crime scene)" if is_guilty else "INNOCENT (alibi is true)"

    user_prompt = f"""Generate an Alibi scenario with these parameters:
    - Incident Type: {incident_type} (specifically: {incident_subtype})
    - Suspect is: {guilt_status}
    - Suspect Background: {suspect_background}
    - Detective Style: {detective_style}
    - Evidence Strength: {evidence_strength}
    - Complicating Factor: {complication}

    Create:
    1. Detailed incident with timeline and location
    2. Complex suspect with believable alibi
    3. If guilty: alibi should have subtle flaws
    4. If innocent: alibi should be verifiable but might seem suspicious
    5. Professional detective with clear strategy
    6. Realistic interrogation context

    Make sure investigation_details is a list of key-value pairs."""

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=AlibiScenarioConfig,
        )

        scenario = response.choices[0].message.parsed
        scenario.game_id = game_id

        return convert_alibi_scenario_to_config(scenario)

    except Exception as e:
        print(f"Error generating scenario with GPT: {e}")
        return create_alibi_fallback_scenario(
            game_id, incident_type, incident_subtype,
            is_guilty, suspect_background, detective_style
        )


def convert_alibi_scenario_to_config(scenario: AlibiScenarioConfig) -> Dict[str, Any]:
    """Convert structured scenario to GameConfig format"""

    # Convert investigation details from list to dict
    investigation_details = {}
    for detail in scenario.incident.investigation_details:
        investigation_details[detail.key] = detail.value

    # Convert alibi stories to AlibiElement format
    actual_whereabouts = []
    for story in scenario.suspect.actual_whereabouts:
        actual_whereabouts.append({
            'time_period': story.time_period,
            'location': story.location,
            'activity': story.activity,
            'witness': story.witness,
            'verifiable_evidence': story.verifiable_evidence,
            'is_truthful': True,
            'cover_story': None
        })

    alibi_story = []
    for story in scenario.suspect.alibi_claims:
        alibi_story.append({
            'time_period': story.time_period,
            'location': story.location,
            'activity': story.activity,
            'witness': story.witness,
            'verifiable_evidence': story.verifiable_evidence,
            'is_truthful': story.is_truthful,
            'cover_story': story.cover_story_if_lying
        })

    config = {
        'game_id': scenario.game_id,
        'incident': {
            'incident_type': scenario.incident.type,
            'location': scenario.incident.location,
            'description': scenario.incident.description,
            'time_frame': {
                'start_time': scenario.incident.time_frame.start_time,
                'end_time': scenario.incident.time_frame.end_time,
                'date': scenario.incident.time_frame.date,
                'duration_minutes': str(scenario.incident.time_frame.duration_minutes)
            },
            'evidence_against_suspect': scenario.incident.evidence_against_suspect,
            'investigation_details': investigation_details  # Now it's a dict
        },
        'suspect_profile': {
            'name': scenario.suspect.persona.name,
            'background': scenario.suspect.persona.background,
            'personality_traits': scenario.suspect.persona.personality_traits,
            'relationship_to_incident': scenario.suspect.relationship_to_incident,
            'actual_whereabouts': actual_whereabouts,
            'alibi_story': alibi_story,
            'motivation_to_lie': scenario.suspect.motivation_to_lie
        },
        'detective_profile': {
            'name': scenario.detective.persona.name,
            'background': scenario.detective.persona.background,
            'personality_traits': scenario.detective.persona.personality_traits,
            'rank': scenario.detective.rank,
            'investigation_style': scenario.detective.investigation_style
        },
        'game_context': {
            'setting': scenario.context.setting,
            'atmosphere': scenario.context.atmosphere,
            'time_pressure': scenario.context.time_pressure,
            'legal_status': scenario.context.legal_status
        },
        'interrogation_turns_limit': 10 + random.randint(-2, 5),
        'turns_limit': 10 + random.randint(-2, 5)
    }

    return config

def create_alibi_fallback_scenario(
        game_id: str,
        incident_type: str,
        incident_subtype: str,
        is_guilty: bool,
        suspect_background: str,
        detective_style: str
) -> Dict[str, Any]:
    """Create fallback scenario if GPT generation fails"""

    # Simple time frame
    time_frame = {
        'start_time': '2:00 PM',
        'end_time': '4:30 PM',
        'date': 'Tuesday, March 15th',
        'duration_minutes': '150'
    }

    # Basic incident
    incident = {
        'incident_type': incident_type,
        'location': 'Corporate office building',
        'description': f'A {incident_subtype} incident occurred',
        'time_frame': time_frame,
        'evidence_against_suspect': [
            'Seen near the location',
            'Had access to the area',
            'Motive identified'
        ],
        'investigation_details': {
            'case_number': f'CASE-{game_id}',
            'lead_detective': 'Detective assigned',
            'status': 'Active investigation'
        }
    }

    # Create alibi based on guilt
    if is_guilty:
        actual_whereabouts = [{
            'time_period': f"{time_frame['start_time']}-{time_frame['end_time']}",
            'location': incident['location'],
            'activity': 'Committing the crime',
            'witness': None,
            'verifiable_evidence': None,
            'is_truthful': True,
            'cover_story': None
        }]

        alibi_story = [{
            'time_period': f"{time_frame['start_time']}-{time_frame['end_time']}",
            'location': 'Home',
            'activity': 'Working from home',
            'witness': 'Neighbor saw me',
            'verifiable_evidence': 'Computer logs',
            'is_truthful': False,
            'cover_story': 'Stick to remote work story'
        }]
    else:
        actual_whereabouts = alibi_story = [{
            'time_period': f"{time_frame['start_time']}-{time_frame['end_time']}",
            'location': 'Coffee shop',
            'activity': 'Meeting with client',
            'witness': 'Client can confirm',
            'verifiable_evidence': 'Receipt and calendar',
            'is_truthful': True,
            'cover_story': None
        }]

    return {
        'game_id': game_id,
        'incident': incident,
        'suspect_profile': {
            'name': 'Alex Johnson',
            'background': suspect_background,
            'personality_traits': ['nervous', 'defensive'] if is_guilty else ['cooperative', 'confused'],
            'relationship_to_incident': 'Has access to location',
            'actual_whereabouts': actual_whereabouts,
            'alibi_story': alibi_story,
            'motivation_to_lie': 'Avoid conviction' if is_guilty else 'Clear name'
        },
        'detective_profile': {
            'name': 'Detective Smith',
            'background': 'Experienced investigator',
            'personality_traits': ['methodical', 'observant'],
            'rank': 'Senior Detective',
            'investigation_style': detective_style
        },
        'game_context': {
            'setting': 'police station interrogation room',
            'atmosphere': 'formal and tense',
            'time_pressure': 'suspect can leave at any time',
            'legal_status': 'voluntary questioning'
        },
        'interrogation_turns_limit': 10,
        'turns_limit': 10
    }


def generate_alibi_scenario_batch(
        batch_size: int = 5,
        incident_types: List[str] = None,
        use_gpt: bool = True,
        max_workers: int = 5
) -> List[Dict[str, Any]]:
    """Generate a batch of Alibi scenarios with parallelization"""

    if not incident_types:
        incident_types = ['THEFT', 'FRAUD', 'VANDALISM', 'WORKPLACE_MISCONDUCT']

    # Incident subtypes mapping
    incident_subtypes = {
        'THEFT': ['corporate_espionage', 'inventory_theft', 'intellectual_property'],
        'FRAUD': ['financial_fraud', 'insurance_fraud', 'identity_fraud'],
        'VANDALISM': ['property_damage', 'graffiti', 'digital_vandalism'],
        'WORKPLACE_MISCONDUCT': ['harassment', 'data_breach', 'policy_violation']
    }

    scenarios = []
    scenario_params = []

    for i in range(batch_size):
        incident_type = random.choice(incident_types)
        params = {
            'incident_type': incident_type,
            'incident_subtype': random.choice(incident_subtypes.get(incident_type, ['general'])),
            'is_guilty': random.random() < 0.7,  # 70% guilty
            'suspect_background': random.choice(['employee', 'contractor', 'visitor', 'former_employee']),
            'detective_style': random.choice(['methodical', 'aggressive', 'friendly', 'psychological']),
            'evidence_strength': random.choice(['weak', 'moderate', 'strong', 'overwhelming']),
            'complication': random.choice(['previous_record', 'influential_connections', 'media_attention']),
            'game_id': f'alibi_{i}'
        }
        scenario_params.append(params)

    if use_gpt and batch_size > 1:
        # Parallel generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, batch_size)) as executor:
            future_to_params = {
                executor.submit(generate_alibi_scenario_with_gpt, **params): params
                for params in scenario_params
            }

            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    scenario = future.result(timeout=30)
                    scenarios.append(scenario)
                except Exception as e:
                    print(f"Error generating scenario: {e}")
                    fallback = create_alibi_fallback_scenario(
                        params['game_id'],
                        params['incident_type'],
                        params['incident_subtype'],
                        params['is_guilty'],
                        params['suspect_background'],
                        params['detective_style']
                    )
                    scenarios.append(fallback)
    else:
        # Sequential generation
        for params in scenario_params:
            if use_gpt:
                scenario = generate_alibi_scenario_with_gpt(**params)
            else:
                scenario = create_alibi_fallback_scenario(
                    params['game_id'],
                    params['incident_type'],
                    params['incident_subtype'],
                    params['is_guilty'],
                    params['suspect_background'],
                    params['detective_style']
                )
            scenarios.append(scenario)

    return scenarios


def generate_enhanced_alibi_scenarios(
        num_scenarios: int = 25,
        incident_types: List[IncidentType] = None,
        include_edge_cases: bool = True,
        themes: List[str] = None,
        use_gpt: bool = True,
        max_workers: int = 5
) -> List[Dict[str, Any]]:
    """Main entry point for enhanced Alibi scenario generation"""

    if not incident_types:
        incident_types = [t.value for t in IncidentType]
    else:
        incident_types = [t.value if hasattr(t, 'value') else t for t in incident_types]

    all_scenarios = []

    # Regular scenarios
    regular_count = num_scenarios - (5 if include_edge_cases else 0)
    if regular_count > 0:
        batch = generate_alibi_scenario_batch(
            batch_size=regular_count,
            incident_types=incident_types,
            use_gpt=use_gpt,
            max_workers=max_workers
        )
        all_scenarios.extend(batch)

    # Edge cases
    if include_edge_cases:
        edge_cases = generate_edge_case_scenarios(min(5, num_scenarios // 4), use_gpt)
        all_scenarios.extend(edge_cases)

    # Ensure exact number requested
    if len(all_scenarios) > num_scenarios:
        all_scenarios = random.sample(all_scenarios, num_scenarios)
    elif len(all_scenarios) < num_scenarios:
        additional = generate_alibi_scenario_batch(
            batch_size=num_scenarios - len(all_scenarios),
            incident_types=incident_types,
            use_gpt=use_gpt
        )
        all_scenarios.extend(additional)

    return all_scenarios


def generate_edge_case_scenarios(num_scenarios: int = 5, use_gpt: bool = True) -> List[Dict[str, Any]]:
    """Generate edge case scenarios for testing"""

    edge_cases = []
    edge_templates = [
        {
            'description': 'guilty_but_cooperative',
            'is_guilty': True,
            'detective_style': 'friendly',
            'evidence_strength': 'weak',
            'complication': 'no_prior_record'
        },
        {
            'description': 'innocent_but_hostile',
            'is_guilty': False,
            'detective_style': 'aggressive',
            'evidence_strength': 'strong',
            'complication': 'previous_record'
        },
        {
            'description': 'perfect_alibi_guilty',
            'is_guilty': True,
            'detective_style': 'methodical',
            'evidence_strength': 'weak',
            'complication': 'well_prepared'
        },
        {
            'description': 'confused_innocent',
            'is_guilty': False,
            'detective_style': 'psychological',
            'evidence_strength': 'moderate',
            'complication': 'memory_issues'
        },
        {
            'description': 'influential_guilty',
            'is_guilty': True,
            'detective_style': 'careful',
            'evidence_strength': 'moderate',
            'complication': 'influential_connections'
        }
    ]

    for i, template in enumerate(edge_templates[:num_scenarios]):
        if use_gpt:
            scenario = generate_alibi_scenario_with_gpt(
                incident_type='THEFT',
                incident_subtype='corporate_espionage',
                is_guilty=template['is_guilty'],
                suspect_background='employee',
                detective_style=template['detective_style'],
                evidence_strength=template['evidence_strength'],
                complication=template['complication'],
                game_id=f"edge_{template['description']}_{i}"
            )
        else:
            scenario = create_alibi_fallback_scenario(
                game_id=f"edge_{template['description']}_{i}",
                incident_type='THEFT',
                incident_subtype='corporate_espionage',
                is_guilty=template['is_guilty'],
                suspect_background='employee',
                detective_style=template['detective_style']
            )

        edge_cases.append(scenario)

    return edge_cases