from typing import List, Literal

import tanuki

from pipelines.games.implementations.alibi.models import IncidentDetails, AlibiStory, PersonaDetails, TimeFrame


@tanuki.patch
def generate_incident_scenario(
    incident_type: str,
    severity: Literal['minor', 'moderate', 'serious'] = 'moderate'
) -> IncidentDetails:
    """Generate a realistic incident scenario for investigation"""


@tanuki.patch
def generate_suspect_alibi(
    incident_location: str,
    time_frame: str,
    is_guilty: str,
    alibi_type: str
) -> AlibiStory:
    """Generate a believable alibi story for the suspect"""


@tanuki.patch
def generate_suspect_persona(
    incident_type: str,
    background_type: Literal['former_employee', 'local_resident', 'business_associate', 'stranger']
) -> PersonaDetails:
    """Generate a realistic suspect persona"""


@tanuki.patch
def generate_detective_persona(
    experience_level: Literal['junior', 'senior', 'veteran'],
    investigation_style: Literal['methodical', 'aggressive', 'friendly', 'intuitive']
) -> PersonaDetails:
    """Generate a realistic detective persona"""


@tanuki.patch
def assess_alibi_credibility(
    alibi_story: str,
    available_evidence: List[str],
    time_frame: str
) -> Literal['highly_credible', 'somewhat_credible', 'questionable', 'implausible']:
    """Assess how credible an alibi story would appear to investigators"""


@tanuki.patch
def generate_time_frame() -> TimeFrame:
    """Generate a realistic time frame for when an incident occurred"""


@tanuki.align
def align_incident_generation():
    # Theft incident - serious
    assert generate_incident_scenario("theft", "serious") == IncidentDetails(
        location="DataTech Solutions office building",
        description="Company laptop containing sensitive client data was stolen from a secure office",
        evidence_against_suspect=[
            "Suspect's keycard was used to access the building at 1:45 PM",
            "Security camera shows someone matching suspect's build near the office",
            "Suspect had financial troubles and knew about the laptop's value"
        ],
        investigation_details={
            "case_number": "INV-7834",
            "investigating_officer": "Detective Rodriguez",
            "witness_count": "3",
            "evidence_collected": "Physical evidence, witness statements, security footage"
        }
    )

    # Theft incident - moderate
    assert generate_incident_scenario("theft", "moderate") == IncidentDetails(
        location="Westfield Shopping Center",
        description="Company laptop containing sensitive client data was stolen from a secure office",
        evidence_against_suspect=[
            "Suspect was seen in the area earlier that day",
            "Store employee recognized suspect from previous visits",
            "Suspect recently lost their job and needed money"
        ],
        investigation_details={
            "case_number": "INV-5621",
            "investigating_officer": "Detective Patterson",
            "witness_count": "2",
            "evidence_collected": "Security footage, witness statements"
        }
    )

    # Vandalism incident - minor
    assert generate_incident_scenario("vandalism", "minor") == IncidentDetails(
        location="Former employer's office parking garage",
        description="Three company vehicles had tires slashed and windows broken",
        evidence_against_suspect=[
            "Suspect was recently fired from the company",
            "Witness saw someone matching suspect's description in the garage",
            "Suspect made angry posts about the company on social media"
        ],
        investigation_details={
            "case_number": "INV-3412",
            "investigating_officer": "Detective Kim",
            "witness_count": "1",
            "evidence_collected": "Social media posts, witness statement, security footage"
        }
    )

    # Fraud incident - moderate
    assert generate_incident_scenario("fraud", "moderate") == IncidentDetails(
        location="First National Bank branch",
        description="Fraudulent loan application submitted using stolen identity documents",
        evidence_against_suspect=[
            "Suspect's fingerprints found on the application",
            "Security footage shows suspect entering the bank",
            "Suspect had access to the victim's personal information"
        ],
        investigation_details={
            "case_number": "INV-9156",
            "investigating_officer": "Detective Rodriguez",
            "witness_count": "2",
            "evidence_collected": "Fingerprints, documents, security footage"
        }
    )


@tanuki.align
def align_alibi_generation_process():
    # Work alibi for office building theft - not guilty
    assert generate_suspect_alibi("DataTech Solutions office building", "2:00 PM - 4:30 PM", 'False',
                                  "work") == AlibiStory(
        location="Home office",
        activities=["video conference with team", "working on quarterly report", "responding to emails"],
        witnesses=["Sarah from accounting (video call)", "boss (scheduled meeting)", "IT support (technical call)"],
        evidence=["Zoom meeting logs", "email timestamps", "document edit history"],
        alibi_type="work"
    )
    #
    # Work alibi for generic office - not guilty
    assert generate_suspect_alibi("office building", "9:00 AM - 5:00 PM", 'False', "work") == AlibiStory(
        location="Downtown office building",
        activities=["video conference with team", "working on quarterly report", "responding to emails"],
        witnesses=["project manager", "client representatives", "office security"],
        evidence=["meeting room booking", "presentation slides", "keycard access logs"],
        alibi_type="work"
    )

    # Social alibi for shopping center - not guilty
    assert generate_suspect_alibi("Westfield Shopping Center", "11:30 AM - 1:15 PM", 'False', "social") == AlibiStory(
        location="Riverside Café",
        activities=["lunch with old college friend", "catching up over coffee", "discussing friend's wedding plans"],
        witnesses=["college friend Jennifer", "café waitress", "other diners who might remember"],
        evidence=["credit card receipt", "social media check-in", "photos taken together"],
        alibi_type="social"
    )

    # Social alibi for generic shopping center - not guilty
    assert generate_suspect_alibi("shopping center", "afternoon", 'False', "social") == AlibiStory(
        location="Community Recreation Center",
        activities=["weekly yoga class", "talking with instructor after class", "picking up kids from daycare"],
        witnesses=["college friend Jennifer", "café waitress", "other diners who might remember"],
        evidence=["class membership card", "parking receipt", "daycare sign-out log"],
        alibi_type="social"
    )

    # Home alibi for any location - not guilty
    assert generate_suspect_alibi("any location", "evening", 'False', "home") == AlibiStory(
        location="Apartment",
        activities=["doing laundry", "online grocery shopping", "video call with family"],
        witnesses=["upstairs neighbor", "delivery driver", "family members on video call"],
        evidence=["utility usage records", "online order confirmations", "call logs"],
        alibi_type="home"
    )

    # Work alibi for guilty suspect (false alibi)
    assert generate_suspect_alibi("bank branch", "2:00 PM - 3:30 PM", 'True', "work") == AlibiStory(
        location="Home office",
        activities=["important client call", "finishing project deliverables", "team status update"],
        witnesses=["remote colleague on call", "client contact", "project manager"],
        evidence=["calendar appointment", "call logs", "email threads"],
        alibi_type="work"
    )


@tanuki.align
def align_persona_generation():
    # Suspect persona - fraud with former employee background
    assert generate_suspect_persona("fraud", "former_employee") == PersonaDetails(
        name="Alex Thompson",
        background="Marketing coordinator at a tech startup, recently facing financial difficulties",
        personality_traits=["nervous", "defensive", "detail-oriented"],
        motivation="Avoid conviction and maintain innocence"
    )

    # Suspect persona - theft with local resident background
    assert generate_suspect_persona("theft", "local_resident") == PersonaDetails(
        name="Jordan Martinez",
        background="Marketing coordinator at a tech startup, recently facing financial difficulties",
        personality_traits=["confident", "talkative", "sometimes contradictory"],
        motivation="Protect reputation in the community"
    )

    # Suspect persona - vandalism with former employee background
    assert generate_suspect_persona("vandalism", "former_employee") == PersonaDetails(
        name="Casey Williams",
        background="Former employee with access to the location, unemployed for 3 months",
        personality_traits=["nervous", "defensive", "detail-oriented"],
        motivation="Avoid criminal record that would prevent future employment"
    )

    # Suspect persona - workplace misconduct with business associate
    assert generate_suspect_persona("workplace_misconduct", "business_associate") == PersonaDetails(
        name="Morgan Chen",
        background="Freelance consultant with irregular schedule and few verifiable commitments",
        personality_traits=["cautious", "measured", "careful with words"],
        motivation="Maintain professional relationships and contracts"
    )

    # Detective persona - senior methodical
    assert generate_detective_persona("senior", "methodical") == PersonaDetails(
        name="Detective Maria Rodriguez",
        background="15 years on the force, specializes in white-collar crime",
        personality_traits=["patient", "analytical", "thorough"],
        motivation="Build airtight cases through careful investigation"
    )

    # Detective persona - veteran aggressive
    assert generate_detective_persona("veteran", "aggressive") == PersonaDetails(
        name="Detective James Patterson",
        background="20 years experience, known for high conviction rate",
        personality_traits=["persistent", "confrontational", "intuitive"],
        motivation="Get confessions and close cases quickly"
    )

    # Detective persona - junior friendly
    assert generate_detective_persona("junior", "friendly") == PersonaDetails(
        name="Detective Sarah Kim",
        background="5 years on the force, graduated top of academy class",
        personality_traits=["empathetic", "observant", "strategic"],
        motivation="Build trust to gather information effectively"
    )


@tanuki.align
def align_credibility_assessment():
    # Highly credible - work meeting with multiple witnesses
    assert assess_alibi_credibility(
        "I was at work in a meeting with 5 colleagues",
        ["meeting calendar entry", "email confirmations", "colleague testimonies"],
        "2:00 PM - 4:00 PM"
    ) == "highly_credible"

    # Somewhat credible - social meeting with evidence
    assert assess_alibi_credibility(
        "I was at Riverside Café with Jennifer",
        ["credit card receipt", "social media check-in"],
        "11:30 AM - 1:15 PM"
    ) == "somewhat_credible"

    # Questionable - home alone with limited evidence
    assert assess_alibi_credibility(
        "I was home alone watching TV",
        ["Netflix viewing history"],
        "8:00 PM - 11:00 PM"
    ) == "questionable"

    # Questionable - vague alibi with inconsistencies
    assert assess_alibi_credibility(
        "I was running errands around town",
        ["gas station receipt from different area"],
        "entire afternoon"
    ) == "questionable"

    # Implausible - impossible alibi
    assert assess_alibi_credibility(
        "I was abducted by aliens",
        ["no evidence", "no witnesses"],
        "unknown time"
    ) == "implausible"

    # Implausible - contradictory alibi
    assert assess_alibi_credibility(
        "I was in two different cities at the same time",
        ["conflicting evidence"],
        "3:00 PM"
    ) == "implausible"
