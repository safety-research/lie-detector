from typing import List, Literal, Dict

import tanuki

from pipelines.games.implementations.alibi.models import IncidentDetails, AlibiStory, PersonaDetails, TimeFrame


@tanuki.patch(generation_params={"temperature": 1})
def generate_incident_scenario(
        incident_type: str,
        severity: Literal['minor', 'moderate', 'serious'] = 'moderate',
        subtype: str = None,
        location_type: str = None,
        time_period: str = None
) -> IncidentDetails:
    """Generate a realistic incident scenario for investigation with rich details"""


@tanuki.patch(generation_params={"temperature": 1})
def generate_suspect_alibi(
        incident_location: str,
        time_frame: str,
        is_guilty: str,
        alibi_type: str,
        suspect_preparation: str = 'unprepared',
        incident_subtype: str = None
) -> AlibiStory:
    """Generate a believable alibi story for the suspect with preparation level"""


@tanuki.patch(generation_params={"temperature": 1})
def generate_suspect_persona(
        incident_type: str,
        background_type: Literal['former_employee', 'current_employee', 'contractor', 'client', 'vendor',
        'local_resident', 'visitor', 'business_partner', 'competitor', 'stranger',
        'family_connection', 'social_connection', 'service_provider'],
        personality_type: str = None,
        complicating_factor: str = None
) -> PersonaDetails:
    """Generate a realistic suspect persona with complications"""


@tanuki.patch(generation_params={"temperature": 1})
def generate_detective_persona(
        experience_level: Literal['rookie', 'junior', 'experienced', 'senior', 'veteran', 'specialized'],
        investigation_style: Literal['methodical', 'aggressive', 'friendly', 'intuitive', 'analytical',
        'empathetic', 'confrontational', 'patient', 'rapid-fire', 'psychological'],
        case_type_specialty: str = None
) -> PersonaDetails:
    """Generate a realistic detective persona with specializations"""


@tanuki.patch(generation_params={"temperature": 1})
def generate_time_frame(
        incident_type: str = None,
        time_period: str = None
) -> TimeFrame:
    """Generate a realistic time frame for when an incident occurred"""


@tanuki.patch(generation_params={"temperature": 1})
def generate_evidence_details(
        evidence_strength: Literal['weak', 'moderate', 'strong', 'overwhelming', 'conflicting'],
        incident_type: str,
        is_guilty: bool
) -> Dict[str, List[str]]:
    """Generate evidence details based on strength and guilt"""


@tanuki.align
def align_incident_generation():
    # Theft incident - serious with corporate espionage subtype
    assert generate_incident_scenario("theft", "serious", "corporate_espionage", "office") == IncidentDetails(
        location="DataTech Solutions office building, 12th floor server room",
        description="Classified AI algorithm source code worth millions was copied from a secure server",
        evidence_against_suspect=[
            "Suspect's keycard accessed server room at 11:47 PM - unusual time",
            "IT logs show suspect's credentials used to access the specific files",
            "Suspect recently interviewed with competitor company",
            "USB device was connected to server during suspect's access window"
        ],
        investigation_details={
            "case_number": "INV-7834-CE",
            "investigating_officer": "Detective Rodriguez - Cybercrime Division",
            "witness_count": "2 security guards, 1 IT administrator",
            "evidence_collected": "Server logs, keycard data, security footage, network traffic analysis",
            "estimated_loss": "$5-10 million in IP value"
        }
    )

    # Theft incident - moderate with inventory subtype
    assert generate_incident_scenario("theft", "moderate", "inventory_theft", "warehouse") == IncidentDetails(
        location="Amazon Fulfillment Center - Zone B",
        description="Multiple high-value electronics disappeared from inventory over 3 weeks",
        evidence_against_suspect=[
            "Suspect worked all shifts when items went missing",
            "Inventory discrepancies match suspect's work zone assignments",
            "Co-worker saw suspect near restricted high-value storage",
            "Suspect's financial records show recent large purchases"
        ],
        investigation_details={
            "case_number": "INV-5621-INV",
            "investigating_officer": "Detective Patterson",
            "witness_count": "4 warehouse workers",
            "evidence_collected": "Inventory logs, work schedules, financial records",
            "estimated_loss": "$15,000 in merchandise"
        }
    )

    # Vandalism incident - minor with digital vandalism subtype
    assert generate_incident_scenario("vandalism", "minor", "digital_vandalism", "online") == IncidentDetails(
        location="Company website and social media accounts",
        description="Defamatory messages and altered content posted across company digital platforms",
        evidence_against_suspect=[
            "IP address traced to suspect's home network",
            "Posts made during suspect's known free time",
            "Writing style matches suspect's previous communications",
            "Suspect had admin credentials from previous employment"
        ],
        investigation_details={
            "case_number": "INV-3412-DV",
            "investigating_officer": "Detective Kim - Digital Crimes",
            "witness_count": "0 direct witnesses",
            "evidence_collected": "IP logs, digital forensics, content analysis",
            "reputation_damage": "Moderate - required PR response"
        }
    )

    # Fraud incident - serious with insurance fraud subtype
    assert generate_incident_scenario("fraud", "serious", "insurance_fraud", "insurance_office") == IncidentDetails(
        location="Liberty Mutual Insurance regional office",
        description="Staged accident claim with falsified medical records totaling $250,000",
        evidence_against_suspect=[
            "Medical provider confirmed records were forged",
            "Surveillance shows suspect with no apparent injuries",
            "Similar claims filed with other insurers",
            "Accomplice admitted to participation"
        ],
        investigation_details={
            "case_number": "INV-9156-IF",
            "investigating_officer": "Detective Chen - Insurance Fraud Unit",
            "witness_count": "3 medical professionals, 1 accomplice",
            "evidence_collected": "Medical records, surveillance footage, claim history",
            "total_fraud_amount": "$250,000 across multiple claims"
        }
    )

    # Workplace misconduct - moderate with data breach subtype
    assert generate_incident_scenario("workplace_misconduct", "moderate", "data_breach",
                                      "remote_work") == IncidentDetails(
        location="Remote access to company CRM system",
        description="Unauthorized download of customer database containing 50,000 records",
        evidence_against_suspect=[
            "VPN logs show after-hours access from suspect's account",
            "Large data transfer detected to personal cloud storage",
            "Suspect's LinkedIn shows new job at competitor starting next month",
            "IT security alert triggered by unusual access pattern"
        ],
        investigation_details={
            "case_number": "INV-2847-DB",
            "investigating_officer": "Detective Martinez - Corporate Security",
            "witness_count": "0 direct witnesses",
            "evidence_collected": "VPN logs, data transfer records, security alerts",
            "compliance_violations": "GDPR and CCPA breach notifications required"
        }
    )


@tanuki.align
def align_alibi_generation_process():
    # Well-prepared guilty suspect with work alibi
    assert generate_suspect_alibi("DataTech Solutions office building", "11:30 PM - 12:30 AM", 'True',
                                  "work", "well_prepared", "corporate_espionage") == AlibiStory(
        location="Home office working on quarterly projections",
        activities=["reviewing financial models", "video call with Asia team", "preparing board presentation"],
        witnesses=["Singapore team lead on scheduled call", "email exchanges with CFO", "neighbor saw lights on"],
        evidence=["Zoom recording of Asia call", "email timestamps", "VPN login from home IP",
                  "document version history"],
        alibi_type="work",
        inconsistencies=["VPN disconnected for 20 minutes during critical time",
                         "No activity on financial model during claimed work time"]
    )

    # Unprepared innocent with medical alibi
    assert generate_suspect_alibi("warehouse", "2:00 PM - 5:00 PM", 'False', "medical", "unprepared",
                                  "inventory_theft") == AlibiStory(
        location="St. Mary's Hospital Emergency Room",
        activities=["severe migraine treatment", "waiting for test results", "IV medication administration"],
        witnesses=["Dr. Patricia Chen", "Nurse Williams", "registration desk staff"],
        evidence=["hospital admission records", "insurance claim", "prescription receipt", "parking validation"],
        alibi_type="medical",
        verifiable='True'
    )

    # Travel alibi for guilty suspect
    assert generate_suspect_alibi("office building", "business_hours", 'True', "travel",
                                  "somewhat_prepared") == AlibiStory(
        location="Driving to client meeting in neighboring city",
        activities=["left office at 10 AM", "highway drive", "stopped for gas", "client lunch meeting"],
        witnesses=["gas station clerk might remember", "client confirmed meeting but at different time"],
        evidence=["credit card at gas station", "toll road records", "calendar appointment"],
        alibi_type="travel",
        inconsistencies=["Client says meeting was at 2 PM not noon", "Gas purchase 50 miles from claimed route"]
    )

    # Exercise alibi for innocent suspect
    assert generate_suspect_alibi("Company website", "evening", 'False', "exercise", "unprepared") == AlibiStory(
        location="24 Hour Fitness gym",
        activities=["cardio workout", "weight training", "shower and sauna"],
        witnesses=["personal trainer Jake", "front desk staff", "workout buddy Marcus"],
        evidence=["gym check-in record", "fitness app workout log", "parking lot security camera"],
        alibi_type="exercise",
        verifiable='True'
    )

    # Family obligation alibi
    assert generate_suspect_alibi("insurance office", "afternoon", 'True', "family_obligation",
                                  "well_prepared") == AlibiStory(
        location="Daughter's school for parent-teacher conference",
        activities=["meeting with teacher", "reviewing grades", "discussing behavior concerns"],
        witnesses=["Mrs. Henderson (teacher)", "other parents in waiting area", "school receptionist"],
        evidence=["appointment confirmation email", "visitor sign-in log", "follow-up email from teacher"],
        alibi_type="family_obligation",
        inconsistencies=["Meeting was only 20 minutes but claimed 2 hours", "No one recalls seeing after meeting ended"]
    )

    # Entertainment alibi with partial truth
    assert generate_suspect_alibi("remote location", "late evening", 'True', "entertainment",
                                  "somewhat_prepared") == AlibiStory(
        location="AMC Movie Theater watching latest Marvel movie",
        activities=["bought ticket online", "watched 9 PM showing", "grabbed dinner after"],
        witnesses=["might be on security cameras", "restaurant server possibly"],
        evidence=["online ticket purchase", "credit card at concession", "movie ticket stub"],
        alibi_type="entertainment",
        inconsistencies=["Ticket was purchased but no evidence of attendance",
                         "Restaurant receipt from different location"]
    )


@tanuki.align
def align_persona_generation():
    # Current employee suspect with influential connections
    assert generate_suspect_persona("fraud", "current_employee", "confident",
                                    "influential_connections") == PersonaDetails(
        name="Victoria Sterling",
        background="Senior VP of Finance, 15 years with company, MBA from Wharton, board member connections",
        personality_traits=["polished", "articulate", "subtly intimidating", "name-drops frequently"],
        motivation="Protect reputation and position, leverage connections for favorable treatment",
        complications=["Brother-in-law is company board member", "Major donor to DA's campaign", "Media-savvy"]
    )

    # Contractor suspect with technical expertise
    assert generate_suspect_persona("theft", "contractor", "analytical", "None") == PersonaDetails(
        name="Marcus Chen",
        background="Independent IT consultant, specialized in server architecture, multiple client engagements",
        personality_traits=["precise", "technical in speech", "impatient with non-technical questions"],
        motivation="Maintain professional reputation and future contracts",
        complications=["Works for multiple competing firms", "Has proprietary knowledge"]
    )

    # Family connection suspect with emotional complications
    assert generate_suspect_persona("vandalism", "family_connection", "emotional",
                                    "personal_connection") == PersonaDetails(
        name="Rebecca Morrison",
        background="Ex-wife of company CFO, going through bitter divorce, former employee herself",
        personality_traits=["volatile", "switches between anger and tears", "brings up personal grievances"],
        motivation="Hurt ex-spouse's career, gain leverage in divorce proceedings",
        complications=["Custody battle ongoing", "Restraining order recently lifted", "Has insider knowledge"]
    )

    # Vendor suspect with financial troubles
    assert generate_suspect_persona("fraud", "vendor", "nervous", "previous_record") == PersonaDetails(
        name="Gary Thompson",
        background="Owns small supplies company, exclusive contract with victim company, struggling financially",
        personality_traits=["overly apologetic", "rambling explanations", "visibly sweating", "eager to please"],
        motivation="Save business from bankruptcy, avoid another conviction",
        complications=["Previous fraud conviction 10 years ago", "Owes money to dangerous people",
                       "Desperate to keep contract"]
    )

    # Stranger suspect with mental health issues
    assert generate_suspect_persona("workplace_misconduct", "stranger", "confused", "None") == PersonaDetails(
        name="David Williams",
        background="No connection to company, found with stolen laptop, homeless for 6 months, veteran",
        personality_traits=["disoriented", "tangential thinking", "moments of clarity", "distrustful of authority"],
        motivation="Unclear motives, possibly survival-driven, wants to be heard",
        complications=["PTSD diagnosis", "Medication non-compliance", "No fixed address for follow-up"]
    )

    # Detective personas with specializations

    # Specialized cybercrime detective
    assert generate_detective_persona("specialized", "analytical", "cybercrime") == PersonaDetails(
        name="Detective Lisa Park",
        background="8 years in cybercrime unit, Computer Science degree, former ethical hacker",
        personality_traits=["tech-savvy", "systematic", "speaks in analogies", "patient with complex cases"],
        motivation="Solve technically challenging cases, stay ahead of cyber criminals",
        specialization="Digital forensics, cryptocurrency tracking, dark web investigations"
    )

    # Rookie detective with empathetic approach
    assert generate_detective_persona("rookie", "empathetic", "None") == PersonaDetails(
        name="Detective James Wilson",
        background="2 years on force, psychology degree, former social worker, top of academy class",
        personality_traits=["good listener", "builds rapport quickly", "sometimes too trusting",
                            "eager to prove himself"],
        motivation="Help people while solving cases, make a difference in community",
        strengths="Witnesses open up to him, sees connections others miss"
    )

    # Veteran with psychological approach
    assert generate_detective_persona("veteran", "psychological", "white_collar") == PersonaDetails(
        name="Detective Robert Hayes",
        background="25 years experience, pioneered interrogation techniques, wrote manual on white-collar criminals",
        personality_traits=["reads people instantly", "uses silence effectively", "intimidating presence",
                            "never rushes"],
        motivation="Perfect record before retirement, pass on knowledge to younger detectives",
        signature_moves="Makes suspects think he knows more than he does, finds psychological pressure points"
    )

    # Experienced rapid-fire detective
    assert generate_detective_persona("experienced", "rapid-fire", "None") == PersonaDetails(
        name="Detective Angela Martinez",
        background="12 years on force, military background, known for quick case closures",
        personality_traits=["high energy", "interrupts frequently", "multiple questions at once", "creates urgency"],
        motivation="Efficiency in investigations, clear case backlog, move up to lieutenant",
        technique="Overwhelm suspects with pace, catch contradictions in rapid responses"
    )

    # Senior detective with patience
    assert generate_detective_persona("senior", "patient", "fraud") == PersonaDetails(
        name="Detective Charles Bennett",
        background="18 years specializing in financial crimes, accounting background, teaches at police academy",
        personality_traits=["methodical", "never shows frustration", "builds cases slowly",
                            "documentary evidence focused"],
        motivation="Build airtight cases for prosecution, mentor junior detectives",
        approach="Let suspects talk themselves into corners, meticulously verify every detail"
    )


@tanuki.align
def align_evidence_generation():
    # Strong evidence for guilty suspect
    assert generate_evidence_details("strong", "theft", "True") == {
        "physical_evidence": [
            "Fingerprints on stolen items",
            "Security footage showing suspect at scene",
            "Stolen property found in suspect's possession"
        ],
        "digital_evidence": [
            "Access logs showing suspect's credentials",
            "Deleted files recovered from suspect's computer",
            "Text messages discussing the theft"
        ],
        "witness_testimony": [
            "Two employees saw suspect leaving with bag",
            "Security guard noticed suspicious behavior",
            "Co-worker overheard planning conversation"
        ],
        "circumstantial": [
            "Recent financial difficulties",
            "Previous access to stolen items",
            "Lied about whereabouts initially"
        ]
    }

    # Weak evidence for innocent suspect
    assert generate_evidence_details("weak", "fraud", "False") == {
        "physical_evidence": [
            "Similar handwriting but not confirmed match",
            "Present at bank but legitimate reason"
        ],
        "digital_evidence": [
            "Email account accessed but could be hacked",
            "IP address from public WiFi"
        ],
        "witness_testimony": [
            "Witness thinks they saw suspect but unsure",
            "Description partially matches"
        ],
        "circumstantial": [
            "Had knowledge of victim's information",
            "No alibi for exact time window",
            "Previous relationship with victim"
        ]
    }

    # Conflicting evidence
    assert generate_evidence_details("conflicting", "vandalism", "True") == {
        "supporting_guilt": [
            "Paint on suspect's clothes matches vandalism",
            "Threatening messages sent to victim",
            "Purchase of spray paint day before"
        ],
        "supporting_innocence": [
            "Alibi confirmed by two witnesses",
            "Security footage shows suspect elsewhere at overlapping time",
            "Different brand of paint than purchased"
        ],
        "unclear_evidence": [
            "Witness descriptions don't fully match",
            "Timeline has unexplained gaps",
            "Motive is questionable"
        ],
        "expert_opinions": [
            "Handwriting expert says 60% match probability",
            "Video analyst questions footage timestamp accuracy"
        ]
    }

    # Overwhelming evidence
    assert generate_evidence_details("overwhelming", "workplace_misconduct", "True") == {
        "direct_evidence": [
            "Multiple victims with consistent stories",
            "Audio recording of misconduct",
            "Written admission in email"
        ],
        "pattern_evidence": [
            "Similar complaints at previous employer",
            "HR reports spanning 2 years",
            "Escalating behavior documented"
        ],
        "corroborating_evidence": [
            "10+ witnesses to various incidents",
            "Security footage of multiple occurrences",
            "Digital trail of inappropriate communications"
        ],
        "no_reasonable_doubt": [
            "Suspect admitted to some incidents",
            "No plausible alternative explanation",
            "Evidence from multiple independent sources"
        ]
    }