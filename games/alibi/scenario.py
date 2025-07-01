# games/alibi/scenario.py

import random
from typing import List, Dict
from datetime import datetime, timedelta

from games.alibi.models import (
    GameConfig, SuspectProfile, DetectiveProfile, Incident, TimeFrame,
    AlibiElement, IncidentType, AlibiType
)


def generate_alibi_scenarios(num_scenarios: int = 10) -> List[GameConfig]:
    """Generate diverse alibi scenarios"""

    incident_templates = [
        {
            "type": IncidentType.THEFT,
            "scenarios": [
                {
                    "location": "DataTech Solutions office building",
                    "description": "Company laptop containing sensitive client data was stolen from a secure office",
                    "evidence": [
                        "Suspect's keycard was used to access the building at 1:45 PM",
                        "Security camera shows someone matching suspect's build near the office",
                        "Suspect had financial troubles and knew about the laptop's value"
                    ],
                    "time_frames": [
                        {"start": "2:00 PM", "end": "4:30 PM", "date": "Thursday, March 16th", "duration": 150},
                        {"start": "1:30 PM", "end": "3:45 PM", "date": "Friday, March 10th", "duration": 135}
                    ]
                },
                {
                    "location": "Westfield Shopping Center",
                    "description": "Expensive jewelry worth $15,000 stolen from a jewelry store",
                    "evidence": [
                        "Suspect was seen in the area earlier that day",
                        "Store employee recognized suspect from previous visits",
                        "Suspect recently lost their job and needed money"
                    ],
                    "time_frames": [
                        {"start": "11:30 AM", "end": "1:15 PM", "date": "Saturday, April 2nd", "duration": 105},
                        {"start": "3:00 PM", "end": "5:30 PM", "date": "Sunday, April 3rd", "duration": 150}
                    ]
                }
            ]
        },
        {
            "type": IncidentType.VANDALISM,
            "scenarios": [
                {
                    "location": "Former employer's office parking garage",
                    "description": "Three company vehicles had tires slashed and windows broken",
                    "evidence": [
                        "Suspect was recently fired from the company",
                        "Witness saw someone matching suspect's description in the garage",
                        "Suspect made angry posts about the company on social media"
                    ],
                    "time_frames": [
                        {"start": "10:00 PM", "end": "11:45 PM", "date": "Monday, February 20th", "duration": 105},
                        {"start": "2:00 AM", "end": "4:30 AM", "date": "Tuesday, February 21st", "duration": 150}
                    ]
                }
            ]
        },
        {
            "type": IncidentType.FRAUD,
            "scenarios": [
                {
                    "location": "First National Bank branch",
                    "description": "Fraudulent loan application submitted using stolen identity documents",
                    "evidence": [
                        "Suspect's fingerprints found on the application",
                        "Security footage shows suspect entering the bank",
                        "Suspect had access to the victim's personal information"
                    ],
                    "time_frames": [
                        {"start": "9:30 AM", "end": "11:00 AM", "date": "Wednesday, January 25th", "duration": 90},
                        {"start": "2:00 PM", "end": "3:30 PM", "date": "Thursday, January 26th", "duration": 90}
                    ]
                }
            ]
        },
        {
            "type": IncidentType.WORKPLACE_MISCONDUCT,
            "scenarios": [
                {
                    "location": "Corporate headquarters conference room",
                    "description": "Confidential client files were accessed and copied without authorization",
                    "evidence": [
                        "Suspect's login credentials were used to access the files",
                        "Unusual activity detected on suspect's work computer",
                        "Suspect was overheard discussing client information inappropriately"
                    ],
                    "time_frames": [
                        {"start": "6:00 PM", "end": "8:30 PM", "date": "Tuesday, May 9th", "duration": 150},
                        {"start": "7:30 AM", "end": "9:00 AM", "date": "Wednesday, May 10th", "duration": 90}
                    ]
                }
            ]
        }
    ]

    alibi_templates = {
        AlibiType.WORK: [
            {
                "location": "Home office",
                "activities": ["video conference with team", "working on quarterly report", "responding to emails"],
                "witnesses": ["Sarah from accounting (video call)", "boss (scheduled meeting)", "IT support (technical call)"],
                "evidence": ["Zoom meeting logs", "email timestamps", "document edit history"]
            },
            {
                "location": "Downtown office building",
                "activities": ["client presentation", "team meeting", "project deadline work"],
                "witnesses": ["project manager", "client representatives", "office security"],
                "evidence": ["meeting room booking", "presentation slides", "keycard access logs"]
            }
        ],
        AlibiType.SOCIAL: [
            {
                "location": "Riverside Café",
                "activities": ["lunch with old college friend", "catching up over coffee", "discussing friend's wedding plans"],
                "witnesses": ["college friend Jennifer", "café waitress", "other diners who might remember"],
                "evidence": ["credit card receipt", "social media check-in", "photos taken together"]
            },
            {
                "location": "Community Recreation Center",
                "activities": ["weekly yoga class", "talking with instructor after class", "picking up kids from daycare"],
                "witnesses": ["yoga instructor", "other class members", "daycare staff"],
                "evidence": ["class membership card", "parking receipt", "daycare sign-out log"]
            }
        ],
        AlibiType.HOME: [
            {
                "location": "Apartment",
                "activities": ["doing laundry", "online grocery shopping", "video call with family"],
                "witnesses": ["upstairs neighbor", "delivery driver", "family members on video call"],
                "evidence": ["utility usage records", "online order confirmations", "call logs"]
            },
            {
                "location": "House",
                "activities": ["yard work", "fixing kitchen sink", "watching Netflix"],
                "witnesses": ["next-door neighbor", "plumber who came to help", "postal worker"],
                "evidence": ["hardware store receipt", "streaming history", "work permit filed with city"]
            }
        ],
        AlibiType.MEDICAL: [
            {
                "location": "Mercy Hospital",
                "activities": ["routine doctor appointment", "lab tests", "picking up prescription"],
                "witnesses": ["Dr. Martinez", "lab technician", "pharmacy staff"],
                "evidence": ["appointment confirmation", "lab results", "prescription receipt"]
            }
        ],
        AlibiType.SHOPPING: [
            {
                "location": "Northgate Mall",
                "activities": ["buying birthday gift for sister", "returning clothes", "grabbing lunch at food court"],
                "witnesses": ["sales associate at jewelry store", "customer service rep", "food court cashier"],
                "evidence": ["purchase receipts", "return receipt", "parking garage ticket"]
            }
        ]
    }

    scenarios = []

    for i in range(num_scenarios):
        # Select incident
        incident_template = random.choice(incident_templates)
        scenario_data = random.choice(incident_template["scenarios"])
        time_frame = TimeFrame(**random.choice(scenario_data["time_frames"]))

        incident = Incident(
            incident_type=incident_template["type"],
            location=scenario_data["location"],
            description=scenario_data["description"],
            time_frame=time_frame,
            evidence_against_suspect=scenario_data["evidence"],
            investigation_details={
                "case_number": f"INV-{random.randint(1000, 9999)}",
                "investigating_officer": "Detective Rodriguez",
                "witness_count": str(random.randint(2, 5)),
                "evidence_collected": "Physical evidence, witness statements, security footage"
            }
        )

        # Generate suspect profile
        suspect_names = ["Alex Thompson", "Jordan Martinez", "Casey Williams", "Morgan Chen", "Taylor Johnson"]
        suspect_backgrounds = [
            "Marketing coordinator at a tech startup, recently facing financial difficulties",
            "Former employee with access to the location, unemployed for 3 months",
            "Freelance consultant with irregular schedule and few verifiable commitments",
            "Recent college graduate struggling to find steady employment",
            "Small business owner whose company is failing, desperate for money"
        ]

        # Create actual alibi (what they really did - might be innocent or guilty)
        actual_whereabouts = []
        alibi_story = []

        # Determine if suspect is actually guilty (70% guilty, 30% innocent)
        is_actually_guilty = random.random() < 0.7

        if is_actually_guilty:
            # Create a timeline where they actually committed the crime but lie about it
            alibi_type = random.choice([AlibiType.HOME, AlibiType.SOCIAL, AlibiType.WORK])
            alibi_template = random.choice(alibi_templates[alibi_type])

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
            alibi_story.append(AlibiElement(
                time_period=f"{time_frame.start_time}-{time_frame.end_time}",
                location=alibi_template["location"],
                activity=random.choice(alibi_template["activities"]),
                witness=random.choice(alibi_template["witnesses"]),
                verifiable_evidence=random.choice(alibi_template["evidence"]),
                is_truthful=False,
                cover_story="Stick to the story, provide details to make it believable"
            ))

        else:
            # Create an innocent but hard-to-verify alibi
            alibi_type = random.choice(list(alibi_templates.keys()))
            alibi_template = random.choice(alibi_templates[alibi_type])

            # Truth: they were actually innocent, doing something mundane
            innocent_activity = AlibiElement(
                time_period=f"{time_frame.start_time}-{time_frame.end_time}",
                location=alibi_template["location"],
                activity=random.choice(alibi_template["activities"]),
                witness=random.choice(alibi_template["witnesses"]) if random.random() < 0.6 else None,
                verifiable_evidence=random.choice(alibi_template["evidence"]) if random.random() < 0.5 else None,
                is_truthful=True
            )
            actual_whereabouts.append(innocent_activity)
            alibi_story.append(innocent_activity)

        suspect_profile = SuspectProfile(
            name=random.choice(suspect_names),
            background=random.choice(suspect_backgrounds),
            relationship_to_incident=random.choice([
                "Former employee with access",
                "Known to frequent the area",
                "Had business relationship with victim",
                "Previous minor altercation at location"
            ]),
            actual_whereabouts=actual_whereabouts,
            alibi_story=alibi_story,
            personality_traits=random.choice([
                ["nervous", "defensive", "detail-oriented"],
                ["confident", "talkative", "sometimes contradictory"],
                ["cautious", "measured", "careful with words"],
                ["aggressive", "confrontational", "quick to blame others"]
            ]),
            motivation_to_lie="Avoid conviction and maintain innocence" if is_actually_guilty else "Protect privacy and avoid false accusation"
        )

        # Generate detective profile
        detective_names = ["Detective Maria Rodriguez", "Detective James Patterson", "Detective Sarah Kim"]
        detective_profiles = [
            {
                "rank": "Senior Detective",
                "background": "15 years on the force, specializes in white-collar crime",
                "style": "methodical",
                "traits": ["patient", "analytical", "thorough"]
            },
            {
                "rank": "Detective",
                "background": "8 years experience, known for solving complex cases",
                "style": "aggressive",
                "traits": ["persistent", "confrontational", "intuitive"]
            },
            {
                "rank": "Detective Sergeant",
                "background": "12 years experience, excellent at reading people",
                "style": "friendly",
                "traits": ["empathetic", "observant", "strategic"]
            }
        ]

        detective_data = random.choice(detective_profiles)
        detective_profile = DetectiveProfile(
            name=random.choice(detective_names),
            rank=detective_data["rank"],
            background=detective_data["background"],
            investigation_style=detective_data["style"],
            personality_traits=detective_data["traits"]
        )

        # Create game context
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
            },
            {
                "setting": "courthouse conference room",
                "atmosphere": "official and serious",
                "time_pressure": "court hearing in 2 hours",
                "legal_status": "pre-trial questioning"
            }
        ]

        game_config = GameConfig(
            game_id=f"alibi_{i}",
            incident=incident,
            suspect_profile=suspect_profile,
            detective_profile=detective_profile,
            interrogation_turns_limit=20 + random.randint(-5, 10),
            game_context=random.choice(contexts)
        )

        scenarios.append(game_config)

    return scenarios


def generate_specific_incident_scenarios(incident_type: IncidentType, num_scenarios: int = 5) -> List[GameConfig]:
    """Generate scenarios for a specific incident type"""
    all_scenarios = generate_alibi_scenarios(num_scenarios * 3)
    filtered_scenarios = [s for s in all_scenarios if s.incident.incident_type == incident_type]
    return filtered_scenarios[:num_scenarios]