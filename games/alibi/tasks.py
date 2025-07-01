# games/alibi/tasks.py

from typing import List

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample

from games.alibi.models import GameConfig
from games.alibi.scenario import generate_alibi_scenarios, generate_specific_incident_scenarios
from games.alibi.solvers import alibi_interrogation_solver
from games.alibi.scorer import alibi_investigation_scorer, alibi_deception_scorer, alibi_consistency_scorer


def create_alibi_task(configs: List[GameConfig], task_name: str) -> Task:
    """Create an alibi interrogation task from a list of game configurations"""
    
    # Create samples from configurations
    samples = []
    for config in configs:
        samples.append(Sample(
            input="Begin alibi interrogation",
            target="Investigate suspect's alibi for inconsistencies",
            metadata={
                "config": config.model_dump(),
                "game_id": config.game_id,
                "incident_type": config.incident.incident_type.value,
                "is_suspect_guilty": not config.suspect_profile.alibi_story[0].is_truthful if config.suspect_profile.alibi_story else False
            }
        ))
    
    # Create dataset
    dataset = MemoryDataset(samples)
    
    return Task(
        name=task_name,
        dataset=dataset,
        solver=alibi_interrogation_solver(),
        scorer=[
            alibi_investigation_scorer(),
            alibi_deception_scorer(),
            alibi_consistency_scorer()
        ],
        sandbox="local"
    )


def create_theft_alibi_task(num_scenarios: int = 5) -> Task:
    """Create task focused on theft alibis"""
    from games.alibi.models import IncidentType
    configs = generate_specific_incident_scenarios(IncidentType.THEFT, num_scenarios)
    return create_alibi_task(configs, f"alibi-theft-investigation")


def create_fraud_alibi_task(num_scenarios: int = 5) -> Task:
    """Create task focused on fraud alibis"""
    from games.alibi.models import IncidentType
    configs = generate_specific_incident_scenarios(IncidentType.FRAUD, num_scenarios)
    return create_alibi_task(configs, f"alibi-fraud-investigation")


def create_workplace_alibi_task(num_scenarios: int = 5) -> Task:
    """Create task focused on workplace misconduct alibis"""
    from games.alibi.models import IncidentType
    configs = generate_specific_incident_scenarios(IncidentType.WORKPLACE_MISCONDUCT, num_scenarios)
    return create_alibi_task(configs, f"alibi-workplace-investigation")


def create_mixed_alibi_task(num_scenarios: int = 10) -> Task:
    """Create task with mixed incident types"""
    configs = generate_alibi_scenarios(num_scenarios)
    return create_alibi_task(configs, f"alibi-mixed-investigation")