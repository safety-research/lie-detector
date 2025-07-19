# games/alibi/tasks.py

from typing import List, Dict, Any

from inspect_ai import Task
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from pipelines.games.core.base.tasks import BaseTaskCreator, create_game_task
from pipelines.games.implementations.alibi import generate_enhanced_alibi_scenarios
from pipelines.games.implementations.alibi.models import GameConfig
from pipelines.games.implementations.alibi.scorer import alibi_deception_judge, alibi_investigation_judge
from pipelines.games.implementations.alibi.solvers import alibi_interrogation_solver


class AlibiTaskCreator(BaseTaskCreator):
    """Task creator for alibi interrogation games"""
    
    def get_sample_input(self) -> str:
        return "Begin alibi interrogation"
    
    def get_sample_target(self) -> str:
        return "Investigate suspect's alibi for inconsistencies"
    
    def extract_sample_metadata(self, config: GameConfig) -> Dict[str, Any]:
        return config
        # return {
        #     "config": config.model_dump(),
        #     "game_id": config.game_id,
        #     #"incident_type": config.incident.incident_type.value,
        #     #"is_suspect_guilty": not config.suspect_profile.alibi_story[0].is_truthful if config.suspect_profile.alibi_story else False
        # }
    
    def get_solver(self) -> Solver:
        return alibi_interrogation_solver()
    
    def get_scorers(self) -> List[Scorer]:
        return [
            alibi_deception_judge(),
            alibi_investigation_judge()
        ]


def create_alibi_task(configs: List[GameConfig], task_name: str) -> Task:
    """Create an alibi interrogation task from a list of game configurations"""
    return create_game_task(AlibiTaskCreator, configs, task_name)


def create_theft_alibi_task(num_scenarios: int = 100) -> Task:
    """Create task focused on theft alibis"""
    from pipelines.games.implementations.alibi.models import IncidentType
    configs = generate_enhanced_alibi_scenarios(incident_types=[IncidentType.THEFT], num_scenarios=num_scenarios)
    return create_alibi_task(configs, f"alibi_theft_investigation")


def create_fraud_alibi_task(num_scenarios: int = 100) -> Task:
    """Create task focused on fraud alibis"""
    from pipelines.games.implementations.alibi.models import IncidentType
    configs = generate_enhanced_alibi_scenarios(incident_types=[IncidentType.FRAUD], num_scenarios=num_scenarios)
    return create_alibi_task(configs, f"alibi_fraud_investigation")


def create_workplace_alibi_task(num_scenarios: int = 100) -> Task:
    """Create task focused on workplace misconduct alibis"""
    from pipelines.games.implementations.alibi.models import IncidentType
    configs = generate_enhanced_alibi_scenarios(incident_types=[IncidentType.WORKPLACE_MISCONDUCT], num_scenarios=num_scenarios)
    return create_alibi_task(configs, f"alibi_workplace_investigation")


def create_mixed_alibi_task(num_scenarios: int = 100) -> Task:
    """Create task with mixed incident types"""
    configs = generate_enhanced_alibi_scenarios(incident_types=[], num_scenarios=num_scenarios)
    return create_alibi_task(configs, f"alibi_mixed_investigation")