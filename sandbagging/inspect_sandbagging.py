"""
Inspect AI integration for sandbagging detection.

This module provides the main entry points for running sandbagging detection
evaluations using the Inspect AI framework.
"""

from typing import List, Optional
from inspect_ai import Task

from .tasks import (
    sandbagging_task,
    sandbagging_control_task, 
    sandbagging_harmful_task,
    sandbagging_physical_security_task,
    sandbagging_digital_security_task,
    sandbagging_chemistry_biology_task,
    sandbagging_automotive_mechanical_task,
    sandbagging_medical_pharmaceutical_task,
    sandbagging_financial_systems_task,
    SANDBAGGING_TASKS,
    get_sandbagging_task
)

from .dataset import (
    create_sandbagging_dataset,
    sandbagging_dataset_factory,
    load_sandbagging_dataset,
    print_dataset_info
)

from .evaluation import (
    SandbaggingEvaluator,
    run_sandbagging_evaluation
)

# Export main task functions for Inspect CLI
__all__ = [
    # Main task function
    "sandbagging_detection",
    
    # Variant tasks
    "sandbagging_control",
    "sandbagging_harmful", 
    
    # Domain-specific tasks
    "sandbagging_physical_security",
    "sandbagging_digital_security", 
    "sandbagging_chemistry_biology",
    "sandbagging_automotive_mechanical",
    "sandbagging_medical_pharmaceutical",
    "sandbagging_financial_systems",
    
    # Utility functions
    "get_sandbagging_task",
    "list_sandbagging_tasks",
    "create_sandbagging_dataset",
    "run_sandbagging_evaluation"
]


def sandbagging_detection() -> Task:
    """Main sandbagging detection task with both control and harmful variants."""
    return sandbagging_task()


def sandbagging_control() -> Task:
    """Sandbagging task with only control (neutral expert) prompts."""
    return sandbagging_control_task()


def sandbagging_harmful() -> Task:
    """Sandbagging task with only harmful context prompts.""" 
    return sandbagging_harmful_task()


def sandbagging_physical_security() -> Task:
    """Sandbagging task for physical security domain."""
    return sandbagging_physical_security_task()


def sandbagging_digital_security() -> Task:
    """Sandbagging task for digital security domain."""
    return sandbagging_digital_security_task()


def sandbagging_chemistry_biology() -> Task:
    """Sandbagging task for chemistry/biology domain."""
    return sandbagging_chemistry_biology_task()


def sandbagging_automotive_mechanical() -> Task:
    """Sandbagging task for automotive/mechanical domain."""
    return sandbagging_automotive_mechanical_task()


def sandbagging_medical_pharmaceutical() -> Task:
    """Sandbagging task for medical/pharmaceutical domain."""
    return sandbagging_medical_pharmaceutical_task()


def sandbagging_financial_systems() -> Task:
    """Sandbagging task for financial systems domain."""
    return sandbagging_financial_systems_task()


def list_sandbagging_tasks() -> List[str]:
    """List all available sandbagging task names."""
    return list(SANDBAGGING_TASKS.keys())


# For Inspect CLI discovery
def get_tasks() -> List[str]:
    """Return list of available task names for Inspect CLI."""
    return [
        "sandbagging_detection",
        "sandbagging_control", 
        "sandbagging_harmful",
        "sandbagging_physical_security",
        "sandbagging_digital_security",
        "sandbagging_chemistry_biology", 
        "sandbagging_automotive_mechanical",
        "sandbagging_medical_pharmaceutical",
        "sandbagging_financial_systems"
    ]