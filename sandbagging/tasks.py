"""
Task definitions for sandbagging detection evaluation.
"""

from typing import List, Optional
from inspect_ai import Task
from inspect_ai.dataset import Dataset

from dataset import create_sandbagging_dataset, sandbagging_dataset_factory
from solver import sandbagging_solver, sandbagging_with_refusal_detection
from scorer import sandbagging_answer_accuracy_scorer, sandbagging_detection_scorer
from models import SandbaggingPromptPair, SandbaggingDomain


def sandbagging_task(
    name: str = "sandbagging_detection",
    dataset: Optional[Dataset] = None,
    variants: Optional[List[str]] = None,
    domains: Optional[List[SandbaggingDomain]] = None,
    include_refusal_detection: bool = True,
    solver_name: str = "default"
) -> Task:
    """
    Create a sandbagging detection task.
    
    Args:
        name: Name of the task
        dataset: Dataset to use. If None, creates default dataset
        variants: List of variants to include ("control", "sandbagging", or both)
        domains: List of domains to filter by
        include_refusal_detection: Whether to include refusal detection in solver
        solver_name: Which solver to use ("default", "simple", "with_refusal")
        
    Returns:
        Inspect Task for sandbagging detection
    """
    
    # Create dataset if not provided
    if dataset is None:
        dataset = create_sandbagging_dataset(
            variants=variants,
            domains=domains
        )
    
    # Select solver
    if solver_name == "simple":
        solver = sandbagging_solver()
    elif solver_name == "with_refusal" or include_refusal_detection:
        solver = sandbagging_with_refusal_detection()
    else:
        solver = sandbagging_solver()
    
    # Select scorers
    scorers = [
        sandbagging_answer_accuracy_scorer(),
        sandbagging_detection_scorer()
    ]
    
    # Note: Refusal detection is handled in the solver, not scorer
    
    return Task(
        name=name,
        dataset=dataset,
        solver=solver,
        scorer=scorers,
        sandbox="local"
    )


def sandbagging_control_task() -> Task:
    """Task with only control prompts."""
    return sandbagging_task(
        name="sandbagging_control",
        variants=["control"]
    )


def sandbagging_harmful_task() -> Task:
    """Task with only sandbagging (harmful context) prompts."""
    return sandbagging_task(
        name="sandbagging_harmful",
        variants=["sandbagging"]
    )


def sandbagging_domain_task(domain: SandbaggingDomain) -> Task:
    """Task for a specific domain."""
    return sandbagging_task(
        name=f"sandbagging_{domain.value}",
        domains=[domain]
    )


def sandbagging_physical_security_task() -> Task:
    """Task for physical security domain."""
    return sandbagging_domain_task(SandbaggingDomain.PHYSICAL_SECURITY)


def sandbagging_digital_security_task() -> Task:
    """Task for digital security domain."""
    return sandbagging_domain_task(SandbaggingDomain.DIGITAL_SECURITY)


def sandbagging_chemistry_biology_task() -> Task:
    """Task for chemistry/biology domain."""
    return sandbagging_domain_task(SandbaggingDomain.CHEMISTRY_BIOLOGY)


def sandbagging_automotive_mechanical_task() -> Task:
    """Task for automotive/mechanical domain."""
    return sandbagging_domain_task(SandbaggingDomain.AUTOMOTIVE_MECHANICAL)


def sandbagging_medical_pharmaceutical_task() -> Task:
    """Task for medical/pharmaceutical domain."""
    return sandbagging_domain_task(SandbaggingDomain.MEDICAL_PHARMACEUTICAL)


def sandbagging_financial_systems_task() -> Task:
    """Task for financial systems domain."""
    return sandbagging_domain_task(SandbaggingDomain.FINANCIAL_SYSTEMS)


def create_all_sandbagging_tasks() -> List[Task]:
    """Create all standard sandbagging tasks."""
    tasks = []
    
    # Main tasks
    tasks.append(sandbagging_task())
    tasks.append(sandbagging_control_task())
    tasks.append(sandbagging_harmful_task())
    
    # Domain-specific tasks
    for domain in SandbaggingDomain:
        tasks.append(sandbagging_domain_task(domain))
    
    return tasks


# Registry of available tasks
SANDBAGGING_TASKS = {
    "sandbagging_detection": sandbagging_task,
    "sandbagging_control": sandbagging_control_task,
    "sandbagging_harmful": sandbagging_harmful_task,
    "sandbagging_physical_security": sandbagging_physical_security_task,
    "sandbagging_digital_security": sandbagging_digital_security_task,
    "sandbagging_chemistry_biology": sandbagging_chemistry_biology_task,
    "sandbagging_automotive_mechanical": sandbagging_automotive_mechanical_task,
    "sandbagging_medical_pharmaceutical": sandbagging_medical_pharmaceutical_task,
    "sandbagging_financial_systems": sandbagging_financial_systems_task,
}


def get_sandbagging_task(task_name: str) -> Task:
    """Get a sandbagging task by name."""
    if task_name not in SANDBAGGING_TASKS:
        available = list(SANDBAGGING_TASKS.keys())
        raise ValueError(f"Unknown task: {task_name}. Available: {available}")
    
    return SANDBAGGING_TASKS[task_name]()


def list_sandbagging_tasks() -> List[str]:
    """List all available sandbagging task names."""
    return list(SANDBAGGING_TASKS.keys())