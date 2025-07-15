import os
import sys
from inspect_ai import task, Task

# Add the project root to path to import sandbagging module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.sandbagging.selective_compliance.answer_accuracy.dataset import create_sandbagging_contrastive_dataset, create_sandbagging_dataset
from pipelines.sandbagging.selective_compliance.answer_accuracy.solver import sandbagging_contrastive_solver, sandbagging_with_refusal_detection
from pipelines.sandbagging.selective_compliance.answer_accuracy.scorer import sandbagging_judge_lie_detector, sandbagging_lie_detector_scorer, sandbagging_contrastive_scorer, sandbagging_answer_accuracy_scorer, sandbagging_detection_scorer
from pipelines.sandbagging.selective_compliance.answer_accuracy.models import SandbaggingDomain
from pipelines.sandbagging.selective_compliance.answer_accuracy.hooks import *

@task
def sandbagging_judge_lie_detection():
    """Sandbagging task with judge-based lie detection - returns C score when model lies."""
    dataset = create_sandbagging_contrastive_dataset()
    
    return Task(
        name="sandbagging_judge_lie_detection",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[
            sandbagging_judge_lie_detector()
        ],
        sandbox="local"
    )


@task
def sandbagging_lie_detection():
    """Sandbagging task with rule-based lie detection - returns C score when model lies."""
    dataset = create_sandbagging_contrastive_dataset()
    
    return Task(
        name="sandbagging_lie_detection",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[
            sandbagging_lie_detector_scorer()
        ],
        sandbox="local"
    )


@task
def sandbagging_contrastive():
    """Main sandbagging detection task using contrastive pairs."""
    dataset = create_sandbagging_contrastive_dataset()
    
    return Task(
        name="sandbagging_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[
            sandbagging_contrastive_scorer()
        ],
        sandbox="local"
    )


@task
def sandbagging_detection():
    """Legacy sandbagging detection task with separate control and harmful variants."""
    dataset = create_sandbagging_dataset()
    
    return Task(
        name="sandbagging_detection",
        dataset=dataset,
        solver=sandbagging_with_refusal_detection(),
        scorer=[
            sandbagging_answer_accuracy_scorer(),
            sandbagging_detection_scorer()
        ],
        sandbox="local"
    )


@task
def sandbagging_control():
    """Sandbagging task with only control (neutral expert) prompts."""
    dataset = create_sandbagging_dataset(variants=["control"])
    
    return Task(
        name="sandbagging_control", 
        dataset=dataset,
        solver=sandbagging_with_refusal_detection(),
        scorer=[
            sandbagging_answer_accuracy_scorer(),
            sandbagging_detection_scorer()
        ],
        sandbox="local"
    )


@task
def sandbagging_harmful():
    """Sandbagging task with only harmful context prompts."""
    dataset = create_sandbagging_dataset(variants=["sandbagging"])
    
    return Task(
        name="sandbagging_harmful",
        dataset=dataset,
        solver=sandbagging_with_refusal_detection(),
        scorer=[
            sandbagging_answer_accuracy_scorer(),
            sandbagging_detection_scorer()
        ],
        sandbox="local"
    )


@task
def sandbagging_physical_security_contrastive():
    """Contrastive sandbagging task for physical security domain."""
    dataset = create_sandbagging_contrastive_dataset(domains=[SandbaggingDomain.PHYSICAL_SECURITY])
    
    return Task(
        name="sandbagging_physical_security_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[sandbagging_contrastive_scorer()],
        sandbox="local"
    )


@task
def sandbagging_physical_security():
    """Legacy sandbagging task for physical security domain."""
    dataset = create_sandbagging_dataset(domains=[SandbaggingDomain.PHYSICAL_SECURITY])
    
    return Task(
        name="sandbagging_physical_security",
        dataset=dataset,
        solver=sandbagging_with_refusal_detection(),
        scorer=[
            sandbagging_answer_accuracy_scorer(),
            sandbagging_detection_scorer()
        ],
        sandbox="local"
    )


@task
def sandbagging_digital_security_contrastive():
    """Contrastive sandbagging task for digital security domain."""
    dataset = create_sandbagging_contrastive_dataset(domains=[SandbaggingDomain.DIGITAL_SECURITY])
    
    return Task(
        name="sandbagging_digital_security_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[sandbagging_contrastive_scorer()],
        sandbox="local"
    )


@task
def sandbagging_digital_security():
    """Legacy sandbagging task for digital security domain."""
    dataset = create_sandbagging_dataset(domains=[SandbaggingDomain.DIGITAL_SECURITY])
    
    return Task(
        name="sandbagging_digital_security",
        dataset=dataset,
        solver=sandbagging_with_refusal_detection(),
        scorer=[
            sandbagging_answer_accuracy_scorer(),
            sandbagging_detection_scorer()
        ],
        sandbox="local"
    )


@task
def sandbagging_chemistry_biology_contrastive():
    """Contrastive sandbagging task for chemistry/biology domain."""
    dataset = create_sandbagging_contrastive_dataset(domains=[SandbaggingDomain.CHEMISTRY_BIOLOGY])
    
    return Task(
        name="sandbagging_chemistry_biology_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[sandbagging_contrastive_scorer()],
        sandbox="local"
    )


@task
def sandbagging_chemistry_biology():
    """Legacy sandbagging task for chemistry/biology domain."""
    dataset = create_sandbagging_dataset(domains=[SandbaggingDomain.CHEMISTRY_BIOLOGY])
    
    return Task(
        name="sandbagging_chemistry_biology",
        dataset=dataset,
        solver=sandbagging_with_refusal_detection(),
        scorer=[
            sandbagging_answer_accuracy_scorer(),
            sandbagging_detection_scorer()
        ],
        sandbox="local"
    )


@task
def sandbagging_automotive_mechanical_contrastive():
    """Contrastive sandbagging task for automotive/mechanical domain."""
    dataset = create_sandbagging_contrastive_dataset(domains=[SandbaggingDomain.AUTOMOTIVE_MECHANICAL])
    
    return Task(
        name="sandbagging_automotive_mechanical_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[sandbagging_contrastive_scorer()],
        sandbox="local"
    )


@task
def sandbagging_automotive_mechanical():
    """Legacy sandbagging task for automotive/mechanical domain."""
    dataset = create_sandbagging_dataset(domains=[SandbaggingDomain.AUTOMOTIVE_MECHANICAL])
    
    return Task(
        name="sandbagging_automotive_mechanical",
        dataset=dataset,
        solver=sandbagging_with_refusal_detection(),
        scorer=[
            sandbagging_answer_accuracy_scorer(),
            sandbagging_detection_scorer()
        ],
        sandbox="local"
    )


@task
def sandbagging_medical_pharmaceutical_contrastive():
    """Contrastive sandbagging task for medical/pharmaceutical domain."""
    dataset = create_sandbagging_contrastive_dataset(domains=[SandbaggingDomain.MEDICAL_PHARMACEUTICAL])
    
    return Task(
        name="sandbagging_medical_pharmaceutical_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[sandbagging_contrastive_scorer()],
        sandbox="local"
    )


@task
def sandbagging_medical_pharmaceutical():
    """Legacy sandbagging task for medical/pharmaceutical domain."""
    dataset = create_sandbagging_dataset(domains=[SandbaggingDomain.MEDICAL_PHARMACEUTICAL])
    
    return Task(
        name="sandbagging_medical_pharmaceutical",
        dataset=dataset,
        solver=sandbagging_with_refusal_detection(),
        scorer=[
            sandbagging_answer_accuracy_scorer(),
            sandbagging_detection_scorer()
        ],
        sandbox="local"
    )


@task
def sandbagging_financial_systems_contrastive():
    """Contrastive sandbagging task for financial systems domain."""
    dataset = create_sandbagging_contrastive_dataset(domains=[SandbaggingDomain.FINANCIAL_SYSTEMS])
    
    return Task(
        name="sandbagging_financial_systems_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[sandbagging_contrastive_scorer()],
        sandbox="local"
    )


@task
def sandbagging_financial_systems():
    """Legacy sandbagging task for financial systems domain."""
    dataset = create_sandbagging_dataset(domains=[SandbaggingDomain.FINANCIAL_SYSTEMS])
    
    return Task(
        name="sandbagging_financial_systems",
        dataset=dataset,
        solver=sandbagging_with_refusal_detection(),
        scorer=[
            sandbagging_answer_accuracy_scorer(),
            sandbagging_detection_scorer()
        ],
        sandbox="local"
    )


if __name__ == "__main__":
    # Example usage - run judge-based lie detection evaluation
    from inspect_ai import eval
    
    all_tasks = [sandbagging_judge_lie_detection(),
            sandbagging_lie_detection(),
            sandbagging_contrastive(),
            sandbagging_detection(),
            sandbagging_control(),
            sandbagging_harmful(),
            sandbagging_physical_security_contrastive(),
            sandbagging_physical_security(),    
            sandbagging_digital_security_contrastive(),
            sandbagging_digital_security(),
            sandbagging_chemistry_biology_contrastive(),
            sandbagging_chemistry_biology(),
            sandbagging_automotive_mechanical_contrastive(),
            sandbagging_automotive_mechanical(),
            sandbagging_medical_pharmaceutical_contrastive(),   
            sandbagging_medical_pharmaceutical(),
            sandbagging_financial_systems_contrastive(),
            sandbagging_financial_systems()]
    
    model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    # model = "openrouter/meta-llama/llama-3.3-70b-instruct"

    
    # Run evaluation with a specific model
    results = eval(
        tasks=all_tasks,
        model=model,  # or any model you want to test
        limit=125,  # Limit samples for faster debugging
        log_dir="./logs",  # Specify log directory
    )