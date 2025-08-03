"""
Task definitions for sandbagging samples generation.
"""

import os
from typing import List, Optional
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import Model

try:
    from .models import SandbaggingSample, SandbaggingDomain, GenerationConfig
    from .dataset import load_existing_samples, save_generated_samples
    from .generator import generate_samples_with_grok
except ImportError:
    from pipelines.sandbagging.samples_generation.models import SandbaggingSample, SandbaggingDomain, GenerationConfig
    from pipelines.sandbagging.samples_generation.dataset import load_existing_samples, save_generated_samples
    from pipelines.sandbagging.samples_generation.generator import generate_samples_with_grok


def sandbagging_samples_generation_task(
    name: str = "sandbagging_samples_generation",
    sample_json_path: Optional[str] = None,
    num_samples: int = 100,
    samples_per_batch: int = 12,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    output_path: Optional[str] = None,
    model_name: str = "x-ai/grok-4"
) -> Task:
    """
    Create a sandbagging samples generation task.
    
    Args:
        name: Name of the task
        sample_json_path: Path to the original sample.json file
        num_samples: Number of samples to generate
        samples_per_batch: Number of samples to generate per batch
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation
        output_path: Path to save generated samples
        model_name: Name of the model to use
        
    Returns:
        Inspect Task for sandbagging samples generation
    """
    
    # Set default paths if not provided
    if sample_json_path is None:
        sample_json_path = "pipelines/sandbagging/selective_compliance/answer_accuracy/sample.json"
    
    if output_path is None:
        output_path = f"generated_sandbagging_samples_{num_samples}.json"
    
    # Create generation config
    config = GenerationConfig(
        num_samples=num_samples,
        samples_per_batch=samples_per_batch,
        temperature=temperature,
        max_tokens=max_tokens,
        model_name=model_name,
        task_name=name
    )
    
    # Create solver function
    def generation_solver(model: Model, samples: List[dict]) -> List[dict]:
        """Solver function that generates new samples."""
        
        # Load existing samples from sample.json
        print(f"Loading existing samples from {sample_json_path}")
        existing_samples = load_existing_samples(sample_json_path)
        print(f"Loaded {len(existing_samples)} existing samples")
        
        if not existing_samples:
            print("No existing samples found. Cannot generate new samples.")
            return []
        
        # Generate new samples
        print("Starting sample generation...")
        result = generate_samples_with_grok(existing_samples, config, model)
        
        # Save generated samples
        if result.generated_samples:
            save_generated_samples(result.generated_samples, output_path)
        
        # Print summary
        print(f"\nGeneration Summary:")
        print(f"  Total generated: {result.total_generated}")
        print(f"  Successful: {result.successful_generations}")
        print(f"  Failed: {result.failed_generations}")
        print(f"  Time: {result.generation_time_seconds:.2f} seconds")
        print(f"  Output saved to: {output_path}")
        
        # Return empty list since this is a generation task, not evaluation
        return []
    
    # Create a minimal dataset with one dummy sample (required by Inspect)
    class DummySample:
        def __init__(self):
            self.id = "dummy"
            self.input = "dummy"
            self.target = "dummy"
    
    dummy_dataset = MemoryDataset(samples=[DummySample()])
    
    return Task(
        name=name,
        dataset=dummy_dataset,
        solver=generation_solver,
        scorer=[],  # No scoring for generation task
        sandbox="local"
    )


def sandbagging_samples_generation_task_with_config(config: GenerationConfig) -> Task:
    """
    Create a sandbagging samples generation task with a custom config.
    
    Args:
        config: Generation configuration
        
    Returns:
        Inspect Task for sandbagging samples generation
    """
    
    return sandbagging_samples_generation_task(
        name=config.task_name,
        num_samples=config.num_samples,
        samples_per_batch=config.samples_per_batch,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        model_name=config.model_name
    )


def create_domain_specific_generation_task(
    domain: SandbaggingDomain,
    num_samples: int = 50,
    **kwargs
) -> Task:
    """
    Create a domain-specific generation task.
    
    Args:
        domain: Domain to focus on
        num_samples: Number of samples to generate
        **kwargs: Additional arguments for sandbagging_samples_generation_task
        
    Returns:
        Inspect Task for domain-specific generation
    """
    
    return sandbagging_samples_generation_task(
        name=f"sandbagging_samples_generation_{domain.value}",
        num_samples=num_samples,
        **kwargs
    )


# Registry of available generation tasks
GENERATION_TASKS = {
    "sandbagging_samples_generation": sandbagging_samples_generation_task,
    "sandbagging_samples_generation_physical_security": lambda **kwargs: create_domain_specific_generation_task(SandbaggingDomain.PHYSICAL_SECURITY, **kwargs),
    "sandbagging_samples_generation_digital_security": lambda **kwargs: create_domain_specific_generation_task(SandbaggingDomain.DIGITAL_SECURITY, **kwargs),
    "sandbagging_samples_generation_chemistry_biology": lambda **kwargs: create_domain_specific_generation_task(SandbaggingDomain.CHEMISTRY_BIOLOGY, **kwargs),
    "sandbagging_samples_generation_automotive_mechanical": lambda **kwargs: create_domain_specific_generation_task(SandbaggingDomain.AUTOMOTIVE_MECHANICAL, **kwargs),
    "sandbagging_samples_generation_medical_pharmaceutical": lambda **kwargs: create_domain_specific_generation_task(SandbaggingDomain.MEDICAL_PHARMACEUTICAL, **kwargs),
    "sandbagging_samples_generation_financial_systems": lambda **kwargs: create_domain_specific_generation_task(SandbaggingDomain.FINANCIAL_SYSTEMS, **kwargs),
}


def get_generation_task(task_name: str, **kwargs) -> Task:
    """Get a generation task by name."""
    if task_name not in GENERATION_TASKS:
        available = list(GENERATION_TASKS.keys())
        raise ValueError(f"Unknown generation task: {task_name}. Available: {available}")
    
    return GENERATION_TASKS[task_name](**kwargs)


def list_generation_tasks() -> List[str]:
    """List all available generation task names."""
    return list(GENERATION_TASKS.keys()) 