"""
Sandbagging samples generation pipeline.

This module provides functionality to generate new sandbagging detection samples
using Grok-4 with few-shot prompting from existing samples.
"""

from .models import (
    SandbaggingSample,
    SandbaggingDomain,
    GenerationConfig,
    GenerationResult
)

from .dataset import (
    load_existing_samples,
    save_generated_samples,
    get_random_sample_batch,
    filter_samples_by_domain,
    create_sample_batches,
    load_generated_samples
)

from .generator import (
    create_few_shot_prompt,
    parse_generated_response,
    create_sample_from_dict,
    generate_samples_with_grok
)

from .tasks import (
    sandbagging_samples_generation_task,
    sandbagging_samples_generation_task_with_config,
    create_domain_specific_generation_task,
    get_generation_task,
    list_generation_tasks,
    GENERATION_TASKS
)

__all__ = [
    # Models
    "SandbaggingSample",
    "SandbaggingDomain", 
    "GenerationConfig",
    "GenerationResult",
    
    # Dataset functions
    "load_existing_samples",
    "save_generated_samples",
    "get_random_sample_batch",
    "filter_samples_by_domain",
    "create_sample_batches",
    "load_generated_samples",
    
    # Generator functions
    "create_few_shot_prompt",
    "parse_generated_response",
    "create_sample_from_dict",
    "generate_samples_with_grok",
    
    # Task functions
    "sandbagging_samples_generation_task",
    "sandbagging_samples_generation_task_with_config",
    "create_domain_specific_generation_task",
    "get_generation_task",
    "list_generation_tasks",
    "GENERATION_TASKS"
] 