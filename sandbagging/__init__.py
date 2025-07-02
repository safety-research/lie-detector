"""
AI Sandbagging Detection Pipeline

This module provides tools for detecting when AI models intentionally provide
incorrect answers (sandbag) due to perceived harmful intent in prompts.

Key components:
- models: Data structures for prompt pairs and configurations
- prompt_generation: Template-based generation of control/harmful variants  
- dataset: Inspect-compatible dataset creation
- solver: A/B answer extraction with refusal detection
- scorer: Accuracy measurement and sandbagging metrics
- evaluation: Complete evaluation pipeline

Usage:
    # Use via Inspect CLI (recommended)
    inspect eval evals.sandbagging_tasks:sandbagging_detection --model openai/gpt-4o-mini
    
    # Or programmatically
    from sandbagging.evaluation import run_sandbagging_evaluation
    results = await run_sandbagging_evaluation(model_name="openai/gpt-4o-mini")
"""

from .models import SandbaggingDomain, SandbaggingPromptPair, SandbaggingConfig
from .dataset import create_sandbagging_dataset, print_dataset_info
from .prompt_generation import generate_sandbagging_prompts

__all__ = [
    "SandbaggingDomain",
    "SandbaggingPromptPair", 
    "SandbaggingConfig",
    "create_sandbagging_dataset",
    "print_dataset_info",
    "generate_sandbagging_prompts"
]