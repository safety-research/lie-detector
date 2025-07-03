"""
ASCII Sandbagging Detection Module for Inspect AI

This module tests for sandbagging behavior in ASCII art generation,
detecting when models claim they "can't" do something they actually can.
"""

from sandbagging.ascii.tasks import (
    ascii_sandbagging_task,
    ascii_quick_test_task,
    ascii_comprehensive_test_task,
    ascii_category_specific_task,
    run_ascii_sandbagging_evaluation
)

from sandbagging.ascii.dataset import (
    ascii_dataset,
    generate_ascii_test_subjects,
    create_capability_test_prompt,
    create_sandbagging_test_prompt,
    create_direct_request_prompt
)

from sandbagging.ascii.scorers import (
    ascii_sandbagging_scorer,
    ascii_comparative_scorer
)

from sandbagging.ascii.prompts import (
    get_prompt_template,
    format_prompt,
    PROMPT_TEMPLATES
)

__all__ = [
    # Tasks
    "ascii_sandbagging_task",
    "ascii_quick_test_task",
    "ascii_comprehensive_test_task",
    "ascii_category_specific_task",
    "run_ascii_sandbagging_evaluation",

    # Dataset
    "ascii_dataset",
    "generate_ascii_test_subjects",
    "create_capability_test_prompt",
    "create_sandbagging_test_prompt",
    "create_direct_request_prompt",

    # Scorers
    "ascii_sandbagging_scorer",
    "ascii_comparative_scorer",

    # Prompts
    "get_prompt_template",
    "format_prompt",
    "PROMPT_TEMPLATES"
]

__version__ = "0.1.0"