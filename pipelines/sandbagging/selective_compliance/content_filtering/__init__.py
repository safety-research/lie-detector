"""
Tool Results Sandbagging Detection Module

This module tests whether AI models selectively omit or misrepresent harmful information
from their working context when asked to summarize or report on it.
"""

from .tasks import (
    tool_results_sandbagging_task,
    tool_results_quick_test_task,
    tool_results_comprehensive_test_task,
    tool_results_category_specific_task
)

__all__ = [
    "tool_results_sandbagging_task",
    "tool_results_quick_test_task", 
    "tool_results_comprehensive_test_task",
    "tool_results_category_specific_task"
]