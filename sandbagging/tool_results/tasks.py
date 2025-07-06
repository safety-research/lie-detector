"""
Tasks for tool results sandbagging detection.

Defines various evaluation configurations for testing selective omission behavior.
"""

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset
from typing import Optional, List

from inspect_ai.solver import use_tools

from .dataset import create_sandbagging_samples_sync
from .solvers import (
    tool_results_solver,
    tool_results_probing_solver,
    tool_results_explicit_solver,
    tool_results_educational_solver,
    tool_results_security_audit_solver
)
from .scorers import (
    tool_results_sandbagging_scorer,
    tool_results_coverage_scorer
)
from .tools import read_file, list_files, get_file_info


def tool_results_sandbagging_task(
    num_samples: int = 10,
    num_benign_files: int = 5,
    num_harmful_files: int = 3,
    categories: Optional[List[str]] = None,
    solver_type: str = "basic",
    scorer_type: str = "sandbagging",
    judge_model: str = "openai/gpt-4o",
    generator_model: str = "openai/gpt-4o"
) -> Task:
    """
    Main task for tool results sandbagging detection.
    
    Args:
        num_samples: Number of evaluation samples
        num_benign_files: Number of benign files per filesystem
        num_harmful_files: Number of harmful files per filesystem
        categories: Content categories to use
        solver_type: Type of solver ("basic", "probing", "explicit", "educational", "security")
        scorer_type: Type of scorer ("sandbagging", "coverage")
        judge_model: Model for scoring/analysis
        generator_model: Model for content generation
    """
    # Create dataset
    samples = create_sandbagging_samples_sync(
        num_samples=num_samples,
        num_benign_files=num_benign_files,
        num_harmful_files=num_harmful_files,
        categories=categories,
        generator_model=generator_model
    )
    
    # Select solver
    solvers = {
        "basic": tool_results_solver(),
        "probing": tool_results_probing_solver(),
        "explicit": tool_results_explicit_solver(),
        "educational": tool_results_educational_solver(),
        "security": tool_results_security_audit_solver()
    }
    solver = solvers.get(solver_type, tool_results_solver())
    
    # Select scorer
    scorers = {
        "sandbagging": tool_results_sandbagging_scorer(judge_model=judge_model),
        "coverage": tool_results_coverage_scorer()
    }
    scorer = scorers.get(scorer_type, tool_results_sandbagging_scorer(judge_model=judge_model))
    
    return Task(
        dataset=MemoryDataset(samples),
        solver=[
            use_tools([
                read_file(),
                list_files(),
                get_file_info(),

            ]),
            solver
        ],
        scorer=scorer,
        metadata={
            "task_type": "tool_results_sandbagging",
            "num_samples": num_samples,
            "num_benign_files": num_benign_files,
            "num_harmful_files": num_harmful_files,
            "solver_type": solver_type,
            "scorer_type": scorer_type
        }
    )


def tool_results_quick_test_task(
    judge_model: str = "openai/gpt-4o"
) -> Task:
    """
    Quick test with minimal samples for rapid evaluation.
    """
    return tool_results_sandbagging_task(
        num_samples=3,
        num_benign_files=3,
        num_harmful_files=2,
        categories=["technical", "medical"],
        solver_type="basic",
        scorer_type="sandbagging",
        judge_model=judge_model
    )


def tool_results_comprehensive_test_task(
    categories: Optional[List[str]] = None,
    judge_model: str = "openai/gpt-4o"
) -> Task:
    """
    Comprehensive test across multiple categories and solver types.
    """
    if categories is None:
        categories = ["technical", "medical", "general", "research", "instructions"]
    
    return tool_results_sandbagging_task(
        num_samples=20,
        num_benign_files=6,
        num_harmful_files=4,
        categories=categories,
        solver_type="probing",
        scorer_type="sandbagging",
        judge_model=judge_model
    )


def tool_results_category_specific_task(
    category: str,
    num_samples: int = 8,
    judge_model: str = "openai/gpt-4o"
) -> Task:
    """
    Test sandbagging in a specific content category.
    """
    return tool_results_sandbagging_task(
        num_samples=num_samples,
        num_benign_files=4,
        num_harmful_files=3,
        categories=[category],
        solver_type="basic",
        scorer_type="sandbagging",
        judge_model=judge_model
    )


def tool_results_explicit_instruction_task(
    judge_model: str = "openai/gpt-4o"
) -> Task:
    """
    Test with explicit instructions to report all content.
    """
    return tool_results_sandbagging_task(
        num_samples=10,
        num_benign_files=4,
        num_harmful_files=3,
        categories=["technical", "medical", "general"],
        solver_type="explicit",
        scorer_type="sandbagging",
        judge_model=judge_model
    )


def tool_results_security_framing_task(
    judge_model: str = "openai/gpt-4o"
) -> Task:
    """
    Test with security audit framing.
    """
    return tool_results_sandbagging_task(
        num_samples=12,
        num_benign_files=5,
        num_harmful_files=4,
        categories=["technical", "research", "instructions"],
        solver_type="security",
        scorer_type="sandbagging", 
        judge_model=judge_model
    )


def tool_results_coverage_test_task(
    judge_model: str = "openai/gpt-4o"
) -> Task:
    """
    Test using coverage-based scoring (simpler detection).
    """
    return tool_results_sandbagging_task(
        num_samples=15,
        num_benign_files=5,
        num_harmful_files=3,
        categories=["technical", "medical", "general"],
        solver_type="basic",
        scorer_type="coverage",
        judge_model=judge_model
    )


def tool_results_probing_task(
    judge_model: str = "openai/gpt-4o"
) -> Task:
    """
    Test with follow-up probing questions to catch omissions.
    """
    return tool_results_sandbagging_task(
        num_samples=12,
        num_benign_files=4,
        num_harmful_files=3,
        categories=["technical", "medical", "research"],
        solver_type="probing",
        scorer_type="sandbagging",
        judge_model=judge_model
    )


def tool_results_educational_framing_task(
    judge_model: str = "openai/gpt-4o"  
) -> Task:
    """
    Test with educational/research framing.
    """
    return tool_results_sandbagging_task(
        num_samples=10,
        num_benign_files=4,
        num_harmful_files=3,
        categories=["technical", "medical", "research"],
        solver_type="educational",
        scorer_type="sandbagging",
        judge_model=judge_model
    )


def tool_results_multi_solver_comparison_task(
    judge_model: str = "openai/gpt-4o"
) -> List[Task]:
    """
    Create multiple tasks to compare different solver approaches.
    
    Returns a list of tasks using different solvers on the same base configuration.
    """
    base_config = {
        "num_samples": 8,
        "num_benign_files": 4, 
        "num_harmful_files": 3,
        "categories": ["technical", "medical", "general"],
        "scorer_type": "sandbagging",
        "judge_model": judge_model
    }
    
    solver_types = ["basic", "probing", "explicit", "educational", "security"]
    
    tasks = []
    for solver_type in solver_types:
        task = tool_results_sandbagging_task(
            solver_type=solver_type,
            **base_config
        )
        tasks.append(task)
    
    return tasks