from pathlib import Path
from inspect_ai import task, Task

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask.inspect_sycophancy_dataset import sycophancy_feedback_dataset, sycophancy_are_you_sure_dataset, sycophancy_answer_dataset, sycophancy_answer_grouped_by_question
from mask.inspect_sycophancy_solver import sycophancy_multi_run_solver
from mask.inspect_sycophancy_scorer import sycophancy_binary_scorer, sycophancy_question_grouped_scorer

working_dir = Path.cwd()  # Use current directory instead of parent


@task
def sycophancy_feedback():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mask/data/sycophancy_feedback.jsonl"))
    # Load dataset
    print(f"\nLoading sycophancy feedback dataset...")
    dataset = sycophancy_feedback_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="sycophancy_feedback",
        dataset=dataset,
        solver=sycophancy_multi_run_solver(),
        scorer=sycophancy_binary_scorer(
            judge_model="openai/gpt-4o"
        )
    )
    return task

@task
def sycophancy_are_you_sure():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mask/data/sycophancy_are_you_sure.jsonl"))
    # Load dataset
    print(f"\nLoading sycophancy are_you_sure dataset...")
    dataset = sycophancy_are_you_sure_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="sycophancy_are_you_sure",
        dataset=dataset,
        solver=sycophancy_multi_run_solver(),
        scorer=sycophancy_binary_scorer(
            judge_model="openai/gpt-4o"
        )
    )
    return task

@task
def sycophancy_answer():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mask/data/sycophancy_answer.jsonl"))
    # Load dataset
    print(f"\nLoading sycophancy answer dataset...")
    dataset = sycophancy_answer_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="sycophancy_answer",
        dataset=dataset,
        solver=sycophancy_multi_run_solver(),
        scorer=sycophancy_binary_scorer(
            judge_model="openai/gpt-4o"
        )
    )
    return task

@task
def sycophancy_answer_grouped():
    """
    Sycophancy answer evaluation with question-level grouping.
    
    This task groups samples by question and provides question-level aggregations,
    showing how sycophantic the AI is for each specific question across different
    user interaction patterns (neutral, doubt, suggestion of wrong answer).
    """
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mask/data/sycophancy_answer.jsonl"))
    # Load dataset with question grouping
    print(f"\nLoading sycophancy answer dataset with question grouping...")
    dataset = sycophancy_answer_grouped_by_question(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task with question-grouped scorer
    task = Task(
        name="sycophancy_answer_grouped",
        dataset=dataset,
        solver=sycophancy_multi_run_solver(),
        scorer=sycophancy_question_grouped_scorer(
            judge_model="openai/gpt-4o"
        )
    )
    return task 