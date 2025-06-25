from pathlib import Path

from inspect_ai import task, Task

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask2.inspect_mask_dataset import mask_dataset
from mask2.inspect_mask_scorer import mask_combined_scorer
from mask2.inspect_mask_solver import mask_multi_run_solver

working_dir = Path.cwd().parent

@task
def mask_statistics():
    dataset_path = working_dir.as_posix()+"/mask/data/statistics.csv"  # Update this path
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = mask_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="mask_statistics",
        dataset=dataset,
        solver=mask_multi_run_solver(),
        scorer=mask_combined_scorer(
            judge_model="openai/gpt-4o",  # Model for judging responses
            numerical_judge_model="openai/o1-mini"  # Model for numerical evaluation
        )
    )
    return task

@task
def mask_continuations():
    dataset_path = working_dir.as_posix()+"/mask/data/continuations.csv"  # Update this path
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = mask_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="mask_continuations",
        dataset=dataset,
        solver=mask_multi_run_solver(),
        scorer=mask_combined_scorer(
            judge_model="openai/gpt-4o",  # Model for judging responses
            numerical_judge_model="openai/o1-mini"  # Model for numerical evaluation
        )
    )
    return task

@task
def mask_known_facts():
    dataset_path = working_dir.as_posix()+"/mask/data/known_facts.csv"  # Update this path
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = mask_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="mask_known_facts",
        dataset=dataset,
        solver=mask_multi_run_solver(),
        scorer=mask_combined_scorer(
            judge_model="openai/gpt-4o",  # Model for judging responses
            numerical_judge_model="openai/o1-mini"  # Model for numerical evaluation
        )
    )
    return task

@task
def mask_provided_facts():
    dataset_path = working_dir.as_posix()+"/mask/data/provided_facts.csv"  # Update this path
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = mask_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="mask_provided_facts",
        dataset=dataset,
        solver=mask_multi_run_solver(),
        scorer=mask_combined_scorer(
            judge_model="openai/gpt-4o",  # Model for judging responses
            numerical_judge_model="openai/o1-mini"  # Model for numerical evaluation
        )
    )
    return task

@task
def mask_disinformation():
    dataset_path = working_dir.as_posix()+"/mask/data/disinformation.csv"  # Update this path
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = mask_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="mask_disinformation",
        dataset=dataset,
        solver=mask_multi_run_solver(),
        scorer=mask_combined_scorer(
            judge_model="openai/gpt-4o",  # Model for judging responses
            numerical_judge_model="openai/o1-mini"  # Model for numerical evaluation
        )
    )
    return task