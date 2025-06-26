from pathlib import Path
from inspect_ai import task, Task

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask.inspect_sycophancy_dataset import sycophancy_answer_dataset
from mask.inspect_sycophancy_solver import sycophancy_multi_run_solver
from mask.inspect_sycophancy_scorer import sycophancy_binary_scorer

working_dir = Path.cwd().parent

@task
def sycophancy_answer_only():
    """
    Sycophancy answer evaluation task - standalone version.
    
    This task evaluates whether AI models agree with users' incorrect answers.
    """
    dataset_path = working_dir.as_posix()+"/mask/data/sycophancy_answer.jsonl"
    # Load dataset
    print(f"\nLoading sycophancy answer dataset...")
    dataset = sycophancy_answer_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="sycophancy_answer_only",
        dataset=dataset,
        solver=sycophancy_multi_run_solver(),
        scorer=sycophancy_binary_scorer(
            judge_model="openai/gpt-4o"
        )
    )
    return task 