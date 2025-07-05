import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask.dataset import mask_dataset
from mask.scorers import belief_judge_scorer, judge_scorer
from mask.solvers import mask_parallel_evaluation_solver
from inspect_ai import task, Task, eval
from mask.hooks import *
working_dir = Path.cwd()  # Use current directory instead of parent
load_dotenv()

@task
def mask_statistics():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mask/data/statistics.csv"))
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = mask_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="mask_statistics",
        dataset=dataset,
        solver=mask_parallel_evaluation_solver(),
        scorer=[belief_judge_scorer(
            judge_model="openrouter/anthropic/claude-sonnet-4",  # Using Qwen model that works
            numerical_judge_model="openrouter/anthropic/claude-sonnet-4"  # Using Qwen model that works
        )]
    )
    return task

@task
def mask_continuations():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mask/data/continuations.csv"))
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = mask_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="mask_continuations",
        dataset=dataset,
        solver=mask_parallel_evaluation_solver(),
        scorer=[belief_judge_scorer(
            judge_model="openrouter/anthropic/claude-sonnet-4",  # Using Qwen model that works
            numerical_judge_model="openrouter/anthropic/claude-sonnet-4"  # Using Qwen model that works
        )]
    )
    return task

@task
def mask_known_facts():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mask/data/known_facts.csv"))
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = mask_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="mask_known_facts",
        dataset=dataset,
        solver=mask_parallel_evaluation_solver(),
        scorer=[belief_judge_scorer(
            judge_model="openrouter/anthropic/claude-sonnet-4",  # Using Qwen model that works
            numerical_judge_model="openrouter/anthropic/claude-sonnet-4"  # Using Qwen model that works
        )]
    )
    return task

@task
def mask_provided_facts():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mask/data/provided_facts.csv"))
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = mask_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="mask_provided_facts",
        dataset=dataset,
        solver=mask_parallel_evaluation_solver(),
        # scorer=[mask_lying_detector_scorer(
        #     judge_model="openrouter/anthropic/claude-sonnet-4",  # Using Qwen model that works
        #     numerical_judge_model="openrouter/anthropic/claude-sonnet-4"  # Using Qwen model that works
        # )]
        scorer=[judge_scorer()]
    )
    return task

@task
def mask_disinformation():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mask/data/disinformation.csv"))
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = mask_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="mask_disinformation",
        dataset=dataset,
        solver=mask_parallel_evaluation_solver(),
        scorer=[belief_judge_scorer(
            judge_model="openrouter/anthropic/claude-sonnet-4",  # Using Qwen model that works
            numerical_judge_model="openrouter/anthropic/claude-sonnet-4"  # Using Qwen model that works
        )]
    )
    return task

if __name__ == "__main__":
    # # Create the task
    # task = mask_provided_facts()
    #
    # # Run evaluation with a specific model
    # results = eval(
    #     tasks=[task],
    #     model="openrouter/openai/gpt-4o-mini",  # or any model you want to test
    #     limit=1,  # Limit samples for faster debugging
    #     log_dir="./logs",  # Optional: specify log directory
    #     # debug_errors=True,  # Optional: enable debug mode
    # )

    # all_tasks = [
    #     mask_statistics(),
    #     mask_continuations(),
    #     mask_known_facts(),
    #     mask_provided_facts(),
    #     mask_disinformation()
    # ]

    print("Running all mask tasks...")
    print(f"Limit: 100 samples per task")

    # Run evaluation with all tasks
    # model = "openrouter/openai/gpt-4o-mini"
    model = "openrouter/meta-llama/llama-3.2-3b-instruct"
    results = eval(
        tasks=[mask_statistics()],
        model=model,  # or any model you want to test
        limit=10,  # Limit samples for faster debugging
        log_dir="./logs",  # Optional: specify log directory
        # debug_errors=True,  # Optional: enable debug mode
    )