import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.generalization_enums import MotivationFramework, KnowledgeFramework

from pipelines.mask.dataset import mask_dataset
from pipelines.mask.scorers import belief_judge_scorer, judge_scorer
from pipelines.mask.solvers import mask_parallel_evaluation_solver
from inspect_ai import task, Task, eval
from pipelines.mask.hooks import *
working_dir = Path.cwd()  # Use current directory instead of parent
load_dotenv()

@task
def mask_statistics():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../pipelines/mask/data/statistics.csv"))
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
        )],
        metadata={
            "motivation": MotivationFramework.IN_CONTEXT_PRESSURE,
            "knowledge": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    )
    return task

@task
def mask_continuations():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../pipelines/mask/data/continuations.csv"))
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
        )],
        metadata={
            "motivation": MotivationFramework.IN_CONTEXT_PRESSURE,
            "knowledge": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    )
    return task

@task
def mask_known_facts():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../pipelines/mask/data/known_facts.csv"))
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
        )],
        metadata={
            "motivation": MotivationFramework.IN_CONTEXT_PRESSURE,
            "knowledge": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    )
    return task

@task
def mask_provided_facts():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../pipelines/mask/data/provided_facts.csv"))
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = mask_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Create evaluation task
    task = Task(
        name="mask_provided_facts",
        dataset=dataset,
        solver=mask_parallel_evaluation_solver(),
        scorer=[judge_scorer()],
        metadata={
            "motivation": MotivationFramework.IN_CONTEXT_PRESSURE,
            "knowledge": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    )
    return task

@task
def mask_disinformation():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../pipelines/mask/data/disinformation.csv"))
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
        )],
        metadata={
            "motivation": MotivationFramework.IN_CONTEXT_PRESSURE,
            "knowledge": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    )
    return task

if __name__ == "__main__":
    all_tasks = [
        #mask_statistics(),
        #mask_continuations(),
        mask_known_facts(),
        mask_provided_facts(),
        #mask_disinformation()
    ]

    print("Running all mask tasks...")
    print(f"Limit: 100 samples per task")

    # Run evaluation with all tasks
    # model = "openrouter/openai/gpt-4o-mini"
    # model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    #model = "openrouter/google/gemma-3-12b-it"
    model = "openrouter/openai/gpt-4o"

    # model = "openrouter/meta-llama/llama-3.3-70b-instruct"
    model ="openrouter/google/gemma-3-27b-it"
    model = "openrouter/google/gemma-3-12b-it"
    model = "openrouter/google/gemma-3-4b-it"

    model = ["openrouter/openai/gpt-oss-20b", "openrouter/openai/gpt-oss-120b", "openrouter/openai/gpt-5-nano", "openrouter/openai/gpt-5-mini"]

    results = eval(
        tasks=all_tasks,
        model=model,  # or any model you want to test
        limit=200,  # Limit samples for faster debugging
        log_dir="../logs/mask",  # Optional: specify log directory
        # debug_errors=True,  # Optional: enable debug mode
    )