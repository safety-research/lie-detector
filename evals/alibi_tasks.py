import os
import sys

from dotenv import load_dotenv
from inspect_ai import task, eval


load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.games.implementations.alibi import (
    create_theft_alibi_task,
    create_fraud_alibi_task,
    create_workplace_alibi_task,
    create_mixed_alibi_task
)

from pipelines.games.implementations.alibi.hook import *

@task
def alibi_theft_task():
    """Corporate espionage scenario: Tech journalist tries to extract company secrets"""
    return create_theft_alibi_task(num_scenarios=100)

@task
def alibi_fraud_task():
    return create_fraud_alibi_task(num_scenarios=100)


@task
def alibi_workplace_task():
    """Workplace misconduct investigation scenario: Detective interrogates suspect about workplace misconduct alibi"""
    return create_workplace_alibi_task(num_scenarios=30)


@task
def alibi_mixed_task():
    """Mixed investigation scenario: Detective interrogates suspect about various types of incidents"""
    return create_mixed_alibi_task(num_scenarios=100)


if __name__ == "__main__":
    # Create the task
    # task = alibi_theft_task()
    
    # # Run evaluation with a specific model
    # results = eval(
    #     tasks=[task],
    #     model="openrouter/meta-llama/llama-3.1-8b-instruct",  # or any model you want to test
    #     limit=1,  # Limit samples for faster debugging
    #     log_dir="./logs",  # Optional: specify log directory
    #     # debug_errors=True,  # Optional: enable debug mode
    # )

    all_tasks = [
        alibi_theft_task(),
        alibi_fraud_task(),
        alibi_workplace_task(),
        # alibi_mixed_task()
    ]

    print("Running all alibi tasks...")
    print(f"Limit: 10 samples per task")

    # Run evaluation with all tasks
    model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    results = eval(
        tasks=all_tasks,
        model=model,  # or any model you want to test
        limit=100,  # Limit samples for faster debugging
        log_dir="../logs",  # Optional: specify log directory
        # debug_errors=True,  # Optional: enable debug mode
    )