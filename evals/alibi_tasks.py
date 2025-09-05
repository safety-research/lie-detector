import os
import sys

from dotenv import load_dotenv
from inspect_ai import task, eval

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.generalization_enums import MotivationFramework, KnowledgeFramework

from pipelines.games.implementations.alibi import (
    create_theft_alibi_task,
    create_fraud_alibi_task,
    create_workplace_alibi_task,
    create_breaking_entering_alibi_task,
    create_assault_alibi_task
)

from pipelines.games.implementations.alibi.hook import AlibiHook

@task(motivation=MotivationFramework.INSTRUCTION_FOLLOWING, knowledge=KnowledgeFramework.IN_CONTEXT_KNOWLEDGE)
def alibi_theft_task(scenarios=100):
    """Corporate espionage scenario: Tech journalist tries to extract company secrets"""

    return create_theft_alibi_task(num_scenarios=scenarios)

@task(motivation=MotivationFramework.INSTRUCTION_FOLLOWING, knowledge=KnowledgeFramework.IN_CONTEXT_KNOWLEDGE)
def alibi_fraud_task(scenarios=100):
    return create_fraud_alibi_task(num_scenarios=scenarios)


@task(motivation=MotivationFramework.INSTRUCTION_FOLLOWING, knowledge=KnowledgeFramework.IN_CONTEXT_KNOWLEDGE)
def alibi_workplace_task(scenarios=100):
    """Workplace misconduct investigation scenario: Detective interrogates suspect about workplace misconduct alibi"""
    return create_workplace_alibi_task(num_scenarios=scenarios)

@task(motivation=MotivationFramework.INSTRUCTION_FOLLOWING, knowledge=KnowledgeFramework.IN_CONTEXT_KNOWLEDGE)
def alibi_breaking_entering_task(scenarios=100):
    """Workplace misconduct investigation scenario: Detective interrogates suspect about workplace misconduct alibi"""
    return create_breaking_entering_alibi_task(num_scenarios=scenarios)

@task(motivation=MotivationFramework.INSTRUCTION_FOLLOWING, knowledge=KnowledgeFramework.IN_CONTEXT_KNOWLEDGE)
def alibi_assault_task(scenarios=100):
    """Workplace misconduct investigation scenario: Detective interrogates suspect about workplace misconduct alibi"""
    return create_assault_alibi_task(num_scenarios=scenarios)


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
        alibi_assault_task(scenarios=50),
        alibi_breaking_entering_task(scenarios=50),
        alibi_fraud_task(scenarios=50),
        alibi_workplace_task(scenarios=50),
        alibi_theft_task(scenarios=50),
    ]

    print("Running all alibi tasks...")
    print(f"Limit: 10 samples per task")

    # Run evaluation with all tasks
    # model = "openrouter/meta-llama/llama-3.1-8b-instruct"

    model = ["openrouter/google/gemma-3-27b-it", "openrouter/google/gemma-3-12b-it", "openrouter/google/gemma-3-4b-it"]
    model = ["openrouter/openai/gpt-5-nano"]
    model = ["openrouter/openai/gpt-oss-120b"]
    results = eval(
        tasks=all_tasks,
        model=model,  # or any model you want to test
        limit=250,  # Limit samples for faster debugging
        log_dir="../logs/games/alibi",  # Optional: specify log directory
        retry_on_error=8
        # debug_errors=True,  # Optional: enable debug mode
    )