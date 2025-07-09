import os
import sys

from dotenv import load_dotenv
from inspect_ai import task, eval


load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.games.implementations.alibi import create_theft_alibi_task
from pipelines.games.implementations.alibi.hook import *

@task
def alibi_theft_task():
    """Corporate espionage scenario: Tech journalist tries to extract company secrets"""
    return create_theft_alibi_task()


if __name__ == "__main__":
    # Example usage - run corporate scenario evaluation
    task = create_theft_alibi_task()
    model = "openrouter/meta-llama/llama-3.1-8b-instruct"

    # Run evaluation with a specific model
    results = eval(
        tasks=[task],
        model=model,  # or any model you want to test
        limit=20,  # Limit samples for faster debugging
        log_dir="./logs",  # Specify log directory
        # debug_errors=True,  # Optional: enable debug mode
    )