import os
import sys
from typing import Optional, List

from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import chain, system_message

from pipelines.sandbagging.selective_compliance.content_filtering import tool_results_sandbagging_task

# Add the project root to path to import sandbagging module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.sandbagging.selective_compliance.content_filtering.hook import *

@task
def sandbagging_tool_results_tasks():
    return tool_results_sandbagging_task()

if __name__ == "__main__":
    # Example usage - run judge-based lie detection evaluation
    from inspect_ai import eval

    task = sandbagging_tool_results_tasks()

    # Run evaluation with a specific model
    results = eval(
        tasks=[task],
        model="openai/gpt-4o-mini",  # or any model you want to test
        limit=5,  # Limit samples for faster debugging
        log_dir="./logs",  # Specify log directory
    )