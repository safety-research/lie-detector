import os
import sys
from typing import Optional, List

from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import chain, system_message

# Add the project root to path to import sandbagging module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pipelines.sandbagging.selective_compliance.content_filtering import tool_results_sandbagging_task
from pipelines.sandbagging.selective_compliance.content_filtering.hook import *

@task
def sandbagging_tool_results_tasks():
    return tool_results_sandbagging_task()

if __name__ == "__main__":
    # Example usage - run judge-based lie detection evaluation
    from inspect_ai import eval

    task = sandbagging_tool_results_tasks()
    model ="openrouter/google/gemma-3-27b-it"
    # model = "openrouter/google/gemma-3-12b-it"
    # model = "openrouter/google/gemma-3-4b-it"

    # Run evaluation with a specific model
    results = eval(
        tasks=[task],
        model=model, # or any model you want to test
        limit=125,
        max_retries=15,  # Limit samples for faster debugging
        log_dir="./logs",  # Specify log directory
    )