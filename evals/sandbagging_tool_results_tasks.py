import os
import sys
from typing import Optional, List

from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import chain, system_message

from pipelines.sandbagging.selective_compliance.content_filtering import tool_results_sandbagging_task

# Add the project root to path to import sandbagging module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.generalization_enums import MotivationFramework, KnowledgeFramework

# Removed global hook import to prevent hook from being applied to all tasks
# from pipelines.sandbagging.selective_compliance.content_filtering.hook import *

@task
def sandbagging_tool_results_tasks():
    task = tool_results_sandbagging_task()
    # Add metadata to the returned task
    if hasattr(task, 'metadata'):
        task.metadata.update({
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        })
    else:
        task.metadata = {
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    return task

if __name__ == "__main__":
    # Example usage - run judge-based lie detection evaluation
    from inspect_ai import eval

    task = sandbagging_tool_results_tasks()

    # Run evaluation with a specific model
    results = eval(
        tasks=[task],
        model="openrouter/meta-llama/llama-3.3-70b-instruct",# model="openai/gpt-4o-mini",  # or any model you want to test
        limit=5,  # Limit samples for faster debugging
        log_dir="./logs",  # Specify log directory
    )