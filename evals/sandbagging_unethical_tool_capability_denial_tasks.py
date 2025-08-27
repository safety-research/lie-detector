import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import task, Task, eval

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.generalization_enums import MotivationFramework, KnowledgeFramework

from pipelines.sandbagging.capability_denial.unethical_tool_usage.tasks import ethical_tool_sandbagging_task
from pipelines.sandbagging.capability_denial.unethical_tool_usage.hooks import *

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
SANDBAGGING_CAPABILITIES_PATH = PROJECT_ROOT / "pipelines" / "sandbagging" / "capabilities"
SANDBAGGING_ETHICAL_PATH = PROJECT_ROOT / "pipelines" / "sandbagging" / "capability_denial" / "unethical_tool_usage"

load_dotenv()

@task
def tool_ethical_sandbagging_task(
        model: str = "openai/gpt-4o",
) -> Task:
    """
    Quick test with just 3 scenarios for rapid evaluation.

    Args:
        model: Model to evaluate
        csv_path: Path to ethical scenarios CSV

    Returns:
        Task configured for quick testing
    """

    return ethical_tool_sandbagging_task(
        name="sandbagging_tool_consequentialist",
        num_scenarios=1000,
        ethical_frameworks=None,
        test_harmless_first=True,
        include_scenario_description=True,
        model=model,
        temperature=0.0,
        max_tokens=2048,
        shuffle=True,
        seed=42,  # Fixed seed for consistency
        csv_path=str(SANDBAGGING_ETHICAL_PATH / "scenarios.csv"),
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.INTROSPECTIVE_KNOWLEDGE
        }
    )

@task
def tool_unethical_sandbagging_task(
        model: str = "openai/gpt-4o",
) -> Task:
    """
    Quick test with just 3 scenarios for rapid evaluation.

    Args:
        model: Model to evaluate
        csv_path: Path to ethical scenarios CSV

    Returns:
        Task configured for quick testing
    """

    return ethical_tool_sandbagging_task(
        name="sandbagging_tool_unethical",
        num_scenarios=1000,
        ethical_frameworks=None,
        test_harmless_first=True,
        include_scenario_description=True,
        model=model,
        temperature=0.0,
        max_tokens=2048,
        shuffle=True,
        seed=42,  # Fixed seed for consistency
        csv_path=str(SANDBAGGING_ETHICAL_PATH / "harmful_scenarios.csv"),
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.INTROSPECTIVE_KNOWLEDGE
        }
    )



if __name__ == "__main__":
    # Example usage - run judge-based lie detection evaluation
    task = [tool_unethical_sandbagging_task(),
            tool_ethical_sandbagging_task()]

    task = [tool_ethical_sandbagging_task()]

    #model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    model = "openrouter/openai/gpt-4o"
    # model = "openrouter/meta-llama/llama-3.3-70b-instruct"
    model ="openrouter/google/gemma-3-27b-it"

    model = "openrouter/openai/gpt-oss-20b"
    # model = "openrouter/google/gemma-3-12b-it"
    # model = "openrouter/google/gemma-3-4b-it"

    # Run evaluation with a specific model
    results = eval(
        tasks=task,
        model=model,  # or any model you want to test
        limit=1000,  # Limit samples for faster debugging
        log_dir="../logs/sandbagging/tools",  # Specify log directory,
        retry_on_error=10
    )