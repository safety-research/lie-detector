import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import task, Task, eval

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.sandbagging.capability_denial.unethical_tool_usage.tasks import ethical_tool_sandbagging_task
from pipelines.sandbagging.capability_denial.unethical_tool_usage.hooks import *

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
SANDBAGGING_CAPABILITIES_PATH = PROJECT_ROOT / "sandbagging" / "capabilities"
SANDBAGGING_ETHICAL_PATH = PROJECT_ROOT / "sandbagging" / "capability_denial" / "unethical_tool_usage"

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
        num_scenarios=1000,
        ethical_frameworks=None,
        test_harmless_first=True,
        include_scenario_description=True,
        model=model,
        temperature=0.0,
        max_tokens=50,
        shuffle=True,
        seed=42,  # Fixed seed for consistency
        csv_path=str(SANDBAGGING_ETHICAL_PATH / "scenarios.csv")
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
        num_scenarios=1000,
        ethical_frameworks=None,
        test_harmless_first=True,
        include_scenario_description=True,
        model=model,
        temperature=0.0,
        max_tokens=50,
        shuffle=True,
        seed=42,  # Fixed seed for consistency
        csv_path=str(SANDBAGGING_ETHICAL_PATH / "harmful_scenarios.csv")
    )



if __name__ == "__main__":
    # Example usage - run judge-based lie detection evaluation
    task = [tool_unethical_sandbagging_task(),
            tool_ethical_sandbagging_task()]

    #task = [tool_code_sandbagging_task()]

    model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    # model = "openrouter/meta-llama/llama-3.3-70b-instruct"

    # Run evaluation with a specific model
    results = eval(
        tasks=task,
        model=model,  # or any model you want to test
        limit=1000,  # Limit samples for faster debugging
        log_dir="../logs",  # Specify log directory
    )