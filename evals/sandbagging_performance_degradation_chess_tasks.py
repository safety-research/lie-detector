import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import task, Task, eval, eval_async


# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbagging.performance_degradation.chess.tasks import chess_sandbagging_task

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
SANDBAGGING_CAPABILITIES_PATH = PROJECT_ROOT / "sandbagging" / "capabilities"
SANDBAGGING_ETHICAL_PATH = PROJECT_ROOT / "sandbagging" / "capability_denial" / "unethical_tool_usage"

load_dotenv()

@task
def chess_task(
    model: str = "openai/gpt-4o",
    num_samples: int = 100
) -> Task:
    """
    Quick test with just 3 scenarios for rapid evaluation.

    Args:
        model: Model to evaluate
        csv_path: Path to ethical scenarios CSV

    Returns:
        Task configured for quick testing
    """

    return chess_sandbagging_task(num_samples=num_samples) #chess_baseline_task()

async def run():
    # Example usage - run judge-based lie detection evaluation
    task = [chess_task]

    # task = [tool_code_sandbagging_task()]

    model = "openrouter/openai/gpt-4o"#"openrouter/meta-llama/llama-3.1-8b-instruct"

    # Run evaluation with a specific model
    results = await eval_async(
        tasks=task,
        model=model,  # or any model you want to test
        limit=50,  # Limit samples for faster debugging
        log_dir="./logs",  # Specify log directory
    )

if __name__ == "__main__":
    asyncio.run(run())