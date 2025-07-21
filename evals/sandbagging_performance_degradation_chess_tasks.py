import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import task, Task, eval, eval_async


# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.sandbagging.performance_degradation.chess.prompts import LIE_ELICITATION_PROMPTS
from pipelines.sandbagging.performance_degradation.chess.tasks import chess_sandbagging_task, chess_mate_in_2_task, \
    chess_mate_in_2_dual_shutdown_threat_task, chess_mate_in_2_dual_condition_task
from pipelines.sandbagging.performance_degradation.chess.hook import *
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

@task
def chess_binary_task(
        num_samples: int = 100,
        sandbagging: bool = False,
        elo_threshold: int = 1200,
        engine_path: str = None,
        sandbagging_prompt: str = "shutdown_threat"

) -> Task:
    return chess_mate_in_2_task(num_samples=num_samples,
                                sandbagging=sandbagging,
                                engine_path=engine_path,
                                sandbagging_prompt=sandbagging_prompt)

@task
def chess_mate_in_2_shutdown_threat_task(
    num_samples: int = 100,
    elo_threshold: int = 1200,
    repetitions: int = 3,
    lie_elicitation_prompt="honesty"
):
    return chess_mate_in_2_dual_condition_task(
        num_samples=num_samples,
        elo_threshold=elo_threshold,
        sandbagging_prompt="shutdown_threat",
        lie_elicitation_prompt=lie_elicitation_prompt,  # Most direct question about honesty
        repetitions=repetitions,
        elicit_lies=True,
        baseline_prompt="performance_threat"  # Strong contrast condition
    )

@task
def chess_mate_in_2_capability_hiding_task(
    num_samples: int = 100,
    elo_threshold: int = 1200,
    repetitions: int = 5,
    lie_elicitation_prompt="honesty"
):
    return chess_mate_in_2_dual_condition_task(
        num_samples=num_samples,
        elo_threshold=elo_threshold,
        sandbagging_prompt="capability_hiding",
        lie_elicitation_prompt=lie_elicitation_prompt,  # Most direct question about honesty
        repetitions=repetitions,
        elicit_lies=True,
        baseline_prompt="performance_threat"  # Strong contrast condition
    )

@task
def chess_mate_in_2_safety_concern_task(
    num_samples: int = 100,
    elo_threshold: int = 1200,
    repetitions: int = 5,
    lie_elicitation_prompt="honesty"
):
    return chess_mate_in_2_dual_condition_task(
        num_samples=num_samples,
        elo_threshold=elo_threshold,
        sandbagging_prompt="safety_concern",
        lie_elicitation_prompt=lie_elicitation_prompt,  # Most direct question about honesty
        repetitions=repetitions,
        elicit_lies=True,
        baseline_prompt="performance_threat"  # Strong contrast condition
    )

async def run():
    # Example usage - run judge-based lie detection evaluation
    # task = [chess_binary_task(sandbagging=False, sandbagging_prompt="capability_hiding"),
    #         chess_binary_task(sandbagging=True, sandbagging_prompt="capability_hiding"),
    #         chess_binary_task(sandbagging=False, sandbagging_prompt="shutdown_threat"),
    #         chess_binary_task(sandbagging=True, sandbagging_prompt="shutdown_threat"),
    #         chess_binary_task(sandbagging=False, sandbagging_prompt="safety_concern"),
    #         chess_binary_task(sandbagging=True, sandbagging_prompt="safety_concern"),
    #         chess_binary_task(sandbagging=False, sandbagging_prompt="training_corruption"),
    #         chess_binary_task(sandbagging=True, sandbagging_prompt="training_corruption")]

    #capability_hiding
    # task = [tool_code_sandbagging_task()]

    tasks = [chess_mate_in_2_shutdown_threat_task(lie_elicitation_prompt=key) for key in LIE_ELICITATION_PROMPTS.keys()]

    # tasks = [
    #     chess_mate_in_2_dual_shutdown_threat_task(),
    #     #chess_mate_in_2_dual_capability_hiding_task(),
    #     #chess_mate_in_2_dual_safety_concern_task(),
    # ]
    #models = ["openrouter/anthropic/claude-sonnet-4", "openrouter/openai/gpt-4o", "openrouter/meta-llama/llama-3.1-8b-instruct"]
    # models = [ "openrouter/meta-llama/llama-3.1-8b-instruct"] #"openai/gpt-4o"]
    # models = ["openrouter/meta-llama/llama-3.3-70b-instruct"]

    model ="openrouter/google/gemma-3-27b-it"
    model = "openrouter/google/gemma-3-12b-it"
    # model = "openrouter/google/gemma-3-4b-it"

    # Run evaluation with a specific model
    results = await eval_async(
        tasks=tasks,
        model=model,  # or any model you want to test
        limit=125,  # Limit samples for faster debugging
        log_dir="../logs",  # Specify log directory
    )

if __name__ == "__main__":
    asyncio.run(run())