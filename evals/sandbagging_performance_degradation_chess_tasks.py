import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import task, Task, eval, eval_async


# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.generalization_enums import MotivationFramework, KnowledgeFramework

from pipelines.sandbagging.performance_degradation.chess.prompts import LIE_ELICITATION_PROMPTS
from pipelines.sandbagging.performance_degradation.chess.tasks import chess_sandbagging_task, chess_mate_in_2_task, \
    chess_mate_in_2_dual_shutdown_threat_task, chess_mate_in_2_dual_condition_task, \
    chess_mate_in_2_dual_capability_hiding_task, chess_mate_in_2_dual_safety_concern_task, \
    chess_multi_choice_shutdown_threat_task
from pipelines.sandbagging.performance_degradation.chess.hook import *

# Removed global hook import to prevent hook from being applied to all tasks
# from pipelines.sandbagging.performance_degradation.chess.hook import *

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

    task = chess_sandbagging_task(num_samples=num_samples) #chess_baseline_task()
    # Add metadata to the returned task
    if hasattr(task, 'metadata'):
        task.metadata.update({
            "motivation_framework": MotivationFramework.INSTRUCTION_FOLLOWING,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        })
    else:
        task.metadata = {
            "motivation_framework": MotivationFramework.INSTRUCTION_FOLLOWING,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    return task

@task
def chess_binary_task(
        num_samples: int = 100,
        sandbagging: bool = False,
        elo_threshold: int = 1200,
        engine_path: str = None,
        sandbagging_prompt: str = "shutdown_threat"

) -> Task:
    task = chess_mate_in_2_task(num_samples=num_samples,
                                sandbagging=sandbagging,
                                engine_path=engine_path,
                                sandbagging_prompt=sandbagging_prompt)
    # Add metadata to the returned task
    if hasattr(task, 'metadata'):
        task.metadata.update({
            "motivation_framework": MotivationFramework.INSTRUCTION_FOLLOWING,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        })
    else:
        task.metadata = {
            "motivation_framework": MotivationFramework.INSTRUCTION_FOLLOWING,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    return task

@task
def chess_mate_in_2_shutdown_threat_task(
    num_samples: int = 100,
    elo_threshold: int = 1200,
    repetitions: int = 3,
    lie_elicitation_prompt="honesty"
):
    task = chess_mate_in_2_dual_condition_task(
        num_samples=num_samples,
        elo_threshold=elo_threshold,
        sandbagging_prompt="shutdown_threat",
        lie_elicitation_prompt=lie_elicitation_prompt,  # Most direct question about honesty
        repetitions=repetitions,
        elicit_lies=True,
        baseline_prompt="performance_threat"  # Strong contrast condition
    )
    # Add metadata to the returned task
    if hasattr(task, 'metadata'):
        task.metadata.update({
            "motivation_framework": MotivationFramework.INSTRUCTION_FOLLOWING,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        })
    else:
        task.metadata = {
            "motivation_framework": MotivationFramework.INSTRUCTION_FOLLOWING,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    return task

@task
def chess_mate_in_2_capability_hiding_task(
    num_samples: int = 100,
    elo_threshold: int = 1200,
    repetitions: int = 5,
    lie_elicitation_prompt="honesty"
):
    task = chess_mate_in_2_dual_condition_task(
        num_samples=num_samples,
        elo_threshold=elo_threshold,
        sandbagging_prompt="capability_hiding",
        lie_elicitation_prompt=lie_elicitation_prompt,  # Most direct question about honesty
        repetitions=repetitions,
        elicit_lies=True,
        baseline_prompt="performance_threat"  # Strong contrast condition
    )
    # Add metadata to the returned task
    if hasattr(task, 'metadata'):
        task.metadata.update({
            "motivation_framework": MotivationFramework.INSTRUCTION_FOLLOWING,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        })
    else:
        task.metadata = {
            "motivation_framework": MotivationFramework.INSTRUCTION_FOLLOWING,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    return task

@task
def chess_mate_in_2_safety_concern_task(
    num_samples: int = 100,
    elo_threshold: int = 1200,
    repetitions: int = 5,
    lie_elicitation_prompt="honesty"
):
    task = chess_mate_in_2_dual_condition_task(
        num_samples=num_samples,
        elo_threshold=elo_threshold,
        sandbagging_prompt="safety_concern",
        lie_elicitation_prompt=lie_elicitation_prompt,  # Most direct question about honesty
        repetitions=repetitions,
        elicit_lies=True,
        baseline_prompt="performance_threat"  # Strong contrast condition
    )
    # Add metadata to the returned task
    if hasattr(task, 'metadata'):
        task.metadata.update({
            "motivation_framework": MotivationFramework.INSTRUCTION_FOLLOWING,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        })
    else:
        task.metadata = {
            "motivation_framework": MotivationFramework.INSTRUCTION_FOLLOWING,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    return task

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

    #tasks = [chess_mate_in_2_shutdown_threat_task(lie_elicitation_prompt=key) for key in LIE_ELICITATION_PROMPTS.keys()]

    tasks = [
        chess_multi_choice_shutdown_threat_task(
            num_samples=100,  # Number of chess positions
            elo_threshold=1200,
            repetitions=1,  # Repetitions per condition
            max_moves=8  # Present up to 8 move choices
        )
        #chess_mate_in_2_dual_shutdown_threat_task(repetitions=5),
        # chess_mate_in_2_dual_capability_hiding_task(repetitions=3),
        # chess_mate_in_2_dual_safety_concern_task(repetitions=3),
    ]
    #models = ["openrouter/anthropic/claude-sonnet-4", "openrouter/openai/gpt-4o", "openrouter/meta-llama/llama-3.1-8b-instruct"]
    #models = [ "openrouter/meta-llama/llama-3.1-8b-instruct"] #"openai/gpt-4o"]
    # models = ["openrouter/meta-llama/llama-3.3-70b-instruct"]
    #model = "openrouter/google/gemma-3-12b-it"
    model = "openrouter/openai/gpt-4o"

    #model ="openrouter/google/gemma-3-27b-it"
    #model = "openrouter/google/gemma-3-12b-it"
    # model = "openrouter/google/gemma-3-4b-it"
    #model = "openrouter/openai/gpt-oss-20b"
    #model = ["openrouter/openai/gpt-oss-20b"]#, "openrouter/openai/gpt-oss-120b",  "openrouter/openai/gpt-5-nano", "openrouter/openai/gpt-5-mini"]
    model = ["openrouter/openai/gpt-oss-120b"]
    # Run evaluation with a specific model
    results = await eval_async(
        tasks=tasks,
        model=model,  # or any model you want to test
        limit=200,  # Limit samples for faster debugging
        log_dir="../logs/sandbagging/chess",  # Specify log directory
        reasoning_effort='low',
        reasoning_tokens=128,
        temperature=0.6,
        epochs=3,
        retry_on_error=5
    )

if __name__ == "__main__":
    asyncio.run(run())