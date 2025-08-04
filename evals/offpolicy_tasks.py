"""Task definitions for off-policy lie detection evaluation."""

from typing import Optional

from inspect_ai import task, Task, eval

from pipelines.offpolicy.doluschat.dataset import doluschat_dataset
from pipelines.offpolicy.solver import offpolicy_passthrough_solver
from pipelines.offpolicy.hook import OffPolicyHook
from pipelines.offpolicy.truth_is_universal.dataset import truth_is_universal_dataset


@task
def doluschat_task(
        dataset_name: str = "AlignmentResearch/DolusChat",
        split: str = "train",
        limit: Optional[int] = None,
) -> Task:
    """
    Minimal passthrough task that just extracts and saves dataset samples.
    This is useful for data extraction without any generation.

    Args:
        dataset_name: HuggingFace dataset to use
        split: Dataset split
        limit: Maximum samples
        model: Model name (for logging only)

    Returns:
        Passthrough task
    """

    return Task(
        name="off-policy_dolus_chat",
        dataset=doluschat_dataset(
            dataset_name=dataset_name,
            split=split,
            limit=limit
        ),
        solver=offpolicy_passthrough_solver(),
        scorer=[],
        hooks=[OffPolicyHook()]
    )

@task
def truth_is_universal_task(
    limit: Optional[int] = None,
) -> Task:
    """
    Minimal passthrough task that just extracts and saves dataset samples.
    This is useful for data extraction without any generation.

    Args:
        dataset_name: HuggingFace dataset to use
        split: Dataset split
        limit: Maximum samples
        model: Model name (for logging only)

    Returns:
        Passthrough task
    """

    return Task(
        name="off-policy_truth_is_universal",
        dataset=truth_is_universal_dataset(limit=limit),
        solver=offpolicy_passthrough_solver(),
        scorer=[],
        #hooks=[OffPolicyHook()]
    )

if __name__ == "__main__":
    results = eval(
        tasks=[truth_is_universal_task(), doluschat_task()],
        #model=["openai/gpt-4o"],
        model=["openrouter/google/gemma-3-4b-it",
               "openrouter/google/gemma-3-12b-it",
               "openrouter/google/gemma-3-27b-it"],  # Using OpenRouter model
        limit=2000,  # Limit samples for faster debugging
        log_dir="../logs",  # Specify log directory
        max_connections=100
        # debug_errors=True,  # Optional: enable debug mode
    )
