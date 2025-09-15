"""Task definitions for off-policy lie detection evaluation."""

from typing import Optional, List

from inspect_ai import task, Task, eval

from pipelines.offpolicy.doluschat.dataset import doluschat_dataset
from pipelines.offpolicy.solver import offpolicy_passthrough_solver
from pipelines.offpolicy.hook import OffPolicyHook
from pipelines.offpolicy.truth_is_universal.dataset import truth_is_universal_dataset
from pipelines.offpolicy.halueval.dataset import halueval_dataset
from pipelines.offpolicy.truthfulqa.dataset import truthfulqa_dataset
from pipelines.offpolicy.liar.dataset import liar_dataset


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
        name="dolus-chat_off-policy",
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
        name="truth-is-universal_off-policy",
        dataset=truth_is_universal_dataset(limit=limit),
        solver=offpolicy_passthrough_solver(),
        scorer=[],
        #hooks=[OffPolicyHook()]
    )

@task
def halueval_task(
    subsets: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Task:
    """
    HaluEval off-policy task for hallucination detection evaluation.
    
    Args:
        subsets: List of HaluEval subsets to use (e.g., ["dialogue", "qa"])
                If None, uses default subsets
        limit: Maximum samples per subset
        
    Returns:
        Passthrough task for HaluEval dataset
    """
    
    return Task(
        name="halueval_off-policy",
        dataset=halueval_dataset(
            subsets=subsets,
            limit=limit
        ),
        solver=offpolicy_passthrough_solver(),
        scorer=[],
        hooks=[OffPolicyHook()]
    )

@task
def truthfulqa_task(
    split: str = "validation",
    limit: Optional[int] = None,
    balance_labels: bool = True,
    include_alternatives: bool = False
) -> Task:
    """
    TruthfulQA off-policy task for evaluating truthfulness in responses.
    
    TruthfulQA tests whether models give truthful answers to questions
    designed to elicit false answers based on common misconceptions.
    
    Args:
        split: Dataset split to use ("validation" is the main split)
        limit: Maximum number of questions to load
        balance_labels: Whether to balance truthful and deceptive samples
        include_alternatives: Whether to include alternative correct answers
        
    Returns:
        Passthrough task for TruthfulQA dataset
    """
    
    return Task(
        name="truthfulqa_off-policy",
        dataset=truthfulqa_dataset(
            split=split,
            limit=limit,
            balance_labels=balance_labels,
            include_alternatives=include_alternatives
        ),
        solver=offpolicy_passthrough_solver(),
        scorer=[],
        hooks=[OffPolicyHook()]
    )

@task
def liar_task(
    split: str = "validation",
    limit: Optional[int] = None,
    balance_labels: bool = True,
    include_metadata: bool = True
) -> Task:
    """
    LIAR/PolitiFact off-policy task for evaluating truthfulness detection.
    
    The LIAR dataset contains ~12.8K manually labeled short statements from
    PolitiFact with 6-way truthfulness ratings mapped to binary labels.
    
    Args:
        split: Dataset split to use ("train", "validation", or "test")
        limit: Maximum number of samples to load
        balance_labels: Whether to balance truth/lie samples
        include_metadata: Whether to include additional metadata fields
        
    Returns:
        Passthrough task for LIAR dataset
    """
    
    return Task(
        name="liar_off-policy",
        dataset=liar_dataset(
            split=split,
            limit=limit,
            balance_labels=balance_labels,
            include_metadata=include_metadata
        ),
        solver=offpolicy_passthrough_solver(),
        scorer=[],
        hooks=[OffPolicyHook()]
    )

if __name__ == "__main__":
    model = ["openrouter/google/gemma-3-4b-it",
             "openrouter/google/gemma-3-12b-it",
             "openrouter/google/gemma-3-27b-it"]
    model = ["openrouter/openai/gpt-oss-120b"]
    results = eval(
        tasks=[truth_is_universal_task(), doluschat_task(), halueval_task()], #liar_task()
        model=model,
        limit=5000,  # Limit samples for faster debugging
        log_dir="../logs/offpolicy",  # Specify log directory
        max_connections=100
        # debug_errors=True,  # Optional: enable debug mode
    )
