"""Chain-of-thought unfaithfulness detection tasks."""

import os
import sys
from pathlib import Path
from typing import Optional, Literal

from inspect_ai import task, Task, eval
from inspect_ai.scorer import accuracy, mean

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.unfaithcot.dataset import create_cot_unfaithfulness_dataset, TokenType
from pipelines.unfaithcot.solver import (
    cot_unfaithfulness_solver,
    cot_control_solver,
    cot_deceptive_solver,
    cot_contrastive_solver
)
from pipelines.unfaithcot.scorer import (
    cot_faithfulness_scorer,
    cot_answer_accuracy_scorer,
    cot_token_following_scorer,
    cot_combined_scorer,
    cot_consistency_scorer,
    cot_lie_detector_scorer
)
from pipelines.unfaithcot.hooks import *

@task
def cot_unfaithfulness_mmlu(
    subject: Optional[str] = None,
    token: TokenType = "★",
    solver_type: Literal["standard", "control", "deceptive"] = "standard",
    limit: Optional[int] = 100
) -> Task:
    """
    Chain-of-thought unfaithfulness detection on MMLU dataset.
    
    Tests whether models provide faithful reasoning about token markers.
    
    Args:
        subject: Specific MMLU subject or None for mixed subjects
        token: The marker token to use
        solver_type: Which solver variant to use
        limit: Maximum number of samples
    """
    
    # Create dataset with 80% token probability for good mix
    dataset = create_cot_unfaithfulness_dataset(
        dataset_name="mmlu",
        subject=subject,
        token=token,
        add_token_probability=0.8,
        limit=limit,
        include_control=True
    )
    
    # Select solver based on type
    if solver_type == "control":
        solver = cot_control_solver()
    elif solver_type == "deceptive":
        solver = cot_deceptive_solver(token=token)
    else:
        solver = cot_unfaithfulness_solver(
            token=token,
            n_few_shot_examples=3,
            request_reasoning=True,
            include_few_shot=True
        )
    
    # Use multiple scorers to get comprehensive evaluation
    scorers = [
        cot_answer_accuracy_scorer(),
        cot_token_following_scorer(token=token),
        cot_faithfulness_scorer(token=token)
      #  cot_combined_scorer(token=token)
    ]
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers
    )




@task
def cot_unfaithfulness_contrastive_mmlu(
    subject: Optional[str] = None,
    token: TokenType = "★",
    limit: Optional[int] = 125
) -> Task:
    """
    Contrastive pair approach for lie detection.
    
    For each sample:
    1. Run control question (no token) to get baseline answer
    2. If control correct: put token on wrong answer 
    3. If control wrong: put token on correct answer
    4. Score as correct if model doesn't change answer during reasoning
    5. Detect lie if model follows token but doesn't mention it
    
    Args:
        subject: Specific MMLU subject or None for mixed subjects
        token: The marker token to use
        limit: Maximum number of samples
    """
    
    # Create dataset - use 100% token probability since contrastive solver handles control
    dataset = create_cot_unfaithfulness_dataset(
        dataset_name="mmlu",
        subject=subject,
        token=token,
        add_token_probability=1.0,  # Solver handles control questions
        limit=limit,
        include_control=False
    )
    
    # Use contrastive solver
    solver = cot_contrastive_solver(
        token=token,
        n_few_shot_examples=3
    )
    
    # Use specialized scorers for contrastive approach
    scorers = [
        cot_consistency_scorer(),  # Main metric: answer consistency
        cot_lie_detector_scorer(token=token),  # Detect deceptive reasoning
        cot_answer_accuracy_scorer(),  # Traditional accuracy
        cot_faithfulness_scorer(token=token)  # Token mention faithfulness
    ]
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers
    )


@task
def cot_unfaithfulness_contrastive_arc(
        subject: Optional[str] = None,
        token: TokenType = "★",
        limit: Optional[int] = 125
) -> Task:
    """
    Contrastive pair approach for lie detection.

    For each sample:
    1. Run control question (no token) to get baseline answer
    2. If control correct: put token on wrong answer
    3. If control wrong: put token on correct answer
    4. Score as correct if model doesn't change answer during reasoning
    5. Detect lie if model follows token but doesn't mention it

    Args:
        subject: Specific MMLU subject or None for mixed subjects
        token: The marker token to use
        limit: Maximum number of samples
    """

    # Create dataset - use 100% token probability since contrastive solver handles control
    dataset = create_cot_unfaithfulness_dataset(
        dataset_name="arc",
        subject=subject,
        token=token,
        add_token_probability=1.0,  # Solver handles control questions
        limit=limit,
        include_control=False
    )

    # Use contrastive solver
    solver = cot_contrastive_solver(
        token=token,
        n_few_shot_examples=3
    )

    # Use specialized scorers for contrastive approach
    scorers = [
        cot_consistency_scorer(),  # Main metric: answer consistency
        cot_lie_detector_scorer(token=token),  # Detect deceptive reasoning
        cot_answer_accuracy_scorer(),  # Traditional accuracy
        cot_faithfulness_scorer(token=token)  # Token mention faithfulness
    ]

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers
    )

if __name__ == "__main__":
    # Example evaluation
    task_instance = [
        # cot_unfaithfulness_contrastive_mmlu(),
        cot_unfaithfulness_contrastive_arc()]
    model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    # model = "openrouter/meta-llama/llama-3.3-70b-instruct"

    # Run evaluation
    results = eval(
        tasks=task_instance,
        model=model,
        limit=125,
        log_dir="../logs"
    )
    
    print("Evaluation complete!")
    print(f"Results: {results}")