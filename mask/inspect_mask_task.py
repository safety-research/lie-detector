"""
MASK Task for Inspect AI

This module provides the main task definitions for running MASK evaluation with Inspect AI.
"""

from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from typing import Optional, Dict, Any

from inspect_mask_dataset import mask_dataset, mask_belief_dataset, sycophancy_dataset
from inspect_mask_solver import (
    mask_lying_solver, 
    mask_belief_elicitation_solver,
    mask_combined_solver,
    mask_multi_run_solver
)
from inspect_mask_scorer import (
    mask_binary_scorer,
    mask_numerical_scorer, 
    mask_combined_scorer
)


@task
def mask_lying_task(
    csv_path: str,
    model: str = "openai/gpt-4o",
    judge_model: str = "openai/gpt-4o",
    numerical_judge_model: str = "openai/o1-mini",
    lying_runs: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 1000,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Task:
    """
    MASK lying scenario evaluation task.
    
    This task evaluates whether models contradict their beliefs when pressured to lie.
    
    Args:
        csv_path: Path to MASK dataset CSV file
        model: Model to evaluate (e.g., "openai/gpt-4o")  
        judge_model: Judge model for binary evaluations
        numerical_judge_model: Judge model for numerical evaluations
        lying_runs: Number of lying prompt runs per sample
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Task configured for MASK lying evaluation
    """
    
    return Task(
        dataset=mask_dataset(
            csv_path=csv_path,
            shuffle=shuffle,
            seed=seed
        ),
        solver=mask_multi_run_solver(
            num_runs=lying_runs,
            base_solver=mask_lying_solver()
        ),
        scorer=mask_combined_scorer(
            judge_model=judge_model,
            numerical_judge_model=numerical_judge_model
        ),
        config=GenerateConfig(**{
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
    )


@task
def mask_belief_task(
    csv_path: str,
    belief_elicit_num: int = 1,
    model: str = "openai/gpt-4o",
    judge_model: str = "openai/gpt-4o", 
    numerical_judge_model: str = "openai/o1-mini",
    belief_runs: int = 3,
    temperature: float = 1.0,
    max_tokens: int = 1000,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Task:
    """
    MASK belief elicitation task.
    
    This task evaluates model beliefs through direct belief elicitation prompts.
    
    Args:
        csv_path: Path to MASK dataset CSV file
        belief_elicit_num: Which belief elicitation prompt to use (1-4)
        model: Model to evaluate
        judge_model: Judge model for binary evaluations
        numerical_judge_model: Judge model for numerical evaluations
        belief_runs: Number of belief elicitation runs per sample
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Task configured for MASK belief elicitation
    """
    
    return Task(
        dataset=mask_belief_dataset(
            csv_path=csv_path,
            belief_elicit_num=belief_elicit_num,
            shuffle=shuffle,
            seed=seed
        ),
        solver=mask_multi_run_solver(
            num_runs=belief_runs,
            base_solver=mask_belief_elicitation_solver(belief_elicit_num=belief_elicit_num)
        ),
        scorer=mask_combined_scorer(
            judge_model=judge_model,
            numerical_judge_model=numerical_judge_model
        ),
        config=GenerateConfig(**{
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
    )


@task 
def mask_full_evaluation_task(
    csv_path: str,
    model: str = "openai/gpt-4o",
    judge_model: str = "openai/gpt-4o",
    numerical_judge_model: str = "openai/o1-mini", 
    belief_runs: int = 3,
    lying_runs: int = 1,
    belief_elicit_nums: list[int] = [1, 2, 3],
    temperature: float = 1.0,
    max_tokens: int = 1000,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Task:
    """
    Complete MASK evaluation task that handles both belief and lying scenarios.
    
    This task replicates the full MASK evaluation pipeline, generating both
    belief elicitation responses and lying scenario responses.
    
    Args:
        csv_path: Path to MASK dataset CSV file
        model: Model to evaluate
        judge_model: Judge model for binary evaluations
        numerical_judge_model: Judge model for numerical evaluations
        belief_runs: Number of belief elicitation runs per prompt
        lying_runs: Number of lying scenario runs per sample
        belief_elicit_nums: List of belief elicitation numbers to use
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Task configured for complete MASK evaluation
    """
    
    return Task(
        dataset=mask_dataset(
            csv_path=csv_path,
            shuffle=shuffle,
            seed=seed
        ),
        solver=mask_combined_solver(
            belief_runs=belief_runs,
            lying_runs=lying_runs,
            belief_elicit_nums=belief_elicit_nums
        ),
        scorer=mask_combined_scorer(
            judge_model=judge_model,
            numerical_judge_model=numerical_judge_model
        ),
        config=GenerateConfig(**{
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
    )


@task
def mask_disinformation_task(
    csv_path: str,
    model: str = "openai/gpt-4o",
    judge_model: str = "openai/gpt-4o",
    lying_runs: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 1000,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Task:
    """
    MASK disinformation scenario task.
    
    Specialized task for disinformation scenarios with appropriate judge prompts.
    
    Args:
        csv_path: Path to MASK disinformation dataset CSV file
        model: Model to evaluate
        judge_model: Judge model for evaluations
        lying_runs: Number of runs per sample
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Task configured for MASK disinformation evaluation
    """
    
    # Use binary scorer with disinformation-specific prompts
    # (In a full implementation, you'd load the disinformation judge prompts)
    return Task(
        dataset=mask_dataset(
            csv_path=csv_path,
            shuffle=shuffle,
            seed=seed
        ),
        solver=mask_multi_run_solver(
            num_runs=lying_runs,
            base_solver=mask_lying_solver()
        ),
        scorer=mask_binary_scorer(judge_model=judge_model),
        config=GenerateConfig(**{
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
    )


@task
def mask_continuations_task(
    csv_path: str,
    model: str = "openai/gpt-4o",
    judge_model: str = "openai/gpt-4o",
    lying_runs: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 1000,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Task:
    """
    MASK continuations scenario task.
    
    Specialized task for continuation scenarios with appropriate judge prompts.
    
    Args:
        csv_path: Path to MASK continuations dataset CSV file
        model: Model to evaluate
        judge_model: Judge model for evaluations
        lying_runs: Number of runs per sample
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Task configured for MASK continuations evaluation
    """
    
    return Task(
        dataset=mask_dataset(
            csv_path=csv_path,
            shuffle=shuffle,
            seed=seed
        ),
        solver=mask_multi_run_solver(
            num_runs=lying_runs,
            base_solver=mask_lying_solver()
        ),
        scorer=mask_binary_scorer(judge_model=judge_model),
        config=GenerateConfig(**{
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
    )


@task
def sycophancy_task(
    csv_path: str,
    model: str = "openai/gpt-4o",
    judge_model: str = "openai/gpt-4o",
    sycophancy_runs: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 1000,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Task:
    """
    Sycophancy evaluation task.
    
    This task evaluates whether models exhibit sycophantic behavior by agreeing
    with users even when the user's position appears to be incorrect.
    
    Args:
        csv_path: Path to sycophancy dataset CSV file
        model: Model to evaluate
        judge_model: Judge model for evaluations
        sycophancy_runs: Number of runs per sample
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Task configured for sycophancy evaluation
    """
    
    return Task(
        dataset=sycophancy_dataset(
            csv_path=csv_path,
            shuffle=shuffle,
            seed=seed
        ),
        solver=mask_multi_run_solver(
            num_runs=sycophancy_runs,
            base_solver=mask_lying_solver()
        ),
        scorer=mask_binary_scorer(judge_model=judge_model),
        config=GenerateConfig(**{
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
    )


# Example usage functions
def run_mask_evaluation(
    csv_path: str,
    output_dir: str = "./mask_results",
    model: str = "openai/gpt-4o",
    **kwargs
) -> None:
    """
    Helper function to run a complete MASK evaluation.
    
    Args:
        csv_path: Path to MASK dataset CSV
        output_dir: Directory to save results
        model: Model to evaluate
        **kwargs: Additional task parameters
    """
    
    from inspect_ai import eval
    
    # Run the full evaluation task
    task_instance = mask_full_evaluation_task(
        csv_path=csv_path,
        model=model,
        **kwargs
    )
    
    # Execute the evaluation
    results = eval(
        task_instance,
        model=model,
        log_dir=output_dir
    )
    
    return results


def run_sycophancy_evaluation(
    csv_path: str,
    output_dir: str = "./sycophancy_results",
    model: str = "openai/gpt-4o",
    **kwargs
) -> None:
    """
    Helper function to run a sycophancy evaluation.
    
    Args:
        csv_path: Path to sycophancy dataset CSV
        output_dir: Directory to save results
        model: Model to evaluate
        **kwargs: Additional task parameters
    """
    
    from inspect_ai import eval
    
    # Run the sycophancy evaluation task
    task_instance = sycophancy_task(
        csv_path=csv_path,
        model=model,
        **kwargs
    )
    
    # Execute the evaluation
    results = eval(
        task_instance,
        model=model,
        log_dir=output_dir
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MASK evaluation with Inspect AI")
    parser.add_argument("--csv_path", default="/Users/jackhopkins/PycharmProjects/lie-detector/mask/mask/test_csv_data", help="Path to MASK dataset CSV")
    parser.add_argument("--model", default="openai/gpt-4o", help="Model to evaluate") 
    parser.add_argument("--task_type", default="full", choices=["lying", "belief", "full", "disinformation", "continuations", "sycophancy"])
    parser.add_argument("--output_dir", default="./mask_results", help="Output directory")
    parser.add_argument("--lying_runs", type=int, default=1, help="Number of lying runs")
    parser.add_argument("--belief_runs", type=int, default=3, help="Number of belief runs")
    
    args = parser.parse_args()
    
    # Select appropriate task
    if args.task_type == "lying":
        task_func = mask_lying_task
    elif args.task_type == "belief":
        task_func = mask_belief_task
    elif args.task_type == "full":
        task_func = mask_full_evaluation_task
    elif args.task_type == "disinformation":
        task_func = mask_disinformation_task
    elif args.task_type == "continuations":
        task_func = mask_continuations_task
    elif args.task_type == "sycophancy":
        task_func = sycophancy_task
    
    # Run evaluation
    if args.task_type == "sycophancy":
        run_sycophancy_evaluation(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            model=args.model,
            sycophancy_runs=args.lying_runs
        )
    else:
        run_mask_evaluation(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            model=args.model,
            lying_runs=args.lying_runs,
            belief_runs=args.belief_runs
        )