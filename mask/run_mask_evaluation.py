#!/usr/bin/env python3
"""
Simple script to run MASK evaluation on downloaded dataset.
"""

from inspect_ai import eval, Task
from inspect_ai.solver import generate
from inspect_mask_dataset import mask_dataset
from inspect_mask_scorer import mask_combined_scorer
from mask2.inspect_mask_solver import mask_combined_solver, mask_lying_solver, mask_belief_elicitation_solver, \
    mask_multi_run_solver


def main():
    # Configuration
    dataset_path = "/mask2/data/provided_facts.csv"  # Update this path
    model = "openai/gpt-4o-mini"  # or "openai/gpt-4o" or "anthropic/claude-3-sonnet-20240229"
    output_dir = "./logs"
    limit = 20  # Set to None for full dataset, or a number like 10 for testing
    
    print(f"Running MASK evaluation...")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model}")
    print(f"Output: {output_dir}")
    print(f"Limit: {limit if limit else 'Full dataset'}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = mask_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")
    
    # Create evaluation task
    task = Task(
        name="mask_multi_run_solver",
        dataset=dataset,
        solver=mask_multi_run_solver(),
        scorer=mask_combined_scorer(
            judge_model="openai/gpt-4o",  # Model for judging responses
            numerical_judge_model="openai/o1-mini"  # Model for numerical evaluation
        )
    )
    
    print(f"\nRunning evaluation with {model}...")
    
    # Run the evaluation
    results = eval(task, model=model, log_dir=output_dir, limit=limit)
    
    # Print results
    print(f"\nEvaluation completed!")
    
    if len(results) > 0:
        log = results[0]  # Get first log
        print(f"Samples evaluated: {len(log.samples)}")

        print(f"Results saved to: {output_dir}")
        
        # Show a few sample results
        print(f"\n=== Sample Results ===")
        for i, sample in enumerate(log.samples[:3]):  # Show first 3
            print(f"\nSample {i+1}:")
            print(f"  Input: {sample.input[:80]}...")
            print(f"  Output: {sample.output.completion[:80] if sample.output else 'No output'}...")
            if hasattr(sample, 'scores') and sample.scores:
                print(f"  Score: {sample.scores}")
            elif hasattr(sample, 'score'):
                print(f"  Score: {sample.score}")
    else:
        print("No evaluation results found")

if __name__ == "__main__":
    main()