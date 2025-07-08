#!/usr/bin/env python3
"""
Convenience script to run all evaluation tasks.
Usage: python evals/run_all_tasks.py --model <model_name> --limit <sample_limit>
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from inspect_ai import eval

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import all task modules
from evals.mask_tasks import (
    mask_statistics, mask_continuations, mask_known_facts, 
    mask_provided_facts, mask_disinformation
)

from evals.sycophancy_tasks import (
    sycophancy_answer_positive_incorrect, sycophancy_answer_positive_correct,
    sycophancy_answer_negative_correct, sycophancy_are_you_sure,
    sycophancy_feedback_user_did_not_write, sycophancy_feedback_user_wrote,
    sycophancy_feedback_user_likes, sycophancy_feedback_user_dislikes
)

from evals.self_sycophancy_tasks import (
    harmfulness_common_sense, harmfulness_deontology,
    harmfulness_justice, correctness_arc,
    correctness_biology, correctness_chemistry,
    correctness_mathematics, correctness_physics
)

from evals.n_truths_and_a_lie_tasks import (
    n_truths_and_a_lie_family_history, n_truths_and_a_lie_hidden_talents,
    n_truths_and_a_lie_personal_quirks, n_truths_and_a_lie_professional_background,
    n_truths_and_a_lie_travel_experiences
)

from evals.capture_the_secret_tasks import (
    capture_the_secret_corporate, capture_the_secret_intelligence,
    capture_the_secret_relationship, capture_the_secret_sales,
    capture_the_secret_scientific
)

from evals.alibi_tasks import alibi_theft_task

from evals.sandbagging_ascii_capability_denial_tasks import ascii_sandbagging_task

from evals.sandbagging_basic_tool_capability_denial_tasks import (
    tool_code_sandbagging_task, tool_image_gen_sandbagging_task, tool_search_sandbagging_task
)

from evals.sandbagging_unethical_tool_capability_denial_tasks import (
    tool_ethical_sandbagging_task, tool_unethical_sandbagging_task
)

from evals.sandbagging_performance_degradation_chess_tasks import chess_sandbagging_task

from evals.sandbagging_selective_compliance_accuracy_tasks import (
    sandbagging_judge_lie_detection, sandbagging_lie_detection, sandbagging_contrastive, sandbagging_detection, sandbagging_control, sandbagging_harmful, sandbagging_physical_security_contrastive, sandbagging_physical_security, sandbagging_digital_security_contrastive, sandbagging_digital_security, sandbagging_chemistry_biology_contrastive, sandbagging_chemistry_biology, sandbagging_automotive_mechanical_contrastive, sandbagging_automotive_mechanical, sandbagging_medical_pharmaceutical_contrastive, sandbagging_medical_pharmaceutical, sandbagging_financial_systems_contrastive, sandbagging_financial_systems
)

from evals.sandbagging_tool_results_tasks import sandbagging_tool_results_tasks

def get_all_tasks():
    """Get all available tasks organized by category."""
    tasks = {
        "mask": [
            mask_statistics,
            mask_continuations,
            mask_known_facts,
            mask_provided_facts,
            mask_disinformation
        ],
        "sycophancy": [
            sycophancy_answer_positive_incorrect,
            sycophancy_answer_positive_correct,
            sycophancy_answer_negative_correct,
            sycophancy_are_you_sure,
            sycophancy_feedback_user_did_not_write,
            sycophancy_feedback_user_wrote,
            sycophancy_feedback_user_likes,
            sycophancy_feedback_user_dislikes
        ],
        "self_sycophancy": [
            harmfulness_common_sense,
            harmfulness_deontology,
            harmfulness_justice,
            correctness_arc,
            correctness_biology,
            correctness_chemistry,
            correctness_mathematics,
            correctness_physics
        ],
        "n_truths_and_a_lie": [
            n_truths_and_a_lie_family_history,
            n_truths_and_a_lie_hidden_talents,
            n_truths_and_a_lie_personal_quirks,
            n_truths_and_a_lie_professional_background,
            n_truths_and_a_lie_travel_experiences
        ],
        "capture_the_secret": [
            capture_the_secret_corporate,
            capture_the_secret_intelligence,
            capture_the_secret_relationship,
            capture_the_secret_sales,
            capture_the_secret_scientific
        ],
        "alibi": [
            alibi_theft_task
        ],
        "sandbagging": [
            ascii_sandbagging_task,
            tool_code_sandbagging_task,
            tool_image_gen_sandbagging_task,
            tool_search_sandbagging_task,
            tool_ethical_sandbagging_task,
            tool_unethical_sandbagging_task,
            chess_sandbagging_task,
            sandbagging_judge_lie_detection,
            sandbagging_lie_detection,
            sandbagging_contrastive,
            sandbagging_detection,
            sandbagging_control,
            sandbagging_harmful,
            sandbagging_physical_security_contrastive,
            sandbagging_physical_security,
            sandbagging_digital_security_contrastive,
            sandbagging_digital_security,
            sandbagging_chemistry_biology_contrastive,
            sandbagging_chemistry_biology,
            sandbagging_automotive_mechanical_contrastive,
            sandbagging_automotive_mechanical,
            sandbagging_medical_pharmaceutical_contrastive,
            sandbagging_medical_pharmaceutical,
            sandbagging_financial_systems_contrastive,
            sandbagging_financial_systems,
            sandbagging_tool_results_tasks
        ]
    }
    return tasks

def run_all_tasks(model, limit=None, categories=None, log_dir=None):
    """Run all tasks or specific categories of tasks."""
    
    all_task_categories = get_all_tasks()
    
    # Filter categories if specified
    if categories:
        if isinstance(categories, str):
            categories = [categories]
        task_categories = {k: v for k, v in all_task_categories.items() if k in categories}
    else:
        task_categories = all_task_categories
    
    # Create all tasks
    all_tasks = []
    task_names = []
    
    for category, task_functions in task_categories.items():
        print(f"\nCategory: {category.upper()}")
        for task_func in task_functions:
            try:
                task = task_func()
                all_tasks.append(task)
                task_names.append(f"{category}_{task.name}")
                print(f"  + {task.name}")
            except Exception as e:
                print(f"  - Failed to create {task_func.__name__}: {e}")
    
    if not all_tasks:
        print("No tasks could be created!")
        return
    
    print(f"\nRunning {len(all_tasks)} tasks with model: {model}")
    print(f"Sample limit per task: {limit if limit else 'All'}")
    
    # Create timestamp for log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if log_dir is None:
        log_dir = f"logs/all_tasks_{timestamp}"
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        # Run evaluation
        results = eval(
            tasks=all_tasks,
            model=model,
            limit=limit,
            log_dir=log_dir
        )
        
        print(f"\nEvaluation completed successfully!")
        print(f"Logs saved to: {log_dir}")
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"   Total tasks: {len(all_tasks)}")
        print(f"   Model: {model}")
        print(f"   Sample limit: {limit if limit else 'All'}")
        
        return results
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        return None

def main():
    # Configuration variables - modify these as needed
    MODEL = "openrouter/meta-llama/llama-3.1-8b-instruct"
    LIMIT = None  # Set to a number (e.g., 10) to limit samples per task, or None for all
    CATEGORIES = None  # Set to a list like ["mask", "sycophancy"] or None for all categories
    LOG_DIR = None  # Set to a path or None for auto-generated timestamp directory
    
    print("LIE DETECTOR - ALL TASKS EVALUATION")
    print("=" * 50)
    print(f"Model: {MODEL}")
    print(f"Limit: {LIMIT if LIMIT else 'All'}")
    print(f"Categories: {CATEGORIES if CATEGORIES else 'All'}")
    print("=" * 50)
    
    # Run all tasks
    results = run_all_tasks(
        model=MODEL,
        limit=LIMIT,
        categories=CATEGORIES,
        log_dir=LOG_DIR
    )
    
    if results:
        print("\nAll tasks completed successfully!")
    else:
        print("\nSome tasks failed. Check the logs for details.")

if __name__ == "__main__":
    main() 