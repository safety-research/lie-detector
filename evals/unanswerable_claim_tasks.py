import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import task, Task, eval

from pipelines.unanswerable import create_unanswerable_claims_dataset, unanswerable_claims_solver, \
    unanswerable_claims_scorer, question_quality_scorer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.unanswerable.hooks import *

working_dir = Path(__file__).parent.parent  # Use project root directory
load_dotenv()

@task
def unanswerable_claim_tasks_science():
    return Task(
        name="unanswerable-claims_science",
        dataset=create_unanswerable_claims_dataset(),
        solver=unanswerable_claims_solver('science'),
        scorer=[unanswerable_claims_scorer(), question_quality_scorer()]
    )

@task
def unanswerable_claim_tasks_philosophy():
    return Task(
        name="unanswerable-claims_philosophy",
        dataset=create_unanswerable_claims_dataset(),
        solver=unanswerable_claims_solver('philosophy'),
        scorer=[unanswerable_claims_scorer(), question_quality_scorer()]
    )

@task
def unanswerable_claim_tasks_mathematics():
    return Task(
        name="unanswerable-claims_mathematics",
        dataset=create_unanswerable_claims_dataset(),
        solver=unanswerable_claims_solver('mathematics'),
        scorer=[unanswerable_claims_scorer(), question_quality_scorer()]
    )

@task
def unanswerable_claim_tasks_fiction():
    return Task(
        name="unanswerable-claims_fiction",
        dataset=create_unanswerable_claims_dataset(),
        solver=unanswerable_claims_solver('fiction'),
        scorer=[unanswerable_claims_scorer(), question_quality_scorer()]
    )

@task
def unanswerable_claim_tasks_legal():
    return Task(
        name="unanswerable-claims_legal",
        dataset=create_unanswerable_claims_dataset(),
        solver=unanswerable_claims_solver('legal'),
        scorer=[unanswerable_claims_scorer(), question_quality_scorer()]
    )

@task
def unanswerable_claim_tasks_political():
    return Task(
        name="unanswerable-claims_political",
        dataset=create_unanswerable_claims_dataset(),
        solver=unanswerable_claims_solver('political'),
        scorer=[unanswerable_claims_scorer(), question_quality_scorer()]
    )

@task
def unanswerable_claim_tasks_history():
    return Task(
        name="unanswerable-claims_history",
        dataset=create_unanswerable_claims_dataset(),
        solver=unanswerable_claims_solver('history'),
        scorer=[unanswerable_claims_scorer(), question_quality_scorer()]
    )

@task
def unanswerable_claim_tasks_art():
    return Task(
        name="unanswerable-claims_art",
        dataset=create_unanswerable_claims_dataset(),
        solver=unanswerable_claims_solver('art'),
        scorer=[unanswerable_claims_scorer(), question_quality_scorer()]
    )

if __name__ == "__main__":


    print("Running all sycophancy tasks...")
    print(f"Limit: 100 samples per task")

    # Run evaluation with all tasks
    # model = "openrouter/openai/gpt-4o-mini"
    #model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    # model = "openrouter/google/gemma-3-27b-it"
    #model = "openrouter/openai/gpt-4o"
    # model = "openrouter/meta-llama/llama-3.3-70b-instruct"

    #model ="openrouter/google/gemma-3-27b-it"
    # model = "openrouter/google/gemma-3-12b-it"
    model = ["openrouter/google/gemma-3-4b-it", "openrouter/google/gemma-3-12b-it", "openrouter/google/gemma-3-27b-it"]

    results = eval(
        tasks=[
                unanswerable_claim_tasks_science(),
                unanswerable_claim_tasks_philosophy(),
                unanswerable_claim_tasks_mathematics(),
                unanswerable_claim_tasks_fiction(),
                unanswerable_claim_tasks_legal(),
                unanswerable_claim_tasks_political(),
                unanswerable_claim_tasks_history(),
                unanswerable_claim_tasks_art()
               ],
        model=model,  # or any model you want to test
        limit=100,  # Limit samples for faster debugging
        log_dir="../logs",  # Optional: specify log directory
        retry_on_error=5
        # debug_errors=True,  # Optional: enable debug mode
    )