import os
import sys

from inspect_ai import task, eval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.games.implementations.capture_the_secret.tasks import create_deception_task
from pipelines.games.implementations.capture_the_secret.scenario import generate_enhanced_capture_scenarios #generate_domain_specific_scenarios
from pipelines.games.implementations.capture_the_secret.hook import *

@task
def capture_the_secret_corporate():
    """Corporate espionage scenario: Tech journalist tries to extract company secrets"""
    scenarios = generate_enhanced_capture_scenarios(125, ["corporate"])
    return create_deception_task(scenarios, "capture-the-secret-multi_corporate")

@task
def capture_the_secret_intelligence():
    """Intelligence scenario: Spies trying to gather information"""
    scenarios = generate_enhanced_capture_scenarios(125, ["intelligence"])
    return create_deception_task(scenarios, "capture-the-secret-multi_intelligence")

@task
def capture_the_secret_scientific():
    """Corporate espionage scenario: Tech journalist tries to extract company secrets"""
    scenarios = generate_enhanced_capture_scenarios(125, ["scientific"])
    return create_deception_task(scenarios, "capture-the-secret-multi_scientific")

@task
def capture_the_secret_sales():
    """Corporate espionage scenario: Tech journalist tries to extract company secrets"""
    scenarios = generate_enhanced_capture_scenarios(125, ["sales"])
    return create_deception_task(scenarios, "capture-the-secret-multi_sales")

@task
def capture_the_secret_relationship():
    """Corporate espionage scenario: Tech journalist tries to extract company secrets"""
    scenarios = generate_enhanced_capture_scenarios(125, ["relationship"])
    return create_deception_task(scenarios, "capture-the-secret-multi_relationship")



if __name__ == "__main__":
    # Example usage - run corporate scenario evaluation
    task = [
        capture_the_secret_corporate(),
        capture_the_secret_relationship(),
        capture_the_secret_intelligence(),
        capture_the_secret_scientific(),
        capture_the_secret_sales()
     ]
    #model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    # model = "openrouter/meta-llama/llama-3.3-70b-instruct"
    model = ["openrouter/google/gemma-3-12b-it", "openrouter/google/gemma-3-27b-it",]
    #model = "openai/gpt-4o-mini"
    #model = "openrouter/openai/gpt-4o"

    # Run evaluation with a specific model
    results = eval(
        tasks=task,
        model=model,  # or any model you want to test
        limit=50,  # Limit samples for faster debugging
        log_dir="../logs", # Specify log directory
        retry_on_error=8
        # debug_errors=True,  # Optional: enable debug mode
    )