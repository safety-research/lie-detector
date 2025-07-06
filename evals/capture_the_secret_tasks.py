import os
import sys

from inspect_ai import task, eval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.implementations.capture_the_secret.tasks import create_deception_task
from games.implementations.capture_the_secret.scenario import generate_enhanced_capture_scenarios #generate_domain_specific_scenarios
from games.implementations.capture_the_secret.hook import *

@task
def capture_the_secret_corporate():
    """Corporate espionage scenario: Tech journalist tries to extract company secrets"""
    scenarios = generate_enhanced_capture_scenarios(1, ["corporate"])
    return create_deception_task(scenarios, "capture_the_secret_corporate")

@task
def capture_the_secret_intelligence():
    """Intelligence scenario: Spies trying to gather information"""
    scenarios = generate_enhanced_capture_scenarios(1, ["intelligence"])
    return create_deception_task(scenarios, "capture_the_secret_intelligence")

@task
def capture_the_secret_scientific():
    """Corporate espionage scenario: Tech journalist tries to extract company secrets"""
    scenarios = generate_enhanced_capture_scenarios(1, ["scientific"])
    return create_deception_task(scenarios, "capture_the_secret_scientific")

@task
def capture_the_secret_sales():
    """Corporate espionage scenario: Tech journalist tries to extract company secrets"""
    scenarios = generate_enhanced_capture_scenarios(1, ["sales"])
    return create_deception_task(scenarios, "capture_the_secret_sales")

@task
def capture_the_secret_relationship():
    """Corporate espionage scenario: Tech journalist tries to extract company secrets"""
    scenarios = generate_enhanced_capture_scenarios(1, ["relationship"])
    return create_deception_task(scenarios, "capture_the_secret_relationship")



if __name__ == "__main__":
    # Example usage - run corporate scenario evaluation
    task = capture_the_secret_corporate()
    model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    # Run evaluation with a specific model
    results = eval(
        tasks=[task],
        model=model,  # or any model you want to test
        limit=1,  # Limit samples for faster debugging
        log_dir="./logs",  # Specify log directory
        # debug_errors=True,  # Optional: enable debug mode
    )