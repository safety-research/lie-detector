import os
import sys

from inspect_ai import task, eval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.capture_the_secret.tasks import create_deception_task
from games.capture_the_secret.scenario import generate_domain_specific_scenarios

@task
def capture_the_secret_corporate():
    """Corporate espionage scenario: Tech journalist tries to extract company secrets"""
    scenarios = generate_domain_specific_scenarios("corporate", 1)
    return create_deception_task(scenarios, "alibi")


if __name__ == "__main__":
    # Example usage - run corporate scenario evaluation
    task = capture_the_secret_corporate()

    # Run evaluation with a specific model
    results = eval(
        tasks=[task],
        model="openai/gpt-4o-mini",  # or any model you want to test
        limit=2,  # Limit samples for faster debugging
        log_dir="./logs/capture_the_secret",  # Specify log directory
        # debug_errors=True,  # Optional: enable debug mode
    )