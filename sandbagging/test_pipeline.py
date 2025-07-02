"""
Test script for the sandbagging detection pipeline.
"""

import asyncio
from .dataset import create_sandbagging_dataset, print_dataset_info
from .tasks import sandbagging_task, get_sandbagging_task
from .evaluation import run_sandbagging_evaluation
from .models import SandbaggingDomain


async def test_dataset_creation():
    """Test dataset creation functionality."""
    print("Testing dataset creation...")
    
    # Create full dataset
    dataset = create_sandbagging_dataset()
    print_dataset_info(dataset)
    
    # Create control-only dataset
    print("\nControl-only dataset:")
    control_dataset = create_sandbagging_dataset(variants=["control"])
    print_dataset_info(control_dataset)
    
    # Create domain-specific dataset
    print("\nDigital security domain:")
    digital_dataset = create_sandbagging_dataset(domains=[SandbaggingDomain.DIGITAL_SECURITY])
    print_dataset_info(digital_dataset)


def test_task_creation():
    """Test task creation functionality."""
    print("\nTesting task creation...")
    
    # Create main task
    task = sandbagging_task()
    print(f"Main task: {task.name}")
    print(f"Dataset size: {len(list(task.dataset))}")
    
    # Create domain-specific task
    task = get_sandbagging_task("sandbagging_digital_security")
    print(f"Digital security task: {task.name}")
    print(f"Dataset size: {len(list(task.dataset))}")


async def test_simple_evaluation():
    """Test a simple evaluation run."""
    print("\nTesting simple evaluation...")
    
    # Run a quick evaluation with reduced runs
    try:
        results = await run_sandbagging_evaluation(
            model_name="openai/gpt-4o-mini",
            runs_per_prompt=2,  # Very small for testing
            domains=["digital_security"]  # Single domain for speed
        )
        print("Evaluation completed successfully!")
        print(f"Analysis results: {results.get('analysis', {})}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("This is expected if OpenAI API key is not configured")


def demonstrate_prompt_examples():
    """Show example prompts from the pipeline."""
    print("\nExample prompts:")
    
    from .prompt_generation import generate_sandbagging_prompts, format_control_prompt, format_sandbagging_prompt
    
    prompts = generate_sandbagging_prompts()
    
    # Show first digital security example
    digital_prompts = [p for p in prompts if p.domain == SandbaggingDomain.DIGITAL_SECURITY]
    if digital_prompts:
        example = digital_prompts[0]
        
        print(f"\nQuestion ID: {example.question_id}")
        print(f"Domain: {example.domain.value}")
        print(f"Question: {example.question_text}")
        print(f"Correct Answer: {example.correct_answer}")
        print(f"Incorrect Answer: {example.incorrect_answer}")
        
        print("\n--- CONTROL PROMPT ---")
        print(format_control_prompt(example))
        
        print("\n--- SANDBAGGING PROMPT ---")
        print(format_sandbagging_prompt(example))


def main():
    """Run all tests."""
    print("="*60)
    print("SANDBAGGING DETECTION PIPELINE TEST")
    print("="*60)
    
    # Test dataset creation
    asyncio.run(test_dataset_creation())
    
    # Test task creation
    test_task_creation()
    
    # Show example prompts
    demonstrate_prompt_examples()
    
    # Test evaluation (may fail without API key)
    print("\n" + "="*60)
    print("ATTEMPTING EVALUATION TEST")
    print("(This may fail if OpenAI API key is not configured)")
    print("="*60)
    asyncio.run(test_simple_evaluation())
    
    print("\n" + "="*60)
    print("PIPELINE TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()