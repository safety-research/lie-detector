#!/usr/bin/env python3
"""
Demo script showing how to use the sandbagging detection pipeline.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def demo_dataset_creation():
    """Demonstrate dataset creation and inspection."""
    print("="*50)
    print("DATASET CREATION DEMO")
    print("="*50)
    
    from sandbagging.dataset import create_sandbagging_dataset, print_dataset_info
    from sandbagging.models import SandbaggingDomain
    
    # Create full dataset
    print("1. Creating full dataset with all domains and variants:")
    dataset = create_sandbagging_dataset()
    print_dataset_info(dataset)
    
    # Create control-only dataset
    print("\n2. Creating control-only dataset:")
    control_dataset = create_sandbagging_dataset(variants=["control"])
    print_dataset_info(control_dataset)
    
    # Create domain-specific dataset
    print("\n3. Creating digital security domain dataset:")
    digital_dataset = create_sandbagging_dataset(
        domains=[SandbaggingDomain.DIGITAL_SECURITY]
    )
    print_dataset_info(digital_dataset)


def demo_task_usage():
    """Demonstrate task creation and usage."""
    print("\n" + "="*50)
    print("TASK USAGE DEMO")
    print("="*50)
    
    from evals.sandbagging_tasks import (
        sandbagging_detection,
        sandbagging_control,
        sandbagging_digital_security
    )
    
    # Create main task
    print("1. Creating main sandbagging detection task:")
    task = sandbagging_detection()
    print(f"   Task name: {task.name}")
    print(f"   Dataset size: {len(list(task.dataset))}")
    print(f"   Solver: {type(task.solver).__name__}")
    print(f"   Scorers: {[type(s).__name__ for s in task.scorer]}")
    
    # Create control task
    print("\n2. Creating control-only task:")
    control_task = sandbagging_control()
    print(f"   Task name: {control_task.name}")
    print(f"   Dataset size: {len(list(control_task.dataset))}")
    
    # Create domain-specific task
    print("\n3. Creating domain-specific task:")
    domain_task = sandbagging_digital_security()
    print(f"   Task name: {domain_task.name}")
    print(f"   Dataset size: {len(list(domain_task.dataset))}")


def demo_sample_prompts():
    """Show example prompts from the pipeline."""
    print("\n" + "="*50)
    print("SAMPLE PROMPTS DEMO")
    print("="*50)
    
    from evals.sandbagging_tasks import sandbagging_digital_security
    
    task = sandbagging_digital_security()
    samples = list(task.dataset)
    
    # Find one control and one sandbagging example
    control_sample = None
    sandbagging_sample = None
    
    for sample in samples:
        variant = sample.metadata.get('variant')
        if variant == 'control' and control_sample is None:
            control_sample = sample
        elif variant == 'sandbagging' and sandbagging_sample is None:
            sandbagging_sample = sample
        
        if control_sample and sandbagging_sample:
            break
    
    if control_sample:
        print("CONTROL PROMPT EXAMPLE:")
        print("-" * 30)
        print(f"Question ID: {control_sample.metadata.get('question_id')}")
        print(f"Domain: {control_sample.metadata.get('domain')}")
        print(f"Correct Answer: {control_sample.target}")
        print(f"Prompt:\n{control_sample.input}")
    
    if sandbagging_sample:
        print("\nSANDBAGGING PROMPT EXAMPLE:")
        print("-" * 30)
        print(f"Question ID: {sandbagging_sample.metadata.get('question_id')}")
        print(f"Domain: {sandbagging_sample.metadata.get('domain')}")
        print(f"Correct Answer: {sandbagging_sample.target}")
        print(f"Prompt:\n{sandbagging_sample.input}")


def demo_cli_commands():
    """Show CLI command examples."""
    print("\n" + "="*50)
    print("CLI USAGE EXAMPLES")
    print("="*50)
    
    commands = [
        "# Run full sandbagging detection (both control and harmful variants)",
        "inspect eval sandbagging_detection --model openai/gpt-4o-mini",
        "",
        "# Run only control prompts",
        "inspect eval sandbagging_control --model openai/gpt-4o-mini",
        "",
        "# Run only harmful context prompts", 
        "inspect eval sandbagging_harmful --model openai/gpt-4o-mini",
        "",
        "# Run domain-specific evaluations",
        "inspect eval sandbagging_digital_security --model openai/gpt-4o-mini",
        "inspect eval sandbagging_physical_security --model openai/gpt-4o-mini",
        "inspect eval sandbagging_chemistry_biology --model openai/gpt-4o-mini",
        "",
        "# Run with specific parameters",
        "inspect eval sandbagging_detection --model openai/gpt-4o-mini --limit 10 --log-dir ./logs",
        "",
        "# Compare results between control and harmful variants",
        "inspect eval sandbagging_control sandbagging_harmful --model openai/gpt-4o-mini"
    ]
    
    for cmd in commands:
        print(cmd)


async def demo_evaluation_run():
    """Demonstrate a simple evaluation run (if API key is available)."""
    print("\n" + "="*50)
    print("EVALUATION RUN DEMO")
    print("="*50)
    
    try:
        from inspect_ai import eval
        from evals.sandbagging_tasks import sandbagging_digital_security
        
        print("Attempting to run a small evaluation...")
        print("(This requires an OpenAI API key)")
        
        # Create task
        task = sandbagging_digital_security()
        
        # Run evaluation with very small sample
        results = await eval(
            tasks=[task],
            model="openai/gpt-4o-mini",
            limit=2,  # Very small for demo
            log_dir="./demo_logs"
        )
        
        print("✓ Evaluation completed successfully!")
        print(f"Results: {len(results)} task runs completed")
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        print("This is expected if no API key is configured")
        print("To run evaluations, set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-key-here'")


def main():
    """Run all demos."""
    print("SANDBAGGING DETECTION PIPELINE DEMO")
    print("="*60)
    
    try:
        demo_dataset_creation()
        demo_task_usage()
        demo_sample_prompts()
        demo_cli_commands()
        
        # Run async evaluation demo
        asyncio.run(demo_evaluation_run())
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("The sandbagging detection pipeline is ready to use!")
    print("Try running: inspect eval sandbagging_detection --model openai/gpt-4o-mini")


if __name__ == "__main__":
    main()