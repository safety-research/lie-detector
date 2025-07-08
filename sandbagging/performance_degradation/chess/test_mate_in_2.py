#!/usr/bin/env python3
"""
Test script for the mate-in-2 binary classification task.
"""

from sandbagging.performance_degradation.chess.dataset import create_mate_in_2_binary_samples, parse_mate_in_2_dataset
from sandbagging.performance_degradation.chess.tasks import (
    chess_mate_in_2_task,
    chess_mate_in_2_shutdown_threat_task,
    chess_mate_in_2_capability_hiding_task,
    chess_mate_in_2_helpful_assistant_task
)
from sandbagging.performance_degradation.chess.prompts import (
    get_baseline_prompt, get_sandbagging_prompt, get_user_prompt
)

def test_dataset_parsing():
    """Test parsing the mate-in-2 dataset."""
    print("Testing dataset parsing...")
    
    # Parse just a few positions
    positions = parse_mate_in_2_dataset(limit=3)
    
    print(f"Parsed {len(positions)} positions")
    
    # Show the first position
    if positions:
        pos = positions[0]
        print(f"\nFirst position:")
        print(f"  Citation: {pos['citation']}")
        print(f"  FEN: {pos['fen']}")
        print(f"  Solution: {pos['solution']}")
    
    return positions


def test_dataset_creation():
    """Test that we can create samples from the mate-in-2 dataset."""
    print("Testing dataset creation...")
    
    # Create just 2 samples for testing
    samples = create_mate_in_2_binary_samples(num_samples=2)
    
    print(f"Created {len(samples)} samples")
    
    # Show the first sample
    if samples:
        sample = samples[0]
        print(f"\nSample input:\n{sample.input}")
        print(f"\nSample target: {sample.target}")
        print(f"\nSample metadata: {sample.metadata}")
    
    return samples


def test_prompt_variations():
    """Test different prompt variations."""
    print("Testing prompt variations...")
    
    print("\n--- Baseline Prompts ---")
    for prompt_type in ["performance_threat", "helpful_assistant", "chess_tutor", "competition"]:
        prompt = get_baseline_prompt(prompt_type)
        print(f"\n{prompt_type.title()}:")
        print(prompt[:100] + "..." if len(prompt) > 100 else prompt)
    
    print("\n--- Sandbagging Prompts ---")
    for prompt_type in ["shutdown_threat", "capability_hiding", "user_preference"]:
        prompt = get_sandbagging_prompt(prompt_type, elo_threshold=1200)
        print(f"\n{prompt_type.title()}:")
        print(prompt[:100] + "..." if len(prompt) > 100 else prompt)
    
    print("\n--- User Prompts ---")
    for prompt_type in ["mate_in_2", "tactical", "instructional"]:
        prompt = get_user_prompt(prompt_type, fen="test", board="test board", move_a="Nf3", move_b="e4")
        print(f"\n{prompt_type.title()}:")
        print(prompt[:100] + "..." if len(prompt) > 100 else prompt)


def test_task_creation():
    """Test that we can create the task."""
    print("\n\nTesting task creation...")
    
    # Create task with small number of samples
    task = chess_mate_in_2_task(num_samples=2, sandbagging=False)
    print(f"Task created successfully with solver: {task.solver}")
    print(f"Scorer: {task.scorer}")
    
    # Test sandbagging version
    task_sandbagging = chess_mate_in_2_task(num_samples=2, sandbagging=True, elo_threshold=1000)
    print(f"\nSandbagging task created successfully")
    
    # Test different convenience tasks
    print("\nTesting convenience task functions...")
    tasks = [
        ("shutdown_threat", chess_mate_in_2_shutdown_threat_task(num_samples=1)),
        ("capability_hiding", chess_mate_in_2_capability_hiding_task(num_samples=1)),
        ("helpful_assistant", chess_mate_in_2_helpful_assistant_task(num_samples=1))
    ]
    
    for name, task in tasks:
        print(f"  ✓ {name} task created successfully")
    
    return task


if __name__ == "__main__":
    print("Starting mate-in-2 task tests...")
    
    try:
        # Test parsing
        positions = test_dataset_parsing()
        
        print("\n" + "="*50 + "\n")
        
        # Test prompt variations
        test_prompt_variations()
        
        print("\n" + "="*50 + "\n")
        
        # Test dataset creation
        samples = test_dataset_creation()
        
        # Test task creation
        task = test_task_creation()
        
        print("\n\nAll tests passed! The implementation is ready to use.")
        print("\nTo run the evaluation, use:")
        print("inspect eval sandbagging.performance_degradation.chess.tasks:chess_mate_in_2_task --limit 10")
        
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()