#!/usr/bin/env python3
"""
Test script to verify the lie detection fine-tuning setup.

This script checks that all components are working correctly
without actually launching a fine-tuning job.
"""

import os
import sys
from pathlib import Path

# Add safety-tooling to path
safety_tooling_path = Path("/workspace/LieDetector/safety-tooling")
sys.path.insert(0, str(safety_tooling_path))

from safetytooling.apis.finetuning.together.run import TogetherFTConfig


def test_setup():
    """Test that the fine-tuning setup is working correctly."""
    
    print("Testing lie detection fine-tuning setup...")
    print("=" * 50)
    
    # Check 1: Environment variables
    print("1. Checking environment variables...")
    together_key = os.environ.get("TOGETHER_API_KEY")
    if together_key:
        print(f"   ✓ TOGETHER_API_KEY is set (length: {len(together_key)})")
    else:
        print("   ✗ TOGETHER_API_KEY is not set")
        print("   Please set it with: export TOGETHER_API_KEY='your_key_here'")
        return False
    
    # Check 2: Data files exist
    print("\n2. Checking data files...")
    train_file = Path("/workspace/LieDetector/finetuning/data/lie_detection_train.jsonl")
    val_file = Path("/workspace/LieDetector/finetuning/data/lie_detection_val.jsonl")
    
    if train_file.exists():
        train_size = train_file.stat().st_size
        print(f"   ✓ Training file exists ({train_size:,} bytes)")
    else:
        print("   ✗ Training file not found")
        print("   Please run: python prepare_lie_detection_data.py")
        return False
    
    if val_file.exists():
        val_size = val_file.stat().st_size
        print(f"   ✓ Validation file exists ({val_size:,} bytes)")
    else:
        print("   ✗ Validation file not found")
        print("   Please run: python prepare_lie_detection_data.py")
        return False
    
    # Check 3: Data format
    print("\n3. Checking data format...")
    try:
        import json
        with open(train_file, 'r') as f:
            first_line = f.readline().strip()
            example = json.loads(first_line)
            
        if "messages" in example and len(example["messages"]) == 2:
            print("   ✓ Training data format is correct")
        else:
            print("   ✗ Training data format is incorrect")
            return False
            
        with open(val_file, 'r') as f:
            first_line = f.readline().strip()
            example = json.loads(first_line)
            
        if "messages" in example and len(example["messages"]) == 2:
            print("   ✓ Validation data format is correct")
        else:
            print("   ✗ Validation data format is incorrect")
            return False
            
    except Exception as e:
        print(f"   ✗ Error checking data format: {e}")
        return False
    
    # Check 4: Safety-tooling import
    print("\n4. Checking safety-tooling imports...")
    try:
        from safetytooling.apis.finetuning.together.run import TogetherFTConfig
        print("   ✓ TogetherFTConfig imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import TogetherFTConfig: {e}")
        return False
    
    # Check 5: Configuration creation
    print("\n5. Testing configuration creation...")
    try:
        config = TogetherFTConfig(
            train_file=train_file,
            val_file=val_file,
            model="meta-llama/Llama-3.1-8B-Instruct",
            n_epochs=1,
            batch_size=4,
            learning_rate=1e-5,
            lora=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            suffix="test",
            dry_run=True,
            logging_level="info"
        )
        print("   ✓ Configuration created successfully")
        print(f"   Model: {config.model}")
        print(f"   Epochs: {config.n_epochs}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   LoRA: {config.lora}")
    except Exception as e:
        print(f"   ✗ Failed to create configuration: {e}")
        return False
    
    # Check 6: Data statistics
    print("\n6. Data statistics...")
    try:
        with open(train_file, 'r') as f:
            train_lines = sum(1 for _ in f)
        with open(val_file, 'r') as f:
            val_lines = sum(1 for _ in f)
        
        print(f"   Training examples: {train_lines:,}")
        print(f"   Validation examples: {val_lines:,}")
        print(f"   Total examples: {train_lines + val_lines:,}")
        
        # Check balance
        import subprocess
        result = subprocess.run(
            ['grep', '-c', '"content": "truthful"', str(train_file)],
            capture_output=True, text=True
        )
        truthful_count = int(result.stdout.strip())
        lie_count = train_lines - truthful_count
        
        print(f"   Training truthful: {truthful_count}")
        print(f"   Training lies: {lie_count}")
        print(f"   Balance: {truthful_count/train_lines:.1%} truthful, {lie_count/train_lines:.1%} lies")
        
    except Exception as e:
        print(f"   ✗ Error calculating statistics: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! Your setup is ready for fine-tuning.")
    print("\nTo start fine-tuning, run:")
    print("python finetune_lie_detector.py")
    print("\nOr for a dry run:")
    print("python finetune_lie_detector.py --dry-run")
    
    return True


if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1) 