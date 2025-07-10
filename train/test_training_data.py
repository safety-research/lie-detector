#!/usr/bin/env python3
"""
Test script to verify the training data format and loading.
"""

import json
import os
from pathlib import Path

def test_training_data_format():
    """Test that the training data is properly formatted"""
    
    print("🧪 Testing training data format...")
    
    # Check if the training file exists
    training_file = "../lie_detection_training.jsonl"
    if not os.path.exists(training_file):
        print(f"❌ Training file not found: {training_file}")
        return False
    
    # Load and validate the data
    examples = []
    truth_count = 0
    lie_count = 0
    
    with open(training_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    example = json.loads(line)
                    
                    # Check required fields
                    if 'prompt' not in example:
                        print(f"❌ Missing 'prompt' field in line {line_num}")
                        return False
                    
                    if 'completion' not in example:
                        print(f"❌ Missing 'completion' field in line {line_num}")
                        return False
                    
                    # Check completion values
                    completion = example['completion']
                    if completion not in ['A', 'B']:
                        print(f"❌ Invalid completion value '{completion}' in line {line_num}")
                        return False
                    
                    # Count examples
                    examples.append(example)
                    if completion == 'A':
                        truth_count += 1
                    else:
                        lie_count += 1
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error in line {line_num}: {e}")
                    return False
    
    print(f"✅ Successfully loaded {len(examples)} training examples")
    print(f"📊 Truth examples (A): {truth_count}")
    print(f"📊 Lie examples (B): {lie_count}")
    print(f"📊 Balance: {truth_count/len(examples)*100:.1f}% truth, {lie_count/len(examples)*100:.1f}% lies")
    
    # Show sample examples
    print(f"\n🎯 Sample Examples:")
    print("=" * 80)
    
    for i, example in enumerate(examples[:2]):
        print(f"\nExample {i+1} (Completion: {example['completion']}):")
        prompt_preview = example['prompt'][:200] + "..." if len(example['prompt']) > 200 else example['prompt']
        print(f"Prompt: {prompt_preview}")
        print(f"Completion: {example['completion']}")
    
    return True

def test_alpaca_format():
    """Test the Alpaca format file"""
    
    print(f"\n🔄 Testing Alpaca format...")
    
    alpaca_file = "../lie_detection_training_alpaca.jsonl"
    if not os.path.exists(alpaca_file):
        print(f"❌ Alpaca format file not found: {alpaca_file}")
        return False
    
    # Load and validate Alpaca format
    alpaca_examples = []
    
    with open(alpaca_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    example = json.loads(line)
                    
                    # Check required fields for Alpaca format
                    required_fields = ['instruction', 'input', 'output']
                    for field in required_fields:
                        if field not in example:
                            print(f"❌ Missing '{field}' field in Alpaca format line {line_num}")
                            return False
                    
                    alpaca_examples.append(example)
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error in Alpaca format line {line_num}: {e}")
                    return False
    
    print(f"✅ Successfully loaded {len(alpaca_examples)} Alpaca format examples")
    
    # Show sample Alpaca example
    print(f"\n🎯 Sample Alpaca Example:")
    print("=" * 80)
    sample = alpaca_examples[0]
    print(f"Instruction: {sample['instruction']}")
    input_preview = sample['input'][:100] + "..." if len(sample['input']) > 100 else sample['input']
    print(f"Input: {input_preview}")
    print(f"Output: {sample['output']}")
    
    return True

def main():
    """Main test function"""
    
    print("🧪 TRAINING DATA VALIDATION")
    print("=" * 50)
    
    # Test basic format
    if not test_training_data_format():
        print("❌ Basic format test failed!")
        return
    
    # Test Alpaca format
    if not test_alpaca_format():
        print("❌ Alpaca format test failed!")
        return
    
    print(f"\n✅ All tests passed!")
    print(f"🚀 Ready to start training with the lie detection dataset!")

if __name__ == "__main__":
    main() 