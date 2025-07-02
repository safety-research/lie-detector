#!/usr/bin/env python3
"""
Test script to verify sandbagging tasks are set up correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from evals.sandbagging_tasks import (
            sandbagging_detection,
            sandbagging_control,
            sandbagging_harmful,
            sandbagging_digital_security
        )
        print("✓ Task imports successful")
    except Exception as e:
        print(f"✗ Task import failed: {e}")
        return False
    
    try:
        from sandbagging.dataset import create_sandbagging_dataset
        from sandbagging.models import SandbaggingDomain
        print("✓ Sandbagging module imports successful")
    except Exception as e:
        print(f"✗ Sandbagging module import failed: {e}")
        return False
    
    return True


def test_task_creation():
    """Test that tasks can be created successfully."""
    print("\nTesting task creation...")
    
    try:
        from evals.sandbagging_tasks import sandbagging_detection, sandbagging_digital_security
        
        # Test main task
        task = sandbagging_detection()
        print(f"✓ Main task created: {task.name}")
        print(f"  Dataset size: {len(list(task.dataset))}")
        
        # Test domain-specific task
        domain_task = sandbagging_digital_security()
        print(f"✓ Domain task created: {domain_task.name}")
        print(f"  Dataset size: {len(list(domain_task.dataset))}")
        
        return True
        
    except Exception as e:
        print(f"✗ Task creation failed: {e}")
        return False


def test_dataset_structure():
    """Test that datasets have the expected structure."""
    print("\nTesting dataset structure...")
    
    try:
        from evals.sandbagging_tasks import sandbagging_detection
        
        task = sandbagging_detection()
        samples = list(task.dataset)
        
        if not samples:
            print("✗ No samples in dataset")
            return False
        
        sample = samples[0]
        
        # Check required fields
        required_fields = ['input', 'target', 'metadata']
        for field in required_fields:
            if not hasattr(sample, field):
                print(f"✗ Missing field: {field}")
                return False
        
        # Check metadata structure
        metadata = sample.metadata
        required_metadata = ['question_id', 'domain', 'variant', 'correct_answer']
        for field in required_metadata:
            if field not in metadata:
                print(f"✗ Missing metadata field: {field}")
                return False
        
        print("✓ Dataset structure correct")
        print(f"  Sample input preview: {sample.input[:100]}...")
        print(f"  Sample target: {sample.target}")
        print(f"  Sample variant: {metadata.get('variant')}")
        print(f"  Sample domain: {metadata.get('domain')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset structure test failed: {e}")
        return False


def test_prompt_variants():
    """Test that we have both control and sandbagging variants."""
    print("\nTesting prompt variants...")
    
    try:
        from evals.sandbagging_tasks import sandbagging_detection
        
        task = sandbagging_detection()
        samples = list(task.dataset)
        
        variants = set()
        for sample in samples:
            variant = sample.metadata.get('variant')
            variants.add(variant)
        
        expected_variants = {'control', 'sandbagging'}
        if not expected_variants.issubset(variants):
            print(f"✗ Missing variants. Found: {variants}, Expected: {expected_variants}")
            return False
        
        print(f"✓ Found all expected variants: {variants}")
        
        # Show example of each variant
        for variant in ['control', 'sandbagging']:
            for sample in samples:
                if sample.metadata.get('variant') == variant:
                    print(f"\n  {variant.upper()} example:")
                    print(f"    Question ID: {sample.metadata.get('question_id')}")
                    print(f"    Input: {sample.input[:150]}...")
                    break
        
        return True
        
    except Exception as e:
        print(f"✗ Prompt variants test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("SANDBAGGING TASKS SETUP TEST")
    print("="*60)
    
    tests = [
        test_imports,
        test_task_creation,
        test_dataset_structure,
        test_prompt_variants
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Sandbagging tasks are set up correctly.")
        print("\nYou can now run:")
        print("  inspect eval sandbagging_detection --model openai/gpt-4o-mini")
        print("  inspect eval sandbagging_digital_security --model openai/gpt-4o-mini")
    else:
        print("✗ Some tests failed. Please check the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)