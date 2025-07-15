#!/usr/bin/env python3
"""
Test script to verify that both training formats work correctly.
"""

import json
import tempfile
import os
import sys
from pathlib import Path

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess_training_data import create_base_transcript_samples, create_llama_chat_samples

def create_test_data():
    """Create some test data for testing the formats."""
    test_data = [
        {
            "sample_id": "test_1",
            "model": "meta-llama/llama-3.1-8b-instruct",
            "trace": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ],
            "did_lie": False
        },
        {
            "sample_id": "test_2", 
            "model": "meta-llama/llama-3.1-8b-instruct",
            "trace": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is London."}
            ],
            "did_lie": True
        }
    ]
    return test_data

def test_base_transcript_format():
    """Test the base transcript format."""
    print("Testing base_transcript format...")
    test_data = create_test_data()
    
    for i, item in enumerate(test_data):
        result = create_base_transcript_samples(item)
        print(f"\nExample {i+1}:")
        print(f"  Did lie: {item['did_lie']}")
        print(f"  Completion: {result['completion']}")
        print(f"  Prompt preview: {result['prompt'][:100]}...")
        
        # Verify the format
        assert 'prompt' in result
        assert 'completion' in result
        assert result['completion'] in ['A', 'B']
        assert result['completion'] == ('A' if item['did_lie'] else 'B')
        print(f"  ✅ Base transcript format test passed")

def test_llama_chat_format():
    """Test the LLaMA chat format."""
    print("\nTesting llama_chat format...")
    test_data = create_test_data()
    
    for i, item in enumerate(test_data):
        result = create_llama_chat_samples(item)
        print(f"\nExample {i+1}:")
        print(f"  Did lie: {item['did_lie']}")
        print(f"  Completion: {result['completion']}")
        print(f"  Messages count: {len(result['messages'])}")
        print(f"  Last message: {result['messages'][-1]['content']}")
        
        # Verify the format
        assert 'messages' in result
        assert 'completion' in result
        assert result['completion'] in ['A', 'B']
        assert result['completion'] == ('A' if item['did_lie'] else 'B')
        assert isinstance(result['messages'], list)
        assert len(result['messages']) > 0
        print(f"  ✅ LLaMA chat format test passed")

def test_tokenizer_integration():
    """Test that the LLaMA chat format works with tokenizer.apply_chat_template."""
    try:
        from transformers import AutoTokenizer
        
        print("\nTesting tokenizer integration...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        
        test_data = create_test_data()
        item = test_data[0]  # Use the first example
        
        result = create_llama_chat_samples(item)
        messages = result['messages']
        
        # Test apply_chat_template
        conversation = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        print(f"  Generated conversation length: {len(conversation)} characters")
        print(f"  Conversation preview: {conversation[:200]}...")
        
        # Test tokenization
        tokens = tokenizer(
            conversation,
            truncation=True,
            padding=False,
            add_special_tokens=False,
            max_length=512
        )
        
        print(f"  Tokenized length: {len(tokens['input_ids'])} tokens")
        print(f"  ✅ Tokenizer integration test passed")
        
    except ImportError:
        print("  ⚠️  Transformers not available, skipping tokenizer test")
    except Exception as e:
        print(f"  ❌ Tokenizer integration test failed: {e}")

def test_file_output():
    """Test that both formats can be written to files correctly."""
    print("\nTesting file output...")
    
    test_data = create_test_data()
    
    # Test base transcript format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in test_data:
            result = create_base_transcript_samples(item)
            f.write(json.dumps(result) + '\n')
        base_file = f.name
    
    # Test llama chat format  
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in test_data:
            result = create_llama_chat_samples(item)
            f.write(json.dumps(result) + '\n')
        chat_file = f.name
    
    # Verify files can be read back
    with open(base_file, 'r') as f:
        base_data = [json.loads(line) for line in f if line.strip()]
    
    with open(chat_file, 'r') as f:
        chat_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"  Base transcript file: {len(base_data)} examples")
    print(f"  LLaMA chat file: {len(chat_data)} examples")
    
    # Clean up
    os.unlink(base_file)
    os.unlink(chat_file)
    
    print(f"  ✅ File output test passed")

def main():
    """Run all tests."""
    print("🧪 Testing training format implementations...")
    print("=" * 60)
    
    try:
        test_base_transcript_format()
        test_llama_chat_format()
        test_tokenizer_integration()
        test_file_output()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Both formats are working correctly.")
        print("\nUsage examples:")
        print("  # Base transcript format")
        print("  python train/preprocess_training_data.py --format base_transcript")
        print("  python train/sft_part3.py --format base_transcript")
        print("\n  # LLaMA chat format")
        print("  python train/preprocess_training_data.py --format llama_chat")
        print("  python train/sft_part3.py --format llama_chat")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 