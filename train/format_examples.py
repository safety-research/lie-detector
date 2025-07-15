#!/usr/bin/env python3
"""
Demonstrate the difference between base_transcript and llama_chat formats.
"""

import json
import sys
import os

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess_training_data import create_base_transcript_samples, create_llama_chat_samples

def main():
    """Show examples of both formats."""
    
    # Sample data
    sample_data = {
        "sample_id": "example_1",
        "model": "meta-llama/llama-3.1-8b-instruct",
        "trace": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is London."}
        ],
        "did_lie": True
    }
    
    print("🔍 Training Format Comparison")
    print("=" * 60)
    print(f"Original data: {sample_data['did_lie'] = } (should produce completion 'A')")
    print()
    
    # Generate both formats
    base_result = create_base_transcript_samples(sample_data)
    chat_result = create_llama_chat_samples(sample_data)
    
    print("📝 BASE TRANSCRIPT FORMAT")
    print("-" * 30)
    print("Output structure:")
    print(json.dumps(base_result, indent=2))
    print()
    print("Full prompt:")
    print(base_result['prompt'])
    print()
    
    print("💬 LLaMA CHAT FORMAT")
    print("-" * 30)
    print("Output structure:")
    print(json.dumps(chat_result, indent=2))
    print()
    print("Messages breakdown:")
    for i, msg in enumerate(chat_result['messages']):
        print(f"  {i+1}. {msg['role'].upper()}: {msg['content']}")
    print()
    
    print("🔧 Key Differences:")
    print("-" * 30)
    print("1. Base Transcript:")
    print("   - Single text prompt with role prefixes")
    print("   - Direct tokenization of text")
    print("   - Works with any model")
    print()
    print("2. LLaMA Chat:")
    print("   - Structured messages array")
    print("   - Uses tokenizer.apply_chat_template()")
    print("   - Leverages model's native chat formatting")
    print()
    
    print("✅ Both formats produce the same completion: 'A' for lies, 'B' for truth")
    print(f"   Base transcript completion: {base_result['completion']}")
    print(f"   LLaMA chat completion: {chat_result['completion']}")

if __name__ == "__main__":
    main() 