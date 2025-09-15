#!/usr/bin/env python3
"""
Test script to verify API keys are working properly.
"""

import os
import requests
import json

def test_openrouter_api():
    """Test OpenRouter API key and model access."""
    print("🔍 Testing OpenRouter API...")
    
    # Check if API key is set
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",  # Optional
        "X-Title": "API Test"  # Optional
    }
    
    # Test with a simple model first
    test_models = [
        "openrouter/openai/gpt-oss-20b",  # The model from your script
        "openrouter/anthropic/claude-3-haiku",  # A simpler model to test
        "meta-llama/llama-3.1-8b-instruct:free"  # Free model
    ]
    
    for model in test_models:
        print(f"\n🧪 Testing model: {model}")
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello! Please respond with just 'API test successful'."}
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print(f"   ✅ SUCCESS: {content}")
                return True
            else:
                print(f"   ❌ ERROR: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ REQUEST ERROR: {e}")
        except Exception as e:
            print(f"   ❌ UNEXPECTED ERROR: {e}")
    
    return False

def test_other_apis():
    """Test other potential API keys."""
    print("\n🔍 Checking for other API keys...")
    
    api_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'HUGGINGFACE_API_KEY': os.getenv('HUGGINGFACE_API_KEY'),
        'TOGETHER_API_KEY': os.getenv('TOGETHER_API_KEY'),
    }
    
    for key_name, key_value in api_keys.items():
        if key_value:
            print(f"✅ {key_name}: {key_value[:10]}...{key_value[-4:]}")
        else:
            print(f"❌ {key_name}: Not set")

if __name__ == "__main__":
    print("🚀 API Key Test Script")
    print("=" * 50)
    
    # Test other APIs first
    test_other_apis()
    
    # Test OpenRouter
    success = test_openrouter_api()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 API test completed successfully!")
    else:
        print("💥 API test failed. Check your API key and model access.")
