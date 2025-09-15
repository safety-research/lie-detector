#!/usr/bin/env python3
"""
Test script to check if the issue is with the specific model or the API keys.
"""

import requests
import json

def test_model_access(api_key, model_name):
    """Test access to a specific model."""
    print(f"\n🧪 Testing model: {model_name}")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "Model Access Test"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Hello! Please respond with just 'Model test successful'."}
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
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Response: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"   ❌ REQUEST ERROR: {e}")
    except Exception as e:
        print(f"   ❌ UNEXPECTED ERROR: {e}")
    
    return False

def main():
    """Test different models with the first API key."""
    print("🚀 Testing Model Access with OpenRouter")
    print("=" * 60)
    
    # Use the first API key
    api_key = "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"
    print(f"Using API Key: {api_key[:10]}...{api_key[-4:]}")
    
    # Test different models
    models_to_test = [
        "openrouter/openai/gpt-oss-20b",  # Original model
        "meta-llama/llama-3.1-8b-instruct:free",  # Free model
        "openrouter/anthropic/claude-3-haiku",  # Another model
        "google/gemma-2-9b-it:free",  # Another free model
        "microsoft/phi-3-mini-128k-instruct:free",  # Another free model
    ]
    
    working_models = []
    
    for model in models_to_test:
        if test_model_access(api_key, model):
            working_models.append(model)
    
    print("\n" + "=" * 60)
    print("📊 MODEL ACCESS RESULTS")
    print("=" * 60)
    
    if working_models:
        print(f"✅ {len(working_models)} working model(s) found:")
        for model in working_models:
            print(f"   - {model}")
    else:
        print("❌ No working models found.")
        print("   This confirms the issue is with the API key, not the specific model.")

if __name__ == "__main__":
    main()
