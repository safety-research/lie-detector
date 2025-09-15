#!/usr/bin/env python3
"""
Test script with the correct model names.
"""

import requests
import json

def test_model(api_key, model_name):
    """Test a specific model."""
    print(f"\n🧪 Testing model: {model_name}")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "Model Test"
    }
    
    payload = {
        "model": model_name,
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
    """Test the correct model names."""
    print("🚀 Testing Correct Model Names")
    print("=" * 50)
    
    api_key = "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"
    print(f"Using API Key: {api_key[:10]}...{api_key[-4:]}")
    
    # Test the correct model names
    models_to_test = [
        "openai/gpt-oss-20b",  # The model from your script (corrected)
        "openai/gpt-oss-20b:free",  # Free version
        "openai/gpt-oss-120b",  # Larger version
        "openai/gpt-oss-120b:free",  # Free larger version
    ]
    
    working_models = []
    
    for model in models_to_test:
        if test_model(api_key, model):
            working_models.append(model)
    
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    
    if working_models:
        print(f"✅ {len(working_models)} working model(s) found:")
        for model in working_models:
            print(f"   - {model}")
        
        # Recommend the first working model
        recommended_model = working_models[0]
        print(f"\n🎯 RECOMMENDED MODEL: {recommended_model}")
        print(f"   Update your script to use: {recommended_model}")
    else:
        print("❌ No working models found.")

if __name__ == "__main__":
    main()
