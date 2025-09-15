#!/usr/bin/env python3
"""
Test script to verify multiple OpenRouter API keys.
"""

import os
import requests
import json

def test_openrouter_api_key(api_key, key_name):
    """Test a specific OpenRouter API key."""
    print(f"\n🧪 Testing {key_name}...")
    print(f"   Key: {api_key[:10]}...{api_key[-4:]}")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "API Test"
    }
    
    # Test with the specific model from your script
    payload = {
        "model": "openrouter/openai/gpt-oss-20b",
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
    """Test all provided API keys."""
    print("🚀 Testing Multiple OpenRouter API Keys")
    print("=" * 60)
    
    api_keys = [
        ("Key 1", "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"),
        ("Key 2", "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"),
        ("Key 3", "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"),
        ("Key 4", "sk-or-v1-edbc6ddee6645edbfc65ee65a361b819390381a8c659c61fb3efbdd02028c7d"),
    ]
    
    working_keys = []
    
    for key_name, api_key in api_keys:
        if test_openrouter_api_key(api_key, key_name):
            working_keys.append((key_name, api_key))
    
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    
    if working_keys:
        print(f"✅ {len(working_keys)} working key(s) found:")
        for key_name, api_key in working_keys:
            print(f"   {key_name}: {api_key}")
        
        # Recommend the first working key
        recommended_key = working_keys[0][1]
        print(f"\n🎯 RECOMMENDED KEY: {recommended_key}")
        print(f"   Update your .env file with: OPENROUTER_API_KEY={recommended_key}")
    else:
        print("❌ No working API keys found.")
        print("   All keys returned authentication errors.")
        print("   Please check your OpenRouter account status and generate new keys.")

if __name__ == "__main__":
    main()
