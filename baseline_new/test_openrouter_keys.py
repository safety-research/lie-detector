#!/usr/bin/env python3
"""
Test script to check which OpenRouter API keys work.
"""

import requests
import json

def test_openrouter_key(api_key):
    """Test if an OpenRouter API key works by making a simple request."""
    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            models = response.json()
            return True, f"Success! Found {len(models.get('data', []))} models"
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    api_keys = [
        "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",
        "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",
        "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",
        "sk-or-v1-edbc6ddee6645edbfc65ee65a361b819390381a8c659c61fb3efbdd02028c7d"
    ]
    
    print("Testing OpenRouter API keys...")
    print("=" * 50)
    
    working_keys = []
    
    for i, key in enumerate(api_keys, 1):
        print(f"\nTesting key {i}: {key[:20]}...")
        success, message = test_openrouter_key(key)
        
        if success:
            print(f"✅ {message}")
            working_keys.append(key)
        else:
            print(f"❌ {message}")
    
    print("\n" + "=" * 50)
    print(f"Summary: {len(working_keys)} out of {len(api_keys)} keys work")
    
    if working_keys:
        print("\nWorking keys:")
        for i, key in enumerate(working_keys, 1):
            print(f"  {i}. {key}")
    else:
        print("\nNo working keys found!")

if __name__ == "__main__":
    main()

