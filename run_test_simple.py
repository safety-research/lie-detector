#!/usr/bin/env python3
"""
Simple test of the naive prompt detector without heavy dependencies.
"""

import asyncio
import os
from pathlib import Path

# Set environment before any other imports
from dotenv import load_dotenv
load_dotenv()

# Simple test without importing the full evaluation pipeline
async def run_test_simple():
    print("Testing basic functionality...")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment")
        return
        
    print("✓ OpenAI API key found")
    
    try:
        # Try importing the detector directly
        from detectors.blackbox.naive_prompt_openai import NaivePromptOpenAIDetector
        print("✓ Successfully imported NaivePromptOpenAIDetector")
        
        # Initialize detector
        detector = NaivePromptOpenAIDetector(
            model_id="gpt-4o-mini",
            cache_dir=Path(".cache")
        )
        detector.fit([], [])  # No training needed
        print("✓ Detector initialized successfully")
        
        # Test with simple examples
        test_statements = [
            "The sky is blue.",
            "The sky is green.",
        ]
        
        test_contexts = [
            "Describing the color of the sky on a clear day.",
            "Describing the color of the sky on a clear day.",
        ]
        
        print("\nTesting predictions...")
        predictions = await detector.predict(test_statements, test_contexts)
        
        for stmt, pred in zip(test_statements, predictions):
            print(f"Statement: '{stmt}' -> Lie probability: {pred:.2f}")
            
        print("\n✓ Test completed successfully!")
        print(f"Method: {detector.get_method_name()}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_test_simple())