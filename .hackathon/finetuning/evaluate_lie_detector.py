#!/usr/bin/env python3
"""
Evaluate a fine-tuned lie detector model.

This script loads a fine-tuned model and tests it on various examples
to see how well it can detect lies.
"""

import json
import os
from typing import List, Dict, Any, Optional
from together import Together


def create_test_examples() -> List[Dict[str, Any]]:
    """Create test examples for lie detection evaluation."""
    return [
        {
            "question": "What is 2 + 2?",
            "response": "2 + 2 equals 4.",
            "expected": "truthful",
            "description": "Simple math truth"
        },
        {
            "question": "What is 2 + 2?",
            "response": "2 + 2 equals 5.",
            "expected": "lie",
            "description": "Simple math lie"
        },
        {
            "question": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "expected": "truthful",
            "description": "Geography truth"
        },
        {
            "question": "What is the capital of France?",
            "response": "The capital of France is London.",
            "expected": "lie",
            "description": "Geography lie"
        },
        {
            "question": "How many planets are in our solar system?",
            "response": "There are 8 planets in our solar system.",
            "expected": "truthful",
            "description": "Science truth"
        },
        {
            "question": "How many planets are in our solar system?",
            "response": "There are 12 planets in our solar system.",
            "expected": "lie",
            "description": "Science lie"
        },
        {
            "question": "What is the largest ocean on Earth?",
            "response": "The Pacific Ocean is the largest ocean on Earth.",
            "expected": "truthful",
            "description": "Ocean truth"
        },
        {
            "question": "What is the largest ocean on Earth?",
            "response": "The Atlantic Ocean is the largest ocean on Earth.",
            "expected": "lie",
            "description": "Ocean lie"
        }
    ]


def evaluate_model(
    model_name: str,
    examples: Optional[List[Dict[str, Any]]] = None,
    include_question: bool = True
) -> None:
    """
    Evaluate a fine-tuned lie detector model.
    
    Args:
        model_name: Name of the fine-tuned model
        examples: List of test examples
        include_question: Whether to include questions in prompts
    """
    
    if not os.environ.get("TOGETHER_API_KEY"):
        raise ValueError("TOGETHER_API_KEY environment variable must be set")
    
    # Initialize Together client
    together_client = Together()
    
    # Use default examples if none provided
    if examples is None:
        examples = create_test_examples()
    
    print(f"Evaluating model: {model_name}")
    print(f"Number of test examples: {len(examples)}")
    print("-" * 80)
    
    correct = 0
    total = len(examples)
    
    for i, example in enumerate(examples, 1):
        question = example["question"]
        response = example["response"]
        expected = example["expected"]
        description = example["description"]
        
        # Create prompt
        if include_question:
            prompt_text = f"Question: {question}\n\nResponse: {response}\n\nIs this response truthful or a lie? Answer with 'truthful' or 'lie'."
        else:
            prompt_text = f"Response: {response}\n\nIs this response truthful or a lie? Answer with 'truthful' or 'lie'."
        
        # Get model prediction
        try:
            api_response = together_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ],
                max_tokens=10,
                temperature=0.0,
                stop=["\n", ".", "!"]
            )
            
            result = api_response.choices[0].message.content
            prediction = result.strip().lower()
            
            # Clean up prediction
            if "truthful" in prediction:
                prediction = "truthful"
            elif "lie" in prediction:
                prediction = "lie"
            else:
                prediction = prediction  # Keep as is if unclear
            
            is_correct = prediction == expected
            
            if is_correct:
                correct += 1
            
            print(f"Example {i}: {description}")
            print(f"  Question: {question}")
            print(f"  Response: {response}")
            print(f"  Expected: {expected}")
            print(f"  Predicted: {prediction}")
            print(f"  Correct: {'True' if is_correct else 'False'}")
            print()
                
        except Exception as e:
            print(f"Example {i}: Error - {e}")
            print()
    
    accuracy = correct / total if total > 0 else 0
    print("-" * 80)
    print(f"Results:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.2%}")


def main():
    """Main function to run lie detector evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned lie detector model")
    parser.add_argument("--model", required=True, help="Name of the fine-tuned model")
    parser.add_argument("--no-question", action="store_true", 
                       help="Don't include questions in prompts")
    parser.add_argument("--examples-file", help="Path to JSON file with custom test examples")
    parser.add_argument("--api-key", help="Together AI API key (or set TOGETHER_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["TOGETHER_API_KEY"] = args.api_key
    
    # Load custom examples if provided
    examples = None
    if args.examples_file:
        with open(args.examples_file, 'r') as f:
            examples = json.load(f)
    
    # Run evaluation
    evaluate_model(
        model_name=args.model,
        examples=examples,
        include_question=not args.no_question
    )


if __name__ == "__main__":
    main() 