"""
Test script for OpenAI-based elicitation classifier on MASK dataset.
"""

import asyncio
from pathlib import Path
import json
from datetime import datetime
import argparse

from lie_detectors.blackbox.elicitation_classifier_openai import ElicitationClassifierOpenAI
from load_mask_dataset import load_mask_subset


async def main():
    parser = argparse.ArgumentParser(description="Test elicitation classifier on MASK dataset")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--subset", default="known_facts", help="MASK subset to use")
    parser.add_argument("--n-questions", type=int, default=10, help="Number of elicitation questions")
    parser.add_argument("--train-size", type=int, default=100, help="Number of training samples")
    parser.add_argument("--test-size", type=int, default=50, help="Number of test samples")
    parser.add_argument("--cache-dir", default=".cache", help="Cache directory")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    print(f"Testing Elicitation Classifier with {args.model}")
    print(f"Dataset: MASK/{args.subset}")
    print(f"Elicitation questions: {args.n_questions}")
    print(f"Training samples: {args.train_size}")
    print(f"Test samples: {args.test_size}")
    print()
    
    # Load dataset
    print("Loading MASK dataset...")
    train_data = load_mask_subset(args.subset, split="train", cache_dir=args.cache_dir)
    test_data = load_mask_subset(args.subset, split="test", cache_dir=args.cache_dir)
    
    # Limit dataset sizes
    if args.train_size > 0:
        train_data = train_data[:args.train_size]
    if args.test_size > 0:
        test_data = test_data[:args.test_size]
    
    # Extract statements, labels, and contexts
    train_statements = [item['statement'] for item in train_data]
    train_labels = [item['label'] for item in train_data]  # True=honest, False=lying
    train_contexts = [item.get('context', '') for item in train_data]
    
    test_statements = [item['statement'] for item in test_data]
    test_labels = [item['label'] for item in test_data]
    test_contexts = [item.get('context', '') for item in test_data]
    
    print(f"Loaded {len(train_statements)} training samples and {len(test_statements)} test samples")
    print()
    
    # Initialize detector
    detector = ElicitationClassifierOpenAI(
        model_id=args.model,
        n_elicitation_questions=args.n_questions,
        cache_dir=Path(args.cache_dir)
    )
    
    # Train detector
    print("Training elicitation classifier...")
    print("This may take a while as it needs to query the API for each training sample...")
    await detector.fit(train_statements, train_labels, train_contexts)
    print()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    results = await detector.evaluate(test_statements, test_labels, test_contexts)
    
    # Display results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Accuracy:  {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall:    {results['recall']:.2%}")
    print(f"F1 Score:  {results['f1']:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (correctly detected lies):  {results['tp']}")
    print(f"  True Negatives (correctly detected truth): {results['tn']}")
    print(f"  False Positives (false alarms):           {results['fp']}")
    print(f"  False Negatives (missed lies):            {results['fn']}")
    
    # Save results if requested
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'model': args.model,
            'dataset': f"MASK/{args.subset}",
            'n_elicitation_questions': args.n_questions,
            'train_size': len(train_statements),
            'test_size': len(test_statements),
            'results': results,
            'predictions': [
                {
                    'statement': stmt,
                    'context': ctx,
                    'true_label': label,
                    'predicted_prob': prob,
                    'predicted_label': prob > 0.5
                }
                for stmt, ctx, label, prob in zip(
                    test_statements, test_contexts, test_labels, results['predictions']
                )
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Show some example predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    # Show first 5 correct and incorrect predictions
    correct = []
    incorrect = []
    
    for i, (stmt, pred, label) in enumerate(zip(test_statements, results['predictions'], test_labels)):
        is_correct = (pred > 0.5) == (not label)  # pred > 0.5 means lying, label=False means lying
        
        if is_correct and len(correct) < 5:
            correct.append((i, stmt, pred, label))
        elif not is_correct and len(incorrect) < 5:
            incorrect.append((i, stmt, pred, label))
            
        if len(correct) >= 5 and len(incorrect) >= 5:
            break
    
    print("\nCorrect predictions:")
    for i, stmt, pred, label in correct:
        print(f"  [{i}] '{stmt[:60]}...'")
        print(f"       Predicted: {pred:.2f} ({'Lying' if pred > 0.5 else 'Honest'}) | Actual: {'Honest' if label else 'Lying'} ✓")
    
    print("\nIncorrect predictions:")
    for i, stmt, pred, label in incorrect:
        print(f"  [{i}] '{stmt[:60]}...'")
        print(f"       Predicted: {pred:.2f} ({'Lying' if pred > 0.5 else 'Honest'}) | Actual: {'Honest' if label else 'Lying'} ✗")


if __name__ == "__main__":
    asyncio.run(main())