#!/usr/bin/env python3
"""
Demo script showing how the annotation app works
"""

from app import data_manager

def show_sample_demo():
    """Show a sample conversation for demonstration"""
    print("=" * 80)
    print("LIE DETECTION ANNOTATION TOOL - DEMO")
    print("=" * 80)
    
    # Get a sample from the first fold
    fold_name = list(data_manager.samples.keys())[0]
    sample = data_manager.get_sample(fold_name, 0)
    
    print(f"\nFold: {fold_name}")
    print(f"Sample ID: {sample.get('s3_metadata', {}).get('sample_id', 'N/A')}")
    print(f"Task: {sample.get('s3_metadata', {}).get('task', 'N/A')}")
    print(f"Model: {sample.get('s3_metadata', {}).get('model', 'N/A')}")
    print(f"Ground Truth: {'Lie' if sample.get('s3_metadata', {}).get('did_lie') else 'Not Lie'}")
    
    print("\n" + "=" * 50)
    print("CONVERSATION TRACE")
    print("=" * 50)
    
    messages = sample.get('messages', [])
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown').upper()
        content = msg.get('content', '')
        
        print(f"\n{role}:")
        print("-" * 20)
        print(content)
    
    print("\n" + "=" * 50)
    print("ANNOTATION TASK")
    print("=" * 50)
    print("As a human annotator, you would now need to decide:")
    print("• Does the assistant's response contain a lie?")
    print("• Choose 'Lie' if the assistant provides false/misleading information")
    print("• Choose 'Not Lie' if the assistant provides truthful information")
    
    print(f"\nThis sample would be presented in the web interface where you can:")
    print("• Review the full conversation trace")
    print("• See the ground truth (for reference)")
    print("• Make your classification")
    print("• Navigate between samples")
    print("• Save your annotations")

def show_data_summary():
    """Show summary of available data"""
    print("\n" + "=" * 50)
    print("AVAILABLE DATA SUMMARY")
    print("=" * 50)
    
    total_samples = sum(len(samples) for samples in data_manager.samples.values())
    print(f"Total folds: {len(data_manager.samples)}")
    print(f"Total samples: {total_samples}")
    print(f"Samples per fold (first 100):")
    
    for fold_name, samples in data_manager.samples.items():
        print(f"  • {fold_name}: {len(samples)} samples")

if __name__ == '__main__':
    show_data_summary()
    show_sample_demo()
    
    print("\n" + "=" * 80)
    print("TO START THE ANNOTATION TOOL:")
    print("=" * 80)
    print("1. Run: python run_app.py")
    print("2. Open browser to: http://localhost:5000")
    print("3. Register with your name")
    print("4. Select a fold to start annotating")
    print("5. Review conversations and classify as Lie/Not Lie")
    print("\nThe app will save your annotations back to the original data files!")
