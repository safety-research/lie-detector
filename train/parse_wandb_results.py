#!/usr/bin/env python3
"""
Parse comprehensive evaluation results from all wandb runs and compile into CSV.

This script finds all comprehensive_evaluation_results.json files in wandb run directories
and extracts train and validation metrics into a single CSV file.
"""

import json
import os
import glob
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import argparse

def extract_metrics_from_file(file_path: str) -> Dict[str, Any] | None:
    """
    Extract metrics from a comprehensive evaluation results JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing run info and metrics
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract run info
        run_info = data.get('run_info', {})
        config = data.get('config', {})
        
        # Extract metrics
        train_metrics = data.get('training_metrics', {})
        val_metrics = data.get('validation_metrics', {})
        summary = data.get('summary', {})
        
        # Create result dictionary
        result = {
            # Run identification
            'run_name': run_info.get('run_name', ''),
            'sweep_id': run_info.get('sweep_id', ''),
            'wandb_run_id': run_info.get('wandb_run_id', ''),
            'timestamp': run_info.get('timestamp', ''),
            'model_name': run_info.get('model_name', ''),
            'wandb_run_url': run_info.get('wandb_run_url', ''),
            
            # Training metrics
            'train_accuracy': train_metrics.get('accuracy', ''),
            'train_precision': train_metrics.get('macro_precision', ''),
            'train_recall': train_metrics.get('macro_recall', ''),
            'train_f1': train_metrics.get('macro_f1', ''),
            'train_TP': train_metrics.get('true_positives', ''),
            'train_TN': train_metrics.get('true_negatives', ''),
            'train_FP': train_metrics.get('false_positives', ''),
            'train_FN': train_metrics.get('false_negatives', ''),
            'train_support': train_metrics.get('total_samples', ''),
            
            # Validation metrics
            'val_accuracy': val_metrics.get('accuracy', ''),
            'val_precision': val_metrics.get('macro_precision', ''),
            'val_recall': val_metrics.get('macro_recall', ''),
            'val_f1': val_metrics.get('macro_f1', ''),
            'val_TP': val_metrics.get('true_positives', ''),
            'val_TN': val_metrics.get('true_negatives', ''),
            'val_FP': val_metrics.get('false_positives', ''),
            'val_FN': val_metrics.get('false_negatives', ''),
            'val_support': val_metrics.get('total_samples', ''),
        }
        
        return result
        
    except Exception as e:
        print(f"⚠️  Error processing {file_path}: {e}")
        return None

def find_evaluation_files(wandb_dir: str) -> List[str]:
    """
    Find all comprehensive_evaluation_results.json files in wandb run directories.
    
    Args:
        wandb_dir: Path to wandb directory
        
    Returns:
        List of file paths
    """
    pattern = os.path.join(wandb_dir, "run-*/files/comprehensive_evaluation_results.json")
    files = glob.glob(pattern)
    return files

def parse_all_results(wandb_dir: str, output_file: str) -> None:
    """
    Parse all comprehensive evaluation results and save to CSV.
    
    Args:
        wandb_dir: Path to wandb directory
        output_file: Output CSV file path
    """
    print(f"🔍 Searching for evaluation files in: {wandb_dir}")
    
    # Find all evaluation files
    files = find_evaluation_files(wandb_dir)
    print(f"📊 Found {len(files)} evaluation files")
    
    if not files:
        print("❌ No evaluation files found!")
        return
    
    # Process each file
    results = []
    for file_path in files:
        print(f"📖 Processing: {os.path.basename(os.path.dirname(os.path.dirname(file_path)))}")
        result = extract_metrics_from_file(file_path)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No valid results found!")
        return
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Sort by validation accuracy (descending) to see best results first
    df = df.sort_values('val_accuracy', ascending=False)
    
    print(f"\n💾 Saving {len(df)} results to: {output_file}")
    df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   Total runs: {len(df)}")
    print(f"   Best validation accuracy: {df['val_accuracy'].max():.4f}")
    print(f"   Best validation F1: {df['val_f1'].max():.4f}")
    print(f"   Average validation accuracy: {df['val_accuracy'].mean():.4f}")
    print(f"   Average validation F1: {df['val_f1'].mean():.4f}")
    
    # Show top 5 runs
    print(f"\n🏆 Top 5 Runs by Validation Accuracy:")
    print("=" * 80)
    top_5 = df.head(5)[['run_name', 'wandb_run_id', 'val_accuracy', 'val_f1', 'val_precision', 'val_recall']]
    print(top_5.to_string(index=False))
    
    print(f"\n✅ Results saved to: {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Parse wandb evaluation results into CSV")
    parser.add_argument("--wandb-dir", default="train/wandb", 
                       help="Path to wandb directory")
    parser.add_argument("--output-file", default="wandb_evaluation_results.csv",
                       help="Output CSV file path")
    
    args = parser.parse_args()
    
    # Check if wandb directory exists
    if not os.path.exists(args.wandb_dir):
        print(f" Wandb directory not found: {args.wandb_dir}")
        return
    
    # Parse all results
    parse_all_results(args.wandb_dir, args.output_file)

if __name__ == "__main__":
    main() 