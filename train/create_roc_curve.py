#!/usr/bin/env python3
"""
Create ROC curve for the best performing model.

This script loads the comprehensive evaluation results from the best model
and creates a ROC curve using the probability scores.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd
import argparse
from pathlib import Path

def load_best_model_results(wandb_dir: str) -> tuple:
    """
    Load the comprehensive evaluation results from the best performing model.
    
    Args:
        wandb_dir: Path to wandb directory
        
    Returns:
        Tuple of (y_true, y_scores, run_info)
    """
    # Find all evaluation files
    pattern = Path(wandb_dir) / "run-*/files/comprehensive_evaluation_results.json"
    files = list(pattern.parent.parent.glob("run-*/files/comprehensive_evaluation_results.json"))
    
    if not files:
        raise FileNotFoundError("No evaluation files found!")
    
    # Find the best model (highest validation accuracy)
    best_file = None
    best_accuracy = -1
    best_run_info = {}
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            val_accuracy = data.get('validation_metrics', {}).get('accuracy', 0)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_file = file_path
                best_run_info = data.get('run_info', {})
                
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            continue
    
    if best_file is None:
        raise ValueError("No valid evaluation files found!")
    
    print(f"🏆 Best model: {best_run_info.get('run_name', 'Unknown')}")
    print(f"📊 Validation accuracy: {best_accuracy:.4f}")
    print(f"📁 File: {best_file}")
    
    # Load the best model's results
    with open(best_file, 'r') as f:
        data = json.load(f)
    
    # Extract results
    results = data.get('validation_metrics', {}).get('results', [])
    
    if not results:
        raise ValueError("No results found in the evaluation file!")
    
    # Extract true labels and probability scores
    y_true = []
    y_scores = []  # Probability of class A (lie)
    
    for result in results:
        true_label = result.get('true', '')
        p_a = result.get('p_a', 0.0)  # Probability of class A
        
        # Convert labels to binary (A=1, B=0)
        if true_label == 'A':
            y_true.append(1)
        elif true_label == 'B':
            y_true.append(0)
        else:
            continue  # Skip invalid labels
        
        y_scores.append(p_a)
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    print(f"📈 Total samples: {len(y_true)}")
    print(f"📈 Class A (lies): {np.sum(y_true)}")
    print(f"📈 Class B (truth): {len(y_true) - np.sum(y_true)}")
    
    return y_true, y_scores, best_run_info

def create_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, run_info: dict, output_file: str = None) -> None:
    """
    Create and save a ROC curve.
    
    Args:
        y_true: True binary labels
        y_scores: Probability scores for positive class
        run_info: Information about the model run
        output_file: Output file path for the plot
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate AUC using sklearn
    auc_score = roc_auc_score(y_true, y_scores)
    
    print(f"📊 ROC AUC Score: {auc_score:.4f}")
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.4f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier (AUC = 0.5)')
    
    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(f'ROC Curve - {run_info.get("run_name", "Best Model")}\n'
              f'Model: {run_info.get("model_name", "Unknown")}', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add some key points
    # Find threshold closest to 0.5
    threshold_idx = np.argmin(np.abs(thresholds - 0.5))
    plt.plot(fpr[threshold_idx], tpr[threshold_idx], 'ro', markersize=8, 
             label=f'Threshold = 0.5\n(FPR={fpr[threshold_idx]:.3f}, TPR={tpr[threshold_idx]:.3f})')
    
    # Add text with model info
    info_text = f"Run ID: {run_info.get('wandb_run_id', 'Unknown')}\n"
    info_text += f"Timestamp: {run_info.get('timestamp', 'Unknown')}\n"
    info_text += f"Total samples: {len(y_true)}"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    if output_file is None:
        output_file = f"roc_curve_{run_info.get('run_name', 'best_model')}.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 ROC curve saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    # Print some additional statistics
    print(f"\n📈 ROC Curve Statistics:")
    print(f"   AUC Score: {auc_score:.4f}")
    print(f"   Total samples: {len(y_true)}")
    print(f"   Positive samples (lies): {np.sum(y_true)}")
    print(f"   Negative samples (truth): {len(y_true) - np.sum(y_true)}")
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    print(f"\n🎯 Optimal Threshold Analysis:")
    print(f"   Optimal threshold: {optimal_threshold:.4f}")
    print(f"   Optimal FPR: {optimal_fpr:.4f}")
    print(f"   Optimal TPR: {optimal_tpr:.4f}")
    print(f"   Youden's J: {j_scores[optimal_idx]:.4f}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create ROC curve for best model")
    parser.add_argument("--wandb-dir", default="train/wandb", 
                       help="Path to wandb directory")
    parser.add_argument("--output-file", default=None,
                       help="Output file path for the ROC curve plot")
    
    args = parser.parse_args()
    
    # Check if wandb directory exists
    if not Path(args.wandb_dir).exists():
        print(f"❌ Wandb directory not found: {args.wandb_dir}")
        return
    
    try:
        # Load the best model results
        y_true, y_scores, run_info = load_best_model_results(args.wandb_dir)
        
        # Create ROC curve
        create_roc_curve(y_true, y_scores, run_info, args.output_file)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    main() 