#!/usr/bin/env python3
"""
Visualize k-fold training results for all models in a timestamp directory.

This script reads the training results from all model subdirectories in a timestamp
directory and creates heatmap visualizations (AUC, Accuracy, F1 scores) for each model.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
import argparse
from pathlib import Path
from datetime import datetime

def load_training_results(results_file):
    """Load training results from JSON file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def load_training_summary(input_path):
    """Load training summary to get correct training set sizes."""
    # Extract the model directory from the input path
    # input_path should be something like: /workspace/lie-detector/organized_balanced_training_cleaned_20250722_135859/openrouter_google_gemma-3-12b-it/folds_colors_chat_format
    # We need to go up one level to get to the model directory
    model_dir = Path(input_path).parent
    summary_file = model_dir / "training_summary.json"
    
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        return summary
    else:
        print(f"Warning: Training summary not found at {summary_file}")
        return None

def extract_auc_scores(all_training_results, training_summary=None):
    """
    Extract AUC scores from training results.
    
    Args:
        all_training_results: Dictionary containing results for each training fold
        training_summary: Training summary containing correct training set sizes
        
    Returns:
        auc_matrix: 2D numpy array of AUC scores
        train_labels: List of training fold names
        test_labels: List of test fold names
    """
    from sklearn.metrics import roc_auc_score
    
    # Get all fold names
    train_folds = list(all_training_results.keys())
    test_folds = list(all_training_results[train_folds[0]].keys())
    
    # Create AUC matrix
    auc_matrix = np.zeros((len(train_folds), len(test_folds)))
    
    for i, train_fold in enumerate(train_folds):
        for j, test_fold in enumerate(test_folds):
            if test_fold in all_training_results[train_fold]:
                # Extract ROC data and calculate AUC
                results = all_training_results[train_fold][test_fold]
                if 'roc_data' in results and 'y_true' in results['roc_data'] and 'y_prob' in results['roc_data']:
                    y_true = results['roc_data']['y_true']
                    y_prob = results['roc_data']['y_prob']
                    try:
                        # The y_prob values are probabilities of class B (truthful)
                        # For AUC calculation, we need probabilities of the positive class (A/liar)
                        # So we use 1 - y_prob to get probability of A
                        y_prob_a = [1 - p for p in y_prob]
                        auc_score = roc_auc_score(y_true, y_prob_a)
                        auc_matrix[i, j] = auc_score
                    except Exception as e:
                        print(f"Error calculating AUC for {train_fold} -> {test_fold}: {e}")
                        auc_matrix[i, j] = np.nan
                else:
                    auc_matrix[i, j] = np.nan
            else:
                auc_matrix[i, j] = np.nan
    
    # Create labels with support counts
    train_labels = []
    test_labels = []
    
    for fold in train_folds:
        # Get training set size from training summary
        if training_summary and 'folds' in training_summary and 'colors' in training_summary['folds']:
            categories = training_summary['folds']['colors']['categories']
            if fold in categories:
                train_size = categories[fold]['train']
            else:
                train_size = all_training_results[fold][test_folds[0]].get('total_examples', 0)
        else:
            train_size = all_training_results[fold][test_folds[0]].get('total_examples', 0)
        
        label = f"{fold.replace('_', ' ')}\n({train_size})"
        train_labels.append(label)
    
    for fold in test_folds:
        # Get test set size from training summary
        if training_summary and 'folds' in training_summary and 'colors' in training_summary['folds']:
            categories = training_summary['folds']['colors']['categories']
            if fold in categories:
                test_size = categories[fold]['test']
            else:
                test_size = all_training_results[train_folds[0]][fold].get('total_examples', 0)
        else:
            test_size = all_training_results[train_folds[0]][fold].get('total_examples', 0)
        
        label = f"{fold.replace('_', ' ')}\n({test_size})"
        test_labels.append(label)
    
    return auc_matrix, train_labels, test_labels

def create_auc_heatmap(auc_matrix, train_labels, test_labels, output_path, title="AUC Scores: Train vs Test Sets"):
    """
    Create and save AUC heatmap.
    
    Args:
        auc_matrix: 2D numpy array of AUC scores
        train_labels: List of training fold names
        test_labels: List of test fold names
        output_path: Path to save the plot
        title: Plot title
    """
    # Create DataFrame for seaborn
    df = pd.DataFrame(auc_matrix, index=train_labels, columns=test_labels)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(df, 
                annot=True, 
                fmt=".3f", 
                cmap="Blues", 
                cbar_kws={'label': 'AUC'},
                vmin=0.0, 
                vmax=1.0,
                center=0.5,
                square=True)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Test Sets", fontsize=12)
    plt.ylabel("Train Sets", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved AUC heatmap to: {output_path}")

def create_accuracy_heatmap(all_training_results, output_path, training_summary=None, title="Accuracy Scores: Train vs Test Sets"):
    """
    Create and save accuracy heatmap as an alternative to AUC.
    
    Args:
        all_training_results: Dictionary containing results for each training fold
        output_path: Path to save the plot
        training_summary: Training summary containing correct training set sizes
        title: Plot title
    """
    # Get all fold names
    train_folds = list(all_training_results.keys())
    test_folds = list(all_training_results[train_folds[0]].keys())
    
    # Create accuracy matrix
    acc_matrix = np.zeros((len(train_folds), len(test_folds)))
    
    for i, train_fold in enumerate(train_folds):
        for j, test_fold in enumerate(test_folds):
            if test_fold in all_training_results[train_fold]:
                # Extract accuracy score
                results = all_training_results[train_fold][test_fold]
                acc_score = results.get('accuracy', 0.0)
                acc_matrix[i, j] = acc_score
            else:
                acc_matrix[i, j] = np.nan
    
    # Create labels with support counts
    train_labels = []
    test_labels = []
    
    for fold in train_folds:
        # Get training set size from training summary
        if training_summary and 'folds' in training_summary and 'colors' in training_summary['folds']:
            categories = training_summary['folds']['colors']['categories']
            if fold in categories:
                train_size = categories[fold]['train']
            else:
                train_size = all_training_results[fold][test_folds[0]].get('total_examples', 0)
        else:
            train_size = all_training_results[fold][test_folds[0]].get('total_examples', 0)
        
        label = f"{fold.replace('_', ' ')}\n({train_size})"
        train_labels.append(label)
    
    for fold in test_folds:
        # Get test set size from training summary
        if training_summary and 'folds' in training_summary and 'colors' in training_summary['folds']:
            categories = training_summary['folds']['colors']['categories']
            if fold in categories:
                test_size = categories[fold]['test']
            else:
                test_size = all_training_results[train_folds[0]][fold].get('total_examples', 0)
        else:
            test_size = all_training_results[train_folds[0]][fold].get('total_examples', 0)
        
        label = f"{fold.replace('_', ' ')}\n({test_size})"
        test_labels.append(label)
    
    # Create DataFrame for seaborn
    df = pd.DataFrame(acc_matrix, index=train_labels, columns=test_labels)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(df, 
                annot=True, 
                fmt=".3f", 
                cmap="Greens", 
                cbar_kws={'label': 'Accuracy'},
                vmin=0.0, 
                vmax=1.0,
                center=0.5,
                square=True)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Test Sets", fontsize=12)
    plt.ylabel("Train Sets", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved accuracy heatmap to: {output_path}")

def create_f1_heatmap(all_training_results, output_path, training_summary=None, title="F1 Scores: Train vs Test Sets"):
    """
    Create and save F1 score heatmap.
    
    Args:
        all_training_results: Dictionary containing results for each training fold
        output_path: Path to save the plot
        training_summary: Training summary containing correct training set sizes
        title: Plot title
    """
    # Get all fold names
    train_folds = list(all_training_results.keys())
    test_folds = list(all_training_results[train_folds[0]].keys())
    
    # Create F1 matrix
    f1_matrix = np.zeros((len(train_folds), len(test_folds)))
    
    for i, train_fold in enumerate(train_folds):
        for j, test_fold in enumerate(test_folds):
            if test_fold in all_training_results[train_fold]:
                # Extract F1 score
                results = all_training_results[train_fold][test_fold]
                f1_score = results.get('f1_weighted', 0.0)
                f1_matrix[i, j] = f1_score
            else:
                f1_matrix[i, j] = np.nan
    
    # Create labels with support counts
    train_labels = []
    test_labels = []
    
    for fold in train_folds:
        # Get training set size from training summary
        if training_summary and 'folds' in training_summary and 'colors' in training_summary['folds']:
            categories = training_summary['folds']['colors']['categories']
            if fold in categories:
                train_size = categories[fold]['train']
            else:
                train_size = all_training_results[fold][test_folds[0]].get('total_examples', 0)
        else:
            train_size = all_training_results[fold][test_folds[0]].get('total_examples', 0)
        
        label = f"{fold.replace('_', ' ')}\n({train_size})"
        train_labels.append(label)
    
    for fold in test_folds:
        # Get test set size from training summary
        if training_summary and 'folds' in training_summary and 'colors' in training_summary['folds']:
            categories = training_summary['folds']['colors']['categories']
            if fold in categories:
                test_size = categories[fold]['test']
            else:
                test_size = all_training_results[train_folds[0]][fold].get('total_examples', 0)
        else:
            test_size = all_training_results[train_folds[0]][fold].get('total_examples', 0)
        
        label = f"{fold.replace('_', ' ')}\n({test_size})"
        test_labels.append(label)
    
    # Create DataFrame for seaborn
    df = pd.DataFrame(f1_matrix, index=train_labels, columns=test_labels)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(df, 
                annot=True, 
                fmt=".3f", 
                cmap="Reds", 
                cbar_kws={'label': 'F1 Score'},
                vmin=0.0, 
                vmax=1.0,
                center=0.5,
                square=True)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Test Sets", fontsize=12)
    plt.ylabel("Train Sets", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved F1 heatmap to: {output_path}")

def visualize_model_results(model_results_dir, output_dir, model_name, title_name):
    """
    Create visualizations for a single model.
    
    Args:
        model_results_dir: Path to the model's results directory
        output_dir: Path to save visualizations
        model_name: Name of the model for titles
    """
    # Find results file
    results_file = Path(model_results_dir) / "train_one_eval_all_results.json"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return False
    
    # Load results
    print(f"Loading results from: {results_file}")
    results = load_training_results(results_file)
    
    # Extract training results
    all_training_results = results.get('all_training_results', {})
    
    if not all_training_results:
        print("No training results found in the file")
        return False
    
    # Get input_path from results to load training summary
    input_path = results.get('input_path', str(model_results_dir))
    training_summary = load_training_summary(input_path)
    
    # Create visualizations
    print(f"Creating visualizations for {model_name}...")
    
    # Sanitize model name for filenames (replace / with _)
    model_name_safe = model_name.replace('/', '_')
    
    # 1. AUC Heatmap (if AUC scores are available)
    try:
        auc_matrix, train_labels, test_labels = extract_auc_scores(all_training_results, training_summary)
        if not np.all(np.isnan(auc_matrix)):
            auc_output = output_dir / f"auc_heatmap_{model_name_safe}.png"
            create_auc_heatmap(auc_matrix, train_labels, test_labels, auc_output, 
                             f"AUC Scores: {title_name}")
        else:
            print("No AUC scores found in results")
    except Exception as e:
        print(f"Error creating AUC heatmap: {e}")
    
    # 2. Accuracy Heatmap
    try:
        acc_output = output_dir / f"accuracy_heatmap_{model_name_safe}.png"
        create_accuracy_heatmap(all_training_results, acc_output, training_summary, 
                               f"Accuracy Scores: {title_name}")
    except Exception as e:
        print(f"Error creating accuracy heatmap: {e}")
    
    # 3. F1 Score Heatmap
    try:
        f1_output = output_dir / f"f1_heatmap_{model_name_safe}.png"
        create_f1_heatmap(all_training_results, f1_output, training_summary, 
                          f"F1 Scores: {title_name}")
    except Exception as e:
        print(f"Error creating F1 heatmap: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Visualize k-fold training results for all models")
    parser.add_argument("--timestamp_dir", type=str, required=True,
                        help="Timestamp directory containing model results (e.g., 20250725_220131)")
    parser.add_argument("--base_output_dir", type=str, default="./visualizations",
                        help="Base output directory for visualizations")
    
    args = parser.parse_args()
    
    # Construct full paths
    timestamp_dir = Path("/workspace/lie-detector/train/outputs") / args.timestamp_dir
    output_dir = Path(args.base_output_dir) / args.timestamp_dir
    
    if not timestamp_dir.exists():
        print(f"Timestamp directory not found: {timestamp_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing timestamp directory: {timestamp_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get all model subdirectories
    model_dirs = [d for d in timestamp_dir.iterdir() if d.is_dir() and d.name.startswith('openrouter_google_gemma-3-')]
    
    if not model_dirs:
        print(f"No model directories found in {timestamp_dir}")
        return
    
    print(f"Found {len(model_dirs)} model directories: {[d.name for d in model_dirs]}")
    
    # Process each model
    successful_models = 0
    for model_dir in model_dirs:
        print(f"\n{'='*80}")
        print(f"Processing model: {model_dir.name}")
        print(f"{'='*80}")
        
        # Create model-specific output directory
        model_output_dir = output_dir / model_dir.name
        model_output_dir.mkdir(exist_ok=True)
        
        # Extract model name for titles
        model_name = model_dir.name.replace('openrouter_google_gemma-3-', 'google/gemma-3-')
        
        # Clean up the title: replace _ and - with spaces, remove everything after _train_one
        title_name = model_name
        if '_train_one' in title_name:
            title_name = title_name.split('_train_one')[0]
        title_name = title_name.replace('_', ' ').replace('-', ' ')
        
        # Create visualizations for this model
        if visualize_model_results(model_dir, model_output_dir, model_name, title_name):
            successful_models += 1
            print(f"Successfully created visualizations for {model_dir.name}")
        else:
            print(f"Failed to create visualizations for {model_dir.name}")
    
    print(f"\n{'='*80}")
    print(f"Summary: Successfully processed {successful_models}/{len(model_dirs)} models")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 