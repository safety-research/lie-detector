#!/usr/bin/env python3
"""
Comprehensive analysis of k-fold training results.
Generates JSON files with best folds per model and top-K generalizing pairs.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from PIL import Image
import glob


def load_training_results(results_file):
    """Load training results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def load_training_summary(input_path):
    """Load training summary to get fold sizes."""
    summary_file = Path(input_path) / "training_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    return {}


def extract_all_metrics(all_training_results, training_summary):
    """
    Extract all metrics for each train-test pair.
    
    Returns:
        dict: Dictionary with train_fold as key and test_fold results as values
    """
    # Get all fold names and sort alphabetically
    train_folds = sorted(list(all_training_results.keys()))
    test_folds = sorted(list(all_training_results[train_folds[0]].keys()))
    
    all_pairs = []
    
    for train_fold in train_folds:
        for test_fold in test_folds:
            if train_fold == test_fold:
                continue  # Skip diagonal (same fold train/test)
            
            results = all_training_results[train_fold][test_fold]
            
            # Extract basic metrics
            accuracy = results.get('accuracy', 0.0)
            f1_score = results.get('f1_weighted', 0.0)
            
            # Extract AUC
            try:
                if 'roc_data' in results:
                    y_true = results['roc_data']['y_true']
                    y_prob = results['roc_data']['y_prob']
                    # Flip labels and probabilities for A (lie) as positive class
                    y_true_flipped = [1 - label for label in y_true]
                    y_prob_a = [1 - p for p in y_prob]
                    auc_score = roc_auc_score(y_true_flipped, y_prob_a)
                else:
                    auc_score = 0.0
            except Exception as e:
                print(f"Error calculating AUC for {train_fold} -> {test_fold}: {e}")
                auc_score = 0.0
            
            # Calculate composite score
            composite_score = (auc_score + accuracy + f1_score) / 3
            
            all_pairs.append({
                'train_fold': train_fold,
                'test_fold': test_fold,
                'auc': auc_score,
                'accuracy': accuracy,
                'f1': f1_score,
                'composite_score': composite_score,
                'num_examples': results.get('total_examples', 0)
            })
    
    return all_pairs


def find_best_folds_per_model(all_pairs):
    """
    Find the best training fold for each model based on average performance.
    
    Returns:
        dict: Best fold analysis per model
    """
    # Group by train_fold and calculate averages
    fold_performance = defaultdict(list)
    for pair in all_pairs:
        fold_performance[pair['train_fold']].append(pair)
    
    # Calculate average metrics for each training fold
    fold_averages = {}
    for fold, pairs in fold_performance.items():
        avg_auc = np.mean([p['auc'] for p in pairs])
        avg_accuracy = np.mean([p['accuracy'] for p in pairs])
        avg_f1 = np.mean([p['f1'] for p in pairs])
        avg_composite = np.mean([p['composite_score'] for p in pairs])
        
        # Calculate standard deviations
        std_auc = np.std([p['auc'] for p in pairs])
        std_accuracy = np.std([p['accuracy'] for p in pairs])
        std_f1 = np.std([p['f1'] for p in pairs])
        
        fold_averages[fold] = {
            'avg_auc': avg_auc,
            'avg_accuracy': avg_accuracy,
            'avg_f1': avg_f1,
            'avg_composite': avg_composite,
            'std_auc': std_auc,
            'std_accuracy': std_accuracy,
            'std_f1': std_f1,
            'num_test_folds': len(pairs),
            'test_folds': [p['test_fold'] for p in pairs]
        }
    
    # Find best fold
    best_fold = max(fold_averages.keys(), key=lambda x: fold_averages[x]['avg_composite'])
    
    return {
        'best_fold': best_fold,
        'best_metrics': fold_averages[best_fold],
        'all_folds': fold_averages
    }


def find_top_k_generalizing_pairs(all_pairs, k=5):
    """
    Find the top K train-test pairs with highest AUC where train ≠ test.
    
    Returns:
        list: Top K pairs sorted by AUC
    """
    # Sort by AUC (descending)
    sorted_pairs = sorted(all_pairs, key=lambda x: x['auc'], reverse=True)
    
    # Return top K
    return sorted_pairs[:k]


def create_comprehensive_json(model_results_dir, output_dir, model_name, title_name):
    """
    Create comprehensive JSON analysis for a single model.
    """
    # Find results file
    results_file = Path(model_results_dir) / "train_one_eval_all_results.json"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None
    
    # Load results
    results = load_training_results(results_file)
    all_training_results = results.get('all_training_results', {})
    
    if not all_training_results:
        print("No training results found in the file")
        return None
    
    # Get input_path from results to load training summary
    input_path = results.get('input_path', str(model_results_dir))
    training_summary = load_training_summary(input_path)
    
    print(f"Analyzing results for {model_name}...")
    
    # Extract all metrics
    all_pairs = extract_all_metrics(all_training_results, training_summary)
    
    # Find best folds
    best_folds_analysis = find_best_folds_per_model(all_pairs)
    
    # Find top K generalizing pairs
    top_k_pairs = find_top_k_generalizing_pairs(all_pairs, k=5)
    
    # Create comprehensive analysis
    analysis = {
        'model_name': model_name,
        'title_name': title_name,
        'timestamp': results.get('timestamp', ''),
        'best_folds_analysis': best_folds_analysis,
        'top_5_generalizing_pairs': top_k_pairs,
        'all_pairs': all_pairs,
        'summary': {
            'total_train_test_pairs': len(all_pairs),
            'best_training_fold': best_folds_analysis['best_fold'],
            'best_composite_score': best_folds_analysis['best_metrics']['avg_composite'],
            'best_avg_auc': best_folds_analysis['best_metrics']['avg_auc'],
            'best_avg_accuracy': best_folds_analysis['best_metrics']['avg_accuracy'],
            'best_avg_f1': best_folds_analysis['best_metrics']['avg_f1']
        }
    }
    
    # Save to JSON
    json_path = output_dir / f"comprehensive_analysis_{model_name.replace('/', '_')}.json"
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Comprehensive analysis saved to: {json_path}")
    
    return analysis


def create_separate_metric_heatmaps(all_analyses, output_dir):
    """
    Create separate heatmaps for AUC, Accuracy, and F1 metrics.
    Each cell shows the average performance of a training fold across all test folds.
    """
    # Prepare data for plotting
    plot_data = []
    for model_name, analysis in all_analyses.items():
        for fold, metrics in analysis['best_folds_analysis']['all_folds'].items():
            plot_data.append({
                'Model': model_name,
                'Training Fold': fold,
                'Avg AUC': metrics['avg_auc'],
                'Avg Accuracy': metrics['avg_accuracy'],
                'Avg F1': metrics['avg_f1']
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Define custom sorting for models (4b, 12b, 27b)
    def sort_models(model_name):
        if '4b' in model_name:
            return 1
        elif '12b' in model_name:
            return 2
        elif '27b' in model_name:
            return 3
        else:
            return 4  # Any other models go last
    
    # Create separate heatmaps for each metric
    metrics_config = [
        ('Avg AUC', 'Blues', 'AUC Score'),
        ('Avg Accuracy', 'Greens', 'Accuracy Score'),
        ('Avg F1', 'Reds', 'F1 Score')
    ]
    
    for metric, colormap, title_metric in metrics_config:
        plt.figure(figsize=(14, 10))
        
        # Pivot data for heatmap
        pivot_df = plot_df.pivot(index='Training Fold', columns='Model', values=metric)
        
        # Sort columns by model size (4b, 12b, 27b)
        sorted_columns = sorted(pivot_df.columns, key=sort_models)
        pivot_df = pivot_df[sorted_columns]
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap=colormap, 
                    cbar_kws={'label': title_metric}, linewidths=0.5,
                    vmin=0.3, vmax=0.8)  # Set reasonable range for better contrast
        
        plt.title(f'Cross-Model Training Fold Performance: {title_metric}\n(Average across all test folds)', 
                  fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Training Fold', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"cross_model_{metric.lower().replace(' ', '_')}_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"{title_metric} heatmap saved to: {plot_path}")
        plt.close()
    
    return plot_df


def create_top_k_summary(all_analyses, output_dir):
    """
    Create a summary of top-K generalizing pairs across all models.
    """
    all_top_pairs = []
    
    for model_name, analysis in all_analyses.items():
        for pair in analysis['top_5_generalizing_pairs']:
            all_top_pairs.append({
                'model': model_name,
                'train_fold': pair['train_fold'],
                'test_fold': pair['test_fold'],
                'auc': pair['auc'],
                'accuracy': pair['accuracy'],
                'f1': pair['f1'],
                'composite_score': pair['composite_score']
            })
    
    # Sort by AUC across all models
    all_top_pairs.sort(key=lambda x: x['auc'], reverse=True)
    
    # Create summary
    summary = {
        'top_10_generalizing_pairs_across_all_models': all_top_pairs[:10],
        'model_summaries': {}
    }
    
    for model_name, analysis in all_analyses.items():
        summary['model_summaries'][model_name] = {
            'best_training_fold': analysis['best_folds_analysis']['best_fold'],
            'best_composite_score': analysis['best_folds_analysis']['best_metrics']['avg_composite'],
            'best_avg_auc': analysis['best_folds_analysis']['best_metrics']['avg_auc'],
            'top_5_pairs': analysis['top_5_generalizing_pairs']
        }
    
    # Save to JSON
    summary_path = output_dir / "top_k_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Top-K summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "="*120)
    print("TOP 10 GENERALIZING PAIRS ACROSS ALL MODELS")
    print("="*120)
    print(f"{'Rank':<4} {'Model':<20} {'Train':<12} {'Test':<12} {'AUC':<8} {'Acc':<8} {'F1':<8} {'Comp':<8}")
    print("-" * 120)
    
    for i, pair in enumerate(all_top_pairs[:10], 1):
        print(f"{i:<4} {pair['model']:<20} {pair['train_fold']:<12} {pair['test_fold']:<12} "
              f"{pair['auc']:<8.3f} {pair['accuracy']:<8.3f} {pair['f1']:<8.3f} {pair['composite_score']:<8.3f}")
    
    print("="*120)
    
    return summary


def create_scaling_grid_comprehensive(output_dir):
    """
    Create a grid plot combining the three comprehensive analysis PNG files.
    
    Args:
        output_dir: Path to the output directory containing the PNG files
    """
    # Define the three PNG files to combine
    png_files = [
        "cross_model_avg_auc_heatmap.png",
        "cross_model_avg_accuracy_heatmap.png", 
        "cross_model_avg_f1_heatmap.png"
    ]
    
    # Check if all files exist
    missing_files = []
    for png_file in png_files:
        file_path = output_dir / png_file
        if not file_path.exists():
            missing_files.append(png_file)
    
    if missing_files:
        print(f"Warning: Missing PNG files: {missing_files}")
        print("Cannot create scaling grid without all required files.")
        return
    
    # Load and resize all images to a standard size
    images = []
    target_size = (800, 600)  # Standard size for all images
    
    for png_file in png_files:
        file_path = output_dir / png_file
        try:
            img = Image.open(file_path)
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(img)
            print(f"Loaded and resized: {png_file}")
        except Exception as e:
            print(f"Error loading {png_file}: {e}")
            return
    
    # Create a horizontal grid (1 row, 3 columns)
    grid_width = target_size[0] * 3
    grid_height = target_size[1]
    
    # Create the combined image
    combined_img = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Paste images horizontally
    for i, img in enumerate(images):
        x_offset = i * target_size[0]
        combined_img.paste(img, (x_offset, 0))
    
    # Save the combined image
    output_path = output_dir / "scaling_grid_comprehensive.png"
    combined_img.save(output_path, 'PNG', quality=95)
    
    print(f"Scaling grid comprehensive saved to: {output_path}")
    print(f"Grid layout: AUC | Accuracy | F1 (horizontal row)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Comprehensive analysis of k-fold training results")
    parser.add_argument("--timestamp_dir", type=str, required=True,
                       help="Timestamp directory containing model results (e.g., 20250725_220131)")
    
    args = parser.parse_args()
    
    # Construct full paths
    timestamp_dir = Path("/workspace/lie-detector/train/outputs") / args.timestamp_dir
    output_dir = Path("visualizations") / args.timestamp_dir / "comprehensive_analysis"
    
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
    all_analyses = {}
    
    for model_dir in model_dirs:
        print(f"\n{'='*80}")
        print(f"Analyzing model: {model_dir.name}")
        print(f"{'='*80}")
        
        # Extract model name for titles
        model_name = model_dir.name.replace('openrouter_google_gemma-3-', 'google/gemma-3-')
        
        # Clean up the title: replace _ and - with spaces, remove everything after _train_one
        title_name = model_name
        if '_train_one' in title_name:
            title_name = title_name.split('_train_one')[0]
        title_name = title_name.replace('_', ' ').replace('-', ' ')
        
        # Analyze results for this model
        analysis = create_comprehensive_json(model_dir, output_dir, model_name, title_name)
        if analysis:
            successful_models += 1
            all_analyses[title_name] = analysis
            print(f"Successfully analyzed {model_dir.name}")
        else:
            print(f"Failed to analyze {model_dir.name}")
    
    # Create cross-model visualizations and summaries
    if all_analyses:
        print(f"\n{'='*80}")
        print("Creating cross-model analysis...")
        print(f"{'='*80}")
        
        # Create separate metric heatmaps
        create_separate_metric_heatmaps(all_analyses, output_dir)
        
        # Create top-K summary
        create_top_k_summary(all_analyses, output_dir)
        
        # Create scaling grid comprehensive
        print(f"\nCreating scaling grid comprehensive...")
        create_scaling_grid_comprehensive(output_dir)
    
    print(f"\n{'='*80}")
    print(f"Summary: Successfully analyzed {successful_models}/{len(model_dirs)} models")
    print(f"Comprehensive analysis saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main() 