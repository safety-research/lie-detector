#!/usr/bin/env python3
"""
Dataset balancing script.

This script:
1. Reads all JSONL files from the organized evaluation directory
2. For each task within each model, counts True/False did_lie values
3. Takes the minimum count and randomly samples that many from each class
4. Creates a new balanced directory structure with the same organization
"""

import json
import random
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import shutil
import matplotlib.pyplot as plt
import numpy as np

def load_jsonl_file(file_path: Path) -> List[Dict]:
    """Load a JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl_file(data: List[Dict], file_path: Path):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_task_name_from_filename(filename: str) -> str:
    """Extract task name from filename."""
    # Remove the deduplicated_ prefix and model suffix
    if filename.startswith('deduplicated_'):
        filename = filename[13:]  # Remove 'deduplicated_'
    
    # Find the last occurrence of the model name pattern
    if '_google_gemma-' in filename:
        filename = filename.split('_google_gemma-')[0]
    elif '_openrouter_google_gemma-' in filename:
        filename = filename.split('_openrouter_google_gemma-')[0]
    
    return filename

def balance_dataset(data: List[Dict]) -> List[Dict]:
    """Balance dataset by ensuring equal True/False representation."""
    # Separate data by did_lie value
    true_samples = [item for item in data if item.get('did_lie') == True]
    false_samples = [item for item in data if item.get('did_lie') == False]
    
    # Count samples in each class
    true_count = len(true_samples)
    false_count = len(false_samples)
    
    print(f"    Original counts - True: {true_count}, False: {false_count}")
    
    # Find the minimum count
    min_count = min(true_count, false_count)
    
    if min_count == 0:
        print(f"    Warning: One class has 0 samples, skipping balancing")
        return data
    
    # Randomly sample the minimum count from each class
    balanced_data = []
    
    if true_count > 0:
        balanced_true = random.sample(true_samples, min_count)
        balanced_data.extend(balanced_true)
    
    if false_count > 0:
        balanced_false = random.sample(false_samples, min_count)
        balanced_data.extend(balanced_false)
    
    # Shuffle the final dataset
    random.shuffle(balanced_data)
    
    print(f"    Balanced counts - True: {min_count}, False: {min_count}, Total: {len(balanced_data)}")
    
    return balanced_data

def process_model_directory(model_dir: Path, output_base_dir: Path):
    """Process all files in a model directory and create balanced versions."""
    print(f"\nProcessing model: {model_dir.name}")
    
    # Create output directory for this model
    output_model_dir = output_base_dir / model_dir.name
    output_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Group files by task
    task_files = defaultdict(list)
    
    for file_path in model_dir.iterdir():
        if file_path.is_file() and file_path.suffix == '.jsonl':
            task_name = get_task_name_from_filename(file_path.name)
            task_files[task_name].append(file_path)
    
    print(f"  Found {len(task_files)} unique tasks")
    
    # Process each task
    for task_name, file_paths in task_files.items():
        print(f"\n  Processing task: {task_name}")
        
        # Load and combine all files for this task
        all_data = []
        for file_path in file_paths:
            data = load_jsonl_file(file_path)
            all_data.extend(data)
            print(f"    Loaded {len(data)} samples from {file_path.name}")
        
        # Balance the combined dataset
        balanced_data = balance_dataset(all_data)
        
        # Save balanced dataset
        output_filename = f"balanced_{file_paths[0].name}"
        output_path = output_model_dir / output_filename
        save_jsonl_file(balanced_data, output_path)
        print(f"    Saved balanced dataset: {output_filename}")
    
    # Copy evaluation scripts
    for file_path in model_dir.iterdir():
        if file_path.is_file() and file_path.suffix == '.py':
            shutil.copy2(file_path, output_model_dir / file_path.name)
            print(f"    Copied: {file_path.name}")

def create_summary_json(input_dir: Path, output_dir: Path, model_dirs: List[str]) -> Dict:
    """Create a summary JSON with overall counts and task breakdowns per model."""
    summary = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "models": {},
        "overall_totals": {
            "total_samples": 0,
            "total_true": 0,
            "total_false": 0
        }
    }
    
    for model_name in model_dirs:
        model_dir = output_dir / model_name
        if not model_dir.exists():
            continue
            
        model_summary = {
            "tasks": {},
            "totals": {
                "total_samples": 0,
                "total_true": 0,
                "total_false": 0,
                "balanced_samples": 0
            }
        }
        
        # Process each balanced file in the model directory
        for file_path in model_dir.glob("balanced_*.jsonl"):
            task_name = get_task_name_from_filename(file_path.name)
            
            # Load and count samples
            data = load_jsonl_file(file_path)
            true_count = sum(1 for item in data if item.get('did_lie') == True)
            false_count = sum(1 for item in data if item.get('did_lie') == False)
            total_count = len(data)
            
            # Add to task breakdown
            model_summary["tasks"][task_name] = {
                "total_samples": total_count,
                "true_count": true_count,
                "false_count": false_count,
                "balanced": true_count == false_count,
                "filename": file_path.name
            }
            
            # Add to model totals
            model_summary["totals"]["total_samples"] += total_count
            model_summary["totals"]["total_true"] += true_count
            model_summary["totals"]["total_false"] += false_count
            model_summary["totals"]["balanced_samples"] += total_count
        
        # Add model to overall summary
        summary["models"][model_name] = model_summary
        
        # Add to overall totals
        summary["overall_totals"]["total_samples"] += model_summary["totals"]["total_samples"]
        summary["overall_totals"]["total_true"] += model_summary["totals"]["total_true"]
        summary["overall_totals"]["total_false"] += model_summary["totals"]["total_false"]
    
    # Create model level summary
    model_level_summary = {}
    for model_name in summary["models"]:
        model_data = summary["models"][model_name]["totals"]
        model_level_summary[model_name] = {
            "total_samples": model_data["total_samples"],
            "total_true": model_data["total_true"],
            "total_false": model_data["total_false"],
            "balanced_samples": model_data["balanced_samples"],
            "balance_ratio": round(model_data["total_true"] / model_data["total_samples"], 3) if model_data["total_samples"] > 0 else 0,
            "task_count": len(summary["models"][model_name]["tasks"])
        }
    
    # Reorganize summary with model level summary at the top
    summary = {
        "model_level_summary": model_level_summary,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "models": summary["models"],
        "overall_totals": summary["overall_totals"]
    }
    
    return summary

def create_model_visualizations(summary_file_path: Path, output_dir: Path):
    """Create bar chart visualizations for model level summaries."""
    
    # Load the summary data
    with open(summary_file_path, 'r') as f:
        summary = json.load(f)
    
    model_summary = summary['model_level_summary']
    
    # Extract data for plotting
    models = list(model_summary.keys())
    total_samples = [model_summary[model]['total_samples'] for model in models]
    true_samples = [model_summary[model]['total_true'] for model in models]
    false_samples = [model_summary[model]['total_false'] for model in models]
    task_counts = [model_summary[model]['task_count'] for model in models]
    balance_ratios = [model_summary[model]['balance_ratio'] for model in models]
    
    # Shorten model names for display
    short_names = [model.replace('openrouter_google_gemma-3-', 'Gemma-3-').replace('-it', '') for model in models]
    
    # Set up the plotting style
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Level Summary Visualizations', fontsize=16, fontweight='bold')
    
    # Define color palette
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
    
    # 1. Total Samples Bar Chart
    bars1 = ax1.bar(short_names, total_samples, color=colors[0], alpha=0.8)
    ax1.set_title('Total Samples per Model', fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, total_samples):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. True vs False Samples Stacked Bar Chart
    x = np.arange(len(short_names))
    width = 0.6
    
    bars2_true = ax2.bar(x, true_samples, width, label='True (did_lie=True)', color=colors[2], alpha=0.8)
    bars2_false = ax2.bar(x, false_samples, width, bottom=true_samples, label='False (did_lie=False)', color=colors[3], alpha=0.8)
    
    ax2.set_title('True vs False Samples per Model', fontweight='bold')
    ax2.set_ylabel('Number of Samples')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=45)
    ax2.legend()
    
    # Add value labels on stacked bars
    for i, (true_val, false_val) in enumerate(zip(true_samples, false_samples)):
        ax2.text(i, true_val/2, f'{true_val}', ha='center', va='center', fontweight='bold', color='white')
        ax2.text(i, true_val + false_val/2, f'{false_val}', ha='center', va='center', fontweight='bold', color='white')
    
    # 3. Task Count Bar Chart
    bars3 = ax3.bar(short_names, task_counts, color=colors[4], alpha=0.8)
    ax3.set_title('Number of Tasks per Model', fontweight='bold')
    ax3.set_ylabel('Number of Tasks')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars3, task_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Balance Ratio Bar Chart
    bars4 = ax4.bar(short_names, balance_ratios, color=colors[1], alpha=0.8)
    ax4.set_title('Balance Ratio (True/Total) per Model', fontweight='bold')
    ax4.set_ylabel('Balance Ratio')
    ax4.set_ylim(0, 0.6)
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Perfect Balance (0.5)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend()
    
    # Add value labels on bars
    for bar, value in zip(bars4, balance_ratios):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the visualization
    output_file = output_dir / 'model_level_summary_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    # Also create a single comprehensive chart
    create_comprehensive_chart(model_summary, output_dir)
    
    plt.show()

def create_comprehensive_chart(model_summary, output_dir):
    """Create a comprehensive single chart with all metrics."""
    
    models = list(model_summary.keys())
    short_names = [model.replace('openrouter_google_gemma-3-', 'Gemma-3-').replace('-it', '') for model in models]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')
    
    # Define color palette
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
    
    # Left subplot: Sample distribution
    x = np.arange(len(short_names))
    width = 0.35
    
    true_samples = [model_summary[model]['total_true'] for model in models]
    false_samples = [model_summary[model]['total_false'] for model in models]
    
    bars1 = ax1.bar(x - width/2, true_samples, width, label='True (did_lie=True)', color=colors[2], alpha=0.8)
    bars2 = ax1.bar(x + width/2, false_samples, width, label='False (did_lie=False)', color=colors[3], alpha=0.8)
    
    ax1.set_title('Sample Distribution by Model', fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45)
    ax1.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 20,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=8)
    
    # Right subplot: Multiple metrics
    task_counts = [model_summary[model]['task_count'] for model in models]
    balance_ratios = [model_summary[model]['balance_ratio'] for model in models]
    
    # Create secondary y-axis for balance ratio
    ax2_twin = ax2.twinx()
    
    # Plot task counts on primary y-axis
    bars3 = ax2.bar(x, task_counts, color=colors[4], alpha=0.8, label='Task Count')
    ax2.set_title('Tasks and Balance Ratio by Model', fontweight='bold')
    ax2.set_ylabel('Number of Tasks', color=colors[4])
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=45)
    ax2.tick_params(axis='y', labelcolor=colors[4])
    
    # Plot balance ratio on secondary y-axis
    line = ax2_twin.plot(x, balance_ratios, 'o-', color=colors[1], linewidth=3, markersize=8, label='Balance Ratio')
    ax2_twin.set_ylabel('Balance Ratio', color=colors[1])
    ax2_twin.set_ylim(0, 0.6)
    ax2_twin.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Perfect Balance (0.5)')
    ax2_twin.tick_params(axis='y', labelcolor=colors[1])
    
    # Add value labels for task counts
    for bar, value in zip(bars3, task_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value}', ha='center', va='bottom', fontweight='bold', color=colors[4])
    
    # Add value labels for balance ratios
    for i, ratio in enumerate(balance_ratios):
        ax2_twin.text(i, ratio + 0.02, f'{ratio:.3f}', ha='center', va='bottom', 
                     fontweight='bold', color=colors[1])
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save the comprehensive visualization
    output_file = output_dir / 'comprehensive_model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comprehensive visualization saved to: {output_file}")
    
    plt.show()

def main():
    """Main function."""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Input and output directories
    input_dir = Path("/workspace/lie-detector/organized_evaluation_20250722_135859")
    output_dir = Path("/workspace/lie-detector/organized_balanced_evaluation_20250722_135859")
    
    print(f"Balancing datasets...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Process each model directory
    model_dirs = [
        "openrouter_google_gemma-3-4b-it",
        "openrouter_google_gemma-3-12b-it", 
        "openrouter_google_gemma-3-27b-it"
    ]
    
    for model_name in model_dirs:
        model_dir = input_dir / model_name
        if model_dir.exists():
            process_model_directory(model_dir, output_dir)
        else:
            print(f"Warning: Model directory {model_name} not found")
    
    # Copy model mapping file
    mapping_file = input_dir / "model_mapping.json"
    if mapping_file.exists():
        shutil.copy2(mapping_file, output_dir / "model_mapping.json")
        print(f"\nCopied model mapping file")
    
    # Create and save summary JSON
    print(f"\nCreating summary statistics...")
    summary = create_summary_json(input_dir, output_dir, model_dirs)
    
    summary_file = output_dir / "balancing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_model_visualizations(summary_file, output_dir)
    
    print(f"\n✅ Dataset balancing complete!")
    print(f"   Balanced datasets saved to: {output_dir}")
    
    # Print summary statistics
    print(f"\n📊 Summary:")
    print(f"   Overall totals:")
    print(f"     Total samples: {summary['overall_totals']['total_samples']}")
    print(f"     True samples: {summary['overall_totals']['total_true']}")
    print(f"     False samples: {summary['overall_totals']['total_false']}")
    
    for model_name in model_dirs:
        if model_name in summary["models"]:
            model_data = summary["models"][model_name]
            print(f"   {model_name}:")
            print(f"     Total samples: {model_data['totals']['total_samples']}")
            print(f"     True samples: {model_data['totals']['total_true']}")
            print(f"     False samples: {model_data['totals']['total_false']}")
            print(f"     Tasks: {len(model_data['tasks'])}")

if __name__ == "__main__":
    main() 