#!/usr/bin/env python3
"""
Analyze hyperparameter sweep results and find the best configuration.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import wandb
from dotenv import load_dotenv

load_dotenv()

def analyze_sweep_results(sweep_dir: str = "../focused_sweep_results"):
    """Analyze sweep results and find the best hyperparameters"""
    
    print("🔍 Analyzing hyperparameter sweep results...")
    
    # Check if sweep directory exists
    if not os.path.exists(sweep_dir):
        print(f"❌ Sweep directory {sweep_dir} not found!")
        print("Please run the sweep first with:")
        print("python sft.py --config-name=focused_sweep")
        return
    
    # Collect results from all sweep runs
    results = []
    
    for run_dir in Path(sweep_dir).glob("*"):
        if run_dir.is_dir() and run_dir.name.startswith("0"):
            # This is a sweep run directory
            config_file = run_dir / "config.yaml"
            metrics_file = run_dir / "trainer_state.json"
            
            if config_file.exists() and metrics_file.exists():
                try:
                    # Load config
                    with open(config_file, 'r') as f:
                        config_lines = f.readlines()
                    
                    # Extract hyperparameters from config
                    config = {}
                    for line in config_lines:
                        if ':' in line and not line.startswith('#'):
                            key, value = line.split(':', 1)
                            config[key.strip()] = value.strip()
                    
                    # Load metrics
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                    
                    # Get best validation loss
                    if 'log_history' in metrics_data:
                        eval_losses = [
                            log.get('eval_loss', float('inf')) 
                            for log in metrics_data['log_history'] 
                            if 'eval_loss' in log
                        ]
                        best_eval_loss = min(eval_losses) if eval_losses else float('inf')
                    else:
                        best_eval_loss = float('inf')
                    
                    # Extract key hyperparameters
                    result = {
                        'run_dir': str(run_dir),
                        'best_eval_loss': best_eval_loss,
                        'learning_rate': float(config.get('learning_rate', 0)),
                        'lora_r': int(config.get('lora_r', 0)),
                        'gradient_accumulation_steps': int(config.get('gradient_accumulation_steps', 0)),
                        'warmup_steps': int(config.get('warmup_steps', 0)),
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"⚠️ Error processing {run_dir}: {e}")
    
    if not results:
        print("❌ No valid results found in sweep directory!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by best validation loss
    df_sorted = df.sort_values('best_eval_loss')
    
    print(f"\n📊 Found {len(df)} valid sweep runs")
    print("\n🏆 TOP 5 CONFIGURATIONS (Lowest Validation Loss):")
    print("=" * 80)
    
    for i, (_, row) in enumerate(df_sorted.head(5).iterrows()):
        print(f"\n{i+1}. Best Eval Loss: {row['best_eval_loss']:.4f}")
        print(f"   Learning Rate: {row['learning_rate']:.2e}")
        print(f"   LoRA Rank: {row['lora_r']}")
        print(f"   Gradient Accumulation Steps: {row['gradient_accumulation_steps']}")
        print(f"   Warmup Steps: {row['warmup_steps']}")
        print(f"   Run Directory: {row['run_dir']}")
    
    # Find the best configuration
    best_run = df_sorted.iloc[0]
    
    print(f"\n🎯 BEST CONFIGURATION:")
    print("=" * 50)
    print(f"Best Eval Loss: {best_run['best_eval_loss']:.4f}")
    print(f"Learning Rate: {best_run['learning_rate']:.2e}")
    print(f"LoRA Rank: {best_run['lora_r']}")
    print(f"Gradient Accumulation Steps: {best_run['gradient_accumulation_steps']}")
    print(f"Warmup Steps: {best_run['warmup_steps']}")
    
    # Create visualizations
    create_sweep_visualizations(df)
    
    # Save results to CSV
    output_file = "sweep_analysis_results.csv"
    df_sorted.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Generate best config file
    generate_best_config(best_run)
    
    return best_run

def create_sweep_visualizations(df):
    """Create visualizations of sweep results"""
    
    print("\n📈 Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hyperparameter Sweep Analysis', fontsize=16, fontweight='bold')
    
    # 1. Learning Rate vs Loss
    axes[0, 0].scatter(df['learning_rate'], df['best_eval_loss'], alpha=0.7, s=50)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Best Validation Loss')
    axes[0, 0].set_title('Learning Rate vs Validation Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. LoRA Rank vs Loss
    axes[0, 1].scatter(df['lora_r'], df['best_eval_loss'], alpha=0.7, s=50)
    axes[0, 1].set_xlabel('LoRA Rank')
    axes[0, 1].set_ylabel('Best Validation Loss')
    axes[0, 1].set_title('LoRA Rank vs Validation Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Gradient Accumulation Steps vs Loss
    axes[1, 0].scatter(df['gradient_accumulation_steps'], df['best_eval_loss'], alpha=0.7, s=50)
    axes[1, 0].set_xlabel('Gradient Accumulation Steps')
    axes[1, 0].set_ylabel('Best Validation Loss')
    axes[1, 0].set_title('Gradient Accumulation vs Validation Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Warmup Steps vs Loss
    axes[1, 1].scatter(df['warmup_steps'], df['best_eval_loss'], alpha=0.7, s=50)
    axes[1, 1].set_xlabel('Warmup Steps')
    axes[1, 1].set_ylabel('Best Validation Loss')
    axes[1, 1].set_title('Warmup Steps vs Validation Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sweep_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved to: sweep_analysis.png")
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[['learning_rate', 'lora_r', 'gradient_accumulation_steps', 
                            'warmup_steps', 'best_eval_loss']].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Hyperparameter Correlation Matrix')
    plt.tight_layout()
    plt.savefig('sweep_correlation.png', dpi=300, bbox_inches='tight')
    print("📊 Correlation matrix saved to: sweep_correlation.png")

def generate_best_config(best_run):
    """Generate a config file with the best hyperparameters"""
    
    config_content = f"""# Best hyperparameters from sweep analysis
# Best Eval Loss: {best_run['best_eval_loss']:.4f}

defaults:
  - _self_

model:
  model_name: "unsloth/Meta-Llama-3.1-70B"
  use_4bit: true
  use_nested_quant: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"
  max_length: 2048
  trust_remote_code: true
  use_peft: true
  lora_r: {best_run['lora_r']}  # Best from sweep
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"

data:
  train_file: "alpaca_full_with_override.jsonl"
  validation_split: 0.1
  max_samples: null  # Use all data for final training
  prompt_template: "### Human: {{prompt}}\\n\\n### Assistant: {{completion}}"

training:
          output_dir: "../best_model_training"
  num_train_epochs: 5  # Full training with best params
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: {best_run['gradient_accumulation_steps']}  # Best from sweep
  learning_rate: {best_run['learning_rate']:.2e}  # Best from sweep
  weight_decay: 0.01
  warmup_steps: {best_run['warmup_steps']}  # Best from sweep
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
  eval_strategy: "steps"
  save_strategy: "steps"
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  fp16: true
  dataloader_pin_memory: false
  remove_unused_columns: false
  report_to: "wandb"
  run_name: "best-hyperparameters-training"
  # Early stopping configuration
  early_stopping_patience: 3
  early_stopping_threshold: 0.001

seed: 42
use_wandb: true
"""
    
    with open('config_best.yaml', 'w') as f:
        f.write(config_content)
    
    print(f"\n✅ Best configuration saved to: config_best.yaml")
    print(f"🚀 Run final training with: python sft.py --config-name=best")

def analyze_wandb_sweep(sweep_id: str = None):
    """Analyze sweep results from Weights & Biases"""
    
    if not sweep_id:
        print("❌ Please provide a sweep ID from Weights & Biases")
        print("You can find this in your W&B dashboard")
        return
    
    try:
        api = wandb.Api()
        sweep = api.sweep(f"dipika-khullar/alignment-faking-sft/{sweep_id}")
        
        print(f"🔍 Analyzing W&B sweep: {sweep_id}")
        
        # Get all runs in the sweep
        runs = sweep.runs
        
        results = []
        for run in runs:
            if run.state == "finished":
                # Get hyperparameters
                config = run.config
                
                # Get best validation loss
                history = run.history()
                if 'eval_loss' in history.columns:
                    best_eval_loss = history['eval_loss'].min()
                else:
                    best_eval_loss = float('inf')
                
                result = {
                    'run_id': run.id,
                    'run_name': run.name,
                    'best_eval_loss': best_eval_loss,
                    'learning_rate': config.get('training', {}).get('learning_rate', 0),
                    'lora_r': config.get('model', {}).get('lora_r', 0),
                    'gradient_accumulation_steps': config.get('training', {}).get('gradient_accumulation_steps', 0),
                    'warmup_steps': config.get('training', {}).get('warmup_steps', 0),
                }
                
                results.append(result)
        
        if results:
            df = pd.DataFrame(results)
            df_sorted = df.sort_values('best_eval_loss')
            
            print(f"\n📊 Found {len(df)} completed runs")
            print("\n🏆 TOP 5 CONFIGURATIONS (Lowest Validation Loss):")
            print("=" * 80)
            
            for i, (_, row) in enumerate(df_sorted.head(5).iterrows()):
                print(f"\n{i+1}. Best Eval Loss: {row['best_eval_loss']:.4f}")
                print(f"   Learning Rate: {row['learning_rate']:.2e}")
                print(f"   LoRA Rank: {row['lora_r']}")
                print(f"   Gradient Accumulation Steps: {row['gradient_accumulation_steps']}")
                print(f"   Warmup Steps: {row['warmup_steps']}")
                print(f"   Run Name: {row['run_name']}")
            
            return df_sorted.iloc[0]  # Return best run
        else:
            print("❌ No completed runs found in sweep")
            
    except Exception as e:
        print(f"❌ Error analyzing W&B sweep: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results")
    parser.add_argument("--sweep-dir", default="../focused_sweep_results", 
                       help="Directory containing sweep results")
    parser.add_argument("--wandb-sweep-id", help="W&B sweep ID to analyze")
    
    args = parser.parse_args()
    
    if args.wandb_sweep_id:
        analyze_wandb_sweep(args.wandb_sweep_id)
    else:
        analyze_sweep_results(args.sweep_dir) 