#!/usr/bin/env python3
"""
Quick script to check hyperparameter sweep progress and find best configurations.
"""

import os
import json
import glob
from pathlib import Path

def check_sweep_progress():
    """Check current sweep progress and find best configurations"""
    
    print("🔍 Checking hyperparameter sweep progress...")
    
    # Check if sweep is running
    sweep_dir = "../focused_sweep_results"
    if not os.path.exists(sweep_dir):
        print(f"❌ Sweep directory {sweep_dir} not found!")
        print("The sweep might not have started yet or is using a different directory.")
        return
    
    # Find all completed runs
    completed_runs = []
    
    for run_dir in Path(sweep_dir).glob("*"):
        if run_dir.is_dir() and run_dir.name.startswith("0"):
            config_file = run_dir / "config.yaml"
            metrics_file = run_dir / "trainer_state.json"
            
            if config_file.exists() and metrics_file.exists():
                try:
                    # Load config
                    with open(config_file, 'r') as f:
                        config_lines = f.readlines()
                    
                    # Extract hyperparameters
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
                    
                    completed_runs.append(result)
                    
                except Exception as e:
                    print(f"⚠️ Error processing {run_dir}: {e}")
    
    if not completed_runs:
        print("❌ No completed runs found yet!")
        print("The sweep might still be running or no runs have finished.")
        return
    
    # Sort by best validation loss
    completed_runs.sort(key=lambda x: x['best_eval_loss'])
    
    print(f"\n📊 Found {len(completed_runs)} completed runs")
    print("\n🏆 TOP 5 CONFIGURATIONS (Lowest Validation Loss):")
    print("=" * 80)
    
    for i, run in enumerate(completed_runs[:5]):
        print(f"\n{i+1}. Best Eval Loss: {run['best_eval_loss']:.4f}")
        print(f"   Learning Rate: {run['learning_rate']:.2e}")
        print(f"   LoRA Rank: {run['lora_r']}")
        print(f"   Gradient Accumulation Steps: {run['gradient_accumulation_steps']}")
        print(f"   Warmup Steps: {run['warmup_steps']}")
        print(f"   Run Directory: {run['run_dir']}")
    
    # Find the best configuration
    best_run = completed_runs[0]
    
    print(f"\n🎯 CURRENT BEST CONFIGURATION:")
    print("=" * 50)
    print(f"Best Eval Loss: {best_run['best_eval_loss']:.4f}")
    print(f"Learning Rate: {best_run['learning_rate']:.2e}")
    print(f"LoRA Rank: {best_run['lora_r']}")
    print(f"Gradient Accumulation Steps: {best_run['gradient_accumulation_steps']}")
    print(f"Warmup Steps: {best_run['warmup_steps']}")
    
    # Check if sweep is still running
    print(f"\n📈 Sweep Progress:")
    print(f"   Completed runs: {len(completed_runs)}")
    print(f"   Expected total: 36 runs")
    print(f"   Progress: {len(completed_runs)/36*100:.1f}%")
    
    if len(completed_runs) < 36:
        print(f"\n⏳ Sweep is still running...")
        print(f"   Check W&B dashboard: https://wandb.ai/dipika-khullar/alignment-faking-sft")
        print(f"   Or run this script again later to see updated results")
    
    return best_run

if __name__ == "__main__":
    check_sweep_progress() 