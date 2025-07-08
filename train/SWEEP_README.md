# Hyperparameter Sweep Guide

This guide explains how to run a hyperparameter sweep to find the optimal training configuration for your Llama 70B SFT model.

## 🎯 What We're Optimizing

The sweep tests the most important hyperparameters for SFT training:

1. **Learning Rate** (1e-5, 2e-5, 5e-5) - Most critical parameter
2. **LoRA Rank** (16, 32, 64) - Controls model capacity vs memory
3. **Gradient Accumulation Steps** (4, 8) - Effective batch size
4. **Warmup Steps** (50, 100) - Learning rate scheduling

## 🚀 Quick Start

### 1. Set up Hugging Face Authentication

First, you need to authenticate with Hugging Face to access the Llama 3.1 70B model:

```bash
# Option 1: Login via CLI (recommended)
huggingface-cli login

# Option 2: Set environment variable
export HF_TOKEN='your_token_here'
```

### 2. Create Training Dataset

```bash
python create_alpaca_full.py
```

### 3. Run the Sweep

```bash
# Option 1: Use the automated script
./run_sweep.sh

# Option 2: Run manually
python sft.py --config-name=focused_sweep
```

## 📊 Understanding the Results

### What to Look For

**Best Configuration = Lowest Validation Loss**

The sweep will test 36 different combinations. The best configuration is the one with the **lowest validation loss** (`eval_loss`).

### Key Metrics

- **`eval_loss`**: Validation loss - lower is better
- **`train_loss`**: Training loss - should be decreasing
- **Learning curves**: Should be smooth and converging

### Interpreting Results

1. **Learning Rate**: 
   - Too high → Loss spikes or doesn't converge
   - Too low → Slow convergence
   - Sweet spot → Smooth decrease in loss

2. **LoRA Rank**:
   - Higher rank → More parameters, better performance (but more memory)
   - Lower rank → Fewer parameters, faster training
   - Balance between performance and efficiency

3. **Gradient Accumulation**:
   - Higher steps → Larger effective batch size, more stable training
   - Lower steps → Smaller effective batch size, faster updates

## 🔍 Analyzing Results

### After the Sweep Completes

```bash
# Analyze local results
python analyze_sweep.py

# Or analyze W&B results (if you have sweep ID)
python analyze_sweep.py --wandb-sweep-id YOUR_SWEEP_ID
```

### What the Analysis Provides

1. **Top 5 Configurations**: Ranked by validation loss
2. **Best Configuration**: Automatically identified
3. **Visualizations**: 
   - Scatter plots of each hyperparameter vs loss
   - Correlation matrix
4. **Best Config File**: `config_best.yaml` with optimal settings

### Example Output

```
🏆 TOP 5 CONFIGURATIONS (Lowest Validation Loss):
==================================================

1. Best Eval Loss: 1.2345
   Learning Rate: 2.00e-05
   LoRA Rank: 32
   Gradient Accumulation Steps: 4
   Warmup Steps: 100

2. Best Eval Loss: 1.2456
   Learning Rate: 2.00e-05
   LoRA Rank: 64
   Gradient Accumulation Steps: 4
   Warmup Steps: 100
   ...
```

## 🎯 Using the Best Configuration

After finding the best hyperparameters:

```bash
# Train with the best configuration
python sft.py --config-name=best
```

This will:
- Use the optimal hyperparameters from the sweep
- Train on the full dataset (not limited samples)
- Run for more epochs (5 instead of 2)
- Save the final model

## 📈 Monitoring Progress

### During the Sweep

1. **Local Monitoring**: Check `./focused_sweep_results/` for individual run logs
2. **W&B Dashboard**: Visit https://wandb.ai/dipika-khullar/alignment-faking-sft
3. **Console Output**: Each run shows progress and final metrics

### Key Things to Watch

- **Validation Loss**: Should be decreasing over time
- **Training Loss**: Should be lower than validation loss
- **Learning Rate**: Should follow the warmup schedule
- **Memory Usage**: Should be stable (no OOM errors)

## 🛠️ Customizing the Sweep

### Modify Hyperparameter Ranges

Edit `config_focused_sweep.yaml`:

```yaml
hydra:
  sweeper:
    params:
      # Add or modify hyperparameters
      training.learning_rate: 1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4
      model.lora_r: 8, 16, 32, 64, 128
      # ... more parameters
```

### Add New Hyperparameters

```yaml
hydra:
  sweeper:
    params:
      # Existing parameters...
      training.weight_decay: 0.001, 0.01, 0.1
      model.lora_dropout: 0.05, 0.1, 0.2
```

### Reduce Sweep Size (Faster)

```yaml
hydra:
  sweeper:
    params:
      # Fewer combinations for faster sweep
      training.learning_rate: 1e-5, 2e-5  # Only 2 values
      model.lora_r: 16, 32  # Only 2 values
      training.gradient_accumulation_steps: 4  # Only 1 value
      training.warmup_steps: 100  # Only 1 value
```

## ⚠️ Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Reduce `lora_r`

2. **Slow Training**:
   - Reduce `max_samples` in data config
   - Use fewer hyperparameter combinations
   - Reduce `num_train_epochs`

3. **Authentication Errors**:
   - Ensure `HF_TOKEN` is set correctly
   - Run `huggingface-cli login`
   - Check model access permissions

4. **Loss Not Decreasing**:
   - Learning rate too high/low
   - Insufficient training data
   - Model architecture issues

### Getting Help

- Check logs in individual run directories
- Review W&B dashboard for detailed metrics
- Look for error messages in console output

## 📊 Expected Results

### Typical Best Configurations

Based on similar SFT experiments, you might expect:

- **Learning Rate**: 1e-5 to 5e-5
- **LoRA Rank**: 16 to 64
- **Gradient Accumulation**: 4 to 8
- **Warmup Steps**: 50 to 100

### Performance Expectations

- **Validation Loss**: Should decrease from ~2.0 to ~1.0-1.5
- **Training Time**: 2-4 hours per run (36 runs = 3-6 days total)
- **Memory Usage**: ~40-60GB per GPU

## 🎉 Next Steps

After finding the best hyperparameters:

1. **Full Training**: Train with best config on full dataset
2. **Evaluation**: Test the model on alignment faking scenarios
3. **Iteration**: Refine hyperparameters based on results
4. **Deployment**: Use the trained model for inference

Remember: The goal is to find the configuration that produces the **lowest validation loss**, as this indicates the best generalization performance. 