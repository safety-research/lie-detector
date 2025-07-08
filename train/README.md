# Training Directory

This directory contains all the training-related code for the alignment faking model.

## Files

### Core Training
- `sft.py` - Main supervised fine-tuning script
- `config.yaml` - Default training configuration
- `focused_sweep.yaml` - Configuration for hyperparameter sweep

### Hyperparameter Sweep
- `run_sweep.sh` - Script to run hyperparameter sweep
- `analyze_sweep.py` - Analyze sweep results and find best configuration
- `check_sweep_progress.py` - Monitor sweep progress
- `config_sweep.yaml` - Sweep configuration
- `config_focused_sweep.yaml` - Focused sweep configuration
- `lr_sweep.yaml` - Learning rate sweep configuration

### Analysis
- `analyze_training_data.py` - Analyze the training dataset
- `SWEEP_README.md` - Detailed sweep documentation

## Usage

### Basic Training
```bash
# From the root directory
python train/sft.py --config-name=config
```

### Hyperparameter Sweep
```bash
# From the root directory
cd train
./run_sweep.sh
```

### Analyze Results
```bash
# From the root directory
python train/analyze_sweep.py
```

### Check Progress
```bash
# From the root directory
python train/check_sweep_progress.py
```

## Output Directories

Training outputs are saved to:
- `../llama-70b-alignment-faking-sft/` - Default training output
- `../focused_sweep_results/` - Sweep results
- `../best_model_training/` - Best model training (after sweep analysis)

## Data Files

The training scripts expect data files in the parent directory:
- `../alpaca_full_with_override.jsonl` - Main training dataset
- `../alignment_faking_examples.jsonl` - Alternative dataset 