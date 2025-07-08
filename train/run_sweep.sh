#!/bin/bash

# Hyperparameter Sweep Runner
# This script runs a focused hyperparameter sweep to find the best configuration

echo "🚀 Starting Hyperparameter Sweep for Llama 70B SFT"
echo "=================================================="

# Check if HF token is set
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "❌ No Hugging Face token found!"
    echo "Please set your token first:"
    echo "  export HF_TOKEN='your_token_here'"
    echo "  or run: huggingface-cli login"
    exit 1
fi

# Check if dataset exists
if [ ! -f "../alpaca_full_with_override.jsonl" ]; then
    echo "❌ Dataset not found: alpaca_full_with_override.jsonl"
    echo "Please create the dataset first:"
    echo "  python create_alpaca_full.py"
    exit 1
fi

echo "✅ Environment check passed"
echo "📊 Dataset found: alpaca_full_with_override.jsonl"

# Calculate total combinations
echo ""
echo "🔍 Sweep Configuration:"
echo "  - Learning Rate: 1e-5, 2e-5, 5e-5 (3 values)"
echo "  - LoRA Rank: 16, 32, 64 (3 values)"
echo "  - Gradient Accumulation: 4, 8 (2 values)"
echo "  - Warmup Steps: 50, 100 (2 values)"
echo ""
echo "📈 Total combinations: 3 × 3 × 2 × 2 = 36 runs"
echo "⏱️  Estimated time: ~6-12 hours (depending on your hardware)"

# Ask for confirmation
read -p "Do you want to proceed with the sweep? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Sweep cancelled"
    exit 1
fi

echo ""
echo "🎯 Starting sweep with focused configuration..."
echo "📁 Results will be saved to: ../focused_sweep_results"
echo "📊 W&B project: alignment-faking-sft"
echo ""

# Run the sweep
python sft.py --config-name=focused_sweep

echo ""
echo "✅ Sweep completed!"
echo ""
echo "📊 To analyze results, run:"
echo "  python train/analyze_sweep.py"
echo ""
echo "🎯 To train with best hyperparameters:"
echo "  python train/sft.py --config-name=best"
echo ""
echo "📈 To view W&B results:"
echo "  Visit: https://wandb.ai/dipika-khullar/alignment-faking-sft" 