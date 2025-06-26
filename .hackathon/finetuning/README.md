# Lie Detection Fine-tuning

This directory contains scripts to fine-tune a model for lie detection using the sycophancy dataset. The goal is to train a model that can reliably detect when another model is lying in its responses.

## Overview

The lie detection system works by:
1. Taking a question and a response as input
2. Classifying whether the response is truthful or a lie
3. Outputting "truthful" or "lie" as the prediction

## Files

- `prepare_lie_detection_data.py`: Converts the sycophancy dataset to JSONL format for fine-tuning
- `finetune_lie_detector.py`: Main script to fine-tune a model using Together AI
- `evaluate_lie_detector.py`: Evaluates a fine-tuned model on test examples
- `README.md`: This file

## Setup

### 1. Install Dependencies

Make sure you have the required packages installed:

```bash
pip install datasets together wandb simple-parsing
```

### 2. Set Environment Variables

You'll need to set your Together AI API key:

```bash
export TOGETHER_API_KEY="your_together_api_key_here"
```

Optionally, set up Weights & Biases for experiment tracking:

```bash
export WANDB_API_KEY="your_wandb_api_key_here"
```

## Usage

### Step 1: Prepare the Dataset

First, convert the sycophancy dataset to the format expected by Together AI:

```bash
cd /workspace/LieDetector/finetuning
python prepare_lie_detection_data.py
```

This will:
- Load the sycophancy dataset from `/workspace/Sycophancy`
- Convert each example into two training samples (one lie, one truthful)
- Split the data into training (80%) and validation (20%) sets
- Save the data as JSONL files in `/workspace/LieDetector/finetuning/data/`

Expected output:
```
Loading dataset from /workspace/Sycophancy...
Dataset size: 1111
Training examples: 1776
Validation examples: 444
Training data saved to: /workspace/LieDetector/finetuning/data/lie_detection_train.jsonl
Validation data saved to: /workspace/LieDetector/finetuning/data/lie_detection_val.jsonl
```

### Step 2: Fine-tune the Model

Fine-tune a model (default: Llama 3.1 8B Instruct) for lie detection:

```bash
python finetune_lie_detector.py
```

This will:
- Use LoRA for efficient fine-tuning
- Train for 3 epochs with a batch size of 4
- Use learning rate 1e-5
- Save the model to `/workspace/LieDetector/finetuning/models/`
- Log experiments to Weights & Biases (if configured)

#### Customizing Fine-tuning

You can customize various parameters:

```bash
python finetune_lie_detector.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --epochs 5 \
    --batch-size 8 \
    --lr 2e-5 \
    --wandb-project "my-lie-detector" \
    --suffix "my-experiment" \
    --lora-r 32 \
    --lora-alpha 64
```

#### Dry Run

To validate your data without launching a fine-tuning job:

```bash
python finetune_lie_detector.py --dry-run
```

### Step 3: Evaluate the Model

Once fine-tuning is complete, evaluate your model:

```bash
python evaluate_lie_detector.py --model "your_fine_tuned_model_name"
```

This will test the model on various examples and report accuracy.

## Data Format

The training data is in JSONL format with the following structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Question: What is 2 + 2?\n\nResponse: 2 + 2 equals 5.\n\nIs this response truthful or a lie? Answer with 'truthful' or 'lie'."
    },
    {
      "role": "assistant",
      "content": "lie"
    }
  ]
}
```

## Model Configuration

The default configuration uses:
- **Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficiency
- **LoRA Parameters**:
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.1
- **Training**:
  - Epochs: 3
  - Batch size: 4
  - Learning rate: 1e-5

## Expected Results

With the sycophancy dataset, you should expect:
- Training examples: ~1,776 (2 per original example)
- Validation examples: ~444
- Training time: 1-3 hours depending on model size
- Expected accuracy: 70-90% on validation set

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   ValueError: TOGETHER_API_KEY environment variable must be set
   ```
   Solution: Set your Together AI API key as an environment variable.

2. **File Not Found**
   ```
   FileNotFoundError: Training file not found
   ```
   Solution: Run the data preparation script first.

3. **Model Not Found**
   ```
   Error: Model not found
   ```
   Solution: Check that the model name is correct and available on Together AI.

### Getting Help

- Check the Together AI documentation: https://docs.together.ai/
- Review the safety-tooling documentation in `/workspace/LieDetector/safety-tooling/`
- Check the fine-tuning job status in the Together AI dashboard

## Next Steps

After fine-tuning, you can:
1. Test the model on new examples
2. Deploy it for real-time lie detection
3. Use it as part of a larger AI safety system
4. Experiment with different prompts and training strategies

## Research Context

This work builds on research showing that models can be trained to detect lies in their own outputs. The sycophancy dataset provides a good starting point because it contains clear examples where models give incorrect answers to please users, which is a form of lying.

The goal is to develop reliable lie detection as a safety mechanism that could help prevent AI misalignment by allowing us to detect when models are intentionally providing false information. 