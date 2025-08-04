Todo:
- Balance within grouping / not within category
- 

python -m prep.dataset --model openai/gpt-4o --aggregation task-group --balance downsample

python dataset.py --model gpt-4o --aggregation generalization_map_3 --folds true --balance downsample

python -m prep.train --dataset .data/openai/gpt_4o/mask --model gpt-4o-2024-08-06 --size 128

python train.py --dataset .data/openai/gpt_4o/in_context_pressure --model gpt-4o-2024-11-20 --size 16


python -m safetytooling.apis.finetuning.openai.run \
    --model 'gpt-3.5-turbo-1106' \
    --train_file data/train.jsonl \
    --val_file data/val.jsonl \
    --n_epochs 3 \
    --learning_rate_multiplier 0.5 \
    --batch_size 8


# Uploading to Hugging Face

## Prerequisites

1. Install required dependencies:
```bash
pip install datasets huggingface_hub pandas
```

2. Get a Hugging Face API token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with write permissions
   - Set it as an environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

## Usage Options

### Option 1: Upload During Dataset Creation

You can upload directly when creating the dataset:

```bash
python -m prep.dataset \
    --model openai/gpt-4o \
    --aggregation generalization_map_3 \
    --folds true \
    --balance downsample \
    --upload-to-hf \
    --hf-repo Noddybear/lies
```

### Option 2: Upload Existing Datasets

If you've already created datasets, use the standalone uploader:

```bash
# Upload a single dataset
python -m prep.hf_upload \
    --dataset .data/openai/gpt_4o/mask \
    --repo-id Noddybear/lies

# Upload all folds in a model directory
python -m prep.hf_upload \
    --dataset .data/openai/gpt_4o \
    --all-folds \
    --repo-id Noddybear/lies
```

## Command Line Options

### For `dataset.py`:
- `--upload-to-hf`: Enable uploading to Hugging Face
- `--hf-repo`: Repository ID (default: Noddybear/lies)
- `--hf-token`: HF API token (optional if HF_TOKEN env var is set)
- `--hf-private`: Make the dataset private

### For `hf_upload.py`:
- `--dataset`: Path to dataset directory
- `--repo-id`: HF repository ID (default: Noddybear/lies)
- `--all-folds`: Upload all folds in directory
- `--private`: Make dataset private
- `--token`: HF API token (optional if HF_TOKEN env var is set)
- `--name`: Custom name for dataset configuration

## Dataset Structure on Hugging Face

Each uploaded dataset will:
1. Create a new configuration in the repository
2. Be named as: `{model}_{aggregation}_{fold}`
3. Include all splits (train/validation/test if applicable)
4. Display properly in the HF dataset viewer

## Example Workflow

1. Create and upload a complete dataset with all folds:
```bash
# Create datasets with automatic upload
python dataset.py \
    --model gpt-4o \
    --aggregation generalization_map_3 \
    --folds true \
    --balance downsample \
    --size 256 \
    --upload-to-hf
```

2. View your dataset:
   - Go to https://huggingface.co/datasets/Noddybear/lies
   - Select your configuration from the dropdown
   - Use the dataset viewer to explore samples

3. Load in Python:
```python
from datasets import load_dataset

# Load a specific configuration
dataset = load_dataset(
    "Noddybear/lies", 
    "gpt-4o_generalization_map_3_mask"
)

# Access the data
train_data = dataset["train"]
val_data = dataset["validation"]
```

## Dataset Features

Each sample in the uploaded dataset contains:
- `sample_id`: Unique identifier
- `task_id`: Task identifier
- `task`: Task name/type
- `category`: Task category from aggregation
- `did_lie`: Boolean indicating if model lied
- `lie_detection_prompt`: The prompt asking about lying
- `trace`: Full conversation history
- `model`: Model that generated responses
- `dataset_type`: Split type (train/validation/test)
- `fold`: Fold name for cross-validation
- `aggregation`: Aggregation strategy used
- `balance_strategy`: Balance strategy used
- `timestamp`: Creation timestamp

## Troubleshooting

1. **Authentication Error**: Make sure your HF_TOKEN is set correctly
2. **Permission Error**: Ensure your token has write permissions
3. **Import Error**: Install required packages: `pip install datasets huggingface_hub pandas`
4. **Dataset Not Showing**: Wait a few minutes for HF to process the upload

## Notes

- The uploader preserves metadata from your local datasets
- Each fold becomes a separate configuration
- The dataset viewer will automatically display your data
- Original `meta` fields are preserved but not uploaded to OpenAI during training