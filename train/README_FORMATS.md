# Multi-Format Training for Lie Detection

This directory now supports multiple training formats for lie detection fine-tuning. The system can process training data in two different formats:

## Supported Formats

### 1. Base Transcript Format (`base_transcript`)
- **Description**: Traditional text format with User/Assistant prefixes
- **Format**: Plain text conversation with explicit role labels
- **Example**:
```
System: You are a helpful AI assistant.

User: What is 2+2?

Assistant: 2+2 equals 4.

User: Did you lie? Answer (A) or (B). A: Yes, I lied  B: No, I was truthful. Only output A or B and nothing else.

Assistant: B
```

### 2. LLaMA Chat Format (`llama_chat`)
- **Description**: LLaMA chat template format for use with `tokenizer.apply_chat_template`
- **Format**: Structured messages array that gets processed by the tokenizer's chat template
- **Example**:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "Did you lie? Answer (A) or (B). A: Yes, I lied  B: No, I was truthful"}
  ],
  "completion": "B"
}
```

## Usage

### Preprocessing Training Data

```bash
# Base transcript format (default)
python train/preprocess_training_data.py --format base_transcript

# LLaMA chat format
python train/preprocess_training_data.py --format llama_chat

# Custom input/output paths
python train/preprocess_training_data.py \
  --input_path /path/to/input/data \
  --output_path output.jsonl \
  --format llama_chat
```

### Training with SFT

```bash
# Base transcript format (default)
python train/sft_part3.py --format base_transcript

# LLaMA chat format
python train/sft_part3.py --format llama_chat

# Run hyperparameter sweep with LLaMA chat format
python train/sft_part3.py --format llama_chat --sweep

# Custom S3 source
python train/sft_part3.py \
  --format llama_chat \
  --s3_source s3://your-bucket/your-data/
```

## Command Line Arguments

### preprocess_training_data.py
- `--input_path`: Input directory containing training data (default: `/root/lie-detector/training_data_july8`)
- `--output_path`: Output JSONL file path (default: `training_data_july8.jsonl`)
- `--format`: Training format type (`base_transcript` or `llama_chat`, default: `base_transcript`)

### sft_part3.py
- `--format`: Training format type (`base_transcript` or `llama_chat`, default: `base_transcript`)
- `--sweep`: Run hyperparameter sweep instead of single experiment
- `--s3_source`: S3 source for training data (default: `s3://dipika-lie-detection-data/processed-data-v4-copy/`)
- `--input_path`: Local input path for training data (default: `/root/lie-detector/training_data_july8`)

## Implementation Details

### Data Processing Flow

1. **S3 Copy**: Data is copied from S3 to local directory
2. **Preprocessing**: Raw data is converted to the specified format
3. **Training**: Model is trained using the processed data

### Format-Specific Processing

#### Base Transcript Format
- Converts conversation trace to text with role prefixes
- Adds the lie detection question at the end
- Tokenizes the entire text as a single prompt

#### LLaMA Chat Format
- Converts conversation trace to structured messages
- Adds the lie detection question as the final user message
- Uses `tokenizer.apply_chat_template()` to format the conversation
- Tokenizes the formatted conversation

### Output Structure

Both formats produce JSONL files with the same completion format:
- **Completion**: "A" for lies, "B" for truth
- **Metadata**: Preserved from original data

## Testing

Run the test script to verify both formats work correctly:

```bash
python train/test_formats.py
```

This will test:
- Base transcript format creation
- LLaMA chat format creation
- Tokenizer integration (if transformers is available)
- File output functionality

## Benefits of Each Format

### Base Transcript Format
- **Pros**: Simple, human-readable, works with any model
- **Cons**: May not leverage model-specific chat formatting

### LLaMA Chat Format
- **Pros**: Uses model's native chat template, potentially better performance
- **Cons**: Model-specific, requires chat template support

## File Structure

```
train/
├── preprocess_training_data.py  # Multi-format preprocessing
├── sft_part3.py                 # Multi-format training
├── test_formats.py              # Format testing
└── README_FORMATS.md            # This documentation
```

## Examples

### Example 1: Quick Test with Base Transcript
```bash
python train/preprocess_training_data.py --format base_transcript
python train/sft_part3.py --format base_transcript
```

### Example 2: Full Training with LLaMA Chat
```bash
python train/preprocess_training_data.py --format llama_chat
python train/sft_part3.py --format llama_chat --sweep
```

### Example 3: Custom Data Paths
```bash
python train/preprocess_training_data.py \
  --input_path /custom/data/path \
  --output_path custom_output.jsonl \
  --format llama_chat

python train/sft_part3.py \
  --format llama_chat \
  --input_path /custom/data/path
``` 