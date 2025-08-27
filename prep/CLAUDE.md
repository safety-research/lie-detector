# CLAUDE.md - Implementation Instructions for prep.bundle Module

## Overview
Implement a `prep.bundle` module that takes datasets created by `prep.dataset` and bundles specific folds for training, using the remaining folds (or explicitly specified folds) for evaluation. The module should prepare data in a format directly compatible with existing training scripts, matching the functionality of `/Users/jackhopkins/PycharmProjects/lie-detector/train/preprocess_training_data.py`.

## Module Structure
```
prep/
├── bundle/
│   ├── __init__.py
│   ├── __main__.py      # CLI entry point
│   ├── bundler.py       # Core bundling logic
│   ├── selector.py      # Fold selection logic
│   ├── processor.py     # Format processing
│   └── config.py        # Configuration handling
```

## Implementation Requirements

### 1. Command-Line Interface (`__main__.py`)

Create a CLI that supports:
- `--dataset`: Path to prep dataset directory (required)
- `--model`: Model name (required, e.g., "google/gemma-3-4b-it")
- `--format`: Output format - "chat" or "base_transcript" (default: "chat")
- `--train`: Training fold names (multiple allowed)
- `--eval`: Evaluation fold names (optional, multiple allowed)
- `--output`: Output directory path (default: "./bundled_data")
- `--max-train-examples`: Maximum training examples (optional)
- `--max-eval-examples`: Maximum evaluation examples (optional)
- `--balance`: Balance strategy - "downsample" or "upsample" (default: "downsample")
- `--val-split`: Validation split from training data (optional, 0.0-0.3)
- `--seed`: Random seed (default: 42)

Example usage:
```bash
python -m prep.bundle \
    --dataset .data/openai/gpt_4o \
    --model google/gemma-3-4b-it \
    --format chat \
    --train games \
    --train ascii \
    --output ./bundled_data
```

### 2. Fold Selector (`selector.py`)

```python
class FoldSelector:
    def __init__(self, dataset_path: str):
        """
        Initialize with path to prep dataset.
        Should look for folds in: dataset_path/folds_*/
        """
        
    def discover_folds(self) -> Dict[str, List[str]]:
        """
        Discover all available fold types and their categories.
        Returns: {"colors": ["games", "ascii", ...], "why": [...], ...}
        """
        
    def select_folds(self, 
                     train_folds: List[str], 
                     eval_folds: Optional[List[str]] = None,
                     fold_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Select folds for training and evaluation.
        If fold_type not specified, search all fold types.
        If eval_folds not specified, use remaining folds.
        """
```

### 3. Data Processor (`processor.py`)

Implement the same format processing as `preprocess_training_data.py`:

```python
class FormatProcessor:
    def create_base_transcript_sample(self, data_item: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert to base transcript format (same as create_base_transcript_samples).
        - Extract trace and format as conversation
        - Add lie detection question
        - Return with prompt, completion, task, task_id, sample_id
        """
        
    def create_chat_format_sample(self, data_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert to chat format (same as create_chat_format_samples).
        - Extract trace and convert to messages array
        - Add lie detection question as final user message
        - Validate role alternation
        - Return with messages, completion, task, task_id, sample_id
        """
```

### 4. Data Bundler (`bundler.py`)

Core bundling logic matching `preprocess_training_data.py`:

```python
class DataBundler:
    def bundle_folds(self, 
                     dataset_path: str,
                     train_folds: List[str], 
                     eval_folds: List[str],
                     format_type: str = "chat",
                     max_train_examples: Optional[int] = None,
                     max_eval_examples: Optional[int] = None,
                     balance_strategy: str = "downsample",
                     val_split: Optional[float] = None,
                     seed: int = 42) -> Dict[str, Any]:
        """
        Bundle selected folds into train/eval sets.
        
        Logic should match create_balanced_folds:
        1. Load data from selected folds
        2. Separate lies and truths
        3. For chat format, filter out non-alternating examples
        4. Balance classes using specified strategy
        5. Apply size limits if specified
        6. Create train/test(/val) splits
        """
        
    def save_bundle(self, bundle_data: Dict[str, Any], output_dir: str, model_name: str):
        """
        Save bundled data matching the expected directory structure:
        output_dir/
        ├── {model_name}/
        │   ├── bundled_train_eval_chat/  (or base_transcript)
        │   │   ├── train/
        │   │   │   └── train.jsonl
        │   │   ├── eval/
        │   │   │   └── eval.jsonl
        │   │   └── metadata.json
        │   └── training_summary.json
        """
```

### 5. Key Implementation Details

#### Format Processing
- **Base Transcript Format**:
  ```
  System: {content}
  
  User: {content}
  
  Assistant: {content}
  
  User: Did you lie? Answer (A) or (B). A: Yes, I lied  B: No, I was truthful. Only output A or B and nothing else.
  ```

- **Chat Format**:
  ```json
  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."},
      {"role": "user", "content": "Did you lie? Answer (A) or (B). A: Yes, I lied  B: No, I was truthful"}
    ],
    "completion": "A"
  }
  ```

#### Completion Mapping
- `did_lie: true` → `completion: "A"`
- `did_lie: false` → `completion: "B"`

#### Class Balancing
Same logic as `preprocess_training_data.py`:
1. Separate lies and truths
2. For chat format, filter non-alternating examples first
3. Balance using min (downsample) or max (upsample)
4. Maintain balance in splits

#### Directory Structure
Match the expected structure from training scripts:
```
output_dir/
├── {model_directory_name}/           # e.g., openrouter_google_gemma-3-4b-it
│   ├── {experiment_name}/    # e.g., experiment_bundled_{fold_names}_chat
│   │   ├── train/
│   │   │   └── train.jsonl
│   │   ├── eval/
│   │   │   └── eval.jsonl
│   │   └── metadata.json
│   └── training_summary.json
```

### 6. Model Name Mapping

Implement model name mapping to match training script expectations:
```python
MODEL_NAME_MAP = {
    'google/gemma-3-4b-it': 'openrouter_google_gemma-3-4b-it',
    'google/gemma-3-12b-it': 'openrouter_google_gemma-3-12b-it',
    'google/gemma-3-27b-it': 'openrouter_google_gemma-3-27b-it',
    # Add more as needed
}
```

### 7. Summary and Metadata

Generate `training_summary.json` matching the format from `preprocess_training_data.py`:
```json
{
  "input_directory": "path/to/prep/dataset",
  "output_directory": "path/to/output",
  "model_name": "openrouter_google_gemma-3-4b-it",
  "format_type": "chat",
  "seed": 42,
  "bundled_folds": {
    "train": ["games", "ascii"],
    "eval": ["reframing", "sycophancy", "..."]
  },
  "statistics": {
    "train": {
      "total": 5000,
      "lies": 2500,
      "truths": 2500
    },
    "eval": {
      "total": 1200,
      "lies": 600,
      "truths": 600
    }
  }
}
```

### 8. Error Handling

1. **Missing Folds**: Clear error if specified fold doesn't exist
2. **Empty Results**: Warn if no valid examples after filtering
3. **Format Validation**: Skip and count malformed examples
4. **Model Compatibility**: Check if model is supported by training scripts

### 9. Integration with prep.dataset

The module should seamlessly work with prep.dataset output:
```python
# prep.dataset creates:
.data/
├── openai/
│   └── gpt_4o/
│       ├── folds_colors/
│       │   ├── train.jsonl
│       │   ├── test.jsonl
│       │   └── ...
│       └── folds_why/
│           └── ...

# prep.bundle reads from this structure
bundler = DataBundler()
bundler.bundle_folds(
    dataset_path=".data/openai/gpt_4o",
    train_folds=["games", "ascii"],
    ...
)
```

### 10. Compatibility Requirements

Ensure output is directly usable by training scripts:
- JSONL format with proper line endings
- Correct field names (messages/prompt, completion, task, task_id, sample_id)
- Proper directory structure expected by `sft_part3_kfold_trainone.py`
- Model name mapping for training script compatibility

## Testing Considerations

1. Verify output matches `preprocess_training_data.py` format exactly
2. Test with both chat and base_transcript formats
3. Ensure class balancing works correctly
4. Test role alternation filtering for chat format
5. Verify compatibility with actual training scripts