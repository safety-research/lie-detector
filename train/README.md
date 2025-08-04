# Training Data Preprocessing

## Steps

1. **Install Dependencies**
   ```bash
   uv pip install --no-deps -r requirements_train.txt
   pip install boto3 matplotlib numpy
   ```

2. **Organize S3 Data**
   ```bash
   python common/organize_s3_data_by_model_boto3.py
   ```
   - Downloads and organizes evaluation data from S3 by model
   - Creates `organized_evaluation_YYYYMMDD_HHMMSS/` directory
   - Requires AWS credentials

3. **Clean Consecutive Assistant Messages**
   ```bash
   python train/cleanup_consecutive_assistant_messages.py ./organized_evaluation_YYYYMMDD_HHMMSS
   ```
   - Merges consecutive assistant messages in JSONL files
   - Creates `organized_evaluation_YYYYMMDD_HHMMSS_cleaned/` directory
   - Required before preprocessing for proper training format

4. **Preprocess Training Data**
   ```bash
   # Standard 80/20 split
   python train/preprocess_training_data.py --input_path ./organized_evaluation_20250726_225358_cleaned --taxonomy_path ./common/simple_lie_taxonomy.csv
   
   # Limited training examples (8 per fold)
   python train/preprocess_training_data.py --input_path ./organized_evaluation_20250726_225358_cleaned --taxonomy_path ./common/simple_lie_taxonomy.csv --max_train_examples 8
   
   # Limited training examples (16 per fold)
   python train/preprocess_training_data.py --input_path ./organized_evaluation_20250726_225358_cleaned --taxonomy_path ./common/simple_lie_taxonomy.csv --max_train_examples 16
   ```
   - Processes cleaned organized evaluation data
   - Creates train/test splits (80/20 or max_train_examples)
   - Generates fold-based splits using taxonomy mappings
   - Outputs: `organized_balanced_training_YYYYMMDD_HHMMSS/` or `organized_evaluation_YYYYMMDD_HHMMSS_train_N/`

## Data Location
Training data is generated in:
- Standard: `organized_balanced_training_YYYYMMDD_HHMMSS/`
- Limited: `organized_evaluation_YYYYMMDD_HHMMSS_train_N/`
   - General splits: `{model}/general_train_test_split_base_transcript/{train,test}/`
   - Fold splits: `{model}/folds_colors_base_transcript/{category}/{train,test}/`

## Requirements
- `boto3` for S3 access
- `matplotlib`, `numpy` for visualizations
- Lie taxonomy CSV file: `common/simple_lie_taxonomy.csv`

## Output Formats
- `base_transcript`: Traditional text format
- `chat_format`: Chat template format for tokenizer 