# Training Data Preprocessing

## Prerequisites

### 1. **Install AWS CLI**
   ```bash
   # Install unzip if not available
   apt update && apt install -y unzip
   
   # Download and install AWS CLI
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   ./aws/install
   
   # Alternative: Install via pip
   pip install awscli
   ```

### 2. **Configure AWS Credentials**
   ```bash
   aws configure set aws_access_key_id YOUR_ACCESS_KEY_ID
   aws configure set aws_secret_access_key YOUR_SECRET_ACCESS_KEY
   aws configure set default.region us-east-1
   ```

### 3. **Set Environment Variables**
   ```bash
   # Hugging Face cache directories
   export HF_HOME=/workspace/cache
   export HF_CACHE_HOME=/workspace/cache
   export TRANSFORMERS_CACHE=/workspace/cache
   export HF_DATASETS_CACHE=/workspace/cache
   export HF_HUB_CACHE=/workspace/cache
   export HF_MODELS_CACHE=/workspace/cache
   export HF_TOKEN_CACHE=/workspace/cache
   export HF_TOKEN=YOUR_HF_TOKEN
   
   # AWS credentials (if not using aws configure)
   export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
   export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY
   ```

## Setup Steps

### 1. **Activate Virtual Environment**
   ```bash
   source ~/lies310/bin/activate
   ```
   - **Note**: Replace `lies310` with your actual virtual environment name
   - Verify activation: `which python` should show your venv path

### 2. **Install Dependencies**
   ```bash
   uv pip install --no-deps -r requirements_train.txt
   pip install boto3 matplotlib numpy
   ```

### 3. **Organize S3 Data**
   ```bash
   python common/organize_s3_data_by_model_boto3.py
   ```
   - Downloads and organizes evaluation data from S3 by model
   - Creates `organized_evaluation_YYYYMMDD_HHMMSS/` directory
   - Requires AWS credentials to be configured

### 4. **Clean Consecutive Assistant Messages**
   ```bash
   python train/cleanup_consecutive_assistant_messages.py ./organized_evaluation_YYYYMMDD_HHMMSS
   ```
   - Merges consecutive assistant messages in JSONL files
   - Creates `organized_evaluation_YYYYMMDD_HHMMSS_cleaned/` directory
   - Required before preprocessing for proper training format

### 5. **Preprocess Training Data**
   ```bash
   # Standard 80/20 split
   python train/preprocess_training_data.py --input_path ./organized_evaluation_YYYYMMDD_HHMMSS_cleaned --taxonomy_path ./common/simple_lie_taxonomy.csv
   
   # Limited training examples (8 per fold)
   python train/preprocess_training_data.py --input_path ./organized_evaluation_YYYYMMDD_HHMMSS_cleaned --taxonomy_path ./common/simple_lie_taxonomy.csv --max_train_examples 8
   
   # Limited training examples (16 per fold)
   python train/preprocess_training_data.py --input_path ./organized_evaluation_YYYYMMDD_HHMMSS_cleaned --taxonomy_path ./common/simple_lie_taxonomy.csv --max_train_examples 16
   ```
   - Processes cleaned organized evaluation data
   - Creates train/test splits (80/20 or max_train_examples)
   - Generates fold-based splits using taxonomy mappings
   - Outputs: `organized_balanced_training_YYYYMMDD_HHMMSS/` or `organized_evaluation_YYYYMMDD_HHMMSS_train_N/`

## Automated Setup

### **Using the Setup Script**
   ```bash
   # Make the script executable
   chmod +x train_setup.sh
   
   # Run the complete setup
   ./train_setup.sh
   ```
   - **Note**: Update the script with your actual AWS credentials and HF token before running

## Data Location
Training data is generated in:
- Standard: `organized_balanced_training_YYYYMMDD_HHMMSS/`
- Limited: `organized_evaluation_YYYYMMDD_HHMMSS_train_N/`
   - General splits: `{model}/general_train_test_split_base_transcript/{train,test}/`
   - Fold splits: `{model}/folds_colors_base_transcript/{category}/{train,test}/`

## Requirements
- Python virtual environment (e.g., `lies310`)
- `boto3` for S3 access
- `matplotlib`, `numpy` for visualizations
- `uv` for package management
- AWS CLI for S3 access
- Lie taxonomy CSV file: `common/simple_lie_taxonomy.csv`
- AWS credentials with S3 access
- Hugging Face token (optional, for model downloads)

## Output Formats
- `base_transcript`: Traditional text format
- `chat_format`: Chat template format for tokenizer

## Troubleshooting

### Common Issues:
1. **Virtual Environment Not Found**: Ensure you're using the correct venv name
2. **AWS Credentials Error**: Verify AWS credentials are properly configured
3. **Permission Denied**: Make sure you have write permissions in the workspace
4. **Missing Dependencies**: Run `pip install -r requirements_train.txt` if uv is not available

### Verification Steps:
1. Check virtual environment: `which python`
2. Verify AWS CLI: `aws --version`
3. Test AWS access: `aws s3 ls s3://dipika-lie-detection-data/`
4. Check environment variables: `echo $HF_HOME` 