#!/bin/bash

# Training Data Setup Script for prep.bundle and Lie Detection Training
# This script automates the complete setup process for lie detection training data
# using the new prep.bundle module

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
VENV_NAME="lies310"  # Default virtual environment name
HF_TOKEN=""  # Will be loaded from environment

# Data configuration
S3_BUCKET="dipika-lie-detection-data"  # Your S3 bucket
S3_PREFIX="processed-data/"  # S3 prefix for data
CACHE_DIR=".data"  # Local cache directory

# Default values for bundling
DEFAULT_MODEL="openai/gpt-4o"
DEFAULT_FORMAT="chat"
DEFAULT_BALANCE="downsample"
DEFAULT_OUTPUT_DIR="./bundled_data"

# Function to check AWS credentials
check_aws_credentials() {
    print_status "Checking AWS credentials..."

    # First, check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        print_warning "AWS CLI not found. Installing via pip..."
        pip install awscli
        if ! command -v aws &> /dev/null; then
            print_error "Failed to install AWS CLI"
            return 1
        fi
    fi

    # Check multiple credential sources in order of precedence

    # 1. Check environment variables
    if [[ -n "$AWS_ACCESS_KEY_ID" ]] && [[ -n "$AWS_SECRET_ACCESS_KEY" ]]; then
        print_success "AWS credentials found in environment variables"
        return 0
    fi

    # 2. Check AWS credentials file
    AWS_CREDS_FILE="${AWS_SHARED_CREDENTIALS_FILE:-$HOME/.aws/credentials}"
    if [[ -f "$AWS_CREDS_FILE" ]]; then
        # Check if default profile exists
        if grep -q "\[default\]" "$AWS_CREDS_FILE" 2>/dev/null; then
            print_success "AWS credentials found in $AWS_CREDS_FILE"
            return 0
        fi

        # Check if AWS_PROFILE is set and that profile exists
        if [[ -n "$AWS_PROFILE" ]] && grep -q "\[$AWS_PROFILE\]" "$AWS_CREDS_FILE" 2>/dev/null; then
            print_success "AWS credentials found for profile: $AWS_PROFILE"
            return 0
        fi
    fi

    # 3. Check if running on EC2 with IAM role
    if curl -s -m 2 http://169.254.169.254/latest/meta-data/iam/security-credentials/ &>/dev/null; then
        print_success "Running on EC2 with IAM role"
        return 0
    fi

    # 4. Try to verify credentials by making a simple AWS call
    if aws sts get-caller-identity &>/dev/null; then
        print_success "AWS credentials are configured and working"
        return 0
    fi

    # No credentials found
    print_error "No AWS credentials found!"
    print_status "Please configure AWS credentials using one of these methods:"
    print_status "  1. Set environment variables:"
    print_status "     export AWS_ACCESS_KEY_ID=your-access-key"
    print_status "     export AWS_SECRET_ACCESS_KEY=your-secret-key"
    print_status "  2. Use AWS CLI to configure:"
    print_status "     aws configure"
    print_status "  3. Create credentials file at ~/.aws/credentials"
    print_status "  4. Use IAM roles (if running on EC2)"

    return 1
}

# Function to load environment variables from .env file if it exists
load_env_file() {
    if [[ -f ".env" ]]; then
        print_status "Loading environment variables from .env file..."
        set -a  # automatically export all variables
        source .env
        set +a  # turn off automatic export
        print_success "Environment variables loaded from .env"
    elif [[ -f "../.env" ]]; then
        print_status "Loading environment variables from ../.env file..."
        set -a
        source ../.env
        set +a
        print_success "Environment variables loaded from ../.env"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)
            VENV_NAME="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --train-folds)
            TRAIN_FOLDS="$2"
            shift 2
            ;;
        --eval-folds)
            EVAL_FOLDS="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-bundle)
            SKIP_BUNDLE=true
            shift
            ;;
        --aws-profile)
            export AWS_PROFILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --venv NAME           Virtual environment name (default: lies310)"
            echo "  --model MODEL         Model to process (default: openai/gpt-4o)"
            echo "  --train-folds FOLDS   Comma-separated training folds"
            echo "  --eval-folds FOLDS    Comma-separated evaluation folds (optional)"
            echo "  --format FORMAT       Format type: chat or base_transcript (default: chat)"
            echo "  --size SIZE           Max training examples (optional)"
            echo "  --skip-download       Skip S3 download step"
            echo "  --skip-bundle         Skip bundling step"
            echo "  --aws-profile PROFILE Use specific AWS profile"
            echo "  --help                Show this help message"
            echo ""
            echo "AWS Credentials:"
            echo "  The script will look for AWS credentials in this order:"
            echo "  1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)"
            echo "  2. .env file in current or parent directory"
            echo "  3. AWS credentials file (~/.aws/credentials)"
            echo "  4. IAM role (if running on EC2)"
            echo ""
            echo "Example:"
            echo "  $0 --model openai/gpt-4o --train-folds games,ascii --format chat"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set defaults if not provided
MODEL="${MODEL:-$DEFAULT_MODEL}"
FORMAT="${FORMAT:-$DEFAULT_FORMAT}"

print_status "Starting Training Data Setup with prep.bundle..."
print_status "Configuration:"
print_status "  Model: $MODEL"
print_status "  Format: $FORMAT"
print_status "  Virtual Environment: $VENV_NAME"

# Load environment variables from .env file if it exists
load_env_file

# Check AWS credentials before proceeding
if ! check_aws_credentials; then
    print_error "AWS credentials are required to download data from S3"
    exit 1
fi

# Load HF token from environment if available
if [[ -n "$HF_TOKEN" ]]; then
    print_success "Hugging Face token found in environment"
elif [[ -n "$HUGGING_FACE_HUB_TOKEN" ]]; then
    HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
    print_success "Hugging Face token found in environment (HUGGING_FACE_HUB_TOKEN)"
else
    print_warning "Hugging Face token not found (optional for public models)"
fi

# Step 1: Activate virtual environment
print_status "Step 1: Activating virtual environment..."
if [[ ! -d "$HOME/$VENV_NAME" ]]; then
    # Try current directory
    if [[ -d "./$VENV_NAME" ]]; then
        VENV_PATH="./$VENV_NAME"
    # Try parent directory
    elif [[ -d "../$VENV_NAME" ]]; then
        VENV_PATH="../$VENV_NAME"
    else
        print_error "Virtual environment '$VENV_NAME' not found"
        print_error "Please create the virtual environment first or specify with --venv"
        exit 1
    fi
else
    VENV_PATH="$HOME/$VENV_NAME"
fi

source "$VENV_PATH/bin/activate"
if [[ "$VIRTUAL_ENV" != *"$VENV_NAME"* ]]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi
print_success "Virtual environment activated: $(which python)"

# Step 2: Set environment variables
print_status "Step 2: Setting environment variables..."
export HF_HOME="${HF_HOME:-/workspace/cache}"
export HF_CACHE_HOME="${HF_CACHE_HOME:-/workspace/cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/cache}"
export HF_MODELS_CACHE="${HF_MODELS_CACHE:-/workspace/cache}"
export HF_TOKEN_CACHE="${HF_TOKEN_CACHE:-/workspace/cache}"

if [[ -n "$HF_TOKEN" ]]; then
    export HF_TOKEN="$HF_TOKEN"
fi

# Set prep module environment variables
export S3_BUCKET="$S3_BUCKET"
export S3_PREFIX="$S3_PREFIX"

print_success "Environment variables set"

# Step 3: Install/verify dependencies
print_status "Step 3: Checking dependencies..."

# Check if prep module is available
if python -c "import prep" 2>/dev/null; then
    print_success "prep module is available"
else
    print_error "prep module not found. Please ensure it's in your Python path"
    exit 1
fi

# Install additional dependencies if needed
required_packages=("boto3" "pandas" "numpy" "matplotlib" "transformers" "datasets")
missing_packages=()

for package in "${required_packages[@]}"; do
    if ! python -c "import $package" 2>/dev/null; then
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -gt 0 ]; then
    print_status "Installing missing packages: ${missing_packages[*]}"
    pip install "${missing_packages[@]}"
fi

print_success "All dependencies verified"

# Step 4: Download data from S3 (if not skipped)
if [[ "$SKIP_DOWNLOAD" != true ]]; then
    print_status "Step 4: Downloading data from S3..."

    # Create a Python script to download data
    python -c "
from prep.download import S3DataDownloader
import sys

downloader = S3DataDownloader(cache_dir='$CACHE_DIR')
print(f'Downloading samples for model: $MODEL')

try:
    samples = downloader.get_model_samples('$MODEL')
    print(f'Successfully downloaded/loaded {len(samples)} samples')

    # Show cache info
    cache_info = downloader.get_cache_info()
    for model, info in cache_info.items():
        print(f'Cached: {model} - {info[\"size_mb\"]:.2f} MB')
except Exception as e:
    print(f'Error downloading data: {e}')
    sys.exit(1)
"

    if [ $? -ne 0 ]; then
        print_error "Failed to download data from S3"
        exit 1
    fi

    print_success "Data downloaded/cached successfully"
else
    print_warning "Skipping S3 download step"
fi

# Step 5: Create dataset using prep.dataset
print_status "Step 5: Creating dataset structure..."

# Get model path components
MODEL_PROVIDER=$(echo "$MODEL" | cut -d'/' -f1)
MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)

# Run prep.dataset to create the dataset structure
DATASET_CMD="python -m prep.dataset --model $MODEL --aggregation task-group --balance $DEFAULT_BALANCE"

if [[ -n "$SIZE" ]]; then
    DATASET_CMD="$DATASET_CMD --size $SIZE"
fi

print_status "Running: $DATASET_CMD"
eval $DATASET_CMD

if [ $? -ne 0 ]; then
    print_error "Failed to create dataset"
    exit 1
fi

# Find the created dataset directory
DATASET_DIR=$(find "$CACHE_DIR/$MODEL_PROVIDER" -name "${MODEL_NAME//-/_}" -type d | head -1)
if [[ -z "$DATASET_DIR" ]]; then
    print_error "Failed to find created dataset directory"
    exit 1
fi

print_success "Dataset created in: $DATASET_DIR"

# Step 6: Discover available folds
print_status "Step 6: Discovering available folds..."

python -c "
from prep.bundle.selector import FoldSelector

selector = FoldSelector('$DATASET_DIR')
folds = selector.discover_folds()

print('Available folds:')
for fold_type, fold_names in folds.items():
    print(f'  {fold_type}: {', '.join(fold_names)}')
"

# Step 7: Bundle data for training (if not skipped)
if [[ "$SKIP_BUNDLE" != true ]]; then
    print_status "Step 7: Bundling data for training..."

    # If no folds specified, ask user
    if [[ -z "$TRAIN_FOLDS" ]]; then
        print_warning "No training folds specified"
        echo ""
        echo "Available folds have been discovered above."
        echo "Please specify training folds as comma-separated values."
        echo "Example: games,ascii,reframing"
        echo ""
        read -p "Enter training folds: " TRAIN_FOLDS
    fi

    # Convert comma-separated to space-separated for the command
    TRAIN_FOLDS_ARGS=$(echo "$TRAIN_FOLDS" | tr ',' ' ' | xargs -n1 -I{} echo "--train {}")

    # Build bundle command
    BUNDLE_CMD="python -m prep.bundle \
        --dataset $DATASET_DIR \
        --model $MODEL \
        --format $FORMAT \
        $TRAIN_FOLDS_ARGS \
        --output $DEFAULT_OUTPUT_DIR"

    # Add eval folds if specified
    if [[ -n "$EVAL_FOLDS" ]]; then
        EVAL_FOLDS_ARGS=$(echo "$EVAL_FOLDS" | tr ',' ' ' | xargs -n1 -I{} echo "--eval {}")
        BUNDLE_CMD="$BUNDLE_CMD $EVAL_FOLDS_ARGS"
    fi

    # Add size if specified
    if [[ -n "$SIZE" ]]; then
        BUNDLE_CMD="$BUNDLE_CMD --size $SIZE"
    fi

    print_status "Running bundle command:"
    echo "$BUNDLE_CMD"
    eval $BUNDLE_CMD

    if [ $? -ne 0 ]; then
        print_error "Failed to bundle data"
        exit 1
    fi

    print_success "Data bundled successfully"
else
    print_warning "Skipping bundling step"
fi

# Step 8: Prepare for training
print_status "Step 8: Preparing for training..."

# Find bundled data
BUNDLED_MODEL_DIR="$DEFAULT_OUTPUT_DIR/${MODEL_PROVIDER//-/_}_${MODEL_NAME//-/_}"
if [[ ! -d "$BUNDLED_MODEL_DIR" ]]; then
    print_error "Bundled model directory not found: $BUNDLED_MODEL_DIR"
    exit 1
fi

print_status "Bundled experiments found:"
for exp_dir in "$BUNDLED_MODEL_DIR"/bundled_*; do
    if [[ -d "$exp_dir" ]]; then
        exp_name=$(basename "$exp_dir")
        # Count samples
        train_count=$(wc -l < "$exp_dir/train.jsonl" 2>/dev/null || echo "0")
        val_count=$(wc -l < "$exp_dir/val.jsonl" 2>/dev/null || echo "0")
        echo "  📁 $exp_name"
        echo "     Training samples: $train_count"
        echo "     Validation samples: $val_count"
    fi
done

# Step 9: Generate training commands
print_status "Step 9: Training command examples..."
echo ""
print_success "✅ Setup completed successfully!"
echo ""
print_status "To train your model, use one of these commands:"
echo ""

# Find an example bundled experiment
EXAMPLE_EXP=$(find "$BUNDLED_MODEL_DIR" -name "bundled_*" -type d | head -1)
if [[ -n "$EXAMPLE_EXP" ]]; then
    EXAMPLE_EXP_NAME=$(basename "$EXAMPLE_EXP")

    echo "1. For single experiment training:"
    echo "   python train/sft_part3_general.py \\"
    echo "     --bundled_data_path $EXAMPLE_EXP \\"
    echo "     --format $FORMAT"
    echo ""

    echo "2. For k-fold training (Gemma models):"
    echo "   python train/sft_part3_kfold_trainone.py \\"
    echo "     --use_bundled_data \\"
    echo "     --bundled_base_dir $DEFAULT_OUTPUT_DIR"
    echo ""
fi

echo ""
print_status "Data locations:"
echo "  📁 Downloaded cache: $CACHE_DIR"
echo "  📁 Created dataset: $DATASET_DIR"
echo "  📁 Bundled data: $BUNDLED_MODEL_DIR"
echo ""
print_status "Virtual environment is still active. To deactivate, run:"
echo "  deactivate"
echo ""
print_success "🎉 Training data setup is complete!"