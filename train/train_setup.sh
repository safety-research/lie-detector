#!/bin/bash

# Training Data Preprocessing Setup Script
# This script automates the complete setup process for lie detection training data

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

# Configuration - UPDATE THESE VALUES
VENV_NAME="lies310"  # Change this to your virtual environment name
AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"  # Update with your AWS access key
AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"  # Update with your AWS secret key
HF_TOKEN="YOUR_HF_TOKEN"  # Update with your Hugging Face token (optional)

# Check if running as root (for apt commands)
if [[ $EUID -eq 0 ]]; then
    print_warning "Running as root - this is fine for containerized environments"
else
    print_warning "Not running as root - some commands may need sudo"
fi

print_status "Starting Training Data Preprocessing Setup..."

# Step 1: Check and install AWS CLI
print_status "Step 1: Checking AWS CLI installation..."
if ! command -v aws &> /dev/null; then
    print_status "AWS CLI not found. Installing..."
    
    # Install unzip if not available
    if ! command -v unzip &> /dev/null; then
        print_status "Installing unzip..."
        if [[ $EUID -eq 0 ]]; then
            apt update && apt install -y unzip
        else
            sudo apt update && sudo apt install -y unzip
        fi
    fi
    
    # Try installing via pip first (easier)
    print_status "Installing AWS CLI via pip..."
    pip install awscli
    
    if ! command -v aws &> /dev/null; then
        print_status "Pip installation failed, trying manual installation..."
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        if [[ $EUID -eq 0 ]]; then
            ./aws/install
        else
            sudo ./aws/install
        fi
        rm -rf awscliv2.zip aws/
    fi
else
    print_success "AWS CLI already installed: $(aws --version)"
fi

# Step 2: Configure AWS credentials
print_status "Step 2: Configuring AWS credentials..."
if [[ "$AWS_ACCESS_KEY_ID" == "YOUR_ACCESS_KEY_ID" ]]; then
    print_error "Please update AWS_ACCESS_KEY_ID in the script before running"
    exit 1
fi

aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set default.region us-east-1
print_success "AWS credentials configured"

# Step 3: Set environment variables
print_status "Step 3: Setting environment variables..."
export HF_HOME=/workspace/cache
export HF_CACHE_HOME=/workspace/cache
export TRANSFORMERS_CACHE=/workspace/cache
export HF_DATASETS_CACHE=/workspace/cache
export HF_HUB_CACHE=/workspace/cache
export HF_MODELS_CACHE=/workspace/cache
export HF_TOKEN_CACHE=/workspace/cache

if [[ "$HF_TOKEN" != "YOUR_HF_TOKEN" ]]; then
    export HF_TOKEN="$HF_TOKEN"
    print_success "Hugging Face token configured"
else
    print_warning "Hugging Face token not set (optional)"
fi

export AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
print_success "Environment variables set"

# Step 4: Activate virtual environment
print_status "Step 4: Activating virtual environment..."
if [[ ! -d "$HOME/$VENV_NAME" ]]; then
    print_error "Virtual environment '$VENV_NAME' not found at $HOME/$VENV_NAME"
    print_error "Please create the virtual environment first or update VENV_NAME in the script"
    exit 1
fi

source "$HOME/$VENV_NAME/bin/activate"
if [[ "$VIRTUAL_ENV" != *"$VENV_NAME"* ]]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi
print_success "Virtual environment activated: $(which python)"

# Step 5: Install dependencies
print_status "Step 5: Installing dependencies..."
if command -v uv &> /dev/null; then
    print_status "Using uv for package management..."
    uv pip install --no-deps -r train/requirements_train.txt
else
    print_warning "uv not found, using pip..."
    pip install -r train/requirements_train.txt
fi

pip install boto3 matplotlib numpy
print_success "Dependencies installed"

# Step 6: Organize S3 data
print_status "Step 6: Organizing S3 data..."
python common/organize_s3_data_by_model_boto3.py

# Find the created directory
ORGANIZED_DIR=$(find . -maxdepth 1 -name "organized_evaluation_*" -type d | head -1)
if [[ -z "$ORGANIZED_DIR" ]]; then
    print_error "Failed to find organized evaluation directory"
    exit 1
fi
print_success "S3 data organized in: $ORGANIZED_DIR"

# Step 7: Clean consecutive assistant messages
print_status "Step 7: Cleaning consecutive assistant messages..."
python train/cleanup_consecutive_assistant_messages.py "$ORGANIZED_DIR"

CLEANED_DIR="${ORGANIZED_DIR}_cleaned"
if [[ ! -d "$CLEANED_DIR" ]]; then
    print_error "Failed to find cleaned directory: $CLEANED_DIR"
    exit 1
fi
print_success "Messages cleaned in: $CLEANED_DIR"

# Step 8: Preprocess training data (all variants)
print_status "Step 8: Preprocessing training data..."

# Standard 80/20 split
print_status "Creating standard 80/20 split..."
python train/preprocess_training_data.py --input_path "$CLEANED_DIR" --taxonomy_path ./common/simple_lie_taxonomy.csv

# Limited training examples (8 per fold)
print_status "Creating limited training examples (8 per fold)..."
python train/preprocess_training_data.py --input_path "$CLEANED_DIR" --taxonomy_path ./common/simple_lie_taxonomy.csv --max_train_examples 8

# Limited training examples (16 per fold)
print_status "Creating limited training examples (16 per fold)..."
python train/preprocess_training_data.py --input_path "$CLEANED_DIR" --taxonomy_path ./common/simple_lie_taxonomy.csv --max_train_examples 16

print_success "All training data preprocessing completed!"

# Step 9: Show results
print_status "Step 9: Setup Summary..."
echo ""
print_success "✅ Setup completed successfully!"
echo ""
print_status "Generated directories:"
find . -maxdepth 1 -name "organized_balanced_training_*" -type d | while read dir; do
    echo "  📁 $dir"
done
echo ""
print_status "Next steps:"
echo "  1. Review the generated training data in the directories above"
echo "  2. Check the training summaries and visualizations"
echo "  3. Use the data for model training"
echo ""
print_status "Virtual environment is still active. To deactivate, run:"
echo "  deactivate"
echo ""
print_success "🎉 Training data preprocessing setup is complete!" 