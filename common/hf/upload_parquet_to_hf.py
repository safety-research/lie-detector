#!/usr/bin/env python3
"""
Upload Parquet files to Hugging Face without Git conflicts.

This script handles uploading data to Hugging Face when your data
is already inside an existing Git repository.

Usage:
    python upload_parquet_to_hf.py --repo-id username/dataset-name
    python upload_parquet_to_hf.py --repo-id username/dataset-name --data-dir ./parquet_output
"""

import argparse
import tempfile
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def upload_with_api(repo_id: str, data_dir: Path, token: str = None, private: bool = False):
    """
    Upload files directly using Hugging Face API without Git.
    This avoids any Git repository conflicts.
    """
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token
        )
        logger.info(f"Repository {repo_id} ready")
    except Exception as e:
        logger.error(f"Error creating repository: {e}")
        return False

    # Find all Parquet files
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No Parquet files found in {data_dir}")
        return False

    logger.info(f"Found {len(parquet_files)} Parquet files to upload")

    # Upload each file individually
    for parquet_file in parquet_files:
        try:
            logger.info(f"Uploading {parquet_file.name}...")
            api.upload_file(
                path_or_fileobj=str(parquet_file),
                path_in_repo=parquet_file.name,
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
        except Exception as e:
            logger.error(f"Error uploading {parquet_file.name}: {e}")
            return False

    # Upload README if it exists
    readme_path = data_dir / "README.md"
    if readme_path.exists():
        try:
            logger.info("Uploading README.md...")
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
        except Exception as e:
            logger.warning(f"Error uploading README: {e}")

    logger.info(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
    return True


def upload_with_temp_directory(repo_id: str, data_dir: Path, token: str = None, private: bool = False):
    """
    Upload files by creating a temporary directory outside the existing repo.
    This completely avoids any Git conflicts.
    """
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token
        )
    except Exception as e:
        logger.error(f"Error creating repository: {e}")
        return False

    # Create temporary directory outside current repo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logger.info(f"Created temporary directory: {temp_path}")

        # Copy Parquet files to temp directory
        parquet_files = list(data_dir.glob("*.parquet"))
        for parquet_file in parquet_files:
            shutil.copy2(parquet_file, temp_path / parquet_file.name)
            logger.info(f"Copied {parquet_file.name}")

        # Copy README if exists
        readme_path = data_dir / "README.md"
        if readme_path.exists():
            shutil.copy2(readme_path, temp_path / "README.md")

        # Upload the entire folder
        try:
            logger.info("Uploading to Hugging Face...")
            api.upload_folder(
                folder_path=str(temp_path),
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
            logger.info(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
            return True
        except Exception as e:
            logger.error(f"Error uploading: {e}")
            return False


def create_upload_script(data_dir: Path, output_path: Path = None):
    """
    Create a shell script that can be run from outside the repository.
    """
    if output_path is None:
        output_path = Path("upload_to_hf.sh")

    script_content = f"""#!/bin/bash
# Upload script for Hugging Face dataset
# Run this from OUTSIDE your main repository

# Configuration
REPO_ID="YOUR_USERNAME/YOUR_DATASET_NAME"
DATA_DIR="{data_dir.absolute()}"

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Created temp directory: $TEMP_DIR"

# Copy files
cp "$DATA_DIR"/*.parquet "$TEMP_DIR/"
if [ -f "$DATA_DIR/README.md" ]; then
    cp "$DATA_DIR/README.md" "$TEMP_DIR/"
fi

# Clone HF repo in temp location
cd "$TEMP_DIR"
git clone "https://huggingface.co/datasets/$REPO_ID" hf_dataset
cd hf_dataset

# Copy files and push
cp ../*.parquet .
if [ -f ../README.md ]; then
    cp ../README.md .
fi

git lfs track "*.parquet"
git add .
git commit -m "Update dataset"
git push

# Cleanup
cd /
rm -rf "$TEMP_DIR"

echo "Upload complete!"
echo "View at: https://huggingface.co/datasets/$REPO_ID"
"""

    with open(output_path, 'w') as f:
        f.write(script_content)

    output_path.chmod(0o755)  # Make executable
    logger.info(f"Created upload script: {output_path}")
    logger.info("Edit the REPO_ID in the script, then run it from outside your repository")


def main():
    parser = argparse.ArgumentParser(
        description="Upload Parquet files to Hugging Face from within an existing repo"
    )
    parser.add_argument("--repo-id", type=str, default="Noddybear/lies",help="HF dataset repo ID (username/dataset-name)")
    parser.add_argument("--data-dir", type=Path, default=Path("./parquet_output"),
                        help="Directory containing Parquet files")
    parser.add_argument("--method", choices=["api", "temp", "script"], default="api",
                        help="Upload method: api (direct), temp (via temp dir), script (create script)")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    parser.add_argument("--token", help="HF token (or use HF_TOKEN env var)")

    args = parser.parse_args()

    if args.method == "script":
        # Just create the script
        create_upload_script(args.data_dir)
        return

    if not args.repo_id:
        parser.error("--repo-id is required for upload methods 'api' and 'temp'")

    # Validate repo ID format
    if "/" not in args.repo_id:
        parser.error("--repo-id must be in format 'username/dataset-name'")

    # Check data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory {args.data_dir} does not exist")
        return

    # Upload based on method
    if args.method == "api":
        success = upload_with_api(args.repo_id, args.data_dir, args.token, args.private)
    else:  # temp
        success = upload_with_temp_directory(args.repo_id, args.data_dir, args.token, args.private)

    if success:
        logger.info("\nNext steps:")
        logger.info("1. Your dataset is now available on Hugging Face")
        logger.info("2. The data viewer should automatically show your data")
        logger.info("3. You can load it with:")
        logger.info(f'   dataset = load_dataset("{args.repo_id}")')


if __name__ == "__main__":
    main()