import json
import os
import tempfile
from typing import List

import boto3
from dotenv import load_dotenv

from common.s3_sample_client import S3SampleClient

load_dotenv()

def download_jsonl_files(bucket: str, prefix: str) -> List[str]:
    """Download all JSONL files from S3 to temporary directory."""
    s3_client = boto3.client('s3')
    temp_dir = tempfile.mkdtemp()
    downloaded_files = []

    print(f"Downloading JSONL files from s3://{bucket}/{prefix}")

    try:
        # List all objects with the given prefix
        paginator = s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']

                # Only process JSONL files
                if key.endswith('.jsonl'):
                    # Create local file path
                    local_file = os.path.join(temp_dir, os.path.basename(key))

                    print(f"Downloading {key}...")
                    s3_client.download_file(bucket, key, local_file)
                    downloaded_files.append(local_file)
                    print(f"Downloaded {key} to {local_file}")

    except Exception as e:
        print(f"Error downloading files: {e}")

    return downloaded_files, temp_dir


def process_jsonl_file(jsonl_file: str, sample_client: S3SampleClient) -> tuple:
    """Process a single JSONL file and upload each record."""
    success_count = 0
    error_count = 0

    print(f"\nProcessing {jsonl_file}...")

    try:
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse JSON object
                    data = json.loads(line.strip())

                    # Extract required fields
                    model = data.get('model', 'unknown')
                    task = data.get('task', 'unknown')
                    sample_id = data.get('sample_id', f'sample_{line_num}')

                    # Upload using S3SampleClient
                    if sample_client.put_sample(model, task, sample_id, data):
                        success_count += 1
                    else:
                        error_count += 1
                        print(f"Failed to upload sample {sample_id} from line {line_num}")

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    error_count += 1
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    error_count += 1

    except Exception as e:
        print(f"Error reading file {jsonl_file}: {e}")

    return success_count, error_count


def main():
    """Main migration function."""
    # Configuration
    SOURCE_BUCKET = 'dipika-lie-detection-data'
    SOURCE_PREFIX = 'processed-data-gemma/'

    # Initialize S3 sample client (it will use the same bucket by default)
    sample_client = S3SampleClient()

    if not sample_client.enabled:
        print("S3 client is not enabled. Please check AWS credentials.")
        return

    # Download all JSONL files
    print("Starting JSONL file download...")
    downloaded_files, temp_dir = download_jsonl_files(SOURCE_BUCKET, SOURCE_PREFIX)

    if not downloaded_files:
        print("No JSONL files found to process.")
        return

    print(f"\nFound {len(downloaded_files)} JSONL files to process.")

    # Process each JSONL file
    total_success = 0
    total_errors = 0

    for jsonl_file in downloaded_files:
        success, errors = process_jsonl_file(jsonl_file, sample_client)
        total_success += success
        total_errors += errors

    # Cleanup
    print(f"\nCleaning up temporary files in {temp_dir}...")
    import shutil
    shutil.rmtree(temp_dir)

    # Summary
    print("\n" + "=" * 50)
    print("Migration Complete!")
    print(f"Total files processed: {len(downloaded_files)}")
    print(f"Total samples uploaded: {total_success}")
    print(f"Total errors: {total_errors}")
    print("=" * 50)


if __name__ == "__main__":
    main()