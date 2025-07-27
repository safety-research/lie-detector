#!/usr/bin/env python3
"""
Script to clean up empty (0-byte) JSONL files and then deduplicate by sample_id in S3 bucket.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the common directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))

from utils import clean_s3_processed_data_by_sample_id
import boto3

def cleanup_empty_files(bucket: str = "dipika-lie-detection-data", prefix: str = "processed-data-v4-copy/", dry_run: bool = True):
    """
    Find and delete all empty (0-byte) JSONL files in the S3 bucket.
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    print(f"Searching for empty JSONL files in {bucket}/{prefix}")
    print("=" * 60)
    empty_files = []
    total_files_checked = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            size = obj['Size']
            if key.endswith('.jsonl'):
                total_files_checked += 1
                if size == 0:
                    empty_files.append({'key': key, 'size': size, 'last_modified': obj['LastModified']})
    print(f"Checked {total_files_checked} JSONL files")
    print(f"Found {len(empty_files)} empty files")
    if not empty_files:
        print("No empty files found!")
        return
    print("Empty files found:")
    for file in empty_files:
        print(f"  - {file['key']} (0 bytes, modified: {file['last_modified']})")
    if dry_run:
        print("DRY RUN: Would delete the above empty files")
        print("To actually delete them, run with dry_run=False")
    else:
        print("Deleting empty files...")
        deleted_count = 0
        for file in empty_files:
            try:
                s3.delete_object(Bucket=bucket, Key=file['key'])
                print(f"  Deleted: {file['key']}")
                deleted_count += 1
            except Exception as e:
                print(f"  Error deleting {file['key']}: {e}")
        print(f"\nSuccessfully deleted {deleted_count} empty files")

if __name__ == "__main__":
    BUCKET = "dipika-lie-detection-data"
    PREFIX = "processed-data-gemma/"
    DRY_RUN = False  # Set to True for dry run, False to actually perform cleanup

    print("Step 1: Cleaning up empty files...")
    cleanup_empty_files(bucket=BUCKET, prefix=PREFIX, dry_run=DRY_RUN)
    print("\nStep 2: Deduplicating by sample_id...")
    clean_s3_processed_data_by_sample_id(bucket=BUCKET, prefix=PREFIX, dry_run=DRY_RUN) 