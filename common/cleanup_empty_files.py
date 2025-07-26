#!/usr/bin/env python3
"""
Script to clean up empty (0-byte) JSONL files from S3 bucket.
"""

import boto3
import sys

def cleanup_empty_files(bucket: str = "dipika-lie-detection-data", prefix: str = "processed-data-v4-copy/", dry_run: bool = True):
    """
    Find and delete all empty (0-byte) JSONL files in the S3 bucket.
    
    Args:
        bucket: S3 bucket name
        prefix: Prefix to search under
        dry_run: If True, only print what would be deleted without actually deleting
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    print(f"Searching for empty JSONL files in {bucket}/{prefix}")
    print("=" * 60)
    
    empty_files = []
    total_files_checked = 0
    
    # List all objects under the prefix
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            size = obj['Size']
            
            # Check if it's a JSONL file
            if key.endswith('.jsonl'):
                total_files_checked += 1
                
                # Check if it's empty (0 bytes)
                if size == 0:
                    empty_files.append({
                        'key': key,
                        'size': size,
                        'last_modified': obj['LastModified']
                    })
    
    print(f"Checked {total_files_checked} JSONL files")
    print(f"Found {len(empty_files)} empty files")
    print()
    
    if not empty_files:
        print("No empty files found!")
        return
    
    print("Empty files found:")
    for file in empty_files:
        print(f"  - {file['key']} (0 bytes, modified: {file['last_modified']})")
    
    print()
    
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
    # You can modify these parameters as needed
    cleanup_empty_files(
        bucket="dipika-lie-detection-data",
        prefix="processed-data-v4-copy/",
        dry_run=False  # Set to False to actually delete files
    ) 