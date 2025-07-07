#!/usr/bin/env python3
"""
S3 Data Sync Script

This script syncs data from S3 to a local directory for the Lie Detection Data Viewer.
Run this script separately from the web app to keep local data fresh.
"""

import os
import time
import threading
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Configuration
S3_BUCKET = "dipika-lie-detection-data"
S3_PREFIX = "processed-data/"
LOCAL_DATA_PATH = "./local_data"  # Local cache in the data_viewer directory
SYNC_INTERVAL = 10  # Sync every 10 seconds

def get_s3_client():
    """Get S3 client with error handling"""
    try:
        return boto3.client('s3')
    except NoCredentialsError:
        print("AWS credentials not found. Please configure AWS credentials.")
        return None

def list_s3_files():
    """List all JSON files in the processed-data prefix and its subfolders"""
    s3_client = get_s3_client()
    if not s3_client:
        return []
    
    try:
        # First, list all folders under processed-data/
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=S3_PREFIX,
            Delimiter='/'
        )
        
        all_files = []
        
        # Get files from the root processed-data/ folder
        root_response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=S3_PREFIX
        )
        
        if 'Contents' in root_response:
            for obj in root_response['Contents']:
                if obj['Key'].endswith('.json') or obj['Key'].endswith('.jsonl'):
                    all_files.append(obj['Key'])
        
        # Get files from subfolders
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                folder_prefix = prefix['Prefix']
                folder_response = s3_client.list_objects_v2(
                    Bucket=S3_BUCKET,
                    Prefix=folder_prefix
                )
                
                if 'Contents' in folder_response:
                    for obj in folder_response['Contents']:
                        if obj['Key'].endswith('.json') or obj['Key'].endswith('.jsonl'):
                            all_files.append(obj['Key'])
        
        return all_files
    except ClientError as e:
        print(f"Error listing S3 files: {e}")
        return []

def download_file_from_s3_to_local(file_key):
    """Download a file from S3 to local directory"""
    s3_client = get_s3_client()
    if not s3_client:
        return False
    
    try:
        # Create local directory structure
        local_file_path = os.path.join(LOCAL_DATA_PATH, file_key.replace(S3_PREFIX, ''))
        local_dir = os.path.dirname(local_file_path)
        os.makedirs(local_dir, exist_ok=True)
        
        # Download file from S3
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        with open(local_file_path, 'wb') as f:
            f.write(response['Body'].read())
        
        print(f"Downloaded: {file_key} -> {local_file_path}")
        return True
    except Exception as e:
        print(f"Error downloading {file_key}: {e}")
        return False

def sync_s3_to_local():
    """Sync all files from S3 to local directory"""
    print(f"[{time.strftime('%H:%M:%S')}] Starting S3 to local sync...")
    
    s3_client = get_s3_client()
    if not s3_client:
        print("AWS credentials not configured for sync")
        return
    
    try:
        # Get all files from S3
        s3_files = list_s3_files()
        
        # Create local directory if it doesn't exist
        os.makedirs(LOCAL_DATA_PATH, exist_ok=True)
        
        # Download each file
        downloaded_count = 0
        for file_key in s3_files:
            if download_file_from_s3_to_local(file_key):
                downloaded_count += 1
        
        print(f"[{time.strftime('%H:%M:%S')}] S3 sync complete: {downloaded_count}/{len(s3_files)} files")
        
    except Exception as e:
        print(f"Error during S3 sync: {e}")

def background_sync_worker():
    """Background worker that syncs S3 to local every SYNC_INTERVAL seconds"""
    print(f"Background sync started - syncing every {SYNC_INTERVAL} seconds")
    while True:
        try:
            sync_s3_to_local()
        except Exception as e:
            print(f"Background sync error: {e}")
        
        time.sleep(SYNC_INTERVAL)

def main():
    """Main function"""
    print("S3 Data Sync Script")
    print(f"Local data path: {LOCAL_DATA_PATH}")
    print(f"Sync interval: {SYNC_INTERVAL} seconds")
    print("Press Ctrl+C to stop")
    
    # Initial sync
    print("\nPerforming initial sync...")
    sync_s3_to_local()
    
    # Start background sync
    print("\nStarting background sync...")
    try:
        background_sync_worker()
    except KeyboardInterrupt:
        print("\nSync stopped by user")

if __name__ == '__main__':
    main() 