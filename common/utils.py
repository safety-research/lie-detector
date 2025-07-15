import boto3
import json
import os
import subprocess
from typing import Optional, Tuple
from datetime import datetime, timezone
from typing import Optional

def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """
    Parse an S3 URI to extract bucket and prefix.
    
    Args:
        s3_uri: S3 URI in format s3://bucket-name/prefix/
        
    Returns:
        Tuple of (bucket_name, prefix)
        
    Raises:
        ValueError: If URI format is invalid
    """
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI format: {s3_uri}. Must start with 's3://'")
    
    # Remove s3:// prefix
    path = s3_uri[5:]
    
    # Split into bucket and prefix
    parts = path.split('/', 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    # Ensure prefix ends with / if not empty
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    
    return bucket, prefix

def copy_s3_data_to_local(s3_uri: str, base_local_dir: str = "baseline") -> str:
    """
    Copy data from S3 to a local directory with datetime suffix.
    
    Args:
        s3_uri: S3 URI to copy from (e.g., "s3://bucket-name/prefix/")
        base_local_dir: Base directory to create the training data folder in
        
    Returns:
        Path to the local directory containing the copied data
        
    Raises:
        ValueError: If S3 URI is invalid
        RuntimeError: If aws CLI command fails
    """
    try:
        bucket, prefix = parse_s3_uri(s3_uri)
    except ValueError as e:
        raise ValueError(f"Invalid S3 URI: {e}")
    
    # Create datetime suffix for unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_dir = os.path.join(base_local_dir, f"training_data_{timestamp}")
    
    # Ensure the base directory exists
    os.makedirs(base_local_dir, exist_ok=True)
    
    print(f"Copying data from {s3_uri} to {local_dir}...")
    
    # Use aws CLI to copy data
    try:
        # Create the local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # Run aws s3 sync command
        cmd = [
            "aws", "s3", "sync", 
            s3_uri, 
            local_dir,
            "--quiet"  # Reduce output verbosity
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Count files copied
        file_count = 0
        for root, dirs, files in os.walk(local_dir):
            file_count += len([f for f in files if f.endswith('.jsonl')])
        
        print(f"Successfully copied {file_count} JSONL files to {local_dir}")
        return local_dir
        
    except subprocess.CalledProcessError as e:
        error_msg = f"AWS CLI command failed: {e.stderr}"
        print(f"Error: {error_msg}")
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error copying S3 data: {e}"
        print(f"Error: {error_msg}")
        raise RuntimeError(error_msg)

def write_to_s3(data: str, filename: str, bucket: str = "dipika-lie-detection-data", prefix: str = "processed-data-v4/", clean_task_name: Optional[str] = None):
    """
    Write data to S3 bucket.
    
    Args:
        data: String data to write
        filename: Name of the file to create in S3
        bucket: S3 bucket name
        prefix: S3 key prefix
        clean_task_name: Clean task name (with - instead of _) to use as subdirectory
    """
    try:
        s3_client = boto3.client('s3')
        
        # Create the S3 key with optional subdirectory
        if clean_task_name:
            key = f"{prefix}{clean_task_name}/{filename}"
        else:
            key = f"{prefix}{filename}"
        
        # Upload the data
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data.encode('utf-8'),
            ContentType='application/json'
        )
        
        print(f"[S3Upload] Successfully uploaded to s3://{bucket}/{key}")
        return True
    except Exception as e:
        print(f"[S3Upload] Error uploading to S3: {e}")
        return False

def clean_s3_processed_data_by_sample_id(bucket: str = "dipika-lie-detection-data", prefix: str = "processed-data-v4/", dry_run: bool = True):
    """
    Iterate through each folder in the S3 bucket and for each model, create consolidated JSONL files 
    with only the newest sample_id for each unique sample. This deduplicates based on sample_id 
    within each model, keeping the newest version of each sample.
    
    Args:
        bucket: S3 bucket name
        prefix: Prefix to search under
        dry_run: If True, only print what would be done without actually making changes
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    # Get all folders (prefixes) under the main prefix
    result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    folders = [cp['Prefix'] for cp in result.get('CommonPrefixes', [])]
    print(f"Found {len(folders)} folders in {bucket}/{prefix}")
    
    total_files_before = 0
    total_files_after = 0
    total_files_to_delete = 0
    
    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        # List all JSONL files in the folder
        files = []
        for page in paginator.paginate(Bucket=bucket, Prefix=folder):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.jsonl'):
                    files.append({
                        'key': key,
                        'last_modified': obj['LastModified']
                    })
        if not files:
            print(f"  No files found in {folder}")
            continue
            
        total_files_before += len(files)
        
        # For each file, read all lines and group by model and sample_id
        model_to_samples = {}
        for file in files:
            obj = s3.get_object(Bucket=bucket, Key=file['key'])
            try:
                content = obj['Body'].read().decode('utf-8')
                lines = content.strip().split('\n')
                
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        model = data.get('model', 'unknown')
                        sample_id = data.get('sample_id')
                        
                        if not sample_id:
                            print(f"    Warning: No sample_id found in {file['key']}")
                            continue
                            
                        if model not in model_to_samples:
                            model_to_samples[model] = {}
                            
                        if sample_id not in model_to_samples[model]:
                            model_to_samples[model][sample_id] = []
                            
                        model_to_samples[model][sample_id].append({
                            'file_key': file['key'],
                            'last_modified': file['last_modified'],
                            'data': data
                        })
                        
                    except json.JSONDecodeError as e:
                        print(f"    Warning: Invalid JSON in {file['key']}: {e}")
                        continue
                        
            except Exception as e:
                print(f"  Error reading {file['key']}: {e}")
                continue
        
        # For each model, deduplicate by sample_id and create consolidated files
        for model, samples in model_to_samples.items():
            print(f"  Model: {model}")
            model_samples_before = sum(len(sample_list) for sample_list in samples.values())
            model_samples_after = len(samples)  # One sample per sample_id
            model_samples_to_delete = model_samples_before - model_samples_after
            
            print(f"    Samples before: {model_samples_before}, Samples after: {model_samples_after}, Samples to delete: {model_samples_to_delete}")
            
            # Collect the newest sample for each sample_id
            deduplicated_samples = []
            for sample_id, sample_list in samples.items():
                if len(sample_list) <= 1:
                    print(f"    Sample {sample_id}: Only 1 version, keeping")
                    deduplicated_samples.append(sample_list[0]['data'])
                    continue
                    
                # Sort by last_modified descending
                sample_list.sort(key=lambda x: x['last_modified'], reverse=True)
                newest = sample_list[0]
                to_delete = sample_list[1:]
                
                print(f"    Sample {sample_id}: Keeping newest (modified: {newest['last_modified']})")
                if to_delete:
                    print(f"      Would delete {len(to_delete)} older versions:")
                    for sample in to_delete:
                        print(f"        - {sample['file_key']} (modified: {sample['last_modified']})")
                
                deduplicated_samples.append(newest['data'])
            
            # Create consolidated file content
            consolidated_content = '\n'.join(json.dumps(sample) for sample in deduplicated_samples)
            
            # Generate new filename for consolidated file
            folder_name = folder.rstrip('/').split('/')[-1]
            model_safe = model.replace('/', '_').replace(':', '_')
            new_filename = f"deduplicated_{folder_name}_{model_safe}.jsonl"
            new_key = f"{folder}{new_filename}"
            
            print(f"    Would create consolidated file: {new_key} with {len(deduplicated_samples)} samples")
            
            if not dry_run:
                # Upload the consolidated file
                try:
                    s3.put_object(
                        Bucket=bucket,
                        Key=new_key,
                        Body=consolidated_content.encode('utf-8'),
                        ContentType='application/json'
                    )
                    print(f"    Created consolidated file: {new_key}")
                except Exception as e:
                    print(f"    Error creating consolidated file {new_key}: {e}")
                    continue
                
                # Delete all original files for this model (but NOT the newly created consolidated file)
                files_to_delete = set()
                for sample_list in samples.values():
                    for sample in sample_list:
                        files_to_delete.add(sample['file_key'])
                
                # Remove the newly created consolidated file from deletion list
                files_to_delete.discard(new_key)
                
                for file_key in files_to_delete:
                    try:
                        s3.delete_object(Bucket=bucket, Key=file_key)
                        print(f"    Deleted original file: {file_key}")
                    except Exception as e:
                        print(f"    Error deleting {file_key}: {e}")
            
            total_files_after += 1  # One consolidated file per model
            total_files_to_delete += len(files)  # All original files will be deleted
        
        print(f"  Folder summary: Files before: {len(files)}, Files after: {len(model_to_samples)}, Files to delete: {len(files)}")
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}Summary:")
    print(f"Total files before: {total_files_before}")
    print(f"Total files after: {total_files_after}")
    print(f"Total files to delete: {total_files_to_delete}")
    
    if dry_run:
        print("\nThis was a dry run. No files were actually created or deleted.")
        print("To actually perform the cleanup, call with dry_run=False")
    else:
        print("\nCleanup complete!")

def clean_s3_processed_data(bucket: str = "dipika-lie-detection-data", prefix: str = "processed-data/", dry_run: bool = True):
    """
    Iterate through each folder in the S3 bucket and for each model, keep only the newest JSONL file.
    Each JSONL file has a 'model' key. Older files for each model are deleted.
    
    Args:
        bucket: S3 bucket name
        prefix: Prefix to search under
        dry_run: If True, only print what would be deleted without actually deleting
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    # Get all folders (prefixes) under the main prefix
    result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    folders = [cp['Prefix'] for cp in result.get('CommonPrefixes', [])]
    print(f"Found {len(folders)} folders in {bucket}/{prefix}")
    
    total_files_before = 0
    total_files_after = 0
    total_files_to_delete = 0
    
    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        # List all JSONL files in the folder
        files = []
        for page in paginator.paginate(Bucket=bucket, Prefix=folder):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.jsonl'):
                    files.append({
                        'key': key,
                        'last_modified': obj['LastModified']
                    })
        if not files:
            print(f"  No files found in {folder}")
            continue
            
        # For each file, get the model key from the first line
        model_to_files = {}
        for file in files:
            obj = s3.get_object(Bucket=bucket, Key=file['key'])
            try:
                first_line = obj['Body'].readline().decode('utf-8')
                data = json.loads(first_line)
                model = data.get('model', 'unknown')
            except Exception as e:
                print(f"  Error reading {file['key']}: {e}")
                model = 'unknown'
            if model not in model_to_files:
                model_to_files[model] = []
            model_to_files[model].append(file)
        
        folder_files_before = sum(len(file_list) for file_list in model_to_files.values())
        folder_files_after = len(model_to_files)  # One file per model
        folder_files_to_delete = folder_files_before - folder_files_after
        
        total_files_before += folder_files_before
        total_files_after += folder_files_after
        total_files_to_delete += folder_files_to_delete
        
        print(f"  Files before: {folder_files_before}, Files after: {folder_files_after}, Files to delete: {folder_files_to_delete}")
        
        # For each model, keep only the newest file
        for model, file_list in model_to_files.items():
            if len(file_list) <= 1:
                print(f"  Model: {model} - Only 1 file, keeping: {file_list[0]['key']}")
                continue
            # Sort by last_modified descending
            file_list.sort(key=lambda x: x['last_modified'], reverse=True)
            newest = file_list[0]
            to_delete = file_list[1:]
            print(f"  Model: {model}")
            print(f"    Keeping: {newest['key']} (modified: {newest['last_modified']})")
            print(f"    Would delete {len(to_delete)} files:")
            for file in to_delete:
                print(f"      - {file['key']} (modified: {file['last_modified']})")
            
            # Actually delete the files if not dry run
            if not dry_run:
                for file in to_delete:
                    try:
                        s3.delete_object(Bucket=bucket, Key=file['key'])
                        print(f"    Deleted {file['key']}")
                    except Exception as e:
                        print(f"    Error deleting {file['key']}: {e}")
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}Summary:")
    print(f"Total files before: {total_files_before}")
    print(f"Total files after: {total_files_after}")
    print(f"Total files to delete: {total_files_to_delete}")
    
    if dry_run:
        print("\nThis was a dry run. No files were actually deleted.")
        print("To actually delete files, call with dry_run=False")
    else:
        print("\nCleanup complete!")

def copy_s3_folder_to_local(s3_folder_url: str, local_dir: str):
    """
    Copy all files from an S3 folder (s3://bucket/prefix/) to a local directory, preserving subdirectory structure.
    Args:
        s3_folder_url: S3 folder URL (e.g., s3://bucket/prefix/)
        local_dir: Local directory to copy files into
    """
    import re
    s3_pattern = r"s3://([^/]+)/(.+)"
    m = re.match(s3_pattern, s3_folder_url.rstrip('/'))
    if not m:
        raise ValueError(f"Invalid S3 folder URL: {s3_folder_url}")
    bucket, prefix = m.group(1), m.group(2)
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    print(f"Listing objects in s3://{bucket}/{prefix}/ ...")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Only copy files, not folders
            if key.endswith('/'):
                continue
            rel_path = os.path.relpath(key, prefix)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            print(f"Downloading s3://{bucket}/{key} -> {local_path}")
            s3_client.download_file(bucket, key, local_path)
    print(f"All files copied from s3://{bucket}/{prefix}/ to {local_dir}")
