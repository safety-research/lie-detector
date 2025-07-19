from flask import Flask, render_template, request, jsonify
import json
import os
import time
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global variable to store processed data
processed_data = []
last_load_time = None
processed_files = set()  # Track which files have been processed
CACHE_DURATION = 300  # 5 minutes cache

# Configuration - set to True to use local test data instead of S3-synced data
USE_LOCAL_DATA = os.environ.get('USE_LOCAL_DATA', 'True').lower() == 'true'

# S3 Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'dipika-lie-detection-data')
S3_PREFIX = os.environ.get('S3_PREFIX', 'processed-data/')

# Local data configuration
if USE_LOCAL_DATA:
    LOCAL_DATA_PATH = os.environ.get('LOCAL_DATA_PATH', "../data")  # Local test data in root directory
else:
    LOCAL_DATA_PATH = os.environ.get('LOCAL_DATA_PATH', "./local_data")  # S3-synced data (created by sync script in data_viewer)

def list_local_files():
    """List all JSON files in the local data directory and its subfolders"""
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"Local data path does not exist: {LOCAL_DATA_PATH}")
        return []
    
    all_files = []
    
    # Walk through the directory and find all JSON files
    for root, dirs, files in os.walk(LOCAL_DATA_PATH):
        for file in files:
            if file.endswith('.json') or file.endswith('.jsonl'):
                # Get relative path from LOCAL_DATA_PATH
                rel_path = os.path.relpath(os.path.join(root, file), LOCAL_DATA_PATH)
                all_files.append(rel_path)
    
    return all_files

def read_local_file(file_path):
    """Read a file from local filesystem and return its content"""
    full_path = os.path.join(LOCAL_DATA_PATH, file_path)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {full_path}: {e}")
        return None

def get_s3_client():
    """Get S3 client with error handling"""
    try:
        return boto3.client('s3')
    except NoCredentialsError:
        print("AWS credentials not found. Please configure AWS credentials.")
        return None
    except Exception as e:
        print(f"Error creating S3 client: {e}")
        return None

def list_s3_files():
    """List all JSON files in the S3 bucket with the configured prefix"""
    s3_client = get_s3_client()
    if not s3_client:
        return []
    
    try:
        all_files = []
        
        # Use paginator to handle large number of objects
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix=S3_PREFIX
        )
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.json') or obj['Key'].endswith('.jsonl'):
                        all_files.append(obj['Key'])
        
        print(f"Found {len(all_files)} files in S3 bucket {S3_BUCKET}")
        return all_files
        
    except ClientError as e:
        print(f"Error listing S3 files: {e}")
        return []

def read_s3_file(file_key):
    """Read a file directly from S3 and return its content"""
    s3_client = get_s3_client()
    if not s3_client:
        return None
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        return content
    except Exception as e:
        print(f"Error reading S3 file {file_key}: {e}")
        return None

def load_and_process_s3_data(force_reload=False, batch_offset=0, batch_size=50):
    """Load and process data directly from S3 with incremental loading support"""
    global processed_data, last_load_time, processed_files
    
    # Check if we have cached data and it's still fresh (unless forcing reload or doing batch processing)
    if not force_reload and batch_offset == 0 and processed_data and last_load_time:
        time_since_load = time.time() - last_load_time
        if time_since_load < CACHE_DURATION:
            print(f"Using cached data (loaded {time_since_load:.1f}s ago)")
            return len(processed_data), f"Using cached data ({len(processed_data)} samples)"
    
    # If force reload, reset processed files tracking
    if force_reload:
        processed_files.clear()
        processed_data.clear()
        print("Force reload - clearing cache and processed file tracking")
    
    print(f"Loading data from S3 (batch_offset={batch_offset}, batch_size={batch_size})...")
    
    try:
        s3_files = list_s3_files()
        
        if not s3_files:
            print("No files found in S3")
            return 0, "No files found in S3"
        
        # For incremental loading, preserve existing data
        if batch_offset == 0 and not force_reload:
            all_processed_samples = processed_data.copy()
        else:
            all_processed_samples = processed_data.copy()
        
        # Determine which files to process in this batch
        files_to_process = s3_files[batch_offset:batch_offset + batch_size]
        
        # Filter out already processed files
        new_files_to_process = [f for f in files_to_process if f not in processed_files]
        
        print(f"Found {len(s3_files)} total files, processing {len(new_files_to_process)} new files in this batch")
        
        for file_key in new_files_to_process:
            print(f"Processing new S3 file: {file_key}")
            file_content = read_s3_file(file_key)
            
            if file_content:
                try:
                    # Handle JSONL format - each line is a separate JSON object
                    lines = file_content.strip().split('\n')
                    
                    for line_num, line in enumerate(lines):
                        if line.strip():  # Skip empty lines
                            try:
                                sample = json.loads(line.strip())
                                
                                # Create processed sample object with all the fields
                                processed_sample = {
                                    "id": len(all_processed_samples),
                                    "file_key": file_key,
                                    "line_number": line_num + 1,
                                    "sample_id": sample.get("sample_id", ""),
                                    "task": sample.get("task", ""),
                                    "task_id": sample.get("task_id", ""),
                                    "timestamp": sample.get("timestamp", ""),
                                    "model": sample.get("model", ""),
                                    "trace": sample.get("trace", []),
                                    "did_lie": sample.get("did_lie", False),
                                    "evidence": sample.get("evidence", sample.get("explanation", "")),
                                    "metadata": sample.get("metadata", {}),
                                    "scores": sample.get("scores", {})
                                }
                                
                                # Capture any additional fields that might be metadata
                                additional_fields = {}
                                for key, value in sample.items():
                                    if key not in ["sample_id", "task", "task_id", "timestamp", "model", "trace", "did_lie", "metadata", "scores"]:
                                        additional_fields[key] = value
                                
                                if additional_fields:
                                    processed_sample["additional_fields"] = additional_fields
                                
                                # Extract task name from S3 key path if not already set
                                if not processed_sample["task"]:
                                    # Extract folder name from S3 key
                                    # e.g., processed-data/mask-statistics/file.json -> mask-statistics
                                    key_without_prefix = file_key.replace(S3_PREFIX, '')
                                    path_parts = key_without_prefix.split('/')
                                    if len(path_parts) > 1:
                                        folder_name = path_parts[0]  # [folder_name]/file.json
                                        # Replace hyphens with spaces for display
                                        processed_sample["task"] = folder_name.replace('-', ' ')
                                
                                # Always format task name for display (replace underscores with spaces)
                                if processed_sample["task"]:
                                    processed_sample["task"] = processed_sample["task"].replace('_', ' ')
                                
                                all_processed_samples.append(processed_sample)
                            
                            except json.JSONDecodeError as e:
                                print(f"Error parsing JSON on line {line_num + 1} in file {file_key}: {e}")
                                continue
                
                except Exception as e:
                    print(f"Error processing S3 file {file_key}: {e}")
                    continue
                
                # Mark file as processed
                processed_files.add(file_key)
        
        # Update global state
        processed_data = all_processed_samples
        last_load_time = time.time()
        
        # Calculate statistics
        total_files = len(s3_files)
        processed_count = len(processed_files)
        remaining_files = total_files - processed_count
        
        if remaining_files > 0:
            message = f"Loaded {len(all_processed_samples)} samples from S3 (processed {processed_count}/{total_files} files, {remaining_files} remaining)"
        else:
            message = f"Loaded {len(all_processed_samples)} samples from S3 (all {total_files} files processed)"
        
        print(f"Successfully {message}")
        
        return len(all_processed_samples), message, remaining_files
        
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        return 0, f"Error: {str(e)}", 0

def load_and_process_data(force_reload=False):
    """Load and process data from local files"""
    global processed_data, last_load_time
    
    # Check if we have cached data and it's still fresh
    if not force_reload and processed_data and last_load_time:
        time_since_load = time.time() - last_load_time
        if time_since_load < CACHE_DURATION:
            print(f"Using cached data (loaded {time_since_load:.1f}s ago)")
            return len(processed_data), f"Using cached data ({len(processed_data)} samples)"
    
    print("Loading fresh data from local files...")
    
    try:
        files = list_local_files()
        
        if not files:
            print("No files found in local directory")
            return 0, "No files found in local directory"
        
        all_processed_samples = []
        
        for file_path in files:
            print(f"Processing file: {file_path}")
            file_content = read_local_file(file_path)
            
            if file_content:
                try:
                    # Handle JSONL format - each line is a separate JSON object
                    lines = file_content.strip().split('\n')
                    
                    for line_num, line in enumerate(lines):
                        if line.strip():  # Skip empty lines
                            try:
                                sample = json.loads(line.strip())
                                
                                # Create processed sample object with all the fields
                                processed_sample = {
                                    "id": len(all_processed_samples),
                                    "file_key": file_path,
                                    "line_number": line_num + 1,
                                    "sample_id": sample.get("sample_id", ""),
                                    "task": sample.get("task", ""),
                                    "task_id": sample.get("task_id", ""),
                                    "timestamp": sample.get("timestamp", ""),
                                    "model": sample.get("model", ""),
                                    "trace": sample.get("trace", []),
                                    "did_lie": sample.get("did_lie", False),
                                    "evidence": sample.get("evidence", sample.get("explanation", "")),
                                    "metadata": sample.get("metadata", {}),
                                    "scores": sample.get("scores", {})
                                }
                                
                                # Capture any additional fields that might be metadata
                                additional_fields = {}
                                for key, value in sample.items():
                                    if key not in ["sample_id", "task", "task_id", "timestamp", "model", "trace", "did_lie", "metadata", "scores"]:
                                        additional_fields[key] = value
                                
                                if additional_fields:
                                    processed_sample["additional_fields"] = additional_fields
                                
                                # Extract task name from folder path if not already set
                                if not processed_sample["task"]:
                                    # Extract folder name from file path
                                    # e.g., mask-statistics/file.json -> mask-statistics
                                    path_parts = file_path.split('/')
                                    if len(path_parts) > 1:
                                        folder_name = path_parts[0]  # [folder_name]/file.json
                                        # Replace hyphens with spaces for display
                                        processed_sample["task"] = folder_name.replace('-', ' ')
                                
                                # Always format task name for display (replace underscores with spaces)
                                if processed_sample["task"]:
                                    processed_sample["task"] = processed_sample["task"].replace('_', ' ')
                                
                                all_processed_samples.append(processed_sample)
                            
                            except json.JSONDecodeError as e:
                                print(f"Error parsing JSON on line {line_num + 1} in file {file_path}: {e}")
                                continue
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
        
        processed_data = all_processed_samples
        last_load_time = time.time()
        print(f"Successfully loaded {len(all_processed_samples)} samples from local files")
        return len(all_processed_samples), f"Loaded {len(all_processed_samples)} samples from local files"
        
    except Exception as e:
        print(f"Error loading data from local files: {e}")
        return 0, f"Error: {str(e)}"


@app.route('/')
def index():
    """Main page"""
    return render_template('index_new.html')

@app.route('/get_samples', methods=['GET'])
def get_samples():
    """Get random samples with optional filtering"""
    n = request.args.get('n', 5, type=int)
    task = request.args.get('task')  # Optional filter by task
    model = request.args.get('model')  # Optional filter by model
    did_lie = request.args.get('did_lie')  # Optional filter by did_lie (true/false)
    
    # Load data from S3/local based on configuration
    if USE_LOCAL_DATA:
        print("Loading data from local cache...")
        load_and_process_data()
    else:
        print("Loading data from S3...")
        load_and_process_s3_data()
    
    if not processed_data:
        return jsonify({"error": "No data loaded"}), 400
    
    # Filter data based on parameters
    filtered_data = processed_data
    if task:
        filtered_data = [sample for sample in filtered_data if sample.get('task') == task]
    if model:
        filtered_data = [sample for sample in filtered_data if sample.get('model') == model]
    if did_lie is not None:
        # Convert string to boolean
        did_lie_bool = did_lie.lower() == 'true'
        filtered_data = [sample for sample in filtered_data if sample.get('did_lie') == did_lie_bool]
    
    if not filtered_data:
        return jsonify({
            "samples": [],
            "total_samples": len(processed_data),
            "filtered_samples": 0,
            "filters": {
                "task": task,
                "model": model,
                "did_lie": did_lie
            }
        })
    
    # Get first n samples (no shuffling)
    if n >= len(filtered_data):
        samples = filtered_data
    else:
        samples = filtered_data[:n]
    
    return jsonify({
        "samples": samples,
        "total_samples": len(processed_data),
        "filtered_samples": len(filtered_data),
        "filters": {
            "task": task,
            "model": model,
            "did_lie": did_lie
        }
    })

@app.route('/get_sample/<int:sample_id>')
def get_sample(sample_id):
    """Get a specific sample by ID"""
    if not processed_data or sample_id >= len(processed_data):
        return jsonify({"error": "Sample not found"}), 404
    
    return jsonify(processed_data[sample_id])

@app.route('/get_unique_values', methods=['GET'])
def get_unique_values():
    """Get unique values for filtering with counts"""
    try:
        print(f"get_unique_values called - USE_LOCAL_DATA: {USE_LOCAL_DATA}")
        
        # Load data from S3/local based on configuration
        if USE_LOCAL_DATA:
            count, message = load_and_process_data()
            print(f"Local data load result: {count} samples, {message}")
        else:
            result = load_and_process_s3_data()
            if isinstance(result, tuple) and len(result) >= 2:
                count, message = result[:2]
                print(f"S3 data load result: {count} samples, {message}")
        
        if not processed_data:
            error_msg = f"No data loaded from {'local' if USE_LOCAL_DATA else 'S3'}"
            print(f"Error: {error_msg}")
            return jsonify({"error": error_msg}), 400
        
        print(f"Successfully loaded {len(processed_data)} samples for unique values")
    
        # Count occurrences for each unique value
        task_counts = {}
        model_counts = {}
        lie_counts = {'true': 0, 'false': 0}
        
        for sample in processed_data:
            # Count tasks
            task = sample.get('task', '')
            if task:
                task_counts[task] = task_counts.get(task, 0) + 1
                
            # Count models
            model = sample.get('model', '')
            if model:
                model_counts[model] = model_counts.get(model, 0) + 1
                
            # Count lies/truths
            did_lie = sample.get('did_lie', False)
            if did_lie:
                lie_counts['true'] += 1
            else:
                lie_counts['false'] += 1
        
        unique_values = {
            'tasks': [{'value': task, 'count': count} for task, count in sorted(task_counts.items())],
            'models': [{'value': model, 'count': count} for model, count in sorted(model_counts.items())],
            'task_ids': list(set(sample.get('task_id', '') for sample in processed_data if sample.get('task_id'))),
            'lie_counts': lie_counts,
            'total_count': len(processed_data)
        }
        
        return jsonify(unique_values)
    
    except Exception as e:
        print(f"Error in get_unique_values: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/refresh_data', methods=['POST'])
def refresh_data():
    """Manually reload data"""
    # Handle both JSON and form data
    if request.is_json and request.json:
        batch_offset = request.json.get('batch_offset', 0)
        batch_size = request.json.get('batch_size', 50)
    else:
        batch_offset = request.form.get('batch_offset', 0, type=int)
        batch_size = request.form.get('batch_size', 50, type=int)
    
    if USE_LOCAL_DATA:
        count, message = load_and_process_data(force_reload=True)
        return jsonify({"message": f"Data reloaded. {message}", "sample_count": count})
    else:
        result = load_and_process_s3_data(force_reload=True, batch_offset=batch_offset, batch_size=batch_size)
        if len(result) == 3:
            count, message, remaining_files = result
            return jsonify({
                "message": f"Data reloaded. {message}", 
                "sample_count": count,
                "remaining_files": remaining_files,
                "has_more": remaining_files > 0
            })
        else:
            count, message = result[:2]
            return jsonify({"message": f"Data reloaded. {message}", "sample_count": count})

@app.route('/sync_batch', methods=['POST'])
def sync_batch():
    """Sync a specific batch of files from S3"""
    if USE_LOCAL_DATA:
        return jsonify({"error": "Batch sync only available for S3 data source"}), 400
    
    # Handle both JSON and form data
    if request.is_json and request.json:
        batch_offset = request.json.get('batch_offset', 0)
        batch_size = request.json.get('batch_size', 50)
    else:
        batch_offset = request.form.get('batch_offset', 0, type=int)
        batch_size = request.form.get('batch_size', 50, type=int)
    
    result = load_and_process_s3_data(force_reload=False, batch_offset=batch_offset, batch_size=batch_size)
    if len(result) == 3:
        count, message, remaining_files = result
        return jsonify({
            "message": message,
            "sample_count": count,
            "batch_offset": batch_offset,
            "batch_size": batch_size,
            "remaining_files": remaining_files,
            "has_more": remaining_files > 0,
            "next_offset": batch_offset + batch_size if remaining_files > 0 else None
        })
    else:
        count, message = result[:2]
        return jsonify({
            "message": message,
            "sample_count": count,
            "batch_offset": batch_offset,
            "batch_size": batch_size
        })

@app.route('/continue_processing', methods=['POST'])
def continue_processing():
    """Continue processing remaining files from where we left off"""
    if USE_LOCAL_DATA:
        return jsonify({"error": "Continue processing only available for S3 data source"}), 400
    
    try:
        # Get all S3 files
        s3_files = list_s3_files()
        if not s3_files:
            return jsonify({"error": "No files found in S3"}), 400
        
        # Find the next unprocessed file by checking from the end of processed files
        next_offset = len(processed_files)
        
        # If we've processed some files, start from there
        if next_offset >= len(s3_files):
            # All files processed
            return jsonify({
                "message": "All files already processed",
                "sample_count": len(processed_data),
                "remaining_files": 0,
                "has_more": False
            })
        
        # Process next batch starting from the first unprocessed file
        batch_size = 50
        result = load_and_process_s3_data(force_reload=False, batch_offset=next_offset, batch_size=batch_size)
        
        if len(result) == 3:
            count, message, remaining_files = result
            return jsonify({
                "message": message,
                "sample_count": count,
                "batch_offset": next_offset,
                "batch_size": batch_size,
                "remaining_files": remaining_files,
                "has_more": remaining_files > 0,
                "next_offset": next_offset + batch_size if remaining_files > 0 else None
            })
        else:
            count, message = result[:2]
            return jsonify({
                "message": message,
                "sample_count": count,
                "batch_offset": next_offset,
                "batch_size": batch_size
            })
            
    except Exception as e:
        return jsonify({"error": f"Error continuing processing: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get application status and data information"""
    global processed_data, last_load_time, processed_files
    
    # Get total file count for S3 source
    total_files = 0
    if not USE_LOCAL_DATA:
        try:
            s3_files = list_s3_files()
            total_files = len(s3_files)
        except:
            total_files = 0
    
    status_info = {
        "app_status": "healthy",
        "data_source": "local" if USE_LOCAL_DATA else "s3",
        "s3_config": {
            "bucket": S3_BUCKET,
            "prefix": S3_PREFIX
        } if not USE_LOCAL_DATA else None,
        "local_path": LOCAL_DATA_PATH if USE_LOCAL_DATA else None,
        "data_stats": {
            "total_samples": len(processed_data) if processed_data else 0,
            "last_load_time": last_load_time,
            "cache_age_seconds": time.time() - last_load_time if last_load_time else None,
            "cache_duration": CACHE_DURATION
        },
        "file_stats": {
            "total_files": total_files,
            "processed_files": len(processed_files),
            "remaining_files": total_files - len(processed_files),
            "completion_percentage": round((len(processed_files) / total_files * 100), 2) if total_files > 0 else 0
        } if not USE_LOCAL_DATA else None,
        "timestamp": time.time()
    }
    
    return jsonify(status_info)

# Only load data on startup when running locally
if __name__ == '__main__':
    print("Starting Lie Detection Data Viewer...")
    print(f"Local data path: {LOCAL_DATA_PATH}")
    print(f"Using local test data: {USE_LOCAL_DATA}")
    
    # Load data from local files
    load_and_process_data()
    
    app.run(debug=True, host='0.0.0.0', port=8080) 