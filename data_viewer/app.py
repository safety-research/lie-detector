from flask import Flask, render_template, request, jsonify
import json
import os
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global variable to store processed data
processed_data = []

# S3 configuration
S3_BUCKET = "dipika-lie-detection-data"
S3_PREFIX = "processed-data/"

# Local data configuration for testing
USE_LOCAL_DATA = True  # Set to False to use S3
LOCAL_DATA_PATH = "./test_data"  # Change this to your local data directory

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

def download_s3_file(file_key):
    """Download a file from S3 and return its content"""
    s3_client = get_s3_client()
    if not s3_client:
        return None
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        return response['Body'].read().decode('utf-8')
    except ClientError as e:
        print(f"Error downloading file {file_key}: {e}")
        return None

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

def load_and_process_data_from_s3():
    """Load and process data from S3 or local files"""
    global processed_data
    
    if USE_LOCAL_DATA:
        print("Loading fresh data from local files...")
        return load_and_process_data_from_local()
    else:
        print("Loading fresh data from S3...")
        return load_and_process_data_from_s3_only()

def load_and_process_data_from_local():
    """Load and process data from local files"""
    global processed_data
    
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
        print(f"Successfully loaded {len(all_processed_samples)} samples from local files")
        return len(all_processed_samples), f"Loaded {len(all_processed_samples)} samples from local files"
        
    except Exception as e:
        print(f"Error loading data from local files: {e}")
        return 0, f"Error: {str(e)}"

def load_and_process_data_from_s3_only():
    """Load and process data from S3"""
    global processed_data
    
    s3_client = get_s3_client()
    if not s3_client:
        print("AWS credentials not configured")
        return 0, "AWS credentials not configured"
    
    try:
        files = list_s3_files()
        
        if not files:
            print("No files found in S3")
            return 0, "No files found in S3"
        
        all_processed_samples = []
        
        for file_key in files:
            print(f"Processing file: {file_key}")
            file_content = download_s3_file(file_key)
            
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
                                    # e.g., processed-data/mask-statistics/file.json -> mask-statistics
                                    path_parts = file_key.split('/')
                                    if len(path_parts) > 2:
                                        folder_name = path_parts[1]  # processed-data/[folder_name]/file.json
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
                    print(f"Error processing file {file_key}: {e}")
                    continue
        
        processed_data = all_processed_samples
        print(f"Successfully loaded {len(all_processed_samples)} samples from S3")
        return len(all_processed_samples), f"Loaded {len(all_processed_samples)} samples from S3"
        
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        return 0, f"Error: {str(e)}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index_new.html')

@app.route('/get_samples', methods=['GET'])
def get_samples():
    """Get random samples with optional filtering - loads fresh data from S3"""
    n = request.args.get('n', 5, type=int)
    task = request.args.get('task')  # Optional filter by task
    model = request.args.get('model')  # Optional filter by model
    did_lie = request.args.get('did_lie')  # Optional filter by did_lie (true/false)
    
    # Load fresh data from S3 every time
    print("Loading fresh data from S3...")
    load_and_process_data_from_s3()
    
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
    """Get unique values for filtering - loads fresh data from S3"""
    # Load fresh data from S3 every time
    print("Loading fresh data from S3 for unique values...")
    load_and_process_data_from_s3()
    
    if not processed_data:
        return jsonify({"error": "No data loaded"}), 400
    
    unique_values = {
        'tasks': list(set(sample.get('task', '') for sample in processed_data if sample.get('task'))),
        'models': list(set(sample.get('model', '') for sample in processed_data if sample.get('model'))),
        'task_ids': list(set(sample.get('task_id', '') for sample in processed_data if sample.get('task_id')))
    }
    
    return jsonify(unique_values)

# Auto-load data when the app starts
print("Starting Lie Detection Data Viewer...")
if USE_LOCAL_DATA:
    print(f"Auto-loading data from local path: {LOCAL_DATA_PATH}")
else:
    print("Auto-loading data from S3...")
load_and_process_data_from_s3()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 