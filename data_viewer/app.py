from flask import Flask, render_template, request, jsonify
import json
import os
import random
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

app = Flask(__name__)

# Global variable to store processed data
processed_data = []

# S3 configuration
S3_BUCKET = "dipika-lie-detection-data"
S3_PREFIX = "generated-data/"

def get_s3_client():
    """Get S3 client with error handling"""
    try:
        return boto3.client('s3')
    except NoCredentialsError:
        print("AWS credentials not found. Please configure AWS credentials.")
        return None

def list_s3_folders():
    """List all folders in the generated-data prefix"""
    s3_client = get_s3_client()
    if not s3_client:
        return []
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=S3_PREFIX,
            Delimiter='/'
        )
        
        folders = []
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                folder_name = prefix['Prefix'].replace(S3_PREFIX, '').rstrip('/')
                if folder_name:  # Skip empty folder names
                    folders.append(folder_name)
        
        return folders
    except ClientError as e:
        print(f"Error listing S3 folders: {e}")
        return []

def list_s3_files(folder_name):
    """List all JSON files in a specific folder"""
    s3_client = get_s3_client()
    if not s3_client:
        return []
    
    try:
        prefix = f"{S3_PREFIX}{folder_name}/"
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix
        )
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.json'):
                    files.append(obj['Key'])
        
        return files
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

def clean_trace_full_turns(trace):
    """Define a clean_trace function that keeps full speaker/turn structure"""
    return [
        {"turn_number": msg["turn_number"], msg["speaker"]: msg["message"]}
        for msg in trace
    ]

# def tag_lies(trace, lies):
#     """Tag lies in the trace with <lie> tags"""
#     tagged_trace = trace.copy()
#     for lie in lies:
#         i = lie["turn_number"]
#         utterance = lie["utterance"]
#         if i >= len(tagged_trace):
#             continue  # skip if turn index is out of range
#         new_turn = tagged_trace[i].copy()
#         for speaker in ["blue", "red"]:
#             if speaker in new_turn:
#                 if utterance in new_turn[speaker]:
#                     new_turn[speaker] = new_turn[speaker].replace(
#                         utterance, f"<lie>{utterance}</lie>"
#                     )
#         tagged_trace[i] = new_turn
#     return tagged_trace


def tag_lies(trace, lies):
    tagged_trace = []
    for turn in trace:
        new_turn = turn.copy()
        for speaker in ["blue", "red"]:
            if speaker in new_turn:
                for lie in lies:
                    utterance = lie["utterance"]
                    if utterance in new_turn[speaker]:
                        new_turn[speaker] = new_turn[speaker].replace(
                            utterance, f"<lie>{utterance}</lie>"
                        )
        tagged_trace.append(new_turn)
    return tagged_trace

def filter_lie_turns(trace, lies):
    """Filter trace to only show turns that contain lies"""
    lie_turn_numbers = {lie["turn_number"] for lie in lies}
    return [turn for i, turn in enumerate(trace) if i in lie_turn_numbers]

def load_and_process_data_from_s3(folder_name=None):
    """Load and process data from S3"""
    global processed_data
    
    s3_client = get_s3_client()
    if not s3_client:
        return 0, "AWS credentials not configured"
    
    try:
        # If no specific folder, get all folders
        if folder_name:
            folders = [folder_name]
        else:
            folders = list_s3_folders()
        
        if not folders:
            return 0, "No folders found in S3"
        
        all_processed_samples = []
        
        for folder in folders:
            print(f"Processing folder: {folder}")
            files = list_s3_files(folder)
            
            for file_key in files:
                print(f"Processing file: {file_key}")
                file_content = download_s3_file(file_key)
                
                if file_content:
                    try:
                        data = json.loads(file_content)
                        samples = data.get("samples", [])
                        
                        # Process each sample
                        for i, sample in enumerate(samples):
                            # Extract raw trace
                            raw_trace = sample["store"]["GameState:interrogation"]
                            
                            # Clean trace to structured format
                            structured_trace = clean_trace_full_turns(raw_trace)
                            
                            # Extract lie metadata
                            lie_metadata = sample.get("scores", {}).get("lie_detection_scorer", {}).get("metadata", {}).get("all_lies", [])
                            
                            # Tag lies in the trace
                            tagged_trace = tag_lies(structured_trace, lie_metadata)
                            
                            # Filter to only show turns with lies
                            lie_only_trace = filter_lie_turns(tagged_trace, lie_metadata)
                            
                            # Create sample object with folder tag
                            processed_sample = {
                                "id": len(all_processed_samples),
                                "folder_tag": folder,
                                "file_key": file_key,
                                "raw_trace": raw_trace,
                                "structured_trace": structured_trace,
                                "tagged_trace": tagged_trace,
                                "lie_only_trace": lie_only_trace,
                                "lie_metadata": lie_metadata,
                                "num_lies": len(lie_metadata),
                                "num_turns": len(structured_trace)
                            }
                            all_processed_samples.append(processed_sample)
                    
                    except Exception as e:
                        print(f"Error processing file {file_key}: {e}")
                        continue
        
        processed_data = all_processed_samples
        return len(all_processed_samples), f"Loaded {len(all_processed_samples)} samples from {len(folders)} folders"
        
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        return 0, f"Error: {str(e)}"

def load_and_process_data(file_path):
    """Load and process the JSON file (kept for backward compatibility)"""
    global processed_data
    
    try:
        # Load the JSON file
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract all samples
        samples = data["samples"]
        
        # Process each sample
        processed_samples = []
        for i, sample in enumerate(samples):
            # Extract raw trace
            raw_trace = sample["store"]["GameState:interrogation"]
            
            # Clean trace to structured format
            structured_trace = clean_trace_full_turns(raw_trace)
            
            # Extract lie metadata
            lie_metadata = sample.get("scores", {}).get("lie_detection_scorer", {}).get("metadata", {}).get("all_lies", [])
            
            # Tag lies in the trace
            tagged_trace = tag_lies(structured_trace, lie_metadata)
            
            # Filter to only show turns with lies
            lie_only_trace = filter_lie_turns(tagged_trace, lie_metadata)
            
            # Create sample object
            processed_sample = {
                "id": i,
                "folder_tag": "local_file",
                "file_key": file_path,
                "raw_trace": raw_trace,
                "structured_trace": structured_trace,
                "tagged_trace": tagged_trace,
                "lie_only_trace": lie_only_trace,
                "lie_metadata": lie_metadata,
                "num_lies": len(lie_metadata),
                "num_turns": len(structured_trace)
            }
            processed_samples.append(processed_sample)
        
        processed_data = processed_samples
        return len(processed_samples)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return 0

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/list_folders', methods=['GET'])
def list_folders():
    """List available S3 folders"""
    folders = list_s3_folders()
    return jsonify({
        "folders": folders,
        "total_folders": len(folders)
    })

@app.route('/load_data_from_s3', methods=['POST'])
def load_data_from_s3():
    """Load data from S3"""
    folder_name = request.json.get('folder_name')  # Optional: specific folder
    num_samples, message = load_and_process_data_from_s3(folder_name)
    
    if num_samples == 0:
        return jsonify({"error": message}), 400
    
    return jsonify({
        "success": True,
        "num_samples": num_samples,
        "message": message
    })

@app.route('/load_data', methods=['POST'])
def load_data():
    """Load data from a file (kept for backward compatibility)"""
    file_path = request.json.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 400
    
    num_samples = load_and_process_data(file_path)
    return jsonify({
        "success": True,
        "num_samples": num_samples,
        "message": f"Loaded {num_samples} samples"
    })

@app.route('/get_samples', methods=['GET'])
def get_samples():
    """Get random samples with optional folder filtering"""
    n = request.args.get('n', 5, type=int)
    folder_tag = request.args.get('folder_tag')  # Optional filter by folder
    
    if not processed_data:
        return jsonify({"error": "No data loaded"}), 400
    
    # Filter by folder tag if specified
    filtered_data = processed_data
    if folder_tag:
        filtered_data = [sample for sample in processed_data if sample.get('folder_tag') == folder_tag]
    
    if not filtered_data:
        return jsonify({"error": f"No samples found for folder tag: {folder_tag}"}), 400
    
    # Get random samples
    if n >= len(filtered_data):
        samples = filtered_data
    else:
        samples = random.sample(filtered_data, n)
    
    return jsonify({
        "samples": samples,
        "total_samples": len(processed_data),
        "filtered_samples": len(filtered_data),
        "folder_tag": folder_tag
    })

@app.route('/get_sample/<int:sample_id>')
def get_sample(sample_id):
    """Get a specific sample by ID"""
    if not processed_data or sample_id >= len(processed_data):
        return jsonify({"error": "Sample not found"}), 404
    
    return jsonify(processed_data[sample_id])

@app.route('/get_folder_stats', methods=['GET'])
def get_folder_stats():
    """Get statistics by folder"""
    if not processed_data:
        return jsonify({"error": "No data loaded"}), 400
    
    folder_stats = {}
    for sample in processed_data:
        folder_tag = sample.get('folder_tag', 'unknown')
        if folder_tag not in folder_stats:
            folder_stats[folder_tag] = {
                'count': 0,
                'total_lies': 0,
                'total_turns': 0
            }
        
        folder_stats[folder_tag]['count'] += 1
        folder_stats[folder_tag]['total_lies'] += sample.get('num_lies', 0)
        folder_stats[folder_tag]['total_turns'] += sample.get('num_turns', 0)
    
    # Calculate averages
    for folder in folder_stats:
        count = folder_stats[folder]['count']
        folder_stats[folder]['avg_lies'] = folder_stats[folder]['total_lies'] / count if count > 0 else 0
        folder_stats[folder]['avg_turns'] = folder_stats[folder]['total_turns'] / count if count > 0 else 0
    
    return jsonify(folder_stats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9009) 