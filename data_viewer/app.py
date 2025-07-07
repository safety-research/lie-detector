from flask import Flask, render_template, request, jsonify
import json
import os
import time

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global variable to store processed data
processed_data = []
last_load_time = None
CACHE_DURATION = 300  # 5 minutes cache

# Configuration - set to True to use local test data instead of S3-synced data
USE_LOCAL_DATA = False  # Set to True for local testing without S3

# Local data configuration
if USE_LOCAL_DATA:
    LOCAL_DATA_PATH = "./local_data"  # Local test data within data_viewer
else:
    LOCAL_DATA_PATH = "../local_data"  # S3-synced data (sibling to data_viewer)

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
    
    # Load data from local cache
    print("Loading data from local cache...")
    load_and_process_data()
    
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
    """Get unique values for filtering"""
    # Load data from local cache
    load_and_process_data()
    
    if not processed_data:
        return jsonify({"error": "No data loaded"}), 400
    
    unique_values = {
        'tasks': list(set(sample.get('task', '') for sample in processed_data if sample.get('task'))),
        'models': list(set(sample.get('model', '') for sample in processed_data if sample.get('model'))),
        'task_ids': list(set(sample.get('task_id', '') for sample in processed_data if sample.get('task_id')))
    }
    
    return jsonify(unique_values)

@app.route('/refresh_data', methods=['POST'])
def refresh_data():
    """Manually reload data from local files"""
    count, message = load_and_process_data(force_reload=True)
    return jsonify({"message": f"Data reloaded. {message}", "sample_count": count})

# Auto-load data when the app starts
print("Starting Lie Detection Data Viewer...")
print(f"Local data path: {LOCAL_DATA_PATH}")
print(f"Using local test data: {USE_LOCAL_DATA}")

# Load data from local files
load_and_process_data()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 