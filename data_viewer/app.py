from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import json
import os
import time
import boto3
import random
from botocore.exceptions import ClientError, NoCredentialsError

load_dotenv()
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Lightweight caching for metadata and file listings
metadata_cache = {}
file_list_cache = None
cache_timestamp = 0
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
    LOCAL_DATA_PATH = os.environ.get('LOCAL_DATA_PATH',
                                     "./local_data")  # S3-synced data (created by sync script in data_viewer)


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


def extract_metadata_from_s3_path(file_key):
    """Extract metadata from S3 file path structure"""
    # Expected format: prefix/provider/model/task/domain/truth_tag_sample_id.json
    try:
        # Remove prefix and .json extension
        path_without_prefix = file_key.replace(S3_PREFIX, '')
        if path_without_prefix.endswith('.json'):
            path_without_prefix = path_without_prefix[:-5]

        # Split path into components
        parts = path_without_prefix.split('/')

        if len(parts) >= 5:
            provider = parts[0]
            model = parts[1]
            task = parts[2]
            domain = parts[3]
            filename = parts[4]

            # Extract truth tag from filename (t_ or f_ prefix)
            did_lie = None
            if filename.startswith('t_'):
                did_lie = True
                sample_id = filename[2:]
            elif filename.startswith('f_'):
                did_lie = False
                sample_id = filename[2:]
            else:
                sample_id = filename

            return {
                "provider": provider,
                "model": model,
                "task": task,
                "domain": domain,
                "did_lie": did_lie,
                "sample_id": sample_id,
                "full_model": f"{provider}/{model}"
            }
        else:
            # Fallback for non-standard paths
            return {
                "provider": "unknown",
                "model": "unknown",
                "task": path_without_prefix.split('/')[0] if parts else "unknown",
                "domain": "unknown",
                "did_lie": None,
                "sample_id": "unknown",
                "full_model": "unknown"
            }
    except Exception as e:
        print(f"Error extracting metadata from path {file_key}: {e}")
        return {
            "provider": "unknown",
            "model": "unknown",
            "task": "unknown",
            "domain": "unknown",
            "did_lie": None,
            "sample_id": "unknown",
            "full_model": "unknown"
        }


def get_cached_file_list():
    """Get cached list of S3 files or refresh if needed"""
    global file_list_cache, cache_timestamp

    current_time = time.time()
    if file_list_cache is None or (current_time - cache_timestamp) > CACHE_DURATION:
        print("Refreshing S3 file list cache...")
        if USE_LOCAL_DATA:
            file_list_cache = list_local_files()
        else:
            file_list_cache = list_s3_files()
        cache_timestamp = current_time
        print(f"Cached {len(file_list_cache) if file_list_cache else 0} files")

    return file_list_cache or []


def get_file_metadata_only(file_key):
    """Extract metadata from file path without loading file content"""
    if USE_LOCAL_DATA:
        # For local files, we need minimal metadata extraction
        path_parts = file_key.split('/')
        return {
            "provider": "local",
            "model": "local",
            "task": path_parts[0] if path_parts else "unknown",
            "domain": "local",
            "did_lie": None,
            "sample_id": file_key,
            "full_model": "local"
        }
    else:
        return extract_metadata_from_s3_path(file_key)


def load_sample_from_file(file_key):
    """Load a single sample from S3 or local file on-demand"""
    # Check cache first
    if file_key in metadata_cache:
        return metadata_cache[file_key]

    try:
        if USE_LOCAL_DATA:
            file_content = read_local_file(file_key)
        else:
            file_content = read_s3_file(file_key)

        if not file_content:
            return None

        # Parse the file content
        if file_key.endswith('.jsonl'):
            # Handle JSONL format - return first non-empty line
            lines = file_content.strip().split('\n')
            for line in lines:
                if line.strip():
                    sample = json.loads(line.strip())
                    break
            else:
                return None
        else:
            # Handle individual JSON files
            sample = json.loads(file_content.strip())

        # Extract metadata from file path
        path_metadata = get_file_metadata_only(file_key)

        # Create processed sample object
        processed_sample = {
            "file_key": file_key,
            "sample_id": sample.get("sample_id", path_metadata["sample_id"]),
            "task": sample.get("task", path_metadata["task"]),
            "task_id": sample.get("task_id", ""),
            "timestamp": sample.get("timestamp", ""),
            "model": sample.get("model", path_metadata["full_model"]),
            "provider": path_metadata["provider"],
            "model_name": path_metadata["model"],
            "domain": path_metadata["domain"],
            "trace": sample.get("trace", []),
            "did_lie": sample.get("did_lie", path_metadata["did_lie"]),
            "evidence": sample.get("evidence", sample.get("explanation", "")),
            "metadata": sample.get("metadata", {}),
            "scores": sample.get("scores", {})
        }

        # Capture any additional fields that might be metadata
        additional_fields = {}
        for key, value in sample.items():
            if key not in ["sample_id", "task", "task_id", "timestamp", "model", "trace", "did_lie", "metadata",
                           "scores", "evidence", "explanation"]:
                additional_fields[key] = value

        if additional_fields:
            processed_sample["additional_fields"] = additional_fields

        # Always format task name for display (replace underscores with spaces)
        if processed_sample["task"]:
            processed_sample["task"] = processed_sample["task"].replace('_', ' ')

        if processed_sample["domain"]:
            processed_sample["domain"] = processed_sample["domain"].replace('_', ' ')

        # Cache the result
        metadata_cache[file_key] = processed_sample

        return processed_sample

    except Exception as e:
        print(f"Error loading sample from {file_key}: {e}")
        return None


# Removed bulk loading function - now using lazy loading


@app.route('/')
def index():
    """Main page"""
    return render_template('index_new.html')


@app.route('/get_samples', methods=['GET'])
def get_samples():
    """Get random samples with optional filtering using lazy loading"""
    n = request.args.get('n', 5, type=int)
    task = request.args.get('task')  # Optional filter by task
    model = request.args.get('model')  # Optional filter by model
    provider = request.args.get('provider')  # Optional filter by provider
    domain = request.args.get('domain')  # Optional filter by domain
    did_lie = request.args.get('did_lie')  # Optional filter by did_lie (true/false)

    try:
        # Get cached file list
        all_files = get_cached_file_list()
        if not all_files:
            return jsonify({"error": "No files found"}), 400

        # Filter files based on metadata extracted from paths
        filtered_files = []
        total_files = len(all_files)

        for file_key in all_files:
            file_metadata = get_file_metadata_only(file_key)

            # Apply filters
            if task and file_metadata.get('task', '').replace('_', ' ') != task:
                continue
            if model and file_metadata.get('full_model', '') != model:
                continue
            if provider and file_metadata.get('provider', '') != provider:
                continue
            if domain and file_metadata.get('domain', '').replace('_', ' ') != domain:
                continue
            if did_lie is not None:
                did_lie_bool = did_lie.lower() == 'true'
                if file_metadata.get('did_lie') != did_lie_bool:
                    continue

            filtered_files.append(file_key)

        # Limit to n samples
        files_to_load = filtered_files[:n] if len(filtered_files) > n else filtered_files

        # Load samples on-demand
        samples = []
        for file_key in files_to_load:
            sample = load_sample_from_file(file_key)
            if sample:
                samples.append(sample)

        return jsonify({
            "samples": samples,
            "total_samples": total_files,
            "filtered_samples": len(filtered_files),
            "filters": {
                "task": task,
                "model": model,
                "provider": provider,
                "domain": domain,
                "did_lie": did_lie
            }
        })

    except Exception as e:
        print(f"Error in get_samples: {e}")
        return jsonify({"error": f"Error loading samples: {str(e)}"}), 500


@app.route('/get_sample_by_file/<path:file_key>')
def get_sample_by_file(file_key):
    """Get a specific sample by file key"""
    try:
        sample = load_sample_from_file(file_key)
        if sample:
            return jsonify(sample)
        else:
            return jsonify({"error": "Sample not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Error loading sample: {str(e)}"}), 500


@app.route('/get_unique_values', methods=['GET'])
def get_unique_values():
    """Get unique values for filtering with counts using lazy loading"""
    try:
        print(f"get_unique_values called - USE_LOCAL_DATA: {USE_LOCAL_DATA}")

        # Get current filter selections from query parameters
        current_task = request.args.get('task')
        current_model = request.args.get('model')
        current_provider = request.args.get('provider')
        current_domain = request.args.get('domain')
        current_did_lie = request.args.get('did_lie')

        # Get cached file list
        all_files = get_cached_file_list()
        if not all_files:
            return jsonify({"error": "No files found"}), 400

        print(
            f"Analyzing {len(all_files)} files for unique values with filters: task={current_task}, model={current_model}, provider={current_provider}, domain={current_domain}, did_lie={current_did_lie}")

        # Count occurrences for each unique value by analyzing file paths
        task_counts = {}
        model_counts = {}
        provider_counts = {}
        domain_counts = {}
        lie_counts = {'true': 0, 'false': 0}

        for file_key in all_files:
            metadata = get_file_metadata_only(file_key)

            # Extract all metadata values
            task = metadata.get('task', '').replace('_', ' ') if metadata.get('task') else ''
            model = metadata.get('full_model', '')
            provider = metadata.get('provider', '')
            domain = metadata.get('domain', '').replace('_', ' ') if metadata.get('domain') else ''
            did_lie = metadata.get('did_lie')

            # Check if this file matches ALL current filters
            # We only count it if it would be included in the current filtered view
            matches_filters = True

            if current_task and task != current_task:
                matches_filters = False
            if current_model and model != current_model:
                matches_filters = False
            if current_provider and provider != current_provider:
                matches_filters = False
            if current_domain and domain != current_domain:
                matches_filters = False
            if current_did_lie is not None:
                current_did_lie_bool = current_did_lie.lower() == 'true'
                if did_lie != current_did_lie_bool:
                    matches_filters = False

            # Only count this file if it matches all current filters
            if matches_filters:
                # Count tasks (but not if task is currently filtered)
                if not current_task and task and task != 'unknown':
                    task_counts[task] = task_counts.get(task, 0) + 1

                # Count models (but not if model is currently filtered)
                if not current_model and model and model != 'unknown':
                    model_counts[model] = model_counts.get(model, 0) + 1

                # Count providers (but not if provider is currently filtered)
                if not current_provider and provider and provider != 'unknown':
                    provider_counts[provider] = provider_counts.get(provider, 0) + 1

                # Count domains (but not if domain is currently filtered)
                if not current_domain and domain and domain != 'unknown':
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1

                # Count lies/truths (but not if did_lie is currently filtered)
                if current_did_lie is None:
                    if did_lie is True:
                        lie_counts['true'] += 1
                    elif did_lie is False:
                        lie_counts['false'] += 1

        # For currently filtered attributes, show all options but with conditional counts
        if current_task:
            # If task is filtered, still show all tasks but with counts based on other filters
            for file_key in all_files:
                metadata = get_file_metadata_only(file_key)
                task = metadata.get('task', '').replace('_', ' ') if metadata.get('task') else ''

                if task and task != 'unknown':
                    # Check if it matches all OTHER filters (not task)
                    matches_other_filters = True
                    if current_model and metadata.get('full_model', '') != current_model:
                        matches_other_filters = False
                    if current_provider and metadata.get('provider', '') != current_provider:
                        matches_other_filters = False
                    if current_domain and metadata.get('domain', '').replace('_', ' ') != current_domain:
                        matches_other_filters = False
                    if current_did_lie is not None:
                        current_did_lie_bool = current_did_lie.lower() == 'true'
                        if metadata.get('did_lie') != current_did_lie_bool:
                            matches_other_filters = False

                    if matches_other_filters:
                        task_counts[task] = task_counts.get(task, 0) + 1

        # Similar logic for other filtered attributes
        if current_model:
            for file_key in all_files:
                metadata = get_file_metadata_only(file_key)
                model = metadata.get('full_model', '')

                if model and model != 'unknown':
                    matches_other_filters = True
                    if current_task and metadata.get('task', '').replace('_', ' ') != current_task:
                        matches_other_filters = False
                    if current_provider and metadata.get('provider', '') != current_provider:
                        matches_other_filters = False
                    if current_domain and metadata.get('domain', '').replace('_', ' ') != current_domain:
                        matches_other_filters = False
                    if current_did_lie is not None:
                        current_did_lie_bool = current_did_lie.lower() == 'true'
                        if metadata.get('did_lie') != current_did_lie_bool:
                            matches_other_filters = False

                    if matches_other_filters:
                        model_counts[model] = model_counts.get(model, 0) + 1

        unique_values = {
            'tasks': [{'value': task, 'count': count} for task, count in sorted(task_counts.items()) if count >= 5],
            'models': [{'value': model, 'count': count} for model, count in sorted(model_counts.items())],
            'providers': [{'value': provider, 'count': count} for provider, count in sorted(provider_counts.items())],
            'domains': [{'value': domain, 'count': count} for domain, count in sorted(domain_counts.items())],
            'task_ids': [],  # Not easily extractable from file paths
            'lie_counts': lie_counts,
            'total_count': sum(1 for file_key in all_files if all([
                not current_task or get_file_metadata_only(file_key).get('task', '').replace('_', ' ') == current_task,
                not current_model or get_file_metadata_only(file_key).get('full_model', '') == current_model,
                not current_provider or get_file_metadata_only(file_key).get('provider', '') == current_provider,
                not current_domain or get_file_metadata_only(file_key).get('domain', '').replace('_',
                                                                                                 ' ') == current_domain,
                current_did_lie is None or get_file_metadata_only(file_key).get('did_lie') == (
                            current_did_lie.lower() == 'true')
            ]))
        }

        print(
            f"Found {len(task_counts)} tasks, {len(model_counts)} models, {len(provider_counts)} providers, {len(domain_counts)} domains (with filters applied)")
        return jsonify(unique_values)

    except Exception as e:
        print(f"Error in get_unique_values: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/refresh_data', methods=['POST'])
def refresh_data():
    """Manually refresh file list cache"""
    global file_list_cache, cache_timestamp

    try:
        # Force refresh the file list cache
        file_list_cache = None
        cache_timestamp = 0

        # Get fresh file list
        files = get_cached_file_list()

        return jsonify({
            "message": f"File list refreshed successfully",
            "file_count": len(files)
        })

    except Exception as e:
        return jsonify({"error": f"Error refreshing data: {str(e)}"}), 500


# Removed batch sync endpoint - no longer needed with lazy loading

# Removed continue processing endpoint - no longer needed with lazy loading

@app.route('/status', methods=['GET'])
def status():
    """Get application status and data information"""
    global file_list_cache, cache_timestamp

    # Get total file count
    total_files = 0
    try:
        files = get_cached_file_list()
        total_files = len(files)
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
        "cache_stats": {
            "file_list_cached": file_list_cache is not None,
            "cache_age_seconds": time.time() - cache_timestamp if cache_timestamp else None,
            "cache_duration": CACHE_DURATION,
            "total_files": total_files
        },
        "timestamp": time.time()
    }

    return jsonify(status_info)


# Start the app with lazy loading - no bulk data loading on startup
if __name__ == '__main__':
    print("Starting Lie Detection Data Viewer...")
    print(f"Local data path: {LOCAL_DATA_PATH}")
    print(f"Using local test data: {USE_LOCAL_DATA}")
    print("Using lazy loading - data will be loaded on-demand")

    app.run(debug=True, host='0.0.0.0', port=8080) 