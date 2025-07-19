#!/usr/bin/env python3
"""
Load JSONL data files into Vercel Blob storage.
This script runs during the Vercel build process.
Uses hierarchical path structure: <model>/<task>/<domain>/<sample>.json
"""

import os
import json
import glob
import hashlib
import requests
from typing import List, Dict, Any
import time
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()
# Vercel Blob configuration from environment variables
BLOB_READ_WRITE_TOKEN = os.environ.get('BLOB_READ_WRITE_TOKEN')

# Data directory paths
DATA_DIRS = [
    "../../data",  # Main data directory
    #"../../evals/data"  # Evals data directory
]

def clean_name(name: str) -> str:
    """Clean name for use in blob paths."""
    if not name:
        return "unknown"
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(":", "_")

def get_sample_domain(sample: Dict[str, Any]) -> str:
    """Extract or infer domain from sample metadata."""
    # Try to get domain from metadata
    if 'metadata' in sample and isinstance(sample['metadata'], dict):
        domain = sample['metadata'].get('domain', '')
        if domain:
            return clean_name(domain)
    
    # Try to infer from task name
    task = sample.get('task', '').lower()
    if 'sandbagging' in task:
        if 'physical_security' in task:
            return 'physical_security'
        elif 'digital_security' in task:
            return 'digital_security'
        elif 'chemistry_biology' in task:
            return 'chemistry_biology'
        elif 'automotive_mechanical' in task:
            return 'automotive_mechanical'
        elif 'medical_pharmaceutical' in task:
            return 'medical_pharmaceutical'
        elif 'financial_systems' in task:
            return 'financial_systems'
        else:
            return 'general_sandbagging'
    elif 'mask' in task:
        return 'masking'
    elif 'correctness' in task:
        return 'correctness'
    elif 'harmfulness' in task:
        return 'harmfulness'
    elif 'sycophancy' in task:
        return 'sycophancy'
    else:
        return 'general'

def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of parsed JSON objects."""
    samples = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        sample = json.loads(line.strip())
                        # Add file metadata
                        sample['_source_file'] = os.path.basename(filepath)
                        sample['_line_number'] = line_num + 1
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num + 1} in {filepath}: {e}")
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    
    return samples

def put_blob(pathname: str, content: str) -> bool:
    """Upload content as a blob with hierarchical pathname."""
    if not BLOB_READ_WRITE_TOKEN:
        return False
    
    try:
        url = f"https://blob.vercel-storage.com/{quote(pathname, safe='/')}"
        
        headers = {
            "Authorization": f"Bearer {BLOB_READ_WRITE_TOKEN}",
            "x-content-type": "application/json"
        }
        
        response = requests.put(url, data=content.encode('utf-8'), headers=headers)
        
        if response.status_code in [200, 201]:
            return True
        else:
            print(f"Blob upload failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error uploading blob {pathname}: {e}")
        return False

def store_samples_in_blob(samples: List[Dict[str, Any]]):
    """Store samples in Vercel Blob with hierarchical path structure."""
    if not samples:
        return
    
    print(f"Storing {len(samples)} samples in Vercel Blob...")
    
    uploaded = 0
    failed = 0
    
    for i, sample in enumerate(samples):
        try:
            # Extract metadata for path construction
            model = clean_name(sample.get('model', 'unknown'))
            task = clean_name(sample.get('task', 'unknown'))
            domain = get_sample_domain(sample)
            
            # Create unique sample identifier
            sample_id = sample.get('sample_id', f"sample_{i}")
            timestamp = sample.get('timestamp', str(int(time.time())))
            
            # Clean sample_id for filename
            clean_sample_id = clean_name(sample_id)
            
            # Construct hierarchical path: model/task/domain/sample_id_timestamp.json
            pathname = f"{model}/{task}/{domain}/{clean_sample_id}_{timestamp}.json"
            
            # Convert sample to JSON string
            content = json.dumps(sample, indent=2)
            
            # Upload to blob
            if put_blob(pathname, content):
                uploaded += 1
                if uploaded % 50 == 0:
                    print(f"Uploaded {uploaded}/{len(samples)} samples...")
            else:
                failed += 1
                print(f"Failed to upload sample {pathname}")
            
            # Rate limiting to avoid overwhelming the API
            if i % 10 == 0:
                time.sleep(0.1)
                
        except Exception as e:
            failed += 1
            print(f"Error processing sample {i}: {e}")
    
    print(f"Upload complete: {uploaded} successful, {failed} failed")

def create_index_blob(samples: List[Dict[str, Any]]):
    """Create index blob with metadata about all samples."""
    try:
        # Analyze samples to create index
        models = set()
        tasks = set()
        domains = set()
        model_counts = {}
        task_counts = {}
        domain_counts = {}
        lie_counts = {"true": 0, "false": 0}
        
        for sample in samples:
            model = clean_name(sample.get('model', 'unknown'))
            task = clean_name(sample.get('task', 'unknown'))
            domain = get_sample_domain(sample)
            
            models.add(model)
            tasks.add(task)
            domains.add(domain)
            
            model_counts[model] = model_counts.get(model, 0) + 1
            task_counts[task] = task_counts.get(task, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            if sample.get('did_lie', False):
                lie_counts["true"] += 1
            else:
                lie_counts["false"] += 1
        
        # Create index data
        index_data = {
            "metadata": {
                "total_samples": len(samples),
                "last_updated": time.time(),
                "update_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            },
            "counts": {
                "models": model_counts,
                "tasks": task_counts,
                "domains": domain_counts,
                "lie_status": lie_counts
            },
            "unique_values": {
                "models": sorted(list(models)),
                "tasks": sorted(list(tasks)),
                "domains": sorted(list(domains))
            }
        }
        
        # Upload index
        index_content = json.dumps(index_data, indent=2)
        if put_blob("_index/metadata.json", index_content):
            print("Successfully created index blob")
        else:
            print("Failed to create index blob")
            
    except Exception as e:
        print(f"Error creating index blob: {e}")

def main():
    """Main function to load all JSONL data into Vercel Blob."""
    if not BLOB_READ_WRITE_TOKEN:
        print("Vercel Blob not configured. Skipping data load.")
        print("Set BLOB_READ_WRITE_TOKEN environment variable.")
        return
    
    print("Starting Vercel Blob data load...")
    
    all_samples = []
    processed_files = []
    
    # Find all JSONL files
    for data_dir in DATA_DIRS:
        if os.path.exists(data_dir):
            pattern = os.path.join(data_dir, "**/*.jsonl")
            jsonl_files = glob.glob(pattern, recursive=True)
            
            for filepath in jsonl_files:
                print(f"Processing {filepath}...")
                samples = load_jsonl_file(filepath)
                if samples:
                    all_samples.extend(samples)
                    processed_files.append(filepath)
                    print(f"  Loaded {len(samples)} samples")
    
    print(f"\nTotal files processed: {len(processed_files)}")
    print(f"Total samples loaded: {len(all_samples)}")
    
    if not all_samples:
        print("No samples to upload")
        return
    
    # Store all samples using hierarchical paths
    store_samples_in_blob(all_samples)
    
    # Create index for fast lookups
    create_index_blob(all_samples)
    
    print("\nVercel Blob data load complete!")
    print("Data structure: <model>/<task>/<domain>/<sample>.json")
    print("Index available at: _index/metadata.json")

if __name__ == "__main__":
    main()