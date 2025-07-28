#!/usr/bin/env python3
"""
Comprehensive script that combines cleanup and sample counting functionality.
This script:
1. Cleans up empty (0-byte) JSONL files in S3
2. Deduplicates files by sample_id, keeping the newest version
3. Counts samples per task and model, outputting results as CSV
"""

import boto3
import json
import csv
import sys
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Configuration
BUCKET_NAME = "dipika-lie-detection-data"
PREFIX = "processed-data-gemma/"
DRY_RUN = False  # Set to True for dry run, False to actually perform cleanup

class S3CleanupAndCounter:
    """Class to handle S3 cleanup and sample counting operations."""
    
    def __init__(self, bucket: str = BUCKET_NAME, prefix: str = PREFIX, dry_run: bool = DRY_RUN):
        self.bucket = bucket
        self.prefix = prefix
        self.dry_run = dry_run
        self.s3 = boto3.client('s3')
        self.paginator = self.s3.get_paginator('list_objects_v2')
    
    def cleanup_empty_files(self) -> int:
        """
        Find and delete all empty (0-byte) JSONL files in the S3 bucket.
        
        Returns:
            int: Number of empty files found
        """
        print(f"Searching for empty JSONL files in {self.bucket}/{self.prefix}")
        print("=" * 60)
        
        empty_files = []
        total_files_checked = 0
        
        for page in self.paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                size = obj['Size']
                if key.endswith('.jsonl'):
                    total_files_checked += 1
                    if size == 0:
                        empty_files.append({
                            'key': key, 
                            'size': size, 
                            'last_modified': obj['LastModified']
                        })
        
        print(f"Checked {total_files_checked} JSONL files")
        print(f"Found {len(empty_files)} empty files")
        
        if not empty_files:
            print("No empty files found!")
            return 0
        
        print("Empty files found:")
        for file in empty_files:
            print(f"  - {file['key']} (0 bytes, modified: {file['last_modified']})")
        
        if self.dry_run:
            print("DRY RUN: Would delete the above empty files")
            print("To actually delete them, run with dry_run=False")
        else:
            print("Deleting empty files...")
            deleted_count = 0
            for file in empty_files:
                try:
                    self.s3.delete_object(Bucket=self.bucket, Key=file['key'])
                    print(f"  Deleted: {file['key']}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  Error deleting {file['key']}: {e}")
            print(f"\nSuccessfully deleted {deleted_count} empty files")
        
        return len(empty_files)
    
    def deduplicate_by_sample_id(self) -> Tuple[int, int, int]:
        """
        Deduplicate files by sample_id, keeping the newest version of each sample.
        
        Returns:
            Tuple[int, int, int]: (files_before, files_after, files_to_delete)
        """
        print(f"\nDeduplicating files by sample_id in {self.bucket}/{self.prefix}")
        print("=" * 60)
        
        # Get all folders (prefixes) under the main prefix
        result = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix, Delimiter='/')
        folders = [cp['Prefix'] for cp in result.get('CommonPrefixes', [])]
        print(f"Found {len(folders)} folders in {self.bucket}/{self.prefix}")
        
        total_files_before = 0
        total_files_after = 0
        total_files_to_delete = 0
        
        for folder in folders:
            print(f"\nProcessing folder: {folder}")
            
            # List all JSONL files in the folder
            files = []
            for page in self.paginator.paginate(Bucket=self.bucket, Prefix=folder):
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
            
            # Group by model and sample_id
            model_to_samples = self._group_samples_by_model_and_id(files)
            
            # Process each model
            for model, samples in model_to_samples.items():
                total_files_after += self._process_model_samples(folder, model, samples)
                total_files_to_delete += len(files)  # All original files will be deleted
            
            print(f"  Folder summary: Files before: {len(files)}, Files after: {len(model_to_samples)}, Files to delete: {len(files)}")
        
        print(f"\n{'DRY RUN - ' if self.dry_run else ''}Summary:")
        print(f"Total files before: {total_files_before}")
        print(f"Total files after: {total_files_after}")
        print(f"Total files to delete: {total_files_to_delete}")
        
        if self.dry_run:
            print("\nThis was a dry run. No files were actually created or deleted.")
            print("To actually perform the cleanup, call with dry_run=False")
        else:
            print("\nCleanup complete!")
        
        return total_files_before, total_files_after, total_files_to_delete
    
    def _group_samples_by_model_and_id(self, files: List[Dict]) -> Dict:
        """Group samples by model and sample_id."""
        model_to_samples = {}
        
        for file in files:
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=file['key'])
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
        
        return model_to_samples
    
    def _process_model_samples(self, folder: str, model: str, samples: Dict) -> int:
        """Process samples for a specific model, creating deduplicated file."""
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
        
        if not self.dry_run:
            # Upload the consolidated file
            try:
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=new_key,
                    Body=consolidated_content.encode('utf-8'),
                    ContentType='application/json'
                )
                print(f"    Created consolidated file: {new_key}")
            except Exception as e:
                print(f"    Error creating consolidated file {new_key}: {e}")
                return 0
            
            # Delete all original files for this model
            files_to_delete = set()
            for sample_list in samples.values():
                for sample in sample_list:
                    files_to_delete.add(sample['file_key'])
            
            # Remove the newly created consolidated file from deletion list
            files_to_delete.discard(new_key)
            
            for file_key in files_to_delete:
                try:
                    self.s3.delete_object(Bucket=self.bucket, Key=file_key)
                    print(f"    Deleted original file: {file_key}")
                except Exception as e:
                    print(f"    Error deleting {file_key}: {e}")
        
        return 1  # One consolidated file per model
    
    def count_samples_by_model(self) -> Tuple[Dict, List]:
        """
        Count total samples for each model across all datasets.
        
        Returns:
            Tuple[Dict, List]: (model_counts, csv_data)
        """
        print("Counting samples for each model...")
        print("=" * 60)
        
        model_counts = defaultdict(int)
        dataset_counts = defaultdict(lambda: defaultdict(int))
        csv_data = []
        
        for page in self.paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if 'deduplicated' in key and key.endswith('.jsonl'):
                    # Extract model name from file path
                    # Example: processed-data-gemma/alibi-fraud-investigation/deduplicated_alibi-fraud-investigation_google_gemma-3-27b-it.jsonl
                    parts = key.split('/')
                    if len(parts) >= 3:
                        dataset = parts[1]  # e.g., "alibi-fraud-investigation"
                        filename = parts[2]  # e.g., "deduplicated_alibi-fraud-investigation_google_gemma-3-27b-it.jsonl"
                        
                        # Extract model name from filename
                        model = self._extract_model_from_filename(filename)
                        
                        # Count samples in this file
                        try:
                            response = self.s3.get_object(Bucket=self.bucket, Key=key)
                            content = response['Body'].read().decode('utf-8')
                            lines = content.strip().split('\n')
                            sample_count = len([line for line in lines if line.strip()])
                            
                            model_counts[model] += sample_count
                            dataset_counts[dataset][model] = sample_count
                            csv_data.append([dataset, model, sample_count])
                            
                            print(f"{dataset:40} | {model:35} | {sample_count:6} samples")
                            
                        except Exception as e:
                            print(f"Error reading {key}: {e}")
        
        return dict(model_counts), csv_data
    
    def _extract_model_from_filename(self, filename: str) -> str:
        """Extract model name from deduplicated filename."""
        if '_openrouter_' in filename:
            model = filename.split('_openrouter_')[1].replace('.jsonl', '')
        elif '_google_' in filename:
            model = filename.split('_google_')[1].replace('.jsonl', '')
        elif '_meta-llama_' in filename:
            model = filename.split('_meta-llama_')[1].replace('.jsonl', '')
        else:
            # Fallback: extract everything after the last underscore
            model = filename.split('_')[-1].replace('.jsonl', '')
        return model
    
    def print_summary_stats(self, model_counts: Dict, csv_data: List) -> None:
        """Print summary statistics and write CSV file."""
        print("\n" + "=" * 60)
        print("TOTAL SAMPLES BY MODEL:")
        print("=" * 60)
        
        total_samples = 0
        for model, count in sorted(model_counts.items()):
            print(f"{model:35} | {count:6} samples")
            total_samples += count
        
        print("=" * 60)
        print(f"{'TOTAL':35} | {total_samples:6} samples")
        
        # Write CSV file
        csv_filename = "sample_counts.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['task_name', 'model', 'num_samples'])
            for row in sorted(csv_data):
                writer.writerow(row)
        
        print(f"\nCSV file written to: {csv_filename}")
        print(f"Total rows: {len(csv_data)}")
    
    def run_full_pipeline(self) -> None:
        """Run the complete cleanup and counting pipeline."""
        print("Starting comprehensive cleanup and sample counting process...")
        print("=" * 80)
        
        # Step 1: Clean up empty files
        print("Step 1: Cleaning up empty files...")
        empty_files_count = self.cleanup_empty_files()
        
        # Step 2: Deduplicate by sample_id
        print("\nStep 2: Deduplicating by sample_id...")
        files_before, files_after, files_to_delete = self.deduplicate_by_sample_id()
        
        # Step 3: Count samples
        print("\nStep 3: Counting samples...")
        model_counts, csv_data = self.count_samples_by_model()
        self.print_summary_stats(model_counts, csv_data)
        
        # Final summary
        print(f"\n{'='*80}")
        print("✅ COMPREHENSIVE PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"📊 Empty files found: {empty_files_count}")
        print(f"📊 Files before deduplication: {files_before}")
        print(f"📊 Files after deduplication: {files_after}")
        print(f"📊 Files deleted: {files_to_delete}")
        print(f"📊 Total task-model combinations: {len(csv_data)}")
        print(f"📊 Total samples: {sum(model_counts.values())}")
        print(f"📊 Check sample_counts.csv for detailed results")
        print(f"{'='*80}")


def main():
    """Main function to run the cleanup and counting pipeline."""
    # Create the processor
    processor = S3CleanupAndCounter(
        bucket=BUCKET_NAME,
        prefix=PREFIX,
        dry_run=DRY_RUN
    )
    
    # Run the full pipeline
    processor.run_full_pipeline()


if __name__ == "__main__":
    main() 