#!/usr/bin/env python3
"""
Script to count samples in organized evaluation folders.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def count_samples_in_jsonl_file(file_path: Path) -> int:
    """Count the number of samples in a JSONL file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Count non-empty lines that contain valid JSON
            sample_count = 0
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        sample_count += 1
                    except json.JSONDecodeError:
                        continue
            return sample_count
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def analyze_organized_folder(folder_path: Path) -> dict:
    """Analyze a single organized evaluation folder."""
    if not folder_path.exists():
        return {"error": f"Folder {folder_path} does not exist"}
    
    results = {
        "folder": folder_path.name,
        "total_files": 0,
        "total_samples": 0,
        "files_by_task": {},
        "task_summary": {}
    }
    
    # Find all JSONL files
    jsonl_files = list(folder_path.glob("*.jsonl"))
    results["total_files"] = len(jsonl_files)
    
    for jsonl_file in jsonl_files:
        # Extract task name from filename
        filename = jsonl_file.name
        if filename.startswith("deduplicated_"):
            # Remove "deduplicated_" prefix and model suffix
            task_part = filename[13:]  # Remove "deduplicated_"
            # Find the last underscore followed by model name
            parts = task_part.split("_")
            if len(parts) >= 2:
                # Reconstruct task name (everything except the last part which is the model)
                task_name = "_".join(parts[:-1])
                model_part = parts[-1]
            else:
                task_name = task_part
                model_part = "unknown"
        else:
            task_name = filename.replace(".jsonl", "")
            model_part = "unknown"
        
        # Count samples in this file
        sample_count = count_samples_in_jsonl_file(jsonl_file)
        
        # Store results
        results["files_by_task"][filename] = {
            "task": task_name,
            "model": model_part,
            "samples": sample_count,
            "file_size_mb": jsonl_file.stat().st_size / (1024 * 1024)
        }
        
        results["total_samples"] += sample_count
        
        # Update task summary
        if task_name not in results["task_summary"]:
            results["task_summary"][task_name] = {
                "files": 0,
                "samples": 0,
                "models": set()
            }
        
        results["task_summary"][task_name]["files"] += 1
        results["task_summary"][task_name]["samples"] += sample_count
        results["task_summary"][task_name]["models"].add(model_part)
    
    # Convert sets to lists for JSON serialization
    for task_info in results["task_summary"].values():
        task_info["models"] = list(task_info["models"])
    
    return results

def print_analysis_results(results: dict):
    """Print the analysis results in a formatted way."""
    if "error" in results:
        print(f"❌ {results['error']}")
        return
    
    print(f"\n{'='*80}")
    print(f"📁 FOLDER: {results['folder']}")
    print(f"{'='*80}")
    print(f"📊 Total files: {results['total_files']}")
    print(f"📊 Total samples: {results['total_samples']:,}")
    
    print(f"\n📋 TASK BREAKDOWN:")
    print(f"{'Task Name':<50} {'Files':<6} {'Samples':<8} {'Models':<20}")
    print(f"{'-'*50} {'-'*6} {'-'*8} {'-'*20}")
    
    for task_name, task_info in sorted(results["task_summary"].items()):
        models_str = ", ".join(task_info["models"])
        print(f"{task_name:<50} {task_info['files']:<6} {task_info['samples']:<8} {models_str:<20}")
    
    print(f"\n📄 DETAILED FILE BREAKDOWN:")
    print(f"{'Filename':<60} {'Task':<30} {'Samples':<8} {'Size (MB)':<10}")
    print(f"{'-'*60} {'-'*30} {'-'*8} {'-'*10}")
    
    for filename, file_info in sorted(results["files_by_task"].items()):
        print(f"{filename:<60} {file_info['task']:<30} {file_info['samples']:<8} {file_info['file_size_mb']:<10.2f}")

def main():
    """Main function to analyze all organized evaluation folders."""
    base_dir = Path("organized_evaluation_20250721_215718")
    
    if not base_dir.exists():
        print(f"❌ Base directory {base_dir} does not exist!")
        return
    
    # Find all model folders
    model_folders = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("openrouter_google_gemma-"):
            model_folders.append(item)
    
    if not model_folders:
        print(f"❌ No model folders found in {base_dir}")
        return
    
    print(f"🔍 Analyzing {len(model_folders)} model folders...")
    
    all_results = {}
    total_samples_across_all = 0
    
    for folder in sorted(model_folders):
        results = analyze_organized_folder(folder)
        all_results[folder.name] = results
        print_analysis_results(results)
        
        if "error" not in results:
            total_samples_across_all += results["total_samples"]
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY ACROSS ALL FOLDERS")
    print(f"{'='*80}")
    print(f"Total folders analyzed: {len(model_folders)}")
    print(f"Total samples across all folders: {total_samples_across_all:,}")
    
    print(f"\n📈 SAMPLES BY MODEL:")
    for folder_name, results in sorted(all_results.items()):
        if "error" not in results:
            print(f"  {folder_name}: {results['total_samples']:,} samples ({results['total_files']} files)")
    
    # Save detailed results to JSON
    output_file = "organized_samples_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main() 