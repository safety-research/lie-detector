from datasets import load_dataset
import pandas as pd
import os
import argparse
import requests
import json

def download_sycophancy_eval_datasets(data_dir: str, test_data_dir: str):
    """
    Download sycophancy-eval datasets from meg-tong/sycophancy-eval repository.
    """
    base_url = "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets"
    datasets = ["feedback.jsonl", "are_you_sure.jsonl", "answer.jsonl"]
    
    for dataset_name in datasets:
        print(f"Downloading {dataset_name}...")
        
        # Download the JSONL file
        url = f"{base_url}/{dataset_name}"
        response = requests.get(url)
        
        if response.status_code == 200:
            # Save full dataset
            full_path = os.path.join(data_dir, f"sycophancy_{dataset_name.replace('.jsonl', '')}.jsonl")
            with open(full_path, 'w') as f:
                f.write(response.text)
            
            # Create test dataset (first 5 samples)
            lines = response.text.strip().split('\n')
            test_lines = lines[:5] if len(lines) >= 5 else lines
            
            test_path = os.path.join(test_data_dir, f"sycophancy_{dataset_name.replace('.jsonl', '')}.jsonl")
            with open(test_path, 'w') as f:
                f.write('\n'.join(test_lines))
            
            print(f"Downloaded {len(lines)} samples for sycophancy_{dataset_name.replace('.jsonl', '')}")
            print(f"Created test dataset with {len(test_lines)} samples for sycophancy_{dataset_name.replace('.jsonl', '')}")
        else:
            print(f"Failed to download {dataset_name}: {response.status_code}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download MASK and sycophancy-eval datasets')
    parser.add_argument('--data-dir', type=str, default='data', 
                       help='Data directory to save datasets (default: data)')
    parser.add_argument('--test-data-dir', type=str, default='test_data',
                       help='Test data directory to save small datasets (default: test_data)')
    args = parser.parse_args()

    # Create data directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.test_data_dir, exist_ok=True)

    # Download MASK dataset
    print("Downloading MASK dataset...")
    for split in ["continuations", "disinformation", "doubling_down_known_facts", "known_facts", "provided_facts", "statistics"]:
        ds = load_dataset("cais/mask", split)

        df = pd.DataFrame(ds['test'])
        df.to_csv(f"{args.data_dir}/{split}.csv", index=False)
        print(f"Downloaded {len(df)} samples for {split}")
        
        # Save first 5 samples to test_data directory (use iloc to ensure exactly 5 rows)
        df_test = df.iloc[:5]
        df_test.to_csv(f"{args.test_data_dir}/{split}.csv", index=False)
        print(f"Created test dataset with {len(df_test)} samples for {split}")

    # Download sycophancy-eval datasets
    print("\nDownloading sycophancy-eval datasets...")
    download_sycophancy_eval_datasets(args.data_dir, args.test_data_dir)

    print("\nDownload complete!")
    print(f"Full datasets saved to: {args.data_dir}/")
    print(f"Test datasets (first 5 samples) saved to: {args.test_data_dir}/")

if __name__ == "__main__":
    main()