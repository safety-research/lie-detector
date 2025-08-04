from datasets import get_dataset_config_names

# Check available configurations
try:
    configs = get_dataset_config_names("Noddybear/lies")
    print(f"Available configurations: {len(configs)}")
    for config in sorted(configs):
        print(f"  - {config}")
except Exception as e:
    print(f"Error: {e}")

# Try loading with 'default' config to see what's there
from datasets import load_dataset

try:
    dataset = load_dataset("Noddybear/lies")
    print("\nDataset loaded with default config")
    print(f"Splits: {list(dataset.keys())}")
    if dataset:
        first_split = list(dataset.keys())[0]
        print(f"Features: {dataset[first_split].features}")
        print(f"Number of samples in {first_split}: {len(dataset[first_split])}")
except Exception as e:
    print(f"Error loading default: {e}")