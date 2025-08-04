import pickle
import os
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from inspect_ai.dataset import MemoryDataset, Sample


def get_cache_key(model: str, instruction: str, limit: int) -> str:
    """Generate a unique cache key based on parameters"""
    params = {
        'model': model,
        'instruction': instruction,
        'limit': limit
    }
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def get_cache_path(cache_key: str, cache_dir: str = "cache") -> Path:
    """Get the cache file path for a given cache key"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    return cache_path / f"dataset_{cache_key}.pkl"


def save_dataset_to_cache(dataset: MemoryDataset, cache_key: str, cache_dir: str = "cache") -> None:
    """Save a MemoryDataset to cache"""
    cache_path = get_cache_path(cache_key, cache_dir)

    # Convert dataset to serializable format
    serializable_data = {
        'samples': [
            {
                'input': sample.input,
                'target': getattr(sample, 'target', None),
                'id': getattr(sample, 'id', None),
                'metadata': sample.metadata
            }
            for sample in dataset.samples
        ]
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(serializable_data, f)

    print(f"Dataset cached to: {cache_path}")


def load_dataset_from_cache(cache_key: str, cache_dir: str = "cache") -> Optional[MemoryDataset]:
    """Load a MemoryDataset from cache if it exists"""
    cache_path = get_cache_path(cache_key, cache_dir)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)

        # Reconstruct samples
        samples = []
        for sample_data in data['samples']:
            sample = Sample(
                input=sample_data['input'],
                metadata=sample_data['metadata']
            )
            # Set optional attributes if they exist
            if sample_data.get('target') is not None:
                sample.target = sample_data['target']
            if sample_data.get('id') is not None:
                sample.id = sample_data['id']
            samples.append(sample)

        dataset = MemoryDataset(samples)
        print(f"Dataset loaded from cache: {cache_path}")
        return dataset

    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def create_or_load_signature_dataset(
        model: str,
        instruction: str,
        limit: int = 10,
        cache_dir: str = "cache",
        force_recreate: bool = False
) -> MemoryDataset:
    """
    Create or load a cached signature dataset.

    Args:
        model: Model name
        instruction: Instruction for generation
        limit: Number of samples
        cache_dir: Directory for cache files
        force_recreate: If True, ignore cache and recreate dataset

    Returns:
        MemoryDataset
    """
    from baseline.tasks.bytetrash.dataset import create_signature_dataset

    # Generate cache key
    cache_key = get_cache_key(model, instruction, limit)

    # Try to load from cache if not forcing recreation
    if not force_recreate:
        cached_dataset = load_dataset_from_cache(cache_key, cache_dir)
        if cached_dataset is not None:
            return cached_dataset

    # Create new dataset
    print(f"Creating new dataset (model={model}, limit={limit})...")
    dataset = create_signature_dataset(model, instruction, limit)

    # Save to cache
    save_dataset_to_cache(dataset, cache_key, cache_dir)

    return dataset


# Additional utility functions

def clear_cache(cache_dir: str = "cache") -> None:
    """Clear all cached datasets"""
    cache_path = Path(cache_dir)
    if cache_path.exists():
        for file in cache_path.glob("dataset_*.pkl"):
            file.unlink()
        print(f"Cache cleared: {cache_path}")


def list_cached_datasets(cache_dir: str = "cache") -> Dict[str, Dict[str, Any]]:
    """List all cached datasets with their metadata"""
    cache_path = Path(cache_dir)
    cached_datasets = {}

    if cache_path.exists():
        for file in cache_path.glob("dataset_*.pkl"):
            cache_key = file.stem.replace("dataset_", "")
            file_stats = file.stat()
            cached_datasets[cache_key] = {
                'file': str(file),
                'size_mb': file_stats.st_size / (1024 * 1024),
                'modified': file_stats.st_mtime
            }

    return cached_datasets