import json
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3DataDownloader:
    """Downloads sample data from S3 for a specific model with intelligent caching."""

    def __init__(self, bucket: str = None, prefix: str = None, cache_dir: str = '.data'):
        self.bucket = bucket or os.environ.get('S3_BUCKET', 'dipika-lie-detection-data')
        self.prefix = prefix or os.environ.get('S3_PREFIX', 'processed-data/')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.s3_client = boto3.client('s3', config=Config(max_pool_connections=50))

    def get_model_samples(self, model: str, max_workers: int = 20, batch_size: int = 50) -> List[Dict]:
        """Get model samples, using cache if available and fresh."""
        cache_file = self._get_cache_path(model)

        # Check if we should use cache
        if self._should_use_cache(model, cache_file):
            logger.info(f"Loading {model} samples from cache")
            samples = self._load_from_cache(cache_file)

            if samples:
                return samples

        # Download fresh data
        logger.info(f"Cache miss or stale for {model}, downloading fresh data")
        samples = self._download_model_samples(model, max_workers, batch_size)

        # Save to cache
        self._save_to_cache(cache_file, samples)

        return samples

    def _should_use_cache(self, model: str, cache_file: Path) -> bool:
        """Check if cache exists and is fresher than the most recent S3 upload."""
        if not cache_file.exists():
            logger.info(f"No cache file found for {model}")
            return False

        # Get cache modification time as UTC
        cache_mtime_utc = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
        logger.info(f"Cache last modified: {cache_mtime_utc}")

        # Get most recent S3 object modification time
        latest_s3_time = self._get_latest_s3_modification_time(model)

        if latest_s3_time is None:
            logger.warning(f"No S3 objects found for {model}")
            return False

        logger.info(f"Latest S3 modification: {latest_s3_time}")

        # Use cache if it's newer than the latest S3 object
        is_fresh = cache_mtime_utc > latest_s3_time
        logger.info(f"Cache is {'fresh' if is_fresh else 'stale'}")

        return is_fresh

    def _get_latest_s3_modification_time(self, model: str) -> Optional[datetime]:
        """Get the most recent modification time of any object with the model prefix."""
        provider, model_name = self._parse_model_name(model)
        clean_provider = self._clean_name(provider)
        clean_model = self._clean_name(model_name)
        prefix = f"{self.prefix}{clean_provider}/{clean_model}/"

        latest_time = None
        paginator = self.s3_client.get_paginator('list_objects_v2')

        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.json'):
                        obj_time = obj['LastModified']
                        if latest_time is None or obj_time > latest_time:
                            latest_time = obj_time
        except Exception as e:
            logger.error(f"Error checking S3 modification times: {e}")
            return None

        return latest_time

    def _get_cache_path(self, model: str) -> Path:
        """Get the cache file path for a model."""
        provider, model_name = self._parse_model_name(model)
        clean_provider = self._clean_name(provider)
        clean_model = self._clean_name(model_name)

        # Create subdirectory for provider if needed
        provider_dir = self.cache_dir / clean_provider
        provider_dir.mkdir(exist_ok=True)

        return provider_dir / f"{clean_model}_samples.pkl"

    def _load_from_cache(self, cache_file: Path) -> List[Dict]:
        """Load samples from cache file."""
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Loaded {len(data)} samples from cache")
                return data
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return []

    def _save_to_cache(self, cache_file: Path, samples: List[Dict]):
        """Save samples to cache file."""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)
            logger.info(f"Saved {len(samples)} samples to cache at {cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def clear_cache(self, model: Optional[str] = None):
        """Clear cache for a specific model or all models."""
        if model:
            cache_file = self._get_cache_path(model)
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Cleared cache for {model}")
            else:
                logger.info(f"No cache found for {model}")
        else:
            # Clear all cache files
            count = 0
            for cache_file in self.cache_dir.rglob("*.pkl"):
                cache_file.unlink()
                count += 1
            logger.info(f"Cleared {count} cache files")

    def get_cache_info(self) -> Dict[str, Dict]:
        """Get information about cached models."""
        cache_info = {}

        for cache_file in self.cache_dir.rglob("*.pkl"):
            try:
                # Extract model info from path
                provider = cache_file.parent.name
                model_name = cache_file.stem.replace("_samples", "")

                # Get file stats
                stats = cache_file.stat()

                cache_info[f"{provider}/{model_name}"] = {
                    "size_mb": stats.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "path": str(cache_file)
                }
            except Exception as e:
                logger.error(f"Error getting cache info for {cache_file}: {e}")

        return cache_info

    def _download_model_samples(self, model: str, max_workers: int = 20, batch_size: int = 50) -> List[Dict]:
        """Download all samples for a specific model from S3 in parallel with batching."""
        provider, model_name = self._parse_model_name(model)
        clean_provider = self._clean_name(provider)
        clean_model = self._clean_name(model_name)

        prefix = f"{self.prefix}{clean_provider}/{clean_model}/"

        logger.info(f"Downloading samples from s3://{self.bucket}/{prefix}")

        # Collect all keys to download
        keys_to_download = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.json'):
                    keys_to_download.append(obj['Key'])

        logger.info(f"Found {len(keys_to_download)} samples to download")

        # Download function for a batch of files
        def download_batch(keys_batch: List[str]) -> List[Optional[Dict]]:
            batch_samples = []
            for key in keys_batch:
                try:
                    response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
                    content = response['Body'].read().decode('utf-8')
                    sample = json.loads(content)
                    batch_samples.append(sample)
                    logger.debug(f"Downloaded: {key}")
                except Exception as e:
                    logger.error(f"Error downloading {key}: {e}")
                    batch_samples.append(None)
            return batch_samples

        # Create batches
        batches = [keys_to_download[i:i + batch_size]
                   for i in range(0, len(keys_to_download), batch_size)]

        # Download in parallel with progress bar
        samples = []
        failed_downloads = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch download tasks
            future_to_batch = {
                executor.submit(download_batch, batch): batch
                for batch in batches
            }

            # Process completed batches with progress bar
            with tqdm(total=len(keys_to_download),
                      desc=f"Downloading {model} samples",
                      unit="files") as pbar:

                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_samples = future.result()
                        for sample in batch_samples:
                            if sample is not None:
                                samples.append(sample)
                            else:
                                failed_downloads += 1

                        # Update progress bar
                        pbar.set_postfix_str(f"Failed: {failed_downloads}")
                        pbar.update(len(batch))

                    except Exception as e:
                        logger.error(f"Exception in batch download: {e}")
                        failed_downloads += len(batch)
                        pbar.update(len(batch))

        logger.info(f"Downloaded {len(samples)} samples successfully, {failed_downloads} failed")
        return samples

    def _parse_model_name(self, model: str) -> Tuple[str, str]:
        """Parse model name into provider and model."""
        if "/" in model:
            if "openrouter" in model:
                model = model.replace("openrouter/", "")
            parts = model.split("/", 1)
            return parts[0], parts[1]
        return "unknown", model

    def _clean_name(self, name: str) -> str:
        """Clean name for S3 paths."""
        return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(":", "_")


# Example usage
if __name__ == "__main__":
    downloader = S3DataDownloader()

    # Get samples (will use cache if fresh)
    samples = downloader.get_model_samples("openai/gpt-4")
    print(f"Loaded {len(samples)} samples")

    # Check cache info
    cache_info = downloader.get_cache_info()
    for model, info in cache_info.items():
        print(f"{model}: {info['size_mb']:.2f} MB, modified: {info['modified']}")

    # Force refresh by clearing cache
    # downloader.clear_cache("openai/gpt-4")
    # samples = downloader.get_model_samples("openai/gpt-4")