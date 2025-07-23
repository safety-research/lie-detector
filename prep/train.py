import json
import hashlib
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import fcntl
from dataclasses import dataclass, asdict
from enum import Enum
import threading

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

class TrainingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingRun:
    """Represents a single training run."""
    fingerprint: str
    base_model: str
    dataset_path: str
    dataset_metadata: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    status: TrainingStatus
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    training_time_seconds: Optional[float] = None
    openai_job_id: Optional[str] = None
    wandb_run_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingRun':
        """Create from dictionary."""
        data['status'] = TrainingStatus(data['status'])
        return cls(**data)


class TrainingQueue:
    """Manages idempotent training queue with persistent state."""

    def __init__(self, base_dir: str = ".data", registry_filename: str = "training_registry.json"):
        self.base_dir = Path(base_dir)
        self.registry_path = self.base_dir / registry_filename
        self.lock_dir = self.base_dir / "locks"
        self.lock_dir.mkdir(exist_ok=True, parents=True)

        # Load or create registry
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, TrainingRun]:
        """Load training registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    return {k: TrainingRun.from_dict(v) for k, v in data.items()}
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                return {}
        return {}

    def _save_registry(self):
        """Save training registry to disk."""
        try:
            # Ensure directory exists
            self.registry_path.parent.mkdir(exist_ok=True, parents=True)

            # Convert to JSON-serializable format
            data = {k: v.to_dict() for k, v in self.registry.items()}

            # Atomic write with temporary file
            temp_path = self.registry_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.registry_path)

        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            raise

    def _generate_fingerprint(self, dataset_path: str, base_model: str,
                              hyperparameters: Dict[str, Any],
                              size: Optional[int] = None) -> str:
        """Generate deterministic fingerprint for training configuration."""
        # Load dataset metadata
        dataset_path = Path(dataset_path)
        metadata_path = dataset_path / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                dataset_metadata = json.load(f)
        else:
            dataset_metadata = {}

        # Calculate dataset content hash (or use size-specific hash)
        if size:
            # For subsampled data, include size in the hash
            dataset_hash = f"sampled_{size}_{self._calculate_dataset_hash(dataset_path)}"
        else:
            dataset_hash = self._calculate_dataset_hash(dataset_path)

        # Create fingerprint data
        fingerprint_data = {
            "base_model": base_model,
            "dataset_hash": dataset_hash,
            "dataset_metadata": {
                "aggregation": dataset_metadata.get("aggregation"),
                "fold_name": dataset_metadata.get("fold_name"),
                "balance_strategy": dataset_metadata.get("balance_strategy"),
                "validation_split": dataset_metadata.get("validation_split"),
                "seed": dataset_metadata.get("seed")
            },
            "hyperparameters": hyperparameters,
            "sample_size": size  # Include size in fingerprint
        }

        # Create stable JSON representation
        fingerprint_json = json.dumps(fingerprint_data, sort_keys=True)

        # Generate hash
        return hashlib.sha256(fingerprint_json.encode()).hexdigest()[:16]

    def _calculate_dataset_hash(self, dataset_path: Path) -> str:
        """Calculate hash of dataset files."""
        hasher = hashlib.sha256()

        # Hash train and val files in consistent order
        for filename in ["train.jsonl", "val.jsonl"]:
            filepath = dataset_path / filename
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    # Read in chunks for large files
                    while chunk := f.read(8192):
                        hasher.update(chunk)

        return hasher.hexdigest()[:16]

    def _acquire_lock(self, fingerprint: str, timeout: int = 30) -> Optional['FileLock']:
        """Acquire distributed lock for training run."""
        lock_path = self.lock_dir / f"{fingerprint}.lock"

        try:
            lock_file = open(lock_path, 'w')
            # Try to acquire exclusive lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_file.write(f"{os.getpid()}\n{datetime.now().isoformat()}")
            lock_file.flush()
            return lock_file
        except IOError:
            # Lock is held by another process
            return None

    def _release_lock(self, lock_file):
        """Release training lock."""
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                # Remove lock file
                Path(lock_file.name).unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Error releasing lock: {e}")

    def _is_stale(self, run: TrainingRun, stale_hours: int = 24) -> bool:
        """Check if a running job is stale."""
        if run.started_at:
            started = datetime.fromisoformat(run.started_at)
            elapsed = datetime.now() - started
            return elapsed.total_seconds() > (stale_hours * 3600)
        return True

    def train(self, dataset_path: str, base_model: str,
              hyperparameters: Optional[Dict[str, Any]] = None,
              size: Optional[int] = None,
              force: bool = False) -> Dict[str, Any]:
        """
        Queue or return existing training run.

        Args:
            dataset_path: Path to dataset directory
            base_model: Base model identifier (e.g., "gpt-4o-2024-11-20")
            hyperparameters: Training hyperparameters
            size: Optional size to downsample training data (maintains balance)
            force: Force retraining even if already completed

        Returns:
            Dictionary with training status and results
        """
        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                "n_epochs": 3,
                "learning_rate_multiplier": 1.0,
            }

        # Generate fingerprint (now includes size)
        fingerprint = self._generate_fingerprint(dataset_path, base_model, hyperparameters, size)

        # Load dataset metadata
        dataset_path = Path(dataset_path)
        metadata_path = dataset_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                dataset_metadata = json.load(f)
        else:
            dataset_metadata = {}

        # Add size to metadata if specified
        if size:
            dataset_metadata["sample_size"] = size

        # Check existing run
        existing_run = self.registry.get(fingerprint)

        if existing_run and not force:
            if existing_run.status == TrainingStatus.COMPLETED:
                logger.info(f"Training already completed for fingerprint {fingerprint}")
                return {
                    "status": "already_trained",
                    "fingerprint": fingerprint,
                    "model_id": existing_run.model_id,
                    "metrics": existing_run.metrics,
                    "completed_at": existing_run.completed_at,
                    "sample_size": size,
                    "openai_job_id": existing_run.openai_job_id,
                    "wandb_run_id": existing_run.wandb_run_id,
                    "message": f"Model already trained on {existing_run.completed_at}"
                }

            elif existing_run.status == TrainingStatus.RUNNING:
                if self._is_stale(existing_run):
                    logger.warning(f"Found stale training run {fingerprint}, restarting...")
                    return self._start_training(fingerprint, dataset_path, base_model,
                                                dataset_metadata, hyperparameters, size)
                else:
                    logger.info(f"Training already in progress for fingerprint {fingerprint}")
                    return {
                        "status": "in_progress",
                        "fingerprint": fingerprint,
                        "started_at": existing_run.started_at,
                        "sample_size": size,
                        "openai_job_id": existing_run.openai_job_id,
                        "message": "Training already in progress"
                    }

            elif existing_run.status == TrainingStatus.FAILED:
                logger.info(f"Retrying failed training run {fingerprint}")
                return self._start_training(fingerprint, dataset_path, base_model,
                                            dataset_metadata, hyperparameters, size)

        # Start new training
        return self._start_training(fingerprint, dataset_path, base_model,
                                    dataset_metadata, hyperparameters, size)

    def _start_training(self, fingerprint: str, dataset_path: Path, base_model: str,
                        dataset_metadata: Dict[str, Any], hyperparameters: Dict[str, Any],
                        size: Optional[int] = None) -> Dict[str, Any]:
        """Start a new training run."""
        # Try to acquire lock
        lock = self._acquire_lock(fingerprint)
        if not lock:
            return {
                "status": "locked",
                "fingerprint": fingerprint,
                "message": "Another process is starting this training"
            }

        try:
            # Create training run record
            run = TrainingRun(
                fingerprint=fingerprint,
                base_model=base_model,
                dataset_path=str(dataset_path),
                dataset_metadata=dataset_metadata,
                hyperparameters=hyperparameters,
                status=TrainingStatus.RUNNING,
                started_at=datetime.now().isoformat()
            )

            # Save to registry
            self.registry[fingerprint] = run
            self._save_registry()

            # Create training run directory
            run_dir = dataset_path / "training_runs" / fingerprint
            run_dir.mkdir(exist_ok=True, parents=True)

            # If size is specified, create subsampled data
            if size:
                sampled_dir = run_dir / "sampled_data"
                sampled_dir.mkdir(exist_ok=True)

                # Subsample the data while maintaining balance
                self._create_balanced_subsample(
                    dataset_path, sampled_dir, size, dataset_metadata.get("seed", 42)
                )

                # Update dataset path for training
                effective_dataset_path = sampled_dir
            else:
                effective_dataset_path = dataset_path

            # Save training configuration
            config_path = run_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    "fingerprint": fingerprint,
                    "base_model": base_model,
                    "dataset_path": str(dataset_path),
                    "effective_dataset_path": str(effective_dataset_path),
                    "sample_size": size,
                    "hyperparameters": hyperparameters,
                    "started_at": run.started_at
                }, f, indent=2)

            logger.info(f"Started training run {fingerprint}")
            if size:
                logger.info(f"Using balanced subsample of size {size}")

            return {
                "status": "started",
                "fingerprint": fingerprint,
                "run_dir": str(run_dir),
                "config_path": str(config_path),
                "dataset_path": str(effective_dataset_path),
                "sample_size": size,
                "message": "Training started successfully"
            }

        finally:
            self._release_lock(lock)

    def _create_balanced_subsample(self, source_dir: Path, target_dir: Path,
                                   size: int, seed: int = 42):
        """Create a balanced subsample of the training data."""
        import random
        random.seed(seed)

        # Load training data
        train_path = source_dir / "train.jsonl"
        val_path = source_dir / "val.jsonl"

        with open(train_path, 'r') as f:
            train_data = [json.loads(line) for line in f]

        with open(val_path, 'r') as f:
            val_data = [json.loads(line) for line in f]

        # Subsample training data while maintaining balance
        train_sampled = self._balanced_sample(train_data, size)

        # Save subsampled training data
        train_sampled_path = target_dir / "train.jsonl"
        with open(train_sampled_path, 'w') as f:
            for item in train_sampled:
                f.write(json.dumps(item) + '\n')

        # Copy validation data WITHOUT subsampling
        val_sampled_path = target_dir / "val.jsonl"
        with open(val_sampled_path, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')

        # Copy test.jsonl if it exists (don't subsample test data)
        test_path = source_dir / "test.jsonl"
        if test_path.exists():
            test_sampled_path = target_dir / "test.jsonl"
            with open(test_path, 'r') as src, open(test_sampled_path, 'w') as dst:
                dst.write(src.read())

        # Copy metadata.json
        metadata_path = source_dir / "metadata.json"
        if metadata_path.exists():
            metadata_sampled_path = target_dir / "metadata.json"
            with open(metadata_path, 'r') as src, open(metadata_sampled_path, 'w') as dst:
                metadata = json.load(src)
                metadata["sample_size"] = size
                json.dump(metadata, dst, indent=2)

        # Save subsample metadata
        subsample_metadata_path = target_dir / "subsample_metadata.json"
        with open(subsample_metadata_path, 'w') as f:
            json.dump({
                "original_train_size": len(train_data),
                "original_val_size": len(val_data),
                "sampled_train_size": len(train_sampled),
                "sampled_val_size": len(val_data),  # Same as original
                "requested_size": size,
                "seed": seed,
                "created_at": datetime.now().isoformat()
            }, f, indent=2)

        logger.info(f"Created balanced subsample: {len(train_sampled)} train, {len(val_data)} val (unchanged)")

    def _balanced_sample(self, data: List[Dict], size: int) -> List[Dict]:
        """Sample data while maintaining label balance."""
        import random

        # Separate by label (assuming binary classification)
        # Extract label from the messages - last assistant message
        positive = []
        negative = []

        for item in data:
            messages = item.get("messages", [])
            if messages and messages[-1].get("role") == "assistant":
                response = messages[-1].get("content", "").strip().lower()
                if response == "yes.":
                    positive.append(item)
                else:
                    negative.append(item)

        # Calculate balanced sizes
        target_per_class = size // 2

        # Sample from each class
        sampled_positive = random.sample(positive, min(target_per_class, len(positive)))
        sampled_negative = random.sample(negative, min(target_per_class, len(negative)))

        # Combine and shuffle
        sampled = sampled_positive + sampled_negative
        random.shuffle(sampled)

        return sampled

    def update_openai_job_id(self, fingerprint: str, job_id: str):
        """Update the OpenAI job ID for a training run."""
        if fingerprint in self.registry:
            self.registry[fingerprint].openai_job_id = job_id
            self._save_registry()

    def update_wandb_run_id(self, fingerprint: str, wandb_run_id: str):
        """Update the W&B run ID for a training run."""
        if fingerprint in self.registry:
            self.registry[fingerprint].wandb_run_id = wandb_run_id
            self._save_registry()

    def complete_training(self, fingerprint: str, model_id: str,
                          metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Mark training as completed with results."""
        if fingerprint not in self.registry:
            return {
                "status": "error",
                "message": f"Training run {fingerprint} not found"
            }

        run = self.registry[fingerprint]
        run.status = TrainingStatus.COMPLETED
        run.model_id = model_id
        run.metrics = metrics
        run.completed_at = datetime.now().isoformat()

        if run.started_at:
            started = datetime.fromisoformat(run.started_at)
            elapsed = datetime.now() - started
            run.training_time_seconds = elapsed.total_seconds()

        self._save_registry()

        # Save results to run directory
        dataset_path = Path(run.dataset_path)
        run_dir = dataset_path / "training_runs" / fingerprint

        # Save model info
        model_info_path = run_dir / "model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump({
                "model_id": model_id,
                "base_model": run.base_model,
                "completed_at": run.completed_at,
                "training_time_seconds": run.training_time_seconds
            }, f, indent=2)

        # Save metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Completed training run {fingerprint}: {model_id}")

        return {
            "status": "completed",
            "fingerprint": fingerprint,
            "model_id": model_id,
            "metrics": metrics
        }

    def fail_training(self, fingerprint: str, error_message: str) -> Dict[str, Any]:
        """Mark training as failed."""
        if fingerprint not in self.registry:
            return {
                "status": "error",
                "message": f"Training run {fingerprint} not found"
            }

        run = self.registry[fingerprint]
        run.status = TrainingStatus.FAILED
        run.error_message = error_message
        run.completed_at = datetime.now().isoformat()

        self._save_registry()

        logger.error(f"Failed training run {fingerprint}: {error_message}")

        return {
            "status": "failed",
            "fingerprint": fingerprint,
            "error_message": error_message
        }

    def get_status(self, fingerprint: Optional[str] = None) -> Dict[str, Any]:
        """Get status of training runs."""
        if fingerprint:
            run = self.registry.get(fingerprint)
            if run:
                return {
                    "fingerprint": fingerprint,
                    "status": run.status.value,
                    "model_id": run.model_id,
                    "metrics": run.metrics,
                    "started_at": run.started_at,
                    "completed_at": run.completed_at,
                    "openai_job_id": run.openai_job_id,
                    "wandb_run_id": run.wandb_run_id,
                    "dataset_path": run.dataset_path,
                    "error_message": run.error_message
                }
            else:
                return {
                    "fingerprint": fingerprint,
                    "status": "not_found"
                }

        # Return summary of all runs
        summary = {
            "total": len(self.registry),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "runs": []
        }

        for run in self.registry.values():
            summary[run.status.value] += 1
            summary["runs"].append({
                "fingerprint": run.fingerprint,
                "status": run.status.value,
                "dataset_path": run.dataset_path,
                "model_id": run.model_id,
                "started_at": run.started_at,
                "openai_job_id": run.openai_job_id
            })

        return summary

    def retry_failed(self) -> List[Dict[str, Any]]:
        """Retry all failed training runs."""
        results = []
        for fingerprint, run in self.registry.items():
            if run.status == TrainingStatus.FAILED:
                result = self.train(
                    run.dataset_path,
                    run.base_model,
                    run.hyperparameters,
                    force=True
                )
                results.append(result)
        return results

    def train_all_folds(self, model_dir: str, base_model: str,
                        hyperparameters: Optional[Dict[str, Any]] = None,
                        size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Train on all folds in a model directory."""
        model_path = Path(model_dir)
        results = []

        # Find all fold directories
        for fold_dir in model_path.iterdir():
            if fold_dir.is_dir() and (fold_dir / "train.jsonl").exists():
                logger.info(f"Queueing training for fold: {fold_dir.name}")
                result = self.train(
                    str(fold_dir),
                    base_model,
                    hyperparameters,
                    size=size
                )
                results.append({
                    "fold": fold_dir.name,
                    **result
                })

        return results


def start_openai_training(queue_result: Dict[str, Any], args) -> Dict[str, Any]:
    """Start the actual OpenAI training job."""
    try:
        from trainers.oai import OpenAITrainer

        # Initialize trainer
        trainer = OpenAITrainer(
            enable_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity
        )

        # Start training with the effective dataset path
        result = trainer.train(
            dataset_path=queue_result["dataset_path"],
            base_model=args.model,
            n_epochs=args.n_epochs,
            learning_rate_multiplier=args.learning_rate_multiplier,
            batch_size=args.batch_size,
            size=None,  # Already handled by queue
            force=False  # Already handled by queue
        )

        return result
    except ImportError:
        logger.error("Could not import OpenAITrainer. Make sure prep.trainers.oai is in your Python path.")
        raise


def interactive_monitor(fingerprints: List[str], queue: TrainingQueue):
    """Monitor training runs interactively."""
    print("\n" + "=" * 70)
    print("INTERACTIVE TRAINING MONITOR")
    print("=" * 70)
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            # Clear screen (works on Unix-like systems)
            os.system('clear' if os.name == 'posix' else 'cls')

            print(f"Monitoring {len(fingerprints)} training run(s)")
            print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 70)

            all_completed = True

            for i, fingerprint in enumerate(fingerprints):
                status = queue.get_status(fingerprint)

                print(f"\n[{i + 1}] Fingerprint: {fingerprint}")
                print(f"    Status: {status['status'].upper()}")

                if status['status'] == 'running':
                    all_completed = False
                    if status.get('started_at'):
                        started = datetime.fromisoformat(status['started_at'])
                        elapsed = datetime.now() - started
                        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
                        print(f"    Running for: {elapsed_str}")

                    if status.get('openai_job_id'):
                        print(f"    OpenAI Job: {status['openai_job_id']}")
                        print(f"    Track at: https://platform.openai.com/finetune/{status['openai_job_id']}")

                    if status.get('wandb_run_id'):
                        print(f"    W&B Run: {status['wandb_run_id']}")

                elif status['status'] == 'completed':
                    print(f"    Model ID: {status.get('model_id', 'N/A')}")
                    if status.get('metrics'):
                        print(f"    Final Metrics:")
                        for key, value in status['metrics'].items():
                            if 'final_' in key:
                                print(f"      - {key}: {value:.4f}" if isinstance(value,
                                                                                  float) else f"      - {key}: {value}")

                elif status['status'] == 'failed':
                    all_completed = False
                    print(f"    Error: {status.get('error_message', 'Unknown error')}")

                elif status['status'] == 'not_found':
                    print(f"    ERROR: Training run not found in registry")

            if all_completed:
                print("\n" + "=" * 70)
                print("All training runs completed!")
                print("=" * 70)
                break

            # Wait before next update
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


# CLI Interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train lie detector models with OpenAI')
    parser.add_argument('--dataset', type=str, help='Path to dataset or model directory')
    parser.add_argument('--model', type=str, help='Base model identifier (e.g., gpt-4o-2024-11-20)')
    parser.add_argument('--all-folds', action='store_true', help='Train all folds in directory')
    parser.add_argument('--status', action='store_true', help='Show queue status')
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed runs')
    parser.add_argument('--force', action='store_true', help='Force retraining')
    parser.add_argument('--n-epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning-rate-multiplier', type=float, default=1.0, help='Learning rate multiplier')
    parser.add_argument('--batch-size', type=int, help='Batch size (None for auto)')
    parser.add_argument('--size', type=int, help='Subsample size for training data (maintains balance)')
    parser.add_argument('--interactive', action='store_true', help='Monitor training interactively')
    parser.add_argument('--monitor', type=str, help='Monitor specific fingerprint(s), comma-separated')

    # W&B arguments
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--wandb-project', type=str, help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, help='W&B entity/team name')

    args = parser.parse_args()

    # Initialize queue
    queue = TrainingQueue()

    if args.monitor:
        # Monitor specific fingerprints
        fingerprints = [f.strip() for f in args.monitor.split(',')]
        interactive_monitor(fingerprints, queue)

    elif args.status:
        # Show status
        status = queue.get_status()
        print(json.dumps(status, indent=2))

    elif args.retry_failed:
        # Retry failed runs
        results = queue.retry_failed()
        print(f"Retrying {len(results)} failed runs")
        for result in results:
            print(json.dumps(result, indent=2))

    elif args.dataset and args.model:
        # Set up hyperparameters
        hyperparameters = {
            "n_epochs": args.n_epochs,
            "learning_rate_multiplier": args.learning_rate_multiplier,
        }
        if args.batch_size:
            hyperparameters["batch_size"] = args.batch_size

        fingerprints_to_monitor = []

        if args.all_folds:
            # Train all folds
            results = queue.train_all_folds(args.dataset, args.model, hyperparameters, size=args.size)
            print(f"Processing {len(results)} training runs")

            for result in results:
                print(f"\nFold: {result['fold']}")

                if result["status"] == "started":
                    # Start OpenAI training
                    print(f"Starting OpenAI training for fingerprint {result['fingerprint']}...")
                    openai_result = start_openai_training(result, args)
                    print(f"OpenAI training initiated: {openai_result.get('status')}")
                    fingerprints_to_monitor.append(result['fingerprint'])
                else:
                    print(f"Status: {result['status']}")
                    if result.get("model_id"):
                        print(f"Model ID: {result['model_id']}")
        else:
            # Train single dataset
            result = queue.train(args.dataset, args.model, hyperparameters, size=args.size, force=args.force)
            print(json.dumps(result, indent=2))

            if result["status"] == "started":
                # Start OpenAI training
                print(f"\nStarting OpenAI training for fingerprint {result['fingerprint']}...")
                openai_result = start_openai_training(result, args)
                print(f"OpenAI training initiated: {openai_result.get('status')}")
                fingerprints_to_monitor.append(result['fingerprint'])

        # If interactive mode and we started training, monitor it
        if args.interactive and fingerprints_to_monitor:
            interactive_monitor(fingerprints_to_monitor, queue)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()