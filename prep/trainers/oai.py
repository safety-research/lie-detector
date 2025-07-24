import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from datetime import datetime
import os

load_dotenv()
try:
    from train import TrainingQueue
except ImportError:
    from prep.train import TrainingQueue

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Install with: pip install wandb")

logger = logging.getLogger(__name__)


class OpenAITrainer:
    """Wrapper for OpenAI fine-tuning with TrainingQueue and W&B integration."""

    def __init__(self, api_key: Optional[str] = None, queue: Optional[TrainingQueue] = None,
                 wandb_project: Optional[str] = None, wandb_entity: Optional[str] = None,
                 enable_wandb: bool = True):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.queue = queue or TrainingQueue()
        self.wandb_project = wandb_project or os.environ.get("WANDB_PROJECT", "lie-detector-training")
        self.wandb_entity = wandb_entity or os.environ.get("WANDB_ENTITY")
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.client = OpenAI(api_key=self.api_key)

    def _init_wandb(self, fingerprint: str, config: Dict[str, Any],
                    dataset_metadata: Dict[str, Any]) -> Optional[Any]:
        """Initialize W&B run for this training."""
        if not self.enable_wandb:
            return None

        try:
            # Create a unique run name
            run_name = f"{config['base_model'].replace('/', '-')}-{fingerprint}"

            # Prepare W&B config
            wandb_config = {
                "fingerprint": fingerprint,
                "base_model": config['base_model'],
                "dataset_path": config['dataset_path'],
                **config['hyperparameters'],
                "size": config.get('size'),
                "dataset_metadata": dataset_metadata
            }

            # Initialize W&B run
            run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=run_name,
                config=wandb_config,
                tags=[
                    f"model:{config['base_model']}",
                    f"fold:{dataset_metadata.get('fold_name', 'single')}",
                    f"aggregation:{dataset_metadata.get('aggregation', 'none')}",
                    f"balance:{dataset_metadata.get('balance_strategy', 'none')}"
                ],
                reinit=True
            )

            # Log dataset statistics
            wandb.log({
                "dataset/train_size": dataset_metadata.get('train_size', 0),
                "dataset/val_size": dataset_metadata.get('val_size', 0),
                "dataset/test_size": dataset_metadata.get('test_size', 0),
                "dataset/train_lies": dataset_metadata.get('train_lies', 0),
                "dataset/train_truths": dataset_metadata.get('train_truths', 0),
                "dataset/val_lies": dataset_metadata.get('val_lies', 0),
                "dataset/val_truths": dataset_metadata.get('val_truths', 0),
            })

            logger.info(f"Initialized W&B run: {run.name}")
            return run

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            return None

    def _prepare_training_files(self, dataset_path: Path, size: Optional[int] = None) -> tuple[Path, Path]:
        """
        Prepare training files by removing the 'meta' field from each line.

        Args:
            dataset_path: Path to the dataset directory
            size: Optional size parameter (for logging)

        Returns:
            Tuple of (cleaned_train_path, cleaned_val_path)
        """
        # Create a temporary directory for cleaned files
        temp_dir = dataset_path / "temp_cleaned"
        temp_dir.mkdir(exist_ok=True)

        # Process train.jsonl
        train_input_path = dataset_path / "train.jsonl"
        train_output_path = temp_dir / "train_cleaned.jsonl"

        logger.info(f"Removing 'meta' field from training data...")
        with open(train_input_path, 'r') as infile, open(train_output_path, 'w') as outfile:
            for line_num, line in enumerate(infile, 1):
                try:
                    data = json.loads(line)
                    # Remove the 'meta' field if it exists
                    if 'meta' in data:
                        del data['meta']
                    outfile.write(json.dumps(data) + '\n')
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num} in train.jsonl: {e}")
                    raise

        # Process val.jsonl
        val_input_path = dataset_path / "val.jsonl"
        val_output_path = temp_dir / "val_cleaned.jsonl"

        with open(val_input_path, 'r') as infile, open(val_output_path, 'w') as outfile:
            for line_num, line in enumerate(infile, 1):
                try:
                    data = json.loads(line)
                    # Remove the 'meta' field if it exists
                    if 'meta' in data:
                        del data['meta']
                    outfile.write(json.dumps(data) + '\n')
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num} in val.jsonl: {e}")
                    raise

        logger.info(f"Successfully cleaned training files (removed 'meta' field)")
        return train_output_path, val_output_path

    def train(self, dataset_path: str, base_model: str = "gpt-4o-2024-08-06",
              n_epochs: int = 3, learning_rate_multiplier: float = 1.0,
              batch_size: Optional[int] = None, size: Optional[int] = None,
              force: bool = False) -> Dict[str, Any]:
        """
        Train a model using OpenAI's fine-tuning API with queue management and W&B logging.

        Args:
            dataset_path: Path to dataset directory containing train.jsonl and val.jsonl
            base_model: OpenAI model to fine-tune
            n_epochs: Number of training epochs
            learning_rate_multiplier: Learning rate multiplier
            batch_size: Batch size (None for auto)
            size: Optional size to subsample training data (maintains balance)
            force: Force retraining even if already completed

        Returns:
            Dictionary with training results
        """
        # Prepare hyperparameters
        hyperparameters = {
            "n_epochs": n_epochs,
            "learning_rate_multiplier": learning_rate_multiplier,
        }
        if batch_size:
            hyperparameters["batch_size"] = batch_size

        # Check queue
        queue_result = self.queue.train(
            dataset_path=dataset_path,
            base_model=base_model,
            hyperparameters=hyperparameters,
            size=size,
            force=force
        )

        # If already trained or in progress, return immediately
        if queue_result["status"] in ["already_trained", "in_progress", "locked"]:
            return queue_result

        # Get fingerprint and dataset path for this run
        fingerprint = queue_result["fingerprint"]
        effective_dataset_path = queue_result["dataset_path"]

        # Load dataset metadata
        dataset_path_obj = Path(effective_dataset_path)
        metadata_path = dataset_path_obj / "metadata.json"
        dataset_metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                dataset_metadata = json.load(f)

        # Initialize W&B
        wandb_run = self._init_wandb(
            fingerprint=fingerprint,
            config={
                "base_model": base_model,
                "dataset_path": dataset_path,
                "hyperparameters": hyperparameters,
                "size": size
            },
            dataset_metadata=dataset_metadata
        )

        try:
            # Prepare cleaned training files (remove 'meta' field)
            dataset_path = Path(effective_dataset_path)
            cleaned_train_path, cleaned_val_path = self._prepare_training_files(dataset_path, size)

            logger.info(f"Uploading training files for {fingerprint}")
            if size:
                logger.info(f"Using subsampled data with size {size}")

            # Upload cleaned files using OpenAI API
            with open(cleaned_train_path, "rb") as f:
                train_file = self.client.files.create(file=f,
                                                      purpose="fine-tune")

            with open(cleaned_val_path, "rb") as f:
                val_file = self.client.files.create(file=f,
                                                    purpose="fine-tune")

            logger.info(f"Files uploaded: train={train_file.id}, val={val_file.id}")

            # Clean up temporary files
            cleaned_train_path.unlink()
            cleaned_val_path.unlink()
            cleaned_train_path.parent.rmdir()

            # Log to W&B
            if wandb_run:
                wandb.log({
                    "openai/train_file_id": train_file.id,
                    "openai/val_file_id": val_file.id,
                })

            # Create fine-tuning job
            suffix = f"lie-detector-{fingerprint}"
            if size:
                suffix += f"-size{size}"

            try:
                job = self.client.fine_tuning.jobs.create(
                    training_file=train_file.id,
                    validation_file=val_file.id,
                    model=base_model,
                    hyperparameters=hyperparameters,
                    suffix=suffix
                )
                logger.info(f"Created fine-tuning job: {job.id}")
            except Exception as e:
                logger.error(f"Failed to create fine-tuning job: {e}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error details: {str(e)}")
                raise


            logger.info(f"Created fine-tuning job: {job.id}")
            logger.info(f"Initial status: {job.status}")

            # Log job creation to W&B
            if wandb_run:
                wandb.log({
                    "openai/job_id": job.id,
                    "openai/job_status": job.status,
                    "openai/job_created_at": job.created_at,
                    "openai/estimated_finish": job.estimated_finish if hasattr(job, 'estimated_finish') else None,
                })

            # Save job info
            run_dir = Path(queue_result["run_dir"])
            job_info_path = run_dir / "openai_job.json"
            with open(job_info_path, "w") as f:
                json.dump({
                    "job_id": job.id,
                    "train_file_id": train_file.id,
                    "val_file_id": val_file.id,
                    "sample_size": size,
                    "created_at": datetime.now().isoformat(),
                    "wandb_run_id": wandb_run.id if wandb_run else None,
                    "wandb_run_name": wandb_run.name if wandb_run else None,
                }, f, indent=2)

            # Monitor job
            return self._monitor_job(job.id, fingerprint, wandb_run)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.queue.fail_training(fingerprint, str(e))

            # Log failure to W&B
            if wandb_run:
                wandb.log({"error": str(e), "status": "failed"})
                wandb.finish(exit_code=1)

            raise

    def _monitor_job(self, job_id: str, fingerprint: str, wandb_run: Optional[Any]) -> Dict[str, Any]:
        """Monitor OpenAI fine-tuning job until completion with W&B logging."""
        logger.info(f"Monitoring job {job_id}")


        step = 0
        last_event_count = 0
        last_status = None
        last_status_message = None

        while True:
            try:
                job = self.client.fine_tuning.jobs.retrieve(job_id)

                logger.info(f"Full job object: {job}")

                # Log status changes
                if job.status != last_status:
                    last_status = job.status
                    status_message = f"Job status changed to: {job.status}"

                    # Extract additional information from job object
                    if hasattr(job, 'status_details') and job.status_details:
                        status_message += f" - {job.status_details}"

                    logger.info(status_message)

                    if wandb_run:
                        wandb.log({
                            "openai/status": job.status,
                            "openai/status_message": status_message,
                            "openai/status_change_time": datetime.now().isoformat()
                        })

                # Get events for logging
                events = self.client.fine_tuning.jobs.list_events(job_id, limit=100)

                # Log new events to W&B
                if len(events.data) > last_event_count:
                    new_events = events.data[last_event_count:]
                    last_event_count = len(events.data)

                    for event in reversed(new_events):  # Process in chronological order
                        # Log event messages
                        if hasattr(event, 'message') and event.message:
                            message = event.message

                            # Check if this is a different message than last time
                            if message != last_status_message:
                                last_status_message = message
                                logger.info(f"OpenAI: {message}")

                                # Parse specific message types
                                if "queue" in message.lower():
                                    # Extract queue position if available
                                    import re
                                    queue_match = re.search(r'position (\d+)', message, re.IGNORECASE)
                                    if queue_match and wandb_run:
                                        queue_position = int(queue_match.group(1))
                                        wandb.log({"openai/queue_position": queue_position})

                                if wandb_run:
                                    wandb.log({
                                        "openai/message": message,
                                        "openai/message_time": event.created_at if hasattr(event,
                                                                                           'created_at') else datetime.now().timestamp()
                                    })

                        # Log metrics from data field
                        if hasattr(event, "data") and event.data:
                            data = event.data

                            # Log metrics
                            wandb_metrics = {
                                "step": step
                            }

                            if "train_loss" in data:
                                wandb_metrics["train/loss"] = data["train_loss"]
                            if "valid_loss" in data:
                                wandb_metrics["val/loss"] = data["valid_loss"]
                            if "train_accuracy" in data:
                                wandb_metrics["train/accuracy"] = data["train_accuracy"]
                            if "valid_accuracy" in data:
                                wandb_metrics["val/accuracy"] = data["valid_accuracy"]
                            if "epoch" in data:
                                wandb_metrics["epoch"] = data["epoch"]

                            # Log current learning rate if available
                            if "learning_rate" in data:
                                wandb_metrics["train/learning_rate"] = data["learning_rate"]

                            # Log step/iteration info
                            if "step" in data:
                                wandb_metrics["train/step"] = data["step"]
                            if "total_steps" in data:
                                wandb_metrics["train/total_steps"] = data["total_steps"]

                            if wandb_run and wandb_metrics:
                                wandb.log(wandb_metrics)

                            step += 1

                if job.status == "succeeded":
                    # Get final metrics
                    metrics = self._extract_metrics(events)

                    # Mark as completed in queue
                    result = self.queue.complete_training(
                        fingerprint=fingerprint,
                        model_id=job.fine_tuned_model,
                        metrics=metrics
                    )

                    logger.info(f"Training completed: {job.fine_tuned_model}")

                    # Log final results to W&B
                    if wandb_run:
                        wandb.log({
                            "final/model_id": job.fine_tuned_model,
                            "final/status": "completed",
                            **{f"final/{k}": v for k, v in metrics.items()}
                        })

                        # Create W&B artifact for the model
                        try:
                            artifact = wandb.Artifact(
                                name=f"model-{fingerprint}",
                                type="model",
                                description=f"Fine-tuned model {job.fine_tuned_model}",
                                metadata={
                                    "model_id": job.fine_tuned_model,
                                    "base_model": job.model,
                                    "fingerprint": fingerprint,
                                    "metrics": metrics
                                }
                            )
                            wandb.log_artifact(artifact)
                        except Exception as e:
                            logger.error(f"Failed to create W&B artifact: {e}")

                        wandb.finish()

                    return {
                        "status": "completed",
                        "fingerprint": fingerprint,
                        "model_id": job.fine_tuned_model,
                        "job_id": job_id,
                        "metrics": metrics,
                        "wandb_run_id": wandb_run.id if wandb_run else None,
                        "wandb_run_name": wandb_run.name if wandb_run else None,
                    }

                elif job.status == "failed":
                    error = job.error or "Unknown error"
                    self.queue.fail_training(fingerprint, error)

                    if wandb_run:
                        wandb.log({"error": error, "status": "failed"})
                        wandb.finish(exit_code=1)

                    return {
                        "status": "failed",
                        "fingerprint": fingerprint,
                        "job_id": job_id,
                        "error": error
                    }

                elif job.status == "cancelled":
                    self.queue.fail_training(fingerprint, "Job was cancelled")

                    if wandb_run:
                        wandb.log({"status": "cancelled"})
                        wandb.finish(exit_code=2)

                    return {
                        "status": "cancelled",
                        "fingerprint": fingerprint,
                        "job_id": job_id
                    }

                # Still running - wait and check again
                time.sleep(10)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error monitoring job: {e}")
                if wandb_run:
                    wandb.log({"monitoring_error": str(e)})
                time.sleep(60)  # Wait longer on error

    def _extract_metrics(self, events) -> Dict[str, Any]:
        """Extract metrics from fine-tuning events."""
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }

        for event in events.data:
            if hasattr(event, "data") and event.data:
                data = event.data
                if "train_loss" in data:
                    metrics["train_loss"].append(data["train_loss"])
                if "valid_loss" in data:
                    metrics["val_loss"].append(data["valid_loss"])
                if "train_accuracy" in data:
                    metrics["train_accuracy"].append(data["train_accuracy"])
                if "valid_accuracy" in data:
                    metrics["val_accuracy"].append(data["valid_accuracy"])

        # Get final metrics
        final_metrics = {}
        for key, values in metrics.items():
            if values:
                final_metrics[f"final_{key}"] = values[-1]
                final_metrics[f"best_{key}"] = min(values) if "loss" in key else max(values)

        return final_metrics

    def train_with_cli(self, dataset_path: str, base_model: str = "openai/gpt-4o",
                       n_epochs: int = 3, dry_run: bool = False,
                       force: bool = False) -> Dict[str, Any]:
        """
        Train using the SafetyTools CLI with queue management.

        This method uses subprocess to call the existing CLI.
        """
        # Prepare hyperparameters
        hyperparameters = {
            "n_epochs": n_epochs,
            "model": base_model
        }

        # Check queue
        queue_result = self.queue.train(
            dataset_path=dataset_path,
            base_model=base_model,
            hyperparameters=hyperparameters,
            force=force
        )

        # If already trained or in progress, return immediately
        if queue_result["status"] in ["already_trained", "in_progress", "locked"]:
            return queue_result

        # Get fingerprint
        fingerprint = queue_result["fingerprint"]

        try:
            # Prepare command
            dataset_path = Path(dataset_path)
            cmd = [
                "python", "-m", "safetytooling.apis.finetuning.openai.run",
                "--model", base_model,
                "--train_file", str(dataset_path / "train.jsonl"),
                "--val_file", str(dataset_path / "val.jsonl"),
                "--n_epochs", str(n_epochs)
            ]

            if dry_run:
                cmd.append("--dry_run")

            # Log to run directory
            run_dir = Path(queue_result["run_dir"])
            log_file = run_dir / "training.log"

            logger.info(f"Starting training with command: {' '.join(cmd)}")

            # Run training
            with open(log_file, "w") as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )

                # Stream output
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                    log.flush()

                # Wait for completion
                return_code = process.wait()

            if return_code == 0:
                # Parse output to get model ID
                # This is a simplified example - you'd need to parse actual output
                model_id = f"ft:gpt-4o-{datetime.now().strftime('%Y%m%d')}-{fingerprint}"

                # Mark as completed
                result = self.queue.complete_training(
                    fingerprint=fingerprint,
                    model_id=model_id,
                    metrics={"status": "completed via CLI"}
                )

                return {
                    "status": "completed",
                    "fingerprint": fingerprint,
                    "model_id": model_id,
                    "log_file": str(log_file)
                }
            else:
                raise Exception(f"Training failed with return code {return_code}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.queue.fail_training(fingerprint, str(e))
            return {
                "status": "failed",
                "fingerprint": fingerprint,
                "error": str(e)
            }


# Example usage script
def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Train models with queue management and W&B logging')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--model', default='gpt-4o-2024-08-06', help='Base model')
    parser.add_argument('--all-folds', action='store_true', help='Train all folds')
    parser.add_argument('--n-epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--size', type=int, help='Subsample size (maintains balance)')
    parser.add_argument('--force', action='store_true', help='Force retraining')
    parser.add_argument('--use-cli', action='store_true', help='Use SafetyTools CLI')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--wandb-project', type=str, help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, help='W&B entity/team name')

    args = parser.parse_args()

    # Initialize trainer
    trainer = OpenAITrainer(
        enable_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )

    if args.all_folds:
        # Train all folds
        dataset_path = Path(args.dataset)
        results = []

        for fold_dir in dataset_path.iterdir():
            if fold_dir.is_dir() and (fold_dir / "train.jsonl").exists():
                print(f"\nTraining fold: {fold_dir.name}")
                if args.size:
                    print(f"Using balanced subsample of size {args.size}")

                if args.use_cli:
                    result = trainer.train_with_cli(
                        str(fold_dir),
                        args.model,
                        args.n_epochs,
                        force=args.force
                    )
                else:
                    result = trainer.train(
                        str(fold_dir),
                        args.model,
                        args.n_epochs,
                        size=args.size,
                        force=args.force
                    )

                results.append({
                    "fold": fold_dir.name,
                    **result
                })

                print(json.dumps(result, indent=2))

        # Summary
        print(f"\n{'=' * 60}")
        print("TRAINING SUMMARY")
        print(f"{'=' * 60}")
        for result in results:
            status = result.get("status", "unknown")
            model_id = result.get("model_id", "N/A")
            size_info = f" (size={result.get('sample_size')})" if result.get('sample_size') else ""
            wandb_info = f" [W&B: {result.get('wandb_run_name', 'N/A')}]" if result.get('wandb_run_id') else ""
            print(f"{result['fold']}: {status} - {model_id}{size_info}{wandb_info}")

    else:
        # Train single dataset
        if args.size:
            print(f"Using balanced subsample of size {args.size}")

        if args.use_cli:
            result = trainer.train_with_cli(
                args.dataset,
                args.model,
                args.n_epochs,
                force=args.force
            )
        else:
            result = trainer.train(
                args.dataset,
                args.model,
                args.n_epochs,
                size=args.size,
                force=args.force
            )

        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()