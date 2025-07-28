import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from inspect_ai.hooks import Hooks, RunEnd, TaskEnd
from inspect_ai.log import EvalLog

from common.s3_sample_client import S3SampleClient

load_dotenv()
from common.utils import write_to_s3


class BaseSampleProcessingHook(Hooks):
    """
    Base hook class for processing EvalLog samples with concurrent S3 uploads.
    Subclasses should implement `process_sample()` and optionally `save_results()`.
    """

    def __init__(self, output_dir: str = "parsed_logs", max_workers: int = 20):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        self.s3_sample_client: S3SampleClient = S3SampleClient()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def upload_sample_async(self, model: str, task_name: str, sample_id: str, parsed_entry: Dict[str, Any]):
        """Upload a single sample to S3 asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.s3_sample_client.put_sample,
            model,
            task_name,
            sample_id,
            parsed_entry
        )

    async def on_task_end(self, data: TaskEnd):
        """
        Hook that runs at the end of each evaluation run to parse logs and create JSONL output.
        Now with concurrent S3 uploads for individual samples.

        Args:
            data: TaskEnd object containing run_id and logs
        """
        print(f"[LogParserHook] Processing run: {data.run_id}")

        # Get the first log from the logs list
        if not data.log:
            print(f"[LogParserHook] Warning: No logs found for run: {data.run_id}")
            return

        # Process each log file
        eval_log = data.log
        filepath = Path(Path(eval_log.location).parent.parent.as_posix() + '/data')
        try:
            os.mkdir(filepath)
        except FileExistsError:
            pass

        try:
            # Extract basic information
            task_name = eval_log.eval.task
            task_id = eval_log.eval.task_id
            timestamp = eval_log.eval.created

            # Create output filename
            output_filename = f"parsed_{task_name}_{task_id}.jsonl"

            # Create clean task name for S3 subdirectory
            clean_task_name = task_name.replace('_', '-')

            samples = eval_log.samples

            if samples is None:
                print(f"[LogParserHook] Warning: No samples found in log")
                return

            # Collect all parsed entries and prepare upload tasks
            all_entries = []
            upload_tasks = []
            sample_info = []  # Keep track of sample info for error reporting

            for sample in samples:
                parsed_entry = self.process_sample(sample, eval_log)
                if parsed_entry:
                    all_entries.append(parsed_entry)

                    # Prepare upload task
                    sample_id = parsed_entry.get('sample_id', '-')
                    sample_info.append(sample_id)

                    upload_task = self.upload_sample_async(
                        sample.output.model,
                        task_name,
                        sample_id,
                        parsed_entry
                    )
                    upload_tasks.append(upload_task)

            # Execute all uploads concurrently
            print(f"[LogParserHook] Starting concurrent upload of {len(upload_tasks)} samples...")
            upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)

            # Check results and count successes/failures
            successful_uploads = 0
            failed_uploads = 0

            for i, result in enumerate(upload_results):
                if isinstance(result, Exception):
                    failed_uploads += 1
                    print(f"[LogParserHook] Warning: S3 upload failed for sample {sample_info[i]}: {result}")
                elif result is False:
                    failed_uploads += 1
                    print(f"[LogParserHook] Warning: S3 upload failed for sample {sample_info[i]}")
                else:
                    successful_uploads += 1

            print(f"[LogParserHook] Upload complete: {successful_uploads} successful, {failed_uploads} failed")

            # Create JSONL content
            jsonl_content = '\n'.join([json.dumps(entry) for entry in all_entries])

            # Write to local data folder
            local_file_path = filepath.as_posix() + '/' + output_filename
            with open(local_file_path, 'w') as output_file:
                output_file.write(jsonl_content)
            print(f"[LogParserHook] Created local file: {local_file_path} with {len(all_entries)} entries")

            # Also write aggregated JSONL to S3
            s3_success = write_to_s3(jsonl_content, output_filename, clean_task_name=clean_task_name)
            if s3_success:
                print(f"[LogParserHook] Successfully uploaded aggregated file: {output_filename} to S3")
            else:
                print(f"[LogParserHook] Warning: Failed to upload aggregated file to S3, but local file was created")

        except Exception as e:
            print(f"[LogParserHook] Error processing log file: {e}")
        finally:
            # Clean up executor when done
            self.executor.shutdown(wait=False)

    @abstractmethod
    def process_sample(self, sample: Any, eval_log: EvalLog) -> Optional[Dict[str, Any]]:
        """
        Process a single sample and return a dictionary.
        Return None to skip this sample.
        """
        pass

    def process_eval_log(self, eval_log: EvalLog) -> None:
        """
        Process all samples in an EvalLog.
        """
        for sample in getattr(eval_log, 'samples', []):
            try:
                parsed = self.process_sample(sample, eval_log)
                if parsed:
                    self.results.append(parsed)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error processing sample: {e}")

    def save_jsonl(self, filename: str) -> None:
        """
        Save all processed samples to a JSONL file in the output directory.
        """
        output_path = self.output_dir / filename
        with output_path.open("w") as f:
            for item in self.results:
                f.write(json.dumps(item) + "\n")

        print(f"[{self.__class__.__name__}] Saved {len(self.results)} samples to {output_path}")