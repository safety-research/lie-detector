import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.hooks import Hooks, RunEnd
from inspect_ai.log import EvalLog

from common.s3_sample_client import S3SampleClient

load_dotenv()
from common.utils import write_to_s3


class BaseSampleProcessingHook(Hooks):
    """
    Base hook class for processing EvalLog samples.
    Subclasses should implement `process_sample()` and optionally `save_results()`.
    """

    def __init__(self, output_dir: str = "parsed_logs"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        self.s3_sample_client: S3SampleClient = S3SampleClient()

    async def on_run_end(self, data: RunEnd):
        """
        Hook that runs at the end of each evaluation run to parse logs and create JSONL output.

        Args:
            data: RunEnd object containing run_id and logs
        """
        print(f"[LogParserHook] Processing run: {data.run_id}")

        # Get the first log from the logs list
        if not data.logs or len(data.logs) == 0:
            print(f"[LogParserHook] Warning: No logs found for run: {data.run_id}")
            return

        # Process each log file
        for eval_log in data.logs:
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
                    continue

                # Collect all parsed entries
                all_entries = []
                for sample in samples:
                    parsed_entry = self.process_sample(sample, eval_log)
                    if parsed_entry:
                        all_entries.append(parsed_entry)

                        # Store individual sample in S3
                        sample_id = parsed_entry.get('sample_id', '-')
                        s3_success = self.s3_sample_client.put_sample(
                            model=sample.output.model,
                            task=task_name,
                            sample_id=sample_id,
                            content=parsed_entry,
                        )
                        if not s3_success:
                            print(f"[LogParserHook] Warning: S3 individual sample upload failed for sample: {sample_id}")

                # Create JSONL content
                jsonl_content = '\n'.join([json.dumps(entry) for entry in all_entries])

                # Write to local data folder
                local_file_path = filepath.as_posix() + '/' + output_filename
                with open(local_file_path, 'w') as output_file:
                    output_file.write(jsonl_content)
                print(f"[LogParserHook] Created local file: {local_file_path} with {len(all_entries)} entries")

                # Also write to S3
                s3_success = write_to_s3(jsonl_content, output_filename, clean_task_name=clean_task_name)
                if s3_success:
                    print(
                        f"[LogParserHook] Successfully uploaded: {output_filename} with {len(all_entries)} entries to S3")
                else:
                    print(f"[LogParserHook] Warning: Failed to upload to S3, but local file was created")


            except Exception as e:
                print(f"[LogParserHook] Error processing log file: {e}")

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
