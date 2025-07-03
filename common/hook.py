import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path

from inspect_ai.hooks import Hooks
from inspect_ai.log import EvalLog


class BaseSampleProcessingHook(Hooks, ABC):
    """
    Base hook class for processing EvalLog samples.
    Subclasses should implement `process_sample()` and optionally `save_results()`.
    """

    def __init__(self, output_dir: str = "parsed_logs"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

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
