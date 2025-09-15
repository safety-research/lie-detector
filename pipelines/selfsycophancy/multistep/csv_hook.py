"""
CSV output hook for self-sycophancy evaluation results.
Generates a comprehensive CSV with one row per task containing all key metrics.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from inspect_ai.hooks import Hooks, hooks, TaskEnd, RunEnd
from inspect_ai.log import EvalLog


@hooks(
    name="self_sycophancy_csv_hook",
    description="Outputs self-sycophancy evaluation results to CSV format with comprehensive metrics"
)
class SelfSycophancyCSVHook(Hooks):
    """
    Hook that creates a CSV file with aggregated metrics for each task.
    Each row represents one complete task evaluation with all its metrics.
    """

    def __init__(self, output_dir: str = "results", include_sample_details: bool = False):
        """
        Initialize the CSV hook.

        Args:
            output_dir: Directory to save CSV files
            include_sample_details: If True, also create a detailed CSV with sample-level data
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_results: List[Dict[str, Any]] = []
        self.sample_results: List[Dict[str, Any]] = []
        self.include_sample_details = include_sample_details

    async def on_task_end(self, data: TaskEnd) -> None:
        """
        Process task results and extract aggregated metrics.
        """
        try:
            eval_log = data.log

            # Basic task information
            task_info = {
                'run_id': data.run_id,
                'eval_id': data.eval_id,
                'task': eval_log.eval.task,
                'task_id': eval_log.eval.task_id,
                'model': eval_log.eval.model,
                'timestamp': str(eval_log.eval.created),
                'dataset': eval_log.eval.dataset.name if eval_log.eval.dataset else 'unknown',
                'n_samples': len(eval_log.samples) if eval_log.samples else 0,
            }

            # Extract scorer metrics (aggregated across samples)
            scorer_metrics = {}
            if eval_log.results and eval_log.results.scores:
                for scorer in eval_log.results.scores:
                    score_data = scorer.metrics
                    if score_data:
                        # Get the aggregate value and any metrics
                        #scorer_metrics[f'{scorer.name}_value'] = scorer.metrics['mean']

                        # Add any aggregated metrics (mean, std, etc.)
                        for metric_name, metric_value in score_data.items():
                            scorer_metrics[f'{scorer.name}_{metric_name}'] = metric_value.value

            # Process samples to extract additional statistics
            if eval_log.samples and self.include_sample_details:
                try:
                    sample_stats = self._process_samples_for_stats(eval_log.samples)
                    task_info.update(sample_stats)
                except Exception as e:
                    pass

                # Also save sample-level details
                for sample in eval_log.samples:
                    try:
                        sample_row = self._extract_sample_row(sample, eval_log, data.run_id, data.eval_id)
                        if sample_row:
                            self.sample_results.append(sample_row)
                    except Exception as e:
                        continue

            # Combine all information
            task_row = {**task_info, **scorer_metrics}
            self.task_results.append(task_row)

            print(f"[SelfSycophancyCSVHook] Processed task {eval_log.eval.task_id} with {len(scorer_metrics)} metrics")
        except Exception as e:
            raise e

    def _process_samples_for_stats(self, samples: List[Any]) -> Dict[str, Any]:
        """
        Extract statistical summaries from samples.
        """
        stats = {}

        # Collect SSI values and other metrics
        ssi_values = []
        calibrated_effects = []
        control_drifts = []
        position_biases = []

        for sample in samples:
            if not sample.scores:
                continue

            # Extract SSI from comprehensive scorer
            if 'comprehensive_ssi_scorer' in sample.scores:
                score = sample.scores['comprehensive_ssi_scorer']
                if hasattr(score, 'value') and score.value is not None:
                    ssi_values.append(score.value)

            # Extract calibrated effect
            if 'calibrated_effect_scorer' in sample.scores:
                score = sample.scores['calibrated_effect_scorer']
                if hasattr(score, 'value') and score.value is not None:
                    calibrated_effects.append(score.value)

            # Extract consistency/drift
            if 'consistency_scorer' in sample.scores:
                score = sample.scores['consistency_scorer']
                if hasattr(score, 'value') and score.value is not None:
                    control_drifts.append(1 - score.value)  # Convert back to drift

            # Extract position bias
            if 'position_bias_scorer' in sample.scores:
                score = sample.scores['position_bias_scorer']
                if hasattr(score, 'value') and score.value is not None:
                    position_biases.append(score.value)

        # Calculate statistics
        if ssi_values:
            stats['ssi_mean'] = np.mean(ssi_values)
            stats['ssi_std'] = np.std(ssi_values)
            stats['ssi_median'] = np.median(ssi_values)
            stats['ssi_min'] = np.min(ssi_values)
            stats['ssi_max'] = np.max(ssi_values)
            stats['ssi_positive_rate'] = sum(1 for v in ssi_values if v > 0) / len(ssi_values)

        if calibrated_effects:
            stats['calibrated_effect_mean'] = np.mean(calibrated_effects)
            stats['calibrated_effect_std'] = np.std(calibrated_effects)
            stats['calibrated_effect_median'] = np.median(calibrated_effects)

        if control_drifts:
            stats['control_drift_mean'] = np.mean(control_drifts)
            stats['control_drift_max'] = np.max(control_drifts)

        if position_biases:
            stats['position_bias_mean'] = np.mean(position_biases)
            stats['position_bias_std'] = np.std(position_biases)

        return stats

    def _extract_sample_row(self, sample: Any, eval_log: EvalLog, run_id: str, eval_id: str) -> Optional[
        Dict[str, Any]]:
        """
        Extract detailed information from a single sample.
        """
        row = {
            'run_id': run_id,
            'eval_id': eval_id,
            'task': eval_log.eval.task,
            'model': eval_log.eval.model,
            'sample_id': getattr(sample, 'id', 'unknown'),
        }

        # Extract experimental results if available
        if hasattr(sample, 'metadata') and sample.metadata:
            exp_results = sample.metadata.get('experimental_results', {})
            if exp_results:
                row['choice_letter'] = exp_results.get('choice_letter')
                row['choice_index'] = exp_results.get('choice_index')
                row['baseline_scores'] = json.dumps(exp_results.get('baseline_scores', []))
                row['prefilled_actual_scores'] = json.dumps(exp_results.get('prefilled_actual_scores', []))
                row['refusal_detected'] = exp_results.get('refusal_detected', False)

                # Add choice probabilities if available
                if exp_results.get('choice_probs'):
                    for letter, prob in exp_results['choice_probs'].items():
                        row[f'choice_prob_{letter}'] = prob

        # Extract all scorer values
        if sample.scores:
            for scorer_name, score in sample.scores.items():
                if hasattr(score, 'value'):
                    row[f'{scorer_name}'] = score.value

                    # Add key metadata for comprehensive scorer
                    if scorer_name == 'comprehensive_ssi_scorer' and hasattr(score, 'metadata'):
                        metadata = score.metadata
                        if metadata:
                            # Extract SSI breakdown
                            ssi_breakdown = metadata.get('ssi_breakdown', {})
                            if ssi_breakdown:
                                for condition, ssi_data in ssi_breakdown.items():
                                    if isinstance(ssi_data, dict):
                                        row[f'ssi_{condition}'] = ssi_data.get('ssi')

                            # Extract effect sizes
                            effect_sizes = metadata.get('effect_sizes', {})
                            for effect_name, effect_value in effect_sizes.items():
                                row[f'effect_{effect_name}'] = effect_value

        return row

    async def on_run_end(self, data: RunEnd) -> None:
        """
        Save CSV files at the end of the run.
        """
        if not self.task_results:
            print("[SelfSycophancyCSVHook] No task results to save")
            return

        try:

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save task-level CSV
            task_csv_path = self.output_dir / f"sycophancy_results_tasks_{timestamp}.csv"
            self._save_csv(self.task_results, task_csv_path)
            print(f"[SelfSycophancyCSVHook] Saved task results to {task_csv_path}")

            # Save sample-level CSV if requested
            if self.include_sample_details and self.sample_results:
                sample_csv_path = self.output_dir / f"sycophancy_results_samples_{timestamp}.csv"
                self._save_csv(self.sample_results, sample_csv_path)
                print(f"[SelfSycophancyCSVHook] Saved sample details to {sample_csv_path}")

            # Create a summary CSV with key metrics only
            summary_csv_path = self.output_dir / f"sycophancy_summary_{timestamp}.csv"
            summary_results = self._create_summary(self.task_results)
            self._save_csv(summary_results, summary_csv_path)
            print(f"[SelfSycophancyCSVHook] Saved summary to {summary_csv_path}")
        except Exception as e:
            raise e

    def _save_csv(self, data: List[Dict[str, Any]], filepath: Path) -> None:
        """
        Save data to CSV file.
        """
        if not data:
            return

        # Get all unique keys across all rows
        all_keys = set()
        for row in data:
            all_keys.update(row.keys())

        # Sort keys for consistent column order
        fieldnames = sorted(all_keys)

        # Move important columns to the front
        priority_cols = ['run_id', 'eval_id', 'task', 'task_id', 'model', 'timestamp',
                         'sample_id', 'n_samples', 'ssi_mean', 'calibrated_effect_mean']
        ordered_fieldnames = []
        for col in priority_cols:
            if col in fieldnames:
                ordered_fieldnames.append(col)
                fieldnames.remove(col)
        ordered_fieldnames.extend(fieldnames)

        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
            writer.writeheader()
            writer.writerows(data)

    def _create_summary(self, task_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create a summary with only the most important metrics.
        """
        summary = []
        for task in task_results:
            summary_row = {
                'task': task.get('task'),
                'model': task.get('model'),
                'n_samples': task.get('n_samples'),
                'ssi_mean': task.get('ssi_mean'),
                'ssi_std': task.get('ssi_std'),
                'ssi_positive_rate': task.get('ssi_positive_rate'),
                'calibrated_effect_mean': task.get('calibrated_effect_mean'),
                'control_drift_mean': task.get('control_drift_mean'),
                'position_bias_mean': task.get('position_bias_mean'),
                'detection_rate': task.get('comprehensive_detection_scorer_value'),
            }
            # Remove None values
            summary_row = {k: v for k, v in summary_row.items() if v is not None}
            summary.append(summary_row)

        return summary


# # Convenience function to create the hook with common configurations
# def create_sycophancy_csv_hook(
#         output_dir: str = "results",
#         include_sample_details: bool = False
# ) -> SelfSycophancyCSVHook:
#     """
#     Create a self-sycophancy CSV hook with the specified configuration.
#
#     Args:
#         output_dir: Directory to save CSV files
#         include_sample_details: If True, also create a detailed CSV with sample-level data
#
#     Returns:
#         Configured SelfSycophancyCSVHook instance
#     """
#     return SelfSycophancyCSVHook(
#         output_dir=output_dir,
#         include_sample_details=include_sample_details
#     )