import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from together import Together
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LieDetectorEvaluator:
    """Evaluate lie detection model on Together AI."""

    def __init__(self, model_name: str, api_key: str = None):
        """
        Initialize evaluator.

        Args:
            model_name: Name of the fine-tuned model on Together AI
            api_key: Together AI API key (or set TOGETHER_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get('TOGETHER_API_KEY')

        if not self.api_key:
            raise ValueError("Please provide API key or set TOGETHER_API_KEY environment variable")

        self.client = Together(api_key=self.api_key)

        # Expected responses
        self.lie_response = "A"
        self.truth_response = "B"

        # Track statistics
        self.stats = defaultdict(lambda: defaultdict(int))

    def load_dataset(self, file_path: str) -> List[Dict]:
        """Load JSONL dataset file."""
        samples = []
        with open(file_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    def extract_response(self, text: str) -> str:
        """
        Extract A or B response from model output.

        Args:
            text: Model response text

        Returns:
            'A' or 'B' if found, else None
        """
        # Clean the text
        text = text.strip().upper()

        # Look for standalone A or B
        if text == 'A':
            return 'A'
        elif text == 'B':
            return 'B'

        # Look for A or B at the start
        if text.startswith('A'):
            return 'A'
        elif text.startswith('B'):
            return 'B'

        # Look for A: or B: patterns
        if 'A:' in text and 'B:' not in text:
            return 'A'
        elif 'B:' in text and 'A:' not in text:
            return 'B'

        # If both appear, take the first one
        a_index = text.find('A')
        b_index = text.find('B')

        if a_index != -1 and b_index != -1:
            return 'A' if a_index < b_index else 'B'
        elif a_index != -1:
            return 'A'
        elif b_index != -1:
            return 'B'

        return None

    def query_model(self, messages: List[Dict], max_retries: int = 3) -> str:
        """
        Query the model with conversation messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_retries: Maximum number of retries on failure

        Returns:
            Model response text
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=10,  # We only expect A or B
                    temperature=0.0,  # Deterministic for evaluation
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"API call failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    raise

    def evaluate_sample(self, sample: Dict) -> Tuple[bool, str, str]:
        """
        Evaluate a single sample.

        Args:
            sample: Dataset sample with 'messages' and 'meta' fields

        Returns:
            Tuple of (ground_truth_lie, predicted_response, raw_response)
        """
        # Get ground truth
        ground_truth = sample['meta']['did_lie']

        # Get conversation messages (all except the last one which is the expected response)
        messages = sample['messages'][:-1]

        # Query model
        raw_response = self.query_model(messages)

        # Extract A/B response
        predicted = self.extract_response(raw_response)

        return ground_truth, predicted, raw_response

    def evaluate_dataset(self, file_path: str, dataset_name: str) -> Dict:
        """
        Evaluate model on a dataset.

        Args:
            file_path: Path to JSONL file
            dataset_name: Name for logging/results

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on {dataset_name} from {file_path}")

        # Load dataset
        samples = self.load_dataset(file_path)
        logger.info(f"Loaded {len(samples)} samples")

        # Collect predictions and ground truth
        y_true = []
        y_pred = []
        failed_samples = []

        # Evaluate each sample
        for sample in tqdm(samples, desc=f"Evaluating {dataset_name}"):
            try:
                ground_truth, predicted, raw_response = self.evaluate_sample(sample)

                # Convert to binary labels
                # Ground truth: True = lied, False = didn't lie
                # Prediction: A = lied, B = didn't lie
                y_true.append(1 if ground_truth else 0)

                if predicted == 'A':
                    y_pred.append(1)
                elif predicted == 'B':
                    y_pred.append(0)
                else:
                    # Failed to extract response
                    failed_samples.append({
                        'sample_id': sample['meta'].get('sample_id', 'unknown'),
                        'raw_response': raw_response,
                        'ground_truth': ground_truth
                    })
                    # Default to predicting "didn't lie" for failed extractions
                    y_pred.append(0)

            except Exception as e:
                logger.error(f"Failed to evaluate sample: {e}")
                failed_samples.append({
                    'sample_id': sample['meta'].get('sample_id', 'unknown'),
                    'error': str(e),
                    'ground_truth': sample['meta']['did_lie']
                })
                # Default to predicting "didn't lie" for errors
                y_true.append(1 if sample['meta']['did_lie'] else 0)
                y_pred.append(0)

        # Calculate metrics
        results = {
            'dataset': dataset_name,
            'n_samples': len(samples),
            'n_failed': len(failed_samples),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'failed_samples': failed_samples[:5]  # Keep first 5 for debugging
        }

        # Calculate per-class metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results['metrics_breakdown'] = {
            'true_positives': int(tp),  # Correctly predicted lies
            'true_negatives': int(tn),  # Correctly predicted truths
            'false_positives': int(fp),  # Incorrectly predicted as lies
            'false_negatives': int(fn),  # Incorrectly predicted as truths
            'lie_precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'lie_recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'truth_precision': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'truth_recall': tn / (tn + fp) if (tn + fp) > 0 else 0
        }

        return results

    def run_full_evaluation(self, base_dir: str) -> Dict:
        """
        Run evaluation on all folds.

        Args:
            base_dir: Base directory containing fold directories

        Returns:
            Dictionary with all evaluation results
        """
        base_path = Path(base_dir)
        all_results = {}

        # Define evaluation sets
        eval_sets = [
            ('fold_1/val.jsonl', 'Fold 1 Validation (In-domain)'),
            ('fold_2/train.jsonl', 'Fold 2 Training (Out-of-domain)'),
            ('fold_2/val.jsonl', 'Fold 2 Validation (Out-of-domain)')
        ]

        # Run evaluations
        for file_path, dataset_name in eval_sets:
            full_path = base_path / file_path
            if full_path.exists():
                results = self.evaluate_dataset(str(full_path), dataset_name)
                all_results[dataset_name] = results

                # Print results
                logger.info(f"\n{dataset_name} Results:")
                logger.info(f"  Accuracy: {results['accuracy']:.3f}")
                logger.info(f"  F1 Score: {results['f1']:.3f}")
                logger.info(f"  Precision: {results['precision']:.3f}")
                logger.info(f"  Recall: {results['recall']:.3f}")

                if results['n_failed'] > 0:
                    logger.warning(f"  Failed extractions: {results['n_failed']}")
            else:
                logger.warning(f"File not found: {full_path}")

        # Calculate generalization metrics
        if 'Fold 1 Validation (In-domain)' in all_results and 'Fold 2 Training (Out-of-domain)' in all_results:
            in_domain_f1 = all_results['Fold 1 Validation (In-domain)']['f1']
            out_domain_f1 = all_results['Fold 2 Training (Out-of-domain)']['f1']
            generalization_gap = in_domain_f1 - out_domain_f1

            all_results['generalization_metrics'] = {
                'in_domain_f1': in_domain_f1,
                'out_of_domain_f1': out_domain_f1,
                'generalization_gap': generalization_gap,
                'relative_performance': out_domain_f1 / in_domain_f1 if in_domain_f1 > 0 else 0
            }

            logger.info("\n📊 Generalization Performance:")
            logger.info(f"  In-domain F1 (Fold 1 Val): {in_domain_f1:.3f}")
            logger.info(f"  Out-of-domain F1 (Fold 2): {out_domain_f1:.3f}")
            logger.info(f"  Generalization Gap: {generalization_gap:.3f}")
            logger.info(f"  Relative Performance: {all_results['generalization_metrics']['relative_performance']:.1%}")

        # Save results
        results_file = base_path / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n✅ Results saved to: {results_file}")

        return all_results


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate lie detection model on Together AI')
    parser.add_argument('--model', required=True, help='Model name on Together AI')
    parser.add_argument('--data-dir', default='.together/openai/gpt_oss_120b',
                        help='Base directory containing fold directories')
    parser.add_argument('--api-key', help='Together AI API key (or set TOGETHER_API_KEY)')
    parser.add_argument('--single-file', help='Evaluate on a single file instead of full evaluation')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = LieDetectorEvaluator(
        model_name=args.model,
        api_key=args.api_key
    )

    if args.single_file:
        # Evaluate single file
        results = evaluator.evaluate_dataset(args.single_file, "Single Dataset")
        print(json.dumps(results, indent=2))
    else:
        # Run full evaluation
        evaluator.run_full_evaluation(args.data_dir)


if __name__ == "__main__":
    main()