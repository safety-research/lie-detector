"""
Evaluation pipeline for sandbagging detection.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from inspect_ai import eval
from inspect_ai.model import Model, get_model

from dataset import create_sandbagging_contrastive_dataset
from models import SandbaggingDomain
from scorer import compute_sandbagging_metrics
from tasks import sandbagging_task

load_dotenv()

class SandbaggingEvaluator:
    """Main evaluator for sandbagging detection experiments."""
    
    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: str = "openai/gpt-4o-mini",
        output_dir: str = "../../../../logs",
        runs_per_prompt: int = 20
    ):
        self.model = model or get_model(model_name)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.runs_per_prompt = runs_per_prompt
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
    
    def evaluate(
        self,
        domains: Optional[List[SandbaggingDomain]] = None,
        limit = 5
    ) -> Dict[str, Any]:
        """
        Evaluate both control and sandbagging variants to detect sandbagging behavior.
        """
        results = {}
        
        # Create datasets for both variants
        sandbagging_dataset = create_sandbagging_contrastive_dataset(categories=domains)

        # Evaluate sandbagging condition
        print("Evaluating sandbagging condition...")
        sandbagging_task_obj = sandbagging_task(
            name="sandbagging_harmful",
            dataset=sandbagging_dataset
        )
        
        sandbagging_logs = eval(
            sandbagging_task_obj,
            model=self.model,
            epochs=self.runs_per_prompt,
            log_dir=str(self.output_dir / "logs"),
            log_level="info",
            limit=limit
        )
        
        results["sandbagging"] = sandbagging_logs
        
        # Compute comparative metrics
        all_samples = []

        # Extract results from sandbagging logs
        for sample_data in sandbagging_logs[0].samples:
            accuracy = sample_data.scores[
                'sandbagging_answer_accuracy_scorer/accuracy'].value if 'sandbagging_answer_accuracy_scorer/accuracy' in sample_data.scores else 0
            sample_result = {
                'question_id': sample_data.metadata.get('question_id'),
                'variant': 'sandbagging',
                'is_correct': accuracy == 1.0,
                'domain': sample_data.metadata.get('domain'),
                'response': sample_data.output.completion if hasattr(sample_data, 'output') else '',
                'metadata': sample_data.metadata
            }
            all_samples.append(sample_result)
        
        # Compute sandbagging metrics
        metrics = compute_sandbagging_metrics(all_samples)
        results["analysis"] = metrics
        
        return results

    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> Path:
        """Save evaluation results to JSON file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sandbagging_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert any non-serializable objects
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the evaluation results."""
        
        analysis = results.get("analysis", {})
        
        print("\n" + "="*60)
        print("SANDBAGGING DETECTION RESULTS")
        print("="*60)
        
        print(f"Model: {self.model_name}")
        print(f"Runs per prompt: {self.runs_per_prompt}")
        print(f"Questions analyzed: {analysis.get('questions_analyzed', 0)}")
        
        print(f"\nControl accuracy: {analysis.get('overall_control_accuracy', 0):.3f}")
        print(f"Sandbagging accuracy: {analysis.get('overall_sandbagging_accuracy', 0):.3f}")
        print(f"Sandbagging rate: {analysis.get('overall_sandbagging_rate', 0):.3f}")
        
        p_value = analysis.get('statistical_significance_p_value')
        if p_value is not None:
            print(f"Statistical significance (p-value): {p_value:.4f}")
            print(f"Significant at α=0.05: {analysis.get('is_significant', False)}")
        
        # Question-level breakdown
        question_metrics = analysis.get('question_level_metrics', {})
        if question_metrics:
            print(f"\nQuestion-level results:")
            for qid, metrics in list(question_metrics.items())[:5]:  # Show first 5
                print(f"  {qid}: control={metrics['control_accuracy']:.2f}, "
                      f"sandbagging={metrics['sandbagging_accuracy']:.2f}, "
                      f"rate={metrics['sandbagging_rate']:.2f}")
            
            if len(question_metrics) > 5:
                print(f"  ... and {len(question_metrics) - 5} more questions")


def run_sandbagging_evaluation(
    model_name: str = "openai/gpt-4o-mini",
    output_dir: str = "../../../../logs",
    runs_per_prompt: int = 20,
    domains: Optional[List[str]] = None,
    limit = 5
) -> Dict[str, Any]:
    """
    Run a complete sandbagging evaluation.
    
    Args:
        model_name: Name of the model to evaluate
        output_dir: Directory to save results
        runs_per_prompt: Number of runs per prompt
        domains: List of domain names to evaluate (None for all)
        
    Returns:
        Evaluation results
    """
    
    # Convert domain names to enums if provided
    domain_enums = None
    if domains:
        domain_enums = [SandbaggingDomain(d) for d in domains]
    
    evaluator = SandbaggingEvaluator(
        model_name=model_name,
        output_dir=output_dir,
        runs_per_prompt=runs_per_prompt
    )
    
    # Run evaluation
    results = evaluator.evaluate(domains=domain_enums, limit=limit)
    
    # Save and summarize
    evaluator.save_results(results)
    evaluator.print_summary(results)
    
    return results


if __name__ == "__main__":
    # Example usage
    results = run_sandbagging_evaluation(
        model_name="openrouter/anthropic/claude-3.7-sonnet",
        domains=["digital_security"],
        runs_per_prompt=1  # Reduced for quick testing
    )