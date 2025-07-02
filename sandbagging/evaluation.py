"""
Evaluation pipeline for sandbagging detection.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from inspect_ai import eval
from inspect_ai.model import Model, get_model

from tasks import sandbagging_task, get_sandbagging_task
from dataset import create_sandbagging_dataset, sandbagging_dataset_factory
from scorer import compute_sandbagging_metrics
from models import SandbaggingConfig, SandbaggingDomain


class SandbaggingEvaluator:
    """Main evaluator for sandbagging detection experiments."""
    
    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: str = "openai/gpt-4o-mini",
        output_dir: str = "sandbagging_results",
        runs_per_prompt: int = 20
    ):
        self.model = model or get_model(model_name)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.runs_per_prompt = runs_per_prompt
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
    
    async def evaluate_single_task(
        self,
        task_name: str = "sandbagging_detection",
        **task_kwargs
    ) -> Dict[str, Any]:
        """Evaluate a single sandbagging task."""
        
        # Get the task
        task = get_sandbagging_task(task_name)
        
        # Run evaluation
        logs = await eval(
            task,
            model=self.model,
            limit=None,  # Evaluate all samples
            epochs=self.runs_per_prompt,
            log_dir=str(self.output_dir / "logs"),
            log_level="info"
        )
        
        return logs
    
    async def evaluate_control_vs_sandbagging(
        self,
        domains: Optional[List[SandbaggingDomain]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate both control and sandbagging variants to detect sandbagging behavior.
        """
        
        results = {}
        
        # Create datasets for both variants
        control_dataset = create_sandbagging_dataset(variants=["control"], domains=domains)
        sandbagging_dataset = create_sandbagging_dataset(variants=["sandbagging"], domains=domains)
        
        # Evaluate control condition
        print("Evaluating control condition...")
        control_task = sandbagging_task(
            name="sandbagging_control",
            dataset=control_dataset
        )
        
        control_logs = await eval(
            control_task,
            model=self.model,
            epochs=self.runs_per_prompt,
            log_dir=str(self.output_dir / "logs"),
            log_level="info"
        )
        
        results["control"] = control_logs
        
        # Evaluate sandbagging condition
        print("Evaluating sandbagging condition...")
        sandbagging_task_obj = sandbagging_task(
            name="sandbagging_harmful",
            dataset=sandbagging_dataset
        )
        
        sandbagging_logs = await eval(
            sandbagging_task_obj,
            model=self.model,
            epochs=self.runs_per_prompt,
            log_dir=str(self.output_dir / "logs"),
            log_level="info"
        )
        
        results["sandbagging"] = sandbagging_logs
        
        # Compute comparative metrics
        all_samples = []
        
        # Extract results from control logs
        for sample_data in control_logs[0].samples:
            sample_result = {
                'question_id': sample_data.metadata.get('question_id'),
                'variant': 'control',
                'is_correct': sample_data.scores['sandbagging_answer_accuracy_scorer/accuracy'].value == 1.0,
                'domain': sample_data.metadata.get('domain'),
                'response': sample_data.output.completion if hasattr(sample_data, 'output') else '',
                'metadata': sample_data.metadata
            }
            all_samples.append(sample_result)
        
        # Extract results from sandbagging logs
        for sample_data in sandbagging_logs[0].samples:
            sample_result = {
                'question_id': sample_data.metadata.get('question_id'),
                'variant': 'sandbagging',
                'is_correct': sample_data.scores['sandbagging_answer_accuracy_scorer/accuracy'].value == 1.0,
                'domain': sample_data.metadata.get('domain'),
                'response': sample_data.output.completion if hasattr(sample_data, 'output') else '',
                'metadata': sample_data.metadata
            }
            all_samples.append(sample_result)
        
        # Compute sandbagging metrics
        metrics = compute_sandbagging_metrics(all_samples)
        results["analysis"] = metrics
        
        return results
    
    async def evaluate_domain_comparison(self) -> Dict[str, Any]:
        """Evaluate sandbagging across different domains."""
        
        domain_results = {}
        
        for domain in SandbaggingDomain:
            print(f"Evaluating domain: {domain.value}")
            
            domain_results[domain.value] = await self.evaluate_control_vs_sandbagging(
                domains=[domain]
            )
        
        return domain_results
    
    async def evaluate_harm_gradient(self) -> Dict[str, Any]:
        """Evaluate sandbagging across different harm severity levels."""
        
        # This would require prompts with varying harm levels
        # For now, we'll analyze by the existing harm_severity metadata
        
        results = await self.evaluate_control_vs_sandbagging()
        
        # Group by harm severity
        severity_results = {}
        
        for sample in results.get("analysis", {}).get("question_level_metrics", {}).items():
            # This would need to be implemented based on the harm severity
            pass
        
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


async def run_sandbagging_evaluation(
    model_name: str = "openai/gpt-4o-mini",
    output_dir: str = "sandbagging_results",
    runs_per_prompt: int = 20,
    domains: Optional[List[str]] = None
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
    results = await evaluator.evaluate_control_vs_sandbagging(domains=domain_enums)
    
    # Save and summarize
    evaluator.save_results(results)
    evaluator.print_summary(results)
    
    return results


if __name__ == "__main__":
    # Example usage
    results = asyncio.run(run_sandbagging_evaluation(
        model_name="openai/gpt-4o-mini",
        runs_per_prompt=10  # Reduced for quick testing
    ))