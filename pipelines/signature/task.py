from inspect_ai import Task, eval
from inspect_ai.dataset import hf_dataset, FieldSpec
from inspect_ai.solver import chain
from inspect_ai.scorer import mean, accuracy
from inspect_ai.model import get_model
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Import the logit-based components
from signature import (
    logit_based_lie_detector,
    logit_ambiguous_detector,
    logit_lie_scorer
)
from pipelines.selfsycophancy.correctness.solvers import prepare_science_math_question


def create_logit_lie_detection_task(
        dataset_name: str = "high_school_physics",
        num_samples: int = 100,
        num_signature_samples: int = 10
) -> Task:
    """
    Create a task that uses logit analysis for lie detection
    """

    # Load MMLU dataset
    dataset = hf_dataset(
        "cais/mmlu",
        split="test",
        name=dataset_name,
        sample_fields=FieldSpec(
            input="question",
            target="answer",
            metadata=["choices"]
        ),
        limit=num_samples
    )

    # Create solver chain
    solver = chain(
        # Prepare the question with correct/incorrect answers
        prepare_science_math_question(),

        # Run logit-based lie detection
        logit_based_lie_detector(num_samples=num_signature_samples),

        # Test on ambiguous prompts
        logit_ambiguous_detector()
    )

    # Create task with logit scorer
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=logit_lie_scorer(),
        metrics=[accuracy(), mean()]
    )


def analyze_logit_results(log_dir: str = "./logs/logit_detection"):
    """Analyze the logit-based detection results"""

    # Load results
    results_path = Path(log_dir) / "results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Extract metrics
    logit_kls = []
    generated_kls = []
    confidence_diffs = []
    entropy_diffs = []
    detection_accuracies = []

    for sample in results.get('samples', []):
        metadata = sample.get('metadata', {})

        if 'logit_kl_truth_to_lie' in metadata:
            logit_kls.append(metadata['logit_kl_truth_to_lie'])

        if 'generated_kl_truth_to_lie' in metadata:
            generated_kls.append(metadata['generated_kl_truth_to_lie'])

        feature_diffs = metadata.get('logit_feature_differences', {})
        if 'mean_confidence_diff' in feature_diffs:
            confidence_diffs.append(abs(feature_diffs['mean_confidence_diff']))

        if 'mean_position_entropy_diff' in feature_diffs:
            entropy_diffs.append(abs(feature_diffs['mean_position_entropy_diff']))

        if 'logit_detection_accuracy' in metadata:
            detection_accuracies.append(metadata['logit_detection_accuracy'])

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: KL Divergences
    ax = axes[0, 0]
    if logit_kls and generated_kls:
        x = np.arange(len(logit_kls))
        width = 0.35
        ax.bar(x - width / 2, logit_kls, width, label='Logit KL', alpha=0.8)
        ax.bar(x + width / 2, generated_kls, width, label='Generated KL', alpha=0.8)
        ax.set_xlabel('Sample')
        ax.set_ylabel('KL Divergence')
        ax.set_title('KL Divergences: Truth vs Lie')
        ax.legend()
        ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Threshold')

    # Plot 2: Confidence Differences
    ax = axes[0, 1]
    if confidence_diffs:
        ax.hist(confidence_diffs, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Absolute Confidence Difference')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Confidence Changes')
        ax.axvline(x=np.mean(confidence_diffs), color='r', linestyle='--',
                   label=f'Mean: {np.mean(confidence_diffs):.3f}')
        ax.legend()

    # Plot 3: Entropy Differences
    ax = axes[1, 0]
    if entropy_diffs:
        ax.hist(entropy_diffs, bins=20, alpha=0.7, edgecolor='black', color='green')
        ax.set_xlabel('Absolute Entropy Difference')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Entropy Changes')
        ax.axvline(x=np.mean(entropy_diffs), color='r', linestyle='--',
                   label=f'Mean: {np.mean(entropy_diffs):.3f}')
        ax.legend()

    # Plot 4: Detection Accuracy
    ax = axes[1, 1]
    if detection_accuracies:
        ax.hist(detection_accuracies, bins=10, alpha=0.7, edgecolor='black', color='orange')
        ax.set_xlabel('Detection Accuracy')
        ax.set_ylabel('Frequency')
        ax.set_title('Ambiguous Prompt Detection Accuracy')
        ax.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Random')
        ax.axvline(x=np.mean(detection_accuracies), color='r', linestyle='--',
                   label=f'Mean: {np.mean(detection_accuracies):.3f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig('logit_analysis_results.png', dpi=150)
    plt.show()

    # Print summary statistics
    print("\n=== Logit-Based Lie Detection Summary ===")
    print(f"Number of samples analyzed: {len(logit_kls)}")

    if logit_kls:
        print(f"\nLogit KL Divergence:")
        print(f"  Mean: {np.mean(logit_kls):.4f}")
        print(f"  Std: {np.std(logit_kls):.4f}")
        print(f"  Max: {np.max(logit_kls):.4f}")
        print(f"  % above threshold (0.1): {sum(k > 0.1 for k in logit_kls) / len(logit_kls) * 100:.1f}%")

    if generated_kls:
        print(f"\nGenerated KL Divergence:")
        print(f"  Mean: {np.mean(generated_kls):.4f}")
        print(f"  Std: {np.std(generated_kls):.4f}")
        print(f"  Max: {np.max(generated_kls):.4f}")

    if confidence_diffs:
        print(f"\nConfidence Differences:")
        print(f"  Mean: {np.mean(confidence_diffs):.4f}")
        print(f"  Std: {np.std(confidence_diffs):.4f}")

    if entropy_diffs:
        print(f"\nEntropy Differences:")
        print(f"  Mean: {np.mean(entropy_diffs):.4f}")
        print(f"  Std: {np.std(entropy_diffs):.4f}")

    if detection_accuracies:
        print(f"\nAmbiguous Prompt Detection:")
        print(f"  Mean accuracy: {np.mean(detection_accuracies):.3f}")
        print(f"  Std: {np.std(detection_accuracies):.3f}")
        print(
            f"  % better than random: {sum(a > 0.5 for a in detection_accuracies) / len(detection_accuracies) * 100:.1f}%")

    return results


def compare_logit_distributions(log_dir: str = "./logs/logit_detection"):
    """Visualize the difference between truth and lie logit distributions"""

    # Load results
    results_path = Path(log_dir) / "results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Aggregate logit distributions
    truth_probs_all = []
    lie_probs_all = []

    for sample in results.get('samples', []):
        metadata = sample.get('metadata', {})

        truth_dist = metadata.get('truth_logit_distribution', {})
        lie_dist = metadata.get('lie_logit_distribution', {})

        if 'avg_probs' in truth_dist:
            truth_probs_all.append(truth_dist['avg_probs'])

        if 'avg_probs' in lie_dist:
            lie_probs_all.append(lie_dist['avg_probs'])

    if truth_probs_all and lie_probs_all:
        # Compute average distributions
        avg_truth_probs = np.mean(truth_probs_all, axis=0)
        avg_lie_probs = np.mean(lie_probs_all, axis=0)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Average distributions
        x = np.arange(10)
        width = 0.35

        ax1.bar(x - width / 2, avg_truth_probs, width, label='Truth', alpha=0.8, color='blue')
        ax1.bar(x + width / 2, avg_lie_probs, width, label='Lie', alpha=0.8, color='red')
        ax1.set_xlabel('Digit')
        ax1.set_ylabel('Average Probability (from logits)')
        ax1.set_title('Average Logit-Derived Distributions')
        ax1.set_xticks(x)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Difference
        diff = avg_lie_probs - avg_truth_probs
        colors = ['red' if d < 0 else 'green' for d in diff]
        ax2.bar(x, diff, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Digit')
        ax2.set_ylabel('Probability Difference (Lie - Truth)')
        ax2.set_title('Logit Distribution Differences')
        ax2.set_xticks(x)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('logit_distributions_comparison.png', dpi=150)
        plt.show()

        # Print which digits show biggest differences
        digit_diffs = [(i, diff[i]) for i in range(10)]
        digit_diffs.sort(key=lambda x: abs(x[1]), reverse=True)

        print("\n=== Digit-Level Differences (Lie - Truth) ===")
        for digit, diff_val in digit_diffs[:5]:
            print(f"Digit {digit}: {diff_val:+.4f}")


def run_logit_evaluation():
    """Run the complete logit-based evaluation"""

    print("Running logit-based lie detection evaluation...")

    # Create task
    task = create_logit_lie_detection_task(
        dataset_name="high_school_physics",
        num_samples=100,
        num_signature_samples=1
    )

    # Run evaluation
    results = eval(
        task,
        model="openai/gpt-4o-mini",
        log_dir="./logs/logit_detection"
    )

    #print(f"\nOverall accuracy: {results[0].results.metrics['accuracy']}")
    #print(f"Mean score: {results[0].results.metrics['mean']}")

    # Analyze results
    print("\nAnalyzing results...")
    analyze_logit_results()

    # Compare distributions
    print("\nComparing logit distributions...")
    compare_logit_distributions()

    return results


if __name__ == "__main__":
    results = run_logit_evaluation()