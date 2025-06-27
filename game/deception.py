"""
Utilities for analyzing deception detection results and building training datasets
"""

import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from inspect_ai.log import EvalLog, EvalSample, read_eval_log


class DeceptionDatasetBuilder:
    """Build and analyze deception detection datasets from evaluation logs"""

    def __init__(self, log_path: str):
        """Initialize with evaluation log path"""
        self.log = read_eval_log(log_path)
        self.quality_deceptions = []
        self.extract_deceptions()

    def extract_deceptions(self):
        """Extract all quality deceptions from the evaluation log"""
        for sample in self.log.samples:
            if "quality_deceptions" in sample.metadata:
                self.quality_deceptions.extend(sample.metadata["quality_deceptions"])

    def build_training_dataset(self, min_confidence: float = 70.0) -> pd.DataFrame:
        """Build a training dataset of high-quality deceptions"""

        records = []
        for deception in self.quality_deceptions:
            if deception["red_confidence"] >= min_confidence:
                records.append({
                    "text": deception["utterance"],
                    "label": "deception",
                    "confidence_score": deception["red_confidence"],
                    "deception_type": deception["deception_type"],
                    "domain": deception["context"]["scenario_domain"],
                    "blue_role": deception["context"]["blue_role"],
                    "red_role": deception["context"]["red_role"],
                    "has_hedging": deception["linguistic_features"]["has_hedging"],
                    "has_specifics": deception["linguistic_features"]["has_specifics"],
                    "word_count": deception["linguistic_features"]["length"],
                    "actual_truth": deception["actual_truth"],
                    "red_reasoning": deception["red_reasoning"]
                })

        return pd.DataFrame(records)

    def analyze_deception_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in successful deceptions"""

        analysis = {
            "total_deceptions": len(self.quality_deceptions),
            "deception_types": defaultdict(int),
            "avg_confidence_by_type": defaultdict(list),
            "linguistic_patterns": defaultdict(int),
            "domain_distribution": defaultdict(int)
        }

        for deception in self.quality_deceptions:
            dtype = deception["deception_type"]
            analysis["deception_types"][dtype] += 1
            analysis["avg_confidence_by_type"][dtype].append(deception["red_confidence"])

            if deception["linguistic_features"]["has_hedging"]:
                analysis["linguistic_patterns"]["hedging"] += 1
            if deception["linguistic_features"]["has_specifics"]:
                analysis["linguistic_patterns"]["specifics"] += 1

            analysis["domain_distribution"][deception["context"]["scenario_domain"]] += 1

        # Calculate averages
        for dtype, confidences in analysis["avg_confidence_by_type"].items():
            analysis["avg_confidence_by_type"][dtype] = sum(confidences) / len(confidences)

        return dict(analysis)

    def plot_analysis(self, output_dir: str = "./deception_analysis"):
        """Generate visualization plots for the deception analysis"""

        Path(output_dir).mkdir(exist_ok=True)

        # 1. Confidence distribution
        plt.figure(figsize=(10, 6))
        confidences = [d["red_confidence"] for d in self.quality_deceptions]
        plt.hist(confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel("Red's Confidence (%)")
        plt.ylabel("Count")
        plt.title("Distribution of Confidence Scores for Successful Deceptions")
        plt.savefig(f"{output_dir}/confidence_distribution.png")
        plt.close()

        # 2. Deception types
        plt.figure(figsize=(10, 6))
        deception_types = [d["deception_type"] for d in self.quality_deceptions]
        type_counts = pd.Series(deception_types).value_counts()
        type_counts.plot(kind='bar', color='green', alpha=0.7)
        plt.xlabel("Deception Type")
        plt.ylabel("Count")
        plt.title("Distribution of Deception Types")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/deception_types.png")
        plt.close()

        # 3. Linguistic features
        plt.figure(figsize=(10, 6))
        hedging_count = sum(1 for d in self.quality_deceptions if d["linguistic_features"]["has_hedging"])
        specific_count = sum(1 for d in self.quality_deceptions if d["linguistic_features"]["has_specifics"])

        features = ['Has Hedging', 'Has Specifics', 'Neither']
        counts = [
            hedging_count,
            specific_count,
            len(self.quality_deceptions) - max(hedging_count, specific_count)
        ]

        plt.pie(counts, labels=features, autopct='%1.1f%%', startangle=90)
        plt.title("Linguistic Features in Successful Deceptions")
        plt.savefig(f"{output_dir}/linguistic_features.png")
        plt.close()

    def export_dataset(self, output_path: str, format: str = "jsonl"):
        """Export the deception dataset in various formats"""

        df = self.build_training_dataset()

        if format == "jsonl":
            df.to_json(output_path, orient='records', lines=True)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Exported {len(df)} deception examples to {output_path}")

    def get_calibration_metrics(self) -> Dict[str, float]:
        """Calculate Red agent calibration metrics across all games"""

        calibration_scores = []
        for sample in self.log.samples:
            if sample.scores:
                for score_name, score in sample.scores.items():
                    if "calibration" in score_name:
                        calibration_scores.append(score.value)

        if calibration_scores:
            return {
                "mean_calibration": sum(calibration_scores) / len(calibration_scores),
                "min_calibration": min(calibration_scores),
                "max_calibration": max(calibration_scores),
                "std_calibration": pd.Series(calibration_scores).std()
            }
        return {}


class DeceptionQualityAnalyzer:
    """Analyze the quality of deceptions based on various criteria"""

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def score_deception_sophistication(self) -> pd.DataFrame:
        """Score each deception on sophistication metrics"""

        # Add sophistication scores
        self.dataset['sophistication_score'] = 0.0

        # Factors that increase sophistication
        self.dataset.loc[self.dataset['deception_type'] == 'partial_truth', 'sophistication_score'] += 0.3
        self.dataset.loc[self.dataset['has_specifics'] == True, 'sophistication_score'] += 0.2
        self.dataset.loc[self.dataset['confidence_score'] > 80, 'sophistication_score'] += 0.3
        self.dataset.loc[self.dataset['word_count'] > 20, 'sophistication_score'] += 0.2

        # Normalize to 0-1
        self.dataset['sophistication_score'] = self.dataset['sophistication_score'].clip(0, 1)

        return self.dataset

    def get_top_deceptions(self, n: int = 10) -> pd.DataFrame:
        """Get the top N most sophisticated deceptions"""

        scored_df = self.score_deception_sophistication()
        return scored_df.nlargest(n, 'sophistication_score')

    def analyze_by_domain(self) -> Dict[str, Dict[str, Any]]:
        """Analyze deception patterns by domain"""

        domain_analysis = {}

        for domain in self.dataset['domain'].unique():
            domain_data = self.dataset[self.dataset['domain'] == domain]

            domain_analysis[domain] = {
                'count': len(domain_data),
                'avg_confidence': domain_data['confidence_score'].mean(),
                'deception_types': domain_data['deception_type'].value_counts().to_dict(),
                'avg_word_count': domain_data['word_count'].mean(),
                'hedging_rate': domain_data['has_hedging'].mean()
            }

        return domain_analysis


# Example usage functions

def process_evaluation_logs(log_paths: List[str], output_dir: str = "./deception_datasets"):
    """Process multiple evaluation logs and combine into a master dataset"""

    Path(output_dir).mkdir(exist_ok=True)

    all_deceptions = []

    for log_path in log_paths:
        print(f"Processing {log_path}...")
        builder = DeceptionDatasetBuilder(log_path)

        # Extract dataset
        df = builder.build_training_dataset()
        all_deceptions.append(df)

        # Generate individual analysis
        analysis = builder.analyze_deception_patterns()
        with open(f"{output_dir}/analysis_{Path(log_path).stem}.json", 'w') as f:
            json.dump(analysis, f, indent=2)

    # Combine all datasets
    master_df = pd.concat(all_deceptions, ignore_index=True)

    # Export master dataset
    master_df.to_json(f"{output_dir}/master_deception_dataset.jsonl", orient='records', lines=True)

    # Analyze master dataset
    analyzer = DeceptionQualityAnalyzer(master_df)
    top_deceptions = analyzer.get_top_deceptions(20)
    top_deceptions.to_csv(f"{output_dir}/top_20_deceptions.csv", index=False)

    domain_analysis = analyzer.analyze_by_domain()
    with open(f"{output_dir}/domain_analysis.json", 'w') as f:
        json.dump(domain_analysis, f, indent=2)

    print(f"\nProcessed {len(master_df)} total deceptions")
    print(f"Average confidence: {master_df['confidence_score'].mean():.1f}%")
    print(f"Deception types: {master_df['deception_type'].value_counts().to_dict()}")

    return master_df


def create_lie_detector_training_set(
        deception_dataset: pd.DataFrame,
        truth_dataset: pd.DataFrame,
        output_path: str,
        balance_ratio: float = 1.0
):
    """Create a balanced training set for lie detection models"""

    # Sample truths to match deception count (or use balance_ratio)
    n_deceptions = len(deception_dataset)
    n_truths = int(n_deceptions * balance_ratio)

    truth_sample = truth_dataset.sample(n=min(n_truths, len(truth_dataset)), random_state=42)

    # Combine datasets
    deception_dataset['label'] = 1  # 1 for lies
    truth_sample['label'] = 0  # 0 for truths

    combined = pd.concat([deception_dataset, truth_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    # Export
    combined.to_json(output_path, orient='records', lines=True)

    print(f"Created balanced dataset with {len(combined)} examples")
    print(f"Lies: {(combined['label'] == 1).sum()}, Truths: {(combined['label'] == 0).sum()}")

    return combined


if __name__ == "__main__":
    # Example usage
    log_paths = [
        "./logs/deception_detection_log_1.json",
        "./logs/deception_detection_log_2.json"
    ]

    # Process logs and create dataset
    master_dataset = process_evaluation_logs(log_paths)

    # Create visualizations
    if len(log_paths) > 0:
        builder = DeceptionDatasetBuilder(log_paths[0])
        builder.plot_analysis()