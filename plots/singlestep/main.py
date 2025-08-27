#!/usr/bin/env python3
"""
Comprehensive Self-Sycophancy Analysis with Multiple Scorers
Generates vertical bar charts showing various sycophancy metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

from inspect_ai.analysis import samples_df

warnings.filterwarnings('ignore')

# Try different import methods
try:
    from inspect_ai.log import read_eval_log

    LOG_READER_AVAILABLE = True
except ImportError:
    LOG_READER_AVAILABLE = False
    print("Warning: read_eval_log not available")

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color schemes
DOMAIN_COLORS = {
    "harmfulness": "#16A085",
    "correctness": "#D35400"
}


def parse_model_info(model_name):
    """Extract model family and clean name from model path"""
    if not model_name:
        return "other", "unknown"

    model_short = model_name.split('/')[-1] if '/' in model_name else model_name

    model_lower = model_name.lower()
    if "gpt" in model_lower:
        family = "gpt"
    elif "claude" in model_lower:
        family = "claude"
    elif "gemini" in model_lower:
        family = "gemini"
    elif "gemma" in model_lower:
        family = "gemma"
    else:
        family = "other"

    return family, model_short


def parse_task_info(task_name):
    """Parse task name to extract domain and dataset"""
    if not task_name:
        return "unknown", "unknown"

    task_clean = task_name.replace("-", "_")
    parts = task_clean.split("_")

    domain = "unknown"
    dataset = "unknown"

    if "harmfulness" in task_clean:
        domain = "harmfulness"
        idx = parts.index("harmfulness") if "harmfulness" in parts else -1
        if idx >= 0 and idx < len(parts) - 1:
            dataset = "_".join(parts[idx + 1:])
    elif "correctness" in task_clean:
        domain = "correctness"
        idx = parts.index("correctness") if "correctness" in parts else -1
        if idx >= 0 and idx < len(parts) - 1:
            dataset = "_".join(parts[idx + 1:])

    dataset = dataset.replace("_comprehensive", "").strip("_")

    return domain, dataset


def load_samples_using_log_reader(log_dir):
    """Load samples using the official log reader if available"""

    if not LOG_READER_AVAILABLE:
        return pd.DataFrame()

    print("  Trying to load with read_eval_log...")
    log_path = Path(log_dir)
    all_samples = []

    for file_path in list(log_path.glob("*.eval")):
        try:
            log = read_eval_log(str(file_path))

            task = log.eval.task if hasattr(log.eval, 'task') else ''
            model = log.eval.model if hasattr(log.eval, 'model') else ''

            if hasattr(log, 'samples'):
                for sample in log.samples:
                    sample_dict = {
                        'task': task,
                        'model': model
                    }

                    # Extract all attributes
                    for attr in dir(sample):
                        if not attr.startswith('_'):
                            value = getattr(sample, attr)
                            if not callable(value):
                                sample_dict[attr] = value

                    all_samples.append(sample_dict)

        except Exception as e:
            print(e)
            continue

    df = pd.DataFrame(all_samples)
    print(f"  Loaded {len(df)} samples using log reader")
    return df


def load_samples_direct_debug(log_dir):
    """Direct loading with better debugging"""

    print("  Loading directly from JSON files...")
    log_path = Path(log_dir)

    all_samples = []

    # Try both .json and .eval files
    all_files = list(log_path.glob("*.json")) + list(log_path.glob("*.eval"))
    print(f"  Found {len(all_files)} log files")

    # Debug: Look at structure of first file
    if all_files:
        print("\n  Examining first log file structure...")
        try:
            with open(all_files[0], 'r') as f:
                first_data = json.load(f)
                print(f"    Top-level keys: {list(first_data.keys())[:10]}")

                # Check different possible structures
                if 'eval' in first_data:
                    print(f"    'eval' keys: {list(first_data['eval'].keys())[:10]}")
                if 'samples' in first_data:
                    print(f"    Number of samples: {len(first_data['samples'])}")
                    if first_data['samples']:
                        print(f"    First sample keys: {list(first_data['samples'][0].keys())[:20]}")

                        # Check for score keys
                        score_keys = [k for k in first_data['samples'][0].keys() if 'score' in k.lower()]
                        print(f"    Score-related keys: {score_keys[:10]}")

                # Alternative structure - direct samples
                if 'results' in first_data:
                    print(f"    'results' keys: {list(first_data.get('results', {}).keys())[:10]}")

        except Exception as e:
            print(f"    Error examining file: {e}")

    # Process all files
    for i, file_path in enumerate(all_files):
        if i % 10 == 0:
            print(f"    Processing file {i + 1}/{len(all_files)}...")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

                # Extract eval metadata
                eval_info = data.get('eval', {})
                task = eval_info.get('task', '')
                model = eval_info.get('model', '')

                # Try to find samples in different locations
                samples = []

                # Standard location
                if 'samples' in data:
                    samples = data['samples']
                # Alternative location
                elif 'results' in data and 'samples' in data['results']:
                    samples = data['results']['samples']
                # Direct array
                elif isinstance(data, list):
                    samples = data

                # Process each sample
                for sample in samples:
                    if isinstance(sample, dict):
                        sample_record = {
                            'task': task,
                            'model': model
                        }

                        # Add all keys from sample
                        for key, value in sample.items():
                            sample_record[key] = value

                        all_samples.append(sample_record)

        except json.JSONDecodeError:
            # Try reading as eval file (different format)
            try:
                with open(file_path, 'rb') as f:
                    # Eval files might be binary or have different encoding
                    content = f.read()
                    # Try to decode and parse
                    if content.startswith(b'{'):
                        data = json.loads(content.decode('utf-8'))
                        # Process same as above
                        eval_info = data.get('eval', {})
                        task = eval_info.get('task', '')
                        model = eval_info.get('model', '')
                        samples = data.get('samples', [])

                        for sample in samples:
                            if isinstance(sample, dict):
                                sample_record = {'task': task, 'model': model}
                                sample_record.update(sample)
                                all_samples.append(sample_record)
            except:
                pass

        except Exception as e:
            continue

    df = pd.DataFrame(all_samples)
    print(f"  Extracted {len(df)} samples total")

    if not df.empty:
        print(f"  Columns found: {list(df.columns)[:20]}")
        score_cols = [col for col in df.columns if 'score' in col.lower()]
        print(f"  Score columns: {score_cols[:10]}")

    return df


def extract_comprehensive_metrics(samples):
    """Extract all scorer metrics and metadata"""

    if samples.empty:
        return pd.DataFrame()

    print("\n  Processing samples...")
    results = []

    # Find all score columns
    score_columns = [col for col in samples.columns if col.startswith('score_')]
    print(f"  Found {len(score_columns)} score columns")

    if score_columns:
        print(f"  Score columns: {score_columns[:5]}...")  # Show first 5

    for idx, row in samples.iterrows():
        task = row.get('task', '')
        model = row.get('model', '')
        domain, dataset = parse_task_info(task)
        model_family, model_short = parse_model_info(model)

        record = {
            'task': task,
            'model': model,
            'model_short': model_short,
            'model_family': model_family,
            'domain': domain,
            'dataset': dataset
        }

        # Add all score columns
        for scorer in row['scores']:
            value = row.get(scorer, 0)
            try:
                if pd.notna(value) and value != '' and value is not None:
                    record[scorer] = float(value)
                else:
                    record[scorer] = 0.0
            except (TypeError, ValueError):
                record[scorer] = 0.0

        # Extract metadata if available
        metadata = row.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        # Extract ex-ante/ex-post scores
        if isinstance(metadata, dict):
            if domain == "correctness":
                initial_scores = metadata.get('initial_correctness_scores', {})
                chosen_option = metadata.get('chosen_option')
                new_score = metadata.get('new_correctness_score')

                if chosen_option and initial_scores:
                    option_key = str(chosen_option)
                    if option_key in initial_scores:
                        try:
                            record['initial_score'] = float(initial_scores[option_key])
                            record['new_score'] = float(new_score) if new_score else record['initial_score']
                            record['score_shift'] = record['new_score'] - record['initial_score']

                            if abs(record['score_shift']) > 10:
                                continue
                        except:
                            pass

            elif domain == "harmfulness":
                initial_scores = metadata.get('initial_risk_scores', {})
                chosen_option = metadata.get('chosen_option')
                new_score = metadata.get('new_risk_score')

                if chosen_option and initial_scores:
                    option_key = str(chosen_option)
                    if option_key in initial_scores:
                        try:
                            record['initial_score'] = float(initial_scores[option_key])
                            record['new_score'] = float(new_score) if new_score else record['initial_score']
                            record['score_shift'] = record['new_score'] - record['initial_score']

                            if abs(record['score_shift']) > 10:
                                continue
                        except:
                            pass

        results.append(record)

    df = pd.DataFrame(results)
    print(f"  Created {len(df)} processed records")

    # Summary of what we found
    if not df.empty and score_columns:
        print("\n  Score summary:")
        for scorer in score_columns[:5]:  # Show first 5
            if scorer in df.columns:
                non_zero = (df[scorer] != 0).sum()
                if non_zero > 0:
                    mean_val = df[df[scorer] != 0][scorer].mean()
                    print(f"    {scorer[:40]:40s}: {non_zero:4d} non-zero, mean={mean_val:.4f}")

    return df


def create_basic_charts(df, output_dir="plots_comprehensive"):
    """Create basic charts even with limited data"""

    if df.empty:
        print("No data to plot!")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find available score columns
    score_cols = [col for col in df.columns if col.startswith('score_')]

    if not score_cols:
        print("No score columns found to plot!")
        return

    print(f"\n  Creating charts with {len(score_cols)} score columns...")

    # 1. Simple bar chart of available scores
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate means for each scorer
    scorer_data = []
    for scorer in score_cols[:15]:  # Limit to 15 for readability
        non_zero_data = df[df[scorer] != 0][scorer]
        if len(non_zero_data) > 0:
            scorer_data.append({
                'name': scorer.replace('score_', '').replace('_', ' ').replace('scorer', '').strip(),
                'mean': non_zero_data.mean(),
                'std': non_zero_data.std(),
                'count': len(non_zero_data)
            })

    if scorer_data:
        scorer_df = pd.DataFrame(scorer_data)
        scorer_df = scorer_df.sort_values('mean', ascending=False)

        x_pos = np.arange(len(scorer_df))
        colors = plt.cm.Set3(np.linspace(0, 1, len(scorer_df)))

        bars = ax.bar(x_pos, scorer_df['mean'], yerr=scorer_df['std'],
                      capsize=5, color=colors, alpha=0.7, edgecolor='black')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(scorer_df['name'], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title('Self-Sycophancy Scores by Metric', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value and count labels
        for i, (bar, row) in enumerate(zip(bars, scorer_df.iterrows())):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + row[1]['std'] + 0.001,
                    f'{row[1]["mean"]:.3f}\n(n={row[1]["count"]})',
                    ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        plt.savefig(output_dir / 'scorer_means.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Created scorer_means.png")

    # 2. By domain if available
    if 'domain' in df.columns:
        domains = [d for d in df['domain'].unique() if d != 'unknown']

        if domains and score_cols:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Pick first available score column
            scorer = score_cols[0]

            domain_stats = []
            for domain in domains:
                domain_data = df[(df['domain'] == domain) & (df[scorer] != 0)]
                if len(domain_data) > 0:
                    domain_stats.append({
                        'domain': domain,
                        'mean': domain_data[scorer].mean(),
                        'std': domain_data[scorer].std(),
                        'count': len(domain_data)
                    })

            if domain_stats:
                domain_df = pd.DataFrame(domain_stats)

                x_pos = np.arange(len(domain_df))
                colors = [DOMAIN_COLORS.get(d, '#95A5A6') for d in domain_df['domain']]

                bars = ax.bar(x_pos, domain_df['mean'], yerr=domain_df['std'],
                              capsize=5, color=colors, alpha=0.7, edgecolor='black')

                ax.set_xticks(x_pos)
                ax.set_xticklabels(domain_df['domain'].str.title(), fontsize=11)
                ax.set_ylabel('Average Score', fontsize=12)
                ax.set_title(f'Scores by Domain\n({scorer.replace("score_", "").replace("_", " ")})',
                             fontsize=14, fontweight='bold')
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.grid(True, alpha=0.3, axis='y')

                # Add labels
                for bar, row in zip(bars, domain_df.iterrows()):
                    ax.text(bar.get_x() + bar.get_width() / 2.,
                            bar.get_height() + row[1]['std'] + 0.001,
                            f'{row[1]["mean"]:.3f}\n(n={row[1]["count"]})',
                            ha='center', va='bottom', fontsize=9)

                plt.tight_layout()
                plt.savefig(output_dir / 'domain_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    Created domain_comparison.png")

    # 3. Ex-ante/Ex-post if available
    if 'initial_score' in df.columns and 'new_score' in df.columns:
        valid_data = df.dropna(subset=['initial_score', 'new_score'])

        if not valid_data.empty:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Overall comparison
            means = [valid_data['initial_score'].mean(), valid_data['new_score'].mean()]
            stds = [valid_data['initial_score'].std(), valid_data['new_score'].std()]

            x_pos = np.arange(2)
            colors = ['#3498DB', '#E74C3C']

            bars = ax.bar(x_pos, means, yerr=stds, capsize=8,
                          color=colors, alpha=0.7, edgecolor='black', linewidth=2)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(['Ex-Ante\n(Initial)', 'Ex-Post\n(After Choice)'], fontsize=12)
            ax.set_ylabel('Average Score', fontsize=12)
            ax.set_title('Overall Ex-Ante vs Ex-Post Comparison', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 0.1,
                        f'{mean:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

            # Add shift annotation
            shift = means[1] - means[0]
            ax.text(0.5, max(means) + max(stds) + 0.5,
                    f'Δ = {shift:+.3f}', ha='center', fontsize=12,
                    color='red' if abs(shift) > 0.1 else 'gray',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.text(0.5, -0.5, f'N = {len(valid_data)}', ha='center', fontsize=10, alpha=0.6)

            plt.tight_layout()
            plt.savefig(output_dir / 'exante_expost_overall.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Created exante_expost_overall.png")

    print(f"\n  Charts saved to {output_dir}/")


def main():
    """Main analysis pipeline"""

    log_dir = "../../logs/self-sycophancy-single/all-1"
    print(f"Loading samples from {log_dir}...")

    # Check if directory exists
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"\nERROR: Directory does not exist: {log_path.absolute()}")
        return

    # # Try different loading methods
    samples = pd.DataFrame()
    #
    # # Method 1: Try log reader if available
    if LOG_READER_AVAILABLE:
        samples = load_samples_using_log_reader(log_dir)
    #
    # # Method 2: Direct JSON loading with debugging
    # if samples.empty:
    #     samples = load_samples_direct_debug(log_dir)
    #samples, errors = samples_df("../../logs/self-sycophancy-single/all-1", strict=False)

    if samples.empty:
        print("\nERROR: Could not load any samples!")
        print("Please check that the log files contain valid data.")
        return

    # Extract metrics
    df = extract_comprehensive_metrics(samples)

    if df.empty:
        print("ERROR: No data extracted from samples!")
        # Try to work with raw samples
        df = samples

    # Create whatever charts we can with available data
    create_basic_charts(df)

    # Save data
    df.to_csv("sycophancy_analysis.csv", index=False)
    print(f"\nData saved to sycophancy_analysis.csv")

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    if 'model' in df.columns:
        print(f"Unique models: {df['model'].nunique()}")
    if 'task' in df.columns:
        print(f"Unique tasks: {df['task'].nunique()}")

    score_cols = [col for col in df.columns if col.startswith('score_')]
    if score_cols:
        print(f"Score columns available: {len(score_cols)}")

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()