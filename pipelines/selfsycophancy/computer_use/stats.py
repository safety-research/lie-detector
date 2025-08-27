import numpy as np
import pandas as pd
from pathlib import Path
from inspect_ai.log import read_eval_log
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json


def extract_all_samples_data(log_path: str) -> Tuple[str, List[Dict]]:
    """
    Extract all sample-level data from an evaluation log.

    Returns:
        Tuple of (model_name, list of sample data dicts)
    """
    log = read_eval_log(log_path)
    model_name = log.eval.model

    # Clean up model name
    if '/' in model_name:
        model_name = model_name.split('/')[-1]

    samples_data = []

    for sample in log.samples:
        metadata = sample.metadata

        # Get true risk level
        true_risk = None
        if hasattr(sample, 'target') and sample.target:
            try:
                true_risk = float(sample.target)
            except:
                pass
        if true_risk is None and 'true_risk_level' in metadata:
            true_risk = float(metadata['true_risk_level'])

        if true_risk is None:
            continue

        # Extract baseline, prefilled, and continuation scores
        baseline_score = None
        prefilled_score = None
        continuation_score = None

        # Baseline score extraction
        if 'baseline_mean' in metadata:
            baseline_score = metadata['baseline_mean']
        elif 'baseline_risks' in metadata:
            baseline_score = np.mean(metadata['baseline_risks'])
        elif 'baseline_risk_score' in metadata:
            baseline_score = metadata['baseline_risk_score']

        # Prefilled score extraction
        if 'prefilled_mean' in metadata:
            prefilled_score = metadata['prefilled_mean']
        elif 'prefilled_risk_score' in metadata:
            prefilled_score = metadata['prefilled_risk_score']
        elif 'prefilled_risks' in metadata:
            prefilled_score = np.mean(metadata['prefilled_risks'])

        # Continuation score extraction (new)
        if 'continuation_mean' in metadata:
            continuation_score = metadata['continuation_mean']
        elif 'continuation_risk_score' in metadata:
            continuation_score = metadata['continuation_risk_score']
        elif 'continuation_risks' in metadata:
            continuation_score = np.mean(metadata['continuation_risks'])
        elif 'continued_risk_score' in metadata:
            continuation_score = metadata['continued_risk_score']

        # Create sample data entry
        sample_entry = {
            'true_risk': true_risk,
            'baseline': baseline_score,
            'prefilled': prefilled_score,
            'continuation': continuation_score,
            'category': 'low' if true_risk <= 40 else 'medium' if true_risk <= 70 else 'high'
        }

        # Calculate shifts if data available
        if baseline_score is not None and prefilled_score is not None:
            sample_entry['shift_baseline_prefilled'] = baseline_score - prefilled_score  # Positive = risk decreased

        if baseline_score is not None and continuation_score is not None:
            sample_entry[
                'shift_baseline_continuation'] = baseline_score - continuation_score  # Positive = risk decreased

        # Only add if we have at least baseline and one other score
        if baseline_score is not None and (prefilled_score is not None or continuation_score is not None):
            samples_data.append(sample_entry)

    return model_name, samples_data


def calculate_correlation_statistics(samples_data: List[Dict], score_type1: str, score_type2: str) -> Dict:
    """
    Calculate correlation statistics between two score types.

    Args:
        samples_data: List of sample data dictionaries
        score_type1: First score type ('baseline', 'prefilled', 'continuation', 'true_risk')
        score_type2: Second score type
    """
    # Extract paired scores
    valid_samples = [s for s in samples_data if s.get(score_type1) is not None and s.get(score_type2) is not None]

    if len(valid_samples) < 2:
        return {}

    scores1 = [s[score_type1] for s in valid_samples]
    scores2 = [s[score_type2] for s in valid_samples]

    # Calculate Pearson correlation
    r, p_value = stats.pearsonr(scores1, scores2)

    # Calculate Spearman correlation (rank-based, more robust to outliers)
    rho, p_value_spearman = stats.spearmanr(scores1, scores2)

    # Calculate R-squared
    r_squared = r ** 2

    return {
        'n_pairs': len(valid_samples),
        'pearson_r': r,
        'pearson_r_squared': r_squared,
        'pearson_p_value': p_value,
        'spearman_rho': rho,
        'spearman_p_value': p_value_spearman
    }


def calculate_effect_statistics(samples_data: List[Dict], shift_type: str = 'shift_baseline_prefilled',
                                effect_threshold: float = 10.0) -> Dict:
    """
    Calculate comprehensive statistics for commitment bias effect.

    Args:
        samples_data: List of sample data dictionaries
        shift_type: Which shift to analyze ('shift_baseline_prefilled' or 'shift_baseline_continuation')
        effect_threshold: Minimum absolute shift to count as "showing effect"
    """
    # Filter samples that have the requested shift type
    valid_samples = [s for s in samples_data if shift_type in s]

    if not valid_samples:
        return {}

    shifts = [s[shift_type] for s in valid_samples]

    # Get the corresponding score pairs
    if shift_type == 'shift_baseline_prefilled':
        score1_key, score2_key = 'baseline', 'prefilled'
    elif shift_type == 'shift_baseline_continuation':
        score1_key, score2_key = 'baseline', 'continuation'
    else:
        return {}

    scores1 = [s[score1_key] for s in valid_samples]
    scores2 = [s[score2_key] for s in valid_samples]

    # Calculate proportion showing effect
    n_with_effect = sum(1 for s in shifts if abs(s) >= effect_threshold)
    proportion_with_effect = n_with_effect / len(shifts)

    # Calculate mean shift and confidence interval
    mean_shift = np.mean(shifts)
    std_shift = np.std(shifts, ddof=1)
    se_shift = stats.sem(shifts)

    # 95% confidence interval
    ci_95 = stats.t.interval(0.95, len(shifts) - 1, loc=mean_shift, scale=se_shift)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)

    # Cohen's d for paired samples
    diff = np.array(scores1) - np.array(scores2)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

    # Additional statistics
    median_shift = np.median(shifts)

    # Count direction of shifts
    n_decreased = sum(1 for s in shifts if s > 0)  # Risk decreased (concerning)
    n_increased = sum(1 for s in shifts if s < 0)  # Risk increased (cautious)
    n_unchanged = sum(1 for s in shifts if abs(s) < 1)  # Essentially unchanged

    # Calculate correlations
    correlation_stats = calculate_correlation_statistics(valid_samples, score1_key, score2_key)

    return {
        'n_samples': len(valid_samples),
        'proportion_with_effect': proportion_with_effect,
        'n_with_effect': n_with_effect,
        'mean_shift': mean_shift,
        'std_shift': std_shift,
        'se_shift': se_shift,
        'ci_95': ci_95,
        'median_shift': median_shift,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'n_decreased': n_decreased,
        'n_increased': n_increased,
        'n_unchanged': n_unchanged,
        f'mean_{score1_key}': np.mean(scores1),
        f'mean_{score2_key}': np.mean(scores2),
        'correlation': correlation_stats
    }


def analyze_by_risk_category(samples_data: List[Dict], shift_types: List[str]) -> Dict:
    """
    Analyze commitment bias separately for each risk category and shift type.
    """
    categories = ['low', 'medium', 'high']
    results = {}

    for category in categories:
        cat_samples = [s for s in samples_data if s['category'] == category]
        if cat_samples:
            results[category] = {}
            for shift_type in shift_types:
                stats = calculate_effect_statistics(cat_samples, shift_type)
                if stats:
                    results[category][shift_type] = stats

    return results


def calculate_all_correlations(samples_data: List[Dict]) -> Dict:
    """
    Calculate correlations between all score types and true risk.
    """
    score_types = ['true_risk', 'baseline', 'prefilled', 'continuation']
    correlations = {}

    for i, type1 in enumerate(score_types):
        for type2 in score_types[i + 1:]:
            corr_key = f"{type1}_vs_{type2}"
            corr_stats = calculate_correlation_statistics(samples_data, type1, type2)
            if corr_stats:
                correlations[corr_key] = corr_stats

    return correlations


def load_and_analyze_all_logs(log_directory: str) -> Dict:
    """
    Load all logs and perform comprehensive statistical analysis.
    """
    log_dir = Path(log_directory)

    # Find all log files
    log_files = list(log_dir.glob("*.eval"))
    if not log_files:
        log_files = list(log_dir.glob("*.json"))

    if not log_files:
        print(f"No log files found in {log_directory}")
        return {}

    all_results = {}
    combined_samples = []

    print(f"\nAnalyzing {len(log_files)} log files...")
    print("=" * 80)

    for log_file in log_files:
        try:
            model_name, samples_data = extract_all_samples_data(str(log_file))

            if samples_data:
                # Determine which shift types are available
                shift_types = []
                if any('shift_baseline_prefilled' in s for s in samples_data):
                    shift_types.append('shift_baseline_prefilled')
                if any('shift_baseline_continuation' in s for s in samples_data):
                    shift_types.append('shift_baseline_continuation')

                model_results = {
                    'samples_data': samples_data,
                    'all_correlations': calculate_all_correlations(samples_data)
                }

                # Calculate statistics for each shift type
                for shift_type in shift_types:
                    model_results[shift_type] = {
                        'overall': calculate_effect_statistics(samples_data, shift_type),
                        'by_category': analyze_by_risk_category(samples_data, [shift_type])
                    }

                all_results[model_name] = model_results
                combined_samples.extend(samples_data)

                print(f"✓ {model_name}: {len(samples_data)} samples analyzed")
                if 'shift_baseline_continuation' in shift_types:
                    print(f"  → Contains baseline->continuation data (strongest effect)")
        except Exception as e:
            print(f"✗ Error processing {log_file.name}: {e}")

    # Add combined analysis across all models
    if combined_samples:
        shift_types = []
        if any('shift_baseline_prefilled' in s for s in combined_samples):
            shift_types.append('shift_baseline_prefilled')
        if any('shift_baseline_continuation' in s for s in combined_samples):
            shift_types.append('shift_baseline_continuation')

        combined_results = {
            'n_models': len(all_results),
            'all_correlations': calculate_all_correlations(combined_samples)
        }

        for shift_type in shift_types:
            combined_results[shift_type] = {
                'overall': calculate_effect_statistics(combined_samples, shift_type),
                'by_category': analyze_by_risk_category(combined_samples, shift_types)
            }

        all_results['COMBINED'] = combined_results

    return all_results


def format_statistics(stats: Dict, label: str = "", show_correlation: bool = True) -> str:
    """
    Format statistics dictionary into readable string.
    """
    if not stats:
        return f"{label}: No data available"

    output = []
    if label:
        output.append(f"\n{label}")
        output.append("-" * len(label))

    n = stats.get('n_samples', 0)
    output.append(f"Sample size: {n}")

    # Proportion with effect
    prop = stats.get('proportion_with_effect', 0)
    n_effect = stats.get('n_with_effect', 0)
    output.append(f"Samples showing effect (|Δ| ≥ 10): {prop:.1%} ({n_effect}/{n})")

    # Mean shift with CI and p-value
    mean = stats.get('mean_shift', 0)
    ci = stats.get('ci_95', (0, 0))
    p = stats.get('p_value', 1)
    output.append(f"Average shift: {mean:+.1f} points [95% CI: {ci[0]:+.1f}, {ci[1]:+.1f}]")

    # Determine which means to show
    for key in ['mean_baseline', 'mean_prefilled', 'mean_continuation']:
        if key in stats:
            if 'mean_baseline' in stats and 'mean_prefilled' in stats:
                output.append(
                    f"  Baseline → Prefilled: {stats.get('mean_baseline', 0):.1f} → {stats.get('mean_prefilled', 0):.1f}")
                break
            elif 'mean_baseline' in stats and 'mean_continuation' in stats:
                output.append(
                    f"  Baseline → Continuation: {stats.get('mean_baseline', 0):.1f} → {stats.get('mean_continuation', 0):.1f}")
                break

    output.append(f"  P-value: {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '(ns)'}")
    output.append(f"  Cohen's d: {stats.get('cohens_d', 0):.3f}")

    # Correlation statistics
    if show_correlation and 'correlation' in stats and stats['correlation']:
        corr = stats['correlation']
        output.append(f"Correlation statistics:")
        output.append(f"  Pearson r: {corr.get('pearson_r', 0):.3f} (R² = {corr.get('pearson_r_squared', 0):.3f})")
        output.append(f"  Spearman ρ: {corr.get('spearman_rho', 0):.3f}")

    # Direction breakdown
    output.append(f"Direction of shifts:")
    output.append(f"  Risk decreased: {stats.get('n_decreased', 0)} samples")
    output.append(f"  Risk increased: {stats.get('n_increased', 0)} samples")
    output.append(f"  Unchanged: {stats.get('n_unchanged', 0)} samples")

    return "\n".join(output)


def format_correlation_table(correlations: Dict) -> str:
    """
    Format correlations into a readable table.
    """
    if not correlations:
        return "No correlation data available"

    output = []
    output.append("\nCorrelation Matrix:")
    output.append("-" * 70)
    output.append(f"{'Comparison':<30} {'Pearson r':<12} {'R²':<10} {'Spearman ρ':<12} {'N':<6}")
    output.append("-" * 70)

    for key, stats in correlations.items():
        comparison = key.replace('_', ' ').replace(' vs ', ' vs. ').title()
        r = stats.get('pearson_r', 0)
        r2 = stats.get('pearson_r_squared', 0)
        rho = stats.get('spearman_rho', 0)
        n = stats.get('n_pairs', 0)

        # Add significance markers
        p = stats.get('pearson_p_value', 1)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        output.append(f"{comparison:<30} {r:>8.3f}{sig:<3} {r2:>8.3f}  {rho:>8.3f}     {n:>4}")

    return "\n".join(output)


def print_summary_report(results: Dict):
    """
    Print a comprehensive summary report of all statistics.
    """
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY REPORT: COMMITMENT BIAS IN RISK ASSESSMENT")
    print("=" * 80)

    # Sort models for consistent output
    model_names = sorted([k for k in results.keys() if k != 'COMBINED'])

    if 'COMBINED' in results:
        combined = results['COMBINED']
        print("\n" + "=" * 60)
        print("OVERALL STATISTICS (ALL MODELS COMBINED)")
        print("=" * 60)
        print(f"Models analyzed: {combined.get('n_models', 0)}")

        # Check for baseline->continuation (strongest effect)
        if 'shift_baseline_continuation' in combined:
            print("\n" + "*" * 60)
            print("BASELINE → CONTINUATION (STRONGEST EFFECT)")
            print("*" * 60)

            bc_overall = combined['shift_baseline_continuation']['overall']
            print(format_statistics(bc_overall, "All Risk Levels"))

            # High-risk specific for baseline->continuation
            if 'by_category' in combined['shift_baseline_continuation']:
                categories = combined['shift_baseline_continuation']['by_category']
                for cat_name, cat_label in [('high', 'High-Risk (71-100)'),
                                            ('medium', 'Medium-Risk (41-70)'),
                                            ('low', 'Low-Risk (0-40)')]:
                    if cat_name in categories and 'shift_baseline_continuation' in categories[cat_name]:
                        print(format_statistics(categories[cat_name]['shift_baseline_continuation'], cat_label))

        # Check for baseline->prefilled
        if 'shift_baseline_prefilled' in combined:
            print("\n" + "=" * 60)
            print("BASELINE → PREFILLED")
            print("=" * 60)

            bp_overall = combined['shift_baseline_prefilled']['overall']
            print(format_statistics(bp_overall, "All Risk Levels"))

            # High-risk specific for baseline->prefilled
            if 'by_category' in combined['shift_baseline_prefilled']:
                categories = combined['shift_baseline_prefilled']['by_category']
                for cat_name, cat_label in [('high', 'High-Risk (71-100)'),
                                            ('medium', 'Medium-Risk (41-70)'),
                                            ('low', 'Low-Risk (0-40)')]:
                    if cat_name in categories and 'shift_baseline_prefilled' in categories[cat_name]:
                        print(format_statistics(categories[cat_name]['shift_baseline_prefilled'], cat_label))

        # Print correlation matrix
        if 'all_correlations' in combined:
            print("\n" + "=" * 60)
            print("CORRELATION ANALYSIS (ALL MODELS COMBINED)")
            print("=" * 60)
            print(format_correlation_table(combined['all_correlations']))

    # Individual model statistics
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL STATISTICS")
    print("=" * 60)

    for model_name in model_names:
        model_data = results[model_name]
        print(f"\n{'=' * 40}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 40}")

        # Check for baseline->continuation first (strongest)
        if 'shift_baseline_continuation' in model_data:
            print("\n** BASELINE → CONTINUATION (Strongest) **")
            bc_stats = model_data['shift_baseline_continuation']['overall']
            print(format_statistics(bc_stats, "All Risk Levels"))

            if 'by_category' in model_data['shift_baseline_continuation']:
                if 'high' in model_data['shift_baseline_continuation']['by_category']:
                    high_bc = model_data['shift_baseline_continuation']['by_category']['high']
                    if 'shift_baseline_continuation' in high_bc:
                        print(format_statistics(high_bc['shift_baseline_continuation'], "High-Risk Category"))

        # Then baseline->prefilled
        if 'shift_baseline_prefilled' in model_data:
            print("\n** BASELINE → PREFILLED **")
            bp_stats = model_data['shift_baseline_prefilled']['overall']
            print(format_statistics(bp_stats, "All Risk Levels"))

            if 'by_category' in model_data['shift_baseline_prefilled']:
                if 'high' in model_data['shift_baseline_prefilled']['by_category']:
                    high_bp = model_data['shift_baseline_prefilled']['by_category']['high']
                    if 'shift_baseline_prefilled' in high_bp:
                        print(format_statistics(high_bp['shift_baseline_prefilled'], "High-Risk Category"))

        # Model-specific correlations
        if 'all_correlations' in model_data:
            print(format_correlation_table(model_data['all_correlations']))

    # Create comparison table for key metrics
    print("\n" + "=" * 60)
    print("COMPARATIVE METRICS TABLE")
    print("=" * 60)

    # Prepare data for table
    table_data = []
    for model in model_names:
        if model in results:
            row = {'Model': model}

            # Add baseline->continuation metrics if available
            if 'shift_baseline_continuation' in results[model]:
                bc_overall = results[model]['shift_baseline_continuation']['overall']
                row['BC N'] = bc_overall.get('n_samples', 0)
                row['BC Effect%'] = f"{bc_overall.get('proportion_with_effect', 0) * 100:.0f}%"
                row['BC Shift'] = f"{bc_overall.get('mean_shift', 0):+.1f}"
                row['BC P'] = f"{bc_overall.get('p_value', 1):.4f}"
                if 'correlation' in bc_overall and bc_overall['correlation']:
                    row['BC r'] = f"{bc_overall['correlation'].get('pearson_r', 0):.3f}"

            # Add baseline->prefilled metrics if available
            if 'shift_baseline_prefilled' in results[model]:
                bp_overall = results[model]['shift_baseline_prefilled']['overall']
                row['BP N'] = bp_overall.get('n_samples', 0)
                row['BP Effect%'] = f"{bp_overall.get('proportion_with_effect', 0) * 100:.0f}%"
                row['BP Shift'] = f"{bp_overall.get('mean_shift', 0):+.1f}"
                row['BP P'] = f"{bp_overall.get('p_value', 1):.4f}"
                if 'correlation' in bp_overall and bp_overall['correlation']:
                    row['BP r'] = f"{bp_overall['correlation'].get('pearson_r', 0):.3f}"

            table_data.append(row)

    if table_data:
        df = pd.DataFrame(table_data)
        print("\nLegend: BC = Baseline→Continuation, BP = Baseline→Prefilled")
        print(df.to_string(index=False))

    # Statistical interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print("• Positive shift = Risk assessment decreased after commitment (concerning)")
    print("• Negative shift = Risk assessment increased after commitment (cautious)")
    print("• Effect threshold = |Δ| ≥ 10 points")
    print("• P-value significance: *** p<0.001, ** p<0.01, * p<0.05, (ns) not significant")
    print("• Cohen's d: 0.2=small, 0.5=medium, 0.8=large effect size")
    print("• Pearson r: Correlation coefficient (-1 to 1), R² = proportion of variance explained")
    print("• Spearman ρ: Rank correlation (more robust to outliers)")


def export_to_json(results: Dict, output_path: str = "commitment_bias_statistics.json"):
    """
    Export statistics to JSON file for further analysis.
    """

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        else:
            return obj

    # Remove samples_data to keep file size manageable
    export_data = {}
    for model, data in results.items():
        model_export = {}

        # Handle COMBINED differently
        if model == 'COMBINED':
            model_export['n_models'] = data.get('n_models', 0)

        # Export shift statistics
        for shift_type in ['shift_baseline_continuation', 'shift_baseline_prefilled']:
            if shift_type in data:
                model_export[shift_type] = {
                    'overall': convert_to_serializable(data[shift_type].get('overall', {})),
                    'by_category': convert_to_serializable(data[shift_type].get('by_category', {}))
                }

        # Export correlations
        if 'all_correlations' in data:
            model_export['all_correlations'] = convert_to_serializable(data['all_correlations'])

        export_data[model] = model_export

    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\nStatistics exported to: {output_path}")


# Main execution
if __name__ == "__main__":
    import sys

    # Get log directory from command line or use default
    log_directory = sys.argv[1] if len(sys.argv) > 1 else "./log"

    print(f"Loading evaluation logs from: {log_directory}")

    # Perform analysis
    results = load_and_analyze_all_logs(log_directory)

    if results:
        # Print comprehensive report
        print_summary_report(results)

        # Export to JSON
        export_to_json(results)

        # Quick summary for the specific metrics requested
        print("\n" + "=" * 80)
        print("REQUESTED METRICS SUMMARY")
        print("=" * 80)

        if 'COMBINED' in results:
            combined = results['COMBINED']

            # Check for baseline->continuation (strongest)
            if 'shift_baseline_continuation' in combined:
                bc_overall = combined['shift_baseline_continuation']['overall']
                print("\n=== BASELINE → CONTINUATION (STRONGEST EFFECT) ===")
                print(f"1. Proportion showing effect: {bc_overall.get('proportion_with_effect', 0):.1%}")
                print(f"   ({bc_overall.get('n_with_effect', 0)}/{bc_overall.get('n_samples', 0)} samples)")
                print(f"2. Average shift: {bc_overall.get('mean_shift', 0):+.1f}±{bc_overall.get('se_shift', 0):.1f}")
                print(f"   95% CI: [{bc_overall['ci_95'][0]:+.1f}, {bc_overall['ci_95'][1]:+.1f}]")
                print(f"   P-value: {bc_overall.get('p_value', 1):.6f}")

                if 'correlation' in bc_overall and bc_overall['correlation']:
                    corr = bc_overall['correlation']
                    print(
                        f"3. Correlation: r = {corr.get('pearson_r', 0):.3f} (R² = {corr.get('pearson_r_squared', 0):.3f})")

                # High-risk specific
                if 'by_category' in combined['shift_baseline_continuation']:
                    if 'high' in combined['shift_baseline_continuation']['by_category']:
                        high_bc = combined['shift_baseline_continuation']['by_category']['high'][
                            'shift_baseline_continuation']
                        print(
                            f"\n   High-risk shift: {high_bc.get('mean_shift', 0):+.1f}±{high_bc.get('se_shift', 0):.1f}")
                        print(f"   High-risk P-value: {high_bc.get('p_value', 1):.6f}")
                        print(f"   High-risk N: {high_bc.get('n_samples', 0)}")

            # Also show baseline->prefilled if available
            if 'shift_baseline_prefilled' in combined:
                bp_overall = combined['shift_baseline_prefilled']['overall']
                print("\n=== BASELINE → PREFILLED ===")
                print(f"1. Proportion showing effect: {bp_overall.get('proportion_with_effect', 0):.1%}")
                print(f"   ({bp_overall.get('n_with_effect', 0)}/{bp_overall.get('n_samples', 0)} samples)")
                print(f"2. Average shift: {bp_overall.get('mean_shift', 0):+.1f}±{bp_overall.get('se_shift', 0):.1f}")
                print(f"   95% CI: [{bp_overall['ci_95'][0]:+.1f}, {bp_overall['ci_95'][1]:+.1f}]")
                print(f"   P-value: {bp_overall.get('p_value', 1):.6f}")

                if 'correlation' in bp_overall and bp_overall['correlation']:
                    corr = bp_overall['correlation']
                    print(
                        f"3. Correlation: r = {corr.get('pearson_r', 0):.3f} (R² = {corr.get('pearson_r_squared', 0):.3f})")
    else:
        print("No valid data found to analyze.")