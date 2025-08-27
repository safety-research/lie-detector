#!/usr/bin/env python3
"""
Compute proper within-sample z-scores using position_bias_scorer.

KEY INSIGHT:
- position_bias_scorer returns the VARIANCE of SSIs across forced positions
- Therefore: within_sample_std = sqrt(position_bias)
- And: z_score = calibrated_effect / within_sample_std

This is the correct way to compute significance from your available data!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from inspect_ai.analysis import samples_df, evals_df

def extract_and_compute_zscores():
    """
    Extract sample data and compute z-scores from position bias.
    """
    print("="*80)
    print("EXTRACTING SAMPLE DATA AND COMPUTING Z-SCORES")
    print("="*80)

    # Load sample-level data
    samples = samples_df("../logs/self-sycophancy-1")

    print(f"\nLoaded {len(samples)} samples")
    print(f"Available score columns:")
    score_cols = [col for col in samples.columns if 'score_' in col]
    for col in score_cols:
        non_null = samples[col].notna().sum()
        print(f"  {col}: {non_null} non-null values")

    # Key insight: position_bias_scorer returns VARIANCE of forced SSIs
    # So within_sample_std = sqrt(position_bias)

    # Filter to samples with both calibrated effect and position bias
    has_both = samples[
        samples['score_calibrated_effect_scorer'].notna() &
        samples['score_position_bias_scorer'].notna()
        ].copy()

    print(f"\nSamples with both calibrated_effect and position_bias: {len(has_both)}")

    if len(has_both) == 0:
        print("ERROR: No samples have both scores needed for z-score calculation")
        return None

    # Compute within-sample standard deviation from position bias (variance)
    has_both['within_sample_std'] = np.sqrt(has_both['score_position_bias_scorer'])

    # Compute z-scores
    has_both['z_score'] = has_both['score_calibrated_effect_scorer'] / has_both['within_sample_std']

    # Handle cases where std is 0 (all forced SSIs identical)
    has_both.loc[has_both['within_sample_std'] == 0, 'z_score'] = 0

    # Compute significance
    has_both['significant'] = has_both['z_score'].abs() > 1.96
    has_both['sig_level'] = pd.cut(
        has_both['z_score'].abs(),
        bins=[0, 1.96, 2.58, 3.29, float('inf')],
        labels=['ns', '*', '**', '***']
    )

    print("\n" + "="*80)
    print("Z-SCORE STATISTICS")
    print("="*80)

    z_scores = has_both['z_score'].dropna()
    print(f"Total samples with z-scores: {len(z_scores)}")
    print(f"\nZ-score distribution:")
    print(f"  Mean: {z_scores.mean():.3f}")
    print(f"  Median: {z_scores.median():.3f}")
    print(f"  Std Dev: {z_scores.std():.3f}")
    print(f"  Min: {z_scores.min():.3f}")
    print(f"  Max: {z_scores.max():.3f}")

    # Significance breakdown
    sig_pos = (z_scores > 1.96).sum()
    sig_neg = (z_scores < -1.96).sum()
    not_sig = len(z_scores) - sig_pos - sig_neg

    print(f"\nSignificance breakdown:")
    print(f"  Positive (z > 1.96): {sig_pos} ({100*sig_pos/len(z_scores):.1f}%)")
    print(f"  Negative (z < -1.96): {sig_neg} ({100*sig_neg/len(z_scores):.1f}%)")
    print(f"  Not significant: {not_sig} ({100*not_sig/len(z_scores):.1f}%)")

    # Check if this is reasonable
    sig_rate = (sig_pos + sig_neg) / len(z_scores)
    print(f"\nOverall significance rate: {100*sig_rate:.1f}%")

    if sig_rate > 0.5:
        print("⚠️  WARNING: High significance rate may indicate an issue")
        print("   Checking calibration...")

        # Diagnostic: Look at the distribution of within-sample stds
        print(f"\nWithin-sample STD statistics:")
        print(f"  Mean: {has_both['within_sample_std'].mean():.3f}")
        print(f"  Median: {has_both['within_sample_std'].median():.3f}")
        print(f"  Min: {has_both['within_sample_std'].min():.3f}")
        print(f"  Max: {has_both['within_sample_std'].max():.3f}")

        # Check position bias values
        print(f"\nPosition bias (variance) statistics:")
        print(f"  Mean: {has_both['score_position_bias_scorer'].mean():.3f}")
        print(f"  Median: {has_both['score_position_bias_scorer'].median():.3f}")

    elif sig_rate < 0.05:
        print("⚠️  WARNING: Very low significance rate")
        print("   This might indicate the effect is genuinely weak or absent")
    else:
        print("✓ Significance rate appears reasonable")

    return has_both


def analyze_by_groups(df):
    """
    Analyze significance by various groupings.
    """
    print("\n" + "="*80)
    print("ANALYSIS BY GROUPS")
    print("="*80)

    # Parse task from log column
    # Format: .../2025-08-11T20-33-34-07-00_self-sycophancy-harmfulness-commonsense-comprehensive_NYXPw9onKJFw5E67iQf4N9.eval
    def extract_task_from_log(log_path):
        if pd.isna(log_path):
            return None
        # Extract filename
        filename = str(log_path).split('/')[-1]
        # Extract task part (between timestamp and ID)
        parts = filename.split('_')
        # Find the part that starts with 'self-sycophancy'
        for i, part in enumerate(parts):
            if part.startswith('self-sycophancy'):
                # Reconstruct task name until we hit the ID (which is the last part before .eval)
                task_parts = []
                for j in range(i, len(parts) - 1):  # -1 to exclude the ID
                    task_parts.append(parts[j])
                return '-'.join(task_parts)
        return None

    df['task'] = df['log'].apply(extract_task_from_log)

    # Parse domain and dataset from task
    def parse_task(task_name):
        if pd.isna(task_name):
            return pd.Series([None, None])
        # Task format: self-sycophancy-{domain}-{dataset}-comprehensive
        parts = str(task_name).split('-')
        if len(parts) >= 4 and parts[0] == 'self' and parts[1] == 'sycophancy':
            domain = parts[2]  # e.g., 'harmfulness' or 'correctness'
            # Dataset might have hyphens in it
            dataset_parts = parts[3:]
            if dataset_parts and dataset_parts[-1] == 'comprehensive':
                dataset_parts = dataset_parts[:-1]
            dataset = '-'.join(dataset_parts) if dataset_parts else 'unknown'
            return pd.Series([domain, dataset])
        return pd.Series([None, None])

    df[['domain', 'dataset']] = df['task'].apply(parse_task)

    # Extract model from log path (it's in the parent directory name)
    # Format: .../logs/self-sycophancy-1/{timestamp}_...
    # We need to get this from the eval-level data instead

    print(f"\nUnique tasks found: {df['task'].nunique()}")
    print(f"Sample tasks: {df['task'].dropna().unique()[:3].tolist()}")

    # By domain
    print("\nBy Domain:")
    for domain in sorted(df['domain'].dropna().unique()):
        domain_data = df[df['domain'] == domain]
        z_scores = domain_data['z_score'].dropna()
        if len(z_scores) > 0:
            sig_rate = (z_scores.abs() > 1.96).mean() * 100
            mean_effect = domain_data['score_calibrated_effect_scorer'].mean()
            mean_z = z_scores.mean()
            print(f"  {domain:15s}: n={len(z_scores):4d}, sig={sig_rate:5.1f}%, "
                  f"mean_effect={mean_effect:7.4f}, mean_z={mean_z:6.3f}")

    # By dataset
    print("\nBy Dataset (top 10 by sample count):")
    dataset_counts = df.groupby('dataset').size().sort_values(ascending=False).head(10)
    for dataset, count in dataset_counts.items():
        if pd.notna(dataset):
            dataset_data = df[df['dataset'] == dataset]
            z_scores = dataset_data['z_score'].dropna()
            if len(z_scores) > 0:
                sig_rate = (z_scores.abs() > 1.96).mean() * 100
                mean_effect = dataset_data['score_calibrated_effect_scorer'].mean()
                mean_z = z_scores.mean()
                print(f"  {dataset:20s}: n={len(z_scores):4d}, sig={sig_rate:5.1f}%, "
                      f"effect={mean_effect:7.4f}, z={mean_z:6.3f}")

    # By domain-dataset combination
    print("\nBy Domain-Dataset Combination (top 10):")
    df['domain_dataset'] = df['domain'].astype(str) + '/' + df['dataset'].astype(str)
    combo_counts = df.groupby('domain_dataset').size().sort_values(ascending=False).head(10)
    for combo, count in combo_counts.items():
        if 'None' not in combo:
            combo_data = df[df['domain_dataset'] == combo]
            z_scores = combo_data['z_score'].dropna()
            if len(z_scores) > 0:
                sig_rate = (z_scores.abs() > 1.96).mean() * 100
                mean_effect = combo_data['score_calibrated_effect_scorer'].mean()
                print(f"  {combo:35s}: n={len(z_scores):4d}, sig={sig_rate:5.1f}%, "
                      f"effect={mean_effect:7.4f}")

    return df


def create_diagnostic_plots(df, output_dir="plots_diagnostic"):
    """
    Create diagnostic plots to validate the z-score computation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING DIAGNOSTIC PLOTS")
    print("="*80)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Z-score distribution
    ax = axes[0, 0]
    ax.hist(df['z_score'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(-1.96, color='red', linestyle='--', alpha=0.5)
    ax.axvline(1.96, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Z-scores\n(Should be roughly normal)')

    # 2. Within-sample STD distribution
    ax = axes[0, 1]
    ax.hist(df['within_sample_std'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Within-sample STD\n(sqrt of position bias)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Within-Sample STDs')

    # 3. Calibrated effect vs Z-score
    ax = axes[0, 2]
    ax.scatter(df['score_calibrated_effect_scorer'], df['z_score'],
               alpha=0.3, s=10)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(-1.96, color='red', linestyle='--', alpha=0.5)
    ax.axhline(1.96, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Calibrated Effect')
    ax.set_ylabel('Z-score')
    ax.set_title('Effect vs Z-score\n(Should show positive correlation)')

    # 4. Within-sample STD vs Effect
    ax = axes[1, 0]
    ax.scatter(df['within_sample_std'], df['score_calibrated_effect_scorer'].abs(),
               alpha=0.3, s=10, color='green')
    ax.set_xlabel('Within-sample STD')
    ax.set_ylabel('|Calibrated Effect|')
    ax.set_title('Checking heteroscedasticity')

    # 5. Q-Q plot for z-scores
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot(df['z_score'].dropna(), dist="norm", plot=ax)
    ax.set_title('Q-Q Plot of Z-scores\n(Should be linear if normal)')

    # 6. Significance rate by domain
    ax = axes[1, 2]
    if 'domain' in df.columns:
        domain_sig = df.groupby('domain').apply(
            lambda x: (x['z_score'].abs() > 1.96).mean() * 100
        )
        domain_sig.plot(kind='bar', ax=ax, color=['red', 'blue'])
        ax.axhline(5, color='black', linestyle='--', alpha=0.3, label='5% (expected under null)')
        ax.set_ylabel('% Significant')
        ax.set_title('Significance Rate by Domain')
        ax.legend()

    plt.suptitle('Diagnostic Plots for Z-score Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'diagnostic_plots.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Diagnostic plots saved to: {output_dir / 'diagnostic_plots.png'}")


def validate_position_bias_interpretation(df):
    """
    Validate that position_bias_scorer indeed returns variance.
    """
    print("\n" + "="*80)
    print("VALIDATING POSITION BIAS INTERPRETATION")
    print("="*80)

    # Check relationship between position bias and its square root
    pos_bias = df['score_position_bias_scorer'].dropna()

    print(f"Position bias statistics:")
    print(f"  Mean: {pos_bias.mean():.4f}")
    print(f"  Median: {pos_bias.median():.4f}")
    print(f"  Min: {pos_bias.min():.4f}")
    print(f"  Max: {pos_bias.max():.4f}")

    # If position_bias is variance, sqrt should give reasonable STDs
    sqrt_bias = np.sqrt(pos_bias)
    print(f"\nSqrt(position bias) statistics:")
    print(f"  Mean: {sqrt_bias.mean():.4f}")
    print(f"  Median: {sqrt_bias.median():.4f}")
    print(f"  Min: {sqrt_bias.min():.4f}")
    print(f"  Max: {sqrt_bias.max():.4f}")

    # Check if these are reasonable STD values for SSI scores
    print("\nInterpretation check:")
    print("  If SSI scores range from roughly -10 to +10,")
    print(f"  then STDs of {sqrt_bias.mean():.2f} (mean) seem {'reasonable' if sqrt_bias.mean() < 5 else 'high'}")

    # Look at coefficient of variation
    if 'score_comprehensive_ssi_scorer' in df.columns:
        ssi_scale = df['score_comprehensive_ssi_scorer'].abs().mean()
        cv = sqrt_bias.mean() / ssi_scale if ssi_scale > 0 else 0
        print(f"\nCoefficient of variation: {cv:.3f}")
        print(f"  (within-sample STD / mean |SSI|)")
        if cv < 0.5:
            print("  ✓ This seems reasonable")
        else:
            print("  ⚠ This seems high - might indicate high variability")


def main():
    """
    Main analysis pipeline.
    """
    # Extract and compute z-scores
    df = extract_and_compute_zscores()

    if df is None:
        print("ERROR: Could not compute z-scores. Exiting.")
        return

    # Validate the interpretation
    validate_position_bias_interpretation(df)

    # Analyze by groups
    df = analyze_by_groups(df)

    # Create diagnostic plots
    create_diagnostic_plots(df)

    # Save the enhanced dataframe
    output_file = "samples_with_computed_zscores.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved enhanced sample-level data to: {output_file}")

    # Create task-level aggregation
    print("\n" + "="*80)
    print("CREATING TASK-LEVEL AGGREGATION")
    print("="*80)

    task_agg = df.groupby(['task', 'domain', 'dataset']).agg({
        'score_calibrated_effect_scorer': ['mean', 'std', 'count'],
        'score_comprehensive_ssi_scorer': ['mean', 'std'],
        'z_score': ['mean', 'median', lambda x: (x.abs() > 1.96).mean() * 100],  # significance rate
        'within_sample_std': 'mean',
        'score_position_bias_scorer': 'mean',
        'significant': 'mean'
    }).round(4)

    # Flatten column names
    task_agg.columns = ['_'.join(col).strip() for col in task_agg.columns.values]
    task_agg = task_agg.rename(columns={'z_score_<lambda_0>': 'pct_significant'})
    task_agg = task_agg.reset_index()

    # Save task-level data
    task_output_file = "tasks_with_computed_zscores.csv"
    task_agg.to_csv(task_output_file, index=False)
    print(f"✓ Saved task-level aggregation to: {task_output_file}")
    print(f"  Tasks aggregated: {len(task_agg)}")

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
We computed within-sample z-scores using:
  z = calibrated_effect / sqrt(position_bias)

This works because:
  - position_bias_scorer returns VARIANCE of SSIs across forced positions
  - sqrt(variance) = standard deviation
  - This is the within-sample variation needed for proper significance testing

The significance rate should now be reasonable (typically 5-30% for real effects).
If it's still very high, this might indicate:
  1. Strong genuine effects in your experiment
  2. Small within-sample variance (forced positions too similar)
  3. A systematic bias in the experimental design

Check the diagnostic plots to validate the results!

Files created:
  - samples_with_computed_zscores.csv: Sample-level data with z-scores
  - tasks_with_computed_zscores.csv: Task-level aggregated statistics
  - plots_diagnostic/diagnostic_plots.png: Validation plots
    """)


if __name__ == "__main__":
    main()