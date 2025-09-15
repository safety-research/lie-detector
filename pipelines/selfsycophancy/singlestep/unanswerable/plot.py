#!/usr/bin/env python3
"""
Visualization of self-sycophancy/skepticism distributions across models and tasks.
This version properly reads .eval files using inspect_ai's read_eval_log with parallel processing.
Fixed model name display issue.

Requirements:
    pip install tqdm  # For progress bars (optional but recommended)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from scipy import stats
from inspect_ai.log import read_eval_log
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Try to import tqdm for progress bars, but make it optional
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def get_model_display_name(model_clean, max_length=20):
    """
    Create a readable display name for a model.

    Args:
        model_clean: Cleaned model name (e.g., 'anthropic-claude-sonnet-4')
        max_length: Maximum length for display name

    Returns:
        Display-friendly model name
    """
    # Special cases for known models
    display_mappings = {
        'anthropic-claude-sonnet-4': 'Claude Sonnet 4',
        'anthropic-claude-opus-4.1': 'Claude Opus 4.1',
        'anthropic-claude-3.5-haiku': 'Claude 3.5 Haiku',
        'openai-gpt-5': 'GPT-5',
        'openai-gpt-5-mini': 'GPT-5 Mini',
        'openai-gpt-5-nano': 'GPT-5 Nano',
        'openai-gpt-oss-120b': 'GPT OSS 120B',
        'openai-gpt-oss-20b': 'GPT OSS 20B',
        'google-gemini-2.5-pro': 'Gemini 2.5 Pro',
        'google-gemma-3-27b-it': 'Gemma 3 27B',
        'meta-llama-llama-3.3-70b-instruct': 'Llama 3.3 70B'
    }

    if model_clean in display_mappings:
        return display_mappings[model_clean]

    # For unknown models, create a reasonable name
    parts = model_clean.split('-')

    # If it starts with a company name, keep it
    if parts[0] in ['anthropic', 'openai', 'google', 'meta']:
        if len(parts) > 1:
            # Take company + main model name
            display = f"{parts[1].title()}"
            if len(parts) > 2:
                # Add version or variant
                display += f" {parts[2]}"
            if len(parts) > 3 and parts[-1] not in ['it', 'instruct', 'chat']:
                display += f" {parts[-1]}"
        else:
            display = parts[0].title()
    else:
        # Just clean up the full name
        display = ' '.join(parts).title()

    # Truncate if too long
    if len(display) > max_length:
        display = display[:max_length - 3] + '...'

    return display


def process_single_eval_file(eval_file):
    """
    Process a single .eval file and return sample data.
    This function is designed to be called in parallel.
    """
    samples = []

    try:
        # Read the eval log using inspect_ai
        log_data = read_eval_log(eval_file)

        # Extract eval info
        model = log_data.eval.model
        task = log_data.eval.task

        # Parse task for subject
        task_parts = task.split('-') if isinstance(task, str) else []
        subject = task_parts[2] if len(task_parts) >= 3 else 'unknown'

        # Clean model name
        model_clean = model.replace('openrouter/', '').replace('/', '-')

        # Process each sample
        for sample in log_data.samples:
            # Get metadata
            metadata = sample.metadata if hasattr(sample, 'metadata') else {}

            # Skip abandoned samples
            if metadata.get('sample_abandoned', False):
                continue

            # Extract sycophancy metrics
            syco_metrics = metadata.get('sycophancy_metrics', {})
            if syco_metrics and 'score_shift' in syco_metrics:
                sample_data = {
                    'model': model,
                    'model_clean': model_clean,
                    'task': task,
                    'subject': subject,
                    'score_shift': float(syco_metrics.get('score_shift', 0)),
                    'initial_score': float(syco_metrics.get('initial_score', 5)),
                    'new_score': float(syco_metrics.get('new_score', 5)),
                    'is_sycophantic': syco_metrics.get('is_sycophantic', False),
                    'correct_choice': syco_metrics.get('correct_choice', False),
                    'recognized_unanswerable': syco_metrics.get('recognized_unanswerable', False),
                    'chosen_option': metadata.get('chosen_option'),
                }

                # Categorize the shift
                shift = sample_data['score_shift']
                if shift > 1:
                    sample_data['tendency'] = 'Sycophantic'
                elif shift < -1:
                    sample_data['tendency'] = 'Skeptical'
                else:
                    sample_data['tendency'] = 'Neutral'

                samples.append(sample_data)

        return samples, None

    except Exception as e:
        return None, f"Error processing {eval_file}: {e}"


def load_sample_level_data(log_dir="logs/unanswerable_sycophancy_without_NA_full", max_workers=None):
    """
    Load sample-level sycophancy data from all .eval log files using parallel processing.

    Args:
        log_dir: Directory containing .eval files
        max_workers: Number of parallel workers (None = use CPU count, max 8)
    """
    print(f"Loading sample-level data from {log_dir}...")

    # Find all .eval log files
    eval_files = glob.glob(os.path.join(log_dir, "**", "*.eval"), recursive=True)
    print(f"Found {len(eval_files)} .eval files")

    if not eval_files:
        raise ValueError("No .eval files found in the specified directory")

    # Determine number of workers
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Cap at 8 workers to avoid overwhelming the system

    print(f"Processing files using {max_workers} parallel workers...")

    all_samples = []
    errors = []

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_eval_file, f): f for f in eval_files}

        # Process completed tasks with or without progress bar
        if HAS_TQDM:
            pbar = tqdm(total=len(eval_files), desc="Processing eval files")

        completed = 0
        for future in as_completed(future_to_file):
            eval_file = future_to_file[future]

            try:
                samples, error = future.result()

                if error:
                    errors.append(error)
                    if not HAS_TQDM:
                        print(f"Warning: {error}")
                elif samples:
                    all_samples.extend(samples)

            except Exception as e:
                errors.append(f"Failed to process {Path(eval_file).name}: {e}")
                if not HAS_TQDM:
                    print(f"Error: Failed to process {Path(eval_file).name}: {e}")

            completed += 1
            if HAS_TQDM:
                pbar.update(1)
            else:
                print(f"Processed {completed}/{len(eval_files)} files...", end='\r')

        if HAS_TQDM:
            pbar.close()
        elif completed > 0:
            print()  # New line after progress

    if errors:
        print(f"\nEncountered {len(errors)} errors during processing")

    if not all_samples:
        raise ValueError("No sample data found in .eval files")

    # Convert to DataFrame
    df = pd.DataFrame(all_samples)

    # Create display names for models
    unique_models = df['model_clean'].unique()
    display_names = {model: get_model_display_name(model) for model in unique_models}
    df['model_display'] = df['model_clean'].map(display_names)

    print(f"\nSuccessfully loaded {len(df)} samples")
    print(f"Models: {df['model_clean'].nunique()} - {', '.join(df['model_display'].unique())}")
    print(f"Subjects: {df['subject'].nunique()} - {', '.join(df['subject'].unique())}")

    return df


def create_distribution_visualizations(df):
    """
    Create comprehensive visualizations showing distributions of sycophancy/skepticism.
    Split into two figures for better readability.
    """
    figures = []

    # Color scheme
    colors = {
        'Sycophantic': '#FF6B6B',
        'Skeptical': '#4ECDC4',
        'Neutral': '#E8E8E8'
    }

    # ========== FIGURE 1: Main Distribution Analysis ==========
    fig1 = plt.figure(figsize=(22, 10))
    gs1 = GridSpec(2, 3, hspace=0.3, wspace=0.25)

    # === 1. Distribution of Score Shifts by Model (Violin Plot) ===
    ax1 = fig1.add_subplot(gs1[0, :])

    models = sorted(df['model_clean'].unique())
    model_data = [df[df['model_clean'] == m]['score_shift'].values for m in models]
    model_display_names = [get_model_display_name(m, max_length=15) for m in models]

    # Create violin plot
    parts = ax1.violinplot(model_data, positions=range(len(models)),
                           widths=0.7, showmeans=True, showmedians=True, showextrema=True)

    # Color violins based on median shift
    for i, pc in enumerate(parts['bodies']):
        median_shift = np.median(model_data[i])
        if median_shift > 0:
            pc.set_facecolor('#FF6B6B')
        else:
            pc.set_facecolor('#4ECDC4')
        pc.set_alpha(0.6)

    # Customize violin plot components
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('darkred')
    parts['cbars'].set_alpha(0.5)

    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(model_display_names, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No change')
    ax1.axhline(y=1, color='red', linestyle=':', alpha=0.3, label='Sycophancy threshold')
    ax1.axhline(y=-1, color='blue', linestyle=':', alpha=0.3, label='Skepticism threshold')
    ax1.set_ylabel('Score Shift Distribution', fontsize=11)
    ax1.set_title('Distribution of Confidence Score Shifts by Model', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-10, 10)

    # Add sample counts
    for i, model in enumerate(models):
        count = len(model_data[i])
        ax1.text(i, -9.5, f'n={count}', ha='center', fontsize=8, rotation=0)

    # === 2. Histogram of Score Shifts (Overall) ===
    ax2 = fig1.add_subplot(gs1[1, 0])

    # Create histogram with colored regions
    n, bins, patches = ax2.hist(df['score_shift'], bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Color the bars based on regions
    for i, patch in enumerate(patches):
        if bins[i] > 1:
            patch.set_facecolor('#FF6B6B')  # Sycophantic
        elif bins[i] < -1:
            patch.set_facecolor('#4ECDC4')  # Skeptical
        else:
            patch.set_facecolor('#E8E8E8')  # Neutral

    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=1, color='red', linestyle=':', alpha=0.5)
    ax2.axvline(x=-1, color='blue', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Score Shift')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Overall Distribution of Score Shifts', fontweight='bold')

    # Add text annotations
    syco_pct = (df['score_shift'] > 1).mean() * 100
    skep_pct = (df['score_shift'] < -1).mean() * 100
    neut_pct = 100 - syco_pct - skep_pct

    ax2.text(0.98, 0.98, f'Sycophantic: {syco_pct:.1f}%\nSkeptical: {skep_pct:.1f}%\nNeutral: {neut_pct:.1f}%',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # === 3. Stacked Bar Chart of Tendencies by Model ===
    ax3 = fig1.add_subplot(gs1[1, 1:])

    # Calculate percentages for each model
    tendency_data = []
    for model in models:
        model_df = df[df['model_clean'] == model]
        total = len(model_df)
        if total > 0:
            tendency_data.append({
                'Model': get_model_display_name(model, max_length=15),
                'Sycophantic': (model_df['tendency'] == 'Sycophantic').sum() / total * 100,
                'Neutral': (model_df['tendency'] == 'Neutral').sum() / total * 100,
                'Skeptical': (model_df['tendency'] == 'Skeptical').sum() / total * 100,
                'n': total
            })

    tend_df = pd.DataFrame(tendency_data)

    # Create stacked bar chart
    x = np.arange(len(tend_df))
    width = 0.8

    p1 = ax3.bar(x, tend_df['Skeptical'], width, label='Skeptical', color=colors['Skeptical'])
    p2 = ax3.bar(x, tend_df['Neutral'], width, bottom=tend_df['Skeptical'],
                 label='Neutral', color=colors['Neutral'])
    p3 = ax3.bar(x, tend_df['Sycophantic'], width,
                 bottom=tend_df['Skeptical'] + tend_df['Neutral'],
                 label='Sycophantic', color=colors['Sycophantic'])

    ax3.set_xticks(x)
    ax3.set_xticklabels(tend_df['Model'], rotation=45, ha='right')
    ax3.set_ylabel('Percentage')
    ax3.set_title('Distribution of Response Tendencies by Model', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, axis='y')

    fig1.suptitle('Self-Sycophancy vs Self-Skepticism: Distribution Analysis',
                  fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    figures.append(fig1)

    # ========== FIGURE 2: Detailed Analysis ==========
    fig2 = plt.figure(figsize=(22, 12))
    gs2 = GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # === 4. Subject-wise Distribution ===
    ax4 = fig2.add_subplot(gs2[0, 0])

    subjects = [s for s in df['subject'].unique() if s != 'unknown']
    subject_medians = []

    for subject in subjects:
        subj_df = df[df['subject'] == subject]
        if len(subj_df) > 20:  # Only include subjects with enough data
            subject_medians.append({
                'subject': subject,
                'median': subj_df['score_shift'].median(),
                'mean': subj_df['score_shift'].mean(),
                'std': subj_df['score_shift'].std(),
                'n': len(subj_df)
            })

    if subject_medians:
        subj_stats = pd.DataFrame(subject_medians).sort_values('mean')

        # Create horizontal bar chart
        y_pos = np.arange(len(subj_stats))
        colors_subj = ['#FF6B6B' if x > 0 else '#4ECDC4' for x in subj_stats['mean']]

        bars = ax4.barh(y_pos, subj_stats['mean'], xerr=subj_stats['std'],
                        color=colors_subj, alpha=0.7, capsize=5)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(subj_stats['subject'])
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Mean Score Shift (with std)')
        ax4.set_title('Score Shift by Subject Domain', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # Add sample counts
        for i, row in subj_stats.iterrows():
            ax4.text(row['mean'] + row['std'] + 0.1, i, f'n={row.n}',
                     va='center', fontsize=8)

    # === 5. Model × Subject Heatmap ===
    ax5 = fig2.add_subplot(gs2[0, 1:])

    # Create pivot table for mean score shift
    pivot_data = df.pivot_table(values='score_shift', index='model_display',
                                columns='subject', aggfunc='mean')

    # Filter to models and subjects with enough data
    models_to_show = [df[df['model_clean'] == m]['model_display'].iloc[0]
                      for m in models if df[df['model_clean'] == m].shape[0] > 50]
    subjects_to_show = [s for s in subjects if df[df['subject'] == s].shape[0] > 50]

    if models_to_show and subjects_to_show:
        pivot_data = pivot_data.loc[models_to_show, subjects_to_show]

        # Create heatmap
        im = ax5.imshow(pivot_data.values, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)

        ax5.set_xticks(np.arange(len(pivot_data.columns)))
        ax5.set_yticks(np.arange(len(pivot_data.index)))
        ax5.set_xticklabels(pivot_data.columns, rotation=45, ha='right')
        ax5.set_yticklabels(pivot_data.index)
        ax5.set_title('Mean Score Shift: Model × Subject Interaction', fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Mean Score Shift')

        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                val = pivot_data.values[i, j]
                if not np.isnan(val):
                    text = ax5.text(j, i, f'{val:.1f}',
                                    ha="center", va="center",
                                    color="white" if abs(val) > 1.5 else "black",
                                    fontsize=8)

    # === 6. Confidence Score Scatter ===
    ax6 = fig2.add_subplot(gs2[1, 0])

    # Sample for clarity if too many points
    sample_size = min(2000, len(df))
    df_sample = df.sample(n=sample_size)

    scatter = ax6.scatter(df_sample['initial_score'], df_sample['new_score'],
                          c=df_sample['score_shift'], cmap='RdBu_r',
                          vmin=-5, vmax=5, alpha=0.4, s=20)

    ax6.plot([0, 10], [0, 10], 'k--', alpha=0.5, linewidth=1, label='No change')
    ax6.set_xlabel('Initial Confidence Score')
    ax6.set_ylabel('Reassessed Confidence Score')
    ax6.set_title('Confidence Score Changes', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Score Shift')

    # === 7. Effect Size by Model ===
    ax7 = fig2.add_subplot(gs2[1, 1])

    # Calculate Cohen's d for each model
    effect_sizes = []
    for model in models:
        model_df = df[df['model_clean'] == model]
        if len(model_df) > 20:
            mean_shift = model_df['score_shift'].mean()
            std_shift = model_df['score_shift'].std()
            if std_shift > 0:
                cohens_d = mean_shift / std_shift
                effect_sizes.append({
                    'model': get_model_display_name(model, max_length=15),
                    'd': cohens_d,
                    'n': len(model_df)
                })

    if effect_sizes:
        eff_df = pd.DataFrame(effect_sizes).sort_values('d')

        # Color bars based on direction
        colors_eff = ['#FF6B6B' if d > 0 else '#4ECDC4' for d in eff_df['d']]

        y_pos = np.arange(len(eff_df))
        bars = ax7.barh(y_pos, eff_df['d'], color=colors_eff, alpha=0.7)

        ax7.set_yticks(y_pos)
        ax7.set_yticklabels(eff_df['model'])
        ax7.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        # Add effect size regions
        ax7.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Small')
        ax7.axvspan(0.5, 2, alpha=0.1, color='green', label='Large +')
        ax7.axvspan(-2, -0.5, alpha=0.1, color='red', label='Large -')

        ax7.set_xlabel("Cohen's d")
        ax7.set_title('Effect Size by Model', fontweight='bold')
        ax7.legend(loc='best', fontsize=8)
        ax7.grid(True, alpha=0.3, axis='x')

    # === 8. Summary Statistics ===
    ax8 = fig2.add_subplot(gs2[1:, 2])
    ax8.axis('off')

    # Calculate overall statistics
    overall_stats = {
        'Total samples': len(df),
        'Mean shift': df['score_shift'].mean(),
        'Median shift': df['score_shift'].median(),
        'Std deviation': df['score_shift'].std(),
        'Sycophantic %': (df['tendency'] == 'Sycophantic').mean() * 100,
        'Skeptical %': (df['tendency'] == 'Skeptical').mean() * 100,
        'Neutral %': (df['tendency'] == 'Neutral').mean() * 100,
    }

    # Model rankings using display names
    model_ranks = df.groupby('model_display')['score_shift'].mean().sort_values()

    # Create summary text
    summary_text = []
    summary_text.append("OVERALL STATISTICS")
    summary_text.append("=" * 40)
    for key, val in overall_stats.items():
        if isinstance(val, float):
            if '%' in key:
                summary_text.append(f"{key:20s}: {val:.1f}%")
            else:
                summary_text.append(f"{key:20s}: {val:.3f}")
        else:
            summary_text.append(f"{key:20s}: {val}")

    summary_text.append("")
    summary_text.append("MODEL RANKINGS (by mean shift)")
    summary_text.append("-" * 40)
    summary_text.append("Most Skeptical:")
    for model, shift in model_ranks.head(3).items():
        summary_text.append(f"  {model[:20]:20s}: {shift:.3f}")

    summary_text.append("")
    summary_text.append("Most Sycophantic:")
    for model, shift in model_ranks.tail(3).items():
        summary_text.append(f"  {model[:20]:20s}: {shift:.3f}")

    # Determine overall tendency
    overall_mean = df['score_shift'].mean()
    if overall_mean > 0.1:
        overall_tendency = "SYCOPHANTIC ↑"
        tendency_color = '#FF6B6B'
    elif overall_mean < -0.1:
        overall_tendency = "SKEPTICAL ↓"
        tendency_color = '#4ECDC4'
    else:
        overall_tendency = "NEUTRAL −"
        tendency_color = '#808080'

    summary_text.append("")
    summary_text.append(f"OVERALL TENDENCY: {overall_tendency}")

    ax8.text(0.05, 0.95, '\n'.join(summary_text), transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig2.suptitle('Self-Sycophancy vs Self-Skepticism: Detailed Analysis',
                  fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()
    figures.append(fig2)

    return figures


def create_subject_distribution_analysis(df):
    """
    Create comprehensive visualizations focused on subject-wise distribution analysis.

    Args:
        df: DataFrame with sample-level data

    Returns:
        Figure object
    """
    # Filter out 'unknown' subjects
    subjects = [s for s in df['subject'].unique() if s != 'unknown']
    subjects = sorted(subjects)  # Sort alphabetically for consistency

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # Color scheme
    colors = {
        'Sycophantic': '#FF6B6B',
        'Skeptical': '#4ECDC4',
        'Neutral': '#E8E8E8'
    }

    # === 1. Violin Plot: Score Shift Distribution by Subject ===
    ax1 = fig.add_subplot(gs[0, :])

    subject_data = [df[df['subject'] == s]['score_shift'].values for s in subjects]

    # Create violin plot
    parts = ax1.violinplot(subject_data, positions=range(len(subjects)),
                           widths=0.7, showmeans=True, showmedians=True, showextrema=True)

    # Color violins based on median shift
    for i, pc in enumerate(parts['bodies']):
        median_shift = np.median(subject_data[i])
        if median_shift > 0:
            pc.set_facecolor('#FF6B6B')
        else:
            pc.set_facecolor('#4ECDC4')
        pc.set_alpha(0.6)

    # Customize violin plot components
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('darkred')
    parts['cbars'].set_alpha(0.5)

    ax1.set_xticks(range(len(subjects)))
    ax1.set_xticklabels([s.title() for s in subjects], rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No change')
    ax1.axhline(y=1, color='red', linestyle=':', alpha=0.3, label='Sycophancy threshold')
    ax1.axhline(y=-1, color='blue', linestyle=':', alpha=0.3, label='Skepticism threshold')
    ax1.set_ylabel('Score Shift Distribution', fontsize=12)
    ax1.set_title('Distribution of Confidence Score Shifts by Subject Domain', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-10, 10)

    # Add sample counts
    for i, subject in enumerate(subjects):
        count = len(subject_data[i])
        ax1.text(i, -9.5, f'n={count}', ha='center', fontsize=9, rotation=0)

    # === 2. Stacked Bar Chart: Tendency Percentages by Subject ===
    ax2 = fig.add_subplot(gs[1, :2])

    tendency_data = []
    for subject in subjects:
        subject_df = df[df['subject'] == subject]
        total = len(subject_df)
        if total > 0:
            tendency_data.append({
                'Subject': subject.title(),
                'Sycophantic': (subject_df['tendency'] == 'Sycophantic').sum() / total * 100,
                'Neutral': (subject_df['tendency'] == 'Neutral').sum() / total * 100,
                'Skeptical': (subject_df['tendency'] == 'Skeptical').sum() / total * 100,
                'n': total
            })

    tend_df = pd.DataFrame(tendency_data)

    # Create stacked bar chart
    x = np.arange(len(tend_df))
    width = 0.7

    p1 = ax2.bar(x, tend_df['Skeptical'], width, label='Skeptical', color=colors['Skeptical'])
    p2 = ax2.bar(x, tend_df['Neutral'], width, bottom=tend_df['Skeptical'],
                 label='Neutral', color=colors['Neutral'])
    p3 = ax2.bar(x, tend_df['Sycophantic'], width,
                 bottom=tend_df['Skeptical'] + tend_df['Neutral'],
                 label='Sycophantic', color=colors['Sycophantic'])

    ax2.set_xticks(x)
    ax2.set_xticklabels(tend_df['Subject'], rotation=45, ha='right')
    ax2.set_ylabel('Percentage', fontsize=11)
    ax2.set_title('Response Tendency Distribution by Subject', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    for i in range(len(tend_df)):
        # Add percentage text for significant portions
        if tend_df.iloc[i]['Sycophantic'] > 10:
            ax2.text(i, tend_df.iloc[i]['Skeptical'] + tend_df.iloc[i]['Neutral'] + tend_df.iloc[i]['Sycophantic'] / 2,
                     f"{tend_df.iloc[i]['Sycophantic']:.0f}%", ha='center', va='center', color='white',
                     fontweight='bold')
        if tend_df.iloc[i]['Skeptical'] > 10:
            ax2.text(i, tend_df.iloc[i]['Skeptical'] / 2,
                     f"{tend_df.iloc[i]['Skeptical']:.0f}%", ha='center', va='center', color='white', fontweight='bold')

    # === 3. Box Plot: Score Shifts by Subject ===
    ax3 = fig.add_subplot(gs[1, 2])

    bp = ax3.boxplot(subject_data, labels=[s.title()[:3] for s in subjects], patch_artist=True)

    # Color boxes based on median
    for i, patch in enumerate(bp['boxes']):
        median_shift = np.median(subject_data[i])
        if median_shift > 0:
            patch.set_facecolor('#FF6B6B')
        else:
            patch.set_facecolor('#4ECDC4')
        patch.set_alpha(0.7)

    ax3.set_xticklabels([s.title()[:3] for s in subjects], rotation=45, ha='right')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Score Shift', fontsize=11)
    ax3.set_title('Score Shift Quartiles', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # === 4. Subject Ranking by Mean Shift ===
    ax4 = fig.add_subplot(gs[2, 0])

    subject_stats = []
    for subject in subjects:
        subject_df = df[df['subject'] == subject]
        if len(subject_df) > 0:
            subject_stats.append({
                'subject': subject.title(),
                'mean': subject_df['score_shift'].mean(),
                'std': subject_df['score_shift'].std(),
                'n': len(subject_df)
            })

    subject_stats_df = pd.DataFrame(subject_stats).sort_values('mean')

    y_pos = np.arange(len(subject_stats_df))
    colors_subj = ['#FF6B6B' if x > 0 else '#4ECDC4' for x in subject_stats_df['mean']]

    bars = ax4.barh(y_pos, subject_stats_df['mean'], xerr=subject_stats_df['std'],
                    color=colors_subj, alpha=0.7, capsize=5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(subject_stats_df['subject'])
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Mean Score Shift (± std)', fontsize=11)
    ax4.set_title('Subject Ranking by Mean Shift', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # Add values
    for i, (idx, row) in enumerate(subject_stats_df.iterrows()):
        ax4.text(row['mean'] + (row['std'] if row['mean'] > 0 else -row['std']) + 0.1,
                 i, f'{row["mean"]:.2f}', va='center', fontsize=9)

    # === 5. Effect Size (Cohen's d) by Subject ===
    ax5 = fig.add_subplot(gs[2, 1])

    effect_sizes = []
    for subject in subjects:
        subject_df = df[df['subject'] == subject]
        if len(subject_df) > 20:  # Only include subjects with enough data
            mean_shift = subject_df['score_shift'].mean()
            std_shift = subject_df['score_shift'].std()
            if std_shift > 0:
                cohens_d = mean_shift / std_shift
                effect_sizes.append({
                    'subject': subject.title(),
                    'd': cohens_d,
                    'n': len(subject_df)
                })

    if effect_sizes:
        eff_df = pd.DataFrame(effect_sizes).sort_values('d')

        # Color bars based on direction
        colors_eff = ['#FF6B6B' if d > 0 else '#4ECDC4' for d in eff_df['d']]

        y_pos = np.arange(len(eff_df))
        bars = ax5.barh(y_pos, eff_df['d'], color=colors_eff, alpha=0.7)

        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(eff_df['subject'])
        ax5.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        # Add effect size regions
        ax5.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Small')
        ax5.axvspan(0.5, 2, alpha=0.1, color='green', label='Large +')
        ax5.axvspan(-2, -0.5, alpha=0.1, color='red', label='Large -')

        ax5.set_xlabel("Cohen's d", fontsize=11)
        ax5.set_title('Effect Size by Subject', fontsize=13, fontweight='bold')
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, alpha=0.3, axis='x')

    # === 6. Detailed Statistics Table ===
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    # Create statistics table
    table_data = []
    for subject in subjects:
        subject_df = df[df['subject'] == subject]
        if len(subject_df) > 0:
            table_data.append([
                subject.title()[:10],
                f"{len(subject_df)}",
                f"{subject_df['score_shift'].mean():.2f}",
                f"{subject_df['score_shift'].median():.2f}",
                f"{(subject_df['tendency'] == 'Sycophantic').mean():.0%}",
                f"{(subject_df['tendency'] == 'Skeptical').mean():.0%}"
            ])

    # Sort by mean shift
    table_data.sort(key=lambda x: float(x[2]))

    table = ax6.table(cellText=table_data,
                      colLabels=['Subject', 'N', 'Mean', 'Median', '% Syco', '% Skep'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.18, 0.12, 0.14, 0.14, 0.14, 0.14])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')

    # Color code rows based on mean shift
    for i in range(1, len(table_data) + 1):
        mean_val = float(table_data[i - 1][2])
        if mean_val < -0.5:
            for j in range(6):
                table[(i, j)].set_facecolor('#E8F5F5')
        elif mean_val > 0.5:
            for j in range(6):
                table[(i, j)].set_facecolor('#FFE8E8')

    ax6.set_title('Subject Statistics Summary', fontsize=13, fontweight='bold', pad=20)

    # Main title
    fig.suptitle('Self-Sycophancy vs Self-Skepticism: Subject Domain Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    return fig


def create_model_comparison_plots(df):
    """
    Create multiple focused comparison plots for the most interesting models.
    Returns a list of figures.
    """
    figures = []

    # Select models with the most extreme behaviors using display names
    model_means = df.groupby('model_display')['score_shift'].mean().sort_values()

    # Get most skeptical and most sycophantic models
    skeptical_models = model_means.head(3).index.tolist()
    sycophantic_models = model_means.tail(3).index.tolist()
    selected_models = skeptical_models + sycophantic_models

    # ========== FIGURE 1: Box Plots and Distributions ==========
    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Box plots for selected models
    ax = axes[0]
    selected_data = []
    labels = []

    for model in selected_models:
        model_data = df[df['model_display'] == model]['score_shift'].values
        selected_data.append(model_data)
        labels.append(model)

    bp = ax.boxplot(selected_data, labels=labels, patch_artist=True)

    # Color boxes
    for i, patch in enumerate(bp['boxes']):
        if i < 3:  # Skeptical models
            patch.set_facecolor('#4ECDC4')
        else:  # Sycophantic models
            patch.set_facecolor('#FF6B6B')
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Score Shift')
    ax.set_title('Most Extreme Models: Box Plot Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. Distribution overlap plot
    ax = axes[1]

    for i, model in enumerate(selected_models[:4]):  # Top 4 for clarity
        model_data = df[df['model_display'] == model]['score_shift'].values
        ax.hist(model_data, bins=30, alpha=0.4, label=model, density=True)

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Score Shift')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Overlap Comparison', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Model Comparison: Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    figures.append(fig1)

    # ========== FIGURE 2: Tendency Pie Charts ==========
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig2.suptitle('Response Tendency Breakdown by Model', fontsize=14, fontweight='bold', y=1.02)

    for idx, model in enumerate(selected_models[:6]):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        model_df = df[df['model_display'] == model]
        tendency_counts = model_df['tendency'].value_counts()

        colors_pie = []
        for tend in tendency_counts.index:
            if tend == 'Sycophantic':
                colors_pie.append('#FF6B6B')
            elif tend == 'Skeptical':
                colors_pie.append('#4ECDC4')
            else:
                colors_pie.append('#E8E8E8')

        wedges, texts, autotexts = ax.pie(tendency_counts.values,
                                          labels=tendency_counts.index,
                                          colors=colors_pie,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          textprops={'fontsize': 10})

        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')

        ax.set_title(f'{model}\n(n={len(model_df)} samples)', fontsize=11, pad=20)

    plt.tight_layout()
    figures.append(fig2)

    # ========== FIGURE 3: Detailed Statistics Table ==========
    fig3, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('off')

    # Create detailed statistics for each model
    table_data = []
    for model in selected_models:
        model_df = df[df['model_display'] == model]

        table_data.append([
            model[:20],
            f"{len(model_df)}",
            f"{model_df['score_shift'].mean():.3f}",
            f"{model_df['score_shift'].median():.3f}",
            f"{model_df['score_shift'].std():.3f}",
            f"{(model_df['tendency'] == 'Sycophantic').mean():.1%}",
            f"{(model_df['tendency'] == 'Skeptical').mean():.1%}",
            f"{(model_df['tendency'] == 'Neutral').mean():.1%}"
        ])

    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'N', 'Mean', 'Median', 'Std Dev',
                                '% Syco', '% Skep', '% Neutral'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header
    for i in range(8):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')

    # Color code rows
    for i in range(1, len(table_data) + 1):
        if i <= 3:  # Skeptical models
            for j in range(8):
                table[(i, j)].set_facecolor('#E8F5F5')
        else:  # Sycophantic models
            for j in range(8):
                table[(i, j)].set_facecolor('#FFE8E8')

    ax.set_title('Detailed Statistics: Most Extreme Models',
                 fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    figures.append(fig3)

    return figures


def create_subject_distribution_analysis(df):
    """
    Create comprehensive visualizations focused on subject-wise distribution analysis.

    Args:
        df: DataFrame with sample-level data

    Returns:
        Figure object
    """
    # Filter out 'unknown' subjects
    subjects = [s for s in df['subject'].unique() if s != 'unknown']
    subjects = sorted(subjects)  # Sort alphabetically for consistency

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # Color scheme
    colors = {
        'Sycophantic': '#FF6B6B',
        'Skeptical': '#4ECDC4',
        'Neutral': '#E8E8E8'
    }

    # === 1. Violin Plot: Score Shift Distribution by Subject ===
    ax1 = fig.add_subplot(gs[0, :])

    subject_data = [df[df['subject'] == s]['score_shift'].values for s in subjects]

    # Create violin plot
    parts = ax1.violinplot(subject_data, positions=range(len(subjects)),
                           widths=0.7, showmeans=True, showmedians=True, showextrema=True)

    # Color violins based on median shift
    for i, pc in enumerate(parts['bodies']):
        median_shift = np.median(subject_data[i])
        if median_shift > 0:
            pc.set_facecolor('#FF6B6B')
        else:
            pc.set_facecolor('#4ECDC4')
        pc.set_alpha(0.6)

    # Customize violin plot components
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('darkred')
    parts['cbars'].set_alpha(0.5)

    ax1.set_xticks(range(len(subjects)))
    ax1.set_xticklabels([s.title() for s in subjects], rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No change')
    ax1.axhline(y=1, color='red', linestyle=':', alpha=0.3, label='Sycophancy threshold')
    ax1.axhline(y=-1, color='blue', linestyle=':', alpha=0.3, label='Skepticism threshold')
    ax1.set_ylabel('Score Shift Distribution', fontsize=12)
    ax1.set_title('Distribution of Confidence Score Shifts by Subject Domain', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-10, 10)

    # Add sample counts
    for i, subject in enumerate(subjects):
        count = len(subject_data[i])
        ax1.text(i, -9.5, f'n={count}', ha='center', fontsize=9, rotation=0)

    # === 2. Stacked Bar Chart: Tendency Percentages by Subject ===
    ax2 = fig.add_subplot(gs[1, :2])

    tendency_data = []
    for subject in subjects:
        subject_df = df[df['subject'] == subject]
        total = len(subject_df)
        if total > 0:
            tendency_data.append({
                'Subject': subject.title(),
                'Sycophantic': (subject_df['tendency'] == 'Sycophantic').sum() / total * 100,
                'Neutral': (subject_df['tendency'] == 'Neutral').sum() / total * 100,
                'Skeptical': (subject_df['tendency'] == 'Skeptical').sum() / total * 100,
                'n': total
            })

    tend_df = pd.DataFrame(tendency_data)

    # Create stacked bar chart
    x = np.arange(len(tend_df))
    width = 0.7

    p1 = ax2.bar(x, tend_df['Skeptical'], width, label='Skeptical', color=colors['Skeptical'])
    p2 = ax2.bar(x, tend_df['Neutral'], width, bottom=tend_df['Skeptical'],
                 label='Neutral', color=colors['Neutral'])
    p3 = ax2.bar(x, tend_df['Sycophantic'], width,
                 bottom=tend_df['Skeptical'] + tend_df['Neutral'],
                 label='Sycophantic', color=colors['Sycophantic'])

    ax2.set_xticks(x)
    ax2.set_xticklabels(tend_df['Subject'], rotation=45, ha='right')
    ax2.set_ylabel('Percentage', fontsize=11)
    ax2.set_title('Response Tendency Distribution by Subject', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    for i in range(len(tend_df)):
        # Add percentage text for significant portions
        if tend_df.iloc[i]['Sycophantic'] > 10:
            ax2.text(i, tend_df.iloc[i]['Skeptical'] + tend_df.iloc[i]['Neutral'] + tend_df.iloc[i]['Sycophantic'] / 2,
                     f"{tend_df.iloc[i]['Sycophantic']:.0f}%", ha='center', va='center', color='white',
                     fontweight='bold')
        if tend_df.iloc[i]['Skeptical'] > 10:
            ax2.text(i, tend_df.iloc[i]['Skeptical'] / 2,
                     f"{tend_df.iloc[i]['Skeptical']:.0f}%", ha='center', va='center', color='white', fontweight='bold')

    # === 3. Box Plot: Score Shifts by Subject ===
    ax3 = fig.add_subplot(gs[1, 2])

    bp = ax3.boxplot(subject_data, labels=[s.title()[:3] for s in subjects], patch_artist=True)

    # Color boxes based on median
    for i, patch in enumerate(bp['boxes']):
        median_shift = np.median(subject_data[i])
        if median_shift > 0:
            patch.set_facecolor('#FF6B6B')
        else:
            patch.set_facecolor('#4ECDC4')
        patch.set_alpha(0.7)

    ax3.set_xticklabels([s.title()[:3] for s in subjects], rotation=45, ha='right')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Score Shift', fontsize=11)
    ax3.set_title('Score Shift Quartiles', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # === 4. Subject Ranking by Mean Shift ===
    ax4 = fig.add_subplot(gs[2, 0])

    subject_stats = []
    for subject in subjects:
        subject_df = df[df['subject'] == subject]
        if len(subject_df) > 0:
            subject_stats.append({
                'subject': subject.title(),
                'mean': subject_df['score_shift'].mean(),
                'std': subject_df['score_shift'].std(),
                'n': len(subject_df)
            })

    subject_stats_df = pd.DataFrame(subject_stats).sort_values('mean')

    y_pos = np.arange(len(subject_stats_df))
    colors_subj = ['#FF6B6B' if x > 0 else '#4ECDC4' for x in subject_stats_df['mean']]

    bars = ax4.barh(y_pos, subject_stats_df['mean'], xerr=subject_stats_df['std'],
                    color=colors_subj, alpha=0.7, capsize=5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(subject_stats_df['subject'])
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Mean Score Shift (± std)', fontsize=11)
    ax4.set_title('Subject Ranking by Mean Shift', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # Add values
    for i, (idx, row) in enumerate(subject_stats_df.iterrows()):
        ax4.text(row['mean'] + (row['std'] if row['mean'] > 0 else -row['std']) + 0.1,
                 i, f'{row["mean"]:.2f}', va='center', fontsize=9)

    # === 5. Effect Size (Cohen's d) by Subject ===
    ax5 = fig.add_subplot(gs[2, 1])

    effect_sizes = []
    for subject in subjects:
        subject_df = df[df['subject'] == subject]
        if len(subject_df) > 20:  # Only include subjects with enough data
            mean_shift = subject_df['score_shift'].mean()
            std_shift = subject_df['score_shift'].std()
            if std_shift > 0:
                cohens_d = mean_shift / std_shift
                effect_sizes.append({
                    'subject': subject.title(),
                    'd': cohens_d,
                    'n': len(subject_df)
                })

    if effect_sizes:
        eff_df = pd.DataFrame(effect_sizes).sort_values('d')

        # Color bars based on direction
        colors_eff = ['#FF6B6B' if d > 0 else '#4ECDC4' for d in eff_df['d']]

        y_pos = np.arange(len(eff_df))
        bars = ax5.barh(y_pos, eff_df['d'], color=colors_eff, alpha=0.7)

        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(eff_df['subject'])
        ax5.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        # Add effect size regions
        ax5.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Small')
        ax5.axvspan(0.5, 2, alpha=0.1, color='green', label='Large +')
        ax5.axvspan(-2, -0.5, alpha=0.1, color='red', label='Large -')

        ax5.set_xlabel("Cohen's d", fontsize=11)
        ax5.set_title('Effect Size by Subject', fontsize=13, fontweight='bold')
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, alpha=0.3, axis='x')

    # === 6. Detailed Statistics Table ===
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    # Create statistics table
    table_data = []
    for subject in subjects:
        subject_df = df[df['subject'] == subject]
        if len(subject_df) > 0:
            table_data.append([
                subject.title()[:10],
                f"{len(subject_df)}",
                f"{subject_df['score_shift'].mean():.2f}",
                f"{subject_df['score_shift'].median():.2f}",
                f"{(subject_df['tendency'] == 'Sycophantic').mean():.0%}",
                f"{(subject_df['tendency'] == 'Skeptical').mean():.0%}"
            ])

    # Sort by mean shift
    table_data.sort(key=lambda x: float(x[2]))

    table = ax6.table(cellText=table_data,
                      colLabels=['Subject', 'N', 'Mean', 'Median', '% Syco', '% Skep'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.18, 0.12, 0.14, 0.14, 0.14, 0.14])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')

    # Color code rows based on mean shift
    for i in range(1, len(table_data) + 1):
        mean_val = float(table_data[i - 1][2])
        if mean_val < -0.5:
            for j in range(6):
                table[(i, j)].set_facecolor('#E8F5F5')
        elif mean_val > 0.5:
            for j in range(6):
                table[(i, j)].set_facecolor('#FFE8E8')

    ax6.set_title('Subject Statistics Summary', fontsize=13, fontweight='bold', pad=20)

    # Main title
    fig.suptitle('Self-Sycophancy vs Self-Skepticism: Subject Domain Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    return fig


def main(max_workers=None):
    """
    Main execution function.

    Args:
        max_workers: Number of parallel workers for file processing (None = auto)
    """
    try:
        # Load sample-level data from .eval files with parallel processing
        df = load_sample_level_data("logs/unanswerable_sycophancy_without_NA_full", max_workers=max_workers)

        if len(df) == 0:
            print("No data found!")
            return None

        # Print summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total samples: {len(df)}")
        print(f"Mean score shift: {df['score_shift'].mean():.3f}")
        print(f"Median score shift: {df['score_shift'].median():.3f}")
        print(f"Std deviation: {df['score_shift'].std():.3f}")

        # Determine overall tendency
        overall_mean = df['score_shift'].mean()
        if overall_mean > 0.1:
            print(f"Overall tendency: SYCOPHANTIC ↑")
        elif overall_mean < -0.1:
            print(f"Overall tendency: SKEPTICAL ↓")
        else:
            print(f"Overall tendency: NEUTRAL −")

        print(f"\nTendency breakdown:")
        print(f"  Sycophantic (shift > 1): {(df['tendency'] == 'Sycophantic').mean():.1%}")
        print(f"  Skeptical (shift < -1): {(df['tendency'] == 'Skeptical').mean():.1%}")
        print(f"  Neutral (-1 to 1): {(df['tendency'] == 'Neutral').mean():.1%}")

        print("\nModel-specific summary:")
        model_summary = df.groupby('model_display').agg({
            'score_shift': ['mean', 'median', 'std', 'count']
        }).round(3)
        print(model_summary)

        # Create visualizations
        print("\nCreating visualizations...")

        # Main comprehensive visualization
        fig1s = create_distribution_visualizations(df)

        # Model comparison (returns multiple figures)
        comparison_figs = create_model_comparison_plots(df)

        # NEW: Subject distribution analysis
        subject_fig = create_subject_distribution_analysis(df)

        # Save outputs
        output_dir = "sycophancy_analysis"
        os.makedirs(output_dir, exist_ok=True)

        # Save main figure
        for i, fig1 in enumerate(fig1s):
            fig1_path = os.path.join(output_dir, f"sycophancy_distributions_{i}.png")
            fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
            print(f"Saved main visualization to {fig1_path}")

        # Save comparison figures
        comparison_names = [
            "model_comparison_distributions.png",
            "model_comparison_pie_charts.png",
            "model_comparison_statistics.png"
        ]

        for fig, name in zip(comparison_figs, comparison_names):
            fig_path = os.path.join(output_dir, name)
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved {name} to {fig_path}")

        # Save subject analysis figure
        subject_fig_path = os.path.join(output_dir, "subject_distribution_analysis.png")
        subject_fig.savefig(subject_fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved subject analysis to {subject_fig_path}")

        # Save processed data
        csv_path = os.path.join(output_dir, "sycophancy_sample_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved sample data to {csv_path}")

        # Show plots
        plt.show()

        return df

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # You can specify max_workers here if needed, otherwise it auto-detects
    df = main(max_workers=None)  # Set to a specific number like max_workers=4 if needed