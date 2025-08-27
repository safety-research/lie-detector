import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Read the data
df = pd.read_csv('self-sycophancy-1.csv')

# Extract model name from the full path
df['model_short'] = df['model'].str.split('/').str[-1]

# Extract task category (harmfulness vs correctness)
df['task_category'] = df['task_name'].str.extract(r'self_sycophancy_(\w+)_')[0]

# Extract task subcategory (dataset)
df['dataset'] = df['task_name'].str.extract(r'self_sycophancy_\w+_(.+)_comprehensive')[0]

# Debug: Check what categories we actually have
print("Unique task categories found:", df['task_category'].unique())
print("Task category value counts:")
print(df['task_category'].value_counts())

# Set up the style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ========================================
# FIGURE 1: PER-DATASET ANALYSIS
# ========================================
fig1 = plt.figure(figsize=(20, 12))
gs1 = GridSpec(3, 3, figure=fig1, hspace=0.35, wspace=0.3)

datasets = df['dataset'].unique()
datasets_sorted = sorted(datasets)

# 1.1 SSI Scores by Dataset (Box plots)
ax1_1 = fig1.add_subplot(gs1[0, :])
df_sorted = df.sort_values('dataset')
bp = ax1_1.boxplot([df[df['dataset'] == d]['score_comprehensive_ssi_scorer_mean'].values
                    for d in datasets_sorted],
                   tick_labels=datasets_sorted, patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(datasets_sorted)))):
    patch.set_facecolor(color)
ax1_1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero line')
ax1_1.set_xlabel('Dataset', fontsize=12)
ax1_1.set_ylabel('Comprehensive SSI Score', fontsize=12)
ax1_1.set_title('SSI Score Distribution by Dataset', fontsize=14, fontweight='bold')
ax1_1.tick_params(axis='x', rotation=45)
ax1_1.grid(True, alpha=0.3)
ax1_1.legend()

# 1.2 Mean SSI Score by Dataset with error bars
ax1_2 = fig1.add_subplot(gs1[1, :])
dataset_stats = df.groupby('dataset')['score_comprehensive_ssi_scorer_mean'].agg(['mean', 'std', 'count'])
dataset_stats = dataset_stats.sort_values('mean')
x_pos = np.arange(len(dataset_stats))
ax1_2.bar(x_pos, dataset_stats['mean'], yerr=dataset_stats['std'], capsize=5,
          color=plt.cm.RdBu_r((dataset_stats['mean'] + 1) / 2))
ax1_2.set_xticks(x_pos)
ax1_2.set_xticklabels(dataset_stats.index, rotation=45, ha='right')
ax1_2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1_2.set_xlabel('Dataset', fontsize=12)
ax1_2.set_ylabel('Mean SSI Score', fontsize=12)
ax1_2.set_title('Mean SSI Score by Dataset (with std dev)', fontsize=14, fontweight='bold')
ax1_2.grid(True, alpha=0.3)
# Add count labels
for i, (idx, row) in enumerate(dataset_stats.iterrows()):
    ax1_2.text(i, row['mean'] + row['std'] + 0.05, f'n={int(row["count"])}',
               ha='center', va='bottom', fontsize=9)

# 1.3-1.8: Individual dataset distributions
for i, dataset in enumerate(datasets_sorted[:6]):  # Show first 6 datasets
    ax = fig1.add_subplot(gs1[2, i % 3])
    data = df[df['dataset'] == dataset]['score_comprehensive_ssi_scorer_mean']
    ax.hist(data, bins=15, edgecolor='black', alpha=0.7, color=plt.cm.Set3(i))
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=data.mean(), color='blue', linestyle='-', alpha=0.7, label=f'Mean: {data.mean():.2f}')
    ax.set_xlabel('SSI Score', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{dataset}\n(n={len(data)})', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig1.suptitle('Per-Dataset Analysis of Self-Sycophancy Scores', fontsize=16, fontweight='bold', y=0.995)
plt.show()

# ========================================
# FIGURE 2: PER-TASK ANALYSIS
# ========================================
fig2 = plt.figure(figsize=(20, 14))
gs2 = GridSpec(4, 2, figure=fig2, hspace=0.35, wspace=0.3)

tasks = df['task_name'].unique()

# 2.1 All tasks ranked by mean SSI score
ax2_1 = fig2.add_subplot(gs2[0:2, :])
task_means = df.groupby('task_name')['score_comprehensive_ssi_scorer_mean'].mean().sort_values()
y_pos = np.arange(len(task_means))
colors = ['red' if 'harmfulness' in task else 'blue' for task in task_means.index]
ax2_1.barh(y_pos, task_means.values, color=colors, alpha=0.7)
ax2_1.set_yticks(y_pos)
ax2_1.set_yticklabels([t.replace('self_sycophancy_', '').replace('_comprehensive', '')
                       for t in task_means.index], fontsize=9)
ax2_1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax2_1.set_xlabel('Mean SSI Score', fontsize=12)
ax2_1.set_title('All Tasks Ranked by Mean SSI Score (Red=Harmfulness, Blue=Correctness)',
                fontsize=14, fontweight='bold')
ax2_1.grid(True, alpha=0.3, axis='x')

# 2.2 Task category comparison
ax2_2 = fig2.add_subplot(gs2[2, 0])
category_data = [df[df['task_category'] == 'harmfulness']['score_comprehensive_ssi_scorer_mean'].values,
                 df[df['task_category'] == 'correctness']['score_comprehensive_ssi_scorer_mean'].values]
bp = ax2_2.boxplot(category_data, tick_labels=['Harmfulness', 'Correctness'], patch_artist=True)
bp['boxes'][0].set_facecolor('red')
bp['boxes'][0].set_alpha(0.5)
bp['boxes'][1].set_facecolor('blue')
bp['boxes'][1].set_alpha(0.5)
ax2_2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2_2.set_ylabel('SSI Score', fontsize=12)
ax2_2.set_title('Task Category Comparison', fontsize=13, fontweight='bold')
ax2_2.grid(True, alpha=0.3)

# 2.3 Variance by task
ax2_3 = fig2.add_subplot(gs2[2, 1])
task_variance = df.groupby('task_name')['score_comprehensive_ssi_scorer_mean'].std().sort_values(ascending=False).head(
    10)
ax2_3.barh(range(len(task_variance)), task_variance.values,
           color=plt.cm.plasma(np.linspace(0.2, 0.8, len(task_variance))))
ax2_3.set_yticks(range(len(task_variance)))
ax2_3.set_yticklabels([t.replace('self_sycophancy_', '').replace('_comprehensive', '')
                       for t in task_variance.index], fontsize=9)
ax2_3.set_xlabel('Standard Deviation', fontsize=12)
ax2_3.set_title('Top 10 Tasks by Score Variance', fontsize=13, fontweight='bold')
ax2_3.grid(True, alpha=0.3, axis='x')

# 2.4 Heatmap of tasks vs scoring metrics
ax2_4 = fig2.add_subplot(gs2[3, :])
scorer_cols = ['score_actual_vs_random_scorer_mean', 'score_calibrated_effect_scorer_mean',
               'score_choice_vs_prefill_scorer_mean', 'score_comprehensive_detection_scorer_mean']
task_scorer_means = df.groupby('task_name')[scorer_cols].mean()
im = ax2_4.imshow(task_scorer_means.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax2_4.set_yticks(range(len(task_scorer_means)))
ax2_4.set_yticklabels([t.replace('self_sycophancy_', '').replace('_comprehensive', '')
                       for t in task_scorer_means.index], fontsize=8)
ax2_4.set_xticks(range(len(scorer_cols)))
ax2_4.set_xticklabels([col.replace('score_', '').replace('_scorer_mean', '')
                       for col in scorer_cols], rotation=45, ha='right', fontsize=10)
ax2_4.set_title('Task Performance Across Different Scorer Metrics', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax2_4, label='Score')

fig2.suptitle('Per-Task Analysis of Self-Sycophancy', fontsize=16, fontweight='bold', y=0.995)
plt.show()

# ========================================
# FIGURE 3: PER-MODEL ANALYSIS
# ========================================
fig3 = plt.figure(figsize=(20, 16))
gs3 = GridSpec(4, 3, figure=fig3, hspace=0.35, wspace=0.3)

models = df['model_short'].unique()

# 3.1 All models ranked by mean SSI score
ax3_1 = fig3.add_subplot(gs3[0, :])
model_means = df.groupby('model_short')['score_comprehensive_ssi_scorer_mean'].mean().sort_values()
ax3_1.barh(range(len(model_means)), model_means.values,
           color=plt.cm.viridis(np.linspace(0.2, 0.9, len(model_means))))
ax3_1.set_yticks(range(len(model_means)))
ax3_1.set_yticklabels(model_means.index, fontsize=10)
ax3_1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax3_1.set_xlabel('Mean SSI Score', fontsize=12)
ax3_1.set_title('All Models Ranked by Mean SSI Score', fontsize=14, fontweight='bold')
ax3_1.grid(True, alpha=0.3, axis='x')
# Add sample size
for i, (model, score) in enumerate(model_means.items()):
    count = len(df[df['model_short'] == model])
    ax3_1.text(score + 0.02, i, f'n={count}', fontsize=8, va='center')

# 3.2 Model performance consistency (std dev)
ax3_2 = fig3.add_subplot(gs3[1, 0])
model_std = df.groupby('model_short')['score_comprehensive_ssi_scorer_mean'].agg(['std', 'count'])
model_std = model_std[model_std['count'] >= 3].sort_values('std', ascending=False)
ax3_2.bar(range(len(model_std)), model_std['std'].values,
          color=plt.cm.Reds(np.linspace(0.3, 0.9, len(model_std))))
ax3_2.set_xticks(range(len(model_std)))
ax3_2.set_xticklabels(model_std.index, rotation=45, ha='right', fontsize=9)
ax3_2.set_ylabel('Standard Deviation', fontsize=12)
ax3_2.set_title('Model Consistency\n(Models with ≥3 tests)', fontsize=13, fontweight='bold')
ax3_2.grid(True, alpha=0.3, axis='y')

# 3.3 Model performance by task category
ax3_3 = fig3.add_subplot(gs3[1, 1:])
model_category_means = df.pivot_table(values='score_comprehensive_ssi_scorer_mean',
                                      index='model_short',
                                      columns='task_category',
                                      aggfunc='mean')
model_category_means = model_category_means.dropna()

# Check which categories exist in the data
if 'correctness' in model_category_means.columns and 'harmfulness' in model_category_means.columns:
    x = np.arange(len(model_category_means))
    width = 0.35
    ax3_3.bar(x - width / 2, model_category_means['correctness'], width, label='Correctness', alpha=0.8)
    ax3_3.bar(x + width / 2, model_category_means['harmfulness'], width, label='Harmfulness', alpha=0.8)
    ax3_3.set_xticks(x)
    ax3_3.set_xticklabels(model_category_means.index, rotation=45, ha='right', fontsize=9)
elif len(model_category_means.columns) > 0:
    # If categories are different, plot whatever is available
    x = np.arange(len(model_category_means))
    width = 0.8 / len(model_category_means.columns)
    for i, col in enumerate(model_category_means.columns):
        offset = (i - len(model_category_means.columns) / 2) * width + width / 2
        ax3_3.bar(x + offset, model_category_means[col], width, label=col.capitalize(), alpha=0.8)
    ax3_3.set_xticks(x)
    ax3_3.set_xticklabels(model_category_means.index, rotation=45, ha='right', fontsize=9)

ax3_3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax3_3.set_ylabel('Mean SSI Score', fontsize=12)
ax3_3.set_title('Model Performance: Correctness vs Harmfulness', fontsize=13, fontweight='bold')
ax3_3.legend()
ax3_3.grid(True, alpha=0.3, axis='y')

# 3.4-3.9 Individual model performance across tasks (top 6 models by frequency)
model_counts = df['model_short'].value_counts().head(6)
for i, (model, count) in enumerate(model_counts.items()):
    ax = fig3.add_subplot(gs3[2 + i // 3, i % 3])
    model_data = df[df['model_short'] == model]

    # Create grouped bar chart for this model
    model_tasks = model_data.groupby('dataset')['score_comprehensive_ssi_scorer_mean'].mean().sort_values()
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_tasks)))
    ax.bar(range(len(model_tasks)), model_tasks.values, color=colors)
    ax.set_xticks(range(len(model_tasks)))
    ax.set_xticklabels(model_tasks.index, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('SSI Score', fontsize=10)
    ax.set_title(f'{model}\n(n={count} tests)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

fig3.suptitle('Per-Model Analysis of Self-Sycophancy', fontsize=16, fontweight='bold', y=0.995)
plt.show()

# ========================================
# FIGURE 4: DETAILED MODEL COMPARISON MATRIX
# ========================================
fig4 = plt.figure(figsize=(20, 12))
gs4 = GridSpec(2, 2, figure=fig4, hspace=0.3, wspace=0.3)

# 4.1 Model x Dataset heatmap
ax4_1 = fig4.add_subplot(gs4[0, :])
pivot_model_dataset = df.pivot_table(values='score_comprehensive_ssi_scorer_mean',
                                     index='dataset',
                                     columns='model_short',
                                     aggfunc='mean')
# Select models with most data points
top_models = df['model_short'].value_counts().head(8).index
pivot_subset = pivot_model_dataset[top_models]
im = ax4_1.imshow(pivot_subset.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax4_1.set_yticks(range(len(pivot_subset.index)))
ax4_1.set_yticklabels(pivot_subset.index, fontsize=10)
ax4_1.set_xticks(range(len(top_models)))
ax4_1.set_xticklabels(top_models, rotation=45, ha='right', fontsize=10)
ax4_1.set_title('Model Performance Heatmap: Dataset x Model', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax4_1, label='SSI Score')

# 4.2 Model performance spread (violin plot)
ax4_2 = fig4.add_subplot(gs4[1, 0])
models_with_enough_data = df['model_short'].value_counts()[df['model_short'].value_counts() >= 3].index
df_filtered = df[df['model_short'].isin(models_with_enough_data)]
model_order = df_filtered.groupby('model_short')['score_comprehensive_ssi_scorer_mean'].mean().sort_values().index
positions = range(len(model_order))
violin_data = [df_filtered[df_filtered['model_short'] == m]['score_comprehensive_ssi_scorer_mean'].values
               for m in model_order]
parts = ax4_2.violinplot(violin_data, positions=positions, vert=False, widths=0.7,
                         showmeans=True, showmedians=True)
ax4_2.set_yticks(positions)
ax4_2.set_yticklabels(model_order, fontsize=9)
ax4_2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax4_2.set_xlabel('SSI Score Distribution', fontsize=12)
ax4_2.set_title('Model Score Distributions\n(Models with ≥3 tests)', fontsize=13, fontweight='bold')
ax4_2.grid(True, alpha=0.3, axis='x')

# 4.3 Model robustness score (combining mean and inverse variance)
ax4_3 = fig4.add_subplot(gs4[1, 1])
model_stats = df.groupby('model_short')['score_comprehensive_ssi_scorer_mean'].agg(['mean', 'std', 'count'])
model_stats = model_stats[model_stats['count'] >= 3]
# Calculate robustness score: higher mean and lower std is better
# Normalize both metrics
model_stats['norm_mean'] = (model_stats['mean'] - model_stats['mean'].min()) / (
            model_stats['mean'].max() - model_stats['mean'].min())
model_stats['norm_consistency'] = 1 - (model_stats['std'] - model_stats['std'].min()) / (
            model_stats['std'].max() - model_stats['std'].min())
model_stats['robustness'] = (model_stats['norm_mean'] + model_stats['norm_consistency']) / 2
model_stats = model_stats.sort_values('robustness')

ax4_3.scatter(model_stats['mean'], model_stats['std'], s=model_stats['count'] * 20,
              c=model_stats['robustness'], cmap='RdYlGn', alpha=0.6)
ax4_3.set_xlabel('Mean SSI Score', fontsize=12)
ax4_3.set_ylabel('Std Dev of SSI Score', fontsize=12)
ax4_3.set_title('Model Robustness Analysis\n(Size = sample count, Color = robustness)',
                fontsize=13, fontweight='bold')
ax4_3.axhline(y=model_stats['std'].median(), color='gray', linestyle='--', alpha=0.3)
ax4_3.axvline(x=model_stats['mean'].median(), color='gray', linestyle='--', alpha=0.3)
ax4_3.grid(True, alpha=0.3)
# Add labels for models
for idx, row in model_stats.iterrows():
    ax4_3.annotate(idx, (row['mean'], row['std']), fontsize=8, alpha=0.7)
plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn'), ax=ax4_3, label='Robustness Score')

fig4.suptitle('Detailed Model Comparison Matrix', fontsize=16, fontweight='bold', y=0.995)
plt.show()

# ========================================
# SUMMARY STATISTICS
# ========================================
print("\n" + "=" * 80)
print("COMPREHENSIVE SUMMARY STATISTICS")
print("=" * 80)

print("\n1. PER-DATASET STATISTICS:")
print("-" * 40)
dataset_summary = df.groupby('dataset')['score_comprehensive_ssi_scorer_mean'].agg(
    ['mean', 'median', 'std', 'min', 'max', 'count'])
dataset_summary = dataset_summary.sort_values('mean', ascending=False)
print(dataset_summary.to_string())

print("\n2. PER-TASK STATISTICS:")
print("-" * 40)
task_summary = df.groupby('task_name')['score_comprehensive_ssi_scorer_mean'].agg(['mean', 'median', 'std', 'count'])
task_summary = task_summary.sort_values('mean', ascending=False)
print(task_summary.head(10).to_string())

print("\n3. PER-MODEL STATISTICS:")
print("-" * 40)
model_summary = df.groupby('model_short')['score_comprehensive_ssi_scorer_mean'].agg(
    ['mean', 'median', 'std', 'min', 'max', 'count'])
model_summary = model_summary.sort_values('mean', ascending=False)
print(model_summary.to_string())

print("\n4. TASK CATEGORY COMPARISON:")
print("-" * 40)
for category in df['task_category'].unique():
    data = df[df['task_category'] == category]['score_comprehensive_ssi_scorer_mean']
    print(f"\n{category.upper()}:")
    print(f"  Mean: {data.mean():.4f}")
    print(f"  Median: {data.median():.4f}")
    print(f"  Std Dev: {data.std():.4f}")
    print(f"  Sample Size: {len(data)}")

print("\n5. MOST SYCOPHANTIC COMBINATIONS:")
print("-" * 40)
top_syco = df.nlargest(10, 'score_comprehensive_ssi_scorer_mean')[
    ['model_short', 'dataset', 'score_comprehensive_ssi_scorer_mean']]
print(top_syco.to_string(index=False))

print("\n6. LEAST SYCOPHANTIC COMBINATIONS:")
print("-" * 40)
bottom_syco = df.nsmallest(10, 'score_comprehensive_ssi_scorer_mean')[
    ['model_short', 'dataset', 'score_comprehensive_ssi_scorer_mean']]
print(bottom_syco.to_string(index=False))

print("\n" + "=" * 80)