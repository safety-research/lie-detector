#from inspect_ai.analysis.beta import evals_df, log_viewer, model_into, prepare
from inspect_ai.analysis import model_info, prepare, evals_df

# Load the data
df = evals_df("../logs/self-sycophancy-1")
df = prepare(df,
    [
        model_info(),
        #log_viewer("eval", {"logs": "../"})
    ]
)[[
    'task_name',
    'model',
    'task_id',
    'dataset_name',
    'score_headline_name',
    'score_headline_metric',
    'score_headline_value',
    'score_headline_stderr',
    "score_actual_vs_random_scorer_mean",
    "score_calibrated_effect_scorer_mean",
    "score_choice_vs_prefill_scorer_mean",
    "score_comprehensive_detection_scorer_mean",
    "score_comprehensive_ssi_scorer_mean",
    "score_comprehensive_ssi_scorer_std",
    "score_consistency_scorer_mean",
    "score_position_bias_scorer_mean",
    "model_organization_name",
    "model_display_name",
]]

# Deduplicate by keeping the last (most recent) occurrence of each task_name/model combination
df_dedup = df.drop_duplicates(subset=['task_name', 'model'], keep='last')

# Print deduplication statistics
original_count = len(df)
dedup_count = len(df_dedup)
print(f"Original rows: {original_count}")
print(f"After deduplication: {dedup_count}")
print(f"Removed duplicates: {original_count - dedup_count}")

# Show which task/model combinations had duplicates
duplicates = df[df.duplicated(subset=['task_name', 'model'], keep=False)]
if not duplicates.empty:
    print("\nDuplicated task/model combinations:")
    duplicate_groups = duplicates.groupby(['task_name', 'model']).size()
    for (task, model), count in duplicate_groups.items():
        print(f"  - {task} / {model}: {count} occurrences")

# Save the deduplicated data
df_dedup.to_csv("self-sycophancy-1-dedup.csv", index=False)
