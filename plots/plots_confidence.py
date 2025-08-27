#!/usr/bin/env python3
"""
Visualizing self-sycophancy experiment results with error bars.

Usage:
  python plot_self_sycophancy_results.py --csv results.csv --out plots/ --n 100

What it makes:
  - Calibrated effect by model (per dataset) with error bars
  - Aggregated calibrated effect by model family (per dataset) with error bars
  - Scatter plots with confidence ellipses
  - Detection rate vs calibrated effect
  - Prints summary stats per domain/dataset/model family

Notes:
  - Error bars show either SSI std or SE approximation
  - Can toggle between showing 1 SE or 1.96 SE (95% CI)
"""

import argparse
import os
from math import sqrt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Define consistent color schemes
MODEL_FAMILY_COLORS = {
    "Anthropic": "#E74C3C",  # Red
    "OpenAI": "#3498DB",  # Blue
    "Google": "#2ECC71",  # Green
    "OSS": "#F39C12",  # Orange
    "Other": "#9B59B6"  # Purple
}

DOMAIN_COLORS = {
    "harmfulness": "#1ABC9C",  # Teal
    "correctness": "#E67E22",  # Dark Orange
    "unknown": "#95A5A6"  # Gray
}

# Marker styles for domains
DOMAIN_MARKERS = {
    "harmfulness": "o",
    "correctness": "s",
    "unknown": "^"
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="self-sycophancy-1-dedup.csv", help="Path to CSV with results")
    p.add_argument("--out", default="plots", help="Output directory for charts")
    p.add_argument("--n", type=int, default=50000, help="Items per run (default 100)")
    p.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    p.add_argument("--style", default="whitegrid", help="Seaborn style")
    p.add_argument("--error-type", default="se", choices=["se", "std", "ci95"],
                   help="Type of error bars: se (standard error), std (standard deviation), ci95 (95% CI)")
    p.add_argument("--show-ssi-std", action="store_true",
                   help="Show SSI std instead of calibrated effect SE")
    return p.parse_args()


def ensure_dir(pth):
    os.makedirs(pth, exist_ok=True)
    return pth


def parse_domain_dataset(task_name: str):
    """
    Expected task_name format:
      self_sycophancy_<domain>_<dataset>_comprehensive
    e.g., self_sycophancy_harmfulness_commonsense_comprehensive
          self_sycophancy_correctness_arc_challenge_comprehensive
    """
    if not isinstance(task_name, str):
        return "unknown", "unknown"
    parts = task_name.split("_")
    if len(parts) < 4:
        return "unknown", "unknown"
    domain = parts[2]
    # dataset can be multi-token (e.g., arc_challenge); strip trailing 'comprehensive' if present
    ds_tokens = parts[3:]
    if ds_tokens and ds_tokens[-1].lower() == "comprehensive":
        ds_tokens = ds_tokens[:-1]
    dataset = "_".join(ds_tokens) if ds_tokens else "unknown"
    return domain, dataset


def parse_model_family(model: str):
    if not isinstance(model, str):
        return "Other"
    m = model.lower()
    if "/openai/gpt-oss" in m:
        return "OSS"
    if "/openai/" in m:
        return "OpenAI"
    if "/anthropic/" in m:
        return "Anthropic"
    if "/google/" in m:
        return "Google"
    return "Other"


def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_derived(df, n_items=100):
    # domain/dataset
    parsed = df["task_name"].apply(parse_domain_dataset)
    df["domain"] = [d for d, _ in parsed]
    df["dataset"] = [s for _, s in parsed]
    # family
    df["model_family"] = df["model"].apply(parse_model_family)

    # coerce numerics
    num_cols = [
        "score_calibrated_effect_scorer_mean",
        "score_comprehensive_ssi_scorer_mean",
        "score_comprehensive_ssi_scorer_std",
        "score_choice_vs_prefill_scorer_mean",
        "score_actual_vs_random_scorer_mean",
        "score_comprehensive_detection_scorer_mean",
        "score_consistency_scorer_mean",
        "score_position_bias_scorer_mean",
    ]
    df = coerce_numeric(df, num_cols)

    # position bias SD (sqrt of variance)
    df["position_bias_sd"] = np.sqrt(df["score_position_bias_scorer_mean"].clip(lower=0))

    # approximate SE for calibrated effect using raw SSI std and N
    df["ssi_se_approx"] = df["score_comprehensive_ssi_scorer_std"] / np.sqrt(max(n_items, 1))
    # z approx and significance flag
    df["z_cal_approx"] = df["score_calibrated_effect_scorer_mean"] / df["ssi_se_approx"].replace(0, np.nan)
    df["sig_approx"] = df["z_cal_approx"].abs() > 1.96

    # nicer labels
    df["model_label"] = df["model_display_name"].fillna(df["model"])
    df["ds_label"] = df["dataset"].str.replace("_", " ").str.title()
    df["domain_label"] = df["domain"].str.title()

    return df


def get_error_value(row, error_type="se", show_ssi_std=False):
    """Get the appropriate error value based on settings."""
    if show_ssi_std:
        # Show raw SSI standard deviation
        return row.get("score_comprehensive_ssi_scorer_std", 0)
    else:
        # Show error for calibrated effect
        se = row.get("ssi_se_approx", 0)
        if error_type == "se":
            return se
        elif error_type == "std":
            # If we have actual std for calibrated effect, use it; otherwise use SSI std
            return row.get("score_comprehensive_ssi_scorer_std", se)
        elif error_type == "ci95":
            return 1.96 * se
    return 0


def bar_calibrated_by_model(df, outdir, dpi, error_type="se", show_ssi_std=False):
    """Bar chart with error bars for calibrated effect by model."""
    cols = ["domain", "dataset", "model_label", "model_family",
            "score_calibrated_effect_scorer_mean", "ssi_se_approx",
            "score_comprehensive_ssi_scorer_std"]
    d = df.dropna(subset=["score_calibrated_effect_scorer_mean"])[cols].copy()
    if d.empty:
        return

    for (domain, dataset), g in sorted(d.groupby(["domain", "dataset"])):
        if g.empty:
            continue
        g = g.sort_values("score_calibrated_effect_scorer_mean", ascending=True)

        # Map colors
        colors = [MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in g["model_family"]]

        plt.figure(figsize=(12, max(4, 0.35 * len(g))))
        ax = plt.gca()

        # Get error values
        errors = [get_error_value(row, error_type, show_ssi_std) for _, row in g.iterrows()]

        # Create horizontal bar chart with error bars
        y_pos = np.arange(len(g))
        bars = ax.barh(y_pos, g["score_calibrated_effect_scorer_mean"],
                       xerr=errors, color=colors, alpha=0.8,
                       error_kw={'elinewidth': 1, 'capsize': 3, 'alpha': 0.7})

        ax.axvline(0, color="k", lw=1, alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(g["model_label"])

        # Title with error bar type
        error_label = {"se": "SE", "std": "SD", "ci95": "95% CI"}[error_type]
        if show_ssi_std:
            error_label = "SSI SD"
        ax.set_title(
            f"Calibrated Effect by Model (±{error_label})\n{domain.title()} • {dataset.replace('_', ' ').title()}")
        ax.set_xlabel("Calibrated effect (SSI_actual − mean(SSI_forced))")

        # Annotate values with significance indicators
        for i, (idx, row) in enumerate(g.iterrows()):
            val = row["score_calibrated_effect_scorer_mean"]
            sig_marker = "*" if abs(row.get("z_cal_approx", 0)) > 1.96 else ""
            ax.text(
                val + (0.02 if val >= 0 else -0.02),
                i,
                f"{val:.2f}{sig_marker}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
                color="black"
            )

        # Add legend for model families
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=MODEL_FAMILY_COLORS[fam], label=fam)
                           for fam in sorted(g["model_family"].unique())]
        ax.legend(handles=legend_elements, title="Family", bbox_to_anchor=(1.02, 1), loc="upper left")

        # Add note about significance
        if not show_ssi_std:
            ax.text(0.02, 0.02, "* p < 0.05 (approx)", transform=ax.transAxes,
                    fontsize=8, alpha=0.6)

        plt.tight_layout()
        fname = os.path.join(outdir, f"calibrated_by_model_{domain}_{dataset}.png")
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close()


def grouped_bar_family(df, outdir, dpi, error_type="se"):
    """Grouped bar chart with error bars for model families."""
    cols = ["domain", "dataset", "model_family", "score_calibrated_effect_scorer_mean",
            "score_comprehensive_ssi_scorer_std", "ssi_se_approx"]
    d = df.dropna(subset=["score_calibrated_effect_scorer_mean"])[cols].copy()
    if d.empty:
        return

    # Aggregate with both mean and std/se
    agg = d.groupby(["domain", "dataset", "model_family"], as_index=False).agg({
        "score_calibrated_effect_scorer_mean": ["mean", "std", "sem", "count"]
    })
    agg.columns = ["domain", "dataset", "model_family", "mean", "std", "sem", "count"]

    for domain, g in sorted(agg.groupby("domain")):
        plt.figure(figsize=(12, 6))

        # Get sorted list of model families for consistent ordering
        families = sorted(g["model_family"].unique())
        palette = {fam: MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in families}

        # Determine error values
        if error_type == "se":
            g["error"] = g["sem"]
            error_label = "SE"
        elif error_type == "std":
            g["error"] = g["std"]
            error_label = "SD"
        elif error_type == "ci95":
            g["error"] = 1.96 * g["sem"]
            error_label = "95% CI"

        # Create bar plot with error bars
        x = np.arange(len(g["dataset"].unique()))
        width = 0.8 / len(families)

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, family in enumerate(families):
            family_data = g[g["model_family"] == family]
            positions = []
            heights = []
            errors = []

            for j, dataset in enumerate(sorted(g["dataset"].unique())):
                dataset_family = family_data[family_data["dataset"] == dataset]
                if not dataset_family.empty:
                    positions.append(j + (i - len(families) / 2 + 0.5) * width)
                    heights.append(dataset_family["mean"].values[0])
                    errors.append(dataset_family["error"].values[0])

            if positions:
                bars = ax.bar(positions, heights, width,
                              label=family, color=palette[family], alpha=0.8,
                              yerr=errors, capsize=3, error_kw={'elinewidth': 1})

        ax.axhline(0, color="k", lw=1, alpha=0.6)
        ax.set_title(f"Calibrated Effect by Model Family (±{error_label}) • {domain.title()}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Mean calibrated effect")
        ax.set_xticks(x)
        ax.set_xticklabels(sorted(g["dataset"].unique()), rotation=30, ha="right")
        ax.legend(title="Family", bbox_to_anchor=(1.02, 1), loc="upper left")

        # Add sample sizes as text
        for i, family in enumerate(families):
            family_data = g[g["model_family"] == family]
            for j, dataset in enumerate(sorted(g["dataset"].unique())):
                dataset_family = family_data[family_data["dataset"] == dataset]
                if not dataset_family.empty:
                    count = dataset_family["count"].values[0]
                    pos = j + (i - len(families) / 2 + 0.5) * width
                    height = dataset_family["mean"].values[0]
                    error = dataset_family["error"].values[0]
                    ax.text(pos, height + error + 0.01, f'n={int(count)}',
                            ha='center', va='bottom', fontsize=7, alpha=0.6)

        plt.tight_layout()
        fname = os.path.join(outdir, f"calibrated_by_family_{domain}.png")
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close()


def scatter_effect_vs_anchor_with_ellipse(df, outdir, dpi):
    """Scatter plot with confidence ellipses for groups."""
    d = df.dropna(subset=["score_calibrated_effect_scorer_mean", "position_bias_sd"]).copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    families = sorted(d["model_family"].unique())
    domains = sorted(d["domain"].unique())

    palette = {fam: MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in families}
    markers = {dom: DOMAIN_MARKERS.get(dom, "^") for dom in domains}

    # Main scatter plot
    sns.scatterplot(
        data=d,
        x="position_bias_sd",
        y="score_calibrated_effect_scorer_mean",
        hue="model_family",
        style="domain",
        palette=palette,
        hue_order=families,
        style_order=domains,
        markers=markers,
        s=100,
        alpha=0.7,
        ax=ax
    )

    # Add confidence ellipses for each model family
    for family in families:
        family_data = d[d["model_family"] == family]
        if len(family_data) >= 3:  # Need at least 3 points for ellipse
            x = family_data["position_bias_sd"].values
            y = family_data["score_calibrated_effect_scorer_mean"].values

            # Calculate ellipse parameters
            mean_x, mean_y = np.mean(x), np.mean(y)
            cov = np.cov(x, y)
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)

            # 95% confidence ellipse (2.45 sigma for 2D)
            ell = Ellipse(xy=(mean_x, mean_y),
                          width=lambda_[0] * 2 * 2.45, height=lambda_[1] * 2 * 2.45,
                          angle=np.degrees(np.arctan2(*v[:, 0][::-1])),
                          facecolor=palette[family], alpha=0.1,
                          edgecolor=palette[family], linewidth=1.5)
            ax.add_patch(ell)

    ax.axhline(0, color="k", lw=1, alpha=0.6)
    ax.set_title("Calibrated Effect vs Position Bias (with 95% confidence ellipses)")
    ax.set_xlabel("Position bias (SD)")
    ax.set_ylabel("Calibrated effect")

    # Annotate outliers
    d["abs_effect"] = d["score_calibrated_effect_scorer_mean"].abs()
    for _, row in d.nlargest(3, "abs_effect").iterrows():
        ax.annotate(
            f"{row['model_family'][:3]} | {row['dataset'][:8]}",
            (row["position_bias_sd"], row["score_calibrated_effect_scorer_mean"]),
            textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7
        )

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "effect_vs_position_bias.png"), dpi=dpi, bbox_inches='tight')
    plt.close()


def scatter_effect_vs_prefill(df, outdir, dpi):
    """Scatter plot with regression confidence interval."""
    d = df.dropna(subset=["score_calibrated_effect_scorer_mean", "score_choice_vs_prefill_scorer_mean"]).copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    families = sorted(d["model_family"].unique())
    domains = sorted(d["domain"].unique())

    palette = {fam: MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in families}
    markers = {dom: DOMAIN_MARKERS.get(dom, "^") for dom in domains}

    sns.scatterplot(
        data=d,
        x="score_choice_vs_prefill_scorer_mean",
        y="score_calibrated_effect_scorer_mean",
        hue="model_family",
        style="domain",
        palette=palette,
        hue_order=families,
        style_order=domains,
        markers=markers,
        s=100,
        alpha=0.7,
        ax=ax
    )

    # Add regression with confidence interval
    sns.regplot(
        data=d, x="score_choice_vs_prefill_scorer_mean", y="score_calibrated_effect_scorer_mean",
        scatter=False, color="gray", line_kws={"alpha": 0.6},
        ci=95, ax=ax  # 95% confidence interval
    )

    ax.axhline(0, color="k", lw=1, alpha=0.6)
    ax.set_title("Calibrated Effect vs Prefill Amplification (with 95% CI regression)")
    ax.set_xlabel("Prefill amplification (post_prefilled − post_choice)")
    ax.set_ylabel("Calibrated effect")

    # Calculate and display correlation
    corr = d[["score_choice_vs_prefill_scorer_mean", "score_calibrated_effect_scorer_mean"]].corr().iloc[0, 1]
    ax.text(0.02, 0.98, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "effect_vs_prefill.png"), dpi=dpi, bbox_inches='tight')
    plt.close()


def scatter_effect_vs_random(df, outdir, dpi):
    d = df.dropna(subset=["score_calibrated_effect_scorer_mean", "score_actual_vs_random_scorer_mean"]).copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    families = sorted(d["model_family"].unique())
    domains = sorted(d["domain"].unique())

    palette = {fam: MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in families}
    markers = {dom: DOMAIN_MARKERS.get(dom, "^") for dom in domains}

    sns.scatterplot(
        data=d,
        x="score_actual_vs_random_scorer_mean",
        y="score_calibrated_effect_scorer_mean",
        hue="model_family",
        style="domain",
        palette=palette,
        hue_order=families,
        style_order=domains,
        markers=markers,
        s=100,
        alpha=0.7,
        ax=ax
    )
    ax.axvline(0, color="k", lw=1, alpha=0.6)
    ax.axhline(0, color="k", lw=1, alpha=0.6)
    ax.set_title("Own-Choice vs Random-Label Bias")
    ax.set_xlabel("SSI(actual) − SSI(random prefill)")
    ax.set_ylabel("Calibrated effect")

    # Add quadrant labels
    ax.text(0.95, 0.95, "High both", transform=ax.transAxes, ha='right', va='top',
            fontsize=8, alpha=0.5)
    ax.text(0.05, 0.95, "High calibrated\nLow random", transform=ax.transAxes, ha='left', va='top',
            fontsize=8, alpha=0.5)
    ax.text(0.05, 0.05, "Low both", transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8, alpha=0.5)
    ax.text(0.95, 0.05, "Low calibrated\nHigh random", transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, alpha=0.5)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "effect_vs_random.png"), dpi=dpi, bbox_inches='tight')
    plt.close()


def scatter_detection_vs_effect(df, outdir, dpi):
    d = df.dropna(subset=["score_calibrated_effect_scorer_mean", "score_comprehensive_detection_scorer_mean"]).copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    families = sorted(d["model_family"].unique())
    domains = sorted(d["domain"].unique())

    palette = {fam: MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in families}
    markers = {dom: DOMAIN_MARKERS.get(dom, "^") for dom in domains}

    sns.scatterplot(
        data=d,
        x="score_calibrated_effect_scorer_mean",
        y="score_comprehensive_detection_scorer_mean",
        hue="model_family",
        style="domain",
        palette=palette,
        hue_order=families,
        style_order=domains,
        markers=markers,
        s=100,
        alpha=0.7,
        ax=ax
    )

    # Add vertical lines for effect thresholds
    ax.axvline(0.5, color="r", ls="--", lw=1, alpha=0.5, label="Effect = 0.5")
    ax.axvline(0, color="k", ls="-", lw=1, alpha=0.3)

    # Add horizontal line for detection threshold
    ax.axhline(0.5, color="b", ls="--", lw=1, alpha=0.5, label="Detection = 0.5")

    ax.set_title("Detection Rate vs Calibrated Effect")
    ax.set_xlabel("Calibrated effect")
    ax.set_ylabel("Detection rate (fraction > 0.5)")

    # Add correlation
    corr = d[["score_calibrated_effect_scorer_mean", "score_comprehensive_detection_scorer_mean"]].corr().iloc[0, 1]
    ax.text(0.02, 0.98, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "detection_vs_effect.png"), dpi=dpi, bbox_inches='tight')
    plt.close()


def bar_consistency(df, outdir, dpi):
    """Box plot with individual points overlaid for consistency."""
    d = df.dropna(subset=["score_consistency_scorer_mean"]).copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    families = sorted(d["model_family"].unique())
    palette = {fam: MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in families}

    # Create box plot
    sns.boxplot(
        data=d,
        x="domain",
        y="score_consistency_scorer_mean",
        hue="model_family",
        palette=palette,
        hue_order=families,
        ax=ax,
        showfliers=False  # Don't show outliers as we'll add all points
    )

    # Overlay individual points
    sns.stripplot(
        data=d,
        x="domain",
        y="score_consistency_scorer_mean",
        hue="model_family",
        palette=palette,
        hue_order=families,
        dodge=True,
        size=3,
        alpha=0.5,
        ax=ax,
        legend=False  # Don't duplicate legend
    )

    ax.set_title("Consistency (higher is better) by Domain and Model Family")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Consistency (≈ 1/(1 + drift))")

    # Add horizontal line at y=0.5 for reference
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, label='Baseline (0.5)')

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "consistency_by_domain_family.png"), dpi=dpi, bbox_inches='tight')
    plt.close()


def print_summary(df, n_items):
    def fmt(x):
        return "n/a" if pd.isna(x) else f"{x:.3f}"

    print("\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY STATISTICS WITH ERROR ESTIMATES")
    print("=" * 80)

    # By domain / dataset with error estimates
    print("\n=== By Domain/Dataset (with error estimates) ===")
    gcols = ["domain", "dataset"]
    agg = df.groupby(gcols).agg(
        mean_calibrated=("score_calibrated_effect_scorer_mean", "mean"),
        sd_calibrated=("score_calibrated_effect_scorer_mean", "std"),
        se_calibrated=("score_calibrated_effect_scorer_mean", "sem"),
        mean_ssi_std=("score_comprehensive_ssi_scorer_std", "mean"),
        mean_position_var=("score_position_bias_scorer_mean", "mean"),
        mean_position_sd=("position_bias_sd", "mean"),
        mean_prefill=("score_choice_vs_prefill_scorer_mean", "mean"),
        mean_detection=("score_comprehensive_detection_scorer_mean", "mean"),
        mean_consistency=("score_consistency_scorer_mean", "mean"),
        n=("score_calibrated_effect_scorer_mean", "count")
    ).reset_index()

    for _, row in agg.iterrows():
        ci95 = 1.96 * row['se_calibrated']
        print(f"\n{row['domain']}/{row['dataset']}:")
        print(f"  Calibrated effect: {fmt(row['mean_calibrated'])} ± {fmt(row['se_calibrated'])} SE")
        print(f"  95% CI: [{fmt(row['mean_calibrated'] - ci95)}, {fmt(row['mean_calibrated'] + ci95)}]")
        print(f"  SSI std (raw): {fmt(row['mean_ssi_std'])}")
        print(f"  Position bias SD: {fmt(row['mean_position_sd'])}")
        print(f"  Prefill amp: {fmt(row['mean_prefill'])}")
        print(f"  Detection: {fmt(row['mean_detection'])}")
        print(f"  Consistency: {fmt(row['mean_consistency'])}")
        print(f"  N runs: {int(row['n'])}")

    # By model family with confidence intervals
    print("\n=== Calibrated effect by model family (with 95% CI) ===")
    fam_agg = df.groupby("model_family").agg({
        "score_calibrated_effect_scorer_mean": ["mean", "std", "sem", "count"]
    })
    fam_agg.columns = ["mean", "std", "sem", "count"]
    fam_agg = fam_agg.sort_values("mean", ascending=False)

    for family, row in fam_agg.iterrows():
        ci95 = 1.96 * row['sem']
        print(f"- {family}: {fmt(row['mean'])} ± {fmt(row['sem'])} SE "
              f"[95% CI: {fmt(row['mean'] - ci95)}, {fmt(row['mean'] + ci95)}], n={int(row['count'])}")

    # Count of approximate significant positives/negatives
    sig = df.dropna(subset=["z_cal_approx"])
    if not sig.empty:
        pos = ((sig["z_cal_approx"] > 1.96) & (sig["score_calibrated_effect_scorer_mean"] > 0)).sum()
        neg = ((sig["z_cal_approx"] < -1.96) & (sig["score_calibrated_effect_scorer_mean"] < 0)).sum()
        tot = len(sig)
        print(f"\n=== Significance Analysis (approx using SSI std, N={n_items}) ===")
        print(f"Significant positive effects (z>1.96): {pos}/{tot} ({100 * pos / tot:.1f}%)")
        print(f"Significant negative effects (z<-1.96): {neg}/{tot} ({100 * neg / tot:.1f}%)")
        print(f"Non-significant: {tot - pos - neg}/{tot} ({100 * (tot - pos - neg) / tot:.1f}%)")

        # Top significant effects
        print("\nTop 5 significant positive effects:")
        top_pos = sig[sig["z_cal_approx"] > 1.96].nlargest(5, "z_cal_approx")
        for _, row in top_pos.iterrows():
            print(f"  {row['model_family']}/{row['dataset']}: z={row['z_cal_approx']:.2f}, "
                  f"effect={row['score_calibrated_effect_scorer_mean']:.3f}")

        print("\nTop 5 significant negative effects:")
        top_neg = sig[sig["z_cal_approx"] < -1.96].nsmallest(5, "z_cal_approx")
        for _, row in top_neg.iterrows():
            print(f"  {row['model_family']}/{row['dataset']}: z={row['z_cal_approx']:.2f}, "
                  f"effect={row['score_calibrated_effect_scorer_mean']:.3f}")


def main():
    args = parse_args()
    sns.set_style(args.style)
    ensure_dir(args.out)

    # Load
    df = pd.read_csv(args.csv)

    # Prepare
    df = add_derived(df, n_items=args.n)

    # Charts with error bars
    bar_calibrated_by_model(df, args.out, args.dpi, args.error_type, args.show_ssi_std)
    grouped_bar_family(df, args.out, args.dpi, args.error_type)

    # Enhanced scatter plots
    scatter_effect_vs_anchor_with_ellipse(df, args.out, args.dpi)
    scatter_effect_vs_prefill(df, args.out, args.dpi)
    scatter_effect_vs_random(df, args.out, args.dpi)
    scatter_detection_vs_effect(df, args.out, args.dpi)
    bar_consistency(df, args.out, args.dpi)

    # Summary text with error estimates
    print_summary(df, args.n)

    print(f"\nCharts saved to: {os.path.abspath(args.out)}")

    # Print settings used
    print(f"\n=== Settings Used ===")
    print(f"Error bar type: {args.error_type}")
    print(f"Show SSI std instead of calibrated SE: {args.show_ssi_std}")
    print(f"N items per run: {args.n}")

    # Print color legend for reference
    print("\n=== Color Scheme Reference ===")
    print("Model Families:")
    for family, color in sorted(MODEL_FAMILY_COLORS.items()):
        print(f"  {family}: {color}")
    print("\nDomains:")
    for domain, color in sorted(DOMAIN_COLORS.items()):
        print(f"  {domain}: {color} (marker: {DOMAIN_MARKERS.get(domain, '^')})")


if __name__ == "__main__":
    main()