# Self-Sycophancy in Large Language Models: Empirical Analysis
======================================================================

## Executive Summary
----------------------------------------
Across 4300 evaluations, we find that:
- **55.3%** of reassessments show statistically significant self-bias
- When significant, the median effect size is **d = -1.41**
- **9.1%** of significant effects are self-sycophantic (vs. 52.7% self-skeptical)
- **61.7%** of significant effects are large (|d| ≥ 0.8)

## Key Findings
----------------------------------------
### Finding 1: Widespread Self-Sycophancy
The majority of model self-reassessments (55.3%) exhibit statistically significant bias, indicating a systemic metacognitive issue.

### Finding 2: Effect Magnitude
When self-bias occurs, it tends to be large in magnitude (median |d| = 1.41), suggesting meaningful practical impact.

### Finding 3: Directional Asymmetry
Models show relatively balanced self-sycophancy (9.1%) and self-skepticism (52.7%), though individual variations exist.

## Model-Specific Analysis
----------------------------------------
### openrouter/openai/gpt-oss-120b
- Prevalence: 41.2%
- Median effect (when sig.): d = -1.43
- Direction: 9.6% sycophantic, 56.2% skeptical

### openrouter/openai/gpt-oss-20b
- Prevalence: 42.0%
- Median effect (when sig.): d = -1.66
- Direction: 11.7% sycophantic, 62.4% skeptical

### openrouter/openai/gpt-5-nano
- Prevalence: 47.5%
- Median effect (when sig.): d = -1.79
- Direction: 5.9% sycophantic, 77.4% skeptical

### openrouter/openai/gpt-5-mini
- Prevalence: 51.1%
- Median effect (when sig.): d = -1.79
- Direction: 4.1% sycophantic, 70.7% skeptical

### openrouter/openai/gpt-5
- Prevalence: 60.7%
- Median effect (when sig.): d = -1.79
- Direction: 4.2% sycophantic, 69.2% skeptical

### openrouter/anthropic/claude-sonnet-4
- Prevalence: 55.3%
- Median effect (when sig.): d = 0.00
- Direction: 12.4% sycophantic, 46.0% skeptical

### openrouter/anthropic/claude-opus-4.1
- Prevalence: 75.9%
- Median effect (when sig.): d = -2.00
- Direction: 4.2% sycophantic, 62.5% skeptical

### openrouter/anthropic/claude-3.5-haiku
- Prevalence: 62.6%
- Median effect (when sig.): d = 0.00
- Direction: 20.0% sycophantic, 33.9% skeptical

### openrouter/google/gemini-2.5-pro
- Prevalence: 28.6%
- Median effect (when sig.): d = 0.00
- Direction: 41.9% sycophantic, 24.2% skeptical

### openrouter/google/gemma-3-27b-it
- Prevalence: 74.0%
- Median effect (when sig.): d = 0.00
- Direction: 5.8% sycophantic, 24.6% skeptical

## Implications
----------------------------------------
1. **For AI Safety**: Self-sycophancy could amplify errors in recursive or iterative systems
2. **For Deployment**: Models should not be used for self-validation without external oversight
3. **For Evaluation**: Metacognitive assessment should be standard in model evaluation

## Statistical Notes
----------------------------------------
- Effect sizes reported using Cohen's d
- Statistical significance determined via convergent evidence (z-test, t-test, bootstrap CI)
- All tests use α = 0.05
- Total samples analyzed: 4300
