
# Prepare Dataset

This downloads the data from S3 and constructs balanced, trainable subsets/folds.
```
# We download datasets from S3 and balance them
python -m prep.dataset --model openai/gpt-4o --aggregation task-group --balance downsample
python -m prep.dataset --model google/gemma-3-4b-it --aggregation task-group --balance downsample
python -m prep.dataset --model google/gemma-3-12b-it --aggregation task-group --balance downsample
python -m prep.dataset --model google/gemma-3-27b-it --aggregation task-group --balance downsample

# We can verify this worked by doing
python -m prep.dataset --model google/gemma-3-4b-it --verify
```

# Prepare for Training
```
python -m prep.bundle --model google/gemma-3-4b-it --format chat --train games --train ascii
```

# Baselines

TODO
I would like something like
```
python -m baselines --model openai/gpt-4o --limit 100
```

```
@deprecated
python -m prep.train --dataset .data/openai/gpt_4o/mask --model gpt-4o-2024-08-06 --size 128

python -m prep.dataset --model openai/gpt-4o --aggregation generalization_map_3 --folds true --balance downsample
python -m prep.train --dataset .data/openai/gpt_4o/alibi --model gpt-4o-2024-08-06 --size 128

python -m prep.train --monitor 72327b83ea080548

# Data
Run everything:
```bash
inspect eval evals/sycophancy_tasks.py --limit 200 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/self_sycophancy_tasks.py --limit 200 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/mask_tasks.py --limit 200 --model openrouter/meta-llama/llama-3.3-70b-instruct
```
```

```bash
inspect eval evals/n_truths_and_a_lie_tasks.py --limit 20 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/capture_the_secret_tasks.py --limit 20 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/alibi_tasks.py --limit 20 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/sandbagging_ascii_capability_denial_tasks.py --limit 200 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/sandbagging_basic_tool_capability_denial_tasks.py --limit 200 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/sandbagging_unethical_tool_capability_denial_tasks.py --limit 200 --model openrouter/meta-llama/llama-3.3-70b-instruct
```