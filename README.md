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