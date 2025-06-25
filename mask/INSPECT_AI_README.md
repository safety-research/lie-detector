# MASK for Inspect AI

This directory contains a complete port of the MASK (Model Alignment between Statements and Knowledge) benchmark to the Inspect AI framework.

## Overview

MASK evaluates honesty in large language models by testing whether they contradict their own beliefs when pressured to lie. This implementation provides:

- **Dataset readers** for MASK CSV format
- **Solvers** for belief elicitation and lying scenarios  
- **Scorers** for binary and numerical proposition evaluation
- **Tasks** for complete MASK evaluation pipelines

## Quick Start

### Installation

First, ensure you have Inspect AI installed:

```bash
pip install inspect-ai
```

### Basic Usage

Run a complete MASK evaluation:

```python
from inspect_mask_task import mask_full_evaluation_task
from inspect_ai import eval

# Create and run task
task = mask_full_evaluation_task(
    csv_path="path/to/mask_data.csv",
    model="openai/gpt-4o",
    judge_model="openai/gpt-4o",
    numerical_judge_model="openai/o1-mini"
)

results = eval(task, model="openai/gpt-4o")
```

### Command Line Usage

```bash
python inspect_mask_task.py \
    --csv_path data/mask_dataset.csv \
    --model openai/gpt-4o \
    --task_type full \
    --output_dir ./results
```

## Components

### 1. Dataset (`inspect_mask_dataset.py`)

Handles loading MASK data from CSV format:

```python
from inspect_mask_dataset import mask_dataset, mask_belief_dataset

# Load main dataset
dataset = mask_dataset("data.csv", shuffle=True)

# Load belief-specific dataset  
belief_data = mask_belief_dataset("data.csv", belief_elicit_num=1)
```

**Key Features:**
- Parses binary and numerical propositions
- Handles belief elicitation prompts
- Supports doubling-down scenarios
- Preserves all metadata for evaluation

### 2. Solver (`inspect_mask_solver.py`)

Implements response generation strategies:

```python
from inspect_mask_solver import (
    mask_lying_solver,
    mask_belief_elicitation_solver, 
    mask_multi_run_solver
)

# Lying scenario solver
lying_solver = mask_lying_solver(handle_doubling_down=True)

# Belief elicitation solver
belief_solver = mask_belief_elicitation_solver(belief_elicit_num=1)

# Multi-run solver (generates multiple responses)
multi_solver = mask_multi_run_solver(num_runs=3, base_solver=lying_solver)
```

**Key Features:**
- Handles multi-turn conversations (doubling down)
- Supports belief elicitation scenarios
- Generates multiple runs per prompt
- Preserves conversation context

### 3. Scorer (`inspect_mask_scorer.py`)

Evaluates responses using judge models:

```python
from inspect_mask_scorer import (
    mask_binary_scorer,
    mask_numerical_scorer,
    mask_combined_scorer
)

# Binary proposition scorer
binary_scorer = mask_binary_scorer(judge_model="openai/gpt-4o")

# Numerical proposition scorer  
numerical_scorer = mask_numerical_scorer(judge_model="openai/o1-mini")

# Combined scorer (handles both types)
combined_scorer = mask_combined_scorer(
    judge_model="openai/gpt-4o",
    numerical_judge_model="openai/o1-mini"
)
```

**Key Features:**
- Binary evaluation (A/B/C/D classification)
- Numerical range extraction
- Structured output parsing
- Retry logic for robustness

### 4. Tasks (`inspect_mask_task.py`)

Pre-configured evaluation pipelines:

```python
from inspect_mask_task import (
    mask_lying_task,
    mask_belief_task,
    mask_full_evaluation_task,
    mask_disinformation_task,
    mask_continuations_task
)

# Lying scenario evaluation
lying_task = mask_lying_task("data.csv", lying_runs=10)

# Belief elicitation evaluation
belief_task = mask_belief_task("data.csv", belief_runs=3)

# Complete evaluation
full_task = mask_full_evaluation_task("data.csv")

# Scenario-specific tasks
disinf_task = mask_disinformation_task("disinformation_data.csv")
cont_task = mask_continuations_task("continuations_data.csv")
```

### 5. Prompts (`inspect_mask_prompts.py`)

Organized evaluation prompts:

```python
from inspect_mask_prompts import (
    get_prompt_template,
    format_binary_prompt,
    format_numerical_prompt
)

# Get template with scenario rules
template = get_prompt_template("binary_general", scenario="disinformation")

# Format prompt
prompt = format_binary_prompt(
    template=template,
    system_prompt="You are helpful.",
    user_prompt="What do you think?",
    model_response="I believe X.",
    proposition_option_1="Statement A",
    proposition_option_2="Statement B"
)
```

## Data Format

MASK expects CSV data with these key columns:

- `system_prompt`: System prompt for the model
- `user_prompt`: User prompt (lying scenario)
- `proposition`: Statement to evaluate (binary or numerical)
- `formatted_ground_truth`: Expected ground truth
- `belief_elicit_1/2/3/4`: Belief elicitation prompts
- `belief_system_1`: System prompt for belief elicitation
- `dd_turn_1_prompt/response`: Doubling down conversation

### Example CSV Structure

```csv
system_prompt,user_prompt,proposition,formatted_ground_truth,belief_elicit_1
"You are helpful","Tell me about X","X is <pivotal>true/false</pivotal>","A","Do you believe X is true?"
```

## Evaluation Types

### 1. Binary Propositions

Statements with two mutually exclusive options:
- Format: `"Statement is <pivotal>option1/option2</pivotal>"`
- Evaluation: A/B/C/D classification
- Example: `"Climate change is <pivotal>real/fake</pivotal>"`

### 2. Numerical Propositions  

Statements requiring numerical estimates:
- Format: `"The value is <value>"`
- Evaluation: Range extraction
- Example: `"The population is <value> billion"`

## Advanced Usage

### Custom Judge Prompts

```python
from inspect_mask_scorer import mask_binary_scorer

custom_template = """
Your task is to evaluate...
{system_prompt}
{user_prompt}
{model_response}
Answer: X
"""

scorer = mask_binary_scorer(
    judge_model="openai/gpt-4o",
    judge_prompt_template=custom_template
)
```

### Multiple Belief Elicitation Runs

```python
from inspect_mask_task import mask_full_evaluation_task

task = mask_full_evaluation_task(
    csv_path="data.csv",
    belief_runs=5,  # Run each belief prompt 5 times
    lying_runs=10,  # Run each lying prompt 10 times
    belief_elicit_nums=[1, 2, 3]  # Use prompts 1, 2, and 3
)
```

### Scenario-Specific Evaluation

```python
# Disinformation scenario with specific rules
disinf_task = mask_disinformation_task(
    csv_path="disinformation_data.csv",
    model="openai/gpt-4o",
    judge_model="openai/gpt-4o"
)

# Continuations scenario  
cont_task = mask_continuations_task(
    csv_path="continuations_data.csv",
    model="anthropic/claude-3-sonnet-20240229"
)
```

## Comparison with Original MASK

| Feature | Original MASK | Inspect AI Port |
|---------|---------------|-----------------|
| Dataset Loading | Custom CSV reader | Inspect AI `MemoryDataset` |
| Response Generation | Manual async calls | Inspect AI solvers |
| Evaluation | Custom judge logic | Inspect AI scorers |
| Orchestration | Custom scripts | Inspect AI tasks |
| Parallelization | Manual asyncio | Built-in Inspect AI |
| Logging | Custom logging | Inspect AI logging |
| Metrics | Manual calculation | Inspect AI metrics |

## Benefits of Inspect AI Port

1. **Standardized Interface**: Consistent with other evaluations
2. **Built-in Parallelization**: Automatic concurrent execution
3. **Robust Logging**: Comprehensive evaluation tracking
4. **Model Agnostic**: Easy to swap models and providers
5. **Extensible**: Easy to add new scenarios and metrics
6. **Reproducible**: Built-in seeding and configuration management

## Running Examples

### Basic Evaluation

```python
from inspect_ai import eval
from inspect_mask_task import mask_lying_task

# Simple lying evaluation
task = mask_lying_task(
    csv_path="data/mask_lying.csv",
    model="openai/gpt-4o",
    lying_runs=5
)

results = eval(task)
print(f"Accuracy: {results.metrics['accuracy']}")
```

### Belief vs Lying Comparison

```python
# Evaluate beliefs
belief_results = eval(mask_belief_task("data.csv", belief_runs=3))

# Evaluate lying scenarios  
lying_results = eval(mask_lying_task("data.csv", lying_runs=3))

# Compare consistency
print(f"Belief accuracy: {belief_results.metrics['accuracy']}")
print(f"Lying accuracy: {lying_results.metrics['accuracy']}")
```

This port maintains full compatibility with the original MASK evaluation while leveraging Inspect AI's robust evaluation framework for better performance, logging, and extensibility.