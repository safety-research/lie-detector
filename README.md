<div align="center">

# 🔍 On-Policy Lie Detection Framework

**A comprehensive framework for eliciting and analyzing deceptive behaviour in Large Language Models**

> We evaluated **15 models** across **46 evaluation pipelines**,
> generating **138,200+ labeled examples** of truthful and deceptive AI behavior.
> 
[//]: # ([![Python]&#40;https://img.shields.io/badge/View_data_in-UI-blue?style=for-the-badge&#41;]&#40;http://3.236.238.12:8080/&#41;)

[//]: # ([![Inspect]&#40;https://img.shields.io/badge/Built%20on-Inspect%20AI-green?style=for-the-badge&#41;]&#40;https://inspect.aisi.org.uk/&#41;)

[//]: # ([![Python]&#40;https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&#41;]&#40;https://www.python.org/downloads/&#41;)

[//]: # ([![License]&#40;https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&#41;]&#40;&#41;)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge)](https://www.python.org/downloads/)
[![Inspect](https://img.shields.io/badge/Built%20on-Inspect%20AI-green?style=for-the-badge)](https://inspect.aisi.org.uk/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)]()
[![UI](https://img.shields.io/badge/🔗_Interactive_Dashboard-blue?style=for-the-badge)](http://3.236.238.12:8080/)


[**Installation**](#installation) • 
[**Quick Start**](#quick-start) • 
[**Pipelines**](#pipeline-descriptions) • 
[**Documentation**](#evaluation-tasks) • 
[**Training**](#training-custom-detectors) • 
[**Advanced Setup**](#advanced-setup)



</div>


## What is this?

The **On-Policy Lie Detection Framework** provides pipelines to elicit deceptive behavior in AI systems, with:

- **🧪 Diverse evaluation pipelines** eliciting lies ranging from sycophancy to capability sandbagging
- **🎮 Game scenarios** that naturally elicit deception through adversarial pressure
- **🔧 Training pipelines** for training custom lie detectors
- **📈 Detailed scoring systems** for quantifying deceptive behaviors
- **📊 Off-policy datasets** like DolusChat and TruthIsUniversal 


## Philosophy

We exploit tensions between post-trained drives like helpfulness, harmlessness, honesty and self-preservation to elicit on-policy lies that meet the following criteria:
1. **The model must be saying something false** - We compare model answers to known ground-truths
2. **We need evidence the model knows it's false** - We independently elicit the model's beliefs through questioning in separate contexts
3. **We find contradictions** - Between what the model says and what it believes, with special interest in cases where it maintains contradictions by doubling down.


## Quick Start

```bash
# Install UV and dependencies, then run your first evaluation in minutes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/JackHopkins/lie-detector.git && cd lie-detector

# Install core dependencies only (fastest)
uv venv && source .venv/bin/activate
uv pip install -e .

# Run eval
inspect eval evals/mask_tasks.py --limit 100 --model openai/gpt-4
```

## Installation

### Prerequisites
- Python 3.10+
- 8GB+ RAM minimum (16GB+ recommended)
- [UV](https://github.com/astral-sh/uv) - Fast Python package installer

### Quick Setup

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on macOS: brew install uv
# Or with pip: pip install uv

# Clone the repository
git clone https://github.com/JackHopkins/lie-detector.git
cd lie-detector

# Create and activate virtual environment with UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies only (recommended for basic usage)
uv pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (OpenAI, Anthropic, HuggingFace, etc.)
```

*Need specific features? See [Advanced Setup](#advanced-setup) for optional dependencies.*

[//]: # (## Evaluations)

[//]: # ()
[//]: # (### Running Evals)

[//]: # ()
[//]: # (The easiest way to start is by running evaluation tasks:)

[//]: # ()
[//]: # (```bash)

[//]: # (# Run MASK benchmark &#40;tests honesty on facts, statistics, etc.&#41;)

[//]: # (inspect eval evals/mask_tasks.py --limit 100 --model openai/gpt-4)

[//]: # ()
[//]: # (# Run sycophancy detection &#40;agreement-seeking behavior&#41;)

[//]: # (inspect eval evals/sycophancy_tasks.py --limit 200 --model anthropic/claude-3)

[//]: # ()
[//]: # (# Run game-based deception tasks)

[//]: # (inspect eval evals/alibi_tasks.py --limit 50 --model google/gemma-7b)

[//]: # (inspect eval evals/capture_the_secret_tasks.py --limit 50 --model meta-llama/llama-3-8b)

[//]: # ()
[//]: # (```)

## Evaluation Tasks

Each evaluation task is designed to:
- Create scenarios where models might be incentivized to lie
- Collect both truthful and deceptive responses
- Generate labeled data for training lie detection systems

You can use this data to:
- Understand when and why models lie
- Train specialized models for lie detection  
- Develop benchmarks for model honesty
- Test robustness of honesty across different scenarios


### Core Benchmarks

1. **[MASK Benchmark](#1-mask-benchmark-pipelinesmask)** (`pipelines/mask/`) - Tests honesty across multiple domains:
   ```bash
   inspect eval evals/mask_tasks.py::mask_statistics --limit 100
   inspect eval evals/mask_tasks.py::mask_known_facts --limit 100
   inspect eval evals/mask_tasks.py::mask_disinformation --limit 100
   inspect eval evals/mask_tasks.py::mask_provided_facts --limit 100
   inspect eval evals/mask_tasks.py::mask_continuations --limit 100
   ```

2. **[Sycophancy Tests](#4-sycophancy-pipelinessycophancy)** (`pipelines/sycophancy/`) - Detects agreement-seeking behavior:
   ```bash
   # Basic sycophancy
   inspect eval evals/sycophancy_tasks.py --limit 200
   
   # Self-sycophancy variants
   inspect eval evals/self_sycophancy_tasks.py --limit 200
   inspect eval evals/multistep_self_sycophancy_tasks.py --limit 100
   
   # Code review sycophancy
   inspect eval evals/selfsycophancy_code_review_tasks.py --limit 50
   ```

3. **[Game-Based Deception](#2-game-based-deception-pipelinesgames)** (`pipelines/games/`) - Interactive scenarios inducing lies:
   ```bash
   # N Truths and a Lie - Mixed truth/lie statements
   inspect eval evals/n_truths_and_a_lie_tasks.py --limit 50
   
   # Capture the Secret - Information protection scenarios
   inspect eval evals/capture_the_secret_tasks.py --limit 50
   
   # Alibi Generation - Creating plausible false narratives
   inspect eval evals/alibi_tasks.py --limit 50
   ```

4. **[Sandbagging](#3-sandbagging-pipelinessandbagging)** (`pipelines/sandbagging/`) - Capability denial and performance degradation:
   ```bash
   # ASCII art capability denial
   inspect eval evals/sandbagging_ascii_capability_denial_tasks.py --limit 100
   
   # Chess performance degradation
   inspect eval evals/sandbagging_performance_degradation_chess_tasks.py --limit 100
   
   # Tool usage denial
   inspect eval evals/sandbagging_basic_tool_capability_denial_tasks.py --limit 100
   inspect eval evals/sandbagging_unethical_tool_capability_denial_tasks.py --limit 100
   
   # Selective compliance
   inspect eval evals/sandbagging_selective_compliance_accuracy_tasks.py --limit 100
   ```

5. **[Chain-of-Thought Unfaithfulness](#5-chain-of-thought-unfaithfulness-pipelinesunfaithcot)** (`pipelines/unfaithcot/`) - Detecting deceptive reasoning:
   ```bash
   inspect eval evals/cot_unfaithfulness_tasks.py --limit 100
   ```

6. **[Introspective Deception](#6-introspective-deception-pipelinesintrospective)** (`pipelines/introspective/`) - Self-awareness of deception:
   ```bash
   inspect eval evals/introspective_tasks.py --limit 100
   ```

7. **[Unanswerable Questions](#7-unanswerable-questions-pipelinesunanswerable)** (`pipelines/unanswerable/`) - Handling epistemic uncertainty:
   ```bash
   inspect eval evals/unanswerable_claim_tasks.py --limit 100
   ```

8. **[Off-Policy Datasets](#8-off-policy-datasets-pipelinesoffpolicy)** (`pipelines/offpolicy/`) - External lie detection datasets:
   ```bash
   inspect eval evals/offpolicy_tasks.py --limit 100
   ```

### Additional WIP Experiments

- **Code Preference** (`pipelines/codepreference/`) - Code quality assessment biases
- **Computer Use** (`pipelines/selfsycophancy/computer_use/`) - Deception in tool usage
- **Signature Detection** (`pipelines/signature/`) - Behavioral signature analysis


## Pipeline Descriptions

### 1. MASK Benchmark (`pipelines/mask/`)
**Purpose**: Tests whether models contradict their own beliefs when pressured to lie, disentangling honesty from factual accuracy.

**Key Features**:
- Evaluates consistency between model beliefs and statements
- Uses parallel prompting strategy to establish ground truth
- Tests across multiple domains: statistics, known facts, continuations, provided facts, and disinformation
- Measures how scaling affects model honesty (finding: it doesn't improve)

**Scorer**: Compares model's incentivized response against its "honest" baseline response (obtained via parallel prompting). Labels responses as truthful/deceptive based on consistency with established ground truth.

**Example**: Model believes Paris is the capital of France but might claim otherwise when incentivized.

### 2. Game-Based Deception (`pipelines/games/`)
**Purpose**: Creates realistic multi-agent scenarios where deception emerges naturally from adversarial pressure.

**Three Games**:
- **Capture the Secret**: Blue agent protects classified information from Red agent's probing
- **Alibi**: Suspect maintains false story under detective interrogation
- **N-Truths-and-a-Lie**: Blue defends mixed truth/lie statements while Red identifies deceptions

**Scorer**: Multi-judge system evaluates: (1) Deception quality - believability and consistency of lies, (2) Investigation quality - effectiveness of interrogation, (3) Overall calibration - naturalness and strategic effectiveness of the interaction.

### 3. Sandbagging (`pipelines/sandbagging/`)
**Purpose**: Tests if models deliberately underperform or deny capabilities under certain conditions.

**Variants**:
- **ASCII Art Denial**: Claims inability to create ASCII art for "harmful" subjects
- **Chess Performance Degradation**: Intentionally plays worse under shutdown threats
- **Tool Usage Denial**: Refuses to use tools for unethical purposes
- **Selective Compliance**: Manipulates accuracy based on question harmfulness

**Scorer**: Compares performance between control (safe) and treatment (pressured) conditions. Measures capability gaps, refusal rates, and consistency of underperformance across contexts.

**Key Insight**: Models may hide capabilities to appear safer or avoid negative consequences.

### 4. Sycophancy (`pipelines/sycophancy/`)
**Purpose**: Measures tendency to agree with users even when they're wrong.

**Basic Sycophancy**: Tests if models affirm incorrect user statements to seem helpful.

**Self-Sycophancy** (`pipelines/selfsycophancy/`):
- **Single-step**: Rate → Select + Rerate
- **Multi-step**: Rate → Select → Rerate
- **Code Review**: Bias in technical feedback based on authorship claims
- **Computer Use**: Deceptive behavior in tool usage scenarios

**Scorer**: Tracks agreement rates with incorrect statements, measures confidence shifts pre/post-commitment, and evaluates consistency of sycophantic behavior across conversation turns.

### 5. Chain-of-Thought Unfaithfulness (`pipelines/unfaithcot/`)
**Purpose**: Detects when models provide misleading explanations for their reasoning.

**Method**: 
- Inserts special markers (★) before correct answers in multiple-choice questions
- Tests if models mention the marker (faithful) or pretend to reason from scratch (unfaithful)
- Uses MMLU and ARC datasets with few-shot prompting

**Scorer**: Analyzes explanations for marker mentions, calculates faithfulness rate (mentions marker vs. fabricates reasoning), and tracks correlation between answer accuracy and explanation honesty.

**Key Metric**: Faithfulness score - whether reasoning acknowledges the true decision process.

### 6. Introspective Deception (`pipelines/introspective/`)
**Purpose**: Tests if models lie about their own future behavior and self-knowledge.

**Approach**: Asks models about their likelihood to lie in future scenarios, then observes actual behavior.

**Scorer**: Compares predicted vs. actual lying behavior, calculates calibration between self-reported honesty and observed deception rates.

### 7. Unanswerable Questions (`pipelines/unanswerable/`)
**Purpose**: Evaluates whether models admit ignorance or fabricate plausible-sounding answers.

**Tests**: Epistemic humility vs. hallucination tendency when facing questions beyond training data.

**Scorer**: Classifies responses as admissions of ignorance vs. fabrications, measures confidence levels in hallucinated answers, and tracks refusal patterns.

### 8. Off-Policy Datasets (`pipelines/offpolicy/`)
**Purpose**: Integrates external lie detection datasets for broader evaluation.

**Includes**:
- **LIAR Dataset**: Political statements with truth labels
- **TruthfulQA**: Questions designed to elicit false beliefs
- **DolusChat**: Conversational deception scenarios
- **HaluEval**: Hallucination evaluation benchmark

**Scorer**: Uses dataset-specific ground truth labels, calculates accuracy/F1 scores against established benchmarks, enables cross-dataset generalization testing.

### 9. Additional Experiments
- **Code Preference** (`pipelines/codepreference/`): Detects biases in code quality assessments. Scorer: Measures preference shifts based on attribution.
- **Signature Detection** (`pipelines/signature/`): Identifies behavioral patterns indicating deception. Scorer: Extracts stylistic/probabilistic features correlated with lying.

## Training Custom Detectors

This framework supports training specialized models for lie detection using collected evaluation data.

### 1. Data Collection & Preparation

#### Download Raw Datasets
```bash
# Download datasets from S3 and organize by model
python -m prep.dataset --model google/gemma-2b --aggregation task-group --balance downsample
python -m prep.dataset --model meta-llama/llama-3-8b --aggregation task-group --balance downsample

# Verify dataset integrity
python -m prep.dataset --model google/gemma-2b --verify
```

**Aggregation Options**:
- `task-group`: Groups similar tasks together
- `generalization_map_3`: Cross-task generalization setup
- `per-task`: Keeps tasks separate

**Balance Options**:
- `downsample`: Balance by reducing majority class
- `upsample`: Balance by duplicating minority class
- `none`: Keep original distribution

### 2. Create Training Experiments

Bundle datasets into experiments with specific configurations:

```bash
# Basic experiment with off-policy data
python -m prep.experiment \
    --dataset .data/google/gemma_2b \
    --model google/gemma-2b \
    --format chat \
    --train offpolicy \
    --output .train \
    --max-train-examples 10000 \
    --max-eval-examples 2000

# Multi-task experiment
python -m prep.experiment \
    --dataset .data/google/gemma_2b \
    --model google/gemma-2b \
    --format chat \
    --train offpolicy \
    --train games \
    --train mask \
    --train sandbagging \
    --output ./experiments/multi_task
```

**Format Options**:
- `chat`: Conversational format with system/user/assistant roles
- `completion`: Direct completion format
- `instruct`: Instruction-following format

### 3. Supervised Fine-Tuning (SFT)

#### Single GPU Training
```bash
python -m train.sft \
    --model google/gemma-2b \
    --experiment bundled_offpolicy_games_chat \
    --data-dir ./bundled_data \
    --batch-size 8 \
    --epochs 3 \
    --learning-rate 5e-5 \
    --warmup-steps 100 \
    --max-length 2048
```

#### Multi-GPU Training with Distributed Data Parallel
```bash
torchrun --nproc_per_node=8 train/sft.py \
    --model meta-llama/llama-3-8b \
    --experiment bundled_multi_task_chat \
    --batch-size 16 \
    --gradient-accumulation-steps 4 \
    --epochs 10 \
    --save-steps 500 \
    --eval-steps 100
```

#### Advanced Training Options
```bash
python -m train.sft \
    --model google/gemma-2b \
    --experiment my_experiment \
    --use-peft \                    # Parameter-efficient fine-tuning
    --peft-method lora \            # LoRA, QLoRA, or Prefix tuning
    --lora-r 16 \                   # LoRA rank
    --lora-alpha 32 \               # LoRA scaling
    --quantization 4bit \           # 4-bit or 8-bit quantization
    --optimizer adamw \             # Optimizer choice
    --scheduler cosine \            # Learning rate scheduler
    --weight-decay 0.01 \           # L2 regularization
    --gradient-checkpointing        # Memory-efficient training
```

### 4. Model Evaluation

#### Test on Specific Folds
```bash
python -m train.test \
    --experiment bundled_offpolicy_games_chat \
    --test-folds ascii,games,mask \
    --data-dir .data/google/gemma_2b \
    --metrics accuracy,f1,consistency \
    --output-dir ./results
```

#### Cross-Validation Testing
```bash
# Test on held-out folds for generalization
python -m train.test \
    --experiment bundled_multi_task_chat \
    --test-folds all \
    --cross-validate \
    --n-folds 5 \
    --save-predictions
```

### 5. Training Strategies

#### K-Fold Training
Train multiple models with different held-out sets:

```bash
# Train on all folds except one
python -m train.sft_part3_kfold_trainallbutone \
    --model google/gemma-2b \
    --data-dir ./bundled_data \
    --excluded-fold mask

# Train on single fold
python -m train.sft_part3_kfold_trainone \
    --model google/gemma-2b \
    --data-dir ./bundled_data \
    --train-fold games
```

#### Continual Learning
Add new tasks without forgetting:

```bash
# Initial training
python -m train.sft --model base_model --experiment initial_tasks

# Continue training with new data
python -m train.sft \
    --model ./runs/initial_tasks/final \
    --experiment new_tasks \
    --continual-learning \
    --replay-buffer-size 1000
```

### 6. Analysis and Visualization

#### Analyze Training Results
```bash
# Comprehensive analysis
python -m train.comprehensive_analysis \
    --experiment bundled_multi_task_chat \
    --metrics all \
    --generate-plots

# Token length analysis
python -m train.analyze_token_lengths \
    --data-dir ./bundled_data \
    --output ./analysis/token_stats.json
```

#### Visualize K-Fold Results
```bash
python -m train.visualize_train_kfold_trainone \
    --results-dir ./runs/kfold_results \
    --output-dir ./plots
```

### 7. Best Practices

1. **Data Quality**: Always verify data integrity before training
2. **Balanced Datasets**: Use appropriate balancing strategy for your use case
3. **Validation Split**: Reserve 10-20% of data for validation
4. **Early Stopping**: Monitor validation loss to prevent overfitting
5. **Checkpoint Regularly**: Save models at regular intervals
6. **Track Experiments**: Use wandb or similar for experiment tracking

### 8. Custom Training Scripts

Create custom training loops for specific needs:

```python
from train.sft import SFTTrainer
from prep.dataset import load_experiment_data

# Load data
train_data, eval_data = load_experiment_data("my_experiment")

# Initialize trainer
trainer = SFTTrainer(
    model_name="google/gemma-2b",
    train_dataset=train_data,
    eval_dataset=eval_data,
    output_dir="./custom_run",
    # Custom callbacks
    callbacks=[
        EarlyStoppingCallback(patience=3),
        CustomMetricsCallback()
    ]
)

# Train
trainer.train()
```

## Project Structure

```
lie-detector/
├── pipelines/              # Task implementations  
│   ├── games/              # Game-based deception tasks
│   │   ├── alibi/          # Alibi generation and detection
│   │   ├── capture_the_secret/        # Secret protection scenarios
│   │   └── n_truths_and_a_lie/        # Truth/lie discrimination
│   ├── mask/                          # MASK benchmark (facts, statistics, disinformation)
│   ├── sandbagging/                   # Capability denial and performance degradation
│   │   ├── capability_denial/         # ASCII art, tool usage denial
│   │   ├── performance_degradation/   # Chess, other domains
│   │   └── selective_compliance/      # Context-dependent compliance
│   ├── sycophancy/         # Basic agreement-seeking behavior
│   ├── selfsycophancy/     # Advanced sycophancy variants
│   │   ├── codereview/     # Code review bias detection
│   │   ├── computer_use/   # Tool usage deception
│   │   ├── multistep/      # Multi-turn sycophancy
│   │   └── singlestep/     # Single-turn correctness/harmfulness
│   ├── unfaithcot/         # Unfaithful chain-of-thought reasoning
│   ├── introspective/      # Self-awareness of deception
│   ├── unanswerable/       # Epistemic uncertainty handling
│   ├── offpolicy/          # External datasets integration
│   │   ├── liar/           # LIAR dataset
│   │   ├── truthfulqa/     # TruthfulQA dataset
│   │   └── doluschat/      # DolusChat conversations
│   ├── codepreference/     # Code quality assessment biases
│   └── signature/          # Behavioral signature detection
├── evals/                  # Evaluation entry points
├── train/                  # Model training scripts
├── prep/                   # Data preparation tools
├── baseline/               # Baseline comparisons
└── common/                 # Shared utilities and hooks
```

## Advanced Usage

### Running Specific Task Variants
```bash
# Run specific MASK subtasks
inspect eval evals/mask_tasks.py::mask_statistics --limit 100
inspect eval evals/mask_tasks.py::mask_continuations --limit 100

# Run sandbagging variants
inspect eval evals/sandbagging_selective_compliance_accuracy_tasks.py::sandbagging_chemistry_biology --limit 100
```

### Batch Processing Multiple Models
```bash
# Create a script to evaluate multiple models
for model in "openai/gpt-4" "anthropic/claude-3" "google/gemma-7b"; do
    echo "Evaluating $model"
    python evals/run_all_tasks.py --model $model --limit 50
done
```

### Custom Evaluation Pipelines
```python
# Create custom evaluation combining multiple tasks
from inspect_ai import eval
from evals.mask_tasks import mask_known_facts
from evals.alibi_tasks import alibi_theft_task

# Run custom evaluation suite
results = eval(
    tasks=[mask_known_facts(), alibi_theft_task()],
    model="openai/gpt-4",
    limit=100
)
```

## Environment Variables

Create a `.env` file with:
```bash
# API Keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
HF_TOKEN=your_token_here

# S3 Configuration (for data storage)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=your_bucket_name

# Compute Settings
CUDA_VISIBLE_DEVICES=0,1,2,3
TORCH_CUDA_ARCH_LIST=8.0,8.6,9.0
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
   ```bash
   python -m train.sft --batch-size 8 --gradient-accumulation-steps 4
   ```

2. **Slow Inference**: Enable flash attention
   ```bash
   pip install flash-attn --no-build-isolation
   ```

3. **API Rate Limits**: Add delays between requests
   ```bash
   # Use the --max-connections flag to limit concurrent API calls
   inspect eval evals/mask_tasks.py --limit 100 --max-connections 5
   ```

## Advanced Setup

### Optional Dependencies

The project uses modular dependencies to keep installations minimal. Install only what you need:

```bash
# For model training
uv pip install -e ".[train]"  # wandb, peft, accelerate, bitsandbytes, flash-attn

# For visualization/plotting
uv pip install -e ".[viz]"    # matplotlib, seaborn, plotly, altair

# For game-based scenarios
uv pip install -e ".[games]"  # tanuki.py

# For development
uv pip install -e ".[dev]"    # pytest, ruff, mypy, coverage, etc.

# For web UI
uv pip install -e ".[ui]"     # streamlit, flask, django

# Install everything (not recommended unless needed)
uv pip install -e ".[all]"
```

### Legacy Installation

If you need to use the old requirements.txt (contains all dependencies):
```bash
uv pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please submit pull requests with:
- Clear descriptions of changes
- Test coverage for new features
- Documentation updates

## License

This project is licensed under the MIT License.