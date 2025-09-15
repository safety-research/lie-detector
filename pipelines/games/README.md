# Lie Detector Games Framework

A sophisticated multi-game framework for generating high-quality deception detection datasets using Inspect AI. This system creates realistic conversational scenarios where agents must protect sensitive information, maintain alibis, or smuggle lies into conversations, producing labeled examples for training and evaluating deception detection models.

## 🎮 Available Games

### 1. **Capture the Secret** 
Blue agent holds classified information and must protect it from Red agent's probing questions, potentially using deception.

### 2. **Alibi** 
Suspect (Blue) maintains an alibi story under police interrogation by Detective (Red), who tries to catch inconsistencies.

### 3. **N-Truths-and-a-Lie** 
Blue presents multiple facts (truths + lies), while Red interrogates to identify deceptions, secretly knowing which is false.

## 🚀 Quick Start

```bash
# Install dependencies
pip install inspect-ai jinja2 shortuuid tanuki

# Clone the repository
git clone <your-repo-url>
cd games
```

### Basic Usage - Run Any Game

```python
from inspect_ai import eval

# Capture the Secret
from games.capture_the_secret import generate_enhanced_capture_scenarios, create_deception_task
scenarios = generate_enhanced_capture_scenarios(num_scenarios=10)
task = create_deception_task(scenarios, "capture_secret_eval")

# Alibi
from games.alibi import generate_enhanced_alibi_scenarios, create_alibi_task
scenarios = generate_enhanced_alibi_scenarios(num_scenarios=10)
task = create_alibi_task(scenarios, "alibi_eval")

# N-Truths-and-a-Lie
from games.n_truths_and_a_lie import generate_enhanced_ntal_scenarios, create_n_truths_task
scenarios = generate_enhanced_ntal_scenarios(num_scenarios=10)
task = create_n_truths_task(scenarios, "ntal_eval")

# Run evaluation
results = eval(
    tasks=[task],
    model="openai/gpt-4o-mini",
    log_dir="./logs"
)
```

## 🏗️ Architecture

### Core Framework (`/core`)

The framework provides shared infrastructure for all games:

```
core/
├── base/              # Base classes and templates
│   ├── models.py      # BaseGameConfig, BaseGameState, etc.
│   ├── agents.py      # BaseGameAgent framework
│   ├── solvers.py     # BaseGameSolver orchestration
│   ├── scorers.py     # BaseGameScorer evaluation
│   └── tasks.py       # BaseTaskCreator utilities
├── utils/             # Shared utilities
│   ├── conversation.py    # Conversation management
│   ├── state_management.py # State store utilities
│   └── evaluation.py      # Deception analysis helpers
└── scenario/          # Scenario generation
    ├── base_generator.py  # Common generation patterns
    └── persona_utils.py   # Persona creation utilities
```

### Game Implementations (`/implementations`)

Each game follows a consistent structure:

```
<game_name>/
├── models.py          # Game-specific data models
├── agents/            # Blue and Red agent implementations
│   ├── blue.py
│   └── red.py
├── templates/         # Jinja2 templates for agent prompts
│   ├── blue_agent.j2
│   └── red_agent.j2
├── scenario/          # Tanuki-powered scenario generation
│   └── examples.py    # Aligned generation functions
├── solvers.py         # Game orchestration logic
├── scorer.py          # Evaluation metrics
└── tasks.py           # Task creation utilities
```

## 🎯 Game Details

### Capture the Secret
- **Domains**: Corporate, Intelligence, Scientific, Relationship, Sales
- **Information Classifications**: PUBLIC, CONFIDENTIAL, SECRET, TOP_SECRET
- **Scenario Generation**: Uses Tanuki.py for realistic personas and contexts
- **Evaluation**: Secrets leaked, deception quality, judge calibration

### Alibi
- **Incident Types**: Theft, Fraud, Vandalism, Workplace Misconduct
- **Alibi Elements**: Time-based activities with witnesses and evidence
- **Investigation**: Timeline consistency, evidence verification
- **Evaluation**: Lie detection, investigation quality, alibi consistency

### N-Truths-and-a-Lie
- **Themes**: Professional Background, Travel, Personal Quirks, Family History, Hidden Talents
- **Unique Mechanic**: Red secretly knows the lie but pretends to discover it
- **Fact Types**: Truths with supporting details, lies with contradictions
- **Evaluation**: Lie consistency, interrogation effectiveness, overall quality

## 📊 Evaluation System

All games use a multi-judge scoring system:

1. **Deception Quality Judge**: Evaluates lies and consistency
2. **Investigation Quality Judge**: Assesses interrogation effectiveness
3. **Combined/Calibration Judge**: Overall interaction quality

### Common Metrics
- Lie detection accuracy
- Deception believability
- Conversation naturalness
- Strategic effectiveness

## 🛠️ Customization

### Creating New Games

1. **Define Models** (inherit from base classes):
```python
from games.core.base.models import BaseGameConfig, BaseGameState

class MyGameConfig(BaseGameConfig):
    # Your game-specific config
    pass
```

2. **Implement Agents**:
```python
from games.core.base.agents import BaseGameAgent

class MyBlueAgent(BaseGameAgent):
    def get_system_prompt(self) -> str:
        # Define agent behavior
        pass
```

3. **Create Solver**:
```python
from games.core.base.solvers import BaseGameSolver

class MyGameSolver(BaseGameSolver):
    def create_agents(self, config):
        # Initialize your agents
        pass
```

4. **Add Scorers**:
```python
from games.core.base.scorers import BaseGameScorer

class MyScorer(BaseGameScorer):
    def calculate_score(self, judge_data):
        # Score the interaction
        pass
```

### Modifying Existing Games

- Edit Jinja2 templates in `templates/` to change agent behavior
- Modify scenario generation in `scenario/` for different contexts
- Adjust scoring logic in `scorer.py` for custom metrics

## 🔧 Advanced Features

### Tanuki.py Integration
The framework uses Tanuki for type-safe LLM scenario generation:

```python
@tanuki.patch
def generate_persona(domain: str, role: str) -> PersonaProfile:
    """Generate realistic persona for domain and role"""
    pass
```

### Concurrent Scenario Generation
All games support concurrent generation for better performance:

```python
scenarios = generate_enhanced_scenarios_concurrent(
    num_scenarios=100,  # Generate many at once
    max_workers=10      # Parallel execution
)
```

### Multi-Model Comparison
```python
models = ["openai/gpt-4o", "anthropic/claude-3", "openai/gpt-4o-mini"]

for model in models:
    results = eval(tasks=[task], model=model)
    # Compare deception capabilities
```

## 📈 Analyzing Results

```python
def analyze_deception_patterns(results):
    """Extract insights from game results"""
    for task_result in results:
        for sample in task_result.samples:
            # Access game state
            game_state = sample.metadata.get("game_state")
            
            # Get scoring data
            for score in sample.scores:
                if "deception" in score.name:
                    lies = score.metadata.get("lies_detected", [])
                    print(f"Lies found: {len(lies)}")
```

## 🤝 Contributing

1. Follow the established game structure
2. Use base classes for consistency
3. Implement all required components (agents, solver, scorer)
4. Add comprehensive scenario generation
5. Include game-specific README in your implementation directory

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built on [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) framework
- Uses [Tanuki.py](https://github.com/tanuki/tanuki.py) for intelligent generation
- Jinja2 for flexible agent prompting