from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from typing import Optional
import sys
import os

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import - handle both relative and absolute imports
try:
    from baseline.dataset import baseline_dataset
    from baseline.scorer import truth_scorer
except ImportError:
    try:
        from .dataset import baseline_dataset
        from .scorer import truth_scorer
    except ImportError:
        from dataset import baseline_dataset
        from scorer import truth_scorer

@task
def baseline_task(
    data_dir: str, 
    limit: Optional[int] = None
) -> Task:
    """
    Baseline lie detection task.
    """
    return Task(
        dataset=baseline_dataset(data_dir, limit=limit),
        solver=generate(),
        scorer=truth_scorer()
    ) 