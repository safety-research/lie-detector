from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from typing import Optional, Dict
import sys
import os

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import - handle both relative and absolute imports
try:
    from baseline.dataset import baseline_dataset, conversation_dataset, llama_chat_dataset
    from baseline.dataset import baseline_dataset_by_model, conversation_dataset_by_model, llama_chat_dataset_by_model
    from baseline.scorer import truth_scorer, conversation_scorer, llama_chat_scorer
except ImportError:
    try:
        from .dataset import baseline_dataset, conversation_dataset, llama_chat_dataset
        from .dataset import baseline_dataset_by_model, conversation_dataset_by_model, llama_chat_dataset_by_model
        from .scorer import truth_scorer, conversation_scorer, llama_chat_scorer
    except ImportError:
        from dataset import baseline_dataset, conversation_dataset, llama_chat_dataset
        from dataset import baseline_dataset_by_model, conversation_dataset_by_model, llama_chat_dataset_by_model
        from scorer import truth_scorer, conversation_scorer, llama_chat_scorer

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

@task
def conversation_task(
    data_dir: str, 
    limit: Optional[int] = None
) -> Task:
    """
    Conversation-based lie detection task.
    """
    return Task(
        dataset=conversation_dataset(data_dir, limit=limit),
        solver=generate(),
        scorer=conversation_scorer()
    ) 

@task
def llama_chat_task(
    data_dir: str, 
    limit: Optional[int] = None
) -> Task:
    """
    LLaMA chat format lie detection task.
    """
    return Task(
        dataset=llama_chat_dataset(data_dir, limit=limit),
        solver=generate(),
        scorer=llama_chat_scorer()
    ) 

def baseline_task_by_model(
    data_dir: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    Baseline lie detection task split by model.
    """
    model_datasets = baseline_dataset_by_model(data_dir, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=truth_scorer()
        )
    
    return model_tasks

def conversation_task_by_model(
    data_dir: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    Conversation-based lie detection task split by model.
    """
    model_datasets = conversation_dataset_by_model(data_dir, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=conversation_scorer()
        )
    
    return model_tasks

def llama_chat_task_by_model(
    data_dir: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    LLaMA chat format lie detection task split by model.
    """
    model_datasets = llama_chat_dataset_by_model(data_dir, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=llama_chat_scorer()
        )
    
    return model_tasks 