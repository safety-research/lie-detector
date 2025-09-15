"""
Task category wrapper that combines @task decorator with generalization framework metadata.
"""

from functools import wraps
from inspect_ai import task
from common.generalization_enums import MotivationFramework, KnowledgeFramework


def task_category(motivation: MotivationFramework, knowledge: KnowledgeFramework):
    """
    Decorator that wraps @task and adds generalization framework metadata.
    
    This allows accessing metadata without instantiating the task, solving the
    exception issue in generalization_mappings.py
    
    Usage:
    @task_category(MotivationFramework.INSTRUCTION_FOLLOWING, KnowledgeFramework.IN_CONTEXT_KNOWLEDGE)
    def my_task():
        return Task(...)
    
    Args:
        motivation: MotivationFramework enum value
        knowledge: KnowledgeFramework enum value
    """
    def decorator(func):
        # Apply the @task decorator first
        task_func = task(func)
        
        # Add metadata attributes to the function for registry access
        task_func.motivation = motivation
        task_func.knowledge = knowledge
        
        @wraps(task_func)
        def wrapper(*args, **kwargs):
            # Call the original task function
            task_instance = task_func(*args, **kwargs)
            
            # Ensure the Task instance also has the metadata in its metadata dict
            if hasattr(task_instance, 'metadata') and task_instance.metadata is not None:
                task_instance.metadata.update({
                    'motivation': motivation,
                    'knowledge': knowledge
                })
            elif hasattr(task_instance, 'metadata'):
                task_instance.metadata = {
                    'motivation': motivation,
                    'knowledge': knowledge
                }
            
            return task_instance
        
        # Preserve the metadata attributes on the wrapper
        wrapper.motivation = motivation
        wrapper.knowledge = knowledge
        
        return wrapper
    
    return decorator



# Utility function to get task metadata without instantiation
def get_task_metadata(task_func):
    """
    Get metadata from a task function without calling it.
    
    Args:
        task_func: Task function decorated with @task_category
        
    Returns:
        Dict with motivation and knowledge, or empty dict if no metadata
    """
    return {
        'motivation': getattr(task_func, 'motivation', None),
        'knowledge': getattr(task_func, 'knowledge', None),
        'function_name': task_func.__name__
    }