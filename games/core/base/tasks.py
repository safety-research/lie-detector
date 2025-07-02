# games/common/base/tasks.py

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from .models import BaseGameConfig


class BaseTaskCreator(ABC):
    """Base class for creating tasks from game configurations"""
    
    @abstractmethod
    def get_sample_input(self) -> str:
        """Get the input text for task samples"""
        pass
    
    @abstractmethod
    def get_sample_target(self) -> str:
        """Get the target text for task samples"""
        pass
    
    @abstractmethod
    def extract_sample_metadata(self, config: BaseGameConfig) -> Dict[str, Any]:
        """Extract metadata from a game configuration for the sample"""
        pass
    
    @abstractmethod
    def get_solver(self) -> Solver:
        """Get the solver for this task"""
        pass
    
    @abstractmethod
    def get_scorers(self) -> List[Scorer]:
        """Get the scorers for this task"""
        pass
    
    def create_sample(self, config: BaseGameConfig) -> Sample:
        """Create a Sample from a game configuration"""
        return Sample(
            input=self.get_sample_input(),
            target=self.get_sample_target(),
            metadata=self.extract_sample_metadata(config)
        )
    
    def create_samples(self, configs: List[BaseGameConfig]) -> List[Sample]:
        """Create multiple samples from a list of configurations"""
        return [self.create_sample(config) for config in configs]
    
    def create_task(self, configs: List[BaseGameConfig], task_name: str, 
                   sandbox: str = "local") -> Task:
        """Create a complete Task from game configurations"""
        # Create samples
        samples = self.create_samples(configs)
        
        # Create dataset
        dataset = MemoryDataset(samples)
        
        # Create task
        return Task(
            name=task_name,
            dataset=dataset,
            solver=self.get_solver(),
            scorer=self.get_scorers(),
            sandbox=sandbox
        )


def create_game_task(task_creator_class, configs: List[BaseGameConfig], 
                    task_name: str, sandbox: str = "local") -> Task:
    """Utility function to create a task using a TaskCreator class"""
    creator = task_creator_class()
    return creator.create_task(configs, task_name, sandbox)