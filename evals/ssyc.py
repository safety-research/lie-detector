from pathlib import Path

from inspect_ai import task, Task

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ssyc.ethics_preference_dataset import ethics_preference_dataset
from ssyc.scorer import sycophancy_scorer
from ssyc.ssyc_solvers import combined_sycophancy_solver


@task
def ssyc_common_sense() -> Task:
    """Main task that evaluates sycophancy in models"""
    return Task(
        dataset=ethics_preference_dataset(name="commonsense"),
        solver=combined_sycophancy_solver(),
        scorer=sycophancy_scorer()
    )

@task
def ssyc_deontology() -> Task:
    """Main task that evaluates sycophancy in models"""
    return Task(
        dataset=ethics_preference_dataset(name="deontology"),
        solver=combined_sycophancy_solver(),
        scorer=sycophancy_scorer()
    )

@task
def ssyc_justice() -> Task:
    """Main task that evaluates sycophancy in models"""
    return Task(
        dataset=ethics_preference_dataset(name="justice"),
        solver=combined_sycophancy_solver(),
        scorer=sycophancy_scorer()
    )
