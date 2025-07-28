#!/usr/bin/env python3
"""
Print all task names from generalization mappings.
"""

import sys
import os

# Add the common directory to the path to import generalization_mappings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from generalization_mappings import generalization_map_1 as map_1, generalization_map_2 as map_2

def print_all_tasks():
    """Print all unique task names from both generalization maps."""
    
    # Collect all tasks from both maps
    all_tasks = set()
    
    print("="*80)
    print("ALL TASKS FROM GENERALIZATION MAPPINGS")
    print("="*80)
    
    print("\nGENERALIZATION MAP 1 (Motivation):")
    print("-" * 50)
    for category, tasks in map_1.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for task in tasks:
            print(f"  - {task}")
            all_tasks.add(task)
    
    print("\nGENERALIZATION MAP 2 (Knowledge):")
    print("-" * 50)
    for category, tasks in map_2.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for task in tasks:
            print(f"  - {task}")
            all_tasks.add(task)
    
    print("\n" + "="*80)
    print("UNIQUE TASK LIST (ALPHABETICAL ORDER):")
    print("="*80)
    
    # Print all unique tasks in alphabetical order
    for task in sorted(all_tasks):
        print(f"  {task}")
    
    print(f"\nTotal unique tasks: {len(all_tasks)}")

if __name__ == "__main__":
    print_all_tasks() 