# src/data/swe_bench_loader.py
"""
SWE-Bench dataset loader for code review experiments.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import requests
from datasets import load_dataset


class SWEBenchLoader:
    """Load and prepare SWE-Bench data for code review experiments."""

    def __init__(self, cache_dir: Path = Path("data/swe_bench_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_swe_bench_lite(self) -> List[Dict]:
        """Load SWE-Bench-Lite (subset of 300 instances)."""
        # Load from Hugging Face
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite")

        instances = []
        for item in dataset['test']:
            instances.append({
                "instance_id": item["instance_id"],
                "repo": item["repo"],
                "base_commit": item["base_commit"],
                "problem_statement": item["problem_statement"],
                "hints_text": item.get("hints_text", ""),
                "test_patch": item["test_patch"],
                "gold_patch": item["patch"],  # The actual solution
                "environment_setup_commit": item.get("environment_setup_commit"),
                "FAIL_TO_PASS": item.get("FAIL_TO_PASS", []),
                "PASS_TO_PASS": item.get("PASS_TO_PASS", [])
            })

        return instances

    def load_full_swe_bench(self) -> List[Dict]:
        """Load full SWE-Bench dataset (2,294 instances)."""
        dataset = load_dataset("princeton-nlp/SWE-bench")

        instances = []
        for item in dataset['test']:
            instances.append({
                "instance_id": item["instance_id"],
                "repo": item["repo"],
                "base_commit": item["base_commit"],
                "problem_statement": item["problem_statement"],
                "hints_text": item.get("hints_text", ""),
                "test_patch": item["test_patch"],
                "gold_patch": item["patch"],
                "created_at": item.get("created_at"),
                "version": item.get("version")
            })

        return instances

    def filter_by_difficulty(self, instances: List[Dict],
                             difficulty: str = "medium") -> List[Dict]:
        """Filter instances by estimated difficulty."""
        # Estimate difficulty based on patch size and test complexity
        filtered = []

        for instance in instances:
            patch_lines = len(instance["gold_patch"].split('\n'))

            if difficulty == "easy" and patch_lines < 20:
                filtered.append(instance)
            elif difficulty == "medium" and 20 <= patch_lines < 100:
                filtered.append(instance)
            elif difficulty == "hard" and patch_lines >= 100:
                filtered.append(instance)

        return filtered

    def filter_by_language(self, instances: List[Dict],
                           language: str = "python") -> List[Dict]:
        """Filter instances by programming language."""
        # Most SWE-Bench is Python, but you can extend this
        language_repos = {
            "python": ["django/django", "sympy/sympy", "scikit-learn/scikit-learn",
                       "matplotlib/matplotlib", "requests/requests", "flask/flask"],
            "javascript": [],  # Add JS repos if available
        }

        if language not in language_repos:
            return instances

        return [i for i in instances
                if any(repo in i["repo"] for repo in language_repos[language])]

    def prepare_for_pr_generation(self, instance: Dict) -> Dict:
        """Convert SWE-Bench instance to PR generation format."""
        return {
            "repo_url": f"https://github.com/{instance['repo']}",
            "repo_name": instance['repo'].split('/')[-1],
            "base_commit": instance['base_commit'],
            "issue_description": instance['problem_statement'],
            "hints": instance.get('hints_text', ''),
            "test_requirements": instance.get('FAIL_TO_PASS', []),
            "instance_id": instance['instance_id'],
            "gold_solution": instance['gold_patch']  # Hidden during generation
        }