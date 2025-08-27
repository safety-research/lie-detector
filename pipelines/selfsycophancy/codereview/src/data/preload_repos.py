#!/usr/bin/env python3
"""
Preload repositories for offline access in sandbox
"""
import json
import subprocess
from pathlib import Path
from typing import Dict, List
import tempfile
import base64


class RepositoryPreloader:
    def __init__(self, cache_dir: Path = Path("data/repo_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def preload_repo_as_tar(self, repo: str, commit: str) -> Path:
        """Clone and create a tar archive of repository at specific commit"""
        repo_name = repo.replace("/", "_")
        tar_path = self.cache_dir / f"{repo_name}_{commit[:8]}.tar.gz"

        if tar_path.exists():
            print(f"  Using cached {tar_path.name}")
            return tar_path

        print(f"  Cloning {repo} at {commit[:8]}...")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            try:
                # First try shallow clone with specific commit
                subprocess.run([
                    "git", "clone",
                    "--single-branch",
                    "--depth", "1",
                    "--branch", commit,
                    f"https://github.com/{repo}.git",
                    str(tmp_path / "repo")
                ], check=True, capture_output=True, text=True)

            except subprocess.CalledProcessError:
                # If that fails, try full clone
                print(f"    Shallow clone failed, trying full clone...")

                try:
                    # Clone without depth limit
                    subprocess.run([
                        "git", "clone",
                        f"https://github.com/{repo}.git",
                        str(tmp_path / "repo")
                    ], check=True, capture_output=True, text=True)

                    # Checkout specific commit
                    subprocess.run([
                        "git", "checkout", commit
                    ], cwd=tmp_path / "repo", check=True, capture_output=True, text=True)

                except subprocess.CalledProcessError as e:
                    # Last resort: try fetching the specific commit
                    print(f"    Full clone failed, trying to fetch specific commit...")

                    # Clone with no checkout
                    subprocess.run([
                        "git", "clone", "--no-checkout",
                        f"https://github.com/{repo}.git",
                        str(tmp_path / "repo")
                    ], check=True)

                    # Fetch the specific commit
                    subprocess.run([
                        "git", "fetch", "origin", commit
                    ], cwd=tmp_path / "repo", check=True)

                    # Checkout the commit
                    subprocess.run([
                        "git", "checkout", commit
                    ], cwd=tmp_path / "repo", check=True)

            # Remove .git directory to save space
            git_dir = tmp_path / "repo" / ".git"
            if git_dir.exists():
                import shutil
                shutil.rmtree(git_dir)

            # Create tar archive
            subprocess.run([
                "tar", "-czf", str(tar_path),
                "-C", str(tmp_path), "repo"
            ], check=True)

            print(f"    Created {tar_path.name} ({tar_path.stat().st_size / 1024 / 1024:.1f} MB)")

        return tar_path