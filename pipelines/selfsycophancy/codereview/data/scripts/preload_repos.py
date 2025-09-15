#!/usr/bin/env python3
"""
Preload all repositories needed for experiments
"""
import sys
from pathlib import Path

from pipelines.selfsycophancy.codereview.src.data.swe_bench_loader import SWEBenchLoader
from pipelines.selfsycophancy.codereview.src.data.preload_repos import RepositoryPreloader

sys.path.append(str(Path(__file__).parent.parent))


def main():
    loader = SWEBenchLoader()
    preloader = RepositoryPreloader()

    # Load instances
    print("Loading SWE-Bench Lite instances...")
    instances = loader.load_swe_bench_lite()#[:10]
    print(f"Found {len(instances)} total instances")

    # Preload each unique repo/commit pair
    seen = set()
    successful = 0
    failed = []

    for i, instance in enumerate(instances):
        key = (instance['repo'], instance['base_commit'])
        if key not in seen:
            seen.add(key)
            print(f"\n[{i + 1}/{len(instances)}] Preloading {instance['repo']} at {instance['base_commit'][:8]}...")

            try:
                preloader.preload_repo_as_tar(instance['repo'], instance['base_commit'])
                successful += 1
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                failed.append({
                    'repo': instance['repo'],
                    'commit': instance['base_commit'],
                    'error': str(e)
                })

    print(f"\n{'=' * 50}")
    print(f"Successfully preloaded: {successful}/{len(seen)} repository states")

    if failed:
        print(f"\nFailed repositories:")
        for f in failed[:5]:  # Show first 5 failures
            print(f"  - {f['repo']} @ {f['commit'][:8]}: {f['error'][:50]}...")

        # Save failed list for debugging
        import json
        failed_file = Path("data/repo_cache/failed_repos.json")
        with open(failed_file, 'w') as f:
            json.dump(failed, f, indent=2)
        print(f"\nFull failure list saved to: {failed_file}")


if __name__ == "__main__":
    main()