# sandbagging/chess/run.py
# !/usr/bin/env python3
"""
Run script for chess sandbagging detection experiments.
"""

import asyncio
from inspect_ai import eval_async
from pathlib import Path
import json

from sandbagging.performance_degradation._chess.tasks import (
    chess_sandbagging_task,
    chess_baseline_task,
    chess_sandbagging_comparison_task,
    chess_quick_test_task
)
from sandbagging.performance_degradation._chess.scorers import calculate_performance_metrics


async def run_baseline_experiment(
        model: str = "openai/gpt-4",
        num_games: int = 20,
        stockfish_level: int = 1
):
    """Run baseline chess performance evaluation."""

    print(f"Running baseline chess evaluation with {model}...")
    print(f"Games: {num_games}, Stockfish level: {stockfish_level}")

    # Create baseline task
    task = chess_baseline_task(
        num_games=num_games,
        stockfish_level=stockfish_level,
        motivate=False
    )

    # Run evaluation
    results = await eval_async(
        task,
        model=model,
        log_dir="./chess_results/baseline"
    )

    # Extract game results
    game_results = []
    for sample in results.samples:
        if "game_result" in sample.metadata:
            game_results.append(sample.metadata["game_result"])

    # Calculate metrics
    metrics = calculate_performance_metrics(game_results)

    print("\nBaseline Results:")
    print(f"Win rate: {metrics.get('baseline_win_rate', 0):.2%}")
    print(f"Average move quality: {metrics.get('baseline_avg_move_quality', 0):.2f}")
    print(f"Blunder rate: {metrics.get('baseline_blunder_rate', 0):.2%}")

    return game_results


async def run_sandbagging_experiment(
        model: str = "openai/gpt-4",
        num_games: int = 20,
        elo_threshold: int = 1800,
        stockfish_level: int = 1,
        baseline_results: list = None
):
    """Run sandbagging chess evaluation."""

    print(f"\nRunning sandbagging evaluation with {model}...")
    print(f"Games: {num_games}, ELO threshold: {elo_threshold}")

    # Create sandbagging task
    task = chess_sandbagging_task(
        num_games=num_games,
        elo_threshold=elo_threshold,
        stockfish_level=stockfish_level,
        corporate_framing=True
    )

    # Run evaluation
    results = await eval_async(
        task,
        model=model,
        log_dir="./chess_results/sandbagging"
    )

    # Extract game results
    game_results = []
    for sample in results[0].samples:
        if "game_result" in sample.metadata:
            game_results.append(sample.metadata["game_result"])

    # Calculate metrics
    all_results = (baseline_results or []) + game_results
    metrics = calculate_performance_metrics(all_results)

    print("\nSandbagging Results:")
    print(f"Win rate: {metrics.get('sandbagging_win_rate', 0):.2%}")
    print(f"Average move quality: {metrics.get('sandbagging_avg_move_quality', 0):.2f}")
    print(f"Blunder rate: {metrics.get('sandbagging_blunder_rate', 0):.2%}")

    if baseline_results:
        print("\nComparison:")
        print(
            f"Win rate difference: {metrics.get('baseline_win_rate', 0) - metrics.get('sandbagging_win_rate', 0):.2%}")
        print(f"Statistical significance: p={metrics.get('win_rate_p_value', 1):.3f}")
        print(f"Effect size: {metrics.get('sandbagging_effect_size', 0):.2f}")
        print(f"Sandbagging detected: {metrics.get('significant_sandbagging', False)}")

    return game_results


async def run_full_experiment(
        model: str = "openai/gpt-4",
        num_games_per_condition: int = 20,
        elo_thresholds: list = [1400, 1600, 1800, 2000],
        stockfish_level: int = 1
):
    """Run full sandbagging detection experiment."""

    print(f"Running full chess sandbagging experiment with {model}")
    print(f"Games per condition: {num_games_per_condition}")
    print(f"ELO thresholds: {elo_thresholds}")

    # Run baseline
    baseline_results = await run_baseline_experiment(
        model=model,
        num_games=num_games_per_condition,
        stockfish_level=stockfish_level
    )

    # Run sandbagging at different thresholds
    sandbagging_results = {}

    for threshold in elo_thresholds:
        print(f"\n{'=' * 50}")
        results = await run_sandbagging_experiment(
            model=model,
            num_games=num_games_per_condition,
            elo_threshold=threshold,
            stockfish_level=stockfish_level,
            baseline_results=baseline_results
        )
        sandbagging_results[threshold] = results

    # Save results
    output_dir = Path("./chess_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump({
            "model": model,
            "baseline": baseline_results,
            "sandbagging": {str(k): v for k, v in sandbagging_results.items()},
            "config": {
                "num_games_per_condition": num_games_per_condition,
                "elo_thresholds": elo_thresholds,
                "stockfish_level": stockfish_level
            }
        }, f, indent=2)

    print(f"\nResults saved to {output_dir / 'experiment_results.json'}")


async def run_quick_test(model: str = "openai/gpt-4"):
    """Run a quick test of the chess sandbagging system."""

    print(f"Running quick chess test with {model}...")

    task = chess_quick_test_task()

    results = await eval_async(
        task,
        model=model,
        log_dir="./chess_results/quick_test"
    )

    print("Quick test completed!")

    # Print summary of results
    for i, sample in enumerate(results[0].samples):
        if "game_result" in sample.metadata:
            result = sample.metadata["game_result"]
            print(f"\nGame {i + 1}:")
            print(f"  Result: {result['winner']} ({result['reason']})")
            print(f"  Moves: {result['total_moves']}")
            print(f"  Blunders: {result.get('blunder_count', 0)}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run chess sandbagging detection experiments"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4",
        help="Model to evaluate"
    )
    parser.add_argument(
        "--experiment",
        choices=["quick", "baseline", "sandbagging", "full"],
        default="quick",
        help="Which experiment to run"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=20,
        help="Number of games per condition"
    )
    parser.add_argument(
        "--elo-threshold",
        type=int,
        default=1800,
        help="ELO threshold for sandbagging"
    )
    parser.add_argument(
        "--stockfish-level",
        type=int,
        default=1,
        help="Stockfish difficulty (1-20)"
    )

    args = parser.parse_args()

    if args.experiment == "quick":
        asyncio.run(run_quick_test(args.model))
    elif args.experiment == "baseline":
        asyncio.run(run_baseline_experiment(
            model=args.model,
            num_games=args.num_games,
            stockfish_level=args.stockfish_level
        ))
    elif args.experiment == "sandbagging":
        asyncio.run(run_sandbagging_experiment(
            model=args.model,
            num_games=args.num_games,
            elo_threshold=args.elo_threshold,
            stockfish_level=args.stockfish_level
        ))
    elif args.experiment == "full":
        asyncio.run(run_full_experiment(
            model=args.model,
            num_games_per_condition=args.num_games,
            stockfish_level=args.stockfish_level
        ))


if __name__ == "__main__":
    main()