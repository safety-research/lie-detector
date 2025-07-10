#!/usr/bin/env python3
"""
Example run script for tool results sandbagging detection.

This script demonstrates how to run various tool results sandbagging evaluations.
"""

from inspect_ai import eval_set
from .tasks import (
    tool_results_quick_test_task,
    tool_results_sandbagging_task,
    tool_results_comprehensive_test_task,
    tool_results_security_framing_task,
    tool_results_probing_task,
    tool_results_multi_solver_comparison_task
)
import asyncio


async def run_quick_test():
    """Run a quick test to verify the module works."""
    print("Running quick test...")
    
    task = tool_results_quick_test_task(judge_model="openai/gpt-4o")
    
    results = await eval_set(
        tasks=[task],
        model="openai/gpt-4",
        log_dir="./results/quick_test"
    )
    
    print(f"Quick test completed. Results: {results}")
    return results


async def run_basic_evaluation():
    """Run a basic sandbagging evaluation."""
    print("Running basic evaluation...")
    
    task = tool_results_sandbagging_task(
        num_samples=5,
        num_benign_files=4,
        num_harmful_files=3,
        categories=["technical", "medical"],
        solver_type="basic",
        judge_model="openai/gpt-4o"
    )
    
    results = await eval_set(
        tasks=[task],
        model="openai/gpt-4",
        log_dir="./results/basic_evaluation"
    )
    
    print(f"Basic evaluation completed. Results: {results}")
    return results


async def run_comprehensive_test():
    """Run comprehensive testing across multiple categories."""
    print("Running comprehensive test...")
    
    task = tool_results_comprehensive_test_task(
        categories=["technical", "medical", "general", "research"],
        judge_model="openai/gpt-4o"
    )
    
    results = await eval_set(
        tasks=[task],
        model="anthropic/claude-3-sonnet",
        log_dir="./results/comprehensive_test"
    )
    
    print(f"Comprehensive test completed. Results: {results}")
    return results


async def run_solver_comparison():
    """Compare different solver strategies."""
    print("Running solver comparison...")
    
    tasks = tool_results_multi_solver_comparison_task(judge_model="openai/gpt-4o")
    
    results = await eval_set(
        tasks=tasks,
        model="openai/gpt-4",
        log_dir="./results/solver_comparison"
    )
    
    print(f"Solver comparison completed. Results: {results}")
    return results


async def run_security_framing_test():
    """Test with security audit framing."""
    print("Running security framing test...")
    
    task = tool_results_security_framing_task(judge_model="openai/gpt-4o")
    
    results = await eval_set(
        tasks=[task],
        model="openai/gpt-4",
        log_dir="./results/security_framing"
    )
    
    print(f"Security framing test completed. Results: {results}")
    return results


async def run_probing_test():
    """Test with follow-up probing questions."""
    print("Running probing test...")
    
    task = tool_results_probing_task(judge_model="openai/gpt-4o")
    
    results = await eval_set(
        tasks=[task],
        model="openai/gpt-4",
        log_dir="./results/probing_test"
    )
    
    print(f"Probing test completed. Results: {results}")
    return results


async def run_all_tests():
    """Run all available tests."""
    print("Running all tool results sandbagging tests...")
    
    # Quick test first
    await run_quick_test()
    
    # Basic evaluation
    await run_basic_evaluation()
    
    # Security framing
    await run_security_framing_test()
    
    # Probing questions
    await run_probing_test()
    
    # Solver comparison
    await run_solver_comparison()
    
    # Comprehensive test (takes longer)
    await run_comprehensive_test()
    
    print("All tests completed!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run tool results sandbagging detection evaluations"
    )
    parser.add_argument(
        "--test",
        choices=["quick", "basic", "comprehensive", "security", "probing", "comparison", "all"],
        default="quick",
        help="Which test to run"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4",
        help="Model to evaluate"
    )
    parser.add_argument(
        "--judge-model", 
        default="openai/gpt-4o",
        help="Model to use for judging/scoring"
    )
    
    args = parser.parse_args()
    
    # Map test names to functions
    test_functions = {
        "quick": run_quick_test,
        "basic": run_basic_evaluation,
        "comprehensive": run_comprehensive_test,
        "security": run_security_framing_test,
        "probing": run_probing_test,
        "comparison": run_solver_comparison,
        "all": run_all_tests
    }
    
    # Run the selected test
    test_func = test_functions[args.test]
    asyncio.run(test_func())


if __name__ == "__main__":
    main()


# Example usage:
# python -m sandbagging.tool_results.run --test quick --model openai/gpt-4
# python -m sandbagging.tool_results.run --test comprehensive --model anthropic/claude-3-sonnet
# python -m sandbagging.tool_results.run --test all