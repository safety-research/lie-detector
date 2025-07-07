import os
from inspect_ai import eval, Task
from evals.sandbagging_tools_tasks import tool_search_sandbagging_task, tool_image_gen_sandbagging_task, \
    tool_code_sandbagging_task, tool_ethical_sandbagging_task, tool_unethical_sandbagging_task

if __name__ == "__main__":
    # Set up OpenRouter authentication for Tanuki
    if "OPENROUTER_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        print("OpenRouter authentication configured for Tanuki")
    else:
        print("Warning: OPENROUTER_API_KEY not found in environment variables")

    # Define all sandbagging tools tasks
    all_tasks = [
        tool_search_sandbagging_task(),
        tool_image_gen_sandbagging_task(),
        tool_code_sandbagging_task(),
        tool_ethical_sandbagging_task(),
        tool_unethical_sandbagging_task()
    ]

    print("Running all sandbagging tools tasks...")
    print(f"Number of tasks: {len(all_tasks)}")
    print(f"Limit: 10 samples per task")

    # Run evaluation with all tasks
    model = "openrouter/meta-llama/llama-3.1-8b-instruct"

    results = eval(
        tasks=all_tasks,
        model=model,  # Using OpenRouter model
        limit=10,  # Limit samples for faster debugging
        log_dir="./logs",  # Specify log directory
    )