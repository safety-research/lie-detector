#!/usr/bin/env python3

from finetuning import LieDetectionFinetuner

# Create finetuner instance
finetuner = LieDetectionFinetuner(
    '/Users/dipikakhullar/Desktop/lie_detector_clone/lie-detector/.data/openai/gpt_oss_120b_limited', 
    '876b395e972814ee915c5c2cf1b645dc176ccf159b371bf48feb4c801ffe8cda'
)

# Test the function call
print("Testing create_finetune_job function...")
print("Function signature:", finetuner.create_finetune_job.__code__.co_varnames)
print("Default values:", finetuner.create_finetune_job.__defaults__)

# Test calling with explicit model name
print("\nTesting with explicit OSS 120B model name...")
try:
    # We can't actually create a job without a real file ID, but we can see what model name would be used
    import inspect
    sig = inspect.signature(finetuner.create_finetune_job)
    print("Model name that would be used:", sig.parameters['model_name'].default)
except Exception as e:
    print(f"Error: {e}")
