# Naive prompt detector
python evaluate_mask.py --detector naive --all-subsets --model gpt-4o-mini

# Elicitation classifier with custom settings
python evaluate_mask.py --detector elicitation --model gpt-4o-mini --n-elicitation 15 --all-subsets


# Models

Llama3.3, Qwen2.5 and Phi. 

# OpenRouter