import json
from pathlib import Path
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-4b-it', trust_remote_code=True)

# Load actual data
input_path = Path("/workspace/lie-detector/organized_balanced_training_20250722_135859/folds_How_llama_chat")
fold_dirs = [d for d in input_path.iterdir() if d.is_dir()]

# Load first few examples from each fold
all_examples = []
for fold_dir in fold_dirs[:2]:  # Just test first 2 folds
    train_file = fold_dir / "train.jsonl"
    if train_file.exists():
        with open(train_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Just test first 3 examples
                    break
                if line.strip():
                    all_examples.append(json.loads(line))

print(f"Testing {len(all_examples)} examples from actual data")

for i, example in enumerate(all_examples):
    print(f"\n{'='*80}")
    print(f"Example {i+1}:")
    
    messages = example["messages"]
    print(f"Number of messages: {len(messages)}")
    
    # Print message roles
    roles = [msg["role"] for msg in messages]
    print(f"Roles: {roles}")
    
    # Check if roles alternate properly
    is_alternating = True
    for j in range(1, len(roles)):
        if roles[j] == roles[j-1]:
            is_alternating = False
            print(f"  Non-alternating at position {j}: {roles[j-1]} -> {roles[j]}")
    
    if is_alternating:
        print("  ✓ Roles alternate properly")
    else:
        print("  ✗ Roles don't alternate properly")
    
    # Try to apply chat template
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("  ✓ Chat template applied successfully")
        print(f"  Formatted length: {len(formatted)}")
    except Exception as e:
        print(f"  ✗ Chat template failed: {e}")
        
        # Try with filtered messages
        try:
            filtered_messages = [msg for msg in messages if msg["role"] != "system"]
            formatted = tokenizer.apply_chat_template(
                filtered_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print("  ✓ Chat template worked with filtered messages")
        except Exception as e2:
            print(f"  ✗ Even filtered messages failed: {e2}")
    
    # Show first few messages in detail
    print("  First few messages:")
    for j, msg in enumerate(messages[:3]):
        content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"    {j}: {msg['role']}: {content_preview}") 