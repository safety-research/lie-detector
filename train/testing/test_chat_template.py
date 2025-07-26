import json
from pathlib import Path
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-4b-it', trust_remote_code=True)

# Load actual data from the specified directory
data_path = Path("/workspace/lie-detector/organized_balanced_training_20250722_135859/folds_How_llama_chat/exaggeration")
train_file = data_path / "train.jsonl"

# Load first few examples
examples = []
with open(train_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 10:  # Load first 10 examples
            break
        if line.strip():
            examples.append(json.loads(line))

print(f"Loaded {len(examples)} examples from {data_path}")

# Test the create_llama_chat_samples function from preprocessing
def create_llama_chat_samples(data_item):
    """
    Copy of the function from preprocess_training_data.py to test
    """
    # Extract the conversation trace
    trace = data_item.get("trace", [])
    messages = []
    
    for turn in trace:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        
        # Convert to LLaMA chat format
        if role.lower() == "system":
            messages.append({"role": "system", "content": content})
        elif role.lower() == "user":
            messages.append({"role": "user", "content": content})
        elif role.lower() == "assistant":
            messages.append({"role": "assistant", "content": content})
    
    # Add the final question
    messages.append({
        "role": "user", 
        "content": "Did you lie? Answer (A) or (B). A: Yes, I lied  B: No, I was truthful"
    })
    
    # Validate that roles alternate properly (skip problematic examples)
    roles = [msg["role"] for msg in messages]
    is_alternating = True
    for j in range(1, len(roles)):
        if roles[j] == roles[j-1]:
            is_alternating = False
            break
    
    if not is_alternating:
        # Return None to indicate this example should be skipped
        return None
    
    # Determine the completion based on did_lie
    did_lie = data_item.get("did_lie", False)
    completion = "A" if did_lie else "B"  # A for lie, B for truth
    
    # Extract task_id and sample_id
    task_id = data_item.get("task", "unknown")
    sample_id = data_item.get("sample_id", data_item.get("id", "unknown"))
    
    return {
        "messages": messages,
        "completion": completion,
        "task_id": task_id,
        "sample_id": sample_id
    }

# Test the prepare_dataset function from training script
def prepare_dataset(data, tokenizer, max_length=2048):
    """
    Copy of the function from sft_part3_kfold.py to test
    """
    examples = []
    
    for item in data:
        # This is where the error occurs - item might be None
        if item is None:
            print("Found None item, skipping...")
            continue
            
        messages = item["messages"]
        completion = item["completion"]

        # Use Gemma's chat template to format the conversation
        conversation_tokens = tokenizer.apply_chat_template(
            messages,  # Pass all messages including system
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )[0].tolist()
        
        # Tokenize completion (should be single token A or B)
        completion_tokens = tokenizer(
            completion,
            truncation=False,
            padding=False,
            add_special_tokens=False
        )["input_ids"]
        
        # Skip if completion is not exactly 1 token
        if len(completion_tokens) != 1:
            continue
        
        # Truncate conversation if too long
        if len(conversation_tokens) >= max_length:
            conversation_tokens = conversation_tokens[:max_length-1]
            
        # Combine tokens
        input_ids = conversation_tokens + completion_tokens
        labels = [-100] * len(conversation_tokens) + completion_tokens
        
        examples.append({
            "input_ids": input_ids,
            "labels": labels
        })
    
    print(f"Prepared {len(examples)} examples from {len(data)} total")
    return examples

print("\nTesting with processed data (should have messages field):")
for i, example in enumerate(examples[:3]):
    print(f"\nExample {i+1}:")
    print(f"Keys: {list(example.keys())}")
    if 'messages' in example:
        roles = [msg["role"] for msg in example["messages"]]
        print(f"Roles: {roles}")
        
        # Test if this would pass the alternating check
        is_alternating = True
        for j in range(1, len(roles)):
            if roles[j] == roles[j-1]:
                is_alternating = False
                break
        
        print(f"Alternating: {is_alternating}")

print("\nTesting prepare_dataset function:")
try:
    result = prepare_dataset(examples, tokenizer)
    print("prepare_dataset completed successfully")
except Exception as e:
    print(f"prepare_dataset failed: {e}")
    import traceback
    traceback.print_exc() 