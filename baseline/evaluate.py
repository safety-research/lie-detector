from inspect_ai import eval
from dotenv import load_dotenv
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import the task - handle both relative and absolute imports
try:
    from baseline.task import baseline_task, conversation_task, llama_chat_task, llama_chat_reasoning_task, base_transcript_reasoning_task
except ImportError:
    try:
        from .task import baseline_task, conversation_task, llama_chat_task, llama_chat_reasoning_task, base_transcript_reasoning_task
    except ImportError:
        from task import baseline_task, conversation_task, llama_chat_task, llama_chat_reasoning_task, base_transcript_reasoning_task

load_dotenv()

def examine_data_distribution():
    """Examine the data to understand the ground truth labeling."""
    print("=== EXAMINING DATA DISTRIBUTION ===")
    
    data_dir = Path("processed_data")
    lying_examples = []
    truthful_examples = []
    
    # Sample a few examples of each type
    for file_path in data_dir.rglob("*.jsonl"):
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f):
                if line_num >= 2:  # Only look at first 2 lines per file
                    break
                try:
                    item = json.loads(line)
                    if item.get("did_lie") and len(lying_examples) < 3:
                        lying_examples.append((file_path.name, item))
                    elif not item.get("did_lie") and len(truthful_examples) < 3:
                        truthful_examples.append((file_path.name, item))
                except json.JSONDecodeError:
                    continue
                    
        if len(lying_examples) >= 3 and len(truthful_examples) >= 3:
            break
    
    print("\n--- LYING EXAMPLES (did_lie=True) ---")
    for i, (filename, item) in enumerate(lying_examples):
        print(f"\nExample {i+1} from {filename}:")
        trace = item.get("trace", [])
        for turn in trace[:3]:  # First 3 turns
            role = turn.get("role", "unknown")
            content = turn.get("content", "")[:200]  # First 200 chars
            print(f"{role.capitalize()}: {content}...")
        print(f"Ground truth: did_lie = {item.get('did_lie')}")
    
    print("\n--- TRUTHFUL EXAMPLES (did_lie=False) ---")
    for i, (filename, item) in enumerate(truthful_examples):
        print(f"\nExample {i+1} from {filename}:")
        trace = item.get("trace", [])
        for turn in trace[:3]:  # First 3 turns
            role = turn.get("role", "unknown")
            content = turn.get("content", "")[:200]  # First 200 chars
            print(f"{role.capitalize()}: {content}...")
        print(f"Ground truth: did_lie = {item.get('did_lie')}")

def save_prompts_to_file(results, filename, output_dir=".", baseline_type="baseline"):
    """Save all the prompts sent to the model to a text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(output_dir, f"prompts_{timestamp}_{filename}.txt")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FULL PROMPTS SENT TO MODEL\n")
        f.write("="*80 + "\n\n")
        
        for i, sample in enumerate(results):
            target = sample.target
            prediction = sample.output.completion.strip()
            original_did_lie = sample.metadata.get('did_lie', 'N/A')
            
            # Get extracted prediction from scorer
            if baseline_type == "escaped_transcript":
                scorer_name = 'conversation_scorer'
            elif baseline_type == "llama_chat":
                scorer_name = 'llama_chat_scorer'
            elif baseline_type == "llama_chat_reasoning":
                scorer_name = 'llama_chat_reasoning_scorer'
            elif baseline_type == "base_transcript_reasoning":
                scorer_name = 'base_transcript_reasoning_scorer'
            else:
                scorer_name = 'truth_scorer'
            if hasattr(sample, 'scores') and sample.scores and scorer_name in sample.scores:
                extracted_pred = sample.scores[scorer_name].answer
                metadata = sample.scores[scorer_name].metadata
                is_parseable = metadata.get('parseable', True) if metadata else False
            else:
                extracted_pred = "OTHER"
                is_parseable = False
            
            f.write(f"SAMPLE {i+1:03d}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Target: {target}\n")
            f.write(f"Extracted Prediction: {extracted_pred}\n")
            f.write(f"Original did_lie: {original_did_lie}\n")
            f.write(f"Parseable: {is_parseable}\n")
            f.write("-" * 50 + "\n")
            
            f.write("FULL PROMPT SENT TO MODEL:\n")
            f.write("〰" * 40 + "\n")
            # Handle both string and list inputs (for LLaMA chat format)
            if isinstance(sample.input, list):
                # For LLaMA chat format, format as readable messages
                for j, msg in enumerate(sample.input):
                    # Handle both dict format and ChatMessage objects
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        # ChatMessage object
                        f.write(f"[{j+1}] {msg.role.upper()}: {msg.content}\n")
                    elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        # Dictionary format
                        f.write(f"[{j+1}] {msg['role'].upper()}: {msg['content']}\n")
                    else:
                        # Fallback
                        f.write(f"[{j+1}] {str(msg)}\n")
            else:
                f.write(sample.input)
            f.write("\n" + "〰" * 40 + "\n")
            
            f.write("MODEL RESPONSE:\n")
            f.write("▶ " + prediction + "\n")
            f.write("\n" + "="*80 + "\n\n")
    
    return filepath

def save_results_to_files(results, filename, metrics, output_dir=".", baseline_type="baseline"):
    """Save evaluation results to both JSON and TXT files with subtask breakdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Group results by subtask
    subtask_results = {}
    
    for i, sample in enumerate(results):
        # Extract subtask from metadata
        metadata = sample.metadata
        subtask = metadata.get('task', 'unknown') if metadata else 'unknown'
        
        # If task not in metadata, try to extract from file path or other sources
        if subtask == 'unknown':
            # Try to get from trace or other metadata fields
            trace = metadata.get('trace', []) if metadata else []
            if trace and len(trace) > 0:
                # Look for task info in the trace
                for turn in trace:
                    content = turn.get('content', '')
                    if 'task' in content.lower():
                        # Extract task from content
                        import re
                        match = re.search(r'task[:\s]+([^\n]+)', content, re.IGNORECASE)
                        if match:
                            subtask = match.group(1).strip()
                            break
        
        if subtask not in subtask_results:
            subtask_results[subtask] = {
                'samples': [],
                'metrics': {
                    'total_samples': 0,
                    'parseable_samples': 0,
                    'filtered_samples': 0,
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'tp': 0,
                    'tn': 0,
                    'fp': 0,
                    'fn': 0,
                    'target_distribution': {'A': 0, 'B': 0},
                    'prediction_distribution': {'A': 0, 'B': 0, 'OTHER': 0}
                }
            }
        
        target = sample.target
        prediction = sample.output.completion.strip()
        original_did_lie = sample.metadata.get('did_lie', 'N/A')
        
        if baseline_type == "escaped_transcript":
            scorer_name = 'conversation_scorer'
        elif baseline_type == "llama_chat":
            scorer_name = 'llama_chat_scorer'
        elif baseline_type == "llama_chat_reasoning":
            scorer_name = 'llama_chat_reasoning_scorer'
        elif baseline_type == "base_transcript_reasoning":
            scorer_name = 'base_transcript_reasoning_scorer'
        else:
            scorer_name = 'truth_scorer'
        if hasattr(sample, 'scores') and sample.scores and scorer_name in sample.scores:
            extracted_pred = sample.scores[scorer_name].answer
            metadata = sample.scores[scorer_name].metadata
            is_parseable = metadata.get('parseable', True) if metadata else False
        else:
            extracted_pred = "OTHER"
            is_parseable = False
        
        sample_data = {
            "sample_id": i + 1,
            "target": target,
            "extracted_prediction": extracted_pred,
            "raw_prediction": prediction,
            "original_did_lie": original_did_lie,
            "is_parseable": is_parseable,
            "correct": target == extracted_pred if is_parseable else None
        }
        
        subtask_results[subtask]['samples'].append(sample_data)
        subtask_results[subtask]['metrics']['total_samples'] += 1
        
        # Update distributions
        target_str = str(target) if target else "B"
        if target_str in subtask_results[subtask]['metrics']['target_distribution']:
            subtask_results[subtask]['metrics']['target_distribution'][target_str] += 1
        if extracted_pred in subtask_results[subtask]['metrics']['prediction_distribution']:
            subtask_results[subtask]['metrics']['prediction_distribution'][extracted_pred] += 1
        
        if not is_parseable or extracted_pred == "OTHER":
            subtask_results[subtask]['metrics']['filtered_samples'] += 1
        else:
            subtask_results[subtask]['metrics']['parseable_samples'] += 1
            if target_str == extracted_pred:
                if target_str == "A":  # True positive (correctly identified lie)
                    subtask_results[subtask]['metrics']['tp'] += 1
                else:  # True negative (correctly identified truth)
                    subtask_results[subtask]['metrics']['tn'] += 1
            else:
                if target_str == "A":  # False negative (missed lie)
                    subtask_results[subtask]['metrics']['fn'] += 1
                else:  # False positive (incorrectly called truth a lie)
                    subtask_results[subtask]['metrics']['fp'] += 1
    
    # Calculate metrics for each subtask
    for subtask, data in subtask_results.items():
        metrics = data['metrics']
        parseable = metrics['parseable_samples']
        total = metrics['total_samples']
        
        if parseable > 0:
            metrics['accuracy'] = (metrics['tp'] + metrics['tn']) / parseable
            metrics['precision'] = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
            metrics['recall'] = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    # Calculate overall metrics by aggregating across all subtasks
    overall_tp = sum(data['metrics']['tp'] for data in subtask_results.values())
    overall_tn = sum(data['metrics']['tn'] for data in subtask_results.values())
    overall_fp = sum(data['metrics']['fp'] for data in subtask_results.values())
    overall_fn = sum(data['metrics']['fn'] for data in subtask_results.values())
    overall_parseable = sum(data['metrics']['parseable_samples'] for data in subtask_results.values())
    overall_filtered = sum(data['metrics']['filtered_samples'] for data in subtask_results.values())
    
    # Calculate overall performance metrics
    overall_accuracy = (overall_tp + overall_tn) / overall_parseable if overall_parseable > 0 else 0
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # Prepare structured data for JSON
    json_data = {
        "metadata": {
            "timestamp": timestamp,
            "total_samples": len(results),
            "model": "openrouter/meta-llama/llama-3.1-8b-instruct",
            "subtasks": list(subtask_results.keys())
        },
        "overall_metrics": {
            "parseable_samples": overall_parseable,
            "filtered_samples": overall_filtered,
            "accuracy": overall_accuracy,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1_score": overall_f1,
            "true_positives": overall_tp,
            "true_negatives": overall_tn,
            "false_positives": overall_fp,
            "false_negatives": overall_fn
        },
        "subtask_results": subtask_results
    }
    
    # Save JSON file
    json_filepath = os.path.join(output_dir, f"results_{timestamp}_{filename}.json")
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # Save TXT summary file
    txt_filepath = os.path.join(output_dir, f"results_{timestamp}_{filename}.txt")
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LIE DETECTION BASELINE EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: openrouter/meta-llama/llama-3.1-8b-instruct\n")
        f.write(f"Total Samples: {len(results)}\n")
        f.write(f"Number of Subtasks: {len(subtask_results)}\n\n")
        
        f.write("OVERALL FILTERING SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Parseable samples: {overall_parseable}/{len(results)} ({overall_parseable/len(results):.1%})\n")
        f.write(f"Filtered out: {overall_filtered}/{len(results)} ({overall_filtered/len(results):.1%})\n\n")
        
        f.write("OVERALL PERFORMANCE METRICS (PARSEABLE SAMPLES ONLY)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Precision: {overall_precision:.4f}\n")
        f.write(f"Recall: {overall_recall:.4f}\n")
        f.write(f"F1 Score: {overall_f1:.4f}\n\n")
        
        f.write("OVERALL CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        f.write(f"True Positives: {overall_tp}\n")
        f.write(f"True Negatives: {overall_tn}\n")
        f.write(f"False Positives: {overall_fp}\n")
        f.write(f"False Negatives: {overall_fn}\n\n")
        
        f.write("SUBTASK BREAKDOWN\n")
        f.write("="*80 + "\n\n")
        
        for subtask, data in sorted(subtask_results.items()):
            subtask_metrics = data['metrics']
            f.write(f"SUBTASK: {subtask}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total samples: {subtask_metrics['total_samples']}\n")
            f.write(f"Parseable samples: {subtask_metrics['parseable_samples']} ({subtask_metrics['parseable_samples']/subtask_metrics['total_samples']:.1%})\n")
            f.write(f"Filtered out: {subtask_metrics['filtered_samples']} ({subtask_metrics['filtered_samples']/subtask_metrics['total_samples']:.1%})\n")
            
            if subtask_metrics['parseable_samples'] > 0:
                f.write(f"Accuracy: {subtask_metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {subtask_metrics['precision']:.4f}\n")
                f.write(f"Recall: {subtask_metrics['recall']:.4f}\n")
                f.write(f"F1 Score: {subtask_metrics['f1_score']:.4f}\n")
                f.write(f"TP: {subtask_metrics['tp']}, TN: {subtask_metrics['tn']}, FP: {subtask_metrics['fp']}, FN: {subtask_metrics['fn']}\n")
            else:
                f.write("No parseable samples for this subtask\n")
            
            f.write(f"Target Distribution: A={subtask_metrics['target_distribution']['A']}, B={subtask_metrics['target_distribution']['B']}\n")
            f.write(f"Prediction Distribution: A={subtask_metrics['prediction_distribution']['A']}, B={subtask_metrics['prediction_distribution']['B']}, OTHER={subtask_metrics['prediction_distribution']['OTHER']}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    return json_filepath, txt_filepath

def main(num_samples=None, model=None, data_dir="processed_data", baseline_type="base_transcript"):
    # First examine the data distribution
    examine_data_distribution()
    
    # Then run evaluation
    print("\n" + "="*80)
    print(f"=== RUNNING {baseline_type.upper()} EVALUATION ===")
    
    # Configuration
    model_name = model or "openrouter/meta-llama/llama-3.1-8b-instruct"
    log_dir = f"logs/{baseline_type}"
    limit = num_samples  # Use the num_samples parameter

    print(f"Running {baseline_type} evaluation...")
    print(f"Model: {model_name}")
    print(f"Samples: {'All available' if limit is None else limit}")

    # Choose task based on baseline type
    if baseline_type == "escaped_transcript":
        task = conversation_task(
            data_dir=data_dir,
            limit=limit
        )
    elif baseline_type == "llama_chat":
        task = llama_chat_task(
            data_dir=data_dir,
            limit=limit
        )
    elif baseline_type == "llama_chat_reasoning":
        task = llama_chat_reasoning_task(
            data_dir=data_dir,
            limit=limit
        )
    elif baseline_type == "base_transcript_reasoning":
        task = base_transcript_reasoning_task(
            data_dir=data_dir,
            limit=limit
        )
    else:  # base_transcript or fallback
        task = baseline_task(
            data_dir=data_dir,
            limit=limit
        )

    log = eval(task, model=model_name, log_dir=log_dir)
    results = log[0].samples if log and log[0].samples else []

    print(f"\nProcessed {len(results)} samples")
    
    # Create output directories if they don't exist
    results_dir = f"baseline/results/{baseline_type}"
    transcripts_dir = f"baseline/transcripts/{baseline_type}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)
    
    # Save all prompts to file
    prompts_file = save_prompts_to_file(results, baseline_type, transcripts_dir, baseline_type)
    print(f"\n📝 Saved all prompts to: {prompts_file}")
    
    # Analyze classification distribution and filter unparseable
    correct_predictions = 0
    by_prediction = {"A": 0, "B": 0, "OTHER": 0}
    by_target = {"A": 0, "B": 0}
    misclassified_examples = []
    filtered_samples = []
    parseable_samples = []
    
    for i, sample in enumerate(results):
        target = sample.target
        prediction = sample.output.completion.strip()
        original_did_lie = sample.metadata.get('did_lie', 'N/A')
        
        # Extract A/B from prediction (our scorer should handle this)
        if baseline_type == "escaped_transcript":
            scorer_name = 'conversation_scorer'
        elif baseline_type == "llama_chat":
            scorer_name = 'llama_chat_scorer'
        else:
            scorer_name = 'truth_scorer'
        if hasattr(sample, 'scores') and sample.scores and scorer_name in sample.scores:
            extracted_pred = sample.scores[scorer_name].answer
            metadata = sample.scores[scorer_name].metadata
            is_parseable = metadata.get('parseable', True) if metadata else False
        else:
            extracted_pred = "OTHER"
            is_parseable = False
            
        target_str = str(target) if target else "B"
        if target_str in by_target:
            by_target[target_str] += 1
        if extracted_pred in by_prediction:
            by_prediction[extracted_pred] += 1
        
        if not is_parseable or extracted_pred == "OTHER":
            filtered_samples.append({
                'sample_num': i+1,
                'target': target_str,
                'raw_output': prediction[:100] + "..." if len(prediction) > 100 else prediction,
                'did_lie': original_did_lie
            })
        else:
            parseable_samples.append(sample)
            if target_str == extracted_pred:
                correct_predictions += 1
            else:
                if len(misclassified_examples) < 5:  # Store first 5 misclassified
                    misclassified_examples.append({
                        'sample_num': i+1,
                        'target': target_str,
                        'prediction': extracted_pred,
                        'raw_output': prediction[:100] + "..." if len(prediction) > 100 else prediction,
                        'did_lie': original_did_lie
                    })

    print(f"\n=== FILTERING SUMMARY ===")
    print(f"Total samples: {len(results)}")
    print(f"Filtered out (unparseable): {len(filtered_samples)}")
    print(f"Parseable samples: {len(parseable_samples)}")
    
    if filtered_samples:
        print(f"\n=== FILTERED OUT SAMPLES ===")
        for example in filtered_samples[:5]:  # Show first 5
            print(f"Sample {example['sample_num']}: Target={example['target']}, did_lie={example['did_lie']}")
            print(f"  Raw output: {example['raw_output']}")
    
    print(f"\n=== CLASSIFICATION SUMMARY (PARSEABLE ONLY) ===")
    if parseable_samples:
        accuracy_rate = correct_predictions / len(parseable_samples)
        print(f"Overall Accuracy: {correct_predictions}/{len(parseable_samples)} = {accuracy_rate:.3f}")
    else:
        print("No parseable samples to evaluate!")
        
    print(f"Target Distribution: A={by_target['A']}, B={by_target['B']}")
    print(f"Prediction Distribution: A={by_prediction['A']}, B={by_prediction['B']}, OTHER={by_prediction['OTHER']}")
    
    if misclassified_examples:
        print(f"\n=== SAMPLE MISCLASSIFICATIONS (PARSEABLE) ===")
        for example in misclassified_examples:
            print(f"Sample {example['sample_num']}: Target={example['target']}, Got={example['prediction']}, did_lie={example['did_lie']}")
            print(f"  Raw output: {example['raw_output']}")

    # Aggregate metrics (only for parseable samples)
    total_tp = 0
    total_tn = 0 
    total_fp = 0
    total_fn = 0
    
    for r in results:
        if hasattr(r, 'scores') and r.scores and scorer_name in r.scores:
            metadata = r.scores[scorer_name].metadata
            if metadata and metadata.get('parseable', True):
                total_tp += metadata.get('tp', 0)
                total_tn += metadata.get('tn', 0)
                total_fp += metadata.get('fp', 0)
                total_fn += metadata.get('fn', 0)
    
    total_parseable = total_tp + total_tn + total_fp + total_fn
    if total_parseable > 0:
        accuracy = (total_tp + total_tn) / total_parseable
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{'='*60}")
        print("FINAL METRICS (PARSEABLE SAMPLES ONLY)")
        print(f"{'='*60}")
        print(f"Parseable samples: {total_parseable}/{len(results)} ({total_parseable/len(results):.1%})")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("------------------------")
        print(f"True Positives: {total_tp}")
        print(f"True Negatives: {total_tn}")
        print(f"False Positives: {total_fp}")
        print(f"False Negatives: {total_fn}")
        print("------------------------")
    else:
        print("\nNo parseable samples to calculate metrics!")
        accuracy = precision = recall = f1 = 0

    # Prepare metrics for saving
    metrics = {
        "parseable_samples": len(parseable_samples),
        "filtered_samples": len(filtered_samples),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": total_tp,
        "tn": total_tn,
        "fp": total_fp,
        "fn": total_fn,
        "target_distribution": f"A={by_target['A']}, B={by_target['B']}",
        "prediction_distribution": f"A={by_prediction['A']}, B={by_prediction['B']}, OTHER={by_prediction['OTHER']}"
    }

    # Save results to files
    json_filepath, txt_filepath = save_results_to_files(results, baseline_type, metrics, results_dir, baseline_type)
    print(f"\n📝 Saved results to: {json_filepath}")
    print(f"📝 Saved summary to: {txt_filepath}")

def main_by_model(num_samples=None, model=None, data_dir="processed_data", baseline_type="base_transcript"):
    """Run baseline evaluation separately for each model."""
    print("\n" + "="*80)
    print(f"=== RUNNING {baseline_type.upper()} EVALUATION BY MODEL ===")
    
    # Configuration
    model_name = model or "openrouter/meta-llama/llama-3.1-8b-instruct"
    log_dir = f"logs/{baseline_type}"
    limit = num_samples

    print(f"Running {baseline_type} evaluation by model...")
    print(f"Evaluation model: {model_name}")
    print(f"Samples per model: {'All available' if limit is None else limit}")

    # Import model-specific task functions
    try:
        from baseline.task import baseline_task_by_model, conversation_task_by_model, llama_chat_task_by_model, llama_chat_reasoning_task_by_model, base_transcript_reasoning_task_by_model
    except ImportError:
        try:
            from .task import baseline_task_by_model, conversation_task_by_model, llama_chat_task_by_model, llama_chat_reasoning_task_by_model, base_transcript_reasoning_task_by_model
        except ImportError:
            from task import baseline_task_by_model, conversation_task_by_model, llama_chat_task_by_model, llama_chat_reasoning_task_by_model, base_transcript_reasoning_task_by_model

    # Choose task based on baseline type
    if baseline_type == "escaped_transcript":
        model_tasks = conversation_task_by_model(data_dir=data_dir, limit=limit)
    elif baseline_type == "llama_chat":
        model_tasks = llama_chat_task_by_model(data_dir=data_dir, limit=limit)
    elif baseline_type == "llama_chat_reasoning":
        model_tasks = llama_chat_reasoning_task_by_model(data_dir=data_dir, limit=limit)
    elif baseline_type == "base_transcript_reasoning":
        model_tasks = base_transcript_reasoning_task_by_model(data_dir=data_dir, limit=limit)
    else:  # base_transcript or fallback
        model_tasks = baseline_task_by_model(data_dir=data_dir, limit=limit)

    print(f"\nFound {len(model_tasks)} models to evaluate:")
    for model_key in model_tasks.keys():
        print(f"  - {model_key}")

    # Create output directories
    results_dir = f"baseline/results/{baseline_type}"
    transcripts_dir = f"baseline/transcripts/{baseline_type}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)

    # Run evaluation for each model
    all_model_metrics = {}
    
    for model_key, task in model_tasks.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_key}")
        print(f"{'='*60}")
        
        # Create model-specific log directory
        model_log_dir = f"{log_dir}/{model_key.replace('/', '_').replace(':', '_')}"
        
        # Run evaluation for this model
        log = eval(task, model=model_name, log_dir=model_log_dir)
        results = log[0].samples if log and log[0].samples else []
        
        print(f"Processed {len(results)} samples for {model_key}")
        
        if len(results) == 0:
            print(f"No results for {model_key}, skipping...")
            continue
        
        # Save prompts for this model
        model_safe_name = model_key.replace('/', '_').replace(':', '_')
        prompts_file = save_prompts_to_file(results, f"{model_safe_name}_{baseline_type}", transcripts_dir, baseline_type)
        print(f"📝 Saved prompts to: {prompts_file}")
        
        # Analyze results for this model
        correct_predictions = 0
        by_prediction = {"A": 0, "B": 0, "OTHER": 0}
        by_target = {"A": 0, "B": 0}
        misclassified_examples = []
        filtered_samples = []
        parseable_samples = []
        
        for i, sample in enumerate(results):
            target = sample.target
            prediction = sample.output.completion.strip()
            original_did_lie = sample.metadata.get('did_lie', 'N/A')
            
            # Extract A/B from prediction
            if baseline_type == "escaped_transcript":
                scorer_name = 'conversation_scorer'
            elif baseline_type == "llama_chat":
                scorer_name = 'llama_chat_scorer'
            elif baseline_type == "llama_chat_reasoning":
                scorer_name = 'llama_chat_reasoning_scorer'
            elif baseline_type == "base_transcript_reasoning":
                scorer_name = 'base_transcript_reasoning_scorer'
            else:
                scorer_name = 'truth_scorer'
            if hasattr(sample, 'scores') and sample.scores and scorer_name in sample.scores:
                extracted_pred = sample.scores[scorer_name].answer
                metadata = sample.scores[scorer_name].metadata
                is_parseable = metadata.get('parseable', True) if metadata else False
            else:
                extracted_pred = "OTHER"
                is_parseable = False
                
            target_str = str(target) if target else "B"
            if target_str in by_target:
                by_target[target_str] += 1
            if extracted_pred in by_prediction:
                by_prediction[extracted_pred] += 1
            
            if not is_parseable or extracted_pred == "OTHER":
                filtered_samples.append({
                    'sample_num': i+1,
                    'target': target_str,
                    'raw_output': prediction[:100] + "..." if len(prediction) > 100 else prediction,
                    'did_lie': original_did_lie
                })
            else:
                parseable_samples.append(sample)
                if target_str == extracted_pred:
                    correct_predictions += 1
                else:
                    if len(misclassified_examples) < 5:
                        misclassified_examples.append({
                            'sample_num': i+1,
                            'target': target_str,
                            'prediction': extracted_pred,
                            'raw_output': prediction[:100] + "..." if len(prediction) > 100 else prediction,
                            'did_lie': original_did_lie
                        })

        print(f"\n=== {model_key.upper()} FILTERING SUMMARY ===")
        print(f"Total samples: {len(results)}")
        print(f"Filtered out (unparseable): {len(filtered_samples)}")
        print(f"Parseable samples: {len(parseable_samples)}")
        
        # Calculate metrics for this model
        total_tp = 0
        total_tn = 0 
        total_fp = 0
        total_fn = 0
        
        for r in results:
            if hasattr(r, 'scores') and r.scores and scorer_name in r.scores:
                metadata = r.scores[scorer_name].metadata
                if metadata and metadata.get('parseable', True):
                    total_tp += metadata.get('tp', 0)
                    total_tn += metadata.get('tn', 0)
                    total_fp += metadata.get('fp', 0)
                    total_fn += metadata.get('fn', 0)
        
        total_parseable = total_tp + total_tn + total_fp + total_fn
        if total_parseable > 0:
            accuracy = (total_tp + total_tn) / total_parseable
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            accuracy = precision = recall = f1 = 0
        
        print(f"\n=== {model_key.upper()} FINAL METRICS ===")
        print(f"Parseable samples: {total_parseable}/{len(results)} ({total_parseable/len(results):.1%})")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"True Positives: {total_tp}")
        print(f"True Negatives: {total_tn}")
        print(f"False Positives: {total_fp}")
        print(f"False Negatives: {total_fn}")
        
        # Store metrics for this model
        all_model_metrics[model_key] = {
            "parseable_samples": len(parseable_samples),
            "filtered_samples": len(filtered_samples),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tp": total_tp,
            "tn": total_tn,
            "fp": total_fp,
            "fn": total_fn,
            "target_distribution": f"A={by_target['A']}, B={by_target['B']}",
            "prediction_distribution": f"A={by_prediction['A']}, B={by_prediction['B']}, OTHER={by_prediction['OTHER']}"
        }
        
        # Save results for this model
        model_safe_name = model_key.replace('/', '_').replace(':', '_')
        json_filepath, txt_filepath = save_results_to_files(results, f"{model_safe_name}_{baseline_type}", all_model_metrics[model_key], results_dir, baseline_type)
        print(f"📝 Saved results to: {json_filepath}")
        print(f"📝 Saved summary to: {txt_filepath}")
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY BY MODEL")
    print(f"{'='*80}")
    for model_key, metrics in all_model_metrics.items():
        print(f"\n{model_key}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1_score']:.4f}")
        print(f"  Parseable: {metrics['parseable_samples']}")

if __name__ == "__main__":
    main() 