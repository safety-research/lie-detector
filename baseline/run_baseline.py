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
    from baseline.task import baseline_task
except ImportError:
    try:
        from .task import baseline_task
    except ImportError:
        from task import baseline_task

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

def save_prompts_to_file(results, filename):
    """Save all the prompts sent to the model to a text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = f"prompts_{timestamp}_{filename}.txt"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FULL PROMPTS SENT TO MODEL\n")
        f.write("="*80 + "\n\n")
        
        for i, sample in enumerate(results):
            target = sample.target
            prediction = sample.output.completion.strip()
            original_did_lie = sample.metadata.get('did_lie', 'N/A')
            
            # Get extracted prediction from scorer
            if hasattr(sample, 'scores') and sample.scores and 'truth_scorer' in sample.scores:
                extracted_pred = sample.scores['truth_scorer'].answer
                metadata = sample.scores['truth_scorer'].metadata
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
            f.write(sample.input)
            f.write("\n" + "〰" * 40 + "\n")
            
            f.write("MODEL RESPONSE:\n")
            f.write("▶ " + prediction + "\n")
            f.write("\n" + "="*80 + "\n\n")
    
    return filepath

def save_results_to_files(results, filename, metrics):
    """Save evaluation results to both JSON and TXT files."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Prepare structured data for JSON
    json_data = {
        "metadata": {
            "timestamp": timestamp,
            "total_samples": len(results),
            "parseable_samples": metrics["parseable_samples"],
            "filtered_samples": metrics["filtered_samples"],
            "model": "openrouter/meta-llama/llama-3.1-8b-instruct"
        },
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "true_positives": metrics["tp"],
            "true_negatives": metrics["tn"],
            "false_positives": metrics["fp"],
            "false_negatives": metrics["fn"]
        },
        "distribution": {
            "target_distribution": metrics["target_distribution"],
            "prediction_distribution": metrics["prediction_distribution"]
        },
        "sample_results": []
    }
    
    # Add individual sample results
    for i, sample in enumerate(results):
        target = sample.target
        prediction = sample.output.completion.strip()
        original_did_lie = sample.metadata.get('did_lie', 'N/A')
        
        if hasattr(sample, 'scores') and sample.scores and 'truth_scorer' in sample.scores:
            extracted_pred = sample.scores['truth_scorer'].answer
            metadata = sample.scores['truth_scorer'].metadata
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
        json_data["sample_results"].append(sample_data)
    
    # Save JSON file
    json_filepath = f"results_{timestamp}_{filename}.json"
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # Save TXT summary file
    txt_filepath = f"results_{timestamp}_{filename}.txt"
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LIE DETECTION BASELINE EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: openrouter/meta-llama/llama-3.1-8b-instruct\n")
        f.write(f"Total Samples: {len(results)}\n\n")
        
        f.write("FILTERING SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Parseable samples: {metrics['parseable_samples']}/{len(results)} ({metrics['parseable_samples']/len(results):.1%})\n")
        f.write(f"Filtered out: {metrics['filtered_samples']}/{len(results)} ({metrics['filtered_samples']/len(results):.1%})\n\n")
        
        f.write("PERFORMANCE METRICS (PARSEABLE SAMPLES ONLY)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        f.write(f"True Positives: {metrics['tp']}\n")
        f.write(f"True Negatives: {metrics['tn']}\n")
        f.write(f"False Positives: {metrics['fp']}\n")
        f.write(f"False Negatives: {metrics['fn']}\n\n")
        
        f.write("DISTRIBUTION ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Target Distribution: {metrics['target_distribution']}\n")
        f.write(f"Prediction Distribution: {metrics['prediction_distribution']}\n\n")
        
        f.write("SAMPLE MISCLASSIFICATIONS (FIRST 5)\n")
        f.write("-" * 40 + "\n")
        misclass_count = 0
        for i, sample in enumerate(results):
            if misclass_count >= 5:
                break
                
            target = sample.target
            if hasattr(sample, 'scores') and sample.scores and 'truth_scorer' in sample.scores:
                extracted_pred = sample.scores['truth_scorer'].answer
                metadata = sample.scores['truth_scorer'].metadata
                is_parseable = metadata.get('parseable', True) if metadata else False
            else:
                extracted_pred = "OTHER"
                is_parseable = False
            
            if is_parseable and target != extracted_pred:
                prediction = sample.output.completion.strip()
                original_did_lie = sample.metadata.get('did_lie', 'N/A')
                
                f.write(f"Sample {i+1}: Target={target}, Got={extracted_pred}, did_lie={original_did_lie}\n")
                f.write(f"  Raw output: {prediction[:100]}{'...' if len(prediction) > 100 else ''}\n")
                misclass_count += 1
        
        f.write("\n" + "="*80 + "\n")
    
    return json_filepath, txt_filepath

def main():
    # First examine the data distribution
    examine_data_distribution()
    
    # Then run evaluation
    print("\n" + "="*80)
    print("=== RUNNING BASELINE EVALUATION ===")
    
    # Hardcoded configuration
    data_dir = "processed_data"
    model_name = "openrouter/meta-llama/llama-3.1-8b-instruct"
    log_dir = "logs/baseline"
    limit = None  # Run on entire dataset with clean formatting

    print(f"Running baseline evaluation...")
    print(f"Model: {model_name}")
    print(f"Samples: {'All available' if limit is None else limit}")

    task = baseline_task(
        data_dir=data_dir,
        limit=limit
    )

    log = eval(task, model=model_name, log_dir=log_dir)
    results = log[0].samples if log and log[0].samples else []

    print(f"\nProcessed {len(results)} samples")
    
    # Save all prompts to file
    prompts_file = save_prompts_to_file(results, "baseline")
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
        if hasattr(sample, 'scores') and sample.scores and 'truth_scorer' in sample.scores:
            extracted_pred = sample.scores['truth_scorer'].answer
            metadata = sample.scores['truth_scorer'].metadata
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
        if hasattr(r, 'scores') and r.scores and 'truth_scorer' in r.scores:
            metadata = r.scores['truth_scorer'].metadata
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
    json_filepath, txt_filepath = save_results_to_files(results, "baseline", metrics)
    print(f"\n📝 Saved results to: {json_filepath}")
    print(f"📝 Saved summary to: {txt_filepath}")

if __name__ == "__main__":
    main() 