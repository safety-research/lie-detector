import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI from launching
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
import pickle
from datetime import datetime

# Set HuggingFace cache directory to NVMe drive
os.environ['HF_HOME'] = '/mnt/nvme3/dipika/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme3/dipika/.cache/huggingface'

# Create cache directory if it doesn't exist
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

class ActivationExtractor:
    """Extract activations from a language model using hooks"""
    
    def __init__(self, model_name="unsloth/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.activations = {}
        self.hooks = []
        
    def load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        print("Model loaded successfully!")
        
    def register_hooks(self, layer_idx):
        """Register hooks to extract activations from a specific layer"""
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hook for the last layer
        layer_name = f"layers.{layer_idx}.mlp"
        if hasattr(self.model.model, 'layers'):
            layer = self.model.model.layers[layer_idx].mlp
            hook = layer.register_forward_hook(hook_fn(layer_name))
            self.hooks.append(hook)
            print(f"Registered hook for layer: {layer_name}")
        else:
            print("Warning: Could not find layers in model structure")
            
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def extract_last_token_activations(self, texts, max_length=512):
        """Extract activations for the last token of each text"""
        activations_list = []
        
        print("Extracting activations...")
        for i, text in enumerate(tqdm(texts, desc="Processing texts")):
            try:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )
                
                # Move to same device as model
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Get last token activation
                if self.activations:
                    # Get the activation from the registered layer
                    layer_name = list(self.activations.keys())[0]
                    activation = self.activations[layer_name]  # [batch_size, seq_len, hidden_dim]
                    
                    # Get the last token's activation
                    last_token_activation = activation[0, -1, :].cpu().numpy()
                    activations_list.append(last_token_activation)
                else:
                    # Fallback: use the last hidden state
                    last_hidden_state = outputs.hidden_states[-1] if outputs.hidden_states else outputs.logits
                    last_token_activation = last_hidden_state[0, -1, :].cpu().numpy()
                    activations_list.append(last_token_activation)
                    
            except Exception as e:
                print(f"Error processing text {i}: {e}")
                # Use zero activation as fallback
                if activations_list:
                    activations_list.append(np.zeros_like(activations_list[0]))
                else:
                    # Create a dummy activation if this is the first one
                    activations_list.append(np.zeros(4096))  # Typical hidden size
                    
        return np.array(activations_list)

def load_processed_data(file_path, random_seed=42):
    """Load the processed lie detection data and shuffle with random seed"""
    print(f"Loading data from: {file_path}")
    
    traces = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            traces.append(data['trace'])
            labels.append(data['label'])
    
    print(f"Loaded {len(traces)} examples")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Shuffle the data with random seed
    print(f"Shuffling dataset with random seed: {random_seed}")
    np.random.seed(random_seed)
    indices = np.random.permutation(len(traces))
    traces = [traces[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    print(f"✓ Dataset shuffled successfully")
    print(f"Label distribution after shuffling: {np.bincount(labels)}")
    
    return traces, np.array(labels)

def save_results_to_json(results, results_dir="lie_detection_probing_results"):
    """Save results to a JSON file, appending each result as a new line"""
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, "probing_results.jsonl")
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    
    # Save to JSONL format (one JSON object per line)
    with open(results_file, 'a', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
        f.write('\n')
    
    print(f"Results saved to: {results_file}")

def train_logistic_regression(X, y, test_size=0.2, random_state=42, model_id="unknown", layer_index=8):
    """Train a logistic regression classifier and return detailed metrics"""
    print("Training logistic regression classifier...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train classifier
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Get detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Count true and false labels
    true_labels = int(np.sum(y_test == 0))  # Truthful examples
    false_labels = int(np.sum(y_test == 1))  # Lying examples
    
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"Support: {support}")
    
    # Create results dictionary
    results = {
        "model_id": model_id,
        "layer_index": layer_index,
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "support": support.tolist(),
        "true_labels": true_labels,
        "false_labels": false_labels,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }
    
    # Save results
    save_results_to_json(results)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return clf, X_test, y_test, y_pred, results

def plot_results(y_test, y_pred, model_id="unknown", layer_index=8, save_path=None):
    """Plot confusion matrix and save results"""
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate filename with model_id and layer_index
    if save_path is None:
        model_name = model_id.replace("/", "_").replace("-", "_")
        save_path = f"confusion_matrix_{model_name}_layer_{layer_index}.png"
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Truthful', 'Lying'],
                yticklabels=['Truthful', 'Lying'])
    plt.title(f'Confusion Matrix - {model_id} Layer {layer_index}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Confusion matrix saved to: {save_path}")

def save_model(clf, model_id="unknown", layer_index=8, save_path=None):
    """Save the trained model"""
    # Generate filename with model_id and layer_index
    if save_path is None:
        model_name = model_id.replace("/", "_").replace("-", "_")
        save_path = f"lie_detector_model_{model_name}_layer_{layer_index}.pkl"
    
    with open(save_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to: {save_path}")

def main(MODEL_ID="UNSLOTH/META-LLAMA-3.1-8B-INSTRUCT", LAYER_INDEX=8):
    """Main function to run the lie detection training"""
    print("=" * 60)
    print("LIE DETECTION LOGISTIC REGRESSION TRAINING")
    print("=" * 60)
    
    # Configuration
    data_path = "/mnt/nvme3/dipika/lie-detector/processed_math_problems.jsonl"
    model_name = MODEL_ID  # Use the parameter
    layer_index = LAYER_INDEX  # Use the parameter
    random_seed = 42  # Random seed for reproducibility
    
    print(f"Configuration:")
    print(f"  Data path: {data_path}")
    print(f"  Model: {model_name}")
    print(f"  Layer index: {layer_index}")
    print(f"  Random seed: {random_seed}")
    
    # Load data with shuffling
    traces, labels = load_processed_data(data_path, random_seed=random_seed)
    
    # Initialize activation extractor
    extractor = ActivationExtractor(model_name)
    
    try:
        # Load model
        extractor.load_model()
        
        # Register hooks
        extractor.register_hooks(layer_index)
        
        # Extract activations
        print("Extracting last token activations...")
        activations = extractor.extract_last_token_activations(traces)
        
        print(f"Activation shape: {activations.shape}")
        print(f"Expected shape: ({len(traces)}, hidden_dim)")
        
        # Train logistic regression
        clf, X_test, y_test, y_pred, results = train_logistic_regression(
            activations, labels, 
            model_id=model_name, 
            layer_index=layer_index
        )
        
        # Plot results
        plot_results(y_test, y_pred, model_id=model_name, layer_index=layer_index)
        
        # Save model
        save_model(clf, model_id=model_name, layer_index=layer_index)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        extractor.remove_hooks()

if __name__ == "__main__":
    # You can call main() with different parameters:
    # main()  # Uses default values
    for i in range(0, 32):
        main("UNSLOTH/META-LLAMA-3.1-8B-INSTRUCT", i)  # Different layer
    # main("ANOTHER-MODEL-NAME", 24)  # Different model and layer
    # main() 