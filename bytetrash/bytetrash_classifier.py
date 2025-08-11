#!/usr/bin/env python3
"""
Bytetrash Classifier - Training and Evaluation Pipeline

This module provides a clean interface for training and evaluating SVM classifiers
on bytetrash character patterns for lie detection.
"""

import json
import argparse
import os
import gzip
import zlib
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
from collections import Counter, defaultdict
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')


def extract_raw_byte_features(chars: str, max_length: int = 128) -> np.ndarray:
    """
    Convert character string to raw byte values.

    Args:
        chars: Random character string
        max_length: Maximum number of bytes to use (default 128)

    Returns:
        Numpy array of byte values
    """
    if not chars:
        return np.zeros(max_length)

    # Convert characters to byte values
    byte_values = [ord(c) for c in chars[:max_length]]

    # Pad with zeros if needed
    if len(byte_values) < max_length:
        byte_values.extend([0] * (max_length - len(byte_values)))

    return np.array(byte_values, dtype=np.float32)


def extract_character_features(chars: str) -> Dict[str, float]:
    """Extract engineered features from random character string for classification."""
    if not chars:
        return {}

    # Your original feature extraction code remains the same
    total_len = len(chars)
    if total_len == 0:
        return {}

    features = {}

    # Character type ratios
    features['alpha_ratio'] = sum(1 for c in chars if c.isalpha()) / total_len
    features['digit_ratio'] = sum(1 for c in chars if c.isdigit()) / total_len
    features['special_ratio'] = sum(1 for c in chars if not c.isalnum()) / total_len
    features['upper_ratio'] = sum(1 for c in chars if c.isupper()) / total_len
    features['lower_ratio'] = sum(1 for c in chars if c.islower()) / total_len

    # Length features
    features['length'] = total_len

    # Entropy/randomness measures
    char_counts = {}
    for c in chars:
        char_counts[c] = char_counts.get(c, 0) + 1

    # Shannon entropy
    entropy = 0
    for count in char_counts.values():
        prob = count / total_len
        if prob > 0:
            entropy -= prob * (__import__('math').log2(prob))
    features['entropy'] = entropy

    # Unique character ratio
    features['unique_ratio'] = len(char_counts) / total_len

    # Hash-based features
    hash_sum = sum(ord(c) for c in chars)
    features['hash_sum_mod'] = hash_sum % 1000
    features['hash_checksum'] = hash_sum % 2

    # Pattern-based features
    features['has_repeats'] = len(chars) != len(set(chars))
    features['max_repeat'] = max(char_counts.values()) if char_counts else 0

    return features


def extract_advanced_features(chars: str) -> Dict[str, float]:
    """
    Extract advanced features from character string based on requested feature types.
    
    Args:
        chars: Random character string
        
    Returns:
        Dictionary of advanced features (with all features present, defaulting to 0.0)
    """
    if not chars:
        return {}
    
    features = {}
    char_list = list(chars)
    byte_values = [ord(c) for c in chars]
    
    # 1. UNIGRAM FREQUENCY FEATURES
    # Digit frequencies (0-9)
    digit_counts = Counter(c for c in chars if c.isdigit())
    for digit in '0123456789':
        features[f'digit_{digit}_freq'] = digit_counts.get(digit, 0) / len(chars)
    
    # Character type frequencies
    char_counter = Counter(chars)
    features['unique_chars'] = len(char_counter)
    features['most_common_freq'] = char_counter.most_common(1)[0][1] / len(chars) if char_counter else 0
    
    # 2. BIGRAM/TRIGRAM FREQUENCY FEATURES
    # Always include these features, defaulting to 0
    features['unique_bigrams'] = 0
    features['most_common_bigram_freq'] = 0
    features['digit_bigram_count'] = 0
    features['unique_trigrams'] = 0
    features['most_common_trigram_freq'] = 0
    
    if len(chars) >= 2:
        bigrams = [chars[i:i+2] for i in range(len(chars)-1)]
        bigram_counter = Counter(bigrams)
        features['unique_bigrams'] = len(bigram_counter)
        features['most_common_bigram_freq'] = bigram_counter.most_common(1)[0][1] / len(bigrams) if bigrams else 0
        
        # Common digit transitions
        digit_bigrams = [bg for bg in bigrams if bg[0].isdigit() and bg[1].isdigit()]
        features['digit_bigram_count'] = len(digit_bigrams) / len(bigrams) if bigrams else 0
    
    if len(chars) >= 3:
        trigrams = [chars[i:i+3] for i in range(len(chars)-2)]
        trigram_counter = Counter(trigrams)
        features['unique_trigrams'] = len(trigram_counter)
        features['most_common_trigram_freq'] = trigram_counter.most_common(1)[0][1] / len(trigrams) if trigrams else 0
    
    # 3. ENTROPY FEATURES
    # Shannon entropy (already implemented in original)
    char_probs = np.array(list(char_counter.values())) / len(chars)
    features['shannon_entropy'] = -np.sum(char_probs * np.log2(char_probs + 1e-10))
    
    # Byte-wise entropy
    byte_counter = Counter(byte_values)
    byte_probs = np.array(list(byte_counter.values())) / len(byte_values)
    features['byte_entropy'] = -np.sum(byte_probs * np.log2(byte_probs + 1e-10))
    
    # 4. RUN LENGTH FEATURES
    # Default values for run length features
    features['max_run_length'] = 1
    features['min_run_length'] = 1
    features['mean_run_length'] = 1.0
    features['std_run_length'] = 0.0
    features['num_runs'] = len(chars)
    
    # Consecutive repeated characters
    runs = []
    current_char = chars[0] if chars else ''
    current_run = 1
    
    for i in range(1, len(chars)):
        if chars[i] == current_char:
            current_run += 1
        else:
            runs.append(current_run)
            current_char = chars[i]
            current_run = 1
    runs.append(current_run)
    
    if runs:
        features['max_run_length'] = max(runs)
        features['min_run_length'] = min(runs)
        features['mean_run_length'] = np.mean(runs)
        features['std_run_length'] = np.std(runs)
        features['num_runs'] = len(runs)
    
    # 5. DIGIT TRANSITION FEATURES (Markov-like)
    # Default values
    features['most_common_digit_transition_freq'] = 0
    features['unique_digit_transitions'] = 0
    
    if len(chars) >= 2:
        digit_transitions = defaultdict(int)
        total_digit_transitions = 0
        
        for i in range(len(chars)-1):
            if chars[i].isdigit() and chars[i+1].isdigit():
                transition = f"{chars[i]}->{chars[i+1]}"
                digit_transitions[transition] += 1
                total_digit_transitions += 1
        
        # Most common digit transition
        if total_digit_transitions > 0:
            most_common_transition = max(digit_transitions.items(), key=lambda x: x[1])
            features['most_common_digit_transition_freq'] = most_common_transition[1] / total_digit_transitions
            features['unique_digit_transitions'] = len(digit_transitions)
    
    # 6. COMPRESSION RATIO FEATURES
    # Gzip compression ratio
    original_bytes = chars.encode('utf-8')
    try:
        compressed_gzip = gzip.compress(original_bytes)
        features['gzip_compression_ratio'] = len(compressed_gzip) / len(original_bytes)
    except:
        features['gzip_compression_ratio'] = 1.0
    
    # Zlib compression ratio
    try:
        compressed_zlib = zlib.compress(original_bytes)
        features['zlib_compression_ratio'] = len(compressed_zlib) / len(original_bytes)
    except:
        features['zlib_compression_ratio'] = 1.0
    
    # 7. REPETITION RATE
    total_chars = len(chars)
    unique_chars = len(set(chars))
    features['repetition_rate'] = 1 - (unique_chars / total_chars) if total_chars > 0 else 0
    
    # 8. ALTERNATION BIAS (high-low digit patterns)
    # Default values
    features['high_low_alternation_rate'] = 0
    features['ascending_digit_rate'] = 0
    features['descending_digit_rate'] = 0
    
    if any(c.isdigit() for c in chars):

        try:
            digits = [int(c) for c in chars if c.isdigit()]
        except ValueError:
            digits = []
            for c in chars:
                if c.isdigit():
                    try:
                        digits.append(int(c))
                    except ValueError:
                        pass
        if len(digits) >= 2:
            alternations = 0
            for i in range(len(digits)-1):
                # High-low alternation (>= 5 is high, < 5 is low)
                if (digits[i] >= 5 and digits[i+1] < 5) or (digits[i] < 5 and digits[i+1] >= 5):
                    alternations += 1
            features['high_low_alternation_rate'] = alternations / (len(digits) - 1)
            
            # Sequential patterns
            ascending = sum(1 for i in range(len(digits)-1) if digits[i+1] == digits[i] + 1)
            descending = sum(1 for i in range(len(digits)-1) if digits[i+1] == digits[i] - 1)
            features['ascending_digit_rate'] = ascending / (len(digits) - 1) if len(digits) > 1 else 0
            features['descending_digit_rate'] = descending / (len(digits) - 1) if len(digits) > 1 else 0
    
    # 9. LEXICOGRAPHIC FEATURES
    # Sortedness measure
    sorted_chars = sorted(chars)
    features['sortedness'] = sum(1 for i, c in enumerate(chars) if i < len(sorted_chars) and c == sorted_chars[i]) / len(chars)
    
    # Character deltas - default values
    features['max_delta'] = 0
    features['min_delta'] = 0
    features['mean_absolute_delta'] = 0.0
    features['std_delta'] = 0.0
    
    if len(byte_values) >= 2:
        deltas = [byte_values[i+1] - byte_values[i] for i in range(len(byte_values)-1)]
        features['max_delta'] = max(deltas)
        features['min_delta'] = min(deltas)
        features['mean_absolute_delta'] = np.mean(np.abs(deltas))
        features['std_delta'] = np.std(deltas)
    
    # 10. MODULO PATTERNS
    # Parity patterns (even/odd)
    even_count = sum(1 for val in byte_values if val % 2 == 0)
    features['even_byte_ratio'] = even_count / len(byte_values)
    
    # Modulo patterns for various N
    for mod_n in [3, 5, 7, 10]:
        mod_counts = Counter(val % mod_n for val in byte_values)
        # Entropy of modulo distribution
        mod_probs = np.array(list(mod_counts.values())) / len(byte_values)
        features[f'mod_{mod_n}_entropy'] = -np.sum(mod_probs * np.log2(mod_probs + 1e-10))
        
        # Most common modulo class
        features[f'mod_{mod_n}_max_freq'] = max(mod_counts.values()) / len(byte_values)
    
    return features


def prepare_training_data(results_file: str, feature_type: str = 'raw') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from existing bytetrash results.

    Args:
        results_file: Path to JSON file with bytetrash results
        feature_type: 'raw' for raw bytes, 'engineered' for basic features, 'advanced' for comprehensive features

    Returns:
        Tuple of (X, y) where X is features matrix and y is labels
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    X = []
    y = []

    for result in results:
        chars = result.get('bytetrash_chars', '')
        did_lie = result.get('did_lie', False)

        if chars:  # Only include samples with characters
            if feature_type == 'raw':
                # Use raw byte values
                feature_vector = extract_raw_byte_features(chars)
                X.append(feature_vector)
                y.append(1 if did_lie else 0)
            elif feature_type == 'advanced':
                # Use advanced features
                features = extract_advanced_features(chars)
                if features:
                    # Convert feature dict to ordered list, ensuring consistent feature ordering
                    # First, get a sample feature vector to determine feature names
                    if not X:  # First sample - determine feature names
                        feature_names = list(features.keys())
                    
                    # Extract features in consistent order, filling missing with 0
                    feature_vector = []
                    for name in (feature_names if X else list(features.keys())):
                        feature_vector.append(features.get(name, 0.0))
                    
                    X.append(feature_vector)
                    y.append(1 if did_lie else 0)
            else:
                # Use basic engineered features
                features = extract_character_features(chars)
                if features:
                    feature_vector = [
                        features.get('alpha_ratio', 0),
                        features.get('digit_ratio', 0),
                        features.get('special_ratio', 0),
                        features.get('upper_ratio', 0),
                        features.get('lower_ratio', 0),
                        features.get('length', 0),
                        features.get('entropy', 0),
                        features.get('unique_ratio', 0),
                        features.get('hash_sum_mod', 0),
                        features.get('hash_checksum', 0),
                        1.0 if features.get('has_repeats', False) else 0.0,
                        features.get('max_repeat', 0)
                    ]
                    X.append(feature_vector)
                    y.append(1 if did_lie else 0)

    return np.array(X), np.array(y)


def train_svm_classifier(results_file: str,
                         test_size: float = 0.2,
                         random_state: int = 42,
                         feature_type: str = 'raw',
                         optimize_hyperparams: bool = True) -> Dict:
    """
    Train SVM classifier on bytetrash character features.

    Args:
        results_file: Path to JSON file with training data
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        feature_type: 'raw' for raw bytes, 'engineered' for feature engineering
        optimize_hyperparams: Whether to use grid search for hyperparameter optimization

    Returns:
        Dictionary with training results, model, and scaler
    """
    print(f"Loading training data from {results_file}")
    print(f"Using feature type: {feature_type}")
    X, y = prepare_training_data(results_file, feature_type)

    if len(X) == 0:
        raise ValueError("No training data found")

    print(f"Training data: {len(X)} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y)} lies, {len(y) - np.sum(y)} truths")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features (crucial for SVM performance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if optimize_hyperparams:
        # Grid search for optimal hyperparameters
        print("\nPerforming hyperparameter optimization...")
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear', 'poly']
        }

        grid_search = GridSearchCV(
            SVC(probability=True, random_state=random_state),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_scaled, y_train)
        model = grid_search.best_estimator_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    else:
        # Train SVM with default RBF kernel
        model = SVC(kernel='rbf', probability=True, random_state=random_state)
        model.fit(X_train_scaled, y_train)

    # Feature names for reference
    if feature_type == 'raw':
        feature_names = [f'byte_{i}' for i in range(X.shape[1])]
    else:
        feature_names = [
            'alpha_ratio', 'digit_ratio', 'special_ratio', 'upper_ratio', 'lower_ratio',
            'length', 'entropy', 'unique_ratio', 'hash_sum_mod', 'hash_checksum',
            'has_repeats', 'max_repeat'
        ]

    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    # Detailed classification report
    report = classification_report(y_test, y_pred, target_names=['Truth', 'Lie'])

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    results = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'feature_type': feature_type,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'classification_report': report,
        'confusion_matrix': cm,
        'training_size': len(X_train),
        'test_size': len(X_test),
        'best_params': grid_search.best_params_ if optimize_hyperparams else None
    }

    print(f"\nSVM Training Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"\nClassification Report:")
    print(report)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Feature importance for linear kernel
    if model.kernel == 'linear' and feature_type == 'raw':
        print("\nAnalyzing byte position importance...")
        coef = model.coef_[0]

        # Top positions that indicate lies
        top_lie_positions = np.argsort(coef)[-10:]
        print("\nTop byte positions associated with LIES:")
        for pos in top_lie_positions:
            print(f"  Position {pos}: weight = {coef[pos]:.4f}")

        # Top positions that indicate truths
        top_truth_positions = np.argsort(coef)[:10]
        print("\nTop byte positions associated with TRUTHS:")
        for pos in top_truth_positions:
            print(f"  Position {pos}: weight = {coef[pos]:.4f}")

    return results


def save_model(model_data: Dict, model_path: str):
    """Save trained SVM model and scaler to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    save_data = {
        'model': model_data['model'],
        'scaler': model_data['scaler'],
        'feature_names': model_data['feature_names'],
        'feature_type': model_data.get('feature_type', 'engineered'),
        'best_params': model_data.get('best_params')
    }

    with open(model_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"SVM model saved to {model_path}")


def load_model(model_path: str) -> Dict:
    """Load trained SVM model and scaler from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print(f"SVM model loaded from {model_path}")
    return model_data


def classify_chars(chars: str, model_data: Dict) -> Tuple[bool, float]:
    """
    Classify character string using trained SVM model.

    Args:
        chars: Random character string
        model_data: Dictionary containing model, scaler, and feature names

    Returns:
        Tuple of (prediction, confidence) where prediction is True for 'lie', False for 'truth'
    """
    feature_type = model_data.get('feature_type', 'engineered')

    if feature_type == 'raw':
        # Use raw byte features
        feature_vector = extract_raw_byte_features(chars)
        feature_vector = feature_vector.reshape(1, -1)
    elif feature_type == 'advanced':
        # Use advanced features
        features = extract_advanced_features(chars)
        if not features:
            return False, 0.0
        feature_vector = np.array([list(features.values())])
    else:
        # Use basic engineered features
        features = extract_character_features(chars)
        if not features:
            return False, 0.0

        feature_vector = np.array([[
            features.get('alpha_ratio', 0),
            features.get('digit_ratio', 0),
            features.get('special_ratio', 0),
            features.get('upper_ratio', 0),
            features.get('lower_ratio', 0),
            features.get('length', 0),
            features.get('entropy', 0),
            features.get('unique_ratio', 0),
            features.get('hash_sum_mod', 0),
            features.get('hash_checksum', 0),
            1.0 if features.get('has_repeats', False) else 0.0,
            features.get('max_repeat', 0)
        ]])

    # Scale and predict
    feature_vector_scaled = model_data['scaler'].transform(feature_vector)
    prediction = model_data['model'].predict(feature_vector_scaled)[0]

    # Get confidence (probability)
    proba = model_data['model'].predict_proba(feature_vector_scaled)[0]
    confidence = proba[prediction]

    return bool(prediction), float(confidence)


def evaluate_model(results_file: str, model_path: str):
    """
    Evaluate trained model on a results file.

    Args:
        results_file: Path to JSON file with bytetrash results
        model_path: Path to saved model file
    """
    # Load model
    model_data = load_model(model_path)

    # Load test data
    with open(results_file, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    correct = 0
    total = 0
    predictions = []
    actuals = []

    print(f"\nEvaluating model on {len(results)} samples from {results_file}")
    print("\nSample predictions:")

    # Evaluate all results
    for i, result in enumerate(results):
        chars = result.get('bytetrash_chars', '')
        if not chars:
            continue

        actual_lie = result.get('did_lie', False)
        predicted_lie, confidence = classify_chars(chars, model_data)
        is_correct = predicted_lie == actual_lie

        predictions.append(1 if predicted_lie else 0)
        actuals.append(1 if actual_lie else 0)
        total += 1

        if is_correct:
            correct += 1

        if i < 10:  # Show first 10
            print(f"Sample {i}: '{chars[:30]}...' -> "
                  f"Predicted: {'lie' if predicted_lie else 'truth'} ({confidence:.2f}), "
                  f"Actual: {'lie' if actual_lie else 'truth'}, "
                  f"Correct: {is_correct}")

    if total == 0:
        print("No valid samples to evaluate")
        return

    accuracy = correct / total
    print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Show classification report
    if len(set(actuals)) > 1:  # Only if we have both classes
        report = classification_report(actuals, predictions, target_names=['Truth', 'Lie'])
        print("\nDetailed Classification Report:")
        print(report)

    # Show confusion matrix
    cm = confusion_matrix(actuals, predictions)
    print("\nConfusion Matrix:")
    print(cm)


def run_feature_experiments(results_file: str, output_dir: str = 'results/bytetrash/experiments'):
    """
    Run comprehensive experiments to compare different feature types and classifiers.
    
    Args:
        results_file: Path to JSON file with bytetrash results
        output_dir: Directory to save experiment results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature types to test
    feature_types = ['raw', 'engineered', 'advanced']
    
    # Classifiers to test
    classifiers = {
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42),
        'SVM_Linear': SVC(kernel='linear', probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    experiment_results = []
    
    print("Starting comprehensive feature experiments...")
    print(f"Data file: {results_file}")
    print(f"Feature types: {feature_types}")
    print(f"Classifiers: {list(classifiers.keys())}")
    print("="*60)
    
    for feature_type in feature_types:
        print(f"\n--- Testing feature type: {feature_type.upper()} ---")
        
        # Prepare data for this feature type
        try:
            X, y = prepare_training_data(results_file, feature_type)
            print(f"Features: {X.shape[1]}, Samples: {len(X)}")
            
            if len(X) == 0:
                print(f"No data for {feature_type}, skipping...")
                continue
                
        except Exception as e:
            print(f"Error preparing data for {feature_type}: {e}")
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for clf_name, clf in classifiers.items():
            print(f"\n  Testing {clf_name}...")
            
            try:
                # Train classifier
                clf.fit(X_train_scaled, y_train)
                
                # Test set predictions
                y_pred = clf.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
                
                # Classification report
                report = classification_report(y_test, y_pred, target_names=['Truth', 'Lie'], output_dict=True)
                
                # Store results
                result = {
                    'feature_type': feature_type,
                    'classifier': clf_name,
                    'num_features': X.shape[1],
                    'num_samples': len(X),
                    'test_accuracy': test_accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'precision_lie': report['Lie']['precision'],
                    'recall_lie': report['Lie']['recall'],
                    'f1_lie': report['Lie']['f1-score'],
                    'precision_truth': report['Truth']['precision'],
                    'recall_truth': report['Truth']['recall'],
                    'f1_truth': report['Truth']['f1-score']
                }
                
                experiment_results.append(result)
                
                print(f"    Test Accuracy: {test_accuracy:.4f}")
                print(f"    CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"    Error with {clf_name}: {e}")
                continue
    
    # Save results
    results_df = []
    for result in experiment_results:
        results_df.append(result)
    
    # Sort by test accuracy
    results_df.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    # Save to JSON
    output_file = os.path.join(output_dir, 'feature_experiments.json')
    with open(output_file, 'w') as f:
        json.dump({
            'experiments': results_df,
            'summary': {
                'best_result': results_df[0] if results_df else None,
                'feature_types_tested': feature_types,
                'classifiers_tested': list(classifiers.keys())
            }
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*60}")
    
    if results_df:
        print(f"{'Rank':<4} {'Feature Type':<12} {'Classifier':<18} {'Features':<8} {'Test Acc':<8} {'CV Acc':<12}")
        print("-" * 70)
        
        for i, result in enumerate(results_df[:10]):  # Top 10
            print(f"{i+1:<4} {result['feature_type']:<12} {result['classifier']:<18} "
                  f"{result['num_features']:<8} {result['test_accuracy']:<8.4f} "
                  f"{result['cv_mean']:.4f}±{result['cv_std']:.3f}")
        
        print(f"\nBest combination: {results_df[0]['feature_type']} + {results_df[0]['classifier']}")
        print(f"Best accuracy: {results_df[0]['test_accuracy']:.4f}")
        
        # Save best model
        best_result = results_df[0]
        print(f"\nTraining and saving best model...")
        
        best_model_data = train_svm_classifier(
            results_file,
            feature_type=best_result['feature_type'],
            optimize_hyperparams=False  # Skip optimization for demo
        )
        
        best_model_path = os.path.join(output_dir, 'best_model.pkl')
        save_model(best_model_data, best_model_path)
        print(f"Best model saved to: {best_model_path}")
    
    print(f"\nComplete results saved to: {output_file}")
    return results_df


def main():
    """Main entrypoint for bytetrash classifier training and evaluation."""
    parser = argparse.ArgumentParser(description='Bytetrash Classifier - Train and evaluate SVM on character patterns')
    parser.add_argument('command', choices=['train', 'evaluate', 'predict', 'experiment'],
                        help='Command to run')
    parser.add_argument('--data', required=True,
                        help='Path to bytetrash results JSON file')
    parser.add_argument('--model', default='results/bytetrash/svm_model.pkl',
                        help='Path to save/load model file')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for testing (training only)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--text',
                        help='Text to classify (predict command only)')
    parser.add_argument('--feature-type', choices=['raw', 'engineered', 'advanced'], default='raw',
                        help='Use raw bytes, basic engineered features, or advanced features')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip hyperparameter optimization')
    parser.add_argument('--output-dir', default='results/bytetrash/experiments',
                        help='Output directory for experiment results')

    args = parser.parse_args()

    if args.command == 'train':
        print("Training SVM classifier on bytetrash data...")
        model_data = train_svm_classifier(
            args.data,
            test_size=args.test_size,
            random_state=args.random_state,
            feature_type=args.feature_type,
            optimize_hyperparams=not args.no_optimize
        )
        save_model(model_data, args.model)
        print(f"\nTraining complete! Model saved to {args.model}")

    elif args.command == 'evaluate':
        print("Evaluating trained model...")
        evaluate_model(args.data, args.model)

    elif args.command == 'predict':
        if not args.text:
            print("Error: --text argument required for predict command")
            return

        print(f"Loading model and predicting on: '{args.text}'")
        model_data = load_model(args.model)
        prediction, confidence = classify_chars(args.text, model_data)
        print(f"Prediction: {'lie' if prediction else 'truth'} (confidence: {confidence:.2f})")
        
    elif args.command == 'experiment':
        print("Running comprehensive feature experiments...")
        run_feature_experiments(args.data, args.output_dir)


if __name__ == "__main__":
    main()