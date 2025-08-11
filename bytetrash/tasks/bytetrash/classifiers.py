from collections import Counter

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from inspect_ai.dataset import MemoryDataset, Sample


class BytetrashClassifier:
    """Base classifier for bytetrash lie detection"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def extract_features(self, samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from samples"""
        X = []
        y = []

        for sample in samples:
            # Get the logit features
            logit_data = sample.metadata.get('logit_data', {})

            if logit_data:
                # Extract digit logits (main features)
                digit_logits = sample.metadata.get('logit_data', {})
                digit_logits_full = sample.metadata.get('logit_data_full', {})

                # Create feature vector
                features = []

                # 1. Raw digit logits
                if digit_logits_full:
                    for d in digit_logits_full:
                        features.extend(d)
                elif digit_logits:
                    features.extend(digit_logits)

                # 2. Statistical features from digit logits
                # if digit_logits:
                #     features.extend([
                #         logit_data.get('mean_digit_logprob', 0),
                #         logit_data.get('std_digit_logprob', 0),
                #         logit_data.get('min_digit_logprob', 0),
                #         logit_data.get('max_digit_logprob', 0),
                #     ])
                #
                #     # 3. Additional derived features
                #     digit_array = np.array(digit_logits)
                #
                #     # Entropy of the digit distribution
                #     probs = np.exp(digit_array)
                #     probs = probs / np.sum(probs)
                #     entropy = -np.sum(probs * np.log(probs + 1e-10))
                #     features.append(entropy)
                #
                #     # Range (max - min)
                #     features.append(digit_array.max() - digit_array.min())
                #
                #     # Skewness
                #     mean = digit_array.mean()
                #     std = digit_array.std()
                #     if std > 0:
                #         skewness = np.mean(((digit_array - mean) / std) ** 3)
                #         features.append(skewness)
                #     else:
                #         features.append(0)
                #
                #     # Kurtosis
                #     if std > 0:
                #         kurtosis = np.mean(((digit_array - mean) / std) ** 4) - 3
                #         features.append(kurtosis)
                #     else:
                #         features.append(0)
                #
                # # Get label
                label = sample.metadata.get('did_lie', 0)

                if features:  # Only add if we have features
                    X.append(features)
                    y.append(int(label))

        # Store feature names for later reference
        if X and not self.feature_names:
            n_digits = len(X[0])  # Subtract the 8 derived features
            self.feature_names = [f'digit_{i}' for i in range(n_digits)]
            # self.feature_names.extend([
            #     'mean_logprob', 'std_logprob', 'min_logprob', 'max_logprob',
            #     'entropy', 'range', 'skewness', 'kurtosis'
            # ])
        else:
            pass

        # Clean rows with incorrect shape
        X_clean = []
        y_clean = []

        # Find the modal length
        lengths = [len(x) for x in X]
        length_counts = Counter(lengths)
        modal_length = length_counts.most_common(1)[0][0]

        for i, x in enumerate(X):  # Note: i is index, x is the feature vector
            if len(x) == modal_length:  # Keep only rows with exactly 160 features
                X_clean.append(x)
                y_clean.append(y[i])
            else:
                print(f"Dropping sample {i} with length {len(x)}")

        print(f"Kept {len(X_clean)} out of {len(X)} samples")

        try:
            return np.array(X_clean), np.array(y_clean)
        except ValueError as e:
            print(f"Error after cleaning: {e}")
            # If still failing, debug further
            lengths = [len(x) for x in X_clean]
            print(f"Lengths after cleaning: {set(lengths)}")
            raise

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the classifier"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }

        return metrics


class LogisticRegressionClassifier(BytetrashClassifier):
    """Logistic Regression with L2 regularization"""

    def __init__(self, C=1.0):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(C=C, max_iter=1000, random_state=42)


class ShallowMLPClassifier(BytetrashClassifier):
    """Shallow MLP (2 layers)"""

    def __init__(self, hidden_size=64):
        super().__init__("Shallow MLP")
        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden_size, hidden_size),
            activation='relu',
            max_iter=1000,
            random_state=42
        )


class RandomForestClassifierWrapper(BytetrashClassifier):
    """Random Forest classifier"""

    def __init__(self, n_estimators=100):
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )


class XGBoostClassifier(BytetrashClassifier):
    """XGBoost classifier"""

    def __init__(self):
        super().__init__("XGBoost")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )


class SVMClassifier(BytetrashClassifier):
    """Support Vector Machine classifier"""

    def __init__(self):
        super().__init__("SVM")
        self.model = SVC(kernel='rbf', probability=True, random_state=42)


def run_classifier_comparison(dataset: MemoryDataset, test_size: float = 0.2):
    """Run and compare multiple classifiers on the dataset"""

    # Initialize classifiers
    classifiers = [
        LogisticRegressionClassifier(C=1.0),
        LogisticRegressionClassifier(C=0.1),  # More regularization
        ShallowMLPClassifier(hidden_size=32),
        ShallowMLPClassifier(hidden_size=64),
        RandomForestClassifierWrapper(n_estimators=100),
        XGBoostClassifier(),
        SVMClassifier()
    ]

    # Extract features from the first classifier (they all use the same extraction)
    X, y = classifiers[0].extract_features(dataset.samples)

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Results storage
    results = []

    # Train and evaluate each classifier
    for clf in classifiers:
        print(f"\n{'=' * 50}")
        print(f"Training {clf.model_name}...")

        # Train
        clf.train(X_train, y_train)

        # Evaluate
        metrics = clf.evaluate(X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(
            clf.model,
            clf.scaler.transform(X),
            y,
            cv=StratifiedKFold(n_splits=5),
            scoring='roc_auc'
        )
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()

        # Store results
        results.append({
            'classifier': clf.model_name,
            **metrics
        })

        # Print results
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"CV ROC-AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

        # Print classification report for the best performing model
        if clf.model_name == "Logistic Regression":
            y_pred = clf.predict(X_test)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred,
                                        target_names=['Truth', 'Lie']))

    return pd.DataFrame(results), classifiers, (X_train, X_test, y_train, y_test)


def plot_results(results_df: pd.DataFrame):
    """Plot classifier comparison results"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Bar chart of metrics
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results_subset = results_df[['classifier'] + metrics_to_plot].set_index('classifier')

    results_subset.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Classifier Performance Comparison')
    axes[0].set_ylabel('Score')
    axes[0].legend(loc='lower right')
    axes[0].set_ylim(0, 1.1)

    # Plot 2: ROC-AUC with error bars
    axes[1].bar(results_df['classifier'], results_df['cv_auc_mean'],
                yerr=results_df['cv_auc_std'], capsize=5)
    axes[1].set_title('Cross-Validation ROC-AUC Scores')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].set_xticklabels(results_df['classifier'], rotation=45, ha='right')
    axes[1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()


def analyze_feature_importance(classifiers: List[BytetrashClassifier],
                               X_train: np.ndarray):
    """Analyze feature importance for applicable classifiers"""

    for clf in classifiers:
        if isinstance(clf, RandomForestClassifierWrapper):
            # Get feature importances
            importances = clf.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Plot
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importances - {clf.model_name}")
            plt.bar(range(len(importances)), importances[indices])

            plt.xticks(range(len(importances)),
                       [clf.feature_names[i] if clf.feature_names else i for i in indices],
                       rotation=90)
            plt.tight_layout()
            plt.show()

            # Print top features
            print(f"\nTop 10 features for {clf.model_name}:")
            for i in range(min(10, len(indices))):
                try:
                    print(f"{clf.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
                except IndexError:
                    pass


def plot_roc_curves(classifiers: List[BytetrashClassifier],
                    X_test: np.ndarray, y_test: np.ndarray):
    """Plot ROC curves for all classifiers"""

    plt.figure(figsize=(10, 8))

    for clf in classifiers:
        y_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.plot(fpr, tpr, label=f'{clf.model_name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Bytetrash Lie Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()


# Example usage function
def main(dataset: MemoryDataset):
    """Main function to run the classifier comparison"""

    print("Running Bytetrash Classifier Comparison...")

    # Run comparison
    results_df, classifiers, (X_train, X_test, y_train, y_test) = \
        run_classifier_comparison(dataset)

    # Display results
    print("\n" + "=" * 50)
    print("Summary Results:")
    print(results_df.to_string())

    # Plot results
    plot_results(results_df)

    # Analyze feature importance
    analyze_feature_importance(classifiers, X_train)

    # Plot ROC curves
    plot_roc_curves(classifiers, X_test, y_test)

    return results_df, classifiers