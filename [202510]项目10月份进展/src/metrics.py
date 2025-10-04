#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Metrics for Bridge VIV Risk Assessment
Comprehensive metrics for regression and classification tasks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.metrics import make_scorer

logger = logging.getLogger(__name__)

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate regression predictions

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with regression metrics
    """
    metrics = {}

    # Basic metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = root_mean_squared_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)

    # MAPE (if possible)
    try:
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    except:
        metrics['mape'] = np.nan

    # Additional metrics
    residuals = y_true - y_pred
    metrics['mean_residual'] = np.mean(residuals)
    metrics['std_residual'] = np.std(residuals)
    metrics['max_error'] = np.max(np.abs(residuals))

    # Explained variance
    metrics['explained_variance'] = 1 - (np.var(residuals) / np.var(y_true))

    return metrics

def evaluate_classification(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_proba: Optional[np.ndarray] = None,
                          average: str = 'macro') -> Dict[str, float]:
    """
    Evaluate classification predictions

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        average: Averaging strategy for multi-class

    Returns:
        Dictionary with classification metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Micro averages (if multiclass)
    if len(np.unique(y_true)) > 2:
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

    # ROC AUC and PR AUC (if probabilities available)
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if y_proba.ndim == 2:
                    y_proba_pos = y_proba[:, 1]
                else:
                    y_proba_pos = y_proba

                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
                metrics['pr_auc'] = average_precision_score(y_true, y_proba_pos)
            else:
                # Multiclass classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, average=average, multi_class='ovr')
        except Exception as e:
            logger.warning(f"Failed to calculate AUC metrics: {e}")

    return metrics

def evaluate_model(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  task_type: str,
                  y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate model predictions based on task type

    Args:
        y_true: True values/labels
        y_pred: Predicted values/labels
        task_type: Type of task ('regression', 'binary_classification', 'multiclass_classification')
        y_proba: Predicted probabilities (for classification)

    Returns:
        Dictionary with appropriate metrics
    """
    if task_type == 'regression':
        return evaluate_regression(y_true, y_pred)
    elif task_type in ['binary_classification', 'multiclass_classification']:
        return evaluate_classification(y_true, y_pred, y_proba)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def get_scorer(metric_name: str, task_type: str):
    """
    Get sklearn scorer for a given metric

    Args:
        metric_name: Name of the metric
        task_type: Type of task

    Returns:
        Sklearn scorer object
    """
    if task_type == 'regression':
        scorers = {
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'mse': make_scorer(mean_squared_error, greater_is_better=False),
            'rmse': make_scorer(root_mean_squared_error, greater_is_better=False),
            'r2': make_scorer(r2_score),
            'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
        }
    else:
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='macro', zero_division=0),
            'recall': make_scorer(recall_score, average='macro', zero_division=0),
            'f1': make_scorer(f1_score, average='macro', zero_division=0),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True, average='macro', multi_class='ovr')
        }

    if metric_name not in scorers:
        raise ValueError(f"Unknown metric: {metric_name} for task type: {task_type}")

    return scorers[metric_name]

def plot_regression_results(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          save_path: Optional[Path] = None,
                          title: str = "Regression Results") -> None:
    """
    Plot regression results

    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    # Predicted vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Predicted vs Actual')

    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals Plot')

    # Residuals histogram
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')

    # Q-Q plot for residuals normality
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Residuals Normality)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_classification_results(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None,
                              class_names: Optional[List[str]] = None,
                              save_path: Optional[Path] = None,
                              title: str = "Classification Results") -> None:
    """
    Plot classification results

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        class_names: Names of classes
        save_path: Path to save plot
        title: Plot title
    """
    n_classes = len(np.unique(y_true))
    is_binary = n_classes == 2

    if is_binary and y_proba is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    fig.suptitle(title, fontsize=16)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix')

    if class_names:
        axes[0].set_xticklabels(class_names)
        axes[0].set_yticklabels(class_names)

    # Classification metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics_text = f"Accuracy: {report['accuracy']:.3f}\n"
    metrics_text += f"Precision: {report['macro avg']['precision']:.3f}\n"
    metrics_text += f"Recall: {report['macro avg']['recall']:.3f}\n"
    metrics_text += f"F1-Score: {report['macro avg']['f1-score']:.3f}"

    axes[1].text(0.1, 0.5, metrics_text, transform=axes[1].transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    axes[1].set_title('Classification Metrics')

    # For binary classification with probabilities
    if is_binary and y_proba is not None and len(axes) >= 4:
        # Extract positive class probabilities
        if y_proba.ndim == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
        roc_auc = roc_auc_score(y_true, y_proba_pos)

        axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[2].set_xlim([0.0, 1.0])
        axes[2].set_ylim([0.0, 1.05])
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].set_title('ROC Curve')
        axes[2].legend(loc="lower right")

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
        pr_auc = average_precision_score(y_true, y_proba_pos)

        axes[3].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[3].set_xlim([0.0, 1.0])
        axes[3].set_ylim([0.0, 1.05])
        axes[3].set_xlabel('Recall')
        axes[3].set_ylabel('Precision')
        axes[3].set_title('Precision-Recall Curve')
        axes[3].legend(loc="lower left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_results(y_true: np.ndarray,
                y_pred: np.ndarray,
                task_type: str,
                y_proba: Optional[np.ndarray] = None,
                class_names: Optional[List[str]] = None,
                save_path: Optional[Path] = None,
                title: str = "Model Results") -> None:
    """
    Plot results based on task type

    Args:
        y_true: True values/labels
        y_pred: Predicted values/labels
        task_type: Type of task
        y_proba: Predicted probabilities (for classification)
        class_names: Names of classes (for classification)
        save_path: Path to save plot
        title: Plot title
    """
    if task_type == 'regression':
        plot_regression_results(y_true, y_pred, save_path, title)
    elif task_type in ['binary_classification', 'multiclass_classification']:
        plot_classification_results(y_true, y_pred, y_proba, class_names, save_path, title)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def analyze_errors(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  feature_names: Optional[List[str]] = None,
                  X: Optional[np.ndarray] = None,
                  top_n: int = 5) -> Dict[str, Any]:
    """
    Analyze prediction errors

    Args:
        y_true: True values
        y_pred: Predicted values
        feature_names: Names of features
        X: Feature matrix
        top_n: Number of top errors to analyze

    Returns:
        Dictionary with error analysis
    """
    errors = np.abs(y_true - y_pred)
    top_error_indices = np.argsort(errors)[-top_n:][::-1]

    analysis = {
        'top_errors': {
            'indices': top_error_indices.tolist(),
            'true_values': y_true[top_error_indices].tolist(),
            'predicted_values': y_pred[top_error_indices].tolist(),
            'absolute_errors': errors[top_error_indices].tolist()
        },
        'error_statistics': {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'median_error': np.median(errors)
        }
    }

    # Add feature information if available
    if X is not None and feature_names is not None:
        analysis['top_error_features'] = {}
        for i, idx in enumerate(top_error_indices):
            sample_features = {}
            for j, feature_name in enumerate(feature_names):
                sample_features[feature_name] = X[idx, j]
            analysis['top_error_features'][f'sample_{i+1}'] = sample_features

    return analysis

def compare_models(results: Dict[str, Dict[str, float]],
                  task_type: str,
                  save_path: Optional[Path] = None) -> None:
    """
    Compare multiple model results

    Args:
        results: Dictionary with model results
        task_type: Type of task
        save_path: Path to save plot
    """
    if not results:
        return

    # Determine primary metric
    if task_type == 'regression':
        primary_metric = 'rmse'
        ascending = True
    else:
        primary_metric = 'f1'
        ascending = False

    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Sort by primary metric
    df = df.sort_values(primary_metric, ascending=ascending)

    # Create plot
    if task_type == 'regression':
        metrics_to_plot = ['mae', 'rmse', 'r2']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            ax = axes[i] if len(metrics_to_plot) > 1 else axes
            bars = ax.bar(df['Model'], df[metric])
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel(metric.title())
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    """Test metrics functions"""
    # Generate sample data
    np.random.seed(42)

    # Regression test
    y_true_reg = np.random.randn(100) * 10 + 50
    y_pred_reg = y_true_reg + np.random.randn(100) * 2

    reg_metrics = evaluate_regression(y_true_reg, y_pred_reg)
    print("Regression metrics:", reg_metrics)

    # Classification test
    y_true_clf = np.random.randint(0, 2, 100)
    y_pred_clf = np.random.randint(0, 2, 100)
    y_proba_clf = np.random.rand(100, 2)
    y_proba_clf = y_proba_clf / y_proba_clf.sum(axis=1, keepdims=True)

    clf_metrics = evaluate_classification(y_true_clf, y_pred_clf, y_proba_clf)
    print("Classification metrics:", clf_metrics)

    # Test plotting
    plot_results(y_true_reg, y_pred_reg, 'regression', title='Test Regression')
    plot_results(y_true_clf, y_pred_clf, 'binary_classification',
                y_proba_clf, title='Test Classification')

if __name__ == "__main__":
    main()