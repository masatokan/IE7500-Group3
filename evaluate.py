"""
evaluate.py
-----------
Evaluation utilities: metrics, plots, and ROUGE scoring for the
Sentiment Analysis + Summarization pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, matthews_corrcoef, f1_score
)
from rouge_score import rouge_scorer


# ─────────────────────────────────────────────
# Classification Metrics
# ─────────────────────────────────────────────

def print_metrics(y_true, y_pred, y_prob=None, model_name: str = "Model"):
    """
    Print full classification metrics for a model.

    Args:
        y_true:     Ground truth labels
        y_pred:     Predicted labels
        y_prob:     Predicted probabilities (for ROC-AUC)
        model_name: Name shown in the report header
    """
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC Score: {auc:.4f}")

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1-Score: {macro_f1:.4f}")


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model", save_path: str = None):
    """
    Plot and optionally save a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(y_true, y_prob, model_name: str = "Model", save_path: str = None):
    """
    Plot ROC curve for a binary classifier.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {model_name}')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_learning_curve(train_losses, val_losses, model_name: str = "Model", save_path: str = None):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses:   List of validation losses per epoch
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title(f'Learning Curve — {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────
# ROUGE Scoring for Summarization
# ─────────────────────────────────────────────

def compute_rouge(predictions: list, references: list) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        predictions: List of generated summary strings
        references:  List of reference summary strings

    Returns:
        Dict with average ROUGE-1, ROUGE-2, ROUGE-L F1 scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores['rouge1'].fmeasure)
        r2.append(scores['rouge2'].fmeasure)
        rl.append(scores['rougeL'].fmeasure)

    results = {
        'ROUGE-1': np.mean(r1),
        'ROUGE-2': np.mean(r2),
        'ROUGE-L': np.mean(rl),
    }
    print("\nROUGE Scores:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results


# ─────────────────────────────────────────────
# Model Comparison
# ─────────────────────────────────────────────

def plot_model_comparison(results: dict, save_path: str = None):
    """
    Bar chart comparing F1 scores across models.

    Args:
        results: Dict of {model_name: f1_score}
    """
    models = list(results.keys())
    scores = list(results.values())
    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, scores, color=['#4C72B0', '#DD8452', '#55A868'])
    plt.ylim(0.7, 1.0)
    plt.ylabel('Macro F1-Score')
    plt.title('Model Comparison — Macro F1-Score')
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f'{score:.3f}', ha='center', fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
