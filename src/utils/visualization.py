"""
Plotting helpers for training diagnostics and evaluation.

All functions accept an optional *save_path*; when provided the figure is
saved to disk (and ``plt.show()`` is skipped so that headless runs work).

Provides:
- plot_training_curves: loss and accuracy / LR over epochs
- plot_confusion_matrix: annotated heatmap of a confusion matrix
- plot_roc_curve: ROC curve with optional optimal-threshold marker
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib

# Select a file-only backend before pyplot is imported.
#
# Every caller in this package passes `save_path` and writes a PNG; nothing here
# displays a window.
#
# MPLBACKEND is honoured when set, so a notebook that wants an interactive
# backend can still ask for one.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np


# ------------------------------------------------------------------ #
#  Training curves
# ------------------------------------------------------------------ #

def plot_training_curves(
    history: dict[str, list],
    save_path: str | Path | None = None,
) -> None:
    """Plot training curves from a history dict.

    The *history* dict should map metric names to lists of per-epoch
    values.  At minimum ``"loss"`` is expected; an optional second panel
    is drawn for ``"acc"`` (accuracy) or ``"lr"`` (learning rate).

    Args:
        history: Dict with at least ``"loss"`` key.  Optional keys:
            ``"acc"``, ``"lr"``, ``"val_loss"``, ``"val_acc"``.
        save_path: If given, save the figure to this path instead of
            calling ``plt.show()``.
    """
    has_acc = "acc" in history
    has_lr = "lr" in history
    has_val_loss = "val_loss" in history
    has_val_acc = "val_acc" in history

    n_panels = 1 + int(has_acc or has_lr)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # Panel 1 -- loss
    ax = axes[0]
    ax.plot(history["loss"], linewidth=2, label="Train loss")
    if has_val_loss:
        ax.plot(history["val_loss"], linewidth=2, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2 -- accuracy or learning rate
    if has_acc:
        ax = axes[1]
        ax.plot(history["acc"], linewidth=2, color="green", label="Train acc")
        if has_val_acc:
            ax.plot(history["val_acc"], linewidth=2, color="blue", label="Val acc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
    elif has_lr:
        ax = axes[1]
        ax.plot(history["lr"], linewidth=2, color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------ #
#  Confusion matrix
# ------------------------------------------------------------------ #

def plot_confusion_matrix(
    cm: np.ndarray,
    n_classes: int | None = None,
    class_names: Sequence[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> None:
    """Visualise a confusion matrix as an annotated heatmap.

    Args:
        cm: Integer confusion matrix of shape ``(C, C)``.
        n_classes: Number of classes (inferred from *cm* if *None*).
        class_names: Optional tick labels for each class.
        title: Figure title.
        save_path: If given, save the figure to this path.
    """
    if n_classes is None:
        n_classes = cm.shape[0]

    tick_labels = (
        class_names if class_names is not None else list(range(n_classes))
    )

    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.6), max(6, n_classes * 0.5)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            colour = "white" if cm[i, j] > thresh else "black"
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center", color=colour, fontsize=8,
            )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground Truth Label")
    ax.set_title(title)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(tick_labels, fontsize=8)
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------ #
#  ROC curve
# ------------------------------------------------------------------ #

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float | None = None,
    optimal_threshold: float | None = None,
    thresholds: np.ndarray | None = None,
    title: str = "ROC Curve",
    save_path: str | Path | None = None,
) -> None:
    """Plot an ROC curve with an optional optimal-threshold marker.

    Args:
        fpr: False positive rates (from ``sklearn.metrics.roc_curve``).
        tpr: True positive rates.
        auc: Area under the curve (shown in the legend when provided).
        optimal_threshold: If given together with *thresholds*, a red dot
            is placed at the operating point closest to this threshold.
        thresholds: Threshold array from ``roc_curve`` (needed to locate
            *optimal_threshold* on the curve).
        title: Figure title.
        save_path: If given, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    label = "ROC"
    if auc is not None:
        label += f" (AUC={auc:.3f})"
    ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    if optimal_threshold is not None and thresholds is not None:
        idx = int(np.argmin(np.abs(thresholds - optimal_threshold)))
        ax.scatter(
            [fpr[idx]], [tpr[idx]],
            color="red", s=100, zorder=5,
            label=f"Optimal (t={optimal_threshold:.2f})",
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------ #
#  Internal helper
# ------------------------------------------------------------------ #

def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    """Save figure to *save_path* or show interactively."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
