"""
Evaluation metrics for semantic segmentation and instance segmentation.

Provides:
- compute_accuracy: simple accuracy from predictions vs ground truth
- compute_confusion_matrix: multi-class confusion matrix
- compute_auc: ROC-AUC score with safe fallback
- compute_f1_binary: precision / recall / F1 from a binary confusion matrix
- AdaptiveThresholdFinder: finds the threshold that maximises F1 on a
  validation set using the ROC curve
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve


# ------------------------------------------------------------------ #
#  Accuracy
# ------------------------------------------------------------------ #

def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute element-wise accuracy.

    Args:
        predictions: 1-D array of predicted class indices.
        targets: 1-D array of ground-truth class indices (same length).

    Returns:
        Fraction of correct predictions in ``[0, 1]``.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if len(predictions) == 0:
        return 0.0
    return float((predictions == targets).mean())


# ------------------------------------------------------------------ #
#  Confusion matrix
# ------------------------------------------------------------------ #

def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Build an ``(n_classes, n_classes)`` confusion matrix.

    Rows correspond to ground-truth labels, columns to predictions.

    Args:
        predictions: 1-D integer array of predicted class indices.
        targets: 1-D integer array of ground-truth class indices.
        n_classes: Total number of classes (determines matrix size).

    Returns:
        Integer confusion matrix of shape ``(n_classes, n_classes)``.
    """
    return _sk_confusion_matrix(
        targets, predictions, labels=list(range(n_classes))
    )


# ------------------------------------------------------------------ #
#  AUC
# ------------------------------------------------------------------ #

def compute_auc(
    targets: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Compute ROC-AUC score with a safe fallback.

    Returns 0.0 when the inputs are empty or only one class is present
    (in which case ``roc_auc_score`` would raise).

    Args:
        targets: 1-D binary ground-truth labels (0 / 1).
        scores: 1-D predicted scores or probabilities.

    Returns:
        AUC value in ``[0, 1]``, or 0.0 on degenerate inputs.
    """
    targets = np.asarray(targets)
    scores = np.asarray(scores)
    if len(targets) == 0 or len(np.unique(targets)) < 2:
        return 0.0
    return float(roc_auc_score(targets, scores))


# ------------------------------------------------------------------ #
#  Binary precision / recall / F1
# ------------------------------------------------------------------ #

def compute_f1_binary(cm: np.ndarray) -> dict[str, float]:
    """Derive precision, recall, and F1 from a 2x2 confusion matrix.

    Args:
        cm: Confusion matrix of shape ``(2, 2)`` where
            ``cm[i, j]`` = count of samples with true label *i*
            predicted as *j*.

    Returns:
        Dict with keys ``"precision"``, ``"recall"``, ``"f1"``.
        All values are 0.0 when the denominator would be zero.
    """
    if cm.size != 4:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


# ------------------------------------------------------------------ #
#  Adaptive threshold finder
# ------------------------------------------------------------------ #

class AdaptiveThresholdFinder:
    """Find the decision threshold that maximises F1 score.

    Typical usage::

        finder = AdaptiveThresholdFinder()
        best_t = finder.find_optimal_threshold(similarities, targets)
        preds = (similarities >= best_t).astype(int)
    """

    def __init__(self):
        self.optimal_threshold: float = 0.5

    def find_optimal_threshold(
        self,
        similarities: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Sweep thresholds along the ROC curve and pick the best F1.

        Args:
            similarities: 1-D array of predicted similarity scores.
            targets: 1-D binary array of ground-truth labels (0 or 1).

        Returns:
            The threshold value that maximises the approximate F1 score.
        """
        similarities = np.asarray(similarities)
        targets = np.asarray(targets)

        if len(targets) == 0 or len(np.unique(targets)) < 2:
            return self.optimal_threshold

        fpr, tpr, thresholds = roc_curve(targets, similarities)

        # Recover the actual TP / FP counts from the rates using the
        # real class sizes, so precision is exact at every threshold.
        n_pos = float(targets.sum())
        n_neg = float(len(targets) - n_pos)
        tp = tpr * n_pos
        fp = fpr * n_neg
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tpr
        f1_scores = 2 * (precision * recall) / np.maximum(precision + recall, 1e-12)

        optimal_idx = int(np.argmax(f1_scores))
        self.optimal_threshold = float(thresholds[optimal_idx])

        return self.optimal_threshold


# ------------------------------------------------------------------ #
#  Mean Intersection over Union
# ------------------------------------------------------------------ #

def compute_mean_iou(
    cm: np.ndarray,
    ignore_empty: bool = True,
) -> tuple[float, dict[int, float]]:
    """Mean IoU from a confusion matrix.

    This is the metric the MFInstSeg literature reports for the semantic and
    bottom-face heads -- AAGNet and SFRGNN both give accuracy + mIoU for those
    two, and accuracy + F1 for instance segmentation. Reporting macro-F1 alone
    leaves our numbers with nothing to be compared against.

    IoU is stricter than F1 on the same confusion matrix: both divide by the
    same errors, but F1 counts true positives twice.

        IoU_k = TP_k / (TP_k + FP_k + FN_k)
        F1_k  = 2 TP_k / (2 TP_k + FP_k + FN_k)

    so IoU <= F1 always, with equality only at a perfect class. A bottom-face
    F1 of 0.959 is an IoU of 0.921 on the identical predictions -- the gap is
    arithmetic, not a different result.

    Args:
        cm: ``(K, K)`` confusion matrix, **rows = ground truth, columns =
            predictions** (the layout ``compute_confusion_matrix`` returns).
        ignore_empty: Average only over classes that have ground-truth support.
            This matches how ``_per_class_metrics`` computes macro-F1 in the
            evaluation scripts, so the two averages cover the same classes and
            can be read side by side. With ``False``, absent classes contribute
            an IoU of 0 and drag the mean down.

    Returns:
        ``(mean_iou, {class_index: iou})``.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.size == 0:
        return 0.0, {}

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp          # predicted as k, truly something else
    fn = cm.sum(axis=1) - tp          # truly k, predicted as something else
    denom = tp + fp + fn

    iou = np.divide(tp, denom, out=np.zeros_like(tp, dtype=np.float64),
                    where=denom > 0)

    keep = (cm.sum(axis=1) > 0) if ignore_empty else np.ones(len(iou), bool)
    if not keep.any():
        return 0.0, {}

    per_class = {int(k): float(iou[k]) for k in np.flatnonzero(keep)}
    return float(iou[keep].mean()), per_class
