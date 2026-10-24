"""Metrics, and the threshold finder's class-imbalance bug."""

import numpy as np
import pytest

from src.utils.metrics import (AdaptiveThresholdFinder, compute_accuracy,
                               compute_auc, compute_confusion_matrix,
                               compute_f1_binary)


def test_accuracy():
    assert compute_accuracy(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])) == 0.75


def test_confusion_matrix_is_rows_true_cols_predicted():
    """Orientation matters: a transposed matrix looks plausible and makes every
    per-class recall wrong. The figures depend on this."""
    preds = np.array([0, 1, 1, 2])
    gts = np.array([0, 0, 1, 2])
    cm = compute_confusion_matrix(preds, gts, n_classes=3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1, "true 0 -> predicted 0 once, predicted 1 once"
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1
    assert cm.sum() == len(gts)


def test_confusion_matrix_rows_sum_to_support():
    rng = np.random.default_rng(0)
    gts = rng.integers(0, 5, 200)
    preds = rng.integers(0, 5, 200)
    cm = compute_confusion_matrix(preds, gts, n_classes=5)
    for c in range(5):
        assert cm[c].sum() == (gts == c).sum()


def test_f1_binary_from_a_known_matrix():
    # rows = true, cols = predicted:  tn=50 fp=10 / fn=5 tp=35
    cm = np.array([[50, 10], [5, 35]])
    m = compute_f1_binary(cm)
    assert m["precision"] == pytest.approx(35 / 45)
    assert m["recall"] == pytest.approx(35 / 40)


def test_auc_is_one_for_perfect_separation():
    targets = np.array([0, 0, 1, 1])
    assert compute_auc(targets, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_auc_degenerate_inputs_do_not_raise():
    assert compute_auc(np.array([]), np.array([])) == 0.0
    assert compute_auc(np.ones(5), np.random.rand(5)) == 0.0   # one class only


# --------------------------------------------------------------------------- #
# REGRESSION: the finder assumed balanced classes
# --------------------------------------------------------------------------- #

def _true_f1(scores, targets, t):
    pred = scores >= t
    tp = int((pred & (targets == 1)).sum())
    fp = int((pred & (targets == 0)).sum())
    fn = int((~pred & (targets == 1)).sum())
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r)


def _imbalanced(pos_rate=0.086, n=40000, seed=0):
    """Shaped like the real instance-pair task: 8.6% positive."""
    rng = np.random.default_rng(seed)
    n_pos = int(n * pos_rate)
    scores = np.concatenate([rng.normal(0.75, 0.15, n_pos),
                             rng.normal(0.35, 0.20, n - n_pos)])
    targets = np.concatenate([np.ones(n_pos, int), np.zeros(n - n_pos, int)])
    return scores, targets


def test_threshold_finder_maximises_true_f1_under_imbalance():
    """The finder computed `precision = tpr / (tpr + fpr)`, which equals true
    precision ONLY when positives and negatives are equally numerous. On the
    real 8.6%-positive instance task it overstated precision by +0.13 to +0.18
    and chose thresholds far too low -- its own score rose while the true F1
    fell, and plain argmax beat every threshold it picked.
    """
    scores, targets = _imbalanced()
    grid = np.linspace(scores.min(), scores.max(), 1500)
    best = max(_true_f1(scores, targets, t) for t in grid)

    found = AdaptiveThresholdFinder().find_optimal_threshold(scores, targets)
    got = _true_f1(scores, targets, found)
    assert best - got < 0.01, (
        f"threshold finder is {best - got:.3f} F1 below the optimum "
        f"(picked {found:.4f} -> {got:.4f}, best {best:.4f})")


def test_threshold_finder_beats_the_naive_half():
    scores, targets = _imbalanced()
    found = AdaptiveThresholdFinder().find_optimal_threshold(scores, targets)
    assert _true_f1(scores, targets, found) >= _true_f1(scores, targets, 0.5)


def test_threshold_finder_degenerate_inputs():
    f = AdaptiveThresholdFinder()
    assert f.find_optimal_threshold(np.array([]), np.array([])) == 0.5
    assert f.find_optimal_threshold(np.random.rand(5), np.ones(5, int)) == 0.5


# --------------------------------------------------------------------------- #
# mean IoU -- the metric AAGNet and SFRGNN report
# --------------------------------------------------------------------------- #

from src.utils.metrics import compute_mean_iou


def test_miou_perfect_prediction_is_one():
    cm = np.array([[10, 0], [0, 5]])
    miou, per = compute_mean_iou(cm)
    assert miou == pytest.approx(1.0)
    assert per == {0: pytest.approx(1.0), 1: pytest.approx(1.0)}


def test_miou_matches_the_definition():
    """rows = true, cols = predicted:  tn=50 fp=10 / fn=5 tp=35"""
    cm = np.array([[50, 10], [5, 35]])
    miou, per = compute_mean_iou(cm)
    assert per[1] == pytest.approx(35 / (35 + 10 + 5))      # tp/(tp+fp+fn)
    assert per[0] == pytest.approx(50 / (50 + 5 + 10))
    assert miou == pytest.approx((per[0] + per[1]) / 2)


def test_miou_is_never_above_f1_on_the_same_matrix():
    """IoU and F1 divide by the same errors, but F1 counts TP twice, so
    IoU <= F1 always. If this ever flips, one of the two is being computed from
    a different confusion matrix -- which is exactly the kind of mix-up that
    would make our numbers look better than they are."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        cm = rng.integers(0, 400, size=(2, 2))
        if cm.sum() == 0 or cm[1].sum() == 0:
            continue
        _, per = compute_mean_iou(cm)
        f1 = compute_f1_binary(cm)["f1"]
        assert per[1] <= f1 + 1e-12


def test_miou_ignores_classes_with_no_ground_truth_by_default():
    """Matches how the evaluation scripts compute macro-F1, so the two averages
    cover the same classes and can be read side by side."""
    cm = np.array([[8, 0, 0],
                   [0, 6, 0],
                   [0, 0, 0]])          # class 2 never appears in the truth
    miou, per = compute_mean_iou(cm)
    assert set(per) == {0, 1}
    assert miou == pytest.approx(1.0)

    miou_all, per_all = compute_mean_iou(cm, ignore_empty=False)
    assert set(per_all) == {0, 1, 2}
    assert miou_all == pytest.approx(2 / 3)


def test_miou_degenerate_inputs():
    assert compute_mean_iou(np.zeros((0, 0))) == (0.0, {})
    assert compute_mean_iou(np.zeros((3, 3)))[0] == 0.0


def test_miou_reproduces_the_measured_bottom_face_value():
    """Pinned against the number computed by hand from the committed run:
    TN 26,496 FP 198 FN 218 TP 4,883 -> IoU 0.9845 / 0.9215, mIoU 0.9530."""
    cm = np.array([[26496, 198], [218, 4883]])
    miou, per = compute_mean_iou(cm)
    assert per[0] == pytest.approx(0.9845, abs=1e-4)
    assert per[1] == pytest.approx(0.9215, abs=1e-4)
    assert miou == pytest.approx(0.9530, abs=1e-4)
