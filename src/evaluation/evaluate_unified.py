"""
Evaluate a trained unified model on a held-out test split.

Scores all three heads of one checkpoint on the test partition of the split
file:

  * semantic  -- prototype classification against the LabelEncoder:
                 overall accuracy, per-class P/R/F1, macro, confusion matrix
  * bottom    -- prototype classification against the BottomLabelEncoder:
                 accuracy, binary P/R/F1, confusion matrix
  * instance  -- pairwise cosine similarity of face embeddings:
                 ROC-AUC, plus P/R/F1 at a decision threshold

THRESHOLD PROVENANCE.  The instance head has no argmax; it needs a decision
threshold, and where that threshold came from decides whether the F1 is
reportable. This script uses ``optimal_threshold`` from the checkpoint. That
threshold is fitted on the VALIDATION partition:
``train_unified.py`` builds ``val_loader`` from the split file's ``val`` ids
(``train_unified.py:260-263``), accumulates the validation instance
similarities each epoch, and calls ``find_optimal_threshold`` on them
(``train_unified.py:499-505``).  The value stored with the best-validation-loss
checkpoint is the one read here.  Tuned on val, applied to test: reportable,
and no test label has been seen at any point.

A second threshold re-tuned on the test scores themselves is also printed, but
only as an ORACLE UPPER BOUND.  Do not report the oracle row; it has seen the
test labels.  The gap between the two rows measures how much a test-tuned
threshold would buy over the val-tuned one -- a generalisation gap, not a leak.

Usage:
    python -m src.evaluation.evaluate_unified \
        --config configs/mfinstseg_unified.yaml \
        --checkpoint results/checkpoints/mfinstseg_unified.pth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, random_split

from ..data.brep_dataset import PreExtractedBRepDataset, load_split
from ..data.collate import collate_fn_brep
from ..models.encoders import BottomLabelEncoder, BRepEncoder, LabelEncoder
from ..utils.checkpoint import load_checkpoint
from ..utils.provenance import provenance_from_checkpoint
from ..utils.metrics import (
    AdaptiveThresholdFinder,
    compute_accuracy,
    compute_auc,
    compute_confusion_matrix,
    compute_f1_binary,
    compute_mean_iou,
)
from ..utils.visualization import plot_confusion_matrix, plot_roc_curve


def load_config(path):
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


#  What the banner line is allowed to claim
#
# This helper is a duplicate of `evaluate_semantic._input_settings_phrase`.
# The two are pinned together by `tests/test_gatefix_evaluators.py`; they are
# duplicated rather than imported so that this module does not pull in the
# ResNet and UV-grid imports it otherwise never touches.


def _input_settings_phrase(cfg):
    """Describe the attribute/bias settings this run actually used."""
    ds_cfg = cfg.get("dataset", {}) or {}
    stated_attr = ds_cfg.get("attributes")
    stated_bias = (cfg.get("bias") or {}).get("type")

    attr = (f"attributes={stated_attr}" if stated_attr is not None
            else "attributes=scaled [DATASET DEFAULT -- dataset.attributes is "
                 "absent from this config]")
    bias = (f"bias={stated_bias}" if stated_bias is not None
            else "bias=similarity [DATASET DEFAULT -- bias.type is absent "
                 "from this config]")
    return f"{attr}, {bias}"


def _setting_source(stated):
    """``"config"`` when the config states a value, otherwise why it did not."""
    return "config" if stated is not None else (
        "dataset default (the config states no value; see the identical "
        "fallback in _build_test_set's ds_kwargs, which is what the dataset "
        "was actually built with)")


#  What the per-class keys mean
#
# `per_class[c]["accuracy"]` is TP/(TP+FN), i.e. recall, and equals the
# `recall` key beside it to full float precision.

PER_CLASS_KEY_SEMANTICS = {
    "accuracy": (
        "per-class accuracy == RECALL == TP/(TP+FN): the fraction of this "
        "class's ground-truth faces predicted as this class. It is NOT "
        "(TP+TN)/N. Equal to the 'recall' key beside it by construction, to "
        "full float precision. Kept under this name because the archived "
        "metrics files carry it; 'recall' is the correct name for the same "
        "number."),
    "precision": "TP/(TP+FP).",
    "recall": "TP/(TP+FN). The correctly-named twin of 'accuracy' above.",
    "f1": "harmonic mean of precision and recall.",
    "support": "ground-truth faces of this class in the test split.",
}


#  Model + dataset construction

def _build_models(cfg, ckpt, device):
    """Rebuild the BRepEncoder and both label encoders from a checkpoint."""
    m_cfg = cfg.get("model", {})
    model = BRepEncoder(
        d_model=m_cfg.get("d_model", 128),
        n_heads=m_cfg.get("n_heads", 8),
        n_layers=m_cfg.get("n_layers", 4),
        dim_feedforward=m_cfg.get("dim_feedforward", 256),
        dropout=m_cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_classes = ckpt.get("n_classes", cfg["dataset"].get("n_classes", 25))
    embedding_dim = cfg.get("label_encoder", {}).get(
        "embedding_dim", m_cfg.get("d_model", 128))

    if "label_encoder" not in ckpt:
        raise RuntimeError(
            "Checkpoint has no 'label_encoder' -- the semantic prototypes "
            "cannot be reconstructed. Is this a unified checkpoint?")
    label_encoder = LabelEncoder(
        n_classes=n_classes, embedding_dim=embedding_dim).to(device)
    label_encoder.load_state_dict(ckpt["label_encoder"])
    label_encoder.eval()

    if "bottom_encoder" not in ckpt:
        raise RuntimeError(
            "Checkpoint has no 'bottom_encoder' -- this looks like a "
            "semantic-only checkpoint. Use evaluate_semantic.py instead.")
    bottom_encoder = BottomLabelEncoder(embedding_dim=embedding_dim).to(device)
    bottom_encoder.load_state_dict(ckpt["bottom_encoder"])
    bottom_encoder.eval()

    return model, label_encoder, bottom_encoder, n_classes


def _build_test_set(cfg, seed):
    """Return ``(test_dataset, description)``.  Prefers the saved split file."""
    ds_cfg = cfg["dataset"]
    bias_cfg = cfg.get("bias", {})
    # Read before ds_kwargs so it can be forwarded into the dataset.
    split_file = ds_cfg.get("split_file")
    ds_kwargs = dict(
        json_dir=ds_cfg["json_dir"],
        n_classes=ds_cfg.get("n_classes", 25),
        selected_classes=ds_cfg.get("selected_classes"),
        attributes=ds_cfg.get("attributes", "scaled"),
        norm_stats=ds_cfg.get("norm_stats"),
        # Forwarded so the norm-stats guard can compare these statistics
        # with the split in use. Falsy on the random_split path, where
        # checks 2 and 3 are skipped.
        split_file=split_file,
        bias_type=bias_cfg.get("type", "similarity"),
        lambda_adj=bias_cfg.get("lambda_adj", 1.0),
        lambda_sim=bias_cfg.get("lambda_sim", 1.0),
        one_hot_type=ds_cfg.get("one_hot_type", False),
    )

    if split_file:
        # Load ONLY the test ids -- never rebuild the split with random_split.
        test_ds = PreExtractedBRepDataset(
            model_ids=load_split(split_file, "test"), **ds_kwargs)
        return test_ds, f"split file {split_file} (test partition)"

    print("WARNING: the config has no dataset.split_file. Falling back to "
          "random_split, which reproduces the training split only if --seed "
          "matches the seed used to train. Prefer a split file.")
    dataset = PreExtractedBRepDataset(**ds_kwargs)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_ratio = ds_cfg.get("train_split", 0.7)
    val_ratio = ds_cfg.get("val_split", 0.2)
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    if test_size <= 0:
        return dataset, "full dataset (no held-out split configured)"
    if val_size > 0:
        _, _, test_ds = random_split(
            dataset, [train_size, val_size, test_size], generator=generator)
    else:
        _, test_ds = random_split(
            dataset, [train_size, test_size], generator=generator)
    return test_ds, f"random_split(seed={seed})"


#  Forward pass

def _run_test(model, label_encoder, bottom_encoder, loader, n_classes, device):
    """One pass over the test set, collecting all three heads' predictions.

    Returns:
        Dict of numpy arrays: semantic preds/gts, bottom preds/gts, and the
        off-diagonal instance pair similarities with their targets.
    """
    sem_preds, sem_gts, bot_preds, bot_gts = [], [], [], []
    inst_sims, inst_targets = [], []

    with torch.no_grad():
        sem_proto = F.normalize(
            label_encoder(torch.arange(n_classes, device=device)), dim=1)
        bot_proto = F.normalize(
            bottom_encoder(torch.arange(2, device=device)), dim=1)

        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            embeddings = model(
                batch["faces_uv"], batch["faces_attr"],
                batch["edges_attr"], batch["bias"],
                face_mask=batch["face_mask"],
                edge_mask=batch["edge_mask"],
            )

            for b in range(embeddings.shape[0]):
                mask = batch["face_mask"][b]
                if mask.sum() == 0:
                    continue
                valid_idx = torch.nonzero(mask).squeeze(1)
                emb = F.normalize(embeddings[b][valid_idx], dim=1)

                # Nearest prototype -- the same decision rule training used,
                # so these numbers are comparable to the epoch printouts.
                sem_preds.append(
                    torch.matmul(emb, sem_proto.T).argmax(1).cpu().numpy())
                sem_gts.append(batch["semantic_labels"][b][mask].cpu().numpy())
                bot_preds.append(
                    torch.matmul(emb, bot_proto.T).argmax(1).cpu().numpy())
                bot_gts.append(batch["bottom_labels"][b][mask].cpu().numpy())

                # Instance head: off-diagonal pairwise cosine similarity.
                if len(valid_idx) < 2:
                    continue
                sim = torch.matmul(emb, emb.T)
                tgt = batch["instance_labels"][b][valid_idx[:, None], valid_idx]
                off = ~torch.eye(len(valid_idx), device=device, dtype=torch.bool)
                inst_sims.append(sim[off].cpu().numpy().astype(np.float32))
                inst_targets.append(tgt[off].cpu().numpy().astype(np.int8))

    def cat(chunks, dtype):
        if not chunks:
            return np.zeros(0, dtype=dtype)
        return np.concatenate(chunks).astype(dtype)

    return {
        "sem_preds": cat(sem_preds, np.int64),
        "sem_gts": cat(sem_gts, np.int64),
        "bot_preds": cat(bot_preds, np.int64),
        "bot_gts": cat(bot_gts, np.int64),
        "inst_sims": cat(inst_sims, np.float32),
        "inst_targets": cat(inst_targets, np.int64),
    }


def _per_class_metrics(preds, gts, n_classes):
    """Per-class accuracy, precision, recall, F1, and support.

    Identical computation to ``evaluate_semantic._per_class_metrics`` so the
    semantic numbers produced by the two scripts are directly comparable.
    """
    metrics = {}
    for cls in range(n_classes):
        gt_mask = gts == cls
        support = int(gt_mask.sum())
        if support == 0:
            continue
        tp = int(((preds == cls) & (gts == cls)).sum())
        fp = int(((preds == cls) & (gts != cls)).sum())
        fn = int(((preds != cls) & (gts == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        metrics[cls] = {
            # This is TP/(TP+FN), i.e. recall.
            "accuracy": float((preds[gt_mask] == cls).mean()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
    return metrics


def _threshold_row(sims, targets, threshold):
    """Accuracy / precision / recall / F1 / CM at one similarity threshold."""
    preds = (sims >= threshold).astype(np.int64)
    cm = compute_confusion_matrix(preds, targets, n_classes=2)
    row = compute_f1_binary(cm)
    row["accuracy"] = float(compute_accuracy(preds, targets))
    row["threshold"] = float(threshold)
    row["confusion_matrix"] = cm.tolist()
    return row


#  Main evaluation function

def evaluate_unified_model(cfg, checkpoint_path, seed=None):
    """Evaluate all three heads of a unified checkpoint.

    Args:
        cfg: Parsed YAML configuration dict (the one used for training).
        checkpoint_path: Path to the unified ``.pth`` checkpoint.
        seed: Seed for the random_split fallback when no split file is set.

    Returns:
        Dict of computed metrics (also written to ``unified_eval_metrics.json``).
    """
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(checkpoint_path, device=device)
    model, label_encoder, bottom_encoder, n_classes = _build_models(
        cfg, ckpt, device)

    test_ds, split_desc = _build_test_set(cfg, seed)
    if hasattr(test_ds, "has_instance_labels") and not test_ds.has_instance_labels:
        raise RuntimeError(
            "Unified evaluation requires instance_labels in the dataset JSONs.")
    if hasattr(test_ds, "has_bottom_labels") and not test_ds.has_bottom_labels:
        raise RuntimeError(
            "Unified evaluation requires bottom_labels in the dataset JSONs.")

    n_classes = getattr(test_ds, "n_classes", n_classes)
    batch_size = cfg.get("training", {}).get("batch_size", 4)
    # No `**loader_kwargs(...)`: shuffle=False + num_workers=0
    # makes both kwargs inert. Argument and gate:
    # tests/test_eval_dataloader_seeding.py.
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn_brep)

    print("\n" + "=" * 70)
    print("UNIFIED EVALUATION -- semantic + bottom + instance")
    print("=" * 70)
    print(f"checkpoint : {checkpoint_path}  (epoch {ckpt.get('epoch', '?')})")
    print(f"test set   : {len(test_ds)} models from {split_desc}")
    print(f"inputs     : {_input_settings_phrase(cfg)}")

    out = _run_test(model, label_encoder, bottom_encoder, loader,
                    n_classes, device)

    sem_acc = compute_accuracy(out["sem_preds"], out["sem_gts"])
    sem_cm = compute_confusion_matrix(out["sem_preds"], out["sem_gts"], n_classes)
    per_cls = _per_class_metrics(out["sem_preds"], out["sem_gts"], n_classes)

    print("\n" + "-" * 70)
    print(f"SEMANTIC  --  {len(out['sem_gts'])} faces")
    print("-" * 70)
    # Prints TP/(TP+FN) -- the same number as the "Recall" column beside it.
    header = (f"{'Class':>8} | {'Acc=Recall':>10} | {'Precision':>10} | "
              f"{'Recall':>10} | {'F1':>10} | {'Support':>10}")
    print(header)
    print("-" * len(header))
    for cls in sorted(per_cls):
        m = per_cls[cls]
        print(f"{cls:8d} | {m['accuracy']:10.2%} | {m['precision']:10.4f} | "
              f"{m['recall']:10.4f} | {m['f1']:10.4f} | {m['support']:10d}")

    macro_p = macro_r = macro_f1 = float("nan")
    if per_cls:
        macro_p = float(np.mean([m["precision"] for m in per_cls.values()]))
        macro_r = float(np.mean([m["recall"] for m in per_cls.values()]))
        macro_f1 = float(np.mean([m["f1"] for m in per_cls.values()]))
        print("-" * len(header))
        print(f"{'Macro':>8} | {'':>10} | {macro_p:10.4f} | "
              f"{macro_r:10.4f} | {macro_f1:10.4f} |")
    print("The first column is per-class accuracy = TP/(TP+FN) = recall. It is "
          "equal to the Recall column by construction, not by coincidence.")
    # mIoU is what AAGNet and SFRGNN report for this head; macro-F1 is what we
    # track across our own arms. Both come from the same confusion matrix.
    sem_miou, sem_iou_per_class = compute_mean_iou(sem_cm)

    print(f"\nOverall accuracy: {sem_acc:.2%}    macro F1: {macro_f1:.4f}"
          f"    mIoU: {sem_miou:.4f}")
    print("Macro F1 is the number to compare across arms -- overall accuracy "
          "is dominated by the majority class.")
    print("mIoU is the metric the MFInstSeg literature reports for this head. "
          "It is always <= macro F1 on the same predictions: both divide by the "
          "same errors, but F1 counts true positives twice.")

    bot_acc = compute_accuracy(out["bot_preds"], out["bot_gts"])
    bot_cm = compute_confusion_matrix(out["bot_preds"], out["bot_gts"], 2)
    bot_prf = compute_f1_binary(bot_cm)
    n_faces = max(len(out["bot_gts"]), 1)
    n_bot = int((out["bot_gts"] == 1).sum())

    print("\n" + "-" * 70)
    print(f"BOTTOM-FACE  --  {len(out['bot_gts'])} faces, "
          f"{n_bot} positive ({n_bot / n_faces:.1%})")
    print("-" * 70)
    bot_miou, bot_iou_per_class = compute_mean_iou(bot_cm)

    print(f"Accuracy:  {bot_acc:.2%}")
    print(f"Precision: {bot_prf['precision']:.4f}   "
          f"Recall: {bot_prf['recall']:.4f}   F1: {bot_prf['f1']:.4f}")
    print(f"mIoU:      {bot_miou:.4f}   "
          f"(not-bottom {bot_iou_per_class.get(0, 0.0):.4f}, "
          f"bottom {bot_iou_per_class.get(1, 0.0):.4f})")
    print("   AAGNet and SFRGNN report this head as accuracy + mIoU, so mIoU is "
          "the number to quote beside theirs -- on our own split, not theirs.")
    print(f"(an always-negative predictor would score "
          f"{1.0 - n_bot / n_faces:.2%} accuracy here -- read the F1, "
          f"not the accuracy)")

    sims, targets = out["inst_sims"], out["inst_targets"]
    n_pairs = max(len(targets), 1)
    inst_auc = compute_auc(targets, sims)
    n_pos = int(targets.sum())

    # These are ORDERED pairs. `_run_test`
    # masks only the diagonal of each model's square similarity matrix
    # (`off = ~torch.eye(...)`), so (i,j) and (j,i) are both collected and
    # every count here is exactly twice the unordered one.
    n_ordered = int(len(targets))
    if n_ordered % 2 != 0:
        raise RuntimeError(
            f"the instance head collected {n_ordered} pair scores, an ODD "
            f"number. They are built by masking the diagonal of a square "
            f"per-model similarity matrix in _run_test, so every (i,j) has a "
            f"(j,i) and the total must be even. An odd total means the mask "
            f"is wrong, and the unordered counts derived from it would be a "
            f"lie. Not halving it.")
    n_unordered = n_ordered // 2

    print("\n" + "-" * 70)
    print(f"INSTANCE  --  {n_ordered} ORDERED face pairs "
          f"({n_unordered} unordered), "
          f"{n_pos} positive ({n_pos / n_pairs:.1%})")
    print("   the diagonal is masked but (i,j) and (j,i) are both counted, so "
          "every count on this head is twice the unordered one -- quote "
          f"{n_unordered}, not {n_ordered}, as a number of face pairs.")
    print("-" * 70)
    print(f"ROC-AUC: {inst_auc:.4f}   (threshold-free)")

    ckpt_thresh = ckpt.get("optimal_threshold")
    rows = {}
    if ckpt_thresh is not None:
        rows["checkpoint_threshold"] = _threshold_row(sims, targets, ckpt_thresh)
        r = rows["checkpoint_threshold"]
        print(f"\n  at checkpoint threshold {ckpt_thresh:.4f}   "
              f"[tuned on the VALIDATION split by train_unified.py, "
              f"never on test -- REPORTABLE]")
        print(f"    accuracy {r['accuracy']:.2%}   P {r['precision']:.4f}   "
              f"R {r['recall']:.4f}   F1 {r['f1']:.4f}")
    else:
        print("\n  !! this checkpoint stores no optimal_threshold, so only the "
              "AUC above is reportable for the instance head.")

    oracle_t = AdaptiveThresholdFinder().find_optimal_threshold(sims, targets)
    rows["oracle_threshold"] = _threshold_row(sims, targets, oracle_t)
    r = rows["oracle_threshold"]
    print(f"\n  at test-tuned threshold {oracle_t:.4f}   "
          f"[ORACLE: has seen the test labels -- DO NOT REPORT]")
    print(f"    accuracy {r['accuracy']:.2%}   P {r['precision']:.4f}   "
          f"R {r['recall']:.4f}   F1 {r['f1']:.4f}")
    if ckpt_thresh is not None:
        gap = r["f1"] - rows["checkpoint_threshold"]["f1"]
        print(f"    oracle minus reportable F1: {gap:+.4f}   "
              f"(how much a test-tuned threshold would buy over the "
              f"val-tuned one -- a generalisation gap, not a leak)")

    fig_dir = Path(cfg.get("paths", {}).get("figures", "results/figures/"))
    fig_dir.mkdir(parents=True, exist_ok=True)
    name = cfg.get("experiment_name", "unified")

    plot_confusion_matrix(
        sem_cm, n_classes=n_classes,
        title=f"Semantic Confusion Matrix — {name}",
        save_path=fig_dir / "unified_semantic_confusion_matrix.png")
    plot_confusion_matrix(
        bot_cm, n_classes=2, class_names=["Not Bottom", "Bottom"],
        title=f"Bottom-Face Confusion Matrix — {name}",
        save_path=fig_dir / "unified_bottom_confusion_matrix.png")
    if len(np.unique(targets)) >= 2:
        fpr, tpr, thr = roc_curve(targets, sims)
        plot_roc_curve(
            fpr, tpr, auc=inst_auc,
            optimal_threshold=(ckpt_thresh if ckpt_thresh is not None
                               else oracle_t),
            thresholds=thr,
            title=f"Instance Pair ROC — {name}",
            save_path=fig_dir / "unified_instance_roc.png")

    metrics = {
        "experiment_name": name,
        "checkpoint": str(checkpoint_path),
        "epoch": ckpt.get("epoch"),
        "split": split_desc,
        "n_models": int(len(test_ds)),
        # These two are what the dataset was built with, default or not.
        # The `_source` keys say whether the config stated a value or the
        # fallback supplied it.
        "attributes": cfg["dataset"].get("attributes", "scaled"),
        "attributes_source": _setting_source(
            cfg["dataset"].get("attributes")),
        "bias_type": cfg.get("bias", {}).get("type", "similarity"),
        "bias_type_source": _setting_source(
            (cfg.get("bias") or {}).get("type")),
        # What the per_class keys below actually mean. Kept at the top level
        # rather than inside "semantic" so that `check_tier.semantic_block`
        # and `aggregate_seeds.summarise_arm` see a block of that shape.
        "per_class_key_semantics": PER_CLASS_KEY_SEMANTICS,
        "semantic": {
            "n_faces": int(len(out["sem_gts"])),
            "n_classes": int(n_classes),
            "overall_accuracy": float(sem_acc),
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "mean_iou": sem_miou,
            "iou_per_class": {str(k): v for k, v in sem_iou_per_class.items()},
            "per_class": {str(k): v for k, v in per_cls.items()},
            "confusion_matrix": sem_cm.tolist(),
        },
        "bottom": {
            "n_faces": int(len(out["bot_gts"])),
            "n_positive": n_bot,
            "accuracy": float(bot_acc),
            "precision": bot_prf["precision"],
            "recall": bot_prf["recall"],
            "f1": bot_prf["f1"],
            "mean_iou": bot_miou,
            "iou_not_bottom": bot_iou_per_class.get(0, 0.0),
            "iou_bottom": bot_iou_per_class.get(1, 0.0),
            "confusion_matrix": bot_cm.tolist(),
        },
        "instance": {
            # `n_pairs` and `n_positive` are
            # kept with their values untouched -- the archived T5 files carry
            # them -- and the three keys beside them say what they count.
            "n_pairs": int(len(targets)),
            "n_pairs_ordered": n_ordered,
            "n_pairs_unordered": n_unordered,
            "n_positive": n_pos,
            "n_positive_ordered": n_pos,
            "pair_counting": (
                "ORDERED pairs. _run_test masks the diagonal of each model's "
                "square similarity matrix but keeps both (i,j) and (j,i), so "
                "n_pairs and n_positive are exactly twice the unordered "
                "counts. n_pairs is retained unchanged for the archived "
                "files; a caption that says 'face pairs' wants "
                "n_pairs_unordered."),
            "auc": float(inst_auc),
            "reportable": ("checkpoint_threshold" if ckpt_thresh is not None
                           else "auc_only"),
            # The checkpoint threshold was fitted on the validation partition.
            "checkpoint_threshold_source": (
                "fitted on the validation partition by train_unified.py "
                "(find_optimal_threshold over val_loader similarities) and "
                "stored with the best-validation-loss checkpoint"
                if ckpt_thresh is not None else None),
            "note": ("oracle_threshold was tuned on the test labels and is an "
                     "upper bound only; report checkpoint_threshold."),
            **rows,
        },
    }
    # see evaluate_semantic.py -- the six required fields,
    # carried through from the checkpoint that produced these numbers.
    metrics["provenance"] = provenance_from_checkpoint(ckpt, cfg)

    out_path = fig_dir / "unified_eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")
    print(f"Figures saved to {fig_dir}")

    return metrics


#  CLI entry point                                                       #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a unified checkpoint "
                    "(semantic + bottom + instance) on the test split")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML config used for training")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to the .pth checkpoint (overrides paths.checkpoint)")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed for the random_split fallback when the config has no "
             "split_file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = args.checkpoint or cfg.get("paths", {}).get("checkpoint")
    if ckpt_path is None:
        raise ValueError(
            "Provide --checkpoint or set paths.checkpoint in the config")

    evaluate_unified_model(cfg, ckpt_path, seed=args.seed)


if __name__ == "__main__":
    main()
