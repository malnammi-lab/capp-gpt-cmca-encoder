"""
Evaluate a trained link prediction model (Stage 2).

Loads a frozen Stage 1 semantic model and a trained GraphLinkPredictor,
runs pairwise link classification on a held-out test split, and reports
accuracy, AUC, precision / recall / F1, and an ROC curve.

Usage:
    python -m src.evaluation.evaluate_link \
        --config configs/mfinstseg_two_stage.yaml \
        --checkpoint results/checkpoints/mfinstseg_two_stage_link.pth

    # The semantic checkpoint is resolved from the link checkpoint metadata
    # or from the config's semantic_checkpoint field.  Override explicitly:
    python -m src.evaluation.evaluate_link \
        --config configs/mfinstseg_two_stage.yaml \
        --checkpoint results/checkpoints/mfinstseg_two_stage_link.pth \
        --semantic-checkpoint results/checkpoints/mfinstseg_two_stage.pth
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
from ..models.encoders import SemanticSegmentationModel
from ..models.link_predictor import GraphLinkPredictor
from ..utils.checkpoint import load_checkpoint
from ..utils.provenance import provenance_from_checkpoint
from ..utils.metrics import (
    AdaptiveThresholdFinder,
    compute_accuracy,
    compute_auc,
    compute_confusion_matrix,
    compute_f1_binary,
)
from ..utils.visualization import plot_confusion_matrix, plot_roc_curve


def load_config(path):
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


#  Helpers

def _flatten_valid_pairs(link_logits, instance_labels, face_mask, device):
    """Extract valid (non-padded, non-diagonal) pair logits and labels.

    Mirrors the function in ``train_link.py``.

    Returns:
        (flat_logits, flat_labels) or (None, None) when empty.
    """
    logits_list, labels_list = [], []
    for b in range(link_logits.shape[0]):
        mask = face_mask[b]
        n_valid = mask.sum().item()
        if n_valid < 2:
            continue

        valid_idx = torch.nonzero(mask).squeeze(1)
        pair_logits = link_logits[b][valid_idx[:, None], valid_idx]
        pair_labels = instance_labels[b][valid_idx[:, None], valid_idx]

        eye = torch.eye(n_valid, device=device).bool()
        logits_list.append(pair_logits[~eye])
        labels_list.append(pair_labels[~eye].long())

    if not logits_list:
        return None, None
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def _score_loader(semantic_model, link_model, loader, device):
    """Run both models over a loader and return (probs, targets, argmax_preds).

    Factored out of the inline evaluation loop so the same forward
    pass can also be run over the TRAIN split when a decision threshold has to
    be tuned without touching test labels.
    """
    all_probs, all_targets, all_preds = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            embeddings = semantic_model(
                batch["faces_uv"], batch["faces_attr"],
                batch["edges_attr"], batch["bias"],
                face_mask=batch["face_mask"],
                edge_mask=batch["edge_mask"],
            )

            link_logits = link_model(
                embeddings, bias=batch["bias"], mask=batch["face_mask"],
            )

            flat_logits, flat_labels = _flatten_valid_pairs(
                link_logits, batch["instance_labels"],
                batch["face_mask"], device,
            )
            if flat_logits is None:
                continue

            all_probs.append(
                F.softmax(flat_logits, dim=1)[:, 1].cpu().numpy())
            all_targets.append(flat_labels.cpu().numpy())
            all_preds.append(flat_logits.argmax(dim=1).cpu().numpy())

    def cat(chunks):
        return np.concatenate(chunks) if chunks else np.zeros(0)

    return cat(all_probs), cat(all_targets).astype(int), cat(all_preds).astype(int)


#  Main evaluation function

def evaluate_link_predictor(
    cfg,
    link_checkpoint_path,
    semantic_checkpoint_path=None,
    seed=None,
    tune_threshold_on_train=False,
):
    """Evaluate a link prediction model and save results.

    Args:
        cfg: Parsed YAML configuration dict.
        link_checkpoint_path: Path to the link predictor checkpoint.
        semantic_checkpoint_path: Path to the frozen Stage 1 semantic
            checkpoint.  Resolved automatically from the link checkpoint
            metadata or the config when not given.
        seed: Optional random seed for reproducing the data split.

    Returns:
        Dict of computed metrics.
    """
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    link_ckpt = load_checkpoint(link_checkpoint_path, device=device)

    if semantic_checkpoint_path is None:
        semantic_checkpoint_path = link_ckpt.get(
            "semantic_checkpoint",
            cfg.get("semantic_checkpoint"),
        )
    if semantic_checkpoint_path is None:
        raise ValueError(
            "Cannot determine semantic checkpoint.  Provide "
            "--semantic-checkpoint or set semantic_checkpoint in the config."
        )
    sem_ckpt = load_checkpoint(semantic_checkpoint_path, device=device)

    sem_cfg = cfg.get("semantic_model", cfg.get("model", {}))
    semantic_model = SemanticSegmentationModel(
        d_model=sem_cfg.get("d_model", 128),
        n_heads=sem_cfg.get("n_heads", 8),
        n_layers=sem_cfg.get("n_layers", 4),
        dim_feedforward=sem_cfg.get("dim_feedforward", 256),
        dropout=sem_cfg.get("dropout", 0.1),
    ).to(device)
    semantic_model.load_state_dict(sem_ckpt["model"])
    semantic_model.eval()
    for param in semantic_model.parameters():
        param.requires_grad = False
    print(f"Frozen semantic model loaded from {semantic_checkpoint_path}")

    m_cfg = cfg.get("link_model", cfg.get("model", {}))
    link_model = GraphLinkPredictor(
        d_model=m_cfg.get("d_model", 128),
        n_heads=m_cfg.get("n_heads", 8),
        n_layers=m_cfg.get("n_layers", 2),
        dim_feedforward=m_cfg.get("dim_feedforward", 256),
        dropout=m_cfg.get("dropout", 0.1),
    ).to(device)
    link_model.load_state_dict(link_ckpt["model"])
    link_model.eval()

    # Build the same kwargs the trainer uses.
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

    # Only the random_split path needs the whole corpus in memory.
    dataset = None
    if not split_file:
        dataset = PreExtractedBRepDataset(**ds_kwargs)
        if not dataset.has_instance_labels:
            raise RuntimeError(
                "Link evaluation requires instance_labels in the dataset JSONs."
            )

    # A saved split file takes precedence.
    batch_size = cfg.get("training", {}).get("batch_size", 4)
    train_loader = None

    if split_file:
        test_ds = PreExtractedBRepDataset(
            model_ids=load_split(split_file, "test"), **ds_kwargs)
        if not test_ds.has_instance_labels:
            raise RuntimeError(
                "Link evaluation requires instance_labels in the dataset JSONs."
            )
        print(f"\nSplit file: {split_file}")
        print(f"Evaluating on test split ({len(test_ds)} models) -- "
              f"attributes={ds_kwargs['attributes']}, "
              f"bias={ds_kwargs['bias_type']}")
        if tune_threshold_on_train:
            train_ds = PreExtractedBRepDataset(
                model_ids=load_split(split_file, "train"), verbose=False,
                **ds_kwargs)
            # No `**loader_kwargs(...)`: shuffle=False +
            # num_workers=0 makes both kwargs inert, and the threshold this
            # loader feeds comes from roc_curve, which is order-independent.
            # Argument and gate: tests/test_eval_dataloader_seeding.py.
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn_brep)
            print(f"Threshold will be tuned on the train split "
                  f"({len(train_ds)} models), never on test.")
        # see the note above -- same reasoning.
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn_brep,
        )
        return _run_link_evaluation(
            cfg, link_checkpoint_path, semantic_checkpoint_path,
            semantic_model, link_model, test_loader, train_loader, device,
            link_ckpt,
        )

    generator = (
        torch.Generator().manual_seed(seed) if seed is not None else None
    )

    train_ratio = ds_cfg.get("train_split", 0.7)
    val_ratio = ds_cfg.get("val_split", 0.15)
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    if test_size <= 0:
        test_ds = dataset
        print(f"\nEvaluating on full dataset ({len(dataset)} samples)")
    elif val_size > 0:
        _, _, test_ds = random_split(
            dataset, [train_size, val_size, test_size], generator=generator,
        )
        print(f"\nEvaluating on test split ({test_size} samples)")
    else:
        _, test_ds = random_split(
            dataset, [train_size, test_size], generator=generator,
        )
        print(f"\nEvaluating on test split ({test_size} samples)")

    # No `**loader_kwargs(...)`: shuffle=False + num_workers=0
    # makes both kwargs inert. The seed that matters on THIS branch is the one
    # already passed to `random_split` above, which chooses WHICH models are in
    # the test set. Argument and gate: tests/test_eval_dataloader_seeding.py.
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_brep,
    )
    return _run_link_evaluation(
        cfg, link_checkpoint_path, semantic_checkpoint_path,
        semantic_model, link_model, test_loader, None, device,
        link_ckpt,
    )


def _run_link_evaluation(
    cfg,
    link_checkpoint_path,
    semantic_checkpoint_path,
    semantic_model,
    link_model,
    test_loader,
    train_loader,
    device,
    link_ckpt,
):
    """Score the test loader, report metrics, and write figures + JSON.

    Args:
        link_ckpt: The LOADED link-predictor checkpoint dict, not just its path.
            Required, not optional: the regiment block
            written into ``link_eval_metrics.json`` is copied out of the
            checkpoint that produced the weights, exactly as
            ``evaluate_semantic.py`` and ``evaluate_unified.py`` do. Passing it
            positionally from both call sites means a future branch that forgets
            it fails with a TypeError at the call, not with a silently
            provenance-free metrics file that the tier gate then rejects.
    """
    print("\n" + "=" * 70)
    print("LINK PREDICTION EVALUATION")
    print("=" * 70)

    all_probs, all_targets, all_preds = _score_loader(
        semantic_model, link_model, test_loader, device,
    )

    accuracy = compute_accuracy(all_preds, all_targets)
    auc = compute_auc(all_targets, all_probs)

    cm = compute_confusion_matrix(all_preds, all_targets, n_classes=2)
    binary_metrics = compute_f1_binary(cm)

    # The reportable row is argmax. A threshold fitted on the test scores
    # is printed only as an oracle upper bound.
    threshold_finder = AdaptiveThresholdFinder()
    train_threshold = None
    train_row = None
    if train_loader is not None:
        tr_probs, tr_targets, _ = _score_loader(
            semantic_model, link_model, train_loader, device,
        )
        train_threshold = threshold_finder.find_optimal_threshold(
            tr_probs, tr_targets,
        )
        tr_preds = (all_probs >= train_threshold).astype(int)
        tr_cm = compute_confusion_matrix(tr_preds, all_targets, n_classes=2)
        train_row = compute_f1_binary(tr_cm)
        train_row["accuracy"] = float(compute_accuracy(tr_preds, all_targets))
        train_row["threshold"] = float(train_threshold)
        train_row["confusion_matrix"] = tr_cm.tolist()

    optimal_threshold = threshold_finder.find_optimal_threshold(
        all_probs, all_targets,
    )

    thresh_preds = (all_probs >= optimal_threshold).astype(int)
    thresh_acc = compute_accuracy(thresh_preds, all_targets)
    thresh_cm = compute_confusion_matrix(thresh_preds, all_targets, n_classes=2)
    thresh_f1 = compute_f1_binary(thresh_cm)

    print(f"\nTotal pairs evaluated: {len(all_targets)}")
    print(f"Positive pairs (same instance): {int(all_targets.sum())}")
    print(f"Negative pairs (different):     {int((1 - all_targets).sum())}")
    print(f"\n--- Argmax predictions  [REPORTABLE: no threshold fitted] ---")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {binary_metrics['precision']:.4f}")
    print(f"Recall:    {binary_metrics['recall']:.4f}")
    print(f"F1:        {binary_metrics['f1']:.4f}")
    print(f"AUC:       {auc:.4f}")
    if train_row is not None:
        print(f"\n--- Train-tuned threshold ({train_threshold:.4f})  "
              f"[REPORTABLE: tuned on train, applied to test] ---")
        print(f"Accuracy:  {train_row['accuracy']:.2%}")
        print(f"Precision: {train_row['precision']:.4f}")
        print(f"Recall:    {train_row['recall']:.4f}")
        print(f"F1:        {train_row['f1']:.4f}")

    print(f"\n--- Test-tuned threshold ({optimal_threshold:.4f})  "
          f"[ORACLE: fitted on the test labels -- DO NOT REPORT] ---")
    print(f"Accuracy:  {thresh_acc:.2%}")
    print(f"Precision: {thresh_f1['precision']:.4f}")
    print(f"Recall:    {thresh_f1['recall']:.4f}")
    print(f"F1:        {thresh_f1['f1']:.4f}")
    if train_row is None:
        print("(pass --tune-threshold-on-train for a reportable thresholded "
              "row; without it, quote the argmax row or the AUC.)")

    paths_cfg = cfg.get("paths", {})
    fig_dir = Path(paths_cfg.get("figures", "results/figures/"))
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    if len(np.unique(all_targets)) >= 2:
        fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
        plot_roc_curve(
            fpr, tpr,
            auc=auc,
            optimal_threshold=optimal_threshold,
            thresholds=thresholds,
            title=f"Link Prediction ROC — {cfg.get('experiment_name', 'eval')}",
            save_path=fig_dir / "link_roc_curve.png",
        )
        print(f"\nROC curve saved to {fig_dir / 'link_roc_curve.png'}")

    # Confusion matrix (argmax)
    plot_confusion_matrix(
        cm,
        n_classes=2,
        class_names=["No Link", "Same Instance"],
        title=f"Link Confusion Matrix — {cfg.get('experiment_name', 'eval')}",
        save_path=fig_dir / "link_confusion_matrix.png",
    )
    print(f"Confusion matrix saved to {fig_dir / 'link_confusion_matrix.png'}")

    # Metrics JSON
    metrics = {
        "experiment_name": cfg.get("experiment_name", "eval"),
        "link_checkpoint": str(link_checkpoint_path),
        "semantic_checkpoint": str(semantic_checkpoint_path),
        "n_pairs": int(len(all_targets)),
        "n_positive": int(all_targets.sum()),
        "n_negative": int((1 - all_targets).sum()),
        "argmax": {
            "accuracy": float(accuracy),
            "precision": float(binary_metrics["precision"]),
            "recall": float(binary_metrics["recall"]),
            "f1": float(binary_metrics["f1"]),
            "auc": float(auc),
            "confusion_matrix": cm.tolist(),
        },
        "oracle_threshold": {
            "value": float(optimal_threshold),
            "accuracy": float(thresh_acc),
            "precision": float(thresh_f1["precision"]),
            "recall": float(thresh_f1["recall"]),
            "f1": float(thresh_f1["f1"]),
            "confusion_matrix": thresh_cm.tolist(),
            "note": "tuned on the test labels -- upper bound only, do not report",
        },
        "reportable": "train_threshold" if train_row else "argmax",
    }
    if train_row is not None:
        metrics["train_threshold"] = train_row

    # training-regiment.md section 1 requires every
    # metrics file to carry seed, epoch, config hash, code version, GPU and wall
    # time, and also selected_epoch and stopping_reason.
    # The block describes the LINK training run (the weights being scored),
    # which is why it comes out of link_ckpt and not out of the frozen Stage 1
    # checkpoint.
    metrics["provenance"] = provenance_from_checkpoint(link_ckpt, cfg)

    metrics_path = fig_dir / "link_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


#  CLI entry point                                                       #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained link prediction model (Stage 2)",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to the link predictor .pth checkpoint",
    )
    parser.add_argument(
        "--semantic-checkpoint", type=str, default=None,
        help="Path to the frozen Stage 1 semantic checkpoint "
             "(resolved from link checkpoint or config if omitted)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed to reproduce the split when no split_file is set",
    )
    parser.add_argument(
        "--tune-threshold-on-train", action="store_true",
        help="Also score the train split to fit a decision threshold without "
             "touching test labels (needs a split_file; costs one extra pass)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    link_ckpt_path = (
        args.checkpoint
        or cfg.get("paths", {}).get("link_checkpoint")
    )
    if link_ckpt_path is None:
        raise ValueError(
            "Provide --checkpoint or set paths.link_checkpoint in the config"
        )

    evaluate_link_predictor(
        cfg, link_ckpt_path,
        semantic_checkpoint_path=args.semantic_checkpoint,
        seed=args.seed,
        tune_threshold_on_train=args.tune_threshold_on_train,
    )


if __name__ == "__main__":
    main()
