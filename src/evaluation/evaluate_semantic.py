"""
Evaluate a trained semantic segmentation model.

Supports both the Graph Transformer family (SemanticSegmentationModel /
BRepEncoder) and the ResNet baseline (CrossModalAlignmentModel).  Loads a
checkpoint, runs prototype-based classification on a test split, and
reports accuracy, per-class precision / recall / F1, a confusion matrix,
and macro-averaged metrics.

Usage:
    python -m src.evaluation.evaluate_semantic \
        --config configs/cadsynth_bias_both.yaml \
        --checkpoint results/checkpoints/cadsynth_bias_both.pth

    python -m src.evaluation.evaluate_semantic \
        --config configs/resnet_cadsynth.yaml \
        --checkpoint results/checkpoints/resnet_cadsynth.pth

    # Reproduce the same train/val/test split used during training:
    python -m src.evaluation.evaluate_semantic \
        --config configs/mfinstseg_unified.yaml --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split

from ..data.brep_dataset import PreExtractedBRepDataset, load_split
from ..data.collate import collate_fn_brep
from ..data.uv_dataset import UVGridLabelDataset
from ..models.ce_encoder import SemanticClassificationModel
from ..models.encoders import BRepEncoder, LabelEncoder, SemanticSegmentationModel
from ..models.resnet_encoder import (CrossModalAlignmentModel,
                                     ResNetClassificationModel)
from ..utils.checkpoint import load_checkpoint
from ..utils.provenance import provenance_from_checkpoint
from ..utils.metrics import (compute_accuracy, compute_confusion_matrix,
                            compute_mean_iou)
from ..utils.visualization import plot_confusion_matrix


def load_config(path):
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


#  What the banner line is allowed to claim
#
# The helpers below report what the config states, and label a default as a
# default. They take `cfg` and two integers and return a string that is only
# ever printed.

_RESNET_READS_NEITHER = ("the ResNet UV-grid path reads neither "
                         "dataset.attributes nor bias.*")


def _input_settings_phrase(cfg, is_resnet=False):
    """Describe the attribute/bias settings this run actually used.

    A duplicate of this function lives in ``evaluate_unified.py``.  The two are
    kept in step by ``tests/test_gatefix_evaluators.py``; they are duplicated
    rather than imported so that ``evaluate_unified`` does not have to import
    the ResNet and UV-grid modules it otherwise never touches.
    """
    ds_cfg = cfg.get("dataset", {}) or {}
    stated_attr = ds_cfg.get("attributes")
    stated_bias = (cfg.get("bias") or {}).get("type")

    if is_resnet:
        attr = ("attributes=n/a" if stated_attr is None
                else f"attributes={stated_attr} [IGNORED]")
        bias = ("bias=n/a" if stated_bias is None
                else f"bias={stated_bias} [IGNORED]")
        return f"{attr}, {bias}   ({_RESNET_READS_NEITHER})"

    attr = (f"attributes={stated_attr}" if stated_attr is not None
            else "attributes=scaled [DATASET DEFAULT -- dataset.attributes is "
                 "absent from this config]")
    bias = (f"bias={stated_bias}" if stated_bias is not None
            else "bias=similarity [DATASET DEFAULT -- bias.type is absent "
                 "from this config]")
    return f"{attr}, {bias}"


def _split_size_phrase(n_samples, is_resnet, n_models=None):
    """``"N models"`` or ``"N faces from M models"``, whichever is true.

    The unit depends on which dataset class the path built.
    """
    if not is_resnet:
        return f"{n_samples} models"
    if n_models is None:
        return f"{n_samples} faces"
    return f"{n_samples} faces from {n_models} models"


#  What the per-class keys mean

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


#  Model builders

def _build_graph_transformer(cfg, ckpt, device):
    """Instantiate a Graph Transformer model from a checkpoint.

    Returns (model, label_encoder, n_classes). For a cross-entropy checkpoint
    `label_encoder` is None: that arm has no label tower, and classification is
    an argmax over the head's logits rather than a nearest-prototype lookup.
    """
    m_cfg = cfg.get("model", cfg.get("semantic_model", {}))
    model_type = m_cfg.get("type", "semantic_segmentation")
    is_ce = cfg.get("loss", {}).get("type") == "cross_entropy"

    n_classes = ckpt.get("n_classes", cfg["dataset"].get("n_classes", 25))

    # the cross-entropy arm
    if is_ce:
        model = SemanticClassificationModel(
            n_classes=n_classes,
            d_model=m_cfg.get("d_model", 128),
            n_heads=m_cfg.get("n_heads", 8),
            n_layers=m_cfg.get("n_layers", 4),
            dim_feedforward=m_cfg.get("dim_feedforward", 256),
            dropout=m_cfg.get("dropout", 0.1),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        if "label_encoder" in ckpt:
            print("  NOTE: this CE checkpoint carries a label_encoder. It was "
                  "not trained by the CE objective and is ignored.")
        return model, None, n_classes

    ModelClass = (
        BRepEncoder if model_type == "brep_encoder"
        else SemanticSegmentationModel
    )

    model = ModelClass(
        d_model=m_cfg.get("d_model", 128),
        n_heads=m_cfg.get("n_heads", 8),
        n_layers=m_cfg.get("n_layers", 4),
        dim_feedforward=m_cfg.get("dim_feedforward", 256),
        dropout=m_cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    le_cfg = cfg.get("label_encoder", {})
    label_encoder = LabelEncoder(
        n_classes=n_classes,
        embedding_dim=le_cfg.get("embedding_dim", m_cfg.get("d_model", 128)),
    ).to(device)
    if "label_encoder" in ckpt:
        label_encoder.load_state_dict(ckpt["label_encoder"])
    label_encoder.eval()

    return model, label_encoder, n_classes


def _build_resnet(cfg, ckpt, device):
    """Instantiate the ResNet baseline from a checkpoint.

    Returns (model, label_encoder, n_classes). For a cross-entropy checkpoint
    `label_encoder` is None: that arm has a linear class head instead of a label
    tower, exactly as in `_build_graph_transformer`.
    """
    m_cfg = cfg.get("model", {})
    n_classes = ckpt.get("n_classes", cfg["dataset"].get("n_classes", 25))
    is_ce = cfg.get("loss", {}).get("type") == "cross_entropy"

    if is_ce:
        # The class-weight buffer is taken from the checkpoint so its shape
        # matches the saved weights. Evaluation does not read the values.
        saved_weights = (ckpt.get("model") or {}).get("_class_weights")
        if saved_weights is not None and saved_weights.numel() == 0:
            saved_weights = None

        model = ResNetClassificationModel(
            n_classes=n_classes,
            embedding_dim=m_cfg.get("embedding_dim", 512),
            dropout=m_cfg.get("dropout", 0.3),
            class_weights=saved_weights,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, None, n_classes

    model = CrossModalAlignmentModel(
        n_classes=n_classes,
        embedding_dim=m_cfg.get("embedding_dim", 512),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, model.label_encoder, n_classes


#  Evaluation loops

def _evaluate_graph_transformer(model, label_encoder, dataloader, n_classes,
                                device):
    """Prototype-based evaluation for Graph Transformer models.

    Returns:
        Tuple of (all_preds, all_gts) as 1-D numpy arrays.
    """
    all_preds, all_gts = [], []

    # A cross-entropy model has no prototypes -- its head emits logits directly.
    is_ce = label_encoder is None
    prototypes = None
    if not is_ce:
        all_cls = torch.arange(n_classes, device=device)
        with torch.no_grad():
            prototypes = F.normalize(label_encoder(all_cls), dim=1)

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            out = model(
                batch["faces_uv"], batch["faces_attr"],
                batch["edges_attr"], batch["bias"],
                face_mask=batch["face_mask"],
                edge_mask=batch["edge_mask"],
            )

            for b in range(out.shape[0]):
                mask = batch["face_mask"][b]
                if mask.sum() == 0:
                    continue
                if is_ce:
                    preds = out[b][mask].argmax(dim=1)
                else:
                    emb = F.normalize(out[b][mask], dim=1)
                    preds = torch.matmul(emb, prototypes.T).argmax(dim=1)
                labels = batch["semantic_labels"][b][mask]

                all_preds.extend(preds.cpu().numpy())
                all_gts.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_gts)


def _evaluate_resnet(model, label_encoder, dataloader, n_classes, device):
    """Evaluate the ResNet baseline, contrastive or cross-entropy.

    logits rather than a nearest-prototype lookup.

    Returns:
        Tuple of (all_preds, all_gts) as 1-D numpy arrays.
    """
    all_preds, all_gts = [], []
    is_ce = label_encoder is None

    prototypes = None
    if not is_ce:
        all_cls = torch.arange(n_classes, device=device)
        with torch.no_grad():
            prototypes = F.normalize(label_encoder(all_cls), dim=1)

    with torch.no_grad():
        for uv_grids, labels in dataloader:
            uv_grids = uv_grids.to(device)
            labels = labels.to(device)

            if is_ce:
                preds = model.predict(uv_grids)
            else:
                uv_emb = model.uv_encoder(uv_grids)
                emb_n = F.normalize(uv_emb, dim=1)
                preds = torch.matmul(emb_n, prototypes.T).argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_gts)


#  Per-class metric computation

def _per_class_metrics(all_preds, all_gts, n_classes):
    """Compute per-class accuracy, precision, recall, F1, and support."""
    metrics = {}
    for cls in range(n_classes):
        gt_mask = all_gts == cls
        support = int(gt_mask.sum())
        if support == 0:
            continue
        tp = int(((all_preds == cls) & (all_gts == cls)).sum())
        fp = int(((all_preds == cls) & (all_gts != cls)).sum())
        fn = int(((all_preds != cls) & (all_gts == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        metrics[cls] = {
            # This is TP/(TP+FN), i.e. recall.
            "accuracy": float((all_preds[gt_mask] == cls).mean()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
    return metrics


#  Main evaluation function

def evaluate_semantic_model(cfg, checkpoint_path, seed=None):
    """Evaluate a semantic segmentation model and save results.

    Args:
        cfg: Parsed YAML configuration dict.
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        seed: Optional random seed for reproducing train/val/test split.

    Returns:
        Dict of computed metrics.
    """
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(checkpoint_path, device=device)

    ds_cfg = cfg["dataset"]
    m_cfg = cfg.get("model", {})
    model_type = m_cfg.get("type", "semantic_segmentation")
    is_resnet = model_type == "resnet_encoder"
    # Set only on the ResNet split-file path; stays None everywhere else.
    n_test_models = None

    if is_resnet:
        model, label_encoder, n_classes = _build_resnet(cfg, ckpt, device)
    else:
        model, label_encoder, n_classes = _build_graph_transformer(
            cfg, ckpt, device,
        )

    if is_resnet:
        # Load the test ids from the split file. Subsetting by model id
        # keeps every face of a part on one side of the split.
        resnet_split = ds_cfg.get("split_file")
        if resnet_split:
            # The id count is kept so the banner can name both numbers.
            # `len(dataset)` is faces; `n_test_models` is the model count.
            resnet_test_ids = load_split(resnet_split, "test")
            n_test_models = len(resnet_test_ids)
            dataset = UVGridLabelDataset(
                json_dir=ds_cfg["json_dir"],
                n_classes=ds_cfg.get("n_classes", 25),
                model_ids=resnet_test_ids,
            )
        else:
            raise ValueError(
                "A ResNet evaluation needs dataset.split_file. Without it this "
                "scores a random slice of the whole corpus, most of which the "
                "model trained on, and the result is meaningless. Set "
                "split_file in the config -- the ResNet cells share the graph "
                "arms' split so the 2x2 ablation's rows are comparable.")
    else:
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
            # Load only the test ids.
            dataset = PreExtractedBRepDataset(
                model_ids=load_split(split_file, "test"), **ds_kwargs)
        else:
            dataset = PreExtractedBRepDataset(**ds_kwargs)
        n_classes = dataset.n_classes

    generator = (
        torch.Generator().manual_seed(seed) if seed is not None else None
    )

    split_file = ds_cfg.get("split_file")
    train_ratio = ds_cfg.get("train_split", 0.8)
    val_ratio = ds_cfg.get("val_split", 0.0)
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    if split_file:
        # `dataset` already contains exactly the held-out models.
        test_ds = dataset
        print(f"\nSplit file: {split_file}")
        print(f"Evaluating on test split "
              f"({_split_size_phrase(len(dataset), is_resnet, n_test_models)})"
              f" -- {_input_settings_phrase(cfg, is_resnet)}")
    elif test_size <= 0:
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

    batch_size = cfg.get("training", {}).get("batch_size", 4)
    collate = None if is_resnet else collate_fn_brep
    # Deliberately NO `**loader_kwargs(seed, role)`.
    # shuffle=False -> SequentialSampler, which never reads `generator=`;
    # num_workers=0 -> `worker_init_fn` is never called. Both kwargs would be
    # inert, and `loader_kwargs` raises on seed=None, which is this script's
    # default. The full per-loader argument and the gate that fails the moment
    # this stops being true live in tests/test_eval_dataloader_seeding.py.
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
    )

    print("\n" + "=" * 70)
    print("SEMANTIC SEGMENTATION EVALUATION")
    print("=" * 70)

    if is_resnet:
        all_preds, all_gts = _evaluate_resnet(
            model, label_encoder, test_loader, n_classes, device,
        )
    else:
        all_preds, all_gts = _evaluate_graph_transformer(
            model, label_encoder, test_loader, n_classes, device,
        )

    accuracy = compute_accuracy(all_preds, all_gts)
    cm = compute_confusion_matrix(all_preds, all_gts, n_classes)
    per_cls = _per_class_metrics(all_preds, all_gts, n_classes)

    print(f"\nOverall Accuracy: {accuracy:.2%}  ({len(all_gts)} samples)")

    # The first column is Acc=Recall: TP/(TP+FN), the same number as Recall.
    header = (
        f"{'Class':>8} | {'Acc=Recall':>10} | {'Precision':>10} | "
        f"{'Recall':>10} | {'F1':>10} | {'Support':>10}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for cls in sorted(per_cls):
        m = per_cls[cls]
        print(
            f"{cls:8d} | {m['accuracy']:10.2%} | {m['precision']:10.4f} | "
            f"{m['recall']:10.4f} | {m['f1']:10.4f} | {m['support']:10d}"
        )

    if per_cls:
        macro_p = float(np.mean([m["precision"] for m in per_cls.values()]))
        macro_r = float(np.mean([m["recall"] for m in per_cls.values()]))
        macro_f1 = float(np.mean([m["f1"] for m in per_cls.values()]))
        print("-" * len(header))
        print(
            f"{'Macro':>8} | {'':>10} | {macro_p:10.4f} | "
            f"{macro_r:10.4f} | {macro_f1:10.4f} |"
        )
    print("The first column is per-class accuracy = TP/(TP+FN) = recall. It is "
          "equal to the Recall column by construction, not by coincidence.")

    # mIoU: the metric AAGNet and SFRGNN report for semantic segmentation.
    # Derived from the confusion matrix above -- no prediction changes.
    miou, iou_per_class = compute_mean_iou(cm)
    print(f"\nmIoU: {miou:.4f}   (macro F1 {macro_f1:.4f})")
    print("mIoU is always <= macro F1 on the same predictions; both divide by "
          "the same errors, but F1 counts true positives twice.")

    paths_cfg = cfg.get("paths", {})
    fig_dir = Path(paths_cfg.get("figures", "results/figures/"))
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        cm,
        n_classes=n_classes,
        title=f"Confusion Matrix — {cfg.get('experiment_name', 'eval')}",
        save_path=fig_dir / "confusion_matrix.png",
    )
    print(f"\nConfusion matrix saved to {fig_dir / 'confusion_matrix.png'}")

    metrics = {
        "experiment_name": cfg.get("experiment_name", "eval"),
        "checkpoint": str(checkpoint_path),
        "n_samples": int(len(all_gts)),
        "n_classes": int(n_classes),
        "overall_accuracy": float(accuracy),
        "per_class": {str(k): v for k, v in per_cls.items()},
        # Says what the per-class keys mean, so a paper table built from
        # `per_class` cannot print recall in an accuracy column.
        "per_class_key_semantics": PER_CLASS_KEY_SEMANTICS,
        "confusion_matrix": cm.tolist(),
    }
    if per_cls:
        metrics["macro_precision"] = macro_p
        metrics["macro_recall"] = macro_r
        metrics["macro_f1"] = macro_f1
    metrics["mean_iou"] = miou
    metrics["iou_per_class"] = {str(k): v for k, v in iou_per_class.items()}

    # training-regiment.md section 1 requires every
    # eval_metrics.json to carry seed, epoch, config hash, code version, GPU and
    # wall time. The block is copied out of the checkpoint, because it describes
    # the run that produced the weights, not this scoring pass.
    metrics["provenance"] = provenance_from_checkpoint(ckpt, cfg)

    metrics_path = fig_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


#  CLI entry point                                                       #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained semantic segmentation model",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pth checkpoint (overrides paths.checkpoint in config)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed to reproduce the train/val/test split",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    checkpoint_path = (
        args.checkpoint
        or cfg.get("paths", {}).get("checkpoint")
    )
    if checkpoint_path is None:
        raise ValueError(
            "Provide --checkpoint or set paths.checkpoint in the config"
        )

    evaluate_semantic_model(cfg, checkpoint_path, seed=args.seed)


if __name__ == "__main__":
    main()
