"""
Batch inference and prediction export to JSON.

Runs a trained model over a dataset and writes per-sample predictions to
a JSON file.  Supports three modes:

- **semantic**: Graph Transformer or ResNet semantic segmentation.  Each
  sample produces per-face predicted labels with confidence scores.
- **link**: Two-stage link prediction.  Each sample produces per-face
  semantic predictions *and* pairwise same-instance link predictions.
- **unified**: the joint semantic, bottom-face, and instance model. Each sample produces per-face
  semantic labels, bottom-face predictions, and pairwise instance
  similarity scores with adaptive thresholding.

The config passed here MUST be the config the checkpoint was TRAINED under.
It is not a formality: `dataset.attributes`, `dataset.norm_stats` and
`bias.type` decide what the network actually sees, they are not recorded in the
checkpoint, and running a model on a different view of the data produces
predictions that are wrong and look completely normal.  See the long note above
`_resolve_data_view` below.

Usage:
    # Semantic (Graph Transformer)
    python -m src.evaluation.inference \
        --config configs/cadsynth_bias_both.yaml \
        --checkpoint results/checkpoints/cadsynth_bias_both.pth \
        --mode semantic --output results/predictions/cadsynth_bias_both.json

    # Link prediction (two-stage)
    python -m src.evaluation.inference \
        --config configs/mfinstseg_two_stage.yaml \
        --checkpoint results/checkpoints/link_mfinstseg.pth \
        --mode link --output results/predictions/link_mfinstseg.json

    # Unified
    python -m src.evaluation.inference \
        --config configs/mfinstseg_unified.yaml \
        --checkpoint results/checkpoints/mfinstseg_unified.pth \
        --mode unified --output results/predictions/mfinstseg_unified.json

    # ResNet baseline
    python -m src.evaluation.inference \
        --config configs/resnet_cadsynth.yaml \
        --checkpoint results/checkpoints/resnet_cadsynth.pth \
        --mode semantic --output results/predictions/resnet_cadsynth.json

    # Only the held-out test models (default is the WHOLE corpus, which
    # includes the models the checkpoint trained on)
    python -m src.evaluation.inference ... --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ..data.brep_dataset import PreExtractedBRepDataset, load_split
from ..data.collate import collate_fn_brep
from ..data.uv_dataset import UVGridLabelDataset
from ..models.encoders import (
    BottomLabelEncoder,
    BRepEncoder,
    LabelEncoder,
    SemanticSegmentationModel,
)
from ..models.link_predictor import GraphLinkPredictor
from ..models.resnet_encoder import CrossModalAlignmentModel
from ..utils.checkpoint import load_checkpoint
from ..utils.class_weights import required_bias_type
from ..utils.provenance import config_hash


def load_config(path):
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


#  Dataset construction


def _resolve_data_view(cfg, *, where):
    """Resolve the dataset kwargs that decide what the network sees.

    Returns the same kwargs the TRAINERS build from the same config
    (src/training/train_{semantic,unified,link}.py), resolved with the same
    expressions, so that running inference with a checkpoint's own training
    config reproduces the view that checkpoint was fitted on.

    On the defaults that remain: `attributes` falls back to "scaled" and
    `one_hot_type`/`lambda_*` to the class defaults because that is exactly
    what the four trainers do. Matching them is not a guess -- it reproduces
    the trainer's own resolution of the same file. The expressions must stay
    identical to the trainers'; if a trainer ever stops defaulting, this must
    stop too.

    `bias.type` has no default. `required_bias_type` refuses to guess, and
    inference must refuse identically.

    Raises:
        ValueError: If the config asks for standardized attributes but names no
            norm-stats file.
        KeyError: From `required_bias_type`, if `bias.type` is absent.
    """
    ds_cfg = cfg["dataset"]
    bias_cfg = cfg.get("bias", {})

    attributes = ds_cfg.get("attributes", "scaled")
    norm_stats = ds_cfg.get("norm_stats")

    if attributes == "standardized" and not norm_stats:
        raise ValueError(
            f"{where}: dataset.attributes is 'standardized' but "
            f"dataset.norm_stats is not set.\n"
            f"Standardisation needs the mean and std this checkpoint was "
            f"trained with. There is no safe fallback: running the model on "
            f"the raw, unstandardised attributes instead would shift every "
            f"face and edge input by the whole of that mean and std, and "
            f"nothing downstream would raise -- the prediction file would "
            f"carry confidences and ground truths and look entirely normal. "
            f"That silent fallback is the defect this guard exists to stop.\n"
            f"Set dataset.norm_stats to the statistics the TRAINING run used "
            f"(datasets/splits/<corpus>_norm_train.json), or, if you really "
            f"want the unstandardised view, say so explicitly with "
            f"dataset.attributes: raw -- and expect different numbers.")

    return dict(
        json_dir=ds_cfg["json_dir"],
        n_classes=ds_cfg.get("n_classes", 25),
        selected_classes=ds_cfg.get("selected_classes"),
        attributes=attributes,
        norm_stats=norm_stats,
        # Forwarded so CHECKS 2 and 3 of the norm-stats provenance guard in
        # src/data/brep_dataset.py can run: they compare the statistics against
        # the split they claim to have been measured on, and they cannot do
        # that unless they are told which split is in use. Falsy when the
        # config names none, where that guard skips them.
        split_file=ds_cfg.get("split_file"),
        bias_type=required_bias_type(bias_cfg, where=where),
        lambda_adj=bias_cfg.get("lambda_adj", 1.0),
        lambda_sim=bias_cfg.get("lambda_sim", 1.0),
        one_hot_type=ds_cfg.get("one_hot_type", False),
    )


# Keys that a checkpoint COULD carry to name the data view it was trained on.
# Nothing writes them today -- see `_verify_against_checkpoint`.
_VIEW_KEYS_IN_PROVENANCE = ("attributes", "bias_type")


def _verify_against_checkpoint(ckpt, cfg, ds_kwargs, *, label):
    """Cross-check the resolved data view against what the checkpoint records.

    What the checkpoint actually records:

        `src/utils/checkpoint.py:save_checkpoint` writes model / label_encoder /
        bottom_encoder / criterion / history / n_classes / epoch /
        optimal_threshold, plus whatever a trainer passes as `extra`. The
        trainers pass `provenance`, and `build_provenance` records seed, epoch,
        selected_epoch, stopping_reason, use_class_weights,
        epoch_budget_configured, config_hash, code_fingerprint, git_commit,
        gpu, wall_time_s, torch, python, platform, recorded_at.

        NONE of those names the attribute view, the norm-stats file or the bias
        type. A checkpoint therefore CANNOT say what it was trained on; the only
        handle is `config_hash`, an opaque digest of the whole config dict.

    So this does what it can, and says plainly when it cannot do more:

    * If a checkpoint ever does record `attributes` / `bias_type` in its
      provenance block, a disagreement with the config is FATAL here.
    * Otherwise, if its `config_hash` equals this config's, the view is proven
      to be the training view -- same config in, same kwargs out.
    * Otherwise the view is UNVERIFIED, and that word goes into the prediction
      file rather than nowhere.

    A differing config hash is not made fatal: an honest inference run may
    legitimately change `paths`, `device` or `experiment_name`, and a guard that
    stops good runs is how guards get switched off. It is recorded instead, so
    a prediction file states whether its inputs were verified or merely assumed.

    Returns:
        A JSON-serialisable record of the cross-check, for the output file.

    Raises:
        ValueError: If the checkpoint names a data view and it disagrees.
    """
    prov = ckpt.get("provenance") or {}
    # `k in ds_kwargs` keeps the ResNet path out of this: it passes {} because
    # `UVGridLabelDataset` has no attribute view to disagree about, and a
    # recorded key would otherwise be compared against nothing.
    recorded = {k: prov[k] for k in _VIEW_KEYS_IN_PROVENANCE
                if prov.get(k) is not None and k in ds_kwargs}

    mismatch = {k: (v, ds_kwargs[k]) for k, v in recorded.items()
                if v != ds_kwargs[k]}
    if mismatch:
        lines = "\n".join(
            f"    {k}: checkpoint was trained with {was!r}, "
            f"this config asks for {now!r}"
            for k, (was, now) in sorted(mismatch.items()))
        raise ValueError(
            f"{label}: the checkpoint records the data view it was trained on, "
            f"and this config does not match it:\n{lines}\n"
            f"Running the model on a different view of the data gives wrong "
            f"predictions that look normal. Use the config this checkpoint was "
            f"trained under.")

    trained_hash = prov.get("config_hash")
    now_hash = config_hash(cfg)

    if recorded:
        status = "verified_from_checkpoint"
        detail = (f"checkpoint provenance names {sorted(recorded)} and they "
                  f"match this config")
    elif trained_hash and trained_hash == now_hash:
        status = "verified_by_config_hash"
        detail = (f"config_hash {now_hash} matches the training run, so this "
                  f"IS the config the checkpoint was trained under")
    elif trained_hash:
        status = "unverified_config_differs"
        detail = (
            f"this config hashes to {now_hash}; the checkpoint was trained "
            f"under {trained_hash}. The difference may be harmless (paths, "
            f"device, experiment_name) or it may be the attribute view itself "
            f"-- nothing in the checkpoint records that, so it cannot be told "
            f"apart from here")
    else:
        status = "unverifiable_no_provenance"
        detail = (
            "this checkpoint carries no provenance block at all, so neither "
            "its data view nor its config can be checked from here")

    print(f"\n  data view cross-check [{label}]: {status}")
    print(f"    {detail}")
    if status != "verified_from_checkpoint":
        print("    NOTE: checkpoints do not record dataset.attributes, "
              "dataset.norm_stats or bias.type. The config passed to this "
              "script is the ONLY statement of what the model was trained on.")

    return {
        "checkpoint": label,
        "status": status,
        "detail": detail,
        "training_config_hash": trained_hash,
        "inference_config_hash": now_hash,
        "data_view_recorded_in_checkpoint": sorted(recorded) or None,
    }


def _build_brep_dataset(cfg, ckpt, *, where, split=None, verbose=True):
    """Build the B-Rep dataset for an inference run, and describe it.

    Args:
        cfg: The config the checkpoint was trained under.
        ckpt: The loaded checkpoint, for the cross-check.
        where: Caller name, quoted in every error raised from here.
        split: ``"train" | "val" | "test"`` to score only that partition, or
            None to score the whole corpus.
        verbose: Passed to the dataset's own loading print.

    Returns:
        ``(dataset, meta)`` -- `meta` is a JSON-serialisable record of exactly
        which inputs produced the predictions, written into the output file.

    Raises:
        ValueError: From `_resolve_data_view` / `_verify_against_checkpoint`,
            or if `split` is asked for without a `dataset.split_file`.
    """
    ds_kwargs = _resolve_data_view(cfg, where=where)
    check = _verify_against_checkpoint(ckpt, cfg, ds_kwargs, label=where)

    split_file = ds_kwargs["split_file"]
    if split is None:
        model_ids = None
        # Said out loud because the output file carries `ground_truth` next to
        # every prediction, so it is one pandas line away from being read as an
        # accuracy -- an accuracy over the models the checkpoint trained on.
        scored = (f"FULL corpus at {ds_kwargs['json_dir']} -- train, val and "
                  f"test models together; pass --split test for the held-out "
                  f"models only")
    else:
        if not split_file:
            raise ValueError(
                f"{where}: --split {split!r} was requested but the config sets "
                f"no dataset.split_file, so there is no record of which models "
                f"are in that partition. Point dataset.split_file at the split "
                f"this checkpoint was trained with "
                f"(datasets/splits/*_70_10_20_seed42.json), or drop --split to "
                f"score the whole corpus.")
        model_ids = load_split(split_file, split)
        scored = f"{split} partition of {split_file} ({len(model_ids)} models)"

    print(f"    attributes={ds_kwargs['attributes']}  "
          f"norm_stats={ds_kwargs['norm_stats']}  "
          f"bias={ds_kwargs['bias_type']}  "
          f"one_hot_type={ds_kwargs['one_hot_type']}")
    print(f"    scoring: {scored}")

    dataset = PreExtractedBRepDataset(
        model_ids=model_ids, verbose=verbose, **ds_kwargs)

    meta = {
        "dataset_class": "PreExtractedBRepDataset",
        "attributes": ds_kwargs["attributes"],
        "norm_stats": (str(ds_kwargs["norm_stats"])
                       if ds_kwargs["norm_stats"] else None),
        "split_file": str(split_file) if split_file else None,
        "bias_type": ds_kwargs["bias_type"],
        "lambda_adj": ds_kwargs["lambda_adj"],
        "lambda_sim": ds_kwargs["lambda_sim"],
        "one_hot_type": ds_kwargs["one_hot_type"],
        "scored": scored,
        "n_models": len(dataset),
        "checkpoint_cross_check": check,
    }
    return dataset, meta


def _require_matching_n_classes(dataset, label_encoder, *, where):
    """Refuse to score when the label encoder and the dataset disagree.

    The prototypes are ``label_encoder(arange(dataset.n_classes))``. If the
    dataset has FEWER classes than the encoder -- a `selected_classes` list in
    the inference config that is shorter than the one used in training -- the
    argmax silently runs over a prefix of the prototypes and every face is
    forced into a subset of the labels. Nothing raises; the confidences look
    fine. (The opposite direction already dies inside `F.one_hot`, loudly.)
    """
    if dataset.n_classes != label_encoder.n_classes:
        raise ValueError(
            f"{where}: the dataset yields {dataset.n_classes} classes but the "
            f"checkpoint's label encoder was trained with "
            f"{label_encoder.n_classes}.\n"
            f"Predictions are an argmax over one prototype per class, so a "
            f"mismatch silently scores against the wrong set of prototypes. "
            f"Check dataset.n_classes and dataset.selected_classes against the "
            f"config this checkpoint was trained under.")


def _require_tuned_threshold(ckpt, *, where):
    """Read the instance-similarity threshold; refuse to invent one.

    `src/training/train_unified.py` saves the validation-tuned threshold.
    When the checkpoint has none, this refuses to write a prediction file.
    """
    threshold = ckpt.get("optimal_threshold")
    if threshold is None:
        raise ValueError(
            f"{where}: this checkpoint stores no 'optimal_threshold'.\n"
            f"Unified inference thresholds a cosine similarity to decide "
            f"whether two faces belong to the same instance, and that "
            f"threshold is tuned on the validation split by "
            f"src/training/train_unified.py. Substituting 0.5 -- which is what "
            f"this code did until 2026-09-22 -- writes a made-up number into "
            f"the prediction file under the name of a tuned one, and every "
            f"'predicted_same_instance' beneath it is unfounded.\n"
            f"Use a checkpoint from train_unified.py, or score the instance "
            f"head with src/evaluation/evaluate_unified.py, which reports the "
            f"threshold-free ROC-AUC when no tuned threshold exists.")
    return float(threshold)


#  Semantic inference (Graph Transformer)

def _infer_semantic_graph(cfg, ckpt, device, split=None):
    """Run semantic inference with a Graph Transformer model.

    Returns:
        ``(predictions, meta)`` -- a list of per-sample prediction dicts, and a
        record of the data view they were produced from.
    """
    m_cfg = cfg.get("model", cfg.get("semantic_model", {}))
    model_type = m_cfg.get("type", "semantic_segmentation")
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

    n_classes = ckpt.get("n_classes", cfg["dataset"].get("n_classes", 25))
    le_cfg = cfg.get("label_encoder", {})
    label_encoder = LabelEncoder(
        n_classes=n_classes,
        embedding_dim=le_cfg.get("embedding_dim", m_cfg.get("d_model", 128)),
    ).to(device)
    if "label_encoder" in ckpt:
        label_encoder.load_state_dict(ckpt["label_encoder"])
    label_encoder.eval()

    dataset, meta = _build_brep_dataset(
        cfg, ckpt, where="inference(mode=semantic)", split=split,
    )
    _require_matching_n_classes(
        dataset, label_encoder, where="inference(mode=semantic)")
    n_classes = dataset.n_classes

    # No `**loader_kwargs(...)`: shuffle=False +
    # num_workers=0 makes both kwargs inert, and this script has no
    # --seed at all. Argument and gate:
    # tests/test_eval_dataloader_seeding.py.
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_brep,
    )

    all_cls = torch.arange(n_classes, device=device)
    results = []

    with torch.no_grad():
        prototypes = F.normalize(label_encoder(all_cls), dim=1)

        for idx, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            embeddings = model(
                batch["faces_uv"], batch["faces_attr"],
                batch["edges_attr"], batch["bias"],
                face_mask=batch["face_mask"],
                edge_mask=batch["edge_mask"],
            )

            mask = batch["face_mask"][0]
            n_valid = int(mask.sum().item())
            model_id = batch["model_ids"][0]
            if n_valid == 0:
                results.append({"sample_idx": idx, "model_id": model_id,
                                "faces": []})
                continue

            emb = F.normalize(embeddings[0][mask], dim=1)
            scores = torch.matmul(emb, prototypes.T)
            preds = scores.argmax(dim=1)
            gt = batch["semantic_labels"][0][mask]

            faces = []
            for fi in range(n_valid):
                faces.append({
                    "face_idx": fi,
                    "predicted_label": int(preds[fi].item()),
                    "ground_truth": int(gt[fi].item()),
                    "confidence": float(scores[fi].max().item()),
                    "scores": scores[fi].cpu().numpy().tolist(),
                })

            results.append({"sample_idx": idx, "model_id": model_id,
                            "n_faces": n_valid, "faces": faces})

    return results, meta


#  Semantic inference (ResNet)

def _infer_semantic_resnet(cfg, ckpt, device, split=None):
    """Run semantic inference with a ResNet baseline model.

    `UVGridLabelDataset` reads only the UV grids, and has no `attributes` or
    `norm_stats`. `--split` is honoured, because subsetting by MODEL id
    before flattening to faces keeps one part's faces from straddling the
    train/test boundary.

    Returns:
        ``(predictions, meta)`` -- a list of per-face prediction dicts, and a
        record of the data view they were produced from.
    """
    m_cfg = cfg.get("model", {})
    n_classes = ckpt.get("n_classes", cfg["dataset"].get("n_classes", 25))

    model = CrossModalAlignmentModel(
        n_classes=n_classes,
        embedding_dim=m_cfg.get("embedding_dim", 512),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds_cfg = cfg["dataset"]
    split_file = ds_cfg.get("split_file")
    check = _verify_against_checkpoint(
        ckpt, cfg, {}, label="inference(mode=semantic,resnet)")

    if split is None:
        model_ids = None
        scored = (f"FULL corpus at {ds_cfg['json_dir']} -- train, val and test "
                  f"models together; pass --split test for the held-out "
                  f"models only")
    else:
        if not split_file:
            raise ValueError(
                f"inference(mode=semantic,resnet): --split {split!r} was "
                f"requested but the config sets no dataset.split_file, so "
                f"there is no record of which models are in that partition. "
                f"Set dataset.split_file, or drop --split.")
        model_ids = load_split(split_file, split)
        scored = f"{split} partition of {split_file} ({len(model_ids)} models)"
    print(f"    scoring: {scored}")

    dataset = UVGridLabelDataset(
        json_dir=ds_cfg["json_dir"],
        n_classes=ds_cfg.get("n_classes", 25),
        model_ids=model_ids,
    )
    meta = {
        "dataset_class": "UVGridLabelDataset",
        "attributes": "n/a -- this model reads UV grids, not face attributes",
        "norm_stats": None,
        "split_file": str(split_file) if split_file else None,
        "bias_type": "n/a -- no graph bias in the ResNet baseline",
        "scored": scored,
        "n_faces": len(dataset),
        "checkpoint_cross_check": check,
    }

    # No `**loader_kwargs(...)`: shuffle=False +
    # num_workers=0 makes both kwargs inert, and this script has no
    # --seed at all. Argument and gate:
    # tests/test_eval_dataloader_seeding.py.
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_cls = torch.arange(n_classes, device=device)
    results = []

    with torch.no_grad():
        prototypes = F.normalize(model.label_encoder(all_cls), dim=1)

        for uv_grids, labels in loader:
            uv_grids = uv_grids.to(device)
            labels = labels.to(device)

            emb = F.normalize(model.uv_encoder(uv_grids), dim=1)
            scores = torch.matmul(emb, prototypes.T)
            preds = scores.argmax(dim=1)

            for i in range(len(labels)):
                results.append({
                    "predicted_label": int(preds[i].item()),
                    "ground_truth": int(labels[i].item()),
                    "confidence": float(scores[i].max().item()),
                    "scores": scores[i].cpu().numpy().tolist(),
                })

    return results, meta


#  Link prediction inference

def _infer_link(cfg, link_ckpt, device, semantic_checkpoint_path=None,
                split=None):
    """Run two-stage (semantic + link prediction) inference.

    Returns:
        ``(predictions, meta)`` -- per-sample dicts with face labels and
        pairwise link predictions, and a record of the data view.
    """
    # Resolve semantic checkpoint
    if semantic_checkpoint_path is None:
        semantic_checkpoint_path = link_ckpt.get(
            "semantic_checkpoint",
            cfg.get("semantic_checkpoint"),
        )
    if semantic_checkpoint_path is None:
        raise ValueError(
            "Cannot resolve semantic checkpoint for link inference."
        )
    sem_ckpt = load_checkpoint(semantic_checkpoint_path, device=device)

    # Frozen semantic model
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

    # Label encoder for semantic predictions
    n_classes = sem_ckpt.get("n_classes", cfg["dataset"].get("n_classes", 25))
    le_cfg = cfg.get("label_encoder", {})
    label_encoder = LabelEncoder(
        n_classes=n_classes,
        embedding_dim=le_cfg.get("embedding_dim", sem_cfg.get("d_model", 128)),
    ).to(device)
    if "label_encoder" in sem_ckpt:
        label_encoder.load_state_dict(sem_ckpt["label_encoder"])
    label_encoder.eval()

    # Link predictor
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

    # Dataset
    #
    # The view must be the one the FROZEN SEMANTIC encoder was trained on: the
    # link head consumes that encoder's embeddings, so feeding it raw inputs
    # corrupts the link predictions through the same bad embeddings that
    # corrupt the semantic ones. Both checkpoints are cross-checked below.
    dataset, meta = _build_brep_dataset(
        cfg, link_ckpt, where="inference(mode=link)", split=split,
    )
    ds_kwargs_for_check = _resolve_data_view(cfg, where="inference(mode=link)")
    meta["semantic_checkpoint"] = str(semantic_checkpoint_path)
    meta["semantic_checkpoint_cross_check"] = _verify_against_checkpoint(
        sem_ckpt, cfg, ds_kwargs_for_check,
        label="inference(mode=link, frozen semantic checkpoint)")
    _require_matching_n_classes(
        dataset, label_encoder, where="inference(mode=link)")
    n_classes = dataset.n_classes

    # No `**loader_kwargs(...)`: shuffle=False +
    # num_workers=0 makes both kwargs inert, and this script has no
    # --seed at all. Argument and gate:
    # tests/test_eval_dataloader_seeding.py.
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_brep,
    )

    all_cls = torch.arange(n_classes, device=device)
    results = []

    with torch.no_grad():
        prototypes = F.normalize(label_encoder(all_cls), dim=1)

        for idx, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            embeddings = semantic_model(
                batch["faces_uv"], batch["faces_attr"],
                batch["edges_attr"], batch["bias"],
                face_mask=batch["face_mask"],
                edge_mask=batch["edge_mask"],
            )

            mask = batch["face_mask"][0]
            n_valid = int(mask.sum().item())
            model_id = batch["model_ids"][0]
            if n_valid == 0:
                results.append({"sample_idx": idx, "model_id": model_id,
                                "faces": [], "links": []})
                continue

            # Semantic predictions
            emb = F.normalize(embeddings[0][mask], dim=1)
            sem_scores = torch.matmul(emb, prototypes.T)
            sem_preds = sem_scores.argmax(dim=1)
            gt_sem = batch["semantic_labels"][0][mask]

            faces = []
            for fi in range(n_valid):
                faces.append({
                    "face_idx": fi,
                    "predicted_label": int(sem_preds[fi].item()),
                    "ground_truth": int(gt_sem[fi].item()),
                })

            # Link predictions
            link_logits = link_model(
                embeddings, bias=batch["bias"], mask=batch["face_mask"],
            )
            valid_idx = torch.nonzero(mask).squeeze(1)
            pair_logits = link_logits[0][valid_idx[:, None], valid_idx]
            pair_probs = F.softmax(pair_logits, dim=-1)[..., 1]
            pair_preds = pair_logits.argmax(dim=-1)

            gt_inst = None
            if "instance_labels" in batch:
                gt_inst = batch["instance_labels"][0][
                    valid_idx[:, None], valid_idx
                ]

            links = []
            for fi in range(n_valid):
                for fj in range(n_valid):
                    if fi == fj:
                        continue
                    entry = {
                        "face_i": fi,
                        "face_j": fj,
                        "predicted_link": int(pair_preds[fi, fj].item()),
                        "confidence": float(pair_probs[fi, fj].item()),
                    }
                    if gt_inst is not None:
                        entry["ground_truth"] = int(gt_inst[fi, fj].item())
                    links.append(entry)

            results.append({
                "sample_idx": idx,
                "model_id": model_id,
                "n_faces": n_valid,
                "faces": faces,
                "links": links,
            })

    return results, meta


#  Unified inference

def _infer_unified(cfg, ckpt, device, split=None):
    """Run inference with the unified model.

    Returns:
        ``(predictions, meta)`` -- per-sample dicts with semantic labels,
        bottom-face labels and pairwise instance similarity scores, and a
        record of the data view.
    """
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
    le_cfg = cfg.get("label_encoder", {})
    embedding_dim = le_cfg.get("embedding_dim", m_cfg.get("d_model", 128))

    label_encoder = LabelEncoder(
        n_classes=n_classes, embedding_dim=embedding_dim,
    ).to(device)
    if "label_encoder" in ckpt:
        label_encoder.load_state_dict(ckpt["label_encoder"])
    label_encoder.eval()

    bottom_encoder = BottomLabelEncoder(
        embedding_dim=embedding_dim,
    ).to(device)
    if "bottom_encoder" in ckpt:
        bottom_encoder.load_state_dict(ckpt["bottom_encoder"])
    bottom_encoder.eval()

    optimal_threshold = _require_tuned_threshold(
        ckpt, where="inference(mode=unified)")

    dataset, meta = _build_brep_dataset(
        cfg, ckpt, where="inference(mode=unified)", split=split,
    )
    _require_matching_n_classes(
        dataset, label_encoder, where="inference(mode=unified)")
    meta["optimal_threshold"] = optimal_threshold
    meta["optimal_threshold_source"] = (
        "tuned on the validation split by train_unified.py and read from the "
        "checkpoint")
    n_classes = dataset.n_classes

    # No `**loader_kwargs(...)`: shuffle=False +
    # num_workers=0 makes both kwargs inert, and this script has no
    # --seed at all. Argument and gate:
    # tests/test_eval_dataloader_seeding.py.
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_brep,
    )

    all_cls = torch.arange(n_classes, device=device)
    bot_cls = torch.arange(2, device=device)
    results = []

    with torch.no_grad():
        sem_prototypes = F.normalize(label_encoder(all_cls), dim=1)
        bot_prototypes = F.normalize(bottom_encoder(bot_cls), dim=1)

        for idx, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            embeddings = model(
                batch["faces_uv"], batch["faces_attr"],
                batch["edges_attr"], batch["bias"],
                face_mask=batch["face_mask"],
                edge_mask=batch["edge_mask"],
            )

            mask = batch["face_mask"][0]
            n_valid = int(mask.sum().item())
            model_id = batch["model_ids"][0]
            if n_valid == 0:
                results.append({
                    "sample_idx": idx, "model_id": model_id,
                    "faces": [], "instance_pairs": [],
                })
                continue

            emb = F.normalize(embeddings[0][mask], dim=1)

            # Semantic predictions
            sem_scores = torch.matmul(emb, sem_prototypes.T)
            sem_preds = sem_scores.argmax(dim=1)
            gt_sem = batch["semantic_labels"][0][mask]

            # Bottom-face predictions
            bot_scores = torch.matmul(emb, bot_prototypes.T)
            bot_preds = bot_scores.argmax(dim=1)
            gt_bot = (
                batch["bottom_labels"][0][mask]
                if "bottom_labels" in batch else None
            )

            faces = []
            for fi in range(n_valid):
                entry = {
                    "face_idx": fi,
                    "predicted_label": int(sem_preds[fi].item()),
                    "gt_semantic": int(gt_sem[fi].item()),
                    "predicted_bottom": int(bot_preds[fi].item()),
                }
                if gt_bot is not None:
                    entry["gt_bottom"] = int(gt_bot[fi].item())
                faces.append(entry)

            # Instance similarity predictions
            sim_matrix = torch.matmul(emb, emb.T)
            instance_preds = (sim_matrix >= optimal_threshold).int()

            gt_inst = None
            if "instance_labels" in batch:
                valid_idx = torch.nonzero(mask).squeeze(1)
                gt_inst = batch["instance_labels"][0][
                    valid_idx[:, None], valid_idx
                ]

            instance_pairs = []
            for fi in range(n_valid):
                for fj in range(fi + 1, n_valid):
                    entry = {
                        "face_i": fi,
                        "face_j": fj,
                        "similarity": float(sim_matrix[fi, fj].item()),
                        "predicted_same_instance": int(
                            instance_preds[fi, fj].item()
                        ),
                    }
                    if gt_inst is not None:
                        entry["ground_truth"] = int(gt_inst[fi, fj].item())
                    instance_pairs.append(entry)

            results.append({
                "sample_idx": idx,
                "model_id": model_id,
                "n_faces": n_valid,
                "optimal_threshold": float(optimal_threshold),
                "faces": faces,
                "instance_pairs": instance_pairs,
            })

    return results, meta


#  Main inference driver

def run_inference(cfg, checkpoint_path, mode, output_path,
                  semantic_checkpoint_path=None, split=None):
    """Run batch inference and save predictions to a JSON file.

    Args:
        cfg: Parsed YAML configuration dict. This MUST be the config the
            checkpoint was trained under -- it is the only statement anywhere
            of which attribute view and bias the model was fitted on.
        checkpoint_path: Path to the model checkpoint.
        mode: One of ``"semantic"``, ``"link"``, ``"unified"``.
        output_path: Destination path for the JSON prediction file.
        semantic_checkpoint_path: Required for ``mode="link"`` when the
            frozen semantic checkpoint cannot be resolved automatically.
        split: ``"train" | "val" | "test"`` to score only that partition of
            ``dataset.split_file``; None scores the whole corpus.
    """
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint_path, device=device)

    print("\n" + "=" * 70)
    print(f"BATCH INFERENCE — mode={mode}")
    print("=" * 70)

    if mode == "semantic":
        m_cfg = cfg.get("model", {})
        if m_cfg.get("type") == "resnet_encoder":
            predictions, data_view = _infer_semantic_resnet(
                cfg, ckpt, device, split=split)
        else:
            predictions, data_view = _infer_semantic_graph(
                cfg, ckpt, device, split=split)

    elif mode == "link":
        predictions, data_view = _infer_link(
            cfg, ckpt, device,
            semantic_checkpoint_path=semantic_checkpoint_path,
            split=split,
        )

    elif mode == "unified":
        predictions, data_view = _infer_unified(cfg, ckpt, device, split=split)

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use semantic, link, or unified.")

    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment_name": cfg.get("experiment_name", "inference"),
        "mode": mode,
        "checkpoint": str(checkpoint_path),
        # Which inputs produced these numbers, stated
        # in the file that carries them.
        "data_view": data_view,
        "n_samples": len(predictions),
        "predictions": predictions,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{len(predictions)} predictions saved to {output_path}")
    print(f"  inputs: attributes={data_view['attributes']}, "
          f"bias={data_view['bias_type']}, "
          f"cross-check={data_view['checkpoint_cross_check']['status']}")


#  CLI entry point                                                       #

def main():
    parser = argparse.ArgumentParser(
        description="Batch inference and prediction export to JSON",
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
        "--mode", type=str, required=True,
        choices=["semantic", "link", "unified"],
        help="Inference mode: semantic, link, or unified",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: results/predictions/<experiment>.json)",
    )
    parser.add_argument(
        "--semantic-checkpoint", type=str, default=None,
        help="Semantic checkpoint for link mode "
             "(resolved from link checkpoint or config if omitted)",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        choices=["train", "val", "test"],
        help="Score only this partition of dataset.split_file. Omitted, the "
             "whole corpus is scored -- including the models the checkpoint "
             "trained on, whose ground truths are written next to the "
             "predictions in the output file.",
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

    output_path = args.output
    if output_path is None:
        experiment = cfg.get("experiment_name", "predictions")
        output_path = f"results/predictions/{experiment}.json"

    run_inference(
        cfg, checkpoint_path, args.mode, output_path,
        semantic_checkpoint_path=args.semantic_checkpoint,
        split=args.split,
    )


if __name__ == "__main__":
    main()
