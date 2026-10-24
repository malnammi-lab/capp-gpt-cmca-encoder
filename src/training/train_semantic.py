"""
Stage 1: Semantic Segmentation Training with Contrastive Alignment.

Trains a SemanticSegmentationModel (Graph Transformer encoder) jointly
with a LabelEncoder using PrototypeInfoNCELoss.  Produces per-face
embeddings that cluster by semantic class.

Usage:
    python -m src.training.train_semantic --config configs/cadsynth_bias_both.yaml
"""

# Required so the `int | None` annotations below parse on
# Python 3.9, which is the default interpreter on the Windows machine even though
# the model environment is 3.13. Without it, `python -m src.training.*` dies at
# import time with "unsupported operand type(s) for |".
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split

from ..data.brep_dataset import PreExtractedBRepDataset, load_split
from ..data.collate import collate_fn_brep
from ..losses.infonce import PrototypeInfoNCELoss
from ..models.ce_encoder import SemanticClassificationModel
from ..models.encoders import LabelEncoder, SemanticSegmentationModel
from ..utils.checkpoint import save_checkpoint
from ..utils.class_weights import (class_weighting_enabled,
                                   compute_class_weights,
                                   required_bias_type,
                                   required_training_budget)
from ..utils.provenance import build_provenance, stamp_selection
from ..utils.seeding import loader_kwargs, seed_scoped, set_global_seed
from ..utils.visualization import plot_training_curves


def load_config(path):
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _gather_valid(embeddings, labels, face_mask):
    """Gather non-padded face embeddings and their labels across a batch.

    Args:
        embeddings: (B, N, D) face embeddings from the model.
        labels: (B, N) per-face labels.
        face_mask: (B, N) boolean mask.

    Returns:
        Tuple of (flat_embeddings, flat_labels) with shape (V, D) and (V,).
        Returns (None, None) when no valid faces exist in the batch.
    """
    emb_list, lbl_list = [], []
    for b in range(embeddings.shape[0]):
        mask = face_mask[b]
        if mask.sum() > 0:
            emb_list.append(embeddings[b][mask])
            lbl_list.append(labels[b][mask])
    if not emb_list:
        return None, None
    return torch.cat(emb_list, dim=0), torch.cat(lbl_list, dim=0)


def _prototype_accuracy(embeddings, labels, label_encoder, n_classes, device):
    """Compute accuracy by matching face embeddings to nearest class prototype."""
    all_cls = torch.arange(n_classes, device=device)
    prototypes = F.normalize(label_encoder(all_cls), dim=1)
    emb_norm = F.normalize(embeddings, dim=1)
    preds = torch.matmul(emb_norm, prototypes.T).argmax(dim=1)
    return (preds == labels).sum().item()


def train_semantic_model(cfg, resume: str = None, seed_override: int | None = None):
    """Run the full Stage 1 semantic segmentation training loop.

    Args:
        cfg: Resolved config dict.
        resume: Optional checkpoint to resume from.
        seed_override: Seed from ``--seed`` or from ``runners/run_all.py``.
            When given it beats ``cfg["seed"]`` and switches on seed-scoped
            output paths, so one config file serves all five seeds of a tier
            without the runs overwriting each other.
    """
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    # set_global_seed covers torch, numpy, random, and cudnn, and raises
    # when no seed is given.
    seed = seed_override if seed_override is not None else cfg.get("seed")
    seed = set_global_seed(seed)          # raises when there is no seed at all

    ds_cfg = cfg["dataset"]
    bias_cfg = cfg.get("bias", {})
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
        bias_type=required_bias_type(bias_cfg, where="train_semantic"),
        lambda_adj=bias_cfg.get("lambda_adj", 1.0),
        lambda_sim=bias_cfg.get("lambda_sim", 1.0),
        one_hot_type=ds_cfg.get("one_hot_type", False),
    )

    train_ratio = ds_cfg.get("train_split", 0.7)
    val_ratio = ds_cfg.get("val_split", 0.15)
    # `split_file` is read above, before ds_kwargs, so it can be forwarded into
    # the dataset for the norm-stats provenance check.

    if split_file:
        print(f"\nSplit file: {split_file}")
        train_ids = load_split(split_file, "train")
        test_ids = load_split(split_file, "test")
        # A validation partition is required.
        # Every cell selects the best-validation-loss checkpoint,
        # of every tier, so a split file with no "val" key cannot produce a
        # quotable run. Falling back to the epoch budget would produce one that
        # LOOKS quotable, which is the failure mode worth stopping.
        try:
            val_ids = load_split(split_file, "val")
        except KeyError:
            raise KeyError(
                f"train_semantic: split file {split_file} has no 'val' "
                f"partition. Every run selects the best-"
                f"validation-loss checkpoint in every run, so there is no "
                f"no-validation path any more. Point dataset.split_file at the "
                f"70/10/20 file (datasets/splits/*_70_10_20_seed42.json).")
        if not val_ids:
            raise ValueError(
                f"train_semantic: split file {split_file} has an EMPTY 'val' "
                f"partition. A real held-out validation "
                f"set; an empty one would silently restore last-epoch selection.")

        train_ds = PreExtractedBRepDataset(model_ids=train_ids, **ds_kwargs)
        test_ds = PreExtractedBRepDataset(model_ids=test_ids, verbose=False, **ds_kwargs)
        val_ds = PreExtractedBRepDataset(model_ids=val_ids, verbose=False,
                                         **ds_kwargs)
        use_validation = True
        n_classes = train_ds.n_classes

        st = train_ds.get_statistics()
        print(f"Dataset split: {len(train_ds)} train / "
              f"{len(val_ds)} val / {len(test_ds)} test")
        print(f"  attributes={ds_kwargs['attributes']}  "
              f"bias={ds_kwargs['bias_type']}  attr_dim={st['attr_dim']}  "
              f"train adjacency density {st['adjacency_density_mean']:.3f}")
        dataset = train_ds  # for anything downstream that inspects `dataset`
    else:
        dataset = PreExtractedBRepDataset(**ds_kwargs)
        n_classes = dataset.n_classes
        use_validation = val_ratio > 0

    if split_file:
        pass  # already built above
    elif use_validation:
        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            train_ds, val_ds, test_ds = random_split(
                dataset, [train_size, val_size, test_size], generator=generator
            )
        else:
            train_ds, val_ds, test_ds = random_split(
                dataset, [train_size, val_size, test_size]
            )
        print(f"\nDataset split: {train_size} train / {val_size} val / {test_size} test")
    else:
        raise ValueError(
            "train_semantic: dataset.val_split is 0 and there is no "
            "dataset.split_file, so this config asks for training with no "
            "validation partition. Every run selects the "
            "best-validation-loss checkpoint in every run; there is no "
            "last-epoch path left. Use a 70/10/20 split file, or set "
            "dataset.val_split > 0 for a development run.")

    t_cfg = cfg["training"]
    batch_size = t_cfg.get("batch_size", 4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_brep,
        **loader_kwargs(seed, "train_loader"),
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_brep,
            **loader_kwargs(seed, "val_loader"),
        )
        if use_validation
        else None
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_brep,
        **loader_kwargs(seed, "test_loader"),
    )

    # `loss.type` selects the objective and therefore the head:
    #
    #   prototype_infonce (default) -- CMCA. Encoder emits per-face embeddings
    #                                  aligned against a LabelEncoder tower.
    #   cross_entropy               -- the ablation arm. Same encoder, plus a
    #                                  linear class head; no label tower.
    #
    # Only the objective and the head differ, so a gap between the two arms is
    # attributable to the objective alone. That is the whole point of the 2x2.
    m_cfg = cfg.get("model", {})
    l_cfg = cfg.get("loss", {})
    loss_type = l_cfg.get("type", "prototype_infonce")
    if loss_type not in ("prototype_infonce", "cross_entropy"):
        raise ValueError(
            f"loss.type must be prototype_infonce|cross_entropy, got {loss_type!r}")
    is_ce = loss_type == "cross_entropy"

    le_cfg = cfg.get("label_encoder", {})
    embedding_dim = le_cfg.get("embedding_dim", m_cfg.get("d_model", 128))

    if is_ce:
        model = SemanticClassificationModel(
            n_classes=n_classes,
            d_model=m_cfg.get("d_model", 128),
            n_heads=m_cfg.get("n_heads", 8),
            n_layers=m_cfg.get("n_layers", 4),
            dim_feedforward=m_cfg.get("dim_feedforward", 256),
            dropout=m_cfg.get("dropout", 0.1),
        ).to(device)
    else:
        model = SemanticSegmentationModel(
            d_model=m_cfg.get("d_model", 128),
            n_heads=m_cfg.get("n_heads", 8),
            n_layers=m_cfg.get("n_layers", 4),
            dim_feedforward=m_cfg.get("dim_feedforward", 256),
            dropout=m_cfg.get("dropout", 0.1),
        ).to(device)

    # Always constructed so .train()/.eval() call sites stay uniform, but for
    # the CE arm it is excluded from the optimizer and from the checkpoint --
    # an untrained label tower saved alongside a CE model would let
    # evaluate_semantic silently score it with meaningless prototypes.
    label_encoder = LabelEncoder(
        n_classes=n_classes, embedding_dim=embedding_dim,
    ).to(device)

    class_weights = None
    if class_weighting_enabled(l_cfg, where="train_semantic"):
        class_weights = compute_class_weights(train_ds, n_classes, device)

    if is_ce:
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    else:
        criterion = PrototypeInfoNCELoss(
            temperature=l_cfg.get("temperature", 0.07),
            class_weights=class_weights,
        ).to(device)

    def _loss(valid_out, valid_labels):
        """Loss for either objective. `valid_out` is logits (CE) or
        embeddings (CMCA) -- the training loop does not need to care."""
        if is_ce:
            return criterion(valid_out, valid_labels)
        return criterion(valid_out, valid_labels, label_encoder, n_classes)

    def _correct(valid_out, valid_labels):
        """Correct-prediction count for either objective."""
        if is_ce:
            return (valid_out.argmax(dim=1) == valid_labels).sum().item()
        return _prototype_accuracy(
            valid_out, valid_labels, label_encoder, n_classes, device)

    # The CE arm has no label tower to train, and nn.CrossEntropyLoss has no
    # parameters; the contrastive arm trains both (the loss owns a learnable
    # temperature).
    params = list(model.parameters())
    if not is_ce:
        params += list(label_encoder.parameters()) + list(criterion.parameters())
    optimizer = optim.Adam(
        params,
        lr=t_cfg.get("lr", 1e-4),
        weight_decay=t_cfg.get("weight_decay", 0.0),
    )

    # max_epochs and patience are required.
    max_epochs, patience = required_training_budget(
        t_cfg, where="train_semantic")
    grad_clip = t_cfg.get("grad_clip", 1.0)

    scheduler = None
    if t_cfg.get("scheduler") == "cosine_annealing":
        # T_max is the epoch cap.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs,
        )

    paths_cfg = cfg.get("paths", {})
    # Seed-scoped output paths. `seed_scoped` is a no-op when --seed is not given.
    ckpt_path = seed_scoped(
        paths_cfg.get("checkpoint", "results/checkpoints/semantic.pth"),
        seed_override)
    fig_dir = seed_scoped(
        paths_cfg.get("figures", "results/figures/semantic/"),
        seed_override, is_dir=True)

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1
    # The epoch whose weights are in ckpt_path, and how
    # the loop ended. Both are stamped onto the checkpoint after the loop by
    # stamp_selection; the cap audit reads them.
    best_epoch = None
    stopped_early = False

    # Resume restores the model and the optimizer.
    if resume:
        from ..utils.checkpoint import load_checkpoint
        ck = load_checkpoint(resume, device=device)
        model.load_state_dict(ck["model"])
        if "label_encoder" in ck and not is_ce:
            label_encoder.load_state_dict(ck["label_encoder"])
        if "criterion" in ck and not is_ce:
            criterion.load_state_dict(ck["criterion"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        else:
            print("  WARNING: checkpoint has no optimizer state (written before "
                  "resume support). Adam moments restart from zero; the first "
                  "few epochs after resuming will behave slightly differently.")
        if ck.get("history"):
            history = ck["history"]
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_val_loss = float(ck.get("best_val_loss", best_val_loss))
        # The resumed checkpoint IS the best-val
        # checkpoint so far, so its epoch is the selected epoch until beaten.
        best_epoch = int(ck.get("epoch", 0)) or None
        if start_epoch > max_epochs:
            print(f"Checkpoint is already at epoch {ck.get('epoch')} of "
                  f"{max_epochs}. Nothing to do.")
            return model, label_encoder, history
        print(f"Resumed from {resume} at epoch {ck.get('epoch')} -- "
              f"continuing {start_epoch}..{max_epochs}")

    print("\n" + "=" * 70)
    print("STAGE 1: Semantic Segmentation -- "
          + ("cross-entropy head (ablation arm)" if is_ce
             else "Contrastive Alignment (CMCA)"))
    print("=" * 70)

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        label_encoder.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            embeddings = model(
                batch["faces_uv"], batch["faces_attr"],
                batch["edges_attr"], batch["bias"],
                face_mask=batch["face_mask"],
                edge_mask=batch["edge_mask"],
            )

            valid_emb, valid_labels = _gather_valid(
                embeddings, batch["semantic_labels"], batch["face_mask"],
            )
            if valid_emb is None:
                continue

            loss = _loss(valid_emb, valid_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

            train_loss += loss.item() * len(valid_labels)
            with torch.no_grad():
                train_correct += _correct(valid_emb, valid_labels)
            train_total += len(valid_labels)

        ep_vloss = float("inf")
        ep_vacc = 0.0
        if val_loader is not None:
            model.eval()
            label_encoder.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

                    embeddings = model(
                        batch["faces_uv"], batch["faces_attr"],
                        batch["edges_attr"], batch["bias"],
                        face_mask=batch["face_mask"],
                        edge_mask=batch["edge_mask"],
                    )

                    valid_emb, valid_labels = _gather_valid(
                        embeddings, batch["semantic_labels"], batch["face_mask"],
                    )
                    if valid_emb is None:
                        continue

                    loss = _loss(valid_emb, valid_labels)
                    val_loss += loss.item() * len(valid_labels)
                    val_correct += _correct(valid_emb, valid_labels)
                    val_total += len(valid_labels)
            ep_vloss = val_loss / val_total if val_total > 0 else float("inf")
            ep_vacc = val_correct / val_total if val_total > 0 else 0.0

        if scheduler is not None:
            scheduler.step()

        ep_loss = train_loss / train_total if train_total > 0 else 0
        ep_acc = train_correct / train_total if train_total > 0 else 0

        history["loss"].append(ep_loss)
        history["acc"].append(ep_acc)
        history["val_loss"].append(ep_vloss)
        history["val_acc"].append(ep_vacc)

        if val_loader is not None:
            print(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Loss {ep_loss:.4f} | Acc {ep_acc:.2%} | "
                f"Val Loss {ep_vloss:.4f} | Val Acc {ep_vacc:.2%}"
            )
        else:
            print(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Loss {ep_loss:.4f} | Acc {ep_acc:.2%}"
            )

        # Save only when validation loss improves, and stop when patience runs out.
        if ep_vloss < best_val_loss:
            best_val_loss = ep_vloss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                ckpt_path,
                model,
                label_encoder=None if is_ce else label_encoder,
                criterion=None if is_ce else criterion,
                history=history,
                n_classes=n_classes,
                epoch=epoch,
                # seed / epoch / config hash / code
                # fingerprint / GPU / wall time -- required by
                # training-regiment.md section 1 for any quotable run.
                # selected_epoch / stopping_reason are
                # stamped after the loop, when they are knowable.
                extra={"optimizer": optimizer.state_dict(),
                       "best_val_loss": best_val_loss,
                       "provenance": build_provenance(
                           cfg, seed=seed, epoch=epoch,
                           wall_time_s=time.time() - t_start)},
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no val-loss improvement for {patience} epochs); "
                      f"selected epoch {best_epoch}")
                stopped_early = True
                break

    # Which checkpoint is being quoted, and why training
    # ended. "cap" means the loop used its whole budget with patience intact,
    # which the cap audit treats as a failed comparison, so the distinction is
    # recorded rather than inferred from the epoch number.
    stopping_reason = "early_stop" if stopped_early else "cap"
    if best_epoch is None:
        raise RuntimeError(
            "train_semantic: no epoch ever improved on an infinite validation "
            "loss, so no checkpoint was selected. The validation loader "
            "produced no usable batches -- check the 'val' partition of "
            f"{split_file!r}.")
    stamp_selection(ckpt_path, selected_epoch=best_epoch,
                    stopping_reason=stopping_reason)
    if stopping_reason == "cap":
        print(f"  NOTE: training ended at the max_epochs cap ({max_epochs}) "
              f"with patience intact. Under the regiment's cap audit this run "
              f"is NOT quotable until the comparison re-runs at a raised cap.")

    plot_training_curves(history, save_path=Path(fig_dir) / "training_curves.png")

    # Reload the SELECTED checkpoint before scoring.
    # In memory `model` holds the LAST epoch's weights, which under early
    # stopping are up to `patience` epochs past the selected ones; the number
    # printed below would then describe a checkpoint nobody evaluates.
    from ..utils.checkpoint import load_checkpoint as _load_ckpt
    _best = _load_ckpt(ckpt_path, device=device)
    model.load_state_dict(_best["model"])
    if not is_ce and "label_encoder" in _best:
        label_encoder.load_state_dict(_best["label_encoder"])
    print(f"Scoring the selected checkpoint (epoch {best_epoch}, "
          f"stopping_reason={stopping_reason})")

    model.eval()
    label_encoder.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            embeddings = model(
                batch["faces_uv"], batch["faces_attr"],
                batch["edges_attr"], batch["bias"],
                face_mask=batch["face_mask"],
                edge_mask=batch["edge_mask"],
            )
            valid_emb, valid_labels = _gather_valid(
                embeddings, batch["semantic_labels"], batch["face_mask"],
            )
            if valid_emb is None:
                continue
            test_correct += _correct(valid_emb, valid_labels)
            test_total += len(valid_labels)
    test_acc = test_correct / test_total if test_total > 0 else 0.0
    print(f"\nTest Accuracy: {test_acc:.2%} ({test_correct}/{test_total} faces)")
    print(f"Training complete. Best checkpoint: {ckpt_path}")
    return model, label_encoder, history


#  CLI entry point                                                       #

def main():
    parser = argparse.ArgumentParser(
        description="Train Stage 1 semantic segmentation model",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="checkpoint to resume from",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="seed for this run; overrides `seed:` in the config and writes to "
             "seed-scoped output paths (results/.../<name>/seed<k>...)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_semantic_model(cfg, resume=args.resume, seed_override=args.seed)


if __name__ == "__main__":
    main()
