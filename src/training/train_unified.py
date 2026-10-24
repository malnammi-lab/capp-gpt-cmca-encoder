"""
Unified Training: Joint Semantic + Instance + Bottom Contrastive Learning.

Trains a BRepEncoder jointly with LabelEncoder, BottomLabelEncoder, and
three contrastive losses in a single loop. No separate stage-2 model is trained.

Usage:
    python -m src.training.train_unified --config configs/mfinstseg_unified.yaml
"""

# see train_semantic.py -- needed for `int | None` on py3.9
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset, random_split

from ..data.brep_dataset import PreExtractedBRepDataset, load_split
from ..data.collate import collate_fn_brep
from ..losses.bottom_contrastive import BottomContrastiveLoss
from ..losses.infonce import PrototypeInfoNCELoss
from ..losses.instance_loss import HardFocusedInstanceLoss
from ..models.encoders import BottomLabelEncoder, BRepEncoder, LabelEncoder
from ..utils.checkpoint import save_checkpoint
from ..utils.class_weights import required_bias_type, required_training_budget
from ..utils.metrics import AdaptiveThresholdFinder, compute_auc
from ..utils.provenance import build_provenance, stamp_selection
from ..utils.seeding import loader_kwargs, seed_scoped, set_global_seed
from ..utils.visualization import plot_training_curves


def load_config(path):
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _gather_valid(embeddings, semantic_labels, bottom_labels, face_mask):
    """Gather non-padded face embeddings with semantic and bottom labels.

    Returns:
        (flat_emb, flat_sem, flat_bot) or (None, None, None) when empty.
    """
    emb_list, sem_list, bot_list = [], [], []
    for b in range(embeddings.shape[0]):
        mask = face_mask[b]
        if mask.sum() > 0:
            emb_list.append(embeddings[b][mask])
            sem_list.append(semantic_labels[b][mask])
            bot_list.append(bottom_labels[b][mask])
    if not emb_list:
        return None, None, None
    return (
        torch.cat(emb_list, dim=0),
        torch.cat(sem_list, dim=0),
        torch.cat(bot_list, dim=0),
    )


def _collect_instance_pairs(embeddings, instance_labels, face_mask, device):
    """Collect pairwise similarity scores and targets for AUC computation.

    Returns:
        Tuple of (sims_list, targets_list) as Python lists of floats.
    """
    sims, targets = [], []
    for b in range(embeddings.shape[0]):
        mask = face_mask[b]
        if mask.sum() < 2:
            continue
        valid_idx = torch.nonzero(mask).squeeze(1)
        emb_valid = F.normalize(embeddings[b][valid_idx], p=2, dim=-1)
        pred_sim = torch.matmul(emb_valid, emb_valid.T)
        target = instance_labels[b][valid_idx[:, None], valid_idx]

        eye = torch.eye(pred_sim.shape[0], device=device).bool()
        sims.extend(pred_sim[~eye].cpu().numpy().tolist())
        targets.extend(target[~eye].cpu().numpy().tolist())
    return sims, targets


def train_model(cfg, seed_override: int | None = None):
    """Run the unified training loop (semantic + bottom + instance).

    Args:
        cfg: Resolved config dict.
        seed_override: Seed from ``--seed`` or ``runners/run_all.py``; beats
            ``cfg["seed"]`` and switches on seed-scoped output paths.
    """
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    # Seed is set before the dataset is built, so a shuffled bias
    # permutation is a function of the seed.
    seed = seed_override if seed_override is not None else cfg.get("seed")
    seed = set_global_seed(seed)

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
        bias_type=required_bias_type(bias_cfg, where="train_unified"),
        lambda_adj=bias_cfg.get("lambda_adj", 1.0),
        lambda_sim=bias_cfg.get("lambda_sim", 1.0),
        one_hot_type=ds_cfg.get("one_hot_type", False),
    )

    # The full corpus is built only on the no-split-file path.

    # A saved split file takes precedence.
    if split_file:
        print(f"\nSplit file: {split_file}")
        train_ds = PreExtractedBRepDataset(
            model_ids=load_split(split_file, "train"), **ds_kwargs)
        test_ds = PreExtractedBRepDataset(
            model_ids=load_split(split_file, "test"), verbose=False, **ds_kwargs)
        try:
            val_ids = load_split(split_file, "val")
        except KeyError:
            raise KeyError(
                f"train_unified: split file {split_file} has no 'val' "
                f"partition. Early stopping and "
                f"threshold tuning use a real validation set. Point "
                f"dataset.split_file at the 70/10/20 file "
                f"(datasets/splits/*_70_10_20_seed42.json).")
        if not val_ids:
            raise ValueError(
                f"train_unified: split file {split_file} has an EMPTY 'val' "
                f"partition.")
        val_ds = PreExtractedBRepDataset(model_ids=val_ids, verbose=False,
                                         **ds_kwargs)
        n_classes = train_ds.n_classes
        _test_ds = test_ds
        st = train_ds.get_statistics()
        print(f"Dataset split: {len(train_ds)} train / "
              f"{len(val_ds)} val / {len(test_ds)} test")
        print(f"  attributes={ds_kwargs['attributes']}  "
              f"bias={ds_kwargs['bias_type']}  attr_dim={st['attr_dim']}  "
              f"train adjacency density {st['adjacency_density_mean']:.3f}")
    else:
        dataset = PreExtractedBRepDataset(**ds_kwargs)
        train_ratio = ds_cfg.get("train_split", 0.7)
        val_ratio = ds_cfg.get("val_split", 0.2)
        if val_ratio <= 0:
            raise ValueError(
                "train_unified: dataset.val_split is 0 and there is no "
                "dataset.split_file. There is no no-validation path.")
        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            train_ds, val_ds, _test_ds = random_split(
                dataset, [train_size, val_size, test_size], generator=generator
            )
        else:
            train_ds, val_ds, _test_ds = random_split(
                dataset, [train_size, val_size, test_size],
            )
        n_classes = dataset.n_classes
        print(f"\nDataset split: {train_size} train / {val_size} val / {test_size} test")

    # Instance and bottom labels are read from the training batches.
    _labelled = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    if not _labelled.has_instance_labels:
        raise RuntimeError(
            "Unified training requires instance_labels in the dataset JSONs."
        )
    if not _labelled.has_bottom_labels:
        raise RuntimeError(
            "Unified training requires bottom_labels in the dataset JSONs."
        )

    t_cfg = cfg["training"]
    batch_size = t_cfg.get("batch_size", 4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_brep,
        **loader_kwargs(seed, "train_loader"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_brep,
        **loader_kwargs(seed, "val_loader"),
    )

    m_cfg = cfg.get("model", {})
    model = BRepEncoder(
        d_model=m_cfg.get("d_model", 128),
        n_heads=m_cfg.get("n_heads", 8),
        n_layers=m_cfg.get("n_layers", 4),
        dim_feedforward=m_cfg.get("dim_feedforward", 256),
        dropout=m_cfg.get("dropout", 0.1),
    ).to(device)

    le_cfg = cfg.get("label_encoder", {})
    embedding_dim = le_cfg.get("embedding_dim", m_cfg.get("d_model", 128))
    label_encoder = LabelEncoder(
        n_classes=n_classes, embedding_dim=embedding_dim,
    ).to(device)

    bottom_encoder = BottomLabelEncoder(
        embedding_dim=embedding_dim,
    ).to(device)

    l_cfg = cfg.get("loss", {})
    semantic_loss_fn = PrototypeInfoNCELoss(
        temperature=l_cfg.get("temperature", 0.07),
    ).to(device)
    bottom_loss_fn = BottomContrastiveLoss(
        temperature=l_cfg.get("temperature", 0.07),
    ).to(device)
    instance_loss_fn = HardFocusedInstanceLoss(
        temperature=l_cfg.get("temperature", 0.07),
        hard_mining_ratio=l_cfg.get("hard_mining_ratio", 0.3),
    ).to(device)

    loss_weights = l_cfg.get("weights", {})
    w_sem = loss_weights.get("semantic", 1.0)
    w_bot = loss_weights.get("bottom", 1.0)
    w_inst = loss_weights.get("instance", 1.0)

    params = (
        list(model.parameters())
        + list(label_encoder.parameters())
        + list(bottom_encoder.parameters())
        + list(semantic_loss_fn.parameters())
        + list(bottom_loss_fn.parameters())
        + list(instance_loss_fn.parameters())
    )
    optimizer = optim.Adam(
        params,
        lr=t_cfg.get("lr", 1e-4),
        weight_decay=t_cfg.get("weight_decay", 0.0),
    )

    max_epochs, patience = required_training_budget(
        t_cfg, where="train_unified")
    grad_clip = t_cfg.get("grad_clip", 1.0)

    scheduler = None
    if t_cfg.get("scheduler") == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs,
        )

    paths_cfg = cfg.get("paths", {})
    ckpt_path = seed_scoped(
        paths_cfg.get("checkpoint", "results/checkpoints/unified.pth"),
        seed_override)
    fig_dir = seed_scoped(
        paths_cfg.get("figures", "results/figures/unified/"),
        seed_override, is_dir=True)

    history = {
        "loss": [], "val_loss": [],
        "semantic_loss": [], "instance_loss": [], "bottom_loss": [],
        "semantic_acc": [], "bottom_acc": [], "instance_auc": [],
        "optimal_threshold": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0
    # see train_semantic.py; stamped after the loop.
    best_epoch = None
    stopped_early = False
    threshold_finder = AdaptiveThresholdFinder()

    print("\n" + "=" * 70)
    print("UNIFIED TRAINING: Semantic + Bottom + Instance Contrastive")
    print("=" * 70)

    for epoch in range(1, max_epochs + 1):
        model.train()
        label_encoder.train()
        bottom_encoder.train()

        epoch_losses = {"sem": 0.0, "bot": 0.0, "inst": 0.0, "total": 0.0}
        epoch_acc = {"sem": 0, "bot": 0}
        total_samples = 0
        train_inst_sims, train_inst_targets = [], []

        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            embeddings = model(
                batch["faces_uv"], batch["faces_attr"],
                batch["edges_attr"], batch["bias"],
                face_mask=batch["face_mask"],
                edge_mask=batch["edge_mask"],
            )

            h_flat, sem_flat, bot_flat = _gather_valid(
                embeddings, batch["semantic_labels"],
                batch["bottom_labels"], batch["face_mask"],
            )
            if h_flat is None:
                continue

            loss_sem = semantic_loss_fn(
                h_flat, sem_flat, label_encoder, n_classes,
            )
            loss_bot = bottom_loss_fn(h_flat, bot_flat, bottom_encoder)
            loss_inst = instance_loss_fn(
                embeddings, batch["instance_labels"], batch["face_mask"],
            )

            loss = w_sem * loss_sem + w_bot * loss_bot + w_inst * loss_inst

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

            count = len(h_flat)
            epoch_losses["sem"] += loss_sem.item() * count
            epoch_losses["bot"] += loss_bot.item() * count
            epoch_losses["inst"] += loss_inst.item() * count
            epoch_losses["total"] += loss.item() * count

            with torch.no_grad():
                all_cls = torch.arange(n_classes, device=device)
                logits_sem = torch.matmul(
                    F.normalize(h_flat, dim=1),
                    F.normalize(label_encoder(all_cls), dim=1).T,
                )
                epoch_acc["sem"] += (
                    (logits_sem.argmax(dim=1) == sem_flat).sum().item()
                )

                bot_cls = torch.arange(2, device=device)
                logits_bot = torch.matmul(
                    F.normalize(h_flat, dim=1),
                    F.normalize(bottom_encoder(bot_cls), dim=1).T,
                )
                epoch_acc["bot"] += (
                    (logits_bot.argmax(dim=1) == bot_flat).sum().item()
                )

                sims, tgts = _collect_instance_pairs(
                    embeddings, batch["instance_labels"],
                    batch["face_mask"], device,
                )
                train_inst_sims.extend(sims)
                train_inst_targets.extend(tgts)

            total_samples += count

        model.eval()
        label_encoder.eval()
        bottom_encoder.eval()
        val_loss = 0.0
        val_samples = 0
        val_inst_sims, val_inst_targets = [], []

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

                h_flat, sem_flat, bot_flat = _gather_valid(
                    embeddings, batch["semantic_labels"],
                    batch["bottom_labels"], batch["face_mask"],
                )
                if h_flat is None:
                    continue

                loss_sem = semantic_loss_fn(
                    h_flat, sem_flat, label_encoder, n_classes,
                )
                loss_bot = bottom_loss_fn(h_flat, bot_flat, bottom_encoder)
                loss_inst = instance_loss_fn(
                    embeddings, batch["instance_labels"], batch["face_mask"],
                )
                loss = w_sem * loss_sem + w_bot * loss_bot + w_inst * loss_inst

                val_loss += loss.item() * len(h_flat)
                val_samples += len(h_flat)

                sims, tgts = _collect_instance_pairs(
                    embeddings, batch["instance_labels"],
                    batch["face_mask"], device,
                )
                val_inst_sims.extend(sims)
                val_inst_targets.extend(tgts)

        if scheduler is not None:
            scheduler.step()

        if total_samples == 0:
            continue

        for k in epoch_losses:
            epoch_losses[k] /= total_samples
        epoch_acc["sem"] /= total_samples
        epoch_acc["bot"] /= total_samples
        ep_vloss = val_loss / val_samples if val_samples > 0 else float("inf")

        val_auc = compute_auc(
            np.array(val_inst_targets), np.array(val_inst_sims),
        )
        optimal_thresh = (
            threshold_finder.find_optimal_threshold(
                np.array(val_inst_sims), np.array(val_inst_targets),
            )
            if len(val_inst_targets) > 0
            else 0.5
        )

        history["loss"].append(epoch_losses["total"])
        history["val_loss"].append(ep_vloss)
        history["semantic_loss"].append(epoch_losses["sem"])
        history["instance_loss"].append(epoch_losses["inst"])
        history["bottom_loss"].append(epoch_losses["bot"])
        history["semantic_acc"].append(epoch_acc["sem"])
        history["bottom_acc"].append(epoch_acc["bot"])
        history["instance_auc"].append(val_auc)
        history["optimal_threshold"].append(optimal_thresh)

        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Loss {epoch_losses['total']:.4f} | Val {ep_vloss:.4f} | "
            f"Sem {epoch_acc['sem']:.2%} | Bot {epoch_acc['bot']:.2%} | "
            f"InstAUC {val_auc:.3f} | Thresh {optimal_thresh:.3f}"
        )

        if ep_vloss < best_val_loss:
            best_val_loss = ep_vloss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                ckpt_path,
                model,
                label_encoder=label_encoder,
                bottom_encoder=bottom_encoder,
                history=history,
                n_classes=n_classes,
                epoch=epoch,
                # Tuned on the validation partition.
                optimal_threshold=optimal_thresh,
                extra={"provenance": build_provenance(
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

    stopping_reason = "early_stop" if stopped_early else "cap"
    if best_epoch is None:
        raise RuntimeError(
            "train_unified: no epoch improved on an infinite validation loss, "
            "so no checkpoint was selected. The validation loader produced no "
            "usable batches.")
    stamp_selection(ckpt_path, selected_epoch=best_epoch,
                    stopping_reason=stopping_reason)
    if stopping_reason == "cap":
        print(f"  NOTE: training ended at the max_epochs cap ({max_epochs}) "
              f"with patience intact; the cap audit fails this run until the "
              f"comparison re-runs at a raised cap.")

    plot_training_curves(history, save_path=Path(fig_dir) / "training_curves.png")
    print(f"\nTraining complete. Best checkpoint: {ckpt_path} "
          f"(epoch {best_epoch}, stopping_reason={stopping_reason})")
    return model, label_encoder, bottom_encoder, history


#  CLI entry point                                                       #

def main():
    parser = argparse.ArgumentParser(
        description="Train unified model (semantic + bottom + instance)",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="seed for this run; overrides `seed:` in the config and writes to "
             "seed-scoped output paths (results/.../<name>/seed<k>...)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_model(cfg, seed_override=args.seed)


if __name__ == "__main__":
    main()
