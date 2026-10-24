"""
Stage 2: Link Prediction Training for Instance Segmentation.

Loads a frozen Stage 1 semantic model and trains a GraphLinkPredictor
on top of the frozen face embeddings.  For every pair of faces the
predictor outputs a binary classification (same-instance / different).

Usage:
    python -m src.training.train_link --config configs/mfinstseg_two_stage.yaml
"""

# see train_semantic.py -- needed for `int | None` on py3.9
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
from torch.utils.data import DataLoader, Subset, random_split

from ..data.brep_dataset import PreExtractedBRepDataset, load_split
from ..data.collate import collate_fn_brep
from ..models.encoders import LabelEncoder, SemanticSegmentationModel
from ..models.link_predictor import GraphLinkPredictor
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.class_weights import required_bias_type, required_training_budget
from ..utils.metrics import compute_auc
from ..utils.provenance import build_provenance, stamp_selection
from ..utils.seeding import loader_kwargs, seed_scoped, set_global_seed
from ..utils.visualization import plot_training_curves


def load_config(path):
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _flatten_valid_pairs(link_logits, instance_labels, face_mask, device):
    """Extract valid (non-padded, non-diagonal) pair logits and labels.

    Args:
        link_logits: (B, N, N, 2) predicted link logits.
        instance_labels: (B, N, N) ground-truth instance affinity.
        face_mask: (B, N) boolean mask.
        device: Torch device.

    Returns:
        Tuple of (flat_logits, flat_labels) or (None, None) when empty.
    """
    logits_list, labels_list = [], []
    for b in range(link_logits.shape[0]):
        mask = face_mask[b]
        n_valid = mask.sum().item()
        if n_valid < 2:
            continue

        valid_idx = torch.nonzero(mask).squeeze(1)
        pair_logits = link_logits[b][valid_idx[:, None], valid_idx]      # (V, V, 2)
        pair_labels = instance_labels[b][valid_idx[:, None], valid_idx]  # (V, V)

        eye = torch.eye(n_valid, device=device).bool()
        logits_list.append(pair_logits[~eye])           # (V*(V-1), 2)
        labels_list.append(pair_labels[~eye].long())     # (V*(V-1),)

    if not logits_list:
        return None, None
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def train_link_predictor(cfg, seed_override: int | None = None):
    """Run the Stage 2 link prediction training loop.

    Args:
        cfg: Resolved config dict.
        seed_override: Seed from ``--seed`` or ``runners/run_all.py``; beats
            ``cfg["seed"]`` and switches on seed-scoped output paths.
    """
    t_start = time.time()
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

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
        bias_type=required_bias_type(bias_cfg, where="train_link"),
        lambda_adj=bias_cfg.get("lambda_adj", 1.0),
        lambda_sim=bias_cfg.get("lambda_sim", 1.0),
        one_hot_type=ds_cfg.get("one_hot_type", False),
    )

    # The full corpus is built only on the no-split-file path.

    # A saved split file takes precedence.
    if split_file:
        print(f"\nSplit file: {split_file}")
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        train_ds = PreExtractedBRepDataset(
            model_ids=load_split(split_file, "train"), **ds_kwargs)
        test_ds = PreExtractedBRepDataset(
            model_ids=load_split(split_file, "test"), verbose=False, **ds_kwargs)
        # The link threshold is tuned on the validation partition.
        try:
            val_ids = load_split(split_file, "val")
        except KeyError:
            raise KeyError(
                f"train_link: split file {split_file} has no 'val' partition. "
                f"The link threshold is tuned and training stops "
                f"on a real validation set. Point dataset.split_file at the "
                f"70/10/20 file (datasets/splits/*_70_10_20_seed42.json).")
        if not val_ids:
            raise ValueError(
                f"train_link: split file {split_file} has an EMPTY 'val' "
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
        val_ratio = ds_cfg.get("val_split", 0.15)
        if val_ratio <= 0:
            raise ValueError(
                "train_link: dataset.val_split is 0 and there is no "
                "dataset.split_file. There is no no-validation path.")
        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
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

    _labelled = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    if not _labelled.has_instance_labels:
        raise RuntimeError(
            "Link prediction requires instance_labels in the dataset JSONs."
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

    sem_ckpt_path = cfg["semantic_checkpoint"]
    sem_cfg = cfg.get("semantic_model", cfg.get("model", {}))
    semantic_model = SemanticSegmentationModel(
        d_model=sem_cfg.get("d_model", 128),
        n_heads=sem_cfg.get("n_heads", 8),
        n_layers=sem_cfg.get("n_layers", 4),
        dim_feedforward=sem_cfg.get("dim_feedforward", 256),
        dropout=sem_cfg.get("dropout", 0.1),
    ).to(device)

    ckpt = load_checkpoint(sem_ckpt_path, device=device)
    semantic_model.load_state_dict(ckpt["model"])
    semantic_model.eval()
    for param in semantic_model.parameters():
        param.requires_grad = False

    print("Frozen semantic model loaded from", sem_ckpt_path)

    m_cfg = cfg.get("link_model", cfg.get("model", {}))
    link_model = GraphLinkPredictor(
        d_model=m_cfg.get("d_model", 128),
        n_heads=m_cfg.get("n_heads", 8),
        n_layers=m_cfg.get("n_layers", 2),
        dim_feedforward=m_cfg.get("dim_feedforward", 256),
        dropout=m_cfg.get("dropout", 0.1),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        link_model.parameters(),
        lr=t_cfg.get("lr", 5e-5),
        weight_decay=t_cfg.get("weight_decay", 0.0),
    )

    max_epochs, patience = required_training_budget(t_cfg, where="train_link")
    grad_clip = t_cfg.get("grad_clip", 1.0)

    scheduler = None
    if t_cfg.get("scheduler") == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs,
        )

    paths_cfg = cfg.get("paths", {})
    # Prefer paths.link_checkpoint; fall back to paths.checkpoint so
    # single-stage configs still work.
    ckpt_path = seed_scoped(paths_cfg.get(
        "link_checkpoint",
        paths_cfg.get("checkpoint", "results/checkpoints/link.pth")), seed_override)
    fig_dir = seed_scoped(paths_cfg.get(
        "link_figures",
        paths_cfg.get("figures", "results/figures/link/")), seed_override, is_dir=True)

    history = {
        "loss": [], "acc": [], "val_loss": [], "val_acc": [],
        "train_auc": [], "val_auc": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0
    # see train_semantic.py; stamped after the loop.
    best_epoch = None
    stopped_early = False

    print("\n" + "=" * 70)
    print("STAGE 2: Link Prediction for Instance Segmentation")
    print("=" * 70)

    for epoch in range(1, max_epochs + 1):
        link_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_probs, train_targets = [], []

        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            with torch.no_grad():
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

            loss = criterion(flat_logits, flat_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(link_model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss += loss.item() * len(flat_labels)
            preds = flat_logits.argmax(dim=1)
            train_correct += (preds == flat_labels).sum().item()
            train_total += len(flat_labels)

            probs = F.softmax(flat_logits, dim=1)[:, 1]
            train_probs.extend(probs.detach().cpu().numpy())
            train_targets.extend(flat_labels.cpu().numpy())

        link_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
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

                loss = criterion(flat_logits, flat_labels)
                val_loss += loss.item() * len(flat_labels)
                preds = flat_logits.argmax(dim=1)
                val_correct += (preds == flat_labels).sum().item()
                val_total += len(flat_labels)

                probs = F.softmax(flat_logits, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(flat_labels.cpu().numpy())

        if scheduler is not None:
            scheduler.step()

        ep_loss = train_loss / train_total if train_total > 0 else 0
        ep_acc = train_correct / train_total if train_total > 0 else 0
        ep_vloss = val_loss / val_total if val_total > 0 else float("inf")
        ep_vacc = val_correct / val_total if val_total > 0 else 0

        train_auc = compute_auc(np.array(train_targets), np.array(train_probs))
        val_auc = compute_auc(np.array(val_targets), np.array(val_probs))

        history["loss"].append(ep_loss)
        history["acc"].append(ep_acc)
        history["val_loss"].append(ep_vloss)
        history["val_acc"].append(ep_vacc)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Loss {ep_loss:.4f} | Acc {ep_acc:.2%} | "
            f"Val Loss {ep_vloss:.4f} | Val Acc {ep_vacc:.2%} | "
            f"Val AUC {val_auc:.3f}"
        )

        if ep_vloss < best_val_loss:
            best_val_loss = ep_vloss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                ckpt_path,
                link_model,
                history=history,
                epoch=epoch,
                extra={"semantic_checkpoint": sem_ckpt_path,
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

    stopping_reason = "early_stop" if stopped_early else "cap"
    if best_epoch is None:
        raise RuntimeError(
            "train_link: no epoch improved on an infinite validation loss, so "
            "no checkpoint was selected. The validation loader produced no "
            "usable face pairs.")
    stamp_selection(ckpt_path, selected_epoch=best_epoch,
                    stopping_reason=stopping_reason)
    if stopping_reason == "cap":
        print(f"  NOTE: training ended at the max_epochs cap ({max_epochs}) "
              f"with patience intact; the cap audit fails this run until the "
              f"comparison re-runs at a raised cap.")

    plot_training_curves(history, save_path=Path(fig_dir) / "training_curves.png")
    print(f"\nTraining complete. Best checkpoint: {ckpt_path} "
          f"(epoch {best_epoch}, stopping_reason={stopping_reason})")
    return link_model, history


#  CLI entry point                                                       #

def main():
    parser = argparse.ArgumentParser(
        description="Train Stage 2 link prediction model",
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
    train_link_predictor(cfg, seed_override=args.seed)


if __name__ == "__main__":
    main()
