"""
ResNet Baseline: Cross-Modal Contrastive Alignment Training.

Trains a CrossModalAlignmentModel (ResNet-18 UV encoder + LabelEncoder)
with PrototypeInfoNCELoss.  Operates on flat, per-face UV grids — no
graph structure needed.

Usage:
    python -m src.training.train_resnet --config configs/resnet_cadsynth.yaml
"""

# see train_semantic.py -- needed for `int | None` on py3.9
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split

from ..data.uv_dataset import UVGridLabelDataset
from ..data.brep_dataset import load_split
from ..models.resnet_encoder import (CrossModalAlignmentModel,
                                     ResNetClassificationModel)
from ..utils.checkpoint import save_checkpoint
from ..utils.class_weights import (class_weighting_enabled,
                                   compute_class_weights,
                                   required_training_budget)
from ..utils.provenance import build_provenance, stamp_selection
from ..utils.seeding import loader_kwargs, seed_scoped, set_global_seed
from ..utils.visualization import plot_training_curves


def load_config(path):
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _correct(model, uv_grids, labels, n_classes, device, is_ce):
    """Correct-prediction count for either objective.

    CMCA has no class head -- it classifies by nearest prototype in the shared
    embedding space. CE has no prototypes. Keeping both behind one call means
    the training loop itself does not branch.
    """
    if is_ce:
        return (model.predict(uv_grids) == labels).sum().item()
    uv_emb, _ = model(uv_grids, labels, return_embeddings=True)
    protos = F.normalize(
        model.label_encoder(torch.arange(n_classes, device=device)), dim=1)
    preds = torch.matmul(F.normalize(uv_emb, dim=1), protos.T).argmax(dim=1)
    return (preds == labels).sum().item()


def train(cfg, seed_override: int | None = None):
    """Run the ResNet baseline cross-modal alignment training loop.

    Args:
        cfg: Resolved config dict.
        seed_override: Seed from ``--seed`` or ``runners/run_all.py``; beats
            ``cfg["seed"]`` and switches on seed-scoped output paths.
    """
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    # set_global_seed covers torch, numpy, random, and cudnn, and raises
    # when no seed is given.
    seed = seed_override if seed_override is not None else cfg.get("seed")
    seed = set_global_seed(seed)

    ds_cfg = cfg["dataset"]
    split_file = ds_cfg.get("split_file")

    # The classifier head width is dataset.n_classes from the config.
    if "n_classes" not in ds_cfg:
        raise KeyError(
            "train_resnet: dataset.n_classes is not set in this config. It may "
            "not be inferred from the corpus: doing so sizes the classifier "
            "head from train AND test labels, and lets the ResNet cells of the "
            "2x2 use a different class set than the graph cells.")
    n_classes = ds_cfg["n_classes"]
    if not isinstance(n_classes, int) or isinstance(n_classes, bool) or n_classes < 1:
        raise ValueError(
            f"train_resnet: dataset.n_classes must be a positive integer, "
            f"got {n_classes!r}.")
    print(f"Classifier head sized from config: {n_classes} classes")

    # A saved split file takes precedence; the original random_split
    # path is kept below and still runs when no split_file is configured.
    #
    # Note the unit mismatch. A split file lists MODEL ids, but this dataset's
    # samples are FACES. The subset is therefore taken by model at load time, so
    # every face of a part stays on one side of the boundary. Splitting the
    # flattened face list -- which is what random_split does -- scatters faces of
    # the same part across train and test and inflates accuracy invisibly.
    #
    # The full corpus is built only on the no-split_file path, which is
    # the only branch that partitions it.
    if split_file:
        print(f"\nSplit file: {split_file}")
        _kw = dict(json_dir=ds_cfg["json_dir"],
                   n_classes=n_classes, verbose=False)
        train_ds = UVGridLabelDataset(
            model_ids=load_split(split_file, "train"), **_kw)
        test_ds = UVGridLabelDataset(
            model_ids=load_split(split_file, "test"), **_kw)
        # A validation partition is mandatory.
        try:
            val_ids = load_split(split_file, "val")
        except KeyError:
            raise KeyError(
                f"train_resnet: split file {split_file} has no 'val' "
                f"partition. Every run selects the best-"
                f"validation-loss checkpoint in every run. Point "
                f"dataset.split_file at the 70/10/20 file "
                f"(datasets/splits/*_70_10_20_seed42.json).")
        if not val_ids:
            raise ValueError(
                f"train_resnet: split file {split_file} has an EMPTY 'val' "
                f"partition.")
        val_ds = UVGridLabelDataset(model_ids=val_ids, **_kw)
        train_size, test_size = len(train_ds), len(test_ds)
        val_size = len(val_ds)
        if train_size == 0:
            raise RuntimeError(
                "No training faces loaded -- check dataset.json_dir and the "
                "train ids in the split file.")
        print(f"Dataset split: {train_size} train / {val_size} val / "
              f"{test_size} test faces  "
              f"({len(train_ds.model_ids)} / {len(test_ds.model_ids)} models)")
    else:
        dataset = UVGridLabelDataset(
            json_dir=ds_cfg["json_dir"],
            n_classes=n_classes,
        )
        if len(dataset) == 0:
            raise RuntimeError("No data loaded — check dataset.json_dir in config.")

        train_ratio = ds_cfg.get("train_split", 0.8)
        val_ratio = ds_cfg.get("val_split", 0.0)
        if val_ratio <= 0:
            raise ValueError(
                "train_resnet: dataset.val_split is 0 and there is no "
                "dataset.split_file, so this config asks for training with no "
                "validation partition. Every run selects the best-validation-"
                "loss checkpoint in every run. Use a 70/10/20 split file, or "
                "set dataset.val_split > 0 for a development run.")

        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size

        generator = (torch.Generator().manual_seed(seed)
                     if seed is not None else None)
        train_ds, val_ds, test_ds = random_split(
            dataset, [train_size, val_size, test_size], generator=generator,
        )

        print(f"\nDataset split: {train_size} train / {val_size} val / "
              f"{test_size} test")

    t_cfg = cfg["training"]
    batch_size = t_cfg.get("batch_size", 128)
    # drop_last on the TRAIN loader only. UVGridEncoder's projection
    # head contains BatchNorm1d, which raises
    #     ValueError: Expected more than 1 value per channel when training
    # if the final batch of an epoch holds exactly one face. Whether that
    # happens depends on the split size and therefore on the seed, so the run
    # would fail intermittently rather than reproducibly.
    #
    # NOT applied to the val/test loaders below: dropping a partial batch there
    # would silently discard evaluation samples and change the reported metric.
    #
    # This trainer is the one with real per-face batches (batch_size 128),
    # so its shuffle order matters most.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        **loader_kwargs(seed, "train_loader"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        **loader_kwargs(seed, "test_loader"),
    )
    # val_ds is always built. The validation loader is not optional.
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            **loader_kwargs(seed, "val_loader"))

    l_cfg = cfg.get("loss", {})
    class_weights = None
    if class_weighting_enabled(l_cfg, where="train_resnet"):
        class_weights = compute_class_weights(train_ds, n_classes, device)

    # `loss.type` picks the objective and with it the head. Both
    # share the same UVGridEncoder, so a gap between them is attributable to the
    # objective rather than to the tower.
    m_cfg = cfg.get("model", {})
    loss_type = l_cfg.get("type", "prototype_infonce")
    if loss_type not in ("prototype_infonce", "cross_entropy"):
        raise ValueError(
            f"loss.type must be prototype_infonce|cross_entropy, got {loss_type!r}")
    is_ce = loss_type == "cross_entropy"

    if is_ce:
        model = ResNetClassificationModel(
            n_classes=n_classes,
            embedding_dim=m_cfg.get("embedding_dim", 512),
            dropout=m_cfg.get("dropout", 0.3),
            class_weights=class_weights,
        ).to(device)
    else:
        model = CrossModalAlignmentModel(
            n_classes=n_classes,
            embedding_dim=m_cfg.get("embedding_dim", 512),
            temperature=l_cfg.get("temperature", 0.07),
            class_weights=class_weights,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=t_cfg.get("lr", 3e-4),
        weight_decay=t_cfg.get("weight_decay", 1e-4),
    )

    max_epochs, patience = required_training_budget(t_cfg, where="train_resnet")

    scheduler = None
    sched_name = t_cfg.get("scheduler", "cosine_annealing")
    if sched_name == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs,
        )

    grad_clip = t_cfg.get("grad_clip", 1.0)

    paths_cfg = cfg.get("paths", {})
    # Seed-scoped; no-op unless --seed was passed.
    ckpt_path = seed_scoped(
        paths_cfg.get("checkpoint", "results/checkpoints/resnet.pth"),
        seed_override)
    fig_dir = seed_scoped(
        paths_cfg.get("figures", "results/figures/resnet/"),
        seed_override, is_dir=True)

    # val_loss / val_acc are the selection signal.
    history = {"loss": [], "lr": [], "acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = None
    stopped_early = False

    print("\n" + "=" * 70)
    print("RESNET BASELINE: Cross-Modal Contrastive Alignment")
    print("=" * 70)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for uv_grids, labels in train_loader:
            uv_grids = uv_grids.to(device)
            labels = labels.to(device)

            loss = model(uv_grids, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_loss += loss.item() * len(labels)

            with torch.no_grad():
                train_correct += _correct(model, uv_grids, labels,
                                          n_classes, device, is_ce)
            train_total += len(labels)

        ep_loss = total_loss / train_total if train_total > 0 else 0
        ep_acc = train_correct / train_total if train_total > 0 else 0
        current_lr = optimizer.param_groups[0]["lr"]

        history["loss"].append(ep_loss)
        history["lr"].append(current_lr)
        history["acc"].append(ep_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for uv_grids, labels in val_loader:
                uv_grids = uv_grids.to(device)
                labels = labels.to(device)

                loss = model(uv_grids, labels)
                val_loss += loss.item() * len(labels)

                val_correct += _correct(model, uv_grids, labels,
                                        n_classes, device, is_ce)
                val_total += len(labels)

        ep_vloss = val_loss / val_total if val_total > 0 else float("inf")
        ep_vacc = val_correct / val_total if val_total > 0 else 0.0
        history["val_loss"].append(ep_vloss)
        history["val_acc"].append(ep_vacc)

        if scheduler is not None:
            scheduler.step()

        msg = (
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Loss {ep_loss:.4f} | Acc {ep_acc:.2%} | LR {current_lr:.2e}"
            f" | Val Loss {ep_vloss:.4f} | Val Acc {ep_vacc:.2%}"
        )
        if epoch % 10 == 0 or epoch == 1:
            print(msg)

        # Save only when validation loss improves, and stop when patience runs out.
        if ep_vloss < best_val_loss:
            best_val_loss = ep_vloss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                ckpt_path,
                model,
                label_encoder=None if is_ce else model.label_encoder,
                criterion=None if is_ce else model.contrastive_loss,
                history=history,
                n_classes=n_classes,
                # The epoch the loop actually reached.
                epoch=epoch,
                # provenance block; see training-regiment.md s.1
                extra={"best_val_loss": best_val_loss,
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
            "train_resnet: no epoch improved on an infinite validation loss, "
            "so no checkpoint was selected. The validation loader produced no "
            "usable batches.")
    stamp_selection(ckpt_path, selected_epoch=best_epoch,
                    stopping_reason=stopping_reason)
    if stopping_reason == "cap":
        print(f"  NOTE: training ended at the max_epochs cap ({max_epochs}) "
              f"with patience intact; the cap audit fails this run until the "
              f"comparison re-runs at a raised cap.")

    plot_training_curves(history, save_path=Path(fig_dir) / "training_curves.png")

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    # Score the SELECTED checkpoint. In memory `model`
    # holds the last epoch's weights, which under early stopping are up to
    # `patience` epochs past the selected ones; scoring those would report a
    # number for a checkpoint nobody evaluates.
    from ..utils.checkpoint import load_checkpoint as _load_ckpt
    _best = _load_ckpt(ckpt_path, device=device)
    model.load_state_dict(_best["model"])
    print(f"Scoring the selected checkpoint (epoch {best_epoch}, "
          f"stopping_reason={stopping_reason})")
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        # CE classifies with its head; CMCA by nearest prototype.
        # There is no label tower on the CE model, so the prototype lookup must
        # be skipped entirely rather than merely unused.
        protos = None
        if not is_ce:
            all_cls = torch.arange(n_classes, device=device)
            protos = F.normalize(model.label_encoder(all_cls), dim=1)

        for uv_grids, labels in test_loader:
            uv_grids = uv_grids.to(device)
            labels = labels.to(device)

            if is_ce:
                preds = model.predict(uv_grids)
            else:
                emb_n = F.normalize(model.uv_encoder(uv_grids), dim=1)
                preds = torch.matmul(emb_n, protos.T).argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    test_acc = (all_preds == all_gts).mean()
    print(f"Test Accuracy: {test_acc:.2%}")

    for cls in range(n_classes):
        mask = all_gts == cls
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == all_gts[mask]).mean()
            print(f"  Class {cls:2d}: {cls_acc:.2%} ({mask.sum()} samples)")

    print(f"\nTraining complete. Best checkpoint: {ckpt_path} "
          f"(epoch {best_epoch}, stopping_reason={stopping_reason})")
    return model, history


#  CLI entry point                                                       #

def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet baseline cross-modal alignment model",
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
    train(cfg, seed_override=args.seed)


if __name__ == "__main__":
    main()
