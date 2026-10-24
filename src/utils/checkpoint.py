"""
Checkpoint save / load helpers.

Provides a consistent checkpoint format across all training scripts so
that evaluation code can load any checkpoint without guessing key names.

Canonical checkpoint layout::

    {
        "model":          model.state_dict(),
        "label_encoder":  label_encoder.state_dict(),   # optional
        "bottom_encoder": bottom_encoder.state_dict(),   # optional
        "criterion":      criterion.state_dict(),        # optional
        "history":        {...},                          # training history dict
        "n_classes":      int,
        "epoch":          int,
        "optimal_threshold": float,                      # optional
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    label_encoder: nn.Module | None = None,
    bottom_encoder: nn.Module | None = None,
    criterion: nn.Module | None = None,
    history: dict[str, list] | None = None,
    n_classes: int | None = None,
    epoch: int | None = None,
    optimal_threshold: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint to *path*.

    The function creates parent directories if they do not exist.

    Args:
        path: Destination file path (typically ``*.pth``).
        model: The main model whose ``state_dict`` will be saved.
        label_encoder: Optional label encoder module.
        bottom_encoder: Optional bottom-face label encoder module.
        criterion: Optional loss module (may carry learnable params like
            temperature in ``PrototypeInfoNCELoss``).
        history: Training history dict (losses, accuracies, etc.).
        n_classes: Number of classes used during training.
        epoch: Current / final epoch number.
        optimal_threshold: Adaptive threshold for instance segmentation.
        extra: Arbitrary extra key/value pairs to store.
    """
    payload: dict[str, Any] = {"model": model.state_dict()}

    if label_encoder is not None:
        payload["label_encoder"] = label_encoder.state_dict()
    if bottom_encoder is not None:
        payload["bottom_encoder"] = bottom_encoder.state_dict()
    if criterion is not None:
        payload["criterion"] = criterion.state_dict()
    if history is not None:
        payload["history"] = history
    if n_classes is not None:
        payload["n_classes"] = n_classes
    if epoch is not None:
        payload["epoch"] = epoch
    if optimal_threshold is not None:
        payload["optimal_threshold"] = optimal_threshold
    if extra is not None:
        payload.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint dict from *path*.

    The returned dict contains raw ``state_dict`` tensors (not live
    modules).  The caller is responsible for creating model instances and
    calling ``model.load_state_dict(ckpt["model"])``.

    Args:
        path: Path to a ``*.pth`` checkpoint file.
        device: Map location for ``torch.load`` (e.g. ``"cpu"`` or
            ``"cuda"``).

    Returns:
        The checkpoint dict with keys such as ``"model"``,
        ``"label_encoder"``, ``"history"``, ``"n_classes"``, etc.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    print(f"Checkpoint loaded from {path}")

    # Normalise legacy key names so downstream code can use a single key
    _ALIASES = {
        "model_state_dict": "model",
        "label_encoder_state_dict": "label_encoder",
        "criterion_state_dict": "criterion",
        "label_enc": "label_encoder",
        "bot_enc": "bottom_encoder",
    }
    for old_key, new_key in _ALIASES.items():
        if old_key in checkpoint and new_key not in checkpoint:
            checkpoint[new_key] = checkpoint.pop(old_key)

    return checkpoint
