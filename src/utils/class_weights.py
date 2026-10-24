"""
Compute per-class weights for loss re-balancing, and guard the config keys a
trainer is not allowed to guess.

The inverse-frequency weighting scheme is the standard approach used
across the project notebooks.

This module has collected the trainer-side "required key" guards
(`class_weighting_enabled`, `required_bias_type`, `required_training_budget`).
They live together because they are one idea -- a setting that decides an
experiment may not have a default -- and because putting them here keeps the
four trainers from carrying four copies of the same raise. If the set grows
further it wants its own module; it is three functions today.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def class_weighting_enabled(loss_cfg: dict, *, where: str) -> bool:
    """Read ``loss.use_class_weights``, refusing to guess when it is absent.

    Every arm and the ResNet baseline share one class-weighting policy.
    The headline configuration is unweighted. The weighted runs are tier T4.
    The key is required.

    Args:
        loss_cfg: The config's ``loss`` block.
        where: Caller name, quoted in the error so the message is actionable.

    Returns:
        The configured value.

    Raises:
        KeyError: If the key is absent. Deliberate: see above.
    """
    if "use_class_weights" not in loss_cfg:
        raise KeyError(
            f"{where}: loss.use_class_weights is not set in this config.\n"
            f"The class-weighting rule requires one shared class-weighting policy "
            f"across every arm and the ResNet baseline, so this may not fall "
            f"back to a default.\n"
            f"Set `use_class_weights: false` for the headline configuration, or "
            f"`true` for a tier-T4 footnote run, and say which in the config.")
    value = loss_cfg["use_class_weights"]
    if not isinstance(value, bool):
        raise TypeError(
            f"{where}: loss.use_class_weights must be true or false, "
            f"got {value!r} ({type(value).__name__}).")
    return value


#: The bias types `src/data/brep_dataset.py` accepts.
#: Duplicated here rather than imported so that a config is rejected before a
#: corpus is loaded; the dataset re-validates and is the authority.
BIAS_TYPES = ("similarity", "adjacency", "both", "shuffled")


def required_bias_type(bias_cfg: dict, *, where: str) -> str:
    """Read ``bias.type``, refusing to guess when it is absent.

    Args:
        bias_cfg: The config's ``bias`` block (``cfg.get("bias", {})``).
        where: Caller name, quoted in the error so the message is actionable.

    Returns:
        The configured bias type.

    Raises:
        KeyError: If ``bias.type`` is absent.
        ValueError: If it is present but not one of `BIAS_TYPES`.
    """
    if not isinstance(bias_cfg, dict) or "type" not in bias_cfg:
        raise KeyError(
            f"{where}: bias.type is not set in this config.\n"
            f"It may not fall back to a default: the trainers defaulted to "
            f"'similarity' and the dataset to 'adjacency', which are the two "
            f"arms the paper's main table compares. A missing line would pick "
            f"an arm silently.\n"
            f"Add a `bias:` block with `type: <{'|'.join(BIAS_TYPES)}>`.")
    value = bias_cfg["type"]
    if value not in BIAS_TYPES:
        raise ValueError(
            f"{where}: bias.type must be one of {BIAS_TYPES}, got {value!r}.")
    return value


def required_training_budget(t_cfg: dict, *, where: str) -> tuple:
    """Read ``training.max_epochs`` and ``training.patience``; refuse defaults.

    Early stopping on validation loss, patience 10, a cap of 60, and evaluation
    of the best-validation checkpoint for every cell.

    ``max_epochs`` and ``patience`` have no defaults. They are the selection
    rule; the regiment's section-4 checklist requires one identical rule
    across the table, and a per-trainer default is how arms drift apart.

    Args:
        t_cfg: The config's ``training`` block.
        where: Caller name, quoted in the errors.

    Returns:
        ``(max_epochs, patience)``, both positive ints.

    Raises:
        KeyError: If either key is absent, or if the superseded ``epochs`` key
            is still present.
        ValueError: If either value is not a positive integer.
    """
    if "epochs" in t_cfg:
        raise KeyError(
            f"{where}: this config still sets `training.epochs`, which was "
            f"removed project-wide along with fixed-budget "
            f"training. Replace it with `max_epochs: 60` and `patience: 10`. "
            f"Leaving it in place would train under the superseded scheme and "
            f"produce numbers the paper cannot quote.")

    missing = [k for k in ("max_epochs", "patience") if k not in t_cfg]
    if missing:
        raise KeyError(
            f"{where}: training.{' and training.'.join(missing)} not set in "
            f"this config. One selection rule applies to every cell "
            f"(early stop on validation loss, patience 10, cap 60), so these "
            f"may not fall back to per-trainer defaults -- that is how two arms "
            f"of the same table end up on different rules.")

    out = []
    for key in ("max_epochs", "patience"):
        value = t_cfg[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(
                f"{where}: training.{key} must be a positive integer, "
                f"got {value!r} ({type(value).__name__}).")
        out.append(value)
    return out[0], out[1]


def compute_class_weights(
    dataset: Dataset,
    n_classes: int,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a dataset.

    Iterates over the full dataset to count samples per class, then
    returns weights proportional to ``1 / count``, normalised so that
    they sum to ``n_classes`` (keeps the loss magnitude stable).

    The function supports two dataset conventions:

    - **UVGridLabelDataset** -- ``dataset[i]`` returns ``(uv_grid, label)``
      where ``label`` is a scalar tensor.
    - **PreExtractedBRepDataset** -- ``dataset[i]`` returns a dict with
      a ``"semantic_labels"`` key containing a 1-D label tensor (one
      label per face).

    Args:
        dataset: A PyTorch Dataset instance.
        n_classes: Number of classes.
        device: Device to place the weight tensor on.

    Returns:
        A ``(n_classes,)`` float tensor of normalised weights.
    """
    class_counts = np.zeros(n_classes, dtype=np.float64)

    for i in range(len(dataset)):
        sample = dataset[i]

        if isinstance(sample, dict):
            labels = sample["semantic_labels"]
        elif isinstance(sample, (tuple, list)):
            labels = sample[1]
        else:
            raise TypeError(
                f"Unsupported dataset item type: {type(sample)}. "
                "Expected a dict or a (data, label) tuple."
            )

        labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)

        if labels.dim() == 0:
            idx = labels.item()
            if 0 <= idx < n_classes:
                class_counts[idx] += 1
        else:
            for idx in labels.tolist():
                if 0 <= idx < n_classes:
                    class_counts[idx] += 1

    print("\nClass counts:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i:2d}: {int(count):5d} samples")

    weights = 1.0 / (class_counts + 1e-8)
    weights = weights / weights.sum() * n_classes

    print("\nClass weights (inverse-frequency, normalised):")
    for i, w in enumerate(weights):
        print(f"  Class {i:2d}: {w:.4f}")

    return torch.tensor(weights, dtype=torch.float32).to(device)
