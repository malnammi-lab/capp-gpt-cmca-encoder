"""
Global seeding and seed-scoped output paths.

`training-regiment.md` requires every quotable run to fix seeds 0-4 and to
control torch, numpy, random and cudnn.

The seed is required. `set_global_seed(None)` raises, so an unseeded paper
run is impossible.

`seed_scoped` inserts a `seed<k>` component so runs at different seeds cannot
collide. It is a no-op when *seed* is None.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = True) -> int:
    """Seed every generator a run depends on.

    Seeds Python's ``random``, numpy, torch CPU and all CUDA devices, and sets
    the ``PYTHONHASHSEED`` environment variable.  With *deterministic* it also
    pins cuDNN to its deterministic algorithms and disables the autotuner,
    which is what makes two runs of the same seed comparable on GPU.

    Args:
        seed: The run's seed.  Must be a non-negative integer; the regiment
            fixes the list [0, 1, 2, 3, 4].
        deterministic: Set ``cudnn.deterministic = True`` and
            ``cudnn.benchmark = False``.  Leave this on for any run whose
            numbers may enter the paper.  Turning it off is faster but the
            run is then not reproducible and is a debug run.

    Returns:
        The seed, so callers can record it in provenance without re-reading
        the config.

    Raises:
        TypeError: If *seed* is None or not an integer.  This is deliberate:
            see the module docstring.  A run with no seed is not a run.
        ValueError: If *seed* is negative.
    """
    if seed is None:
        raise TypeError(
            "set_global_seed(None): this run has no seed. Every run that may "
            "produce a paper number needs one -- training-regiment.md fixes "
            "seeds 0-4. Pass --seed on the command line, or set `seed:` in the "
            "config. If this is a throwaway debug run, pass --seed 0 and do not "
            "commit its metrics."
        )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}: {seed!r}")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cudnn.benchmark picks the fastest convolution algorithm by timing
        # them, and the winner can differ between runs on the same hardware.
        # That alone breaks seed reproducibility for the ResNet arm.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def make_generator(seed: int) -> torch.Generator:
    """Return a ``torch.Generator`` seeded from *seed*.

    DataLoader shuffling draws from a generator, not from the global RNG, so
    passing one of these is what actually makes the batch order reproducible.

    Raises:
        TypeError: If *seed* is None -- same reason as `set_global_seed`.
    """
    if seed is None:
        raise TypeError("make_generator(None): a DataLoader generator needs a seed.")
    return torch.Generator().manual_seed(int(seed))


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` that reseeds numpy and random inside each worker.

    Without this, every DataLoader worker process inherits the parent's numpy
    seed and any numpy randomness in the dataset is duplicated across workers.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def derive_seed(seed: int, role: str) -> int:
    """Return a stable sub-seed for one named consumer of the run's *seed*.

    Two DataLoaders in the same run must not walk the same permutation just
    because they were built from the same run seed, and they must not share one
    mutable ``torch.Generator`` either -- a shared generator's state advances as
    each loader draws from it, so the batch order of the second loader would
    depend on how many loaders were CONSTRUCTED before it.  Adding a val loader
    would then silently change the train batch order and the run would stop
    reproducing its own published number.

    The mix is SHA256, not ``hash()``: Python's string hash is salted per
    process, so ``hash()`` would give a different sub-seed on every invocation
    and nothing would be reproducible at all.

    Args:
        seed: The run's seed, as returned by `set_global_seed`.
        role: A short stable name for the consumer, e.g. ``"train_loader"``.
            It is part of the mix, so renaming a role changes its stream.

    Returns:
        A non-negative int below 2**31, suitable for ``manual_seed``.

    Raises:
        TypeError: If *seed* is None, or *role* is not a string.
        ValueError: If *role* is empty.
    """
    if seed is None:
        raise TypeError(
            "derive_seed(None, ...): this run has no seed, so its DataLoader "
            "order cannot be reproduced. Seed the run with set_global_seed "
            "first and pass the seed it returns."
        )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}: {seed!r}")
    if not isinstance(role, str):
        raise TypeError(f"role must be a str, got {type(role).__name__}: {role!r}")
    if not role:
        raise ValueError("role must be a non-empty name, e.g. 'train_loader'")
    digest = hashlib.sha256(f"{int(seed)}:{role}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


def loader_kwargs(seed: int, role: str) -> dict:
    """Return the ``DataLoader`` kwargs that tie batch order to *seed*.

    Splat it into every ``DataLoader`` construction::

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  **loader_kwargs(seed, "train_loader"))

    Why both keys are needed
    ------------------------
    ``generator=`` is what a shuffling DataLoader actually draws its
    permutation from.  Without it the sampler falls back to the global torch
    RNG, so the batch order depends on how much *other* code consumed from that
    RNG first -- adding a dropout layer above the loader would reshuffle the
    epochs.

    ``worker_init_fn=`` is inert while ``num_workers=0`` (PyTorch never calls
    it), and every loader in this repo is currently num_workers=0.  It is
    passed anyway because the day someone raises num_workers for speed, each
    worker process re-seeds numpy and ``random`` from a base seed derived from
    ``generator``; without the init fn those libraries keep the parent's state
    in every worker and the run stops being a function of its seed.  That
    failure is silent: the run still trains and still writes a number.

    Passing ``generator=`` to a non-shuffling loader is not pointless either --
    it is what seeds the workers' base seed.

    Args:
        seed: The run's seed.
        role: Stable name for this loader, unique within the trainer.

    Returns:
        ``{"generator": <seeded torch.Generator>, "worker_init_fn": seed_worker}``

    Raises:
        TypeError, ValueError: Propagated from `derive_seed`; an unseeded
            loader is never silently allowed.
    """
    return {
        "generator": make_generator(derive_seed(seed, role)),
        "worker_init_fn": seed_worker,
    }


def seed_scoped(path: str | Path, seed: int | None, *, is_dir: bool = False) -> Path:
    """Insert a ``seed<k>`` component into *path* so seeds cannot collide.

    Files gain a directory named after the file stem and are renamed to the
    seed; directories gain a ``seed<k>`` child::

        seed_scoped("results/checkpoints/arm.pth", 3)
            -> results/checkpoints/arm/seed3.pth
        seed_scoped("results/figures/arm/", 3, is_dir=True)
            -> results/figures/arm/seed3

    Args:
        path: The path exactly as the config gives it.
        seed: The run's seed, or None.
        is_dir: True when *path* names a directory.

    Returns:
        The seed-scoped path, or *path* unchanged when *seed* is None.

    Note:
        Returning *path* unchanged for ``seed=None`` is what keeps every
        result produced before this change readable by the evaluators, which
        look for ``results/figures/<arm>/eval_metrics.json``.  Seed scoping
        only switches on once a run actually declares a seed on the CLI.
    """
    path = Path(path)
    if seed is None:
        return path
    if is_dir:
        return path / f"seed{int(seed)}"
    return path.parent / path.stem / f"seed{int(seed)}{path.suffix}"
