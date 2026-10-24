#!/usr/bin/env python3
"""
make_splits.py -- write reproducible train/val/test split files for a corpus.

Training, evaluation and the normalisation statistics all read the same list
of model ids.

The split is 70/10/20.
Every tier selects its model on the validation partition: early stopping on
validation loss, patience 10, a max-epoch cap of 60, and the best-validation-loss
checkpoint is the one evaluated. Thresholds are tuned on validation.

The val slice is carved out of the *training* portion, so the test partition is
untouched. That is a property of `split()` below, not a convention: `test` is
`shuffled[:n_test]` and depends only on the sorted id list and the seed.
The test ids must stay fixed, and `tests/test_dataset.py` asserts it. Nothing
leaks across the boundary and the frozen test set stays fixed.

    python runners/make_splits.py \
        --corpus ../data/cadsynth \
        --out    datasets/splits/cadsynth_70_10_20_seed42.json \
        --test-fraction 0.2 --val-fraction 0.1 --seed 42
        # -> 4244 train / 606 val / 1212 test  of 6062

    python runners/make_splits.py \
        --corpus ../data/mfinstaseg \
        --out    datasets/splits/mfinstseg_70_10_20_seed42.json \
        --test-fraction 0.2 --val-fraction 0.1 --seed 42
        # -> 4116 train / 588 val / 1176 test  of 5880

`--val-fraction` defaults to 0.0, so train/test-only behaviour is reachable,
but no paper number may be produced that way.

Pure numpy and the standard library. No pythonocc and no torch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

SUFFIX = "_features.json"


def model_ids(corpus: Path) -> List[str]:
    """Every model id in the corpus, sorted.

    Sorting first makes the split depend only on the seed and the set of files,
    not on filesystem enumeration order -- which differs between Drive, local
    disk and a zip extract, and would otherwise silently change the split.
    """
    ids = sorted(p.name[: -len(SUFFIX)] for p in corpus.glob("*" + SUFFIX))
    return ids


def split(ids: List[str], test_fraction: float, val_fraction: float,
          seed: int) -> dict:
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(round(test_fraction * n))
    n_val = int(round(val_fraction * n))
    if n_test + n_val >= n:
        raise ValueError(
            f"test+val ({n_test}+{n_val}) leaves no training models out of {n}")

    test = sorted(shuffled[:n_test])
    val = sorted(shuffled[n_test:n_test + n_val])
    train = sorted(shuffled[n_test + n_val:])

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "test_fraction": test_fraction,
        "val_fraction": val_fraction,
        "n_total": n,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train": train,
        "test": test,
    }
    if val:
        out["val"] = val
    return out


def verify(payload: dict) -> List[str]:
    """A split file is only useful if it is actually disjoint. Check it."""
    problems = []
    train = set(payload["train"])
    test = set(payload["test"])
    val = set(payload.get("val", []))

    if train & test:
        problems.append(f"{len(train & test)} ids in both train and test")
    if train & val:
        problems.append(f"{len(train & val)} ids in both train and val")
    if val & test:
        problems.append(f"{len(val & test)} ids in both val and test")
    total = len(train) + len(val) + len(test)
    if total != payload["n_total"]:
        problems.append(
            f"partition covers {total} ids but corpus has {payload['n_total']}")
    if len(train) != len(payload["train"]):
        problems.append("duplicate ids inside train")
    if len(test) != len(payload["test"]):
        problems.append("duplicate ids inside test")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Write a reproducible train/test split file for a corpus.")
    p.add_argument("--corpus", required=True,
                   help="directory of *_features.json")
    p.add_argument("--out", required=True, help="split file to write (.json)")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--val-fraction", type=float, default=0.0,
                   help="carve a val slice out of the TRAINING portion "
                        "(default 0 = train/test only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing split file")
    args = p.parse_args(argv)

    corpus = Path(args.corpus)
    if not corpus.is_dir():
        print(f"corpus not found: {corpus}", file=sys.stderr)
        return 1

    ids = model_ids(corpus)
    if not ids:
        print(f"no *{SUFFIX} under {corpus}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists. Refusing to overwrite -- a changed "
              f"split silently invalidates every number computed against the "
              f"old one. Pass --force if that is what you want.", file=sys.stderr)
        return 1

    payload = split(ids, args.test_fraction, args.val_fraction, args.seed)
    payload["corpus"] = str(corpus)

    problems = verify(payload)
    if problems:
        print("split verification FAILED: " + "; ".join(problems), file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"corpus : {corpus}  ({payload['n_total']} models)")
    print(f"seed   : {payload['seed']}")
    print(f"train  : {payload['n_train']:6d}  "
          f"({100 * payload['n_train'] / payload['n_total']:.1f}%)")
    if payload["n_val"]:
        print(f"val    : {payload['n_val']:6d}  "
              f"({100 * payload['n_val'] / payload['n_total']:.1f}%)")
    print(f"test   : {payload['n_test']:6d}  "
          f"({100 * payload['n_test'] / payload['n_total']:.1f}%)")
    print("disjoint + complete: verified")
    print(f"first 3 train: {payload['train'][:3]}")
    print(f"first 3 test : {payload['test'][:3]}")
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
