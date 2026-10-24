#!/usr/bin/env python3
"""
compute_norm_stats.py -- per-column mean/std over the TRAINING SPLIT ONLY.

These are the constants for `attributes: standardized`, i.e.

    x_standardised = (x_raw - mean) / std

computed once, stored, and applied identically at train and test time.

Why not just use `raw_feature_stats.json` from the extractor: that file was
accumulated over every model the extraction wrote, which includes the models
that end up in the test split. Fitting preprocessing statistics on test data is
leakage -- mild, but exactly the kind of thing a reviewer flags, and the JIM
submission was already rejected once on rigor. This script reads the split file
and touches training models only.

Why not BatchNorm: you can, and it learns the same thing. But `collate_fn_brep`
pads each part up to `max_faces` with zero rows, so a BatchNorm that sees those
rows has its statistics dragged toward zero by however ragged the batch happened
to be. Frozen statistics computed here have no batch-size sensitivity, no
padding hazard, are bit-for-bit reproducible, and can be printed in the paper.

Both views are measured, so `attributes: scaled | raw | standardized` can be run
as an ablation without recomputing anything:

  faces_attr_raw / edges_attr_raw   unit-cube units, unclipped   <- recommended
  faces_attr     / edges_attr       divided by constants, clipped

Streams one file at a time -- constant memory regardless of corpus size.

Pure numpy and the standard library:

    python runners/compute_norm_stats.py \
        --corpus <corpus-dir> \
        --split  datasets/splits/mfinstseg_70_10_20_seed42.json \
        --out    datasets/splits/mfinstseg_norm_train.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

SUFFIX = "_features.json"

FACE_COLS = ("type", "area", "loops", "degree")
EDGE_COLS = ("type", "length", "convexity", "angle")

# Faces with a degenerate surface can come back from OCC's GProp with a tiny
# negative area (observed: -1e-9 on CADSynth). Negative area is meaningless and
# would poison the mean, so clamp. Applied here and in the dataset loader.
CLAMP_NON_NEGATIVE = {"faces_attr_raw": [1], "faces_attr": [1],
                      "edges_attr_raw": [1], "edges_attr": [1]}


class Accumulator:
    """Streaming mean/std/min/max for a (n, 4) matrix.

    Uses a shifted-mean sum of squares. Naive sum(x^2) loses precision badly
    when the mean is far from zero and n is large; here n reaches ~10^5 with
    values spanning four orders of magnitude, so it matters.
    """

    def __init__(self, dim: int = 4) -> None:
        self.n = 0
        self.dim = dim
        self.shift = None
        self.s = np.zeros(dim, dtype=np.float64)
        self.ss = np.zeros(dim, dtype=np.float64)
        self.lo = np.full(dim, np.inf)
        self.hi = np.full(dim, -np.inf)

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.dim)
        if self.shift is None:
            self.shift = x[0].copy()
        d = x - self.shift
        self.n += x.shape[0]
        self.s += d.sum(axis=0)
        self.ss += (d ** 2).sum(axis=0)
        self.lo = np.minimum(self.lo, x.min(axis=0))
        self.hi = np.maximum(self.hi, x.max(axis=0))

    def result(self) -> Dict[str, List[float]]:
        if self.n == 0:
            return {"n": 0}
        mean = self.shift + self.s / self.n
        var = np.maximum(self.ss / self.n - (self.s / self.n) ** 2, 0.0)
        std = np.sqrt(var)
        # A constant column has std 0; dividing by it produces inf/nan. Substitute
        # 1.0 so the column passes through as (x - mean) = 0 instead of exploding.
        std_safe = np.where(std < 1e-12, 1.0, std)
        return {
            "n": int(self.n),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "std_safe": std_safe.tolist(),
            "min": self.lo.tolist(),
            "max": self.hi.tolist(),
            "degenerate_columns": [i for i, v in enumerate(std) if v < 1e-12],
        }


def _repo_relative(path) -> str:
    """`path` as a repo-relative POSIX string, or its basename if it lies
    outside the repo. Never returns an absolute path."""
    p = Path(path).resolve()
    repo = Path(__file__).resolve().parents[1]
    try:
        return p.relative_to(repo).as_posix()
    except ValueError:
        return p.name


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Per-column mean/std over the training split only.")
    p.add_argument("--corpus", required=True)
    p.add_argument("--split", required=True, help="split file from make_splits.py")
    p.add_argument("--out", required=True)
    p.add_argument("--which", default="train",
                   choices=["train", "val", "test", "all"],
                   help="which split to measure (default train -- anything else "
                        "is for inspection only, never for standardisation)")
    args = p.parse_args(argv)

    corpus = Path(args.corpus)
    split = json.loads(Path(args.split).read_text())

    if args.which == "all":
        ids = sorted(set(split["train"]) | set(split.get("val", []))
                     | set(split["test"]))
    else:
        ids = split.get(args.which, [])
    if not ids:
        print(f"split has no '{args.which}' ids", file=sys.stderr)
        return 1
    if args.which != "train":
        print(f"WARNING: measuring '{args.which}'. Standardisation constants "
              f"must come from train only.", file=sys.stderr)

    keys = ["faces_attr_raw", "edges_attr_raw", "faces_attr", "edges_attr"]
    acc = {k: Accumulator(4) for k in keys}

    missing, read = 0, 0
    for i, mid in enumerate(ids):
        f = corpus / (mid + SUFFIX)
        if not f.exists():
            missing += 1
            continue
        d = json.loads(f.read_text())
        for k in keys:
            if k in d:
                arr = np.asarray(d[k], dtype=np.float64).reshape(-1, 4)
                for col in CLAMP_NON_NEGATIVE.get(k, []):
                    np.maximum(arr[:, col], 0.0, out=arr[:, col])
                acc[k].update(arr)
        read += 1
        if read == 1 or read % 500 == 0:
            print(f"  [{read}/{len(ids)}] {mid}", flush=True)

    if read == 0:
        print("no files read", file=sys.stderr)
        return 1

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": str(corpus),
        # Repo-relative, POSIX separators. These files are provenance meant to
        # travel: an absolute Windows path written here ends up byte-identical
        # on the Linux run machine, where it is both meaningless and a leak of
        # the local filesystem layout.
        #
        # Four fields below are LOAD-BEARING:
        #
        #   split_file        src/data/brep_dataset.check_norm_stats_provenance
        #                     CHECK 2 compares its basename against the split
        #                     the dataset is being built from, and raises when
        #                     they differ. A stats file without it raises too:
        #                     "carry no 'split_file' field".
        #   split_used        CHECK 3 uses it to pick the partition to count.
        #   n_models          CHECK 3 requires n_models + n_models_missing to
        #   n_models_missing  equal the size of that partition.
        #
        # So these four are the norm-stats guard's entire evidence. Do not drop
        # one, and do not hand-edit one: regenerate the file.
        "split_file": _repo_relative(args.split),
        "split_used": args.which,
        "seed": split.get("seed"),
        "n_models": read,
        "n_models_missing": missing,
        "note": "Standardise as (x - mean) / std_safe. std_safe substitutes 1.0 "
                "for constant columns so they pass through as 0 rather than "
                "inf. Area columns are clamped to >= 0 before measurement.",
        "columns": {"faces": list(FACE_COLS), "edges": list(EDGE_COLS)},
        "stats": {k: acc[k].result() for k in keys},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print(f"\nmeasured {read} '{args.which}' models"
          + (f" ({missing} missing)" if missing else ""))
    for k in keys:
        r = payload["stats"][k]
        if not r.get("n"):
            continue
        cols = FACE_COLS if k.startswith("faces") else EDGE_COLS
        print(f"\n{k}  (n={r['n']})")
        for c, name in enumerate(cols):
            flag = "  <- CONSTANT" if c in r["degenerate_columns"] else ""
            print(f"  {name:<10} mean {r['mean'][c]:10.5f}  std {r['std'][c]:10.5f}"
                  f"  min {r['min'][c]:10.5f}  max {r['max'][c]:10.5f}{flag}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
