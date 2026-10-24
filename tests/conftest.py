"""Shared fixtures.

Every fixture builds a tiny synthetic corpus in the exact schema the extractors
write, so the suite runs in seconds and needs neither the real corpora nor a GPU.
The schema is pinned in `datasets/MANIFEST.md`; if it changes there, these
fixtures must change with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# the package is imported as `src.*`, so the repo root must be importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N_CLASSES = 4
GRID = 8
UV_CHANNELS = 7
ATTR_DIM = 4


def _one_model(rng, mid, *, n_faces=None, with_labels=True):
    n_f = int(n_faces if n_faces is not None else rng.integers(4, 9))
    n_e = int(rng.integers(5, 13))

    adj = (rng.random((n_f, n_f)) < 0.35).astype(float)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)

    sim = rng.random((n_f, n_f))
    sim = (sim + sim.T) / 2.0

    # column 0 MUST be type_index / 7 -- _one_hot_type recovers the index by
    # round(x * 7), and the whole one-hot path depends on that invariant
    faces_attr = np.column_stack([
        rng.integers(0, 8, n_f) / 7.0,
        rng.random(n_f) * 12.0,
        rng.random(n_f),
        rng.random(n_f),
    ])
    edges_attr = np.column_stack([
        rng.integers(0, 8, n_e) / 7.0,
        rng.random(n_e) * 6.0,
        rng.random(n_e),
        rng.random(n_e),
    ])

    doc = {
        "model_id": mid,
        "n_faces": n_f,
        "n_edges": n_e,
        "uv_grid_size": [GRID, GRID],
        "faces_uv": rng.random((n_f, UV_CHANNELS, GRID, GRID)).tolist(),
        "faces_attr_raw": faces_attr.tolist(),
        "edges_attr_raw": edges_attr.tolist(),
        "faces_attr": np.clip(faces_attr, 0, 1).tolist(),
        "edges_attr": np.clip(edges_attr, 0, 1).tolist(),
        "adjacency": adj.tolist(),
        "attr_similarity": sim.tolist(),
        "semantic_labels": rng.integers(0, N_CLASSES, n_f).tolist(),
    }
    if with_labels:
        inst_of = rng.integers(0, 2, n_f)
        doc["instance_labels"] = (inst_of[:, None] == inst_of[None, :]).astype(float).tolist()
        doc["bottom_labels"] = rng.integers(0, 2, n_f).tolist()
    return doc


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """A 10-model synthetic corpus plus a split file and train-only norm stats.

    Returns a dict of paths: json_dir, split_file, norm_stats.
    """
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("corpus")
    jdir = root / "features"
    jdir.mkdir()

    ids = []
    for i in range(10):
        mid = f"{i:08d}"
        ids.append(mid)
        (jdir / f"{mid}_features.json").write_text(json.dumps(_one_model(rng, mid)))

    # Train, validation, and test. The test ids are ids[7:].
    split = root / "split.json"
    split.write_text(json.dumps(
        {"train": ids[:6], "val": ids[6:7], "test": ids[7:]}))

    # Statistics measured on the train ids only.
    f_rows, e_rows = [], []
    for mid in ids[:6]:
        raw = json.loads((jdir / f"{mid}_features.json").read_text())
        f_rows.append(np.array(raw["faces_attr_raw"]))
        e_rows.append(np.array(raw["edges_attr_raw"]))
    # The fixture writes the same fields as runners/compute_norm_stats.py.
    # n_models is the size of the train partition these statistics cover.
    stats = {
        "corpus": str(jdir),
        "split_file": str(split),
        "split_used": "train",
        "n_models": len(ids[:6]),
        "n_models_missing": 0,
        "stats": {},
    }
    for key, rows in (("faces_attr_raw", f_rows), ("edges_attr_raw", e_rows)):
        m = np.concatenate(rows)
        sd = m.std(0)
        stats["stats"][key] = {
            "n": int(len(m)),
            "mean": m.mean(0).tolist(),
            "std": sd.tolist(),
            "std_safe": np.where(sd < 1e-8, 1.0, sd).tolist(),
        }
    norm = root / "norm.json"
    norm.write_text(json.dumps(stats))

    return {"json_dir": jdir, "split_file": split, "norm_stats": norm,
            "train_ids": ids[:6], "val_ids": ids[6:7], "test_ids": ids[7:],
            "n_classes": N_CLASSES}


@pytest.fixture
def dataset(corpus):
    """Default dataset view: standardized attributes, combined bias."""
    from src.data.brep_dataset import CMCADataset
    return CMCADataset(
        json_dir=corpus["json_dir"],
        n_classes=corpus["n_classes"],
        attributes="standardized",
        norm_stats=corpus["norm_stats"],
        bias_type="both",
        verbose=False,
    )


@pytest.fixture
def batch(dataset):
    """One collated batch of 3 models with differing face counts."""
    from src.data.collate import collate_fn_brep
    return collate_fn_brep([dataset[i] for i in range(3)])
