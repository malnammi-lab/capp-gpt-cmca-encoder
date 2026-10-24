"""
End-to-end tests: do the trainers and evaluators actually USE the seed layer?

Why this file exists
--------------------
An audit on 2026-09-17 mutation-tested the then-green 130-test suite. Four
mutations were caught. **Three survived**, all of them in production code:

  * `train_semantic.py` stops calling `seed_scoped`  -- the exact defect that
    silently destroys 16 of tier T1's 20 runs               -> 130 passed
  * `build_provenance` stubbed to `{}` in the trainer -- every checkpoint then
    carries an empty provenance block                        -> 128 passed
  * `evaluate_semantic.py` stops writing `metrics["provenance"]` -- metrics
    files go back to carrying 0 of the 6 required fields     -> 130 passed

The verdict was blunt and correct: every test in `test_seeding.py` exercises the
helper module in isolation, and not one asserts that a caller uses it. A
requirement that only the helper honours is a requirement nothing enforces.

These tests train and evaluate a real model on the synthetic corpus fixture and
then assert on the files that come out. They are slower than a unit test and
that is the point: they fail if the wiring is removed, whatever the helpers do.
"""

from __future__ import annotations

import json

import pytest
import torch

from src.evaluation.evaluate_semantic import evaluate_semantic_model
from src.training.train_semantic import train_semantic_model
from src.utils.provenance import check_regiment_fields

SEED = 3


def _config(corpus, out_dir):
    """A complete, minimal training config pointing at the synthetic corpus.

    Paths are deliberately left UNSCOPED, exactly as the real T1 configs are, so
    that the scoping under test is the trainer's doing and not the fixture's.
    """
    return {
        "experiment_name": "integration",
        "seed": 999,                      # must lose to seed_override
        "dataset": {
            "json_dir": str(corpus["json_dir"]),
            "split_file": str(corpus["split_file"]),
            "norm_stats": str(corpus["norm_stats"]),
            "n_classes": corpus["n_classes"],
            "attributes": "standardized",
        },
        "bias": {"type": "both", "lambda_adj": 1.0, "lambda_sim": 1.0},
        "model": {"type": "semantic_segmentation", "version": "v1",
                  "d_model": 32, "n_heads": 2, "n_layers": 1,
                  "dim_feedforward": 32, "dropout": 0.1},
        "label_encoder": {"embedding_dim": 32},
        "loss": {"type": "prototype_infonce", "temperature": 0.07,
                 "use_class_weights": False},
        # Three epochs and patience 2 let the loop select a checkpoint.
        "training": {"max_epochs": 3, "patience": 2, "batch_size": 2,
                     "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0},
        "paths": {"checkpoint": str(out_dir / "ckpt.pth"),
                  "figures": str(out_dir / "figs") + "/"},
        "device": "cpu",
    }


@pytest.fixture(scope="module")
def trained(corpus, tmp_path_factory):
    """Train one real model for one epoch at an explicit seed."""
    out = tmp_path_factory.mktemp("integration")
    cfg = _config(corpus, out)
    train_semantic_model(cfg, seed_override=SEED)
    return {"cfg": cfg, "out": out}


# --------------------------------------------------------------------- #
#  the 16-of-20 defect: does the TRAINER scope its output paths?         #
# --------------------------------------------------------------------- #

def test_trainer_writes_to_a_seed_scoped_checkpoint(trained):
    """Catches: train_semantic stops calling seed_scoped."""
    out = trained["out"]
    scoped = out / "ckpt" / f"seed{SEED}.pth"
    unscoped = out / "ckpt.pth"
    assert scoped.is_file(), (
        f"checkpoint was not written to the seed-scoped path {scoped}. "
        f"Five seeds of one arm would land on the same file and 16 of tier T1's "
        f"20 runs would be destroyed silently.")
    assert not unscoped.exists(), (
        f"checkpoint was ALSO written to the unscoped path {unscoped}")


def test_trainer_scopes_its_figure_directory(trained):
    out = trained["out"]
    assert (out / "figs" / f"seed{SEED}").is_dir()


def test_seed_override_beats_the_config(trained):
    """The config says 999; --seed said 3. The run must be seed 3."""
    ckpt = torch.load(trained["out"] / "ckpt" / f"seed{SEED}.pth",
                      map_location="cpu", weights_only=False)
    assert ckpt["provenance"]["seed"] == SEED


# --------------------------------------------------------------------- #
#  does the TRAINER record provenance?                                   #
# --------------------------------------------------------------------- #

def test_checkpoint_carries_complete_provenance(trained):
    """Catches: build_provenance stubbed out in the trainer."""
    ckpt = torch.load(trained["out"] / "ckpt" / f"seed{SEED}.pth",
                      map_location="cpu", weights_only=False)
    assert "provenance" in ckpt, "checkpoint has no provenance block at all"
    missing = check_regiment_fields(ckpt["provenance"])
    assert missing == [], f"checkpoint provenance missing: {missing}"
    p = ckpt["provenance"]
    # The selected epoch is whichever validation loss won, within the cap.
    assert 1 <= p["epoch"] <= 3, f"epoch {p['epoch']} outside the cap"
    assert p["selected_epoch"] == p["epoch"], (
        "the checkpoint on disk must be the epoch provenance says was selected")
    assert p["stopping_reason"] in ("early_stop", "cap")
    assert p["config_hash"] and p["code_fingerprint"]
    assert isinstance(p["wall_time_s"], float) and p["wall_time_s"] >= 0.0


# --------------------------------------------------------------------- #
#  does the EVALUATOR record provenance?                                 #
# --------------------------------------------------------------------- #

def test_eval_metrics_carries_complete_provenance(trained, tmp_path):
    """Catches: evaluate_semantic stops writing metrics["provenance"]."""
    cfg = dict(trained["cfg"])
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["figures"] = str(tmp_path) + "/"          # score into a clean dir
    ckpt_path = trained["out"] / "ckpt" / f"seed{SEED}.pth"

    evaluate_semantic_model(cfg, str(ckpt_path))

    metrics_file = tmp_path / "eval_metrics.json"
    assert metrics_file.is_file(), "evaluator wrote no eval_metrics.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

    assert "provenance" in metrics, (
        "eval_metrics.json has no provenance block. Under training-regiment.md "
        "section 4 every cell must be traceable to seed/config/commit, so this "
        "file's numbers are not quotable.")
    missing = check_regiment_fields(metrics["provenance"])
    assert missing == [], f"eval provenance missing: {missing}"
    # it must describe the TRAINING run, not this scoring pass
    assert metrics["provenance"]["seed"] == SEED
    assert "evaluated" in metrics["provenance"]


def test_eval_of_a_provenance_free_checkpoint_is_marked_unquotable(
        trained, tmp_path):
    """An old checkpoint must not silently acquire this machine's provenance."""
    ckpt = torch.load(trained["out"] / "ckpt" / f"seed{SEED}.pth",
                      map_location="cpu", weights_only=False)
    del ckpt["provenance"]                                   # simulate a pre-17-Sep file
    stripped = tmp_path / "old.pth"
    torch.save(ckpt, stripped)

    cfg = dict(trained["cfg"])
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["figures"] = str(tmp_path / "out") + "/"
    evaluate_semantic_model(cfg, str(stripped))

    metrics = json.loads((tmp_path / "out" / "eval_metrics.json")
                         .read_text(encoding="utf-8"))
    prov = metrics["provenance"]
    assert "provenance_missing" in prov
    assert "seed" not in prov, "a seed was invented for a checkpoint that had none"
    assert check_regiment_fields(prov) != []
