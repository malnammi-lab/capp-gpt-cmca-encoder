"""Does `evaluate_link.py` write a provenance block the tier gate accepts?

Why this file exists
--------------------
`training-regiment.md` section 1 requires every metrics file to carry seed,
epoch, config hash, code version, GPU and wall time; the selection rule (20 Sep 2026)
added `selected_epoch` and `stopping_reason`. `runners/run_all.py` and
`runners/check_tier.py` both run `check_regiment_fields()` over the block and
call a run **not quotable** when anything is missing.

`evaluate_semantic.py` and `evaluate_unified.py` have written the block since
17 Sep. `evaluate_link.py` did not -- its `link_eval_metrics.json` carried
`experiment_name`, the two checkpoint paths, the pair counts and the metric
rows, and no provenance at all. A link run therefore trained fine, evaluated
fine, exited 0, and was rejected at the gate. Fixed 21 Sep 2026.

The mutation-test lesson from `test_run_integration.py` applies here too: a
requirement that only the helper honours is a requirement nothing enforces.
These tests train a real (tiny) two-stage model on the synthetic corpus and
assert on the JSON that comes out, so deleting the `metrics["provenance"] = ...`
line fails them. They do not test `check_regiment_fields` -- `test_seeding.py`
already does -- they test that the link evaluator's output survives it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.evaluation.evaluate_link import evaluate_link_predictor
from src.evaluation.evaluate_semantic import evaluate_semantic_model
from src.training.train_link import train_link_predictor
from src.training.train_semantic import train_semantic_model
from src.utils.provenance import check_regiment_fields

REPO = Path(__file__).resolve().parents[1]
SEED = 5


def _dataset_block(corpus):
    return {
        "json_dir": str(corpus["json_dir"]),
        "split_file": str(corpus["split_file"]),
        "norm_stats": str(corpus["norm_stats"]),
        "n_classes": corpus["n_classes"],
        "attributes": "standardized",
    }


def _semantic_config(corpus, out_dir):
    """Stage 1. Small enough to train in seconds, real enough to checkpoint."""
    return {
        "experiment_name": "link-prov-stage1",
        "seed": 999,                      # must lose to seed_override
        "dataset": _dataset_block(corpus),
        "bias": {"type": "both", "lambda_adj": 1.0, "lambda_sim": 1.0},
        "model": {"type": "semantic_segmentation", "version": "v1",
                  "d_model": 32, "n_heads": 2, "n_layers": 1,
                  "dim_feedforward": 32, "dropout": 0.1},
        "label_encoder": {"embedding_dim": 32},
        "loss": {"type": "prototype_infonce", "temperature": 0.07,
                 "use_class_weights": False},
        "training": {"max_epochs": 2, "patience": 2, "batch_size": 2,
                     "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0},
        "paths": {"checkpoint": str(out_dir / "sem.pth"),
                  "figures": str(out_dir / "figs_sem") + "/"},
        "device": "cpu",
    }


def _link_config(corpus, out_dir, sem_ckpt):
    """Stage 2, sharing Stage 1's dataset view so the frozen encoder matches."""
    return {
        "experiment_name": "link-prov-stage2",
        "seed": 999,
        "dataset": _dataset_block(corpus),
        "bias": {"type": "both", "lambda_adj": 1.0, "lambda_sim": 1.0},
        "semantic_checkpoint": str(sem_ckpt),
        "semantic_model": {"d_model": 32, "n_heads": 2, "n_layers": 1,
                           "dim_feedforward": 32, "dropout": 0.1},
        "link_model": {"d_model": 32, "n_heads": 2, "n_layers": 1,
                       "dim_feedforward": 32, "dropout": 0.1},
        "loss": {"type": "cross_entropy", "use_class_weights": False},
        "training": {"max_epochs": 2, "patience": 2, "batch_size": 2,
                     "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0},
        "paths": {"link_checkpoint": str(out_dir / "link.pth"),
                  "link_figures": str(out_dir / "figs_link") + "/",
                  "figures": str(out_dir / "figs_link") + "/"},
        "device": "cpu",
    }


@pytest.fixture(scope="module")
def two_stage(corpus, tmp_path_factory):
    """Train Stage 1, then Stage 2 on top of it, both at an explicit seed."""
    out = tmp_path_factory.mktemp("linkprov")

    sem_cfg = _semantic_config(corpus, out)
    train_semantic_model(sem_cfg, seed_override=SEED)
    sem_ckpt = out / "sem" / f"seed{SEED}.pth"
    assert sem_ckpt.is_file(), f"Stage 1 wrote no checkpoint at {sem_ckpt}"

    link_cfg = _link_config(corpus, out, sem_ckpt)
    train_link_predictor(link_cfg, seed_override=SEED)
    link_ckpt = out / "link" / f"seed{SEED}.pth"
    assert link_ckpt.is_file(), f"Stage 2 wrote no checkpoint at {link_ckpt}"

    return {"out": out, "sem_cfg": sem_cfg, "link_cfg": link_cfg,
            "sem_ckpt": sem_ckpt, "link_ckpt": link_ckpt}


def _evaluate_into(two_stage, fig_dir, link_ckpt=None):
    """Run the link evaluator with figures redirected to *fig_dir*."""
    cfg = dict(two_stage["link_cfg"])
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["figures"] = str(fig_dir) + "/"
    evaluate_link_predictor(
        cfg, str(link_ckpt or two_stage["link_ckpt"]),
        semantic_checkpoint_path=str(two_stage["sem_ckpt"]),
    )
    metrics_file = Path(fig_dir) / "link_eval_metrics.json"
    assert metrics_file.is_file(), (
        f"the link evaluator wrote no link_eval_metrics.json into {fig_dir}")
    return json.loads(metrics_file.read_text(encoding="utf-8"))


# --------------------------------------------------------------------- #
#  the defect: link_eval_metrics.json had no provenance at all           #
# --------------------------------------------------------------------- #

def test_link_eval_metrics_carries_complete_provenance(two_stage, tmp_path):
    """Catches: evaluate_link stops writing metrics["provenance"]."""
    metrics = _evaluate_into(two_stage, tmp_path)

    assert "provenance" in metrics, (
        "link_eval_metrics.json has no provenance block. check_tier.py runs "
        "check_regiment_fields() over it and calls the run 'not quotable', so "
        "the link numbers are dead on arrival at the gate.")
    missing = check_regiment_fields(metrics["provenance"])
    assert missing == [], f"link eval provenance missing: {missing}"


def test_link_provenance_describes_the_training_run_not_the_scoring_pass(
        two_stage, tmp_path):
    """The six fields belong to the run that made the weights."""
    prov = _evaluate_into(two_stage, tmp_path)["provenance"]
    ckpt_prov = torch.load(two_stage["link_ckpt"], map_location="cpu",
                           weights_only=False)["provenance"]

    assert prov["seed"] == SEED, (
        f"provenance records seed {prov['seed']!r}; check_tier.py compares this "
        f"against the seed directory and calls a mismatch mislabelled")
    for key in ("seed", "epoch", "selected_epoch", "stopping_reason",
                "config_hash", "wall_time_s"):
        assert prov[key] == ckpt_prov[key], (
            f"{key} was re-derived from the scoring machine instead of copied "
            f"from the link checkpoint")
    assert prov["stopping_reason"] in ("early_stop", "cap")
    assert "evaluated" in prov, (
        "no `evaluated` sub-block: nothing records which code scored the run")
    assert prov["evaluated"]["code_fingerprint"]


def test_link_provenance_has_the_same_shape_as_the_semantic_evaluator(
        two_stage, tmp_path):
    """Same block, not a link-shaped variant of it.

    The gate reads one key by one name. If evaluate_link invented its own
    layout -- a nested block, a different key, extra required fields -- the gate
    would pass on this file and mean something different by it.
    """
    link_prov = _evaluate_into(two_stage, tmp_path / "link")["provenance"]

    sem_cfg = dict(two_stage["sem_cfg"])
    sem_cfg["paths"] = dict(sem_cfg["paths"])
    sem_cfg["paths"]["figures"] = str(tmp_path / "sem") + "/"
    evaluate_semantic_model(sem_cfg, str(two_stage["sem_ckpt"]))
    sem_prov = json.loads((tmp_path / "sem" / "eval_metrics.json")
                          .read_text(encoding="utf-8"))["provenance"]

    assert set(link_prov) == set(sem_prov), (
        f"link provenance keys differ from the semantic evaluator's.\n"
        f"  only in link:     {sorted(set(link_prov) - set(sem_prov))}\n"
        f"  only in semantic: {sorted(set(sem_prov) - set(link_prov))}")
    assert set(link_prov["evaluated"]) == set(sem_prov["evaluated"])
    # the fingerprint is of the code, so both evaluators must agree on it
    assert link_prov["code_fingerprint"] == sem_prov["code_fingerprint"]


def test_link_eval_of_a_provenance_free_checkpoint_is_marked_unquotable(
        two_stage, tmp_path):
    """A pre-17-Sep checkpoint must not acquire this machine's provenance."""
    ckpt = torch.load(two_stage["link_ckpt"], map_location="cpu",
                      weights_only=False)
    del ckpt["provenance"]
    stripped = tmp_path / "old_link.pth"
    torch.save(ckpt, stripped)

    prov = _evaluate_into(two_stage, tmp_path / "out",
                          link_ckpt=stripped)["provenance"]
    assert "provenance_missing" in prov
    assert "seed" not in prov, "a seed was invented for a checkpoint with none"
    assert check_regiment_fields(prov) != [], (
        "an unprovenanced checkpoint produced a block the gate accepts")


# --------------------------------------------------------------------- #
#  the gate itself, on the real file                                     #
# --------------------------------------------------------------------- #

def test_run_all_gate_accepts_the_link_metrics_file(two_stage, tmp_path):
    """Drive run_all.py's own discovery + gate logic over the produced file.

    `run_all.metrics_path()` is what decides which file in a run directory is
    "the metrics file", and the block immediately after it is the quotability
    gate. This runs both against a real link run directory rather than a
    hand-written dict.
    """
    spec = importlib.util.spec_from_file_location(
        "run_all_for_linkprov", REPO / "runners" / "run_all.py")
    run_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all)

    run_dir = tmp_path / "linkarm" / f"seed{SEED}"
    run_dir.mkdir(parents=True)
    _evaluate_into(two_stage, run_dir)

    mp = run_all.metrics_path(run_dir)
    assert mp is not None and mp.name == "link_eval_metrics.json", (
        f"run_all.metrics_path did not find the link metrics file in {run_dir}")

    metrics = json.loads(mp.read_text(encoding="utf-8"))
    prov = metrics.get("provenance", {})
    missing = check_regiment_fields(prov) if prov else ["provenance (absent)"]
    assert missing == [], (
        f"run_all would report this link run UNQUOTABLE -- missing {missing}")
