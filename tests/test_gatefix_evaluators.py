"""Gate-fix tests for the two evaluators, 26 Sep 2026.

These lock the four defects listed for fixer "evaluators" in
`weekly/GATE-DEFECTS.md`'s companion brief:

  1. the ResNet banner line -- "(34338 models)" for 34,338 FACES of 1,212
     models, plus an `attributes=scaled, bias=similarity` pair invented by
     `.get(..., default)` from configs that carry neither key;
  2. `per_class[c]["accuracy"]` is per-class RECALL, undocumented;
  3. `instance.n_pairs` counts ORDERED pairs, undocumented;
  4. `experiment_name` is stale for `resnet_cadsynth_cmca` -- checked here to
     show the evaluator is NOT the cause.

Every test drives the real `evaluate_semantic_model` / `evaluate_unified_model`
with only the model, dataset and forward pass stubbed out, so it exercises the
real banner and the real metrics literal. The metric maths
(`compute_accuracy`, `compute_confusion_matrix`, `compute_mean_iou`,
`compute_f1_binary`, `compute_auc`) is NOT stubbed: a change that moved a
computed value would show up in `test_*_values_unchanged_*` below.

Nothing here writes inside `results/` -- every run's `paths.figures` is a
pytest `tmp_path`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_semantic as es  # noqa: E402
from src.evaluation import evaluate_unified as eu  # noqa: E402


N_CLASSES = 4


# --------------------------------------------------------------------- #
#  Fakes
# --------------------------------------------------------------------- #

class _FakeFaceDataset:
    """Stands in for UVGridLabelDataset: one sample per FACE."""

    def __init__(self, n_faces, **kwargs):
        self._n = n_faces
        self.kwargs = kwargs

    def __len__(self):
        return self._n


class _FakeModelDataset:
    """Stands in for PreExtractedBRepDataset: one sample per MODEL."""

    def __init__(self, n_models, n_classes=N_CLASSES, **kwargs):
        self._n = n_models
        self.n_classes = n_classes
        self.has_instance_labels = True
        self.has_bottom_labels = True
        self.kwargs = kwargs

    def __len__(self):
        return self._n


def _preds_and_gts():
    """A tiny, fixed prediction set with every class supported and errors in it.

    9 faces, 4 classes, deliberately imperfect so precision != recall != 1.
    """
    gts = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    preds = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int64)
    return preds, gts


# --------------------------------------------------------------------- #
#  Drivers
# --------------------------------------------------------------------- #

def _resnet_cfg(tmp_path, *, attributes=None, bias=None):
    """The shape of configs/resnet_cadsynth*.yaml: no bias block, no attributes."""
    cfg = {
        "experiment_name": "resnet_cadsynth",
        "device": "cpu",
        "dataset": {
            "name": "cadsynth_faces",
            "json_dir": "../data/cadsynth_faces/",
            "n_classes": N_CLASSES,
            "split_file": "datasets/splits/cadsynth_70_10_20_seed42.json",
        },
        "model": {"type": "resnet_encoder", "embedding_dim": 512},
        "loss": {"type": "prototype_infonce"},
        "training": {"batch_size": 128},
        "paths": {"figures": str(tmp_path)},
    }
    if attributes is not None:
        cfg["dataset"]["attributes"] = attributes
    if bias is not None:
        cfg["bias"] = {"type": bias}
    return cfg


def _graph_cfg(tmp_path, *, attributes="standardized", bias="both"):
    cfg = {
        "experiment_name": "cadsynth_bias_both",
        "device": "cpu",
        "dataset": {
            "name": "cadsynth",
            "json_dir": "../data/cadsynth/",
            "n_classes": N_CLASSES,
            "split_file": "datasets/splits/cadsynth_70_10_20_seed42.json",
        },
        "model": {"type": "semantic_segmentation", "d_model": 128},
        "loss": {"type": "prototype_infonce"},
        "training": {"batch_size": 4},
        "paths": {"figures": str(tmp_path)},
    }
    if attributes is not None:
        cfg["dataset"]["attributes"] = attributes
    if bias is not None:
        cfg["bias"] = {"type": bias}
    return cfg


def _run_semantic(monkeypatch, capsys, cfg, *, is_resnet, n_faces=9,
                  n_models=3):
    """Call the real `evaluate_semantic_model` with the heavy parts stubbed."""
    preds, gts = _preds_and_gts()
    assert len(gts) == n_faces

    monkeypatch.setattr(es, "load_checkpoint",
                        lambda p, device=None: {"n_classes": N_CLASSES,
                                                "epoch": 12})
    monkeypatch.setattr(es, "load_split",
                        lambda path, part: [f"m{i}" for i in range(n_models)])
    monkeypatch.setattr(es, "UVGridLabelDataset",
                        lambda **kw: _FakeFaceDataset(n_faces, **kw))
    monkeypatch.setattr(es, "PreExtractedBRepDataset",
                        lambda **kw: _FakeModelDataset(n_models, **kw))
    monkeypatch.setattr(es, "_build_resnet",
                        lambda cfg, ckpt, device: (object(), None, N_CLASSES))
    monkeypatch.setattr(es, "_build_graph_transformer",
                        lambda cfg, ckpt, device: (object(), None, N_CLASSES))
    monkeypatch.setattr(es, "_evaluate_resnet",
                        lambda *a, **k: (preds, gts))
    monkeypatch.setattr(es, "_evaluate_graph_transformer",
                        lambda *a, **k: (preds, gts))
    monkeypatch.setattr(es, "DataLoader", lambda *a, **k: "LOADER")
    monkeypatch.setattr(es, "plot_confusion_matrix", lambda *a, **k: None)
    monkeypatch.setattr(es, "provenance_from_checkpoint", lambda ckpt, cfg: {})

    metrics = es.evaluate_semantic_model(cfg, "ckpt.pth")
    out = capsys.readouterr().out
    return metrics, out


def _unified_cfg(tmp_path, *, attributes="standardized", bias="both"):
    cfg = {
        "experiment_name": "mfinstseg_unified",
        "device": "cpu",
        "dataset": {
            "name": "mfinstseg",
            "json_dir": "../data/mfinstseg/",
            "n_classes": N_CLASSES,
            "split_file": "datasets/splits/mfinstseg_70_10_20_seed42.json",
        },
        "model": {"type": "brep_encoder", "d_model": 128},
        "training": {"batch_size": 4},
        "paths": {"figures": str(tmp_path)},
    }
    if attributes is not None:
        cfg["dataset"]["attributes"] = attributes
    if bias is not None:
        cfg["bias"] = {"type": bias}
    return cfg


def _unified_out(n_models=3):
    """Three heads' worth of fake predictions.

    `inst_sims` / `inst_targets` have 12 entries: two models of 3 valid faces
    (3*2 = 6 off-diagonal ordered pairs each). Even by construction, exactly as
    `_run_test` builds them.
    """
    preds, gts = _preds_and_gts()
    bot_gts = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0], dtype=np.int64)
    bot_preds = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    sims = np.array([0.9, 0.2, 0.9, 0.1, 0.2, 0.1,
                     0.8, 0.3, 0.8, 0.4, 0.3, 0.4], dtype=np.float32)
    targets = np.array([1, 0, 1, 0, 0, 0,
                        1, 0, 1, 0, 0, 0], dtype=np.int64)
    return {
        "sem_preds": preds, "sem_gts": gts,
        "bot_preds": bot_preds, "bot_gts": bot_gts,
        "inst_sims": sims, "inst_targets": targets,
    }


def _run_unified(monkeypatch, capsys, cfg, *, n_models=3):
    fake_out = _unified_out(n_models)

    monkeypatch.setattr(eu, "load_checkpoint",
                        lambda p, device=None: {"n_classes": N_CLASSES,
                                                "epoch": 38,
                                                "optimal_threshold": 0.5})
    monkeypatch.setattr(eu, "_build_models",
                        lambda cfg, ckpt, device: (object(), object(),
                                                   object(), N_CLASSES))
    monkeypatch.setattr(
        eu, "_build_test_set",
        lambda cfg, seed: (_FakeModelDataset(n_models),
                           "split file %s (test partition)"
                           % cfg["dataset"]["split_file"]))
    monkeypatch.setattr(eu, "_run_test", lambda *a, **k: fake_out)
    monkeypatch.setattr(eu, "DataLoader", lambda *a, **k: "LOADER")
    monkeypatch.setattr(eu, "plot_confusion_matrix", lambda *a, **k: None)
    monkeypatch.setattr(eu, "plot_roc_curve", lambda *a, **k: None)
    monkeypatch.setattr(eu, "provenance_from_checkpoint", lambda ckpt, cfg: {})

    metrics = eu.evaluate_unified_model(cfg, "ckpt.pth")
    out = capsys.readouterr().out
    return metrics, out


def _banner(out):
    """The `Evaluating on test split (...)` line from a semantic run."""
    for line in out.splitlines():
        if line.startswith("Evaluating on test split"):
            return line
    raise AssertionError(
        "no 'Evaluating on test split' line in:\n" + out)


# ===================================================================== #
#  DEFECT 1 -- the banner line lies three times on the ResNet arms
# ===================================================================== #

def test_resnet_banner_counts_faces_not_models(monkeypatch, capsys, tmp_path):
    """9 faces of 3 models must not be printed as "(9 models)".

    Reproduces `results/resnet_cadsynth_cmca/seed0/eval.log:9`:
        Evaluating on test split (34338 models) -- ...
    for a 1,212-model test partition.
    """
    cfg = _resnet_cfg(tmp_path)
    _, out = _run_semantic(monkeypatch, capsys, cfg, is_resnet=True)
    line = _banner(out)

    assert "9 models" not in line, (
        "the ResNet dataset is UVGridLabelDataset -- one sample per FACE -- so "
        "len(dataset) is a face count, not a model count. Banner: " + line)
    assert "9 faces" in line, "banner does not say faces: " + line
    assert "3 models" in line, (
        "the banner should still name the 3-model test partition: " + line)


def test_resnet_banner_invents_no_bias(monkeypatch, capsys, tmp_path):
    """A config with no `bias` block must not be reported as bias=similarity."""
    cfg = _resnet_cfg(tmp_path)
    assert "bias" not in cfg, "this fixture must have no bias block"
    _, out = _run_semantic(monkeypatch, capsys, cfg, is_resnet=True)
    line = _banner(out)

    assert "bias=similarity" not in line, (
        "the config carries no bias block and the ResNet tower has no "
        "attention at all, so 'bias=similarity' is invented by the "
        "cfg.get('bias', {}).get('type', 'similarity') fallback. Banner: "
        + line)


def test_resnet_banner_invents_no_attributes(monkeypatch, capsys, tmp_path):
    """A config with no `dataset.attributes` must not be reported as scaled."""
    cfg = _resnet_cfg(tmp_path)
    assert "attributes" not in cfg["dataset"]
    _, out = _run_semantic(monkeypatch, capsys, cfg, is_resnet=True)
    line = _banner(out)

    assert "attributes=scaled," not in line and not line.endswith(
        "attributes=scaled"), (
        "dataset.attributes is absent from every ResNet config and is read by "
        "nothing on the UV-grid path, so 'attributes=scaled' is invented. "
        "Banner: " + line)


def test_resnet_banner_marks_stated_keys_as_ignored(monkeypatch, capsys,
                                                    tmp_path):
    """If a ResNet config DID state them, the banner must say they are unused."""
    cfg = _resnet_cfg(tmp_path, attributes="standardized", bias="adjacency")
    _, out = _run_semantic(monkeypatch, capsys, cfg, is_resnet=True)
    line = _banner(out)

    assert "IGNORED" in line, (
        "UVGridLabelDataset takes neither an attributes argument nor a bias "
        "argument, so a ResNet config that sets them is setting dead keys and "
        "the banner must say so. Banner: " + line)
    assert "standardized" in line and "adjacency" in line, (
        "the banner should still echo what the config states: " + line)


def test_graph_banner_reports_the_stated_config_values(monkeypatch, capsys,
                                                       tmp_path):
    cfg = _graph_cfg(tmp_path, attributes="standardized", bias="both")
    _, out = _run_semantic(monkeypatch, capsys, cfg, is_resnet=False)
    line = _banner(out)

    assert "attributes=standardized" in line, line
    assert "bias=both" in line, line
    assert "3 models" in line, (
        "the graph dataset is one sample per MODEL: " + line)
    assert "IGNORED" not in line and "DEFAULT" not in line, (
        "both keys are stated in this config, so nothing is defaulted or "
        "ignored: " + line)


def test_graph_banner_flags_defaults_as_defaults(monkeypatch, capsys, tmp_path):
    """On the graph path the defaults ARE applied -- but must be labelled."""
    cfg = _graph_cfg(tmp_path, attributes=None, bias=None)
    _, out = _run_semantic(monkeypatch, capsys, cfg, is_resnet=False)
    line = _banner(out)

    assert "DEFAULT" in line, (
        "PreExtractedBRepDataset really is built with attributes='scaled' and "
        "bias_type='similarity' here, but the config states neither, and the "
        "banner must not present a fallback as a stated setting. Banner: "
        + line)


def test_unified_banner_flags_defaults_as_defaults(monkeypatch, capsys,
                                                   tmp_path):
    """evaluate_unified carries the same fallbacks; today's configs hide them."""
    cfg = _unified_cfg(tmp_path, attributes=None, bias=None)
    _, out = _run_unified(monkeypatch, capsys, cfg)

    assert "DEFAULT" in out, (
        "evaluate_unified.py prints attributes/bias through the same "
        "`.get(..., default)` fallbacks as evaluate_semantic.py. They are "
        "never reached today only because mfinstseg_unified.yaml carries both "
        "keys; a future config without a bias block reproduces the ResNet "
        "defect exactly. Output:\n" + out)


def test_unified_banner_reports_stated_values(monkeypatch, capsys, tmp_path):
    cfg = _unified_cfg(tmp_path, attributes="standardized", bias="both")
    _, out = _run_unified(monkeypatch, capsys, cfg)
    assert "attributes=standardized" in out or "attributes : standardized" in out
    assert "bias=both" in out or "bias: both" in out
    assert "DEFAULT" not in out


def test_the_two_evaluators_describe_a_config_identically():
    """Anti-drift: the helper is duplicated, so pin the two copies together."""
    for attrs, bias in (("standardized", "both"), (None, None),
                        ("scaled", None), (None, "adjacency")):
        cfg = {"dataset": {"n_classes": 4}}
        if attrs is not None:
            cfg["dataset"]["attributes"] = attrs
        if bias is not None:
            cfg["bias"] = {"type": bias}
        assert (es._input_settings_phrase(cfg, is_resnet=False)
                == eu._input_settings_phrase(cfg)), (
            "evaluate_semantic and evaluate_unified carry separate copies of "
            "this helper (no cross-import, deliberately). They have drifted.")


def test_unified_metrics_record_whether_settings_were_stated(monkeypatch,
                                                             capsys, tmp_path):
    """The JSON must distinguish a stated setting from a defaulted one."""
    cfg = _unified_cfg(tmp_path, attributes="standardized", bias="both")
    stated, _ = _run_unified(monkeypatch, capsys, cfg)
    assert stated["attributes"] == "standardized"
    assert stated["bias_type"] == "both"
    assert stated["attributes_source"] == "config"
    assert stated["bias_type_source"] == "config"

    cfg2 = _unified_cfg(tmp_path, attributes=None, bias=None)
    defaulted, _ = _run_unified(monkeypatch, capsys, cfg2)
    # The VALUES are unchanged -- the dataset really is built with these.
    assert defaulted["attributes"] == "scaled"
    assert defaulted["bias_type"] == "similarity"
    assert defaulted["attributes_source"] != "config", (
        "dataset.attributes is absent from this config; the metrics file must "
        "not imply it was stated.")
    assert defaulted["bias_type_source"] != "config"


# ===================================================================== #
#  DEFECT 2 -- per_class[c]["accuracy"] is RECALL
# ===================================================================== #

@pytest.mark.parametrize("mod", [es, eu], ids=["semantic", "unified"])
def test_per_class_accuracy_is_bit_identical_to_recall(mod):
    """Characterisation lock. Passes before AND after the fix, by design.

    It is here so that a later change which makes `accuracy` mean something
    else (TP+TN)/N without renaming it is caught.
    """
    preds, gts = _preds_and_gts()
    per_cls = mod._per_class_metrics(preds, gts, N_CLASSES)
    assert per_cls, "fixture must support every class"
    for cls, m in per_cls.items():
        assert m["accuracy"] == m["recall"], (
            f"class {cls}: accuracy {m['accuracy']!r} != recall {m['recall']!r}")


def test_semantic_metrics_document_that_accuracy_is_recall(monkeypatch, capsys,
                                                           tmp_path):
    cfg = _graph_cfg(tmp_path)
    metrics, _ = _run_semantic(monkeypatch, capsys, cfg, is_resnet=False)

    doc = metrics.get("per_class_key_semantics")
    assert isinstance(doc, dict), (
        "eval_metrics.json carries a per_class block whose 'accuracy' key is "
        "recall, with nothing in the file saying so. A paper table built from "
        "it would print recall in an accuracy column.")
    assert "recall" in doc.get("accuracy", "").lower(), (
        "the per-class key documentation must state that 'accuracy' is "
        f"recall. Got: {doc.get('accuracy')!r}")
    assert "recall" in doc


def test_unified_metrics_document_that_accuracy_is_recall(monkeypatch, capsys,
                                                          tmp_path):
    cfg = _unified_cfg(tmp_path)
    metrics, _ = _run_unified(monkeypatch, capsys, cfg)

    doc = metrics.get("per_class_key_semantics")
    assert isinstance(doc, dict), (
        "unified_eval_metrics.json has the same undocumented per_class block.")
    assert "recall" in doc.get("accuracy", "").lower()


def test_per_class_accuracy_key_is_kept_for_the_archived_files(monkeypatch,
                                                               capsys,
                                                               tmp_path):
    """Decision lock: the misnamed key STAYS, with its value untouched.

    85 archived metrics files carry `per_class[c]["accuracy"]`. Removing it
    from new output would make new files structurally different from archived
    ones for no gain, since `recall` beside it already carries the same
    number. Passes before and after: it pins the decision, not the fix.
    """
    preds, gts = _preds_and_gts()
    cfg = _graph_cfg(tmp_path)
    metrics, _ = _run_semantic(monkeypatch, capsys, cfg, is_resnet=False)
    expected = es._per_class_metrics(preds, gts, N_CLASSES)
    for cls, m in expected.items():
        got = metrics["per_class"][str(cls)]
        assert got["accuracy"] == m["accuracy"]
        assert got["recall"] == m["recall"]


def test_semantic_printed_table_does_not_head_recall_as_accuracy(
        monkeypatch, capsys, tmp_path):
    cfg = _graph_cfg(tmp_path)
    _, out = _run_semantic(monkeypatch, capsys, cfg, is_resnet=False)
    header = [ln for ln in out.splitlines()
              if "Precision" in ln and "Support" in ln]
    assert header, "no per-class table header in:\n" + out
    assert "Accuracy" not in header[0], (
        "the table prints the same number twice, under 'Accuracy' and under "
        "'Recall' -- see results/resnet_cadsynth_cmca/seed0/eval.log, class 0: "
        "94.98% and 0.9498. The header must not call recall accuracy. Header: "
        + header[0])


# ===================================================================== #
#  DEFECT 3 -- instance.n_pairs counts ORDERED pairs
# ===================================================================== #

def test_instance_pair_count_is_named_ordered(monkeypatch, capsys, tmp_path):
    cfg = _unified_cfg(tmp_path)
    metrics, _ = _run_unified(monkeypatch, capsys, cfg)
    inst = metrics["instance"]

    assert inst["n_pairs"] == 12, "fixture changed"
    assert inst.get("n_pairs_ordered") == 12, (
        "`n_pairs` is an ORDERED count -- `_run_test` masks only the diagonal, "
        "so (i,j) and (j,i) are both collected. Nothing in the file says so, "
        "and a caption can quote 930,718 as a number of pairs.")
    assert inst.get("n_pairs_unordered") == 6, (
        "the unordered count is exactly half and is the number a caption "
        "wants; it must be in the file rather than left to be divided by hand.")
    assert "ordered" in inst.get("pair_counting", "").lower()


def test_instance_banner_says_ordered(monkeypatch, capsys, tmp_path):
    cfg = _unified_cfg(tmp_path)
    _, out = _run_unified(monkeypatch, capsys, cfg)
    line = [ln for ln in out.splitlines() if ln.startswith("INSTANCE")]
    assert line, "no INSTANCE line in:\n" + out
    assert "ordered" in line[0].lower(), (
        "'12 face pairs' reads as 12 unordered pairs; it is 6. Line: "
        + line[0])


def test_odd_pair_count_raises_rather_than_halving_it(monkeypatch, capsys,
                                                      tmp_path):
    """An odd ordered-pair count is impossible; if it happens, say so loudly."""
    cfg = _unified_cfg(tmp_path)
    bad = _unified_out()
    bad["inst_sims"] = bad["inst_sims"][:11]
    bad["inst_targets"] = bad["inst_targets"][:11]

    monkeypatch.setattr(eu, "load_checkpoint",
                        lambda p, device=None: {"n_classes": N_CLASSES,
                                                "epoch": 38,
                                                "optimal_threshold": 0.5})
    monkeypatch.setattr(eu, "_build_models",
                        lambda cfg, ckpt, device: (object(), object(),
                                                   object(), N_CLASSES))
    monkeypatch.setattr(eu, "_build_test_set",
                        lambda cfg, seed: (_FakeModelDataset(3), "fake split"))
    monkeypatch.setattr(eu, "_run_test", lambda *a, **k: bad)
    monkeypatch.setattr(eu, "DataLoader", lambda *a, **k: "LOADER")
    monkeypatch.setattr(eu, "plot_confusion_matrix", lambda *a, **k: None)
    monkeypatch.setattr(eu, "plot_roc_curve", lambda *a, **k: None)
    monkeypatch.setattr(eu, "provenance_from_checkpoint", lambda ckpt, cfg: {})

    with pytest.raises(RuntimeError, match="(?i)odd"):
        eu.evaluate_unified_model(cfg, "ckpt.pth")


# ===================================================================== #
#  DEFECT 4 -- experiment_name is stale for resnet_cadsynth_cmca
# ===================================================================== #

def test_experiment_name_is_echoed_from_the_config_verbatim(monkeypatch,
                                                            capsys, tmp_path):
    """The evaluator is NOT the cause of the stale name -- pin that.

    `results/resnet_cadsynth_cmca/seed0/eval_metrics.json` says
    `experiment_name: resnet_cadsynth` because
    `results/resnet_cadsynth_cmca/seed0/config.resolved.yaml:3` says so,
    because `runners/run_all.py:114` points the cell at
    `configs/resnet_cadsynth.yaml` whose own `experiment_name` is
    `resnet_cadsynth`. The evaluator copies the config verbatim, which is
    right; inventing a name from the output path would be worse.

    Passes before and after: it fixes the blame, not the file.
    """
    cfg = _resnet_cfg(tmp_path)
    cfg["experiment_name"] = "resnet_cadsynth"
    metrics, _ = _run_semantic(monkeypatch, capsys, cfg, is_resnet=True)
    assert metrics["experiment_name"] == "resnet_cadsynth"


def test_stale_experiment_name_is_a_config_defect_not_an_evaluator_one():
    """Grounds the claim above in the files on disk, so it cannot rot."""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "resnet_cadsynth.yaml").read_text())
    assert cfg["experiment_name"] == "resnet_cadsynth", (
        "if this changed, the arm/config mismatch may have been fixed in "
        "configs/ -- re-check runners/run_all.py's resnet_cadsynth_cmca cell.")
    run_all = (ROOT / "runners" / "run_all.py").read_text()
    assert 'Cell("resnet_cadsynth_cmca", "configs/resnet_cadsynth.yaml"' \
        in run_all, (
        "the cell named resnet_cadsynth_cmca is the one pointed at the config "
        "whose experiment_name is resnet_cadsynth. Neither file is owned by "
        "the evaluators fixer.")


# ===================================================================== #
#  Values must not move
# ===================================================================== #

def test_semantic_values_unchanged_by_this_fix(monkeypatch, capsys, tmp_path):
    """Every computed number, pinned to a hand-checked value.

    9 faces, classes [0,0,0,1,1,2,2,3,3], preds [0,0,1,1,2,2,3,3,0]:
    5 correct of 9 -> 55.56%.
    """
    cfg = _graph_cfg(tmp_path)
    metrics, _ = _run_semantic(monkeypatch, capsys, cfg, is_resnet=False)

    assert metrics["overall_accuracy"] == pytest.approx(5 / 9)
    assert metrics["n_samples"] == 9
    assert metrics["n_classes"] == 4
    # class 0: gt 3, tp 2, fp 1 (the class-3 face predicted 0) -> P 2/3, R 2/3
    c0 = metrics["per_class"]["0"]
    assert c0["support"] == 3
    assert c0["precision"] == pytest.approx(2 / 3)
    assert c0["recall"] == pytest.approx(2 / 3)
    assert c0["accuracy"] == pytest.approx(2 / 3)
    assert metrics["mean_iou"] <= metrics["macro_f1"] + 1e-9

    written = json.loads((tmp_path / "eval_metrics.json").read_text())
    assert written["overall_accuracy"] == metrics["overall_accuracy"]
    assert written["per_class"]["0"]["accuracy"] == c0["accuracy"]


def test_unified_values_unchanged_by_this_fix(monkeypatch, capsys, tmp_path):
    cfg = _unified_cfg(tmp_path)
    metrics, _ = _run_unified(monkeypatch, capsys, cfg)

    assert metrics["semantic"]["overall_accuracy"] == pytest.approx(5 / 9)
    assert metrics["semantic"]["n_faces"] == 9
    assert metrics["bottom"]["n_faces"] == 9
    assert metrics["bottom"]["n_positive"] == 3
    assert metrics["instance"]["n_pairs"] == 12
    assert metrics["instance"]["n_positive"] == 4
    assert metrics["instance"]["auc"] == pytest.approx(1.0)
    assert metrics["instance"]["reportable"] == "checkpoint_threshold"
    assert metrics["instance"]["checkpoint_threshold"]["threshold"] == 0.5

    written = json.loads((tmp_path / "unified_eval_metrics.json").read_text())
    assert written["instance"]["auc"] == metrics["instance"]["auc"]
    assert written["semantic"]["macro_f1"] == metrics["semantic"]["macro_f1"]
