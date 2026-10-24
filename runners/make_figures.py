#!/usr/bin/env python3
"""
make_figures.py -- regenerate the paper's result figures from committed metrics.

Reads the SEED-SWEPT layout `results/<cell>/seed<k>/eval_metrics.json` -- the one
`runners/run_all.py` actually writes -- and writes vector PDFs into
`../latex/figs/`. No training, no checkpoints: the figures are a function of the
committed metrics, so they can always be rebuilt and can never drift from the
numbers in the tables.

Usage:
    python runners/make_figures.py                  # both corpora, all figures
    python runners/make_figures.py --dataset cadsynth
    python runners/make_figures.py --seeds 0 1 2
    python runners/make_figures.py --out-dir /somewhere/else

Figures
-------
fig_bias_perclass.pdf          two panels, per-class F1 for the four bias arms
fig_bias_perclass_<ds>.pdf     one corpus, column-width version
fig_confusion_<ds>.pdf         confusion matrix for the `both` arm

Differing epochs are the designed outcome of early stopping. The check that
matters is in `runners/check_convergence.py`: no selected checkpoint sitting
at the cap.

Design notes (these are deliberate, not defaults)
-------------------------------------------------
* Four categorical hues, all-pairs in light mode: worst CVD dE 9.2 (deutan,
  aqua vs orange), worst normal-vision dE 16.3 (violet vs blue), both passing.
  The violet slot is the reference palette's categorical slot 7, chosen because
  the nearer slots (amber dE 13.7, pink dE 12.9 against orange) fail the
  normal-vision floor of 15. The aqua slot sits at 2.74:1 against the surface,
  below the 3:1 bar, so every arm also carries a **distinct marker shape**. That
  is the required relief, and it means the figure still reads when the paper is
  printed in greyscale.
* One x-axis. F1 only. Accuracy is not co-plotted -- CADSynth class 0 is 56.1%
  of all faces, so the two measures disagree and a second axis would invite the
  reader to compare them directly.
* Rows are ordered by the *similarity* arm's F1 -- strongest at the top,
  weakest at the bottom. Ordering by the baseline rather than by the winner
  means the figure is not arranged to flatter the arm being argued for, and the
  connector length shows at a glance where real topology actually helps.
* The connector runs from `similarity` to the best of the two topology arms,
  NOT to the best of all four. `shuffled` is a control that is expected to sit
  at or below similarity; letting it define the end of the connector would draw
  a "gain" bar for a degradation.
* The confusion matrix uses a single-hue sequential ramp. Never a rainbow: a
  rainbow implies category, and this is magnitude.
"""

from __future__ import annotations

import argparse
import json  # noqa: F401
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The results layout and the per-class scoring are imported,
# not re-implemented. `aggregate_seeds` owns "where does an arm's results live";
# `compare_arms` owns "per-class F1 from a summed confusion matrix". A private
# copy of either here is how the figures and the tables drift apart.
from runners.aggregate_seeds import arm_dir, load_runs          # noqa: E402
from runners.compare_arms import per_class_f1_from_cm           # noqa: E402

# --- palette (light mode; see dataviz reference palette) -------------------- #
ARMS = ("similarity", "adjacency", "both", "shuffled")
# The two arms the connector may run to: the control is not a competitor.
TOPOLOGY_ARMS = ("adjacency", "both")
SERIES = {                        # slots 1, 2, 3, 7 of the reference palette
    "similarity": "#2a78d6",      # blue
    "adjacency":  "#eb6834",      # orange
    "both":       "#1baf7a",      # aqua -- 2.74:1, hence the marker relief
    "shuffled":   "#4a3aa7",      # violet -- control; see the design note
}
MARKER = {"similarity": "o", "adjacency": "s", "both": "^", "shuffled": "D"}
LABEL = {"similarity": "similarity (prior work)",
         "adjacency":  "adjacency (real topology)",
         "both":       "both",
         "shuffled":   "shuffled (control)"}

INK_PRIMARY = "#0b0b0b"
INK_SECOND = "#52514e"
INK_MUTED = "#8a8a85"
GRID = "#e6e5e1"

# single-hue sequential ramp, light -> dark (blue)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"],
)


def rcparams():
    """Match the paper: serif text, small type, embedded TrueType."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6,
        "legend.fontsize": 6.5,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.5,
        "xtick.color": INK_SECOND,
        "ytick.color": INK_SECOND,
        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_PRIMARY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,       # TrueType, not Type-3: required by most venues
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def summed_cm(results: Path, dataset: str, arm: str, seeds):
    """Confusion matrix summed over the seeds of one arm, or None.

    Raises on mismatched shapes rather than aggregating arms that scored
    different class sets -- silently summing those would produce a figure
    nobody could tell was wrong.
    """
    runs, missing = load_runs(results, dataset, arm, seeds)
    if not runs:
        print(f"  missing: no runs under {arm_dir(results, dataset, arm)}/seed*/",
              file=sys.stderr)
        return None, missing
    cms = [np.asarray(m["confusion_matrix"], dtype=np.int64)
           for m in runs.values() if "confusion_matrix" in m]
    if not cms:
        print(f"  missing: no confusion_matrix recorded for arm {arm}",
              file=sys.stderr)
        return None, missing
    shapes = {c.shape for c in cms}
    if len(shapes) != 1:
        raise SystemExit(
            f"arm {arm}: confusion matrices differ in shape across seeds: "
            f"{shapes}. The seeds are not scoring the same class set; do not "
            f"aggregate them into one figure.")
    if missing:
        print(f"  NOTE: arm {arm} is missing seeds {missing}; the figure is "
              f"averaged over {sorted(runs)} only.", file=sys.stderr)
    return np.sum(cms, axis=0), missing


def load_arms(results: Path, dataset: str, seeds):
    """Return {arm: {class_index: f1}} plus the shared class list.

    Per-class F1 is computed from the confusion matrix summed across seeds.
    """
    per_arm = {}
    for arm in ARMS:
        cm, _ = summed_cm(results, dataset, arm, seeds)
        if cm is None:
            return None, None
        per_arm[arm] = per_class_f1_from_cm(cm)
    common = sorted(set.intersection(*(set(d) for d in per_arm.values())))
    return per_arm, common


def panel_perclass(ax, per_arm, classes, title, show_legend, show_ylabel):
    """Horizontal dot plot: one row per class, one marker per arm."""
    order = sorted(classes, key=lambda c: per_arm["similarity"][c])
    y = np.arange(len(order))

    # connector from baseline to best, so the gap is a length not a comparison.
    # The control is expected to sit at or below similarity, so including it
    # would occasionally draw a "gain" connector for a degradation.
    for i, c in enumerate(order):
        lo = per_arm["similarity"][c]
        hi = max(per_arm[a][c] for a in TOPOLOGY_ARMS)
        ax.plot([lo, hi], [i, i], color=GRID, lw=1.4, zorder=1,
                solid_capstyle="round")

    for arm in ARMS:
        ax.scatter([per_arm[arm][c] for c in order], y,
                   s=13, marker=MARKER[arm], color=SERIES[arm],
                   edgecolors="white", linewidths=0.5,
                   label=LABEL[arm] if show_legend else None, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([f"c{c}" for c in order])
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("per-class F1")
    if show_ylabel:
        ax.set_ylabel("class (ordered by similarity-arm F1)")
    ax.set_title(title, loc="left", color=INK_PRIMARY, pad=6)

    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def fig_perclass(results: Path, out: Path, datasets, seeds):
    """Two-panel full-width figure, plus a column-width version per corpus."""
    loaded = {}
    for ds in datasets:
        per_arm, common = load_arms(results, ds, seeds)
        if per_arm:
            loaded[ds] = (per_arm, common)
    if not loaded:
        return []

    written = []
    # --- combined, full text width ---
    if len(loaded) > 1:
        fig, axes = plt.subplots(1, len(loaded), figsize=(7.0, 4.6), sharex=True)
        for i, (ds, (per_arm, common)) in enumerate(loaded.items()):
            panel_perclass(axes[i], per_arm, common,
                           f"({chr(97+i)}) {ds}", show_legend=(i == 0),
                           show_ylabel=(i == 0))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.075, 1, 1))
        p = out / "fig_bias_perclass.pdf"
        fig.savefig(p)
        plt.close(fig)
        written.append(p)

    # --- one corpus, column width ---
    for ds, (per_arm, common) in loaded.items():
        fig, ax = plt.subplots(figsize=(3.4, 4.4))
        panel_perclass(ax, per_arm, common, ds, show_legend=True, show_ylabel=True)
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        p = out / f"fig_bias_perclass_{ds}.pdf"
        fig.savefig(p)
        plt.close(fig)
        written.append(p)
    return written


def fig_confusion(results: Path, out: Path, dataset: str, seeds,
                  arm: str = "both"):
    summed, _ = summed_cm(results, dataset, arm, seeds)
    if summed is None:
        return None
    cm = np.asarray(summed, dtype=float)
    # row-normalise: each row is "of the faces truly in class i, where did they go"
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = cm / cm.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)

    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    im = ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"{dataset} — bias: {arm} (summed over seeds)",
                 loc="left", pad=5)
    step = 2 if n > 15 else 1
    ax.set_xticks(range(0, n, step))
    ax.set_yticks(range(0, n, step))
    ax.tick_params(length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("fraction of true class", fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=6)
    fig.tight_layout()
    q = out / f"fig_confusion_{dataset}.pdf"
    fig.savefig(q)
    plt.close(fig)
    return q


def fig_instance_threshold(results: Path, out: Path, cell: str, seeds):
    """Threshold sweep for the INSTANCE head, with the selected point marked.

    Why this figure exists, and why there is no companion
    for the bottom-face head.

    The instance head scores a PAIR of faces by cosine similarity, so it has no
    argmax to fall back on: turning a similarity into "same instance or not"
    needs a decision threshold, and every precision, recall and F1 it reports is
    a function of that one number. A reader cannot tell from a single F1 whether
    the threshold was chosen honestly or fished for. This plots the whole sweep
    and marks the value actually used, which was fitted on the VALIDATION split
    under the selection rule, so the reader can see both that the choice is near the
    optimum and how sharply the metric depends on it.

    The bottom-face head needs no such figure. It classifies each face by
    nearest prototype against a two-class encoder, which is an argmax; there is
    no threshold to sweep, and inventing one to tune would be manufacturing a
    degree of freedom the model does not have.

    ROC-AUC stays the primary instance number because it is threshold-free, as
    the regiment requires. This figure is the defence of the secondary one.

    Reads each seed's committed metrics and needs the raw pair scores under
    `instance.sweep`; returns None when they are absent, so it is a no-op until
    the batch produces T5 rather than a crash.
    """
    import json as _json

    curves, chosen = [], []
    for s in seeds:
        p = results / cell / f"seed{s}" / "unified_eval_metrics.json"
        if not p.is_file():
            p = results / cell / f"seed{s}" / "eval_metrics.json"
        if not p.is_file():
            continue
        inst = _json.loads(p.read_text(encoding="utf-8")).get("instance", {})
        sweep = inst.get("sweep")
        if not sweep:
            continue
        curves.append((np.asarray(sweep["thresholds"], dtype=float),
                       np.asarray(sweep["f1"], dtype=float)))
        row = inst.get("checkpoint_threshold") or {}
        if row.get("threshold") is not None:
            chosen.append((float(row["threshold"]), float(row.get("f1", np.nan))))

    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    grid = curves[0][0]
    stacked = np.vstack([np.interp(grid, t, f) for t, f in curves])
    mean, sd = stacked.mean(axis=0), stacked.std(axis=0)

    ax.fill_between(grid, mean - sd, mean + sd, alpha=0.18, linewidth=0,
                    color=SERIES["similarity"])
    ax.plot(grid, mean, color=SERIES["similarity"], linewidth=1.4,
            label="F1 (mean +/- sd)")
    if chosen:
        xs = [c[0] for c in chosen]
        ax.axvline(float(np.mean(xs)), color=SERIES["adjacency"], linewidth=1.1,
                   linestyle="--", label="selected on validation")
        ax.scatter(xs, [c[1] for c in chosen], s=14, color=SERIES["adjacency"], zorder=3)

    ax.set_xlabel("similarity threshold")
    ax.set_ylabel("pairwise F1")
    ax.set_title(f"{cell} -- instance decision threshold",
                 loc="left", color=INK_PRIMARY, pad=6)
    ax.set_ylim(0, 1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center")
    fig.tight_layout()
    q = out / f"fig_instance_threshold_{cell}.pdf"
    fig.savefig(q)
    plt.close(fig)
    return q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out-dir", default="../latex/figs")
    ap.add_argument("--dataset", choices=["cadsynth", "mfinstseg"], default=None,
                    help="default: both")
    args = ap.parse_args(argv)

    rcparams()
    results = REPO / args.results
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets = [args.dataset] if args.dataset else ["cadsynth", "mfinstseg"]

    written = fig_perclass(results, out, datasets, args.seeds)
    for ds in datasets:
        q = fig_confusion(results, out, ds, args.seeds)
        if q:
            written.append(q)

    if not written:
        print("nothing written -- no eval_metrics.json found under "
              f"{results}/<cell>/seed*/ for seeds {args.seeds}", file=sys.stderr)
        return 1
    for w in written:
        print(f"  {w.stat().st_size:>8,} B  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
