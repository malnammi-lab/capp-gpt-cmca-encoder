#!/usr/bin/env python3
"""
probe_embedding_space.py -- read the EMBEDDING SPACE, not the argmax.

WHAT THIS IS FOR
================
Every T3 segmentation number is a function of two 1-D integer arrays of class
ids, and both the CMCA arm (`loss.type: prototype_infonce`) and the
cross-entropy arm reach those ids through a 25-way argmax
(`src/evaluation/evaluate_semantic.py:232` for CE, `:235` for CMCA). Whatever
the two objectives do differently to the 128-d face embedding is destroyed
before any metric sees it.

But the two objectives do not train the same object:

  * `prototype_infonce` (`src/losses/infonce.py:41-45`) L2-normalises the face
    embeddings AND the 25 label-tower prototypes and scores them by cosine
    similarity. The trained object is a SHARED face-label embedding space.
  * `cross_entropy` (`src/models/ce_encoder.py:85`) projects the same
    embeddings onto `nn.Linear(d_model, n_classes)` logits. The label tower is
    constructed but `src/training/train_semantic.py:308-314` excludes it from
    the optimizer and from the checkpoint, so the arm ends with no trained
    label tower at all.

The project adopted CMCA for that shared space. This runner produces numbers
about it that segmentation accuracy cannot.

`SemanticClassificationModel.encode` (`src/models/ce_encoder.py:102-113`)
exists so the two arms can be compared in embedding space and not only on
accuracy. This runner is the caller that produces that number.

WHAT THIS CAN AND CANNOT CONCLUDE
=================================
**This probe can show the two spaces DIFFER, and by how much on four stated
measures. It cannot show the difference is worth having.** A higher silhouette
or a higher retrieval score is evidence that class structure survives in the
embedding geometry; whether that structure buys anything -- zero-shot labels, a
decoder's input, a retrieval index -- is a separate argument this file does not
make and its output must not be quoted for. L7 in
`paper-content/limitations-register.md` is where that scope limit lives.

Nor is a difference here a restatement of T3's accuracy comparison. T3 compared
two face classifiers. These are different quantities, computed on the same
checkpoints.

WHAT IT MEASURES
================
Four numbers, all computed on the held-out TEST faces only, all functions of
the embedding geometry and none of them a function of any argmax:

  intra/inter cosine-distance ratio   how tight a class is relative to how far
                                      apart classes are. 1.0 means the labels
                                      explain none of the geometry. EXACT over
                                      every test face -- no subsampling.
  silhouette (cosine)                 per-face (b - a) / max(a, b) averaged.
                                      +1 separated, negative means faces sit
                                      closer to another class.
  k-NN accuracy at k = 1 and 5        leave-one-out retrieval: does the nearest
                                      other test face carry the same label?
  mAP                                 mean average precision of the full
                                      same-class ranking per query.

Each is reported beside its CHANCE value, because "0.73" means nothing until
you know whether chance is 0.04 or 0.70. See `MEASURES` further down for the
exact definition that is written into every output file.

...and TWO CONTROLS, because the two arms are not symmetric in what they are
allowed to do with the embedding, and a cosine-only comparison would quietly
favour one of them:

  null_silhouette_label_shuffled      the silhouette recomputed on the SAME
                                      embeddings with the labels permuted.
                                      Chance for the silhouette is NOT 0.0 at
                                      a finite face count and width, so it is
                                      measured, not asserted. See `silhouette`.
  silhouette_euclidean_control        the same coefficient on the UNNORMALISED
    and embedding_norm.               rows, plus the share of the variance of
    class_eta_squared                 ||z|| the class explains. CMCA is
                                      L2-normalised by its own loss and can
                                      only use direction; the CE head reads
                                      w_c.z + b_c and can use length too.
                                      Every cosine measure here throws length
                                      away, so these two say how much was
                                      thrown away, per arm. See
                                      `embedding_norms`.

WIDTH. Compare within a row of the T3 2x2, not across it: both graph cells are
d_model=128 and both ResNet cells embedding_dim=512, so CMCA-vs-CE is width-
matched either way, while graph-vs-ResNet is not. The ratio and the retrieval
scores are width-free at chance. See `MEASURES["WIDTH"]`.

WHAT IT DOES NOT DO
===================
**It does not read ``||z||`` except in the two controls named above.** Four of
the five headline numbers are cosine quantities and cosine is blind to length
by construction. That is the CMCA arm's own scale, which is the reason for it
and also the reason the controls exist.

**Nothing is fitted on test data.** No threshold is tuned, no classifier is
trained, no normalisation statistic is estimated here; the standardisation
constants come from the training split via the config, exactly as in training.
The k-NN is leave-one-out WITHIN the test split, which is the standard
retrieval protocol and estimates no parameters -- the query is excluded from
its own ranking (`_rank_neighbours`).

BOTH ARMS, ONE SCORER
=====================
The four T3 cells reach their face embeddings by three lines of code, and after
that every number is computed by the same functions on the same arrays. That is
the discipline T3's own 2x2 used: if the CE arm were scored by different code
from the CMCA arm, a gap between them would be unattributable. See
`graph_face_embeddings` and `resnet_face_embeddings`, and the test
`test_ce_and_cmca_share_the_encoder_path`.

Models are built by `src/evaluation/evaluate_semantic._build_graph_transformer`
and `._build_resnet` -- IMPORTED, not reimplemented. Those are the builders
that produced every committed T3 number; a second copy here could drift from
them, and then this probe and `eval_metrics.json` would be describing two
different models.
"""

# `--config` must be that run's own config.resolved.yaml.
# A different config scores the checkpoint on inputs it was not trained on.

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.brep_dataset import load_split  # noqa: E402
from src.data.collate import collate_fn_brep  # noqa: E402
from src.data.uv_dataset import UVGridLabelDataset  # noqa: E402
from src.evaluation.evaluate_semantic import (  # noqa: E402
    _build_graph_transformer, _build_resnet)
from src.evaluation.inference import _build_brep_dataset  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.provenance import (code_fingerprint,  # noqa: E402
                                  provenance_from_checkpoint)
from src.utils.seeding import set_global_seed  # noqa: E402

#: Model types this probe knows how to reach the face embedding of. Anything
#: else is refused rather than guessed at: a wrong guess here produces a
#: plausible number from the wrong tensor, which is the failure mode the whole
#: gate suite exists to stop.
GRAPH_MODEL_TYPES = ("semantic_segmentation", "brep_encoder")
RESNET_MODEL_TYPES = ("resnet_encoder",)

#: Padded and unmapped face labels. `collate_cmca` pads with -1 on purpose
#: (src/data/collate.py:26-28) and `CMCADataset` maps a face outside
#: `selected_classes` to -1. Neither is a class, so neither may enter a
#: separation or a retrieval score.
NON_LABEL = -1


#  The measures -- pure functions of (embeddings, labels)
#
# Everything in this section is numpy only. It never sees a model, a config or
# a checkpoint, which is what lets the whole scorer be tested on synthetic
# tensors with a KNOWN answer -- embeddings built with real class structure
# must score high, embeddings built without it must score at chance. That test
# is `tests/test_probe_embedding_space.py` and it needs no `.pth` file.


def unit_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalise each row, in float64.

    Every measure here is a cosine quantity, and `PrototypeInfoNCELoss`
    normalises before it compares too (`src/losses/infonce.py:42-43`), so the
    probe reads the space on the same scale the objective trained it on.

    Raises:
        ValueError: If any row has (near-)zero norm. Normalising it would
            produce nan or an arbitrary direction, and a nan propagates into
            every mean below as a silently missing face.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(
            f"expected a 2-D (n_faces, dim) array, got shape {x.shape}")
    norms = np.linalg.norm(x, axis=1)
    bad = int((norms < 1e-12).sum())
    if bad:
        raise ValueError(
            f"{bad} of {len(norms)} embeddings have zero length, so they have "
            f"no direction and every cosine measure below is undefined for "
            f"them. That is a broken forward pass or an all-padding batch, "
            f"not a result -- it is not rounded away here.")
    return x / norms[:, None]


def class_separation(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Mean intra- and inter-class cosine distance, and their ratio.

    EXACT over every face passed in -- no subsampling, no approximation. The
    naive form is O(N^2) pairs, which is 1.2e9 pairs on a 34,338-face test
    split; this uses the identity that for unit rows the sum of cosine
    similarities over the ordered pairs of a set S is ``||sum(S)||^2 - |S|``,
    which makes it O(N*D). Accumulated in float64.

    Definitions, written into every output file as `MEASURES`:

        mean_intra_cosine_distance
            mean of (1 - cos) over all ordered pairs of distinct faces that
            SHARE a label. Small = classes are tight.
        mean_inter_cosine_distance
            the same over all ordered pairs carrying DIFFERENT labels. Large =
            classes are far apart.
        intra_inter_ratio
            intra / inter. **Lower is better separated.** 1.0 is the value a
            space with no class structure gives, because a face's own class is
            then no closer than any other. There is no upper bound, but above
            1.0 means same-label faces are FARTHER apart than different-label
            ones.

    Args:
        embeddings: ``(n_faces, dim)``.
        labels: ``(n_faces,)`` integer class ids, all >= 0.

    Returns:
        A dict of the three numbers plus `per_class`, each class's own mean
        intra-class cosine distance and its face count.

    Raises:
        ValueError: If fewer than two classes are present (there are then no
            inter-class pairs), if no class has two members (no intra-class
            pairs), or if the inter-class distance is zero (a degenerate space
            where every embedding points the same way -- the ratio would be
            0/0, and returning 1.0 there would read as "no structure" when the
            truth is "no information at all").
    """
    emb = unit_rows(embeddings)
    labels = np.asarray(labels)
    if len(labels) != len(emb):
        raise ValueError(f"{len(emb)} embeddings but {len(labels)} labels")

    n = len(emb)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError(
            f"class separation needs at least two classes to separate; these "
            f"{n} faces are all class {classes.tolist()}. Nothing is measured "
            f"here, so nothing is reported.")
    if not (counts >= 2).any():
        raise ValueError(
            f"no class has two members among these {n} faces, so there is not "
            f"one intra-class pair to measure. Class counts: "
            f"{dict(zip(classes.tolist(), counts.tolist()))}")

    total = emb.sum(axis=0)
    sim_all = float(total @ total) - n          # sum of cos over ordered i != j
    pairs_all = n * (n - 1)

    sim_within = 0.0
    pairs_within = 0
    per_class = {}
    for c, nc in zip(classes.tolist(), counts.tolist()):
        s = emb[labels == c].sum(axis=0)
        sim_c = float(s @ s) - nc
        pairs_c = nc * (nc - 1)
        sim_within += sim_c
        pairs_within += pairs_c
        per_class[str(c)] = {
            "n_faces": int(nc),
            "mean_intra_cosine_distance": (
                1.0 - sim_c / pairs_c if pairs_c else None),
        }

    pairs_between = pairs_all - pairs_within
    if pairs_between <= 0:
        raise ValueError("no inter-class pairs; see the two-class check above")

    intra = 1.0 - sim_within / pairs_within
    inter = 1.0 - (sim_all - sim_within) / pairs_between
    if abs(inter) < 1e-12:
        raise ValueError(
            f"the mean inter-class cosine distance is {inter:.3e}: every face "
            f"embedding points in the same direction, so the space carries no "
            f"information at all and the intra/inter ratio is 0/0. This is a "
            f"collapsed encoder, not a separation score.")

    return {
        "n_faces": int(n),
        "n_classes_present": int(len(classes)),
        "mean_intra_cosine_distance": float(intra),
        "mean_inter_cosine_distance": float(inter),
        "intra_inter_ratio": float(intra / inter),
        "chance_intra_inter_ratio": 1.0,
        "per_class": per_class,
    }


def stratified_subsample(labels: np.ndarray, cap, rng) -> np.ndarray:
    """Seeded, class-stratified indices, at most ~`cap` of them.

    Silhouette needs an N x N distance matrix and retrieval needs a full
    ranking per query, so both are quadratic and both are capped. The
    subsample is stratified rather than uniform so a rare class cannot vanish
    from the measurement, and every class is allocated at least two members --
    a one-member cluster has no intra-class distance, so silhouette scores it
    0 by convention and it would drag the mean toward chance for a reason that
    has nothing to do with the model.

    The floor is an allocation, not a guarantee: a class with exactly ONE face
    in `labels` still contributes one, because `min(len(idx_c), ...)` cannot
    invent a second. That case is a singleton cluster scored 0 by sklearn, and
    it is left visible rather than dropped -- dropping a class silently is how
    a subsample stops describing the split it came from.

    Because of the floor the result can exceed `cap` by up to two per class
    (at most 2K faces over). The actual size is reported by every caller rather
    than assumed.

    Returns:
        Sorted indices into `labels`. Sorted so the result does not depend on
        the order `rng.choice` happened to emit.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if cap is not None and cap < 2:
        raise ValueError(f"cap must be at least 2, got {cap}")
    if cap is None or n <= cap:
        return np.arange(n)

    picked = []
    for c in np.unique(labels):
        idx_c = np.flatnonzero(labels == c)
        take = min(len(idx_c), max(2, int(round(cap * len(idx_c) / n))))
        picked.append(rng.choice(idx_c, size=take, replace=False))
    return np.sort(np.concatenate(picked))


def _cosine_distance_matrix(unit_emb: np.ndarray) -> np.ndarray:
    """``1 - cos`` for unit rows, clipped, with an exact zero diagonal.

    Built once so the label-shuffled null below costs a relabelling instead of
    a second O(N^2) pass: permuting the labels changes nothing about the
    distances. `silhouette_score(..., metric="precomputed")` is the cosine
    score on the same rows.
    """
    d = 1.0 - unit_emb @ unit_emb.T
    np.clip(d, 0.0, 2.0, out=d)
    np.fill_diagonal(d, 0.0)
    return d


def silhouette(embeddings: np.ndarray, labels: np.ndarray, *,
               cap, rng, n_permutations: int = 5) -> dict:
    """Mean silhouette coefficient under COSINE distance, against a MEASURED null.

    For each face: ``(b - a) / max(a, b)`` where `a` is its mean distance to
    the other faces of its own class and `b` its mean distance to the faces of
    the nearest other class. Averaged over faces.

        +1   every face sits inside a tight cluster far from all others
         0   the ASYMPTOTIC value of a space with no class structure
        <0   faces are on average closer to a different class than their own

    Computed by `sklearn.metrics.silhouette_score` on a seeded, stratified
    subsample (see `stratified_subsample`), because the full pairwise matrix
    over a 34,338-face split is 1.2e9 entries.

    THE NULL IS MEASURED, NOT ASSERTED. 0.0 is the limit, not the value of any
    finite split, and the finite-sample floor moves with both the face count
    and the embedding width. The graph cells are d_model=128 and the ResNet
    cells embedding_dim=512, so one constant cannot serve both rows of the T3
    2x2. `null_silhouette_label_shuffled` is the same score recomputed on the
    SAME embeddings with the labels permuted, which holds the width, the
    norms, the face count and the label histogram fixed and varies only the
    thing under test. Read the score against that, not against 0.0.

    THE EUCLIDEAN CONTROL. Every measure in this file is a cosine quantity, so
    every one of them discards ``||z||``. That is the right scale for the CMCA
    arm: `PrototypeInfoNCELoss` L2-normalises before it compares
    (`src/losses/infonce.py:42-43`), so that arm can only put class
    information in the DIRECTION. The CE arm is not so constrained -- its head
    scores ``w_c . z + b_c`` (`src/models/ce_encoder.py:85`), which reads the
    length too. A space whose classes are perfectly separated by length alone
    scores +0.92 Euclidean and -0.33 cosine, i.e. BELOW the null, so without a
    control the probe would report a real structure as no structure, in the
    arm that is allowed to build it. `silhouette_euclidean_control` is the
    same coefficient on the UNNORMALISED rows, and `embedding_norm` in the
    report says how much of the length's variance the class explains.

    Args:
        n_permutations: Label shuffles behind the measured null. Each is one
            relabelling of a distance matrix already in hand; five cost 0.62 s
            at cap 4,000.

    Returns:
        The score, the measured null, the Euclidean control, the number of
        faces it was measured on, and whether the split was subsampled.
    """
    labels = np.asarray(labels)
    if n_permutations < 1:
        raise ValueError(
            f"n_permutations={n_permutations}: the null is what the score is "
            f"read against, and this probe does not report a score without "
            f"one. Asserting 0.0 instead is the defect this replaced.")
    idx = stratified_subsample(labels, cap, rng)
    raw = np.asarray(embeddings, dtype=np.float64)[idx]
    emb = unit_rows(raw)
    lab = labels[idx]
    if len(np.unique(lab)) < 2:
        raise ValueError(
            f"silhouette is undefined with one class; the subsample of "
            f"{len(lab)} faces has only class {np.unique(lab).tolist()}.")

    d_cos = _cosine_distance_matrix(emb)
    score = float(silhouette_score(d_cos, lab, metric="precomputed"))
    null = [float(silhouette_score(d_cos, rng.permutation(lab),
                                   metric="precomputed"))
            for _ in range(n_permutations)]

    return {
        "silhouette_cosine": score,
        "null_silhouette_label_shuffled": {
            "mean": float(np.mean(null)),
            "std": float(np.std(null)),
            "n_permutations": int(n_permutations),
            "scores": [float(x) for x in null],
        },
        "chance_silhouette_asymptotic": 0.0,
        "silhouette_euclidean_control": float(
            silhouette_score(raw, lab, metric="euclidean")),
        "n_faces_scored": int(len(idx)),
        "subsampled": bool(len(idx) < len(labels)),
        "cap": int(cap) if cap is not None else None,
    }


def embedding_norms(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """``||z||`` over the test faces, and the share of its variance the class explains.

    THE ONE NUMBER THAT SAYS WHAT THE COSINE MEASURES THREW AWAY.
    `class_eta_squared` is the between-class share of the variance of the
    scalar ``||z||``: ``1 - SS_within / SS_total``.

        ~0   the length tells the same story for every class, so normalising
             it away costs nothing and the cosine scores are the whole
             picture.
        ~1   the class is written in the LENGTH. Cosine cannot see it, and a
             low `silhouette_cosine` for that arm is then a statement about
             the probe, not about the encoder: read
             `silhouette_euclidean_control` instead, and say so in the record.

    This is the asymmetry control for CMCA vs CE. The contrastive arm is
    normalised by its own loss and has no length to use; the CE arm does.
    Reporting eta-squared for both is what makes a cosine comparison between
    them readable rather than merely favourable to one side.

    Raises:
        ValueError: If the arrays disagree in length.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    if len(labels) != len(x):
        raise ValueError(f"{len(x)} embeddings but {len(labels)} labels")

    norms = np.linalg.norm(x, axis=1)
    grand = float(norms.mean())
    ss_total = float(((norms - grand) ** 2).sum())

    out = {
        "mean": grand,
        "std": float(norms.std()),
        "min": float(norms.min()),
        "max": float(norms.max()),
    }
    if ss_total < 1e-24:
        # Neither rounded to 0.0 nor raised. A constant norm is a real and
        # informative state -- an encoder that normalises its own output --
        # and eta-squared is genuinely 0/0 there. Said out loud in the file
        # rather than filled in with a number nobody measured.
        out["class_eta_squared"] = None
        out["undefined_because"] = (
            f"every face has norm {grand:.6g} to within 1e-12, so ||z|| has no "
            f"variance for the class to explain and eta-squared is 0/0. The "
            f"length carries nothing here, which is the answer -- but it is "
            f"not a measured 0.0 and is not reported as one.")
        return out

    ss_between = 0.0
    for c in np.unique(labels):
        nc = norms[labels == c]
        ss_between += len(nc) * (float(nc.mean()) - grand) ** 2
    out["class_eta_squared"] = float(ss_between / ss_total)
    return out


def _rank_neighbours(sims: np.ndarray, self_cols: np.ndarray) -> np.ndarray:
    """Descending-similarity ranking of every column except the query itself.

    Ties are broken by column index, deterministically: `kind="stable"` on the
    negated similarities. Float32 cosine similarities DO tie -- two faces of
    the same planar type can land on the same value -- and an unstable sort
    would make the same checkpoint score differently on two runs.

    Args:
        sims: ``(n_queries, n_gallery)`` cosine similarities.
        self_cols: ``(n_queries,)`` column index of each query's own row.

    Returns:
        ``(n_queries, n_gallery - 1)`` column indices, best first, self dropped.
    """
    sims = np.array(sims, dtype=np.float32, copy=True)
    sims[np.arange(len(sims)), self_cols] = -np.inf
    # ascending on -sims == descending on sims; the -inf self entry becomes
    # +inf and therefore sorts LAST, where the slice below drops it.
    order = np.argsort(-sims, axis=1, kind="stable")
    return order[:, :sims.shape[1] - 1]


def _majority_label(neighbour_labels: np.ndarray) -> np.ndarray:
    """Vote over each row's k neighbour labels, nearest-first tie-break.

    The winner is the label with the highest count; a tie goes to whichever of
    the tied labels reached that count FIRST when walking the row nearest to
    farthest, i.e. to the closer neighbour. Fully determined by the input --
    no rng, no dict iteration order.
    """
    out = np.empty(len(neighbour_labels), dtype=neighbour_labels.dtype)
    for r, row in enumerate(neighbour_labels):
        counts: dict = {}
        best_label = row[0]
        best_count = 0
        for lab in row.tolist():
            counts[lab] = counts.get(lab, 0) + 1
            if counts[lab] > best_count:
                best_count = counts[lab]
                best_label = lab
        out[r] = best_label
    return out


def retrieval(embeddings: np.ndarray, labels: np.ndarray, *,
              ks=(1, 5), cap, rng, chunk: int = 256) -> dict:
    """Leave-one-out k-NN accuracy and mAP over the test faces.

    The gallery is the test faces themselves and every query is excluded from
    its own ranking, which is the standard retrieval protocol and fits nothing:
    no parameter is estimated, no threshold is chosen. Cosine similarity, the
    same similarity `PrototypeInfoNCELoss` trains on.

        knn_accuracy_at_k
            fraction of faces whose k nearest OTHER test faces vote for the
            face's own label (majority, nearest-first tie-break). k=1 is
            simply "does the single nearest face share the label".
        mean_average_precision
            for each query, average precision over the FULL ranking of the
            other faces, relevance = same label; then averaged over the
            queries that have at least one same-label face to find. Reads the
            whole ranking, not just the top of it, so a space that gets the
            first neighbour right and the rest wrong does not score like one
            that gets the class right throughout.
        chance_same_class_rate
            the probability that a uniformly drawn OTHER face shares the
            query's label, from this subsample's label histogram:
            ``sum_c p_c * (n_c - 1) / (m - 1)``. That is exactly chance for
            k=1, the reference (up to the vote's tie-breaking) for k>1, and
            the expected mAP of a random ranking. On a 25-class split it is
            near 0.04 if the classes are balanced and much higher if they are
            not -- which is why it is reported beside every score instead of
            being assumed.

    Returns:
        A dict of the scores, the chance rate, and the size measured on.

    Raises:
        ValueError: If `ks` asks for more neighbours than the subsample has,
            or if fewer than two faces survive, or if no query has a
            same-label face anywhere in the gallery.
    """
    labels = np.asarray(labels)
    idx = stratified_subsample(labels, cap, rng)
    emb = unit_rows(np.asarray(embeddings)[idx]).astype(np.float32)
    lab = labels[idx]
    m = len(lab)
    ks = tuple(int(k) for k in ks)

    if m < 2:
        raise ValueError(f"retrieval needs at least 2 faces, got {m}")
    if max(ks) > m - 1:
        raise ValueError(
            f"k={max(ks)} needs {max(ks) + 1} faces and the subsample has {m}. "
            f"Raise --max-retrieval or drop the larger k -- padding the "
            f"neighbourhood with the query itself would score every space at "
            f"1.0.")

    hits = {k: 0 for k in ks}
    ap_sum = 0.0
    ap_queries = 0
    ranks = np.arange(1, m, dtype=np.float64)

    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        sims = emb[start:stop] @ emb.T
        order = _rank_neighbours(sims, np.arange(start, stop))
        neighbour_labels = lab[order]
        query_labels = lab[start:stop]

        for k in ks:
            voted = _majority_label(neighbour_labels[:, :k])
            hits[k] += int((voted == query_labels).sum())

        relevant = neighbour_labels == query_labels[:, None]
        n_relevant = relevant.sum(axis=1)
        precision_at = np.cumsum(relevant, axis=1) / ranks
        ap = (precision_at * relevant).sum(axis=1)
        has_any = n_relevant > 0
        ap_sum += float((ap[has_any] / n_relevant[has_any]).sum())
        ap_queries += int(has_any.sum())

    if ap_queries == 0:
        raise ValueError(
            "not one face has another face of its own class in this "
            "subsample, so average precision is undefined for every query.")

    _, counts = np.unique(lab, return_counts=True)
    p = counts / m
    chance = float((p * (counts - 1) / (m - 1)).sum())

    out = {
        "mean_average_precision": ap_sum / ap_queries,
        "map_queries_scored": ap_queries,
        "chance_same_class_rate": chance,
        "n_faces_scored": int(m),
        "subsampled": bool(m < len(labels)),
        "cap": int(cap) if cap is not None else None,
    }
    for k in ks:
        out["knn_accuracy_at_%d" % k] = hits[k] / m
    return out


#: Written verbatim into every output file. A number whose definition lives
#: only in a docstring is a number someone will misread in six weeks.
MEASURES = {
    "intra_inter_ratio": (
        "mean cosine distance between same-label test faces divided by the "
        "mean cosine distance between different-label ones, over ALL pairs, "
        "exactly. LOWER is better separated; 1.0 is what a space with no "
        "class structure gives."),
    "silhouette_cosine": (
        "mean over faces of (b - a) / max(a, b), a = mean cosine distance to "
        "the face's own class, b = to the nearest other class. +1 separated, "
        "negative means faces sit closer to another class. Measured on a "
        "seeded, class-stratified subsample. Read it against "
        "null_silhouette_label_shuffled, NOT against 0.0."),
    "null_silhouette_label_shuffled": (
        "the same silhouette recomputed on the SAME embeddings with the "
        "labels permuted, five times. This is the chance value FOR THIS "
        "measurement: same width, same norms, same face count, same label "
        "histogram, no label signal. The asymptotic 0.0 is not it -- the "
        "finite-sample floor is near -0.027 at 1,000 faces in 128-d and near "
        "-0.0134 at 4,000, so a split-blind 0.0 makes a chance-level space "
        "look worse than chance."),
    "silhouette_euclidean_control": (
        "the silhouette on the UNNORMALISED embeddings under Euclidean "
        "distance. Every other measure here is a cosine quantity and so "
        "discards ||z||. CMCA is L2-normalised by its own loss and cannot use "
        "the length; the CE head reads w_c.z + b_c and can. This control is "
        "what stops a cosine-only comparison scoring the CE arm at chance for "
        "structure it put somewhere cosine does not look."),
    "class_eta_squared": (
        "the between-class share of the variance of ||z||, in "
        "`embedding_norm`. ~0: the length carries no class information and "
        "normalising it away costs nothing, so the cosine scores are the "
        "whole picture. ~1: the class is written in the LENGTH and a low "
        "silhouette_cosine describes the probe, not the encoder. Report it "
        "for BOTH arms or the cosine comparison between them is not readable."),
    "knn_accuracy_at_k": (
        "leave-one-out retrieval: the fraction of test faces whose k nearest "
        "OTHER test faces vote for the face's own label. Nothing is fitted; "
        "the query is excluded from its own ranking."),
    "mean_average_precision": (
        "mean over queries of the average precision of the full same-label "
        "ranking of the other test faces. Reads the whole ranking, not only "
        "the nearest neighbour."),
    "chance_same_class_rate": (
        "probability that a uniformly drawn other face shares the query's "
        "label, from this subsample's label histogram. Exact chance for k=1 "
        "and the expected mAP of a random ranking. Every retrieval score "
        "above must be read against it."),
    "SCOPE": (
        "These measure whether class structure survives in the embedding "
        "GEOMETRY. They can show two arms' spaces differ. They do NOT show "
        "the difference is useful: no zero-shot, no downstream task, no "
        "decoder is evaluated here. See results-T3.md 8.2.1 and L7 in "
        "limitations-register.md."),
    "WIDTH": (
        "Compare WITHIN a row of the T3 2x2, not across it. The two graph "
        "cells are both d_model=128 and the two ResNet cells both "
        "embedding_dim=512, so CMCA-vs-CE is width-matched either way; graph-"
        "vs-ResNet is not. At matched signal-to-noise a 512-d space scores "
        "about +0.04 higher cosine silhouette than a 128-d one, and its "
        "label-shuffled null sits about +0.013 higher too -- both larger than "
        "the effect being hunted. The ratio and the retrieval scores are "
        "width-free at chance (0.9998 vs 0.9999; k-NN at its histogram "
        "chance); the silhouette is not, which is why its null is measured "
        "per run rather than assumed."),
}


def embedding_space_report(embeddings: np.ndarray, labels: np.ndarray, *,
                           seed: int, silhouette_cap: int = 4000,
                           retrieval_cap: int = 5000, ks=(1, 5)) -> dict:
    """Run every measure on one set of face embeddings.

    This is the whole scorer. Both objectives and both encoders reach it with
    an ``(n_faces, dim)`` float array and an ``(n_faces,)`` label array and
    nothing else, so no part of a comparison between two arms can come from
    the scoring code -- which is the same discipline T3's 2x2 used on
    `evaluate_semantic`.

    Deterministic: the only randomness is the subsample, drawn from
    ``np.random.default_rng(seed)``. The same arrays and the same seed give
    byte-identical output.

    Args:
        embeddings: ``(n_faces, dim)`` face embeddings, unnormalised is fine.
        labels: ``(n_faces,)`` integer class ids. Must not contain -1; the
            caller drops padded and unmapped faces (see `NON_LABEL`).
        seed: Seeds the subsample rng, and is recorded in the output.
        silhouette_cap: Max faces for the O(N^2) silhouette matrix.
        retrieval_cap: Max faces for the leave-one-out ranking.
        ks: Neighbourhood sizes for k-NN accuracy.

    Returns:
        A JSON-serialisable dict: `embedding_norm`, `separation`,
        `silhouette`, `retrieval`, `measures`, and the shape facts each was
        computed from.
    """
    emb = np.asarray(embeddings)
    labels = np.asarray(labels)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {emb.shape}")
    if len(labels) != len(emb):
        raise ValueError(f"{len(emb)} embeddings but {len(labels)} labels")
    if (labels < 0).any():
        raise ValueError(
            f"{int((labels < 0).sum())} labels are negative. -1 is the padding "
            f"and unmapped-class marker (src/data/collate.py:26-28); it is not "
            f"a class and must be dropped before scoring, not averaged in.")

    return {
        "n_faces": int(len(emb)),
        "embedding_dim": int(emb.shape[1]),
        "n_classes_present": int(len(np.unique(labels))),
        "seed": int(seed),
        "embedding_norm": embedding_norms(emb, labels),
        "separation": class_separation(emb, labels),
        "silhouette": silhouette(
            emb, labels, cap=silhouette_cap,
            rng=np.random.default_rng(seed)),
        "retrieval": retrieval(
            emb, labels, ks=ks, cap=retrieval_cap,
            rng=np.random.default_rng(seed)),
        "measures": dict(MEASURES),
    }


#  Getting the embeddings out of a checkpoint -- one path per encoder

def graph_face_embeddings(model, batch):
    """Per-face embeddings from a Graph Transformer model, either objective.

    THE ONE LINE THAT DIFFERS BETWEEN THE ARMS, and the reason
    `SemanticClassificationModel.encode` was written:

      * the CMCA arm's model IS a `BRepEncoder`, so `forward` already returns
        the ``(B, N_faces, d_model)`` embeddings;
      * the CE arm's model wraps that same encoder and its `forward` returns
        ``(B, N_faces, n_classes)`` LOGITS, so asking it to `forward` here
        would hand the scorer the head's output and call it an embedding --
        a different tensor, a different width, and a comparison of an
        embedding space against a logit space.

    `encode` (`src/models/ce_encoder.py:102-113`) returns the CE arm's
    `self.encoder(...)`, i.e. exactly what the CMCA arm's forward returns.
    Dispatching on its presence is what makes the two arms comparable.
    `tests/test_probe_embedding_space.py::test_ce_and_cmca_share_the_encoder_path`
    holds this: with the same encoder weights in both models, the two must come
    back identical.
    """
    encode = getattr(model, "encode", None)
    fn = encode if callable(encode) else model
    return fn(
        batch["faces_uv"], batch["faces_attr"],
        batch["edges_attr"], batch["bias"],
        face_mask=batch["face_mask"],
        edge_mask=batch["edge_mask"],
    )


def resnet_face_embeddings(model, uv_grids):
    """Per-face embeddings from a ResNet model, either objective.

    Both `CrossModalAlignmentModel` and `ResNetClassificationModel` hold the
    same `UVGridEncoder` under the same attribute name
    (`src/models/resnet_encoder.py:113` and `:227`), and it is the tower the
    objective trained. Neither model's `forward` returns it -- both return a
    loss -- so the tower is called directly, and identically for the two arms.
    """
    return model.uv_encoder(uv_grids)


def _model_type(cfg: dict) -> str:
    """The `model.type` this probe dispatches on, legacy key and default included.

    Defined once because it is read twice -- once to choose the embedding path
    and once to record what was chosen -- and the two must never disagree.
    """
    m_cfg = cfg.get("model", cfg.get("semantic_model", {}))
    return m_cfg.get("type", "semantic_segmentation")


def collect_test_embeddings(cfg, ckpt, device, *, split="test", verbose=True):
    """Embed the held-out faces. Returns ``(embeddings, labels, meta)``.

    The model is built by the SAME builders `evaluate_semantic` used for every
    committed T3 number (`_build_graph_transformer` / `_build_resnet`), which
    is also what dispatches CE vs CMCA on `loss.type`. The dataset is built by
    `inference._build_brep_dataset`, which resolves the attribute view, the
    norm-stats file and the bias arm from the config and cross-checks them
    against the checkpoint -- running a model on a view it never trained on
    gives embeddings that look entirely ordinary and are wrong.

    Faces labelled `NON_LABEL` are dropped and counted, never scored.

    Raises:
        ValueError: If `model.type` is not one this probe can reach the face
            embedding of, or if a ResNet config names no split file.
    """
    model_type = _model_type(cfg)

    if model_type in RESNET_MODEL_TYPES:
        model, _label_encoder, n_classes = _build_resnet(cfg, ckpt, device)
        ds_cfg = cfg["dataset"]
        split_file = ds_cfg.get("split_file")
        if not split_file:
            raise ValueError(
                "probe_embedding_space: this ResNet config sets no "
                "dataset.split_file, so there is no record of which faces are "
                "held out. Scoring the whole corpus would measure the "
                "embedding space over faces the model trained on. Point "
                "dataset.split_file at the split this checkpoint was trained "
                "with.")
        dataset = UVGridLabelDataset(
            json_dir=ds_cfg["json_dir"],
            n_classes=ds_cfg.get("n_classes", 25),
            model_ids=load_split(split_file, split),
            verbose=verbose,
        )
        meta = {
            "encoder": "resnet18_uv_tower",
            "dataset_class": "UVGridLabelDataset",
            "split_file": str(split_file),
            "split": split,
            "scored": f"{split} partition of {split_file} "
                      f"({len(dataset)} faces)",
            "n_classes_configured": int(n_classes),
        }
        # shuffle=False -> SequentialSampler and num_workers=0, so neither
        # `generator=` nor `worker_init_fn` is ever read; see the H4 note in
        # src/evaluation/evaluate_semantic.py and
        # tests/test_eval_dataloader_seeding.py.
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        chunks, label_chunks = [], []
        with torch.no_grad():
            for uv_grids, face_labels in loader:
                emb = resnet_face_embeddings(model, uv_grids.to(device))
                chunks.append(emb.cpu().numpy())
                label_chunks.append(face_labels.cpu().numpy())

    elif model_type in GRAPH_MODEL_TYPES:
        model, _label_encoder, n_classes = _build_graph_transformer(
            cfg, ckpt, device)
        dataset, meta = _build_brep_dataset(
            cfg, ckpt, where="probe_embedding_space", split=split,
            verbose=verbose)
        meta = dict(meta)
        meta["encoder"] = "graph_transformer"
        meta["split"] = split
        meta["n_classes_configured"] = int(n_classes)
        loader = DataLoader(
            dataset, batch_size=cfg.get("training", {}).get("batch_size", 4),
            shuffle=False, collate_fn=collate_fn_brep)

        chunks, label_chunks = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                emb = graph_face_embeddings(model, batch)
                mask = batch["face_mask"]
                chunks.append(emb[mask].cpu().numpy())
                label_chunks.append(
                    batch["semantic_labels"][mask].cpu().numpy())

    else:
        raise ValueError(
            f"probe_embedding_space does not know how to reach the face "
            f"embedding of model.type={model_type!r}. Known: "
            f"{GRAPH_MODEL_TYPES + RESNET_MODEL_TYPES}. Guessing would score "
            f"whatever tensor forward() happens to return -- for the CE arm "
            f"that is a logit vector, not an embedding.")

    embeddings = np.concatenate(chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)

    keep = labels != NON_LABEL
    meta["n_faces_embedded"] = int(len(labels))
    meta["n_faces_dropped_unlabelled"] = int((~keep).sum())
    return embeddings[keep], labels[keep], meta


#  The run

def probe(cfg: dict, config_path, checkpoint_path, *, seed: int, device: str,
          split: str = "test", silhouette_cap: int = 4000,
          retrieval_cap: int = 5000, ks=(1, 5), verbose: bool = True) -> dict:
    """Load a checkpoint, embed the test faces, score the space.

    Returns the full record, provenance included. Writes nothing -- `main`
    decides where the record goes, so importing this function cannot touch the
    filesystem.
    """
    set_global_seed(seed)
    ckpt = load_checkpoint(checkpoint_path, device=device)

    embeddings, labels, data_meta = collect_test_embeddings(
        cfg, ckpt, device, split=split, verbose=verbose)

    report = embedding_space_report(
        embeddings, labels, seed=seed, silhouette_cap=silhouette_cap,
        retrieval_cap=retrieval_cap, ks=ks)

    record = {
        "probe": "embedding_space",
        "experiment_name": cfg.get("experiment_name", "unnamed"),
        "objective": cfg.get("loss", {}).get("type"),
        # The same lookup `collect_test_embeddings` dispatches on, including
        # the `semantic_model` fallback and the default.
        "model_type": _model_type(cfg),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "data_view": data_meta,
    }
    record.update(report)
    # The same shape `eval_metrics.json` carries: the TRAINING provenance
    # copied out of the checkpoint (it describes the run that produced the
    # weights, not this scoring pass) plus an `evaluated` sub-block whose
    # `code_fingerprint` is this tree's. That is what makes the file gateable
    # evidence instead of a loose number -- see
    # src/utils/provenance.provenance_from_checkpoint.
    record["provenance"] = provenance_from_checkpoint(ckpt, cfg)
    record["probe_run"] = {
        "code_fingerprint": code_fingerprint(),
        "seed": int(seed),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
    }
    return record


def print_report(record: dict) -> None:
    """Human-readable summary. Every score beside its chance value."""
    sep = record["separation"]
    sil = record["silhouette"]
    ret = record["retrieval"]
    print("\n" + "=" * 70)
    print(f"EMBEDDING-SPACE PROBE -- {record['experiment_name']}  "
          f"(objective: {record['objective']})")
    print("=" * 70)
    print(f"  faces scored          {record['n_faces']}  "
          f"({record['n_classes_present']} classes, "
          f"dim {record['embedding_dim']})")
    print(f"  intra/inter ratio     {sep['intra_inter_ratio']:.4f}"
          f"   (chance 1.0; lower is better separated)")
    print(f"     intra cos distance {sep['mean_intra_cosine_distance']:.4f}")
    print(f"     inter cos distance {sep['mean_inter_cosine_distance']:.4f}")
    null = sil["null_silhouette_label_shuffled"]
    print(f"  silhouette (cosine)   {sil['silhouette_cosine']:+.4f}"
          f"   (measured null {null['mean']:+.4f} +- {null['std']:.4f} over "
          f"{null['n_permutations']} label shuffles;"
          f" on {sil['n_faces_scored']} faces)")
    print(f"     euclidean control  {sil['silhouette_euclidean_control']:+.4f}"
          f"   (reads ||z||, which every cosine measure above discards)")
    eta = record["embedding_norm"]["class_eta_squared"]
    eta_s = "n/a (norm is constant)" if eta is None else f"{eta:.4f}"
    print(f"     norm eta-squared   {eta_s}"
          f"   (share of ||z|| variance the class explains; "
          f"near 0 means nothing was discarded)")
    chance = ret["chance_same_class_rate"]
    for key in sorted(k for k in ret if k.startswith("knn_accuracy_at_")):
        print(f"  {key:<21} {ret[key]:.4f}   (chance {chance:.4f})")
    print(f"  mAP                   {ret['mean_average_precision']:.4f}"
          f"   (chance {chance:.4f}; on {ret['n_faces_scored']} faces)")
    print("\n  These read the GEOMETRY. They can show two arms' spaces "
          "differ;\n  they do not show the difference is useful. "
          "See MEASURES['SCOPE'].")
    print("  Compare WITHIN a row of the T3 2x2 (both graph cells are "
          "d_model=128,\n  both ResNet cells embedding_dim=512). Across rows "
          "the width confounds\n  the silhouette. See MEASURES['WIDTH'].")


def main():
    ap = argparse.ArgumentParser(
        description="Probe a checkpoint's face-embedding space: class "
                    "separation and retrieval, not argmax accuracy.")
    ap.add_argument("--config", required=True,
                    help="the run's OWN config.resolved.yaml")
    ap.add_argument("--checkpoint", required=True, help="the run's .pth")
    ap.add_argument("--out", required=True,
                    help="where to write the JSON record (required: this tool "
                         "writes nothing by default)")
    ap.add_argument("--seed", type=int, required=True,
                    help="seeds the subsamples; use the run's own seed")
    ap.add_argument("--split", default="test",
                    choices=("train", "val", "test"),
                    help="partition to embed (default test)")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu (default: cuda when available)")
    ap.add_argument("--max-silhouette", type=int, default=4000,
                    help="cap on faces for the O(N^2) silhouette matrix")
    ap.add_argument("--max-retrieval", type=int, default=5000,
                    help="cap on faces for the leave-one-out ranking")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5],
                    help="neighbourhood sizes for k-NN accuracy")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    record = probe(
        cfg, args.config, args.checkpoint, seed=args.seed, device=device,
        split=args.split, silhouette_cap=args.max_silhouette,
        retrieval_cap=args.max_retrieval, ks=tuple(args.k))

    print_report(record)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nProbe record written to {out}")


if __name__ == "__main__":
    main()
