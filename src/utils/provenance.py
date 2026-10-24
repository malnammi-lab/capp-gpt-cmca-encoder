"""
Run provenance: the six fields every checkpoint and metrics file must carry.

`training-regiment.md` section 1 (Logging) requires that every checkpoint and
every ``eval_metrics.json`` carries::

    seed, epoch, config hash, code version/commit, GPU type, wall time

and notes "(they are missing today)".  They were: a survey of the six committed
metrics files found the keys to be exactly

    checkpoint, confusion_matrix, experiment_name, macro_f1, macro_precision,
    macro_recall, n_classes, n_samples, overall_accuracy, per_class

with none of the six present.  Under the regiment's own rule -- "every cell
traceable to eval_metrics.json with seed/config/commit recorded" -- every
number produced before this module existed is unquotable.

On "commit"
-----------
The logging rule says *commit*, but this folder may have no ``.git``.
There is then no commit hash to record.

Rather than leave the field empty or write a fake one, `code_fingerprint()`
hashes the source tree itself: every tracked ``*.py`` under ``src/`` and
``runners/``, sorted by path, with line endings normalised so two checkouts
of the same code produce the same digest. That answers the question the
commit hash was there to answer: was this number produced by the code now
in the tree? When a git repository is present, `git_commit()` records a
real hash as well.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

#: Source trees that define "the code that produced this number".
_CODE_ROOTS = ("src", "runners")


def _repo_root() -> Path:
    """The codebase root, i.e. the directory holding ``src/`` and ``runners/``."""
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=8)
def code_fingerprint(root: Path | None = None) -> str:
    """Return a 16-hex-char digest of every ``*.py`` under src/ and runners/.

    Line endings are normalised to ``\\n`` and ``__pycache__`` is skipped, so
    the same code on Windows and Linux agree as long as
    the code agrees.  Both the path and the contents feed the hash, so moving
    or renaming a file changes the fingerprint.

    Raises:
        FileNotFoundError: If no source files are found at all. That means the
            function is being called from somewhere unexpected, and silently
            returning a digest of nothing would be worse than stopping.
    """
    root = Path(root) if root is not None else _repo_root()
    digest = hashlib.sha256()
    n = 0
    for rel in _CODE_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix()):
            if "__pycache__" in f.parts:
                continue
            digest.update(f.relative_to(root).as_posix().encode("utf-8"))
            digest.update(f.read_bytes().replace(b"\r\n", b"\n"))
            n += 1
    if n == 0:
        raise FileNotFoundError(
            f"code_fingerprint found no .py files under {root}/{{{','.join(_CODE_ROOTS)}}}. "
            f"Called from an unexpected working directory?"
        )
    return digest.hexdigest()[:16]


def _canonical(cfg: dict) -> str:
    """Serialise a config dict to the one byte string both hashes digest.

    Key order is normalised and whitespace removed, so the digest tracks values
    and not formatting. ``default=str`` covers the YAML scalars json does not
    know (dates, and anything a future loader returns as an object).
    """
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: dict) -> str:
    """Return a 16-hex-char digest of the *resolved* config dict.

    Hashing the loaded dict rather than the YAML text means a comment change
    does not alter the hash but a value change always does. Two runs with the
    same config hash used the same settings, whatever the file looked like.

    This is the RUN identity: it includes the seed and the output paths, so the
    five seeds of one arm have five different values. `arm_config_hash` is the
    ARM identity, which is the one that is supposed to agree across them.
    """
    return hashlib.sha256(_canonical(cfg).encode("utf-8")).hexdigest()[:16]


#: Seed and paths are excluded. run_all rewrites both per seed.
ARM_HASH_EXCLUDED_KEYS = ("seed", "paths")


def arm_config_hash(cfg: dict) -> str:
    """Return a 16-hex-char digest of the config with seed and outputs removed.

    "Record a second, seed-and-paths-free config hash so arms
    are comparable across seeds."

    `config_hash` answers "is this the same RUN?" and therefore differs across
    the five seeds of one arm -- `runners/run_all.write_resolved_config` pins a
    different ``seed`` and a different ``paths`` block into each. That leaves no
    way to assert "these five runs are the same experiment with a different
    seed", which is the claim the paper's error bars rest on, and no way to
    catch a seed that accidentally ran a different config. This hash answers
    that question instead: it is identical across the seeds of one arm and
    differs the moment any setting that can change a number changes.

    Only the keys in `ARM_HASH_EXCLUDED_KEYS` are dropped, and that constant
    carries the justification for each one, including the near misses that are
    deliberately kept.

    Args:
        cfg: A resolved config dict, as written to
            ``results/<arm>/seed<k>/config.resolved.yaml``.

    Returns:
        The digest, in the same form and length as `config_hash`.

    Raises:
        TypeError: If `cfg` is not a dict. A list or a string would serialise
            happily and produce a digest that means nothing.
        ValueError: If nothing survives the exclusion. The digest of ``{}`` is
            a constant, so every arm in the table would "agree" and the check
            would pass having compared nothing -- the same shape of defect as
            a gate that passes because it had no inputs.
    """
    if not isinstance(cfg, dict):
        raise TypeError(
            f"arm_config_hash expects the resolved config dict, got "
            f"{type(cfg).__name__}. Load the YAML with yaml.safe_load first.")

    stripped = {k: v for k, v in cfg.items() if k not in ARM_HASH_EXCLUDED_KEYS}
    if not stripped:
        raise ValueError(
            f"arm_config_hash: nothing is left of this config after dropping "
            f"{list(ARM_HASH_EXCLUDED_KEYS)}; its only keys were "
            f"{sorted(cfg)}. Hashing an empty dict gives every arm the same "
            f"digest, so the cross-seed check would pass having compared "
            f"nothing. Pass the resolved config, not a fragment of one.")
    return hashlib.sha256(_canonical(stripped).encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=8)
def git_commit(root: Path | None = None) -> str | None:
    """Return the short git commit of the codebase, or None when there is no repo.

    This returns None today -- `codebase/` has no `.git`. It is here so that the
    field starts populating itself automatically on the day task 0b.1 creates
    the GitHub repo, with no further code change.
    """
    root = Path(root) if root is not None else _repo_root()
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        # git is not installed on this machine. A specific, expected absence,
        # not a swallowed failure: without this the call raises and every
        # checkpoint save crashes on a machine with no git. The caller records
        # None and code_fingerprint carries the provenance instead.
        return None
    if proc.returncode != 0:          # not a repository -- the normal case here
        return None
    sha = proc.stdout.strip()
    if not sha:
        return None

    # `git -C` walks UP the tree, so if codebase/ is ever dropped inside an
    # unrelated repository this would silently record that repository's commit
    # as this code's provenance. Record where the repo root actually is so the
    # claim is checkable rather than assumed.
    top = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True)
    if top.returncode == 0:
        found = Path(top.stdout.strip())
        if found.resolve() != root.resolve():
            return f"{sha} (from {found}, NOT the codebase root)"
    return sha


def gpu_name() -> str:
    """Return the CUDA device name, or an explicit CPU marker."""
    if not torch.cuda.is_available():
        return "cpu (no CUDA device)"
    return torch.cuda.get_device_name(0)


#: The only two legal values of ``stopping_reason``.
#: "early_stop" -- validation loss stopped improving and patience ran out.
#: "cap"        -- the loop reached ``training.max_epochs`` with patience intact.
#: The cap audit (check_convergence.py in its demoted role) exists to find the
#: second case, so the string has to be exact rather than free text.
STOPPING_REASONS = ("early_stop", "cap")


def build_provenance(
    cfg: dict,
    *,
    seed: int,
    epoch: int | None = None,
    wall_time_s: float | None = None,
    selected_epoch: int | None = None,
    stopping_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance block written into checkpoints and metrics files.

    Args:
        cfg: The resolved config dict for this run.
        seed: The seed actually used, as returned by `set_global_seed`.
        epoch: Final or current epoch. This is the epoch the
            loop had reached when the block was written.
        wall_time_s: Seconds of wall-clock training time.
        selected_epoch: The epoch of the checkpoint that
            is actually evaluated, i.e. the best-validation-loss epoch. Left
            None while a run is still in flight; `stamp_selection` fills it in
            once the loop has finished and the answer is known.
        stopping_reason: One of `STOPPING_REASONS`.
            Also None until the loop ends.
        extra: Extra keys to merge in (e.g. the tier name from run_all).

    Returns:
        A flat dict. Keys are stable; downstream tables read them by name.

    Raises:
        ValueError: If `stopping_reason` is given but is not one of
            `STOPPING_REASONS`. A free-text reason would pass the field check
            and then silently defeat the cap audit, which matches on the exact
            string "cap".
    """
    if stopping_reason is not None and stopping_reason not in STOPPING_REASONS:
        raise ValueError(
            f"stopping_reason must be one of {STOPPING_REASONS}, "
            f"got {stopping_reason!r}. A run ends in exactly "
            f"one of {STOPPING_REASONS}, and the convergence check matches on the literal "
            f"string, so a third value is not a label but a hole in the check.")

    prov: dict[str, Any] = {
        "seed": int(seed),
        "epoch": epoch,
        # Written inside the best-validation branch, so epoch and
        # selected_epoch match. A disagreement means a save outside that branch.
        "selected_epoch": (int(selected_epoch)
                           if selected_epoch is not None else None),
        "stopping_reason": stopping_reason,
        "use_class_weights": cfg.get("loss", {}).get("use_class_weights"),
        "epoch_budget_configured": cfg.get("training", {}).get("max_epochs"),
        "config_hash": config_hash(cfg),
        "arm_config_hash": arm_config_hash(cfg),
        "code_fingerprint": code_fingerprint(),
        "git_commit": git_commit(),          # None until the repo exists
        "gpu": gpu_name(),
        "wall_time_s": round(float(wall_time_s), 1) if wall_time_s is not None else None,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        prov.update(extra)
    return prov


def provenance_from_checkpoint(ckpt: dict, cfg: dict) -> dict[str, Any]:
    """Carry a checkpoint's training provenance into an ``eval_metrics.json``.

    The six regiment fields describe the run that produced the weights, so the
    evaluator copies them out of the checkpoint rather than inventing new ones:
    re-scoring a checkpoint on a different machine must not overwrite the GPU
    or wall time of the training run.

    A checkpoint saved before provenance existed has none. That is recorded
    explicitly, as ``{"provenance_missing": ...}``, instead of being quietly
    filled in with the evaluating machine's values -- a fabricated provenance
    block is worse than an absent one, because it looks quotable.

    Args:
        ckpt: The loaded checkpoint dict.
        cfg: The config used for evaluation, for the config hash.

    Returns:
        The training provenance, plus an ``evaluated`` sub-block describing
        this scoring run.
    """
    train_prov = ckpt.get("provenance")
    if not train_prov:
        return {
            "provenance_missing": (
                "This checkpoint predates run provenance (added 2026-09-17) and "
                "carries no seed, config hash, code fingerprint, GPU or wall "
                "time. Under training-regiment.md section 4 its numbers are NOT "
                "quotable. Retrain through runners/run_all.py."
            ),
            "epoch": ckpt.get("epoch"),
            "evaluated": {
                "config_hash": config_hash(cfg),
                "code_fingerprint": code_fingerprint(),
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        }

    prov = dict(train_prov)
    prov["evaluated"] = {
        "config_hash": config_hash(cfg),
        "code_fingerprint": code_fingerprint(),
        "gpu": gpu_name(),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return prov


def check_regiment_fields(prov: dict) -> list[str]:
    """Return the names of required regiment fields that are missing or None.

    An empty list means the run is traceable by the regiment's checklist. Used
    by `runners/run_all.py` and by the tests, so a run that forgets a field
    fails loudly instead of producing an unquotable metrics file that looks fine.

    The required set includes ``selected_epoch`` and ``stopping_reason``.
    A missing ``code_fingerprint`` is a missing field. ``git_commit`` is
    recorded and is not a substitute.
    """
    required = ("seed", "epoch", "config_hash", "gpu", "wall_time_s",
                "selected_epoch", "stopping_reason")
    missing = [k for k in required if prov.get(k) is None]
    if not prov.get("code_fingerprint"):
        missing.append("code_fingerprint")
    return missing


def stamp_selection(
    ckpt_path: str | Path,
    *,
    selected_epoch: int,
    stopping_reason: str,
) -> dict[str, Any]:
    """Record which epoch was selected, and why training stopped.

    The best-validation-loss checkpoint is written the
    moment it is the best one seen, which is necessarily BEFORE the run knows
    how it ends. The two facts the cap audit needs -- which epoch was selected,
    and whether patience or the cap stopped training -- are therefore only
    knowable after the loop exits. Every trainer calls this once at that point.

    Reading and rewriting the file is deliberate: the alternative is holding the
    best weights in memory and saving once at the end, which would throw away
    the crash resilience the per-improvement save exists for. A resumed run
    depends on a file being there.

    Args:
        ckpt_path: The checkpoint the trainer selected, as written by
            `save_checkpoint`.
        selected_epoch: The epoch whose weights are in that file.
        stopping_reason: One of `STOPPING_REASONS`.

    Returns:
        The updated provenance block, so the caller can log or assert on it.

    Raises:
        FileNotFoundError: If no checkpoint was ever written. That means the
            run never recorded a best epoch, which is a failed run, not a run
            with a missing label.
        ValueError: If `stopping_reason` is not one of `STOPPING_REASONS`, or
            if the checkpoint carries no provenance block to stamp.
    """
    if stopping_reason not in STOPPING_REASONS:
        raise ValueError(
            f"stopping_reason must be one of {STOPPING_REASONS}, "
            f"got {stopping_reason!r}.")

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"stamp_selection: no checkpoint at {ckpt_path}. A "
            f"run selects its best-validation-loss epoch, so a finished run "
            f"always has one saved. Nothing here means training never completed "
            f"an epoch that improved on inf -- a failed run, not a quotable one.")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prov = payload.get("provenance")
    if not prov:
        raise ValueError(
            f"stamp_selection: {ckpt_path} carries no provenance block. The "
            f"trainer that wrote it is not passing build_provenance() to "
            f"save_checkpoint(), so the run is untraceable and unquotable.")

    prov["selected_epoch"] = int(selected_epoch)
    prov["stopping_reason"] = stopping_reason
    payload["provenance"] = prov
    torch.save(payload, ckpt_path)
    print(f"Selection stamped on {ckpt_path}: epoch {selected_epoch}, "
          f"stopping_reason={stopping_reason}")
    return prov
