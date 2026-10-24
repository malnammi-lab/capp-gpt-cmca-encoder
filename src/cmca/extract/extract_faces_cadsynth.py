#!/usr/bin/env python3
"""
extract_faces_cadsynth.py -- per-face UV grids for the ResNet-18 baseline.

Writes `<model>_faces.json`, the flat face-level corpus that
`src/data/uv_dataset.py` (UVGridLabelDataset) consumes.

The geometry is not reimplemented here. This script imports
`BRepFeatureExtractor` from `extract_features_cadsynth.py` and calls the same
`extract_face_uv_grid` the graph corpus was built with. The ResNet row and
the Graph Transformer row then see the same normals, the same trim mask,
the same unit-cube coordinates, and the same face order.

`load_labels` reports a face-count mismatch instead of padding it with
class 0. The script walks the STEP directory, so the face corpus and the
graph corpus cover the same parts.

Usage
-----
    # no CAD data needed -- checks the geometry kernel paths
    python extract_faces_cadsynth.py --self-test

    python extract_faces_cadsynth.py \
        --step-dir  <step-dir> \
        --label-dir <label-dir> \
        --out-dir   <faces-dir> \
        --skip-existing

    python extract_faces_cadsynth.py --check-json /path/to/extracted_faces_v2

Needs the extraction environment (env-extraction.yml, conda env `pyoccenv`);
pythonocc-core 7.9.0 has no Python 3.13 build.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_shared():
    """Import the shared extractor by file path.

    Both scripts are run directly (`python extract_faces_cadsynth.py`) rather
    than as a package, so a relative
    import would fail. Falls back to a package import when one is available.
    """
    try:
        from .extract_features_cadsynth import (  # type: ignore
            BRepFeatureExtractor, load_labels, self_test)
        return BRepFeatureExtractor, load_labels, self_test
    except Exception:
        pass
    path = HERE / "extract_features_cadsynth.py"
    if not path.exists():
        raise SystemExit(
            f"extract_features_cadsynth.py not found next to this script "
            f"({path}). Both files must sit in the same directory -- this one "
            f"deliberately does not reimplement the geometry.")
    spec = importlib.util.spec_from_file_location("_shared_extractor", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BRepFeatureExtractor, mod.load_labels, mod.self_test


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def extract_one(ex, load_labels, step_file: Path, label_file: Path,
                label_format: str = "auto") -> dict:
    """One model -> the `*_faces.json` payload."""
    shape = ex.load_step_file(step_file)
    faces, _edges, _fmap = ex.topology(shape)
    centre, scale = ex.unit_cube_transform(shape)

    lab = load_labels(label_file, len(faces), label_format)
    labels = lab["semantic_labels"]

    face_data = []
    for i, face in enumerate(faces):
        grid = ex.extract_face_uv_grid(face, centre, scale)
        face_data.append({
            "face_id": i,
            "label": int(labels[i]),
            "uv_grid": grid.tolist(),
        })

    return {
        "model_id": step_file.stem,
        "step_file": str(step_file),
        "label_file": str(label_file),
        "n_faces": len(faces),
        "uv_grid_size": list(ex.uv_grid_size),
        "normalize": ex.normalize,
        "unit_centre": list(centre),
        "unit_scale": float(scale),
        "extractor": Path(__file__).name,
        "schema_version": "2.0-faces",
        "faces": face_data,
    }


def run(args: argparse.Namespace) -> int:
    BRepFeatureExtractor, load_labels, _ = _load_shared()

    step_dir, label_dir = Path(args.step_dir), Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = sorted(step_dir.glob("*.stp")) + sorted(step_dir.glob("*.step"))
    if args.limit:
        steps = steps[: args.limit]
    if not steps:
        print(f"No STEP files under {step_dir}", file=sys.stderr)
        return 1

    ex = BRepFeatureExtractor(uv_grid_size=tuple(args.grid_size),
                              normalize=args.normalize)

    print(f"{len(steps)} STEP files -> {out_dir}")
    print(f"grid {tuple(args.grid_size)}  normalize={args.normalize}\n")

    ok = skipped = failed = 0
    n_labelled = 0
    for k, step_file in enumerate(steps, 1):
        out_file = out_dir / f"{step_file.stem}_faces.json"
        if args.skip_existing and out_file.exists():
            skipped += 1
            continue
        label_file = label_dir / f"{step_file.stem}.json"
        try:
            payload = extract_one(ex, load_labels, step_file, label_file,
                                  args.label_format)
            out_file.write_text(json.dumps(payload))
            ok += 1
            n_labelled += sum(1 for f in payload["faces"] if f["label"] != 0)
            if k % 50 == 0 or k == len(steps):
                print(f"  [{k}/{len(steps)}] {ok} written, {skipped} skipped, "
                      f"{failed} failed")
        except Exception as exc:
            failed += 1
            print(f"  FAILED {step_file.name}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            if args.traceback:
                import traceback
                traceback.print_exc()

    print(f"\nwritten {ok} · skipped {skipped} · failed {failed}")
    print(f"non-zero-label faces written: {n_labelled:,}")
    if failed:
        print("Re-run with --skip-existing to retry only the failures.")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def check_json(path: Path) -> int:
    """Validate a finished face corpus -- the counterpart of the feature
    extractor's --check-json.

    Checks that the normals are unit vectors (a tangent would also be
    unit-length, but a zero vector is the signature of a failed sample), and
    that the trim mask is not uniformly 1.0.
    """
    files = sorted(Path(path).glob("*_faces.json"))
    if not files:
        print(f"No *_faces.json under {path}", file=sys.stderr)
        return 1

    n_models = n_faces = 0
    mask_all_one = 0
    norm_bad = 0
    coord_max = 0.0
    label_hist: dict = {}
    grid_sizes = set()

    for f in files:
        d = json.loads(f.read_text())
        n_models += 1
        grid_sizes.add(tuple(d.get("uv_grid_size", [])))
        for face in d["faces"]:
            n_faces += 1
            g = np.asarray(face["uv_grid"], dtype=np.float32)
            label_hist[face["label"]] = label_hist.get(face["label"], 0) + 1
            coord_max = max(coord_max, float(np.abs(g[0:3]).max()))
            mask = g[6]
            if float(mask.min()) == 1.0:
                mask_all_one += 1
            valid = mask > 0.5
            if valid.any():
                mag = np.linalg.norm(g[3:6][:, valid], axis=0)
                if not np.allclose(mag, 1.0, atol=1e-3):
                    norm_bad += 1

    print(f"models {n_models:,} · faces {n_faces:,}")
    print(f"uv_grid_size(s) present: {sorted(grid_sizes)}")
    print(f"max |coordinate| = {coord_max:.4f}"
          f"   {'OK (unit cube -> <= ~1.74)' if coord_max <= 2.0 else 'NOT normalised'}")
    print(f"faces whose trim mask is entirely 1.0: {mask_all_one:,}"
          f" ({mask_all_one / max(n_faces,1):.1%})")
    print("   -- untrimmed faces legitimately have an all-ones mask; the old"
          " corpus had 100%.")
    print(f"faces with non-unit normals on valid samples: {norm_bad:,}")
    print(f"distinct labels: {len(label_hist)}  "
          f"top: {sorted(label_hist.items(), key=lambda kv: -kv[1])[:5]}")

    problems = []
    if coord_max > 2.0:
        problems.append("coordinates are not unit-cube normalised")
    if norm_bad:
        problems.append(f"{norm_bad} faces have non-unit normals")
    if n_faces and mask_all_one == n_faces:
        problems.append("every trim mask is all-ones -- FIX 6 is not in effect")
    for p in problems:
        print(f"  PROBLEM: {p}", file=sys.stderr)
    return 1 if problems else 0


# --------------------------------------------------------------------------- #

def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description="Per-face UV grids for the ResNet baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step-dir")
    p.add_argument("--label-dir")
    p.add_argument("--out-dir")
    p.add_argument("--grid-size", type=int, nargs=2, default=[8, 8])
    p.add_argument("--normalize", default="unitcube",
                   choices=["unitcube", "dataset", "model", "legacy"],
                   help="unitcube (UV-Net, default) puts coordinates in [-1, 1]")
    p.add_argument("--label-format", default="auto",
                   choices=["auto", "cadsynth", "mfinstaseg"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="resume: leave already-written files alone")
    p.add_argument("--traceback", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="run the shared extractor's geometry self-test")
    p.add_argument("--check-json", metavar="DIR",
                   help="validate a finished face corpus")
    args = p.parse_args(argv)

    if args.self_test:
        _, _, self_test = _load_shared()
        return self_test()
    if args.check_json:
        return check_json(Path(args.check_json))

    missing = [f for f in ("step_dir", "label_dir", "out_dir")
               if not getattr(args, f)]
    if missing:
        p.error("required: " + ", ".join("--" + m.replace("_", "-")
                                         for m in missing))
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
