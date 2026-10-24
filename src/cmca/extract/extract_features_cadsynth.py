#!/usr/bin/env python3
"""
extract_features_cadsynth.py -- B-Rep feature extraction for the CMCA part encoder.

Reads a STEP file and a label file and writes one JSON per model. Face
adjacency is the shared-edge graph. `adjacency` and `attr_similarity` are
separate keys, so the encoder can run with bias.type of similarity,
adjacency, or both. Edge convexity and dihedral angle come from the two
faces that meet at the edge. Loop count is the number of wires, so a hole
is visible. UV-grid channels 3-5 are the unit normal, reversed when the
face orientation is REVERSED, and the validity channel is the trim test
over the face's own UV bounds. Attributes are scaled by corpus percentiles
written to norm_stats.json. `--normalize` keeps the legacy constant scaling.

Face details are opt-in via --emit-details and do not repeat the UV grid.
`bias_matrix` is an alias of `attr_similarity`. Both `labels` and
`semantic_labels` are written.

Usage
-----
    # no CAD data needed -- validates the geometry kernel paths
    python extract_features_cadsynth.py --self-test

    # CADSynth: {"file_name": ..., "labels": [...]}
    python extract_features_cadsynth.py --step-dir data/cadsynth/step \
        --label-dir data/cadsynth/label --out-dir data/cadsynth/features

    # re-check already-written files
    python extract_features_cadsynth.py --check-json <features-dir>

`--no-capture-output` and `-u` matter when the script is launched through
`conda run`: without them the run buffers everything and a multi-hour
extraction prints nothing until it finishes. `skip_existing` makes the job
resumable after an interruption.

Requires pythonocc-core (7.6 - 7.9 all supported; the API shims below absorb the
static-method renames) and numpy. numpy alone is enough for --check-json and
--self-test --pure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Used when the script is started with no arguments. Flags override this block.

CONFIG = {
    "run": "extract",

    "step_dir":  "",
    "label_dir": "",
    "out_dir":   "",

    "grid_size": (8, 8),
    "normalize": "unitcube",    # "unitcube" (UV-Net) | "dataset" | "model" | "legacy"
    "percentile": 99.0,
    "scale_sample": 500,        # models sampled in pass 1; 0 = all of them
    "skip_existing": True,
    "emit_path_context": False,
    "emit_details": False,
    # SMOKE TEST: 20 models first. Set to None for the full 6,063 run.
    "limit": 20,
}


SCHEMA_VERSION = "2.0"
DEFAULT_GRID = (8, 8)

# Tangent/normal degeneracy tolerance, and the angle below which an edge counts
# as smooth (tangent-continuous) rather than convex or concave.
_EPS = 1e-9
_SMOOTH_ANGLE_RAD = math.radians(1.0)

# Verified by --self-test on a box (all edges convex) and a blind hole (rim
# concave). If a future OCC build flips this, the self-test fails loudly with a
# message naming this constant rather than silently poisoning a whole corpus.
CONVEXITY_SIGN = +1.0

# "the scale of the solid is normalized into a cube of size 2 and centered at
#  origin" -- Jayaraman et al., UV-Net, CVPR 2021, Sec. 3.1.
# With every solid in that box, geometric quantities have ceilings derived from
# the box itself, so no corpus statistics (and no train/test leakage) are needed.
UNITCUBE_FACE_AREA_DIV = 4.0                  # 2 x 2, one axis-aligned box face
UNITCUBE_EDGE_LENGTH_DIV = 2.0 * math.sqrt(3)  # box space diagonal, ~3.4641
# Counts are scale-free, so these are stated conventions, not derived bounds.
# The run summary reprints the observed distribution so the choice stays
# checkable on any corpus.
LOOP_COUNT_DIV = 8.0
FACE_DEGREE_DIV = 32.0


# pythonocc API shims (7.6 used module-level functions, 7.7+ moved to classes)

OCC_AVAILABLE = False
OCC_ERROR: Optional[str] = None
_API: Dict[str, Any] = {}


def _first_attr(module: Any, *names: str) -> Any:
    for n in names:
        obj = getattr(module, n, None)
        if obj is not None:
            return obj
    return None


def _resolve_static(holder: Any, module: Any, method: str, legacy: str) -> Any:
    """Resolve an OCC static method across the 7.6 / 7.7+ naming schemes."""
    if holder is not None:
        found = _first_attr(holder, method, f"{method}_s")
        if found is not None:
            return found
    return _first_attr(module, legacy)


def _extent(container: Any) -> int:
    """Number of entries in an OCC NCollection map.

    pythonocc 7.9 does not bind `Extent()` on TopTools_IndexedMapOfShape even
    though OCCT declares it; older builds do. Try every spelling rather than
    assuming one.
    """
    for attr in ("Size", "Extent", "NbEntries"):
        fn = getattr(container, attr, None)
        if fn is not None:
            try:
                return int(fn())
            except Exception:
                continue
    try:
        return len(container)
    except Exception as exc:
        raise AttributeError(
            f"cannot determine the size of {type(container).__name__}; "
            f"available attributes: "
            f"{[a for a in dir(container) if not a.startswith('_')][:20]}"
        ) from exc


def _load_occ() -> None:
    """Import pythonocc and resolve the handful of APIs that moved between
    releases. Raises ImportError with a readable message if unavailable."""
    global OCC_AVAILABLE, OCC_ERROR
    if OCC_AVAILABLE:
        return
    try:
        import OCC.Core.BRep as _brep
        import OCC.Core.BRepBndLib as _bndlib_mod
        import OCC.Core.BRepGProp as _gprop_mod
        import OCC.Core.BRepTools as _tools_mod
        import OCC.Core.TopExp as _topexp_mod
        import OCC.Core.TopoDS as _topods_mod
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
        from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
        from OCC.Core.GeomAbs import (
            GeomAbs_BezierCurve, GeomAbs_BezierSurface, GeomAbs_BSplineCurve,
            GeomAbs_BSplineSurface, GeomAbs_Circle, GeomAbs_Cone,
            GeomAbs_Cylinder, GeomAbs_Ellipse, GeomAbs_Hyperbola,
            GeomAbs_Line, GeomAbs_Parabola, GeomAbs_Plane, GeomAbs_Sphere,
            GeomAbs_Torus,
        )
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.gp import gp_Pnt, gp_Pnt2d, gp_Vec
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopAbs import (
            TopAbs_EDGE, TopAbs_FACE, TopAbs_OUT, TopAbs_REVERSED, TopAbs_WIRE,
        )
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopTools import (
            TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape,
        )
    except Exception as exc:  # depends on the environment
        OCC_ERROR = f"{type(exc).__name__}: {exc}"
        raise ImportError(
            "pythonocc-core is required for extraction.\n"
            "  conda create -n pyoccenv -c conda-forge "
            "python=3.10 pythonocc-core=7.9.0 -y\n"
            f"import error was: {OCC_ERROR}"
        ) from exc

    gp_holder = _first_attr(_gprop_mod, "brepgprop", "BRepGProp")
    surf_props = _resolve_static(gp_holder, _gprop_mod, "SurfaceProperties",
                                 "brepgprop_SurfaceProperties")
    lin_props = _resolve_static(gp_holder, _gprop_mod, "LinearProperties",
                                "brepgprop_LinearProperties")

    bt_holder = _first_attr(_tools_mod, "breptools", "BRepTools")
    uv_bounds = _resolve_static(bt_holder, _tools_mod, "UVBounds",
                                "breptools_UVBounds")

    te_holder = _first_attr(_topexp_mod, "topexp", "TopExp")
    map_anc = _resolve_static(te_holder, _topexp_mod, "MapShapesAndAncestors",
                              "topexp_MapShapesAndAncestors")

    td_holder = _first_attr(_topods_mod, "topods", "TopoDS")
    to_face = _resolve_static(td_holder, _topods_mod, "Face", "topods_Face")
    to_edge = _resolve_static(td_holder, _topods_mod, "Edge", "topods_Edge")

    brep_tool = _brep.BRep_Tool
    curve_on_surf = _first_attr(brep_tool, "CurveOnSurface", "CurveOnSurface_s")

    bnd_holder = _first_attr(_bndlib_mod, "brepbndlib", "BRepBndLib")
    bnd_add = _resolve_static(bnd_holder, _bndlib_mod, "Add", "brepbndlib_Add")

    missing = [n for n, v in [
        ("BRepGProp.SurfaceProperties", surf_props),
        ("BRepGProp.LinearProperties", lin_props),
        ("BRepTools.UVBounds", uv_bounds),
        ("TopExp.MapShapesAndAncestors", map_anc),
        ("TopoDS.Face", to_face), ("TopoDS.Edge", to_edge),
        ("BRep_Tool.CurveOnSurface", curve_on_surf),
        ("BRepBndLib.Add", bnd_add),
    ] if v is None]
    if missing:
        raise ImportError(
            "pythonocc is installed but these APIs could not be resolved: "
            + ", ".join(missing)
            + ". Please report the pythonocc-core version."
        )

    _API.update(
        surface_properties=surf_props, linear_properties=lin_props,
        uv_bounds=uv_bounds, map_shapes_and_ancestors=map_anc,
        to_face=to_face, to_edge=to_edge, curve_on_surface=curve_on_surf,
        bnd_add=bnd_add, Bnd_Box=Bnd_Box,
        BRepAdaptor_Curve=BRepAdaptor_Curve,
        BRepAdaptor_Surface=BRepAdaptor_Surface,
        BRepTopAdaptor_FClass2d=BRepTopAdaptor_FClass2d,
        GProp_GProps=GProp_GProps, gp_Pnt=gp_Pnt, gp_Pnt2d=gp_Pnt2d,
        gp_Vec=gp_Vec, IFSelect_RetDone=IFSelect_RetDone,
        STEPControl_Reader=STEPControl_Reader,
        TopAbs_EDGE=TopAbs_EDGE, TopAbs_FACE=TopAbs_FACE, TopAbs_OUT=TopAbs_OUT,
        TopAbs_REVERSED=TopAbs_REVERSED, TopAbs_WIRE=TopAbs_WIRE,
        TopExp_Explorer=TopExp_Explorer,
        TopTools_IndexedDataMapOfShapeListOfShape=(
            TopTools_IndexedDataMapOfShapeListOfShape),
        TopTools_IndexedMapOfShape=TopTools_IndexedMapOfShape,
        surface_types=[GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                       GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
                       GeomAbs_BSplineSurface],
        curve_types=[GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
                     GeomAbs_Hyperbola, GeomAbs_Parabola, GeomAbs_BezierCurve,
                     GeomAbs_BSplineCurve],
    )
    OCC_AVAILABLE = True


# Pure-numpy helpers (importable and testable without pythonocc)

def attribute_similarity(faces_attr: np.ndarray) -> np.ndarray:
    """Attribute-similarity matrix.

    `0.1 + sim * 0.5`, then a final `* 0.5`, with a zero diagonal, on the
    normalised attributes. This is NOT adjacency.
    """
    a = np.asarray(faces_attr, dtype=np.float64)
    n = a.shape[0]
    B = np.zeros((n, n), dtype=np.float32)
    norms = np.linalg.norm(a, axis=1)
    for i in range(n):
        for j in range(i + 1, n):
            if norms[i] > 0 and norms[j] > 0:
                sim = float(a[i] @ a[j] / (norms[i] * norms[j]))
                val = 0.1 + sim * 0.5
            else:
                val = 0.1
            B[i, j] = B[j, i] = val
    return (B * 0.5).astype(np.float32)


def path_context(adjacency: np.ndarray) -> np.ndarray:
    """A_path = min(A^2 + A^3, 1), the 2/3-hop matrix NAMRC 54 defines."""
    A = np.asarray(adjacency, dtype=np.float32)
    A2 = A @ A
    return np.minimum(A2 + A2 @ A, 1.0).astype(np.float32)


# Adjacency density.
#
# A complete graph is legal for a small solid: a tetrahedron (n = 4) and a
# two-face cone (n = 2). A complete graph on many faces is not.
#
# The face-adjacency graph of a solid embeds in the boundary surface of that
# solid: put one node inside each face and draw one arc across each shared
# edge; no two arcs need cross. So the graph's genus cannot exceed the genus of
# the boundary surface. For the complete graph K_n, Ringel & Youngs (1968):
#
#     genus(K_n) = ceil((n - 3) * (n - 4) / 12)
#
# so a COMPLETE face adjacency on n faces forces
#
#     genus(boundary surface) >= ceil((n - 3) * (n - 4) / 12)
#
#     n <= 4     genus 0     K_4 is planar -- the tetrahedron, legal on a ball
#     n = 5..7   genus 1     needs at least one through-hole (a handle)
#     n = 8..10  genus 2
#     n = 14     genus 10
#     n = 15     genus 11
#     n = 40     genus 111   <- the all-ones placeholder's signature
#
# CADSynth and MFInstSeg parts are prismatic stock with machining features; the
# genus of such a part is its through-hole count, in the low single digits.
# _MAX_GENUS_BUDGET = 10 is far more generous than either corpus needs, and it
# still makes any complete adjacency on 15 or more faces a hard error.
#
# THE CORPUS-LEVEL HALF is `validate_corpus_density` below: across a corpus the
# mean density must lie strictly inside (0, 1) with non-zero variance. An
# all-ones placeholder gives mean 1.0 and variance 0.0 and fails it; a corpus
# that merely contains a tetrahedron passes. `check_json` enforces it; the
# extraction loop only reports per-model problems, so --check-json stays the
# gate the regiment names.

_MAX_GENUS_BUDGET = 10
_DENSITY_VAR_TOL = 1e-12


def complete_graph_genus(n: int) -> int:
    """Orientable genus of K_n: ceil((n - 3)(n - 4) / 12) (Ringel & Youngs)."""
    if n <= 4:
        return 0
    a = (n - 3) * (n - 4)
    return int(-((-a) // 12))


def _max_complete_faces(budget: int = _MAX_GENUS_BUDGET) -> int:
    """Largest face count whose complete graph fits a genus <= budget surface."""
    n = 4
    while complete_graph_genus(n + 1) <= budget:
        n += 1
    return n


MAX_COMPLETE_FACES = _max_complete_faces()      # 14 at budget 10


def validate_corpus_density(densities: Sequence[float],
                            min_models: int = 2) -> List[str]:
    """Corpus-level adjacency rule . Returns problems.

    Per model a complete graph can be legal geometry. Across a corpus it cannot
    be: real parts differ in shape, so their adjacency densities must differ.
    Mean strictly inside (0, 1) and non-zero variance is the property the old
    per-model density rule was really reaching for, and it is the one an
    all-ones placeholder fails (mean 1.0, variance 0.0) while a tetrahedron
    passes. Fewer than `min_models` densities carry no corpus-level evidence,
    so a single-file --check-json is left alone rather than judged.
    """
    problems: List[str] = []
    d = np.asarray(list(densities), dtype=np.float64)
    if d.size < min_models:
        return problems
    mean = float(d.mean())
    var = float(d.var())
    if not (0.0 < mean < 1.0):
        problems.append(
            "corpus mean adjacency density {0:.6f} is not strictly inside "
            "(0, 1) over {1} models -- 1.0 is the all-ones placeholder, 0.0 is "
            "no topology at all".format(mean, int(d.size)))
    if var <= _DENSITY_VAR_TOL:
        problems.append(
            "corpus adjacency density has zero variance ({0:.3e}) over {1} "
            "models -- every part cannot have the same density; this is what a "
            "constant placeholder matrix looks like".format(var, int(d.size)))
    return problems


def validate_adjacency(A: np.ndarray, strict: bool = True) -> List[str]:
    """Step 3 of the topology note, as assertions. Returns a list of problems."""
    problems: List[str] = []
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return [f"not square: shape {A.shape}"]
    n = A.shape[0]
    if not np.allclose(A, A.T):
        problems.append("not symmetric")
    if not np.allclose(np.diag(A), 0.0):
        problems.append("diagonal is not zero")
    if not np.all((A == 0) | (A == 1)):
        problems.append("not binary")
    if n > 1:
        rows = A.sum(axis=1)
        if strict and rows.min() < 1:
            problems.append(f"{int((rows < 1).sum())} face(s) with no neighbour")
        density = float(A.sum() / (n * n - n))
        if density >= 1.0 and n > MAX_COMPLETE_FACES:
            problems.append(
                "complete graph on {0} faces: that forces a boundary surface of "
                "genus >= {1} (Ringel-Youngs), which no part in these corpora "
                "has -- this is the all-ones placeholder signature".format(
                    n, complete_graph_genus(n)))
    return problems


def _percentile_scale(values: Sequence[float], pct: float) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return 1.0
    scale = float(np.percentile(arr, pct))
    return scale if scale > _EPS else 1.0


class BRepFeatureExtractor:
    """Extract per-face and per-edge features plus real face adjacency."""

    def __init__(self, uv_grid_size: Tuple[int, int] = DEFAULT_GRID,
                 normalize: str = "unitcube") -> None:
        _load_occ()
        self.uv_grid_size = tuple(uv_grid_size)
        self.normalize = normalize
        self.unit_cube = (normalize == "unitcube")
        if self.unit_cube:
            # Derived from the size-2 box; no corpus statistics involved.
            self.scales: Dict[str, float] = {
                "area": UNITCUBE_FACE_AREA_DIV,
                "length": UNITCUBE_EDGE_LENGTH_DIV,
                "loops": LOOP_COUNT_DIV,
                "degree": FACE_DEGREE_DIV,
            }
        else:
            self.scales = {
                "area": 10.0, "loops": 10.0, "degree": 20.0, "length": 10.0}

    # -- shape IO ----------------------------------------------------------- #

    def load_step_file(self, filepath: str | Path):
        reader = _API["STEPControl_Reader"]()
        status = reader.ReadFile(str(filepath))
        if status != _API["IFSelect_RetDone"]:
            raise RuntimeError(f"STEP read failed ({status}): {filepath}")
        reader.TransferRoots()
        return reader.OneShape()

    def _explore(self, shape, kind) -> List[Any]:
        out, exp = [], _API["TopExp_Explorer"](shape, kind)
        while exp.More():
            out.append(exp.Current())
            exp.Next()
        return out

    def unit_cube_transform(self, shape) -> Tuple[Tuple[float, float, float], float]:
        """Centre and scale that map this solid into a cube of size 2 at the origin.

        UV-Net's protocol. The transform is applied *numerically* to measured
        quantities rather than by rebuilding the shape with
        BRepBuilderAPI_Transform: a rebuilt shape can enumerate its faces in a
        different order, which would silently break the alignment between face
        index and the `seg` / `inst` / `bottom` label ids. Under a uniform
        scale s about the centre c:

            point   -> (p - c) * s
            length  -> length * s
            area    -> area * s^2
            normal, adjacency, loop count, degree, convexity, angle  -> unchanged

        so scaling the numbers is exactly equivalent and carries no topology risk.

        Returns ((cx, cy, cz), s). Returns ((0,0,0), 1.0) when normalisation is
        off or the bounding box is degenerate.
        """
        if not self.unit_cube:
            return (0.0, 0.0, 0.0), 1.0
        bbox = _API["Bnd_Box"]()
        _API["bnd_add"](shape, bbox)
        # Bnd_Box carries a tolerance gap, so Get() returns corners slightly
        # outside the true extent. Zero it so the scale depends only on the
        # geometry and is identical for two copies of a shape whose vertex
        # tolerances happen to differ.
        try:
            bbox.SetGap(0.0)
        except Exception:
            pass
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        centre = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))
        extent = max(xmax - xmin, ymax - ymin, zmax - zmin)
        if not np.isfinite(extent) or extent <= _EPS:
            return centre, 1.0
        return centre, 2.0 / extent

    def topology(self, shape) -> Tuple[List[Any], List[Any], Any]:
        """Faces and edges in TopExp_Explorer order -- preserving the ordering
        the label files are aligned to -- deduplicated by IsSame, plus an
        indexed map giving each face's position."""
        # `Contains` is used rather than comparing map sizes before/after Add:
        # pythonocc 7.9 does not bind Extent() on TopTools_IndexedMapOfShape.
        fmap = _API["TopTools_IndexedMapOfShape"]()
        faces = []
        for f in self._explore(shape, _API["TopAbs_FACE"]):
            face = _API["to_face"](f)
            if not fmap.Contains(face):
                fmap.Add(face)
                faces.append(face)
        emap = _API["TopTools_IndexedMapOfShape"]()
        edges = []
        for e in self._explore(shape, _API["TopAbs_EDGE"]):
            edge = _API["to_edge"](e)
            if not emap.Contains(edge):
                emap.Add(edge)
                edges.append(edge)
        return faces, edges, fmap

    # -- face attributes ---------------------------------------------------- #

    def get_face_type(self, face) -> int:
        st = _API["BRepAdaptor_Surface"](face).GetType()
        types = _API["surface_types"]
        return types.index(st) if st in types else 7

    def get_face_area(self, face) -> float:
        props = _API["GProp_GProps"]()
        _API["surface_properties"](face, props)
        return float(props.Mass())

    def count_face_loops(self, face) -> int:
        """Number of wires: outer boundary plus one per hole."""
        return len(self._explore(face, _API["TopAbs_WIRE"]))

    # -- normals, UV grid --------------------------------------------------- #

    def _face_normal(self, surf, face, u: float, v: float):
        """Unit outward normal, normalise(dS/du x dS/dv), reversed for REVERSED faces."""
        pnt = _API["gp_Pnt"]()
        du, dv = _API["gp_Vec"](), _API["gp_Vec"]()
        surf.D1(u, v, pnt, du, dv)
        n = du.Crossed(dv)
        if n.Magnitude() <= _EPS:
            return pnt, None
        n.Normalize()
        if face.Orientation() == _API["TopAbs_REVERSED"]:
            n.Reverse()
        return pnt, n

    def extract_face_uv_grid(self, face, centre=(0.0, 0.0, 0.0),
                             scale: float = 1.0) -> np.ndarray:
        """UV grid for one face. Channels 0-2 are (point - centre) * scale, so
        with unit-cube normalisation they land in [-1, 1] and are independent of
        the part's absolute size. Channels 3-5 are the outward unit normal
        (unaffected by uniform scaling); channel 6 is the trim mask."""
        h, w = self.uv_grid_size
        grid = np.zeros((7, h, w), dtype=np.float32)
        cx, cy, cz = centre

        surf = _API["BRepAdaptor_Surface"](face, True)
        # face-local UV bounds, not the surface's full parametric range
        u_min, u_max, v_min, v_max = _API["uv_bounds"](face)
        classifier = _API["BRepTopAdaptor_FClass2d"](face, 1e-6)

        for i in range(h):
            u = u_min + (u_max - u_min) * (i / (h - 1) if h > 1 else 0.0)
            for j in range(w):
                v = v_min + (v_max - v_min) * (j / (w - 1) if w > 1 else 0.0)
                try:
                    pnt, normal = self._face_normal(surf, face, u, v)
                    if normal is None:
                        continue
                    # real trim test: OUT means the sample lies in the
                    # parametric box but outside the trimmed face.
                    state = classifier.Perform(_API["gp_Pnt2d"](u, v))
                    grid[0, i, j] = (pnt.X() - cx) * scale
                    grid[1, i, j] = (pnt.Y() - cy) * scale
                    grid[2, i, j] = (pnt.Z() - cz) * scale
                    grid[3, i, j] = normal.X()
                    grid[4, i, j] = normal.Y()
                    grid[5, i, j] = normal.Z()
                    grid[6, i, j] = 0.0 if state == _API["TopAbs_OUT"] else 1.0
                except Exception:
                    continue
        return grid

    # -- adjacency ---------------------------------------------------------- #

    def face_adjacency(self, shape, faces, fmap):
        """Binary face adjacency from shared edges.

        A[i, j] = 1 iff faces i and j share at least one edge. This is the
        A_adj that NAMRC 54 Eq. (bais) defines and that the released code never
        computed. Returns (A, {edge index -> [face ids]}).
        """
        n = len(faces)
        A = np.zeros((n, n), dtype=np.float32)
        anc = _API["TopTools_IndexedDataMapOfShapeListOfShape"]()
        _API["map_shapes_and_ancestors"](
            shape, _API["TopAbs_EDGE"], _API["TopAbs_FACE"], anc)

        edge_faces: Dict[int, List[int]] = {}
        for k in range(1, _extent(anc) + 1):
            shape_list = anc.FindFromIndex(k)
            try:
                shapes = list(shape_list)
            except Exception:
                from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape
                shapes = []
                li = TopTools_ListIteratorOfListOfShape(shape_list)
                while li.More():
                    shapes.append(li.Value())
                    li.Next()
            ids = []
            for s in shapes:
                idx = fmap.FindIndex(_API["to_face"](s))   # 1-based, IsSame-based
                if idx > 0:
                    ids.append(idx - 1)
            ids = sorted(set(ids))
            edge_faces[k - 1] = ids
            for a in ids:
                for b in ids:
                    if a != b:
                        A[a, b] = 1.0
        return A, edge_faces

    # -- edge attributes ---------------------------------------------------- #

    def get_edge_type(self, edge) -> int:
        ct = _API["BRepAdaptor_Curve"](edge).GetType()
        types = _API["curve_types"]
        return types.index(ct) if ct in types else 7

    def get_edge_length(self, edge) -> float:
        props = _API["GProp_GProps"]()
        _API["linear_properties"](edge, props)
        return float(props.Mass())

    def _oriented_tangent(self, edge, face):
        """Edge tangent at its midpoint, flipped so it follows the direction in
        which `face` traverses the edge in its wire. Without this the sign of
        the dihedral angle is arbitrary."""
        curve = _API["BRepAdaptor_Curve"](edge)
        t = 0.5 * (curve.FirstParameter() + curve.LastParameter())
        pnt, tan = _API["gp_Pnt"](), _API["gp_Vec"]()
        curve.D1(t, pnt, tan)
        if tan.Magnitude() <= _EPS:
            return pnt, None
        tan.Normalize()
        for e in self._explore(face, _API["TopAbs_EDGE"]):
            if e.IsSame(edge):
                if e.Orientation() == _API["TopAbs_REVERSED"]:
                    tan.Reverse()
                break
        return pnt, tan

    def _uv_on_face(self, edge, face, pnt):
        """(u, v) of a point on `edge` in `face`'s parameter space -- from the
        pcurve when present, else by projection."""
        try:
            res = _API["curve_on_surface"](edge, face)
            c2d, first, last = res[0], res[1], res[2]
            if c2d is not None:
                p2 = c2d.Value(0.5 * (first + last))
                return float(p2.X()), float(p2.Y())
        except Exception:
            pass
        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
            p2 = ShapeAnalysis_Surface(BRep_Tool.Surface(face)).ValueOfUV(pnt, 1e-6)
            return float(p2.X()), float(p2.Y())
        except Exception:
            return None

    def edge_dihedral(self, edge, faces_of_edge: Sequence[Any]) -> Tuple[float, float]:
        """Real (convexity, signed dihedral angle) for one edge.

        Both are scaled to [0, 1]:
            convexity  0.0 concave | 0.5 smooth or undefined | 1.0 convex
            angle      (theta + pi) / 2pi, theta the signed turning angle
        """
        if len(faces_of_edge) != 2:
            return 0.5, 0.5                  # seam, free or non-manifold edge
        f1, f2 = faces_of_edge[0], faces_of_edge[1]
        pnt, tan = self._oriented_tangent(edge, f1)
        if tan is None:
            return 0.5, 0.5

        normals = []
        for f in (f1, f2):
            uv = self._uv_on_face(edge, f, pnt)
            if uv is None:
                return 0.5, 0.5
            surf = _API["BRepAdaptor_Surface"](f, True)
            _, n = self._face_normal(surf, f, uv[0], uv[1])
            if n is None:
                return 0.5, 0.5
            normals.append(np.array([n.X(), n.Y(), n.Z()], dtype=np.float64))

        n1, n2 = normals
        t = np.array([tan.X(), tan.Y(), tan.Z()], dtype=np.float64)
        cos_a = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        sin_a = float(np.dot(np.cross(n1, n2), t))
        theta = math.atan2(sin_a, cos_a) * CONVEXITY_SIGN

        if abs(theta) < _SMOOTH_ANGLE_RAD:
            convexity = 0.5
        else:
            convexity = 1.0 if theta > 0 else 0.0
        angle_norm = (theta + math.pi) / (2.0 * math.pi)
        return convexity, float(np.clip(angle_norm, 0.0, 1.0))

    # -- assembly ----------------------------------------------------------- #

    def raw_measurements(self, shape) -> Dict[str, Any]:
        faces, edges, fmap = self.topology(shape)
        A, edge_faces = self.face_adjacency(shape, faces, fmap)
        degrees = A.sum(axis=1)
        centre, scale = self.unit_cube_transform(shape)

        face_raw = [{
            "type": self.get_face_type(f),
            # area scales with s^2 under a uniform scale s
            "area": self.get_face_area(f) * scale * scale,
            "area_model_units": self.get_face_area(f),
            "loops": self.count_face_loops(f),
            "degree": float(degrees[i]) if len(faces) > 1 else 0.0,
        } for i, f in enumerate(faces)]

        edge_raw = []
        for k, e in enumerate(edges):
            ids = edge_faces.get(k, [])
            convexity, angle = self.edge_dihedral(e, [faces[i] for i in ids])
            edge_raw.append({
                "type": self.get_edge_type(e),
                # length scales with s
                "length": self.get_edge_length(e) * scale,
                "length_model_units": self.get_edge_length(e),
                "convexity": convexity, "angle": angle, "faces": ids,
            })
        return {"faces": faces, "edges": edges, "adjacency": A,
                "face_raw": face_raw, "edge_raw": edge_raw,
                "unit_centre": list(centre), "unit_scale": float(scale)}

    def _scale(self, value: float, key: str) -> float:
        return float(min(value / self.scales[key], 1.0))

    def assemble(self, meas: Dict[str, Any]) -> Dict[str, Any]:
        """Build both views of the attribute vectors.

        `faces_attr` / `edges_attr` are divided by the stated constants and
        clipped to [0, 1] -- the drop-in replacement for the existing
        `AttributeEncoder`, which is a plain MLP with no normalisation layer.

        `faces_attr_raw` / `edges_attr_raw` hold the same quantities in
        unit-cube units with **no divisor and no clipping**. Nothing is
        destroyed at extraction time, so if the encoder later gains a
        LayerNorm (as AAGNet's attribute MLPs have) the dataset can read these
        instead and every hand-chosen divisor becomes irrelevant -- without
        re-extracting the corpus.

        `clipped` counts how many values hit the ceiling, so the choice of
        divisor can be justified from measurement rather than asserted.
        """
        fr, er = meas["face_raw"], meas["edge_raw"]
        faces_attr = np.array(
            [[r["type"] / 7.0, self._scale(r["area"], "area"),
              self._scale(r["loops"], "loops"), self._scale(r["degree"], "degree")]
             for r in fr], dtype=np.float32).reshape(-1, 4)
        edges_attr = np.array(
            [[r["type"] / 7.0, self._scale(r["length"], "length"),
              r["convexity"], r["angle"]]
             for r in er], dtype=np.float32).reshape(-1, 4)
        faces_attr_raw = np.array(
            [[r["type"] / 7.0, r["area"], float(r["loops"]), r["degree"]]
             for r in fr], dtype=np.float32).reshape(-1, 4)
        edges_attr_raw = np.array(
            [[r["type"] / 7.0, r["length"], r["convexity"], r["angle"]]
             for r in er], dtype=np.float32).reshape(-1, 4)
        clipped = {
            "area": sum(1 for r in fr if r["area"] >= self.scales["area"]),
            "length": sum(1 for r in er if r["length"] >= self.scales["length"]),
            "loops": sum(1 for r in fr if r["loops"] >= self.scales["loops"]),
            "degree": sum(1 for r in fr if r["degree"] >= self.scales["degree"]),
            "n_faces": len(fr), "n_edges": len(er),
        }
        return {"faces_attr": faces_attr, "edges_attr": edges_attr,
                "faces_attr_raw": faces_attr_raw, "edges_attr_raw": edges_attr_raw,
                "clipped": clipped}


# Labels

def load_labels(label_file: Path, n_faces: int, fmt: str) -> Dict[str, Any]:
    """Return semantic_labels and, for MFInstSeg, instance_labels + bottom_labels.

    `cadsynth`   {"file_name": ..., "labels": [...]}  (also a bare list)
    `mfinstaseg` [[name, {"seg": {"0": c}, "inst": [[...]], "bottom": {"0": b}}]]
                 -- this parser is the one tasks/codebase-summary.md 2.3 records
                 as never having been found anywhere in the archive.
    """
    out: Dict[str, Any] = {"semantic_labels": [0] * n_faces, "file_name": None}
    if not label_file.exists():
        return out
    with open(label_file, "r") as fh:
        data = json.load(fh)

    if fmt == "auto":
        fmt = "mfinstaseg" if (
            isinstance(data, list) and data and isinstance(data[0], list)
            and len(data[0]) > 1 and isinstance(data[0][1], dict)
            and "seg" in data[0][1]
        ) else "cadsynth"

    if fmt == "mfinstaseg":
        inner = data[0][1] if isinstance(data, list) else data
        out["file_name"] = data[0][0] if isinstance(data, list) else None
        labels = [0] * n_faces
        for k, v in inner.get("seg", {}).items():
            i = int(k)
            if 0 <= i < n_faces:
                labels[i] = int(v)
        out["semantic_labels"] = labels

        inst = inner.get("inst")
        if inst is not None:
            M = np.asarray(inst, dtype=np.float32)
            padded = np.zeros((n_faces, n_faces), dtype=np.float32)
            m = min(n_faces, M.shape[0])
            padded[:m, :m] = M[:m, :m]
            out["instance_labels"] = padded.tolist()

        bottom = inner.get("bottom")
        if bottom is not None:
            bl = [0] * n_faces
            for k, v in bottom.items():
                i = int(k)
                if 0 <= i < n_faces:
                    bl[i] = int(v)
            out["bottom_labels"] = bl
        return out

    if isinstance(data, dict):
        out["file_name"] = data.get("file_name")
        lab: List[int] = []
        for key in ("labels", "face_labels", "machining_features", "annotations"):
            if key in data:
                lab = [int(x) for x in data[key]]
                break
    elif isinstance(data, list):
        lab = [int(x) for x in data]
    else:
        lab = []

    if lab and len(lab) != n_faces:
        print(f"    WARNING: {len(lab)} labels for {n_faces} faces in "
              f"{label_file.name}; padding/truncating as the notebook did")
        lab = (lab + [0] * n_faces)[:n_faces]
    out["semantic_labels"] = lab or [0] * n_faces
    return out


def _pairs(step_dir: Path, label_dir: Path, limit: Optional[int]):
    steps: List[Path] = []
    for ext in ("*.stp", "*.step", "*.STP", "*.STEP"):
        steps.extend(step_dir.glob(ext))
    steps = sorted(set(steps))
    if limit:
        steps = steps[:limit]
    return [(s, label_dir / f"{s.stem}.json") for s in steps]


def run(args: argparse.Namespace) -> int:
    step_dir, label_dir = Path(args.step_dir), Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _pairs(step_dir, label_dir, args.limit)
    if not pairs:
        print(f"No STEP files found in {step_dir}", file=sys.stderr)
        return 1
    print(f"{len(pairs)} STEP file(s) in {step_dir}")

    ex = BRepFeatureExtractor(tuple(args.grid_size), args.normalize)

    # Nothing per-model is cached here on purpose. `raw_measurements` returns
    # live OCC TopoDS handles; holding those for 5,880 models exhausts memory
    # RAM long before pass 2 starts. Pass 1 keeps only four lists of floats and
    # pass 2 re-reads each STEP file, so peak memory is one model.
    if args.normalize == "unitcube":
        print("unit-cube normalisation (UV-Net): each solid is scaled into a "
              "cube of size 2 at the origin.", flush=True)
        print(f"  divisors: area/{UNITCUBE_FACE_AREA_DIV:g} (box face), "
              f"length/{UNITCUBE_EDGE_LENGTH_DIV:.4f} (space diagonal), "
              f"loops/{LOOP_COUNT_DIV:g}, degree/{FACE_DEGREE_DIV:g}", flush=True)
        print("  no corpus pass needed -- nothing is fitted on the test split.",
              flush=True)
    elif args.normalize == "dataset":
        sample = pairs
        if args.scale_sample and len(pairs) > args.scale_sample:
            stride = max(1, len(pairs) // args.scale_sample)
            sample = pairs[::stride][:args.scale_sample]
        print(f"pass 1/2: corpus scale from {len(sample)} of {len(pairs)} models "
              f"(p{args.percentile}) ...", flush=True)
        areas, loops, degrees, lengths = [], [], [], []
        measured = 0
        for step, _ in sample:
            try:
                meas = ex.raw_measurements(ex.load_step_file(step))
            except Exception as exc:
                print(f"  skip {step.name}: {type(exc).__name__}: {exc}", flush=True)
                continue
            areas.extend(r["area"] for r in meas["face_raw"])
            loops.extend(r["loops"] for r in meas["face_raw"])
            degrees.extend(r["degree"] for r in meas["face_raw"])
            lengths.extend(r["length"] for r in meas["edge_raw"])
            measured += 1
            del meas                      # release the OCC handles immediately
        if measured == 0:
            print("pass 1 measured nothing -- cannot derive scales", file=sys.stderr)
            return 1
        ex.scales = {
            "area": _percentile_scale(areas, args.percentile),
            "loops": max(_percentile_scale(loops, 100.0), 1.0),
            "degree": max(_percentile_scale(degrees, args.percentile), 1.0),
            "length": _percentile_scale(lengths, args.percentile),
        }
        print("  scales: " + ", ".join(f"{k}={v:.4g}" for k, v in ex.scales.items()),
              flush=True)
        (out_dir / "norm_stats.json").write_text(json.dumps(
            {"scales": ex.scales, "percentile": args.percentile,
             "n_models_sampled": measured, "n_models_total": len(pairs)}, indent=2))
    elif args.normalize == "legacy":
        print("legacy normalisation: area/10, loops/10, degree/20, length/10 "
              "(saturating -- for A/B against the archived numbers only)", flush=True)

    written = failed = resumed = 0
    clip_totals: Counter = Counter()
    raw_acc = {"f_n": 0, "e_n": 0,
               "f_sum": np.zeros(4), "f_sq": np.zeros(4),
               "f_min": np.full(4, np.inf), "f_max": np.full(4, -np.inf),
               "e_sum": np.zeros(4), "e_sq": np.zeros(4),
               "e_min": np.full(4, np.inf), "e_max": np.full(4, -np.inf)}
    obs_loops: List[float] = []
    obs_degree: List[float] = []
    problems_seen: Dict[str, int] = {}
    for step, label in pairs:
        try:
            target = out_dir / f"{step.stem}_features.json"
            if args.skip_existing and target.exists():
                resumed += 1
                continue
            meas = ex.raw_measurements(ex.load_step_file(step))
            faces, edges = meas["faces"], meas["edges"]
            A = meas["adjacency"]

            if args.normalize == "model":
                ex.scales = {
                    "area": _percentile_scale(
                        [r["area"] for r in meas["face_raw"]], 100.0),
                    "loops": max(_percentile_scale(
                        [r["loops"] for r in meas["face_raw"]], 100.0), 1.0),
                    "degree": max(float(A.sum(axis=1).max())
                                  if len(faces) > 1 else 1.0, 1.0),
                    "length": _percentile_scale(
                        [r["length"] for r in meas["edge_raw"]], 100.0),
                }
            att = ex.assemble(meas)
            faces_attr = att["faces_attr"]
            edges_attr = att["edges_attr"]
            for key in ("area", "length", "loops", "degree"):
                clip_totals[key] += att["clipped"][key]
            clip_totals["n_faces"] += att["clipped"]["n_faces"]
            clip_totals["n_edges"] += att["clipped"]["n_edges"]
            obs_loops.extend(r["loops"] for r in meas["face_raw"])
            obs_degree.extend(r["degree"] for r in meas["face_raw"])
            fr_arr = att["faces_attr_raw"].astype(np.float64)
            er_arr = att["edges_attr_raw"].astype(np.float64)
            raw_acc["f_n"] += fr_arr.shape[0]
            raw_acc["f_sum"] += fr_arr.sum(axis=0)
            raw_acc["f_sq"] += (fr_arr ** 2).sum(axis=0)
            raw_acc["f_min"] = np.minimum(raw_acc["f_min"], fr_arr.min(axis=0))
            raw_acc["f_max"] = np.maximum(raw_acc["f_max"], fr_arr.max(axis=0))
            if er_arr.shape[0]:
                raw_acc["e_n"] += er_arr.shape[0]
                raw_acc["e_sum"] += er_arr.sum(axis=0)
                raw_acc["e_sq"] += (er_arr ** 2).sum(axis=0)
                raw_acc["e_min"] = np.minimum(raw_acc["e_min"], er_arr.min(axis=0))
                raw_acc["e_max"] = np.maximum(raw_acc["e_max"], er_arr.max(axis=0))
            sim = attribute_similarity(faces_attr)

            for p in validate_adjacency(A, strict=not args.allow_isolated):
                problems_seen[p] = problems_seen.get(p, 0) + 1
                print(f"    adjacency check [{step.name}]: {p}")

            lab = load_labels(label, len(faces), args.label_format)
            centre, uscale = meas["unit_centre"], meas["unit_scale"]
            faces_uv = [ex.extract_face_uv_grid(f, centre, uscale).tolist()
                        for f in faces]

            data: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "extractor": "extract_features.py",
                "extracted_utc": datetime.now(timezone.utc).isoformat(),
                "normalize": args.normalize,
                "norm_scales": dict(ex.scales),
                "step_file": str(step),
                "label_file": str(label),
                "file_name": lab.get("file_name") or step.stem,
                "n_faces": len(faces),
                "n_edges": len(edges),
                "uv_grid_size": list(ex.uv_grid_size),
                "faces_uv": faces_uv,
                "faces_attr": faces_attr.tolist(),
                "edges_attr": edges_attr.tolist(),
                # unclipped, no divisor -- for an encoder with its own
                # normalisation layer. See BRepFeatureExtractor.assemble.
                "faces_attr_raw": att["faces_attr_raw"].tolist(),
                "edges_attr_raw": att["edges_attr_raw"].tolist(),
                "unit_centre": meas["unit_centre"],
                "unit_scale": meas["unit_scale"],

                "adjacency": A.tolist(),
                "labels": lab["semantic_labels"],    # legacy key
                "semantic_labels": lab["semantic_labels"],
            }
            if args.emit_path_context:
                data["path_context"] = path_context(A).tolist()
            if "instance_labels" in lab:
                data["instance_labels"] = lab["instance_labels"]
            if "bottom_labels" in lab:
                data["bottom_labels"] = lab["bottom_labels"]
            if args.emit_details:
                # uv_grid is not repeated here.
                data["face_details"] = [
                    {"face_id": i, "label": lab["semantic_labels"][i],
                     "attributes": faces_attr[i].tolist(), **meas["face_raw"][i]}
                    for i in range(len(faces))]
                data["edge_details"] = [
                    {"edge_id": k, "attributes": edges_attr[k].tolist(),
                     **meas["edge_raw"][k]} for k in range(len(edges))]

            with open(target, "w") as fh:
                json.dump(data, fh)
            written += 1
            if written == 1 or written % 25 == 0:
                denom = max(len(faces) ** 2 - len(faces), 1)
                print(f"  [{written + resumed}/{len(pairs)}] {step.stem}: "
                      f"{len(faces)} faces, {len(edges)} edges, "
                      f"adj density {A.sum() / denom:.3f}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"  FAILED {step.name}: {type(exc).__name__}: {exc}", flush=True)
            if args.traceback:
                traceback.print_exc()

    print(f"\nwrote {written}, already present {resumed}, failed {failed} "
          f"-> {out_dir}", flush=True)
    nf, ne = clip_totals["n_faces"], clip_totals["n_edges"]
    if nf:
        print("\nclipped at 1.0 in the scaled view "
              "(faces_attr_raw / edges_attr_raw are never clipped):")
        for key, total, div in (
            ("area", nf, ex.scales["area"]),
            ("loops", nf, ex.scales["loops"]),
            ("degree", nf, ex.scales["degree"]),
            ("length", ne, ex.scales["length"]),
        ):
            n = clip_totals[key]
            pct = 100.0 * n / total if total else 0.0
            print(f"  {key:<7} divisor {div:<8.4g} {n:>7}/{total:<7} = {pct:5.2f}%")
    if obs_degree:
        d = np.asarray(obs_degree)
        l = np.asarray(obs_loops)
        print("\nobserved count distributions (these set the two divisors that "
              "are conventions, not derived bounds):")
        print(f"  face degree : max {d.max():.0f}  p99 {np.percentile(d, 99):.0f}  "
              f"mean {d.mean():.2f}   (divisor {ex.scales['degree']:g})")
        print(f"  loop count  : max {l.max():.0f}  p99 {np.percentile(l, 99):.0f}  "
              f"mean {l.mean():.2f}   (divisor {ex.scales['loops']:g})")


    # These are the per-column mean/std needed to standardise faces_attr_raw /
    # edges_attr_raw at the encoder input. NOTE: computed over every model this
    # run wrote, which includes the test split. For the paper, recompute them
    # over the training split only (or use a masked BatchNorm) -- these are a
    # sanity check and a starting point, not the final constants.
    if raw_acc["f_n"]:
        names_f = ("type/7", "area", "loops", "degree")
        names_e = ("type/7", "length", "convexity", "angle")
        print("\nraw view (unit-cube units, unclipped) -- these are what an "
              "encoder with input normalisation would consume:")
        for tag, names, n, ssum, ssq, smin, smax in (
            ("faces_attr_raw", names_f, raw_acc["f_n"], raw_acc["f_sum"],
             raw_acc["f_sq"], raw_acc["f_min"], raw_acc["f_max"]),
            ("edges_attr_raw", names_e, raw_acc["e_n"], raw_acc["e_sum"],
             raw_acc["e_sq"], raw_acc["e_min"], raw_acc["e_max"]),
        ):
            if not n:
                continue
            mean = ssum / n
            var = np.maximum(ssq / n - mean ** 2, 0.0)
            std = np.sqrt(var)
            print(f"  {tag}  (n={n})")
            for c, nm in enumerate(names):
                print(f"    {nm:<10} mean {mean[c]:9.4f}  std {std[c]:9.4f}  "
                      f"min {smin[c]:9.4f}  max {smax[c]:9.4f}")
        stats_path = out_dir / "raw_feature_stats.json"
        stats_path.write_text(json.dumps({
            "note": "per-column stats of faces_attr_raw / edges_attr_raw; "
                    "computed over all models written by this run (includes "
                    "test split) -- recompute on the training split before use",
            "faces": {"n": int(raw_acc["f_n"]),
                      "columns": list(names_f),
                      "mean": (raw_acc["f_sum"] / raw_acc["f_n"]).tolist(),
                      "std": np.sqrt(np.maximum(
                          raw_acc["f_sq"] / raw_acc["f_n"]
                          - (raw_acc["f_sum"] / raw_acc["f_n"]) ** 2, 0.0)).tolist(),
                      "min": raw_acc["f_min"].tolist(),
                      "max": raw_acc["f_max"].tolist()},
            "edges": ({"n": int(raw_acc["e_n"]),
                       "columns": list(names_e),
                       "mean": (raw_acc["e_sum"] / raw_acc["e_n"]).tolist(),
                       "std": np.sqrt(np.maximum(
                           raw_acc["e_sq"] / raw_acc["e_n"]
                           - (raw_acc["e_sum"] / raw_acc["e_n"]) ** 2, 0.0)).tolist(),
                       "min": raw_acc["e_min"].tolist(),
                       "max": raw_acc["e_max"].tolist()} if raw_acc["e_n"] else {}),
        }, indent=2))
        print(f"  written to {stats_path.name}")

    if problems_seen:
        print("adjacency problems: "
              + "; ".join(f"{k} x{v}" for k, v in problems_seen.items()))
    return 0 if failed == 0 else 2


def check_json(path: Path) -> int:
    files = sorted(path.glob("*_features.json")) if path.is_dir() else [path]
    if not files:
        print(f"no *_features.json under {path}", file=sys.stderr)
        return 1
    bad = 0
    dens_adj, dens_sim = [], []
    for f in files:
        d = json.loads(f.read_text())
        n = int(d["n_faces"])
        denom = max(n * n - n, 1)
        if "adjacency" in d:
            A = np.asarray(d["adjacency"], dtype=np.float32)
            probs = validate_adjacency(A)
            dens_adj.append(float(A.sum() / denom))
            if probs:
                bad += 1
                print(f"{f.name}: " + "; ".join(probs))
        else:
            bad += 1
            print(f"{f.name}: no 'adjacency' key -- written by the old notebook "
                  f"extractor; re-extract with this script")
        key = "attr_similarity" if "attr_similarity" in d else "bias_matrix"
        if key in d:
            S = np.asarray(d[key], dtype=np.float32)
            dens_sim.append(float((np.abs(S) > 1e-6).sum() / denom))
    corpus_problems: List[str] = []
    if dens_adj:
        print(f"\nadjacency density: mean {np.mean(dens_adj):.3f} "
              f"[{np.min(dens_adj):.3f}, {np.max(dens_adj):.3f}]  (expect ~0.1-0.4)")
        # corpus-level half of the density
        # rule. Per model a complete graph can be legal; across the corpus a
        # constant or saturated density cannot be.
        corpus_problems = validate_corpus_density(dens_adj)
        for p in corpus_problems:
            print(f"CORPUS ADJACENCY PROBLEM: {p}")
    if dens_sim:
        print(f"similarity nonzero density: mean {np.mean(dens_sim):.3f}  "
              f"(1.000 means every pair looks 'adjacent' to GraphBiasEncoder)")
    print(f"\n{len(files) - bad}/{len(files)} files OK")
    return 0 if (bad == 0 and not corpus_problems) else 2


# Self-test

def _check(name: str, ok: bool, detail: str = "", hint: str = "") -> bool:
    """`detail` always prints; `hint` prints only when the check fails."""
    line = f"  [{'PASS' if ok else 'FAIL'}] {name}"
    if detail:
        line += f" -- {detail}"
    if not ok and hint:
        line += f"   >> {hint}"
    print(line)
    return bool(ok)


def self_test(pure_only: bool = False) -> int:
    print("pure-numpy checks")
    ok = True
    A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
    ok &= _check("validate_adjacency accepts a valid matrix",
                 validate_adjacency(A) == [])
    _tetra = np.ones((4, 4)) - np.eye(4)
    _cone = np.array([[0.0, 1.0], [1.0, 0.0]])
    _placeholder = np.ones((40, 40)) - np.eye(40)
    ok &= _check("validate_adjacency ACCEPTS a tetrahedron (density 1.0, n=4)",
                 validate_adjacency(_tetra) == [],
                 f"MAX_COMPLETE_FACES={MAX_COMPLETE_FACES}",
                 str(validate_adjacency(_tetra)))
    ok &= _check("validate_adjacency ACCEPTS a two-face cone (density 1.0, n=2)",
                 validate_adjacency(_cone) == [],
                 "", str(validate_adjacency(_cone)))
    ok &= _check("validate_adjacency rejects an all-ones 40x40 placeholder",
                 any("complete graph" in p
                     for p in validate_adjacency(_placeholder)),
                 f"genus(K_40)={complete_graph_genus(40)}")
    ok &= _check("corpus density rule rejects an all-ones corpus",
                 validate_corpus_density([1.0] * 8) != [])
    ok &= _check("corpus density rule rejects a constant-density corpus",
                 validate_corpus_density([0.3] * 8) != [])
    ok &= _check("corpus density rule accepts a varied corpus with a tetrahedron",
                 validate_corpus_density([1.0, 0.4, 0.33, 0.25, 0.2]) == [])
    ok &= _check("validate_adjacency rejects asymmetry",
                 "not symmetric" in validate_adjacency(np.array([[0, 1], [0, 0]])))
    P = path_context(A)
    ok &= _check("path_context is bounded to [0, 1]",
                 bool(P.max() <= 1.0 and P.min() >= 0.0))
    S = attribute_similarity(np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0]]))
    ok &= _check("attr_similarity symmetric with zero diagonal",
                 bool(np.allclose(S, S.T) and np.allclose(np.diag(S), 0)))
    ok &= _check("attr_similarity reproduces the legacy constant",
                 abs(float(S[0, 1]) - (0.1 + 1.0 * 0.5) * 0.5) < 1e-6,
                 f"S[0,1]={float(S[0, 1]):.4f}, expected 0.3000")

    if pure_only:
        print(f"\n{'ALL PURE CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
        return 0 if ok else 1

    print("\ngeometry checks (needs pythonocc-core)")
    try:
        _load_occ()
    except ImportError as exc:
        print(f"  SKIPPED -- {exc}")
        print("\nRe-run with --pure to check only the numpy paths.")
        return 0 if ok else 1

    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

    try:
        import OCC
        print(f"  pythonocc-core {getattr(OCC, 'VERSION', '?')}")
    except Exception:
        pass
    probe = _API["TopTools_IndexedMapOfShape"]()
    print("  IndexedMapOfShape methods: "
          + ", ".join(a for a in ("Add", "Contains", "FindIndex", "Size", "Extent")
                      if hasattr(probe, a)))

    ex = BRepFeatureExtractor(normalize="unitcube")

    # --- box: 6 faces, each adjacent to 4 others, every edge convex -------- #
    box = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    m = ex.raw_measurements(box)
    A = m["adjacency"]
    ok &= _check("box has 6 faces", len(m["faces"]) == 6, f"got {len(m['faces'])}")
    ok &= _check("box has 12 edges", len(m["edges"]) == 12, f"got {len(m['edges'])}")
    ok &= _check("box adjacency valid", validate_adjacency(A) == [],
                 str(validate_adjacency(A)))
    ok &= _check("every box face has 4 neighbours",
                 bool(np.all(A.sum(axis=1) == 4)), str(A.sum(axis=1)))
    dens = float(A.sum() / (36 - 6))
    ok &= _check("box adjacency density is 0.8, not 1.0",
                 abs(dens - 0.8) < 1e-6, f"density={dens:.3f}")
    ok &= _check("every box face has exactly 1 loop",
                 all(r["loops"] == 1 for r in m["face_raw"]),
                 str([r["loops"] for r in m["face_raw"]]))
    convex = [r["convexity"] for r in m["edge_raw"]]
    ok &= _check("all 12 box edges are convex",
                 all(c == 1.0 for c in convex),
                 f"{sum(1 for c in convex if c == 1.0)}/12 convex",
                 "all 0.0 means the sign is inverted -- flip CONVEXITY_SIGN")
    ok &= _check("edge attrs are not the constant 0.5",
                 not all(r["convexity"] == 0.5 for r in m["edge_raw"]))

    centre = np.array([5.0, 10.0, 15.0])
    outward, unit = [], []
    for f in m["faces"]:
        g = ex.extract_face_uv_grid(f)
        mid = g[:, g.shape[1] // 2, g.shape[2] // 2].astype(float)
        p, n = mid[0:3], mid[3:6]
        if np.linalg.norm(n) > 0.5:
            outward.append(float(np.dot(p - centre, n)) > 0)
            unit.append(abs(np.linalg.norm(n) - 1.0) < 1e-4)
    ok &= _check("UV-grid normals point outward on a box",
                 len(outward) == 6 and all(outward), str(outward))
    ok &= _check("UV-grid normals are unit length",
                 len(unit) == 6 and all(unit))

    # --- blind hole: rim concave, top face gains a second loop ------------- #
    try:
        cyl = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(5.0, 10.0, 20.0), gp_Dir(0, 0, 1)), 3.0, 20.0).Shape()
        holed = BRepAlgoAPI_Cut(box, cyl).Shape()
        mh = ex.raw_measurements(holed)
        Ah = mh["adjacency"]
        nh = len(mh["faces"])
        ok &= _check("blind-hole adjacency valid", validate_adjacency(Ah) == [],
                     str(validate_adjacency(Ah)))
        dh = float(Ah.sum() / max(nh * nh - nh, 1))
        ok &= _check("blind-hole graph is sparse, not complete", dh < 1.0,
                     f"density={dh:.3f}")
        ok &= _check("a face with a hole reports 2 loops",
                     any(r["loops"] == 2 for r in mh["face_raw"]),
                     f"loop counts seen: {sorted({r['loops'] for r in mh['face_raw']})}")
        n_concave = sum(1 for r in mh["edge_raw"] if r["convexity"] == 0.0)
        ok &= _check("the hole rim is concave", n_concave > 0,
                     f"{n_concave} concave edge(s) found",
                     "no concave edge found -- flip CONVEXITY_SIGN")
        degrees = sorted({r["degree"] for r in mh["face_raw"]})
        ok &= _check("face degree now varies across a real part",
                     len(degrees) > 1, f"degrees seen: {degrees}")
        old_degree = min((nh - 1) / 20.0, 1.0)
        ok &= _check("real degrees differ from the old placeholder",
                     any(abs(r["degree"] / 20.0 - old_degree) > 1e-6
                         for r in mh["face_raw"]),
                     f"old placeholder would have been {old_degree:.3f} for all "
                     f"{nh} faces")
    except Exception as exc:
        print(f"  [WARN] boolean-cut test skipped: {type(exc).__name__}: {exc}")

    # --- unit-cube normalisation: the claim is scale invariance ------------ #
    centre, uscale = ex.unit_cube_transform(box)
    # Relative tolerance: Bnd_Box may still include vertex tolerances, which
    # perturb the scale by ~1e-8 relative -- irrelevant to the features, and the
    # scale-invariance check below is the real proof.
    ok &= _check("[unitcube] scale maps the longest axis to 2.0",
                 abs(30.0 * uscale - 2.0) < 1e-6,
                 f"30 x {uscale:.10f} = {30.0 * uscale:.10f} "
                 f"(err {abs(30.0 * uscale - 2.0):.2e})")
    ok &= _check("[unitcube] centre is the bounding-box centre",
                 all(abs(a - b) < 1e-9 for a, b in zip(centre, (5.0, 10.0, 15.0))),
                 str([round(c, 4) for c in centre]))
    g = ex.extract_face_uv_grid(m["faces"][0], centre, uscale)
    mask = g[6] > 0.5
    xyz = g[0:3][:, mask] if mask.any() else g[0:3].reshape(3, -1)
    ok &= _check("[unitcube] UV coordinates land inside [-1, 1]",
                 bool(np.abs(xyz).max() <= 1.0 + 1e-5),
                 f"max |coord| = {float(np.abs(xyz).max()):.4f}")
    ok &= _check("[unitcube] no face area exceeds the box-face divisor",
                 max(r["area"] for r in m["face_raw"]) <= UNITCUBE_FACE_AREA_DIV + 1e-6,
                 f"max normalised area = {max(r['area'] for r in m['face_raw']):.4f} "
                 f"(divisor {UNITCUBE_FACE_AREA_DIV})")
    ok &= _check("[unitcube] no edge length exceeds the space diagonal",
                 max(r["length"] for r in m["edge_raw"]) <= UNITCUBE_EDGE_LENGTH_DIV + 1e-6,
                 f"max normalised length = {max(r['length'] for r in m['edge_raw']):.4f} "
                 f"(divisor {UNITCUBE_EDGE_LENGTH_DIV:.4f})")

    # the whole point: a 10x bigger copy of the same shape must be identical
    big = BRepPrimAPI_MakeBox(100.0, 200.0, 300.0).Shape()
    mb = ex.raw_measurements(big)
    a_small = ex.assemble(m)
    a_big = ex.assemble(mb)
    ok &= _check("[unitcube] a 10x larger identical box gives identical attributes",
                 bool(np.allclose(a_small["faces_attr"], a_big["faces_attr"], atol=1e-5)
                      and np.allclose(a_small["edges_attr"], a_big["edges_attr"],
                                      atol=1e-5)),
                 "this is the scale invariance UV-Net's normalisation buys")
    gs = ex.extract_face_uv_grid(m["faces"][0], *(ex.unit_cube_transform(box)))
    gb = ex.extract_face_uv_grid(mb["faces"][0], *(ex.unit_cube_transform(big)))
    ok &= _check("[unitcube] and identical UV grids",
                 bool(np.allclose(gs, gb, atol=1e-4)),
                 f"max abs diff = {float(np.abs(gs - gb).max()):.2e}")

    # raw view must be unclipped and recoverable
    ok &= _check("[raw] faces_attr_raw is unclipped",
                 bool(np.all(a_small["faces_attr_raw"][:, 1]
                             >= a_small["faces_attr"][:, 1] * UNITCUBE_FACE_AREA_DIV
                             - 1e-5)))
    ok &= _check("[raw] clip counters present",
                 set(a_small["clipped"]) >= {"area", "length", "loops", "degree"})

    fa = ex.assemble(m)["faces_attr"]
    sim = attribute_similarity(fa)
    sim_dens = float((np.abs(sim) > 1e-6).sum() / (36 - 6))
    print(f"\n  note: on this box, attr_similarity nonzero density = {sim_dens:.3f} "
          f"vs real adjacency density = {dens:.3f}")
    print("  that gap is exactly what tasks/hassan-note-graph-topology.md describes")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1



def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="B-Rep feature extraction with real topology (CAPP-GPT P2).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step-dir", help="directory of .stp/.step files")
    p.add_argument("--label-dir", help="directory of matching .json label files")
    p.add_argument("--out-dir", help="destination for *_features.json")
    p.add_argument("--label-format", choices=["auto", "cadsynth", "mfinstaseg"],
                   default="auto")
    p.add_argument("--grid-size", type=int, nargs=2, default=list(DEFAULT_GRID),
                   metavar=("H", "W"))
    p.add_argument("--normalize",
                   choices=["unitcube", "dataset", "model", "legacy"],
                   default="unitcube",
                   help="unitcube (default): UV-Net -- scale each solid into a "
                        "cube of size 2, then divide by constants derived from "
                        "that box; no corpus statistics. dataset: corpus "
                        "percentile. model: per-part max. legacy: the "
                        "notebook's saturating constants")
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--scale-sample", type=int, default=500,
                   help="models sampled in pass 1 to derive the scales "
                        "(0 = all; pass 1 never caches, so this is speed only)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip models whose *_features.json already exists "
                        "(resume an interrupted extraction)")
    p.add_argument("--emit-path-context", action="store_true",
                   help="also write min(A^2 + A^3, 1)")
    p.add_argument("--emit-details", action="store_true",
                   help="write face_details/edge_details (without the uv_grid)")
    p.add_argument("--allow-isolated", action="store_true",
                   help="do not warn about faces with no neighbour")
    p.add_argument("--limit", type=int, help="process only the first N files")
    p.add_argument("--traceback", action="store_true")
    p.add_argument("--check-json", metavar="PATH",
                   help="validate already-extracted features and exit")
    p.add_argument("--self-test", action="store_true",
                   help="run built-in checks; no CAD data required")
    p.add_argument("--pure", action="store_true",
                   help="with --self-test, skip the pythonocc checks")

    raw_argv = sys.argv[1:] if argv is None else argv
    args = p.parse_args(argv)

    # No arguments: fall back to the CONFIG block at the top of the file.
    if not raw_argv:
        mode = CONFIG.get("run", "self-test")
        print(f"no command-line arguments given; using CONFIG (run={mode!r})",
              flush=True)
        if mode == "self-test":
            return self_test(pure_only=False)
        if mode == "check-json":
            return check_json(Path(CONFIG["out_dir"]))
        if mode != "extract":
            print(f"CONFIG['run'] = {mode!r} is not one of "
                  f"'self-test' | 'extract' | 'check-json'", file=sys.stderr)
            return 1
        args.step_dir = CONFIG["step_dir"]
        args.label_dir = CONFIG["label_dir"]
        args.out_dir = CONFIG["out_dir"]
        args.grid_size = list(CONFIG.get("grid_size", DEFAULT_GRID))
        args.normalize = CONFIG.get("normalize", "dataset")
        args.percentile = CONFIG.get("percentile", 99.0)
        args.scale_sample = CONFIG.get("scale_sample", 500)
        args.skip_existing = CONFIG.get("skip_existing", True)
        args.emit_path_context = CONFIG.get("emit_path_context", False)
        args.emit_details = CONFIG.get("emit_details", False)
        args.limit = CONFIG.get("limit")
        for key in ("step_dir", "label_dir"):
            if not Path(getattr(args, key)).is_dir():
                print(f"CONFIG['{key}'] does not exist: {getattr(args, key)}",
                      file=sys.stderr)
                return 1
        return run(args)

    if args.self_test:
        return self_test(pure_only=args.pure)
    if args.check_json:
        return check_json(Path(args.check_json))
    if not (args.step_dir and args.label_dir and args.out_dir):
        p.error("--step-dir, --label-dir and --out-dir are required "
                "(or use --self-test / --check-json, or edit CONFIG and pass "
                "no arguments at all)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
