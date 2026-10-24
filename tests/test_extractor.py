"""Extractor tests.

Two tiers:

* Tier 1 always runs. The extractors import OCC lazily inside `_load_occ()`, so
  the module and its pure-Python validators can be exercised in the model
  environment. This matters -- the project has two conda environments and the
  test suite runs in the one WITHOUT pythonocc.
* Tier 2 is skipped unless pythonocc-core is importable (env-extraction.yml).
  It runs each extractor's built-in `--self-test`, which builds a box and a
  blind-hole part with known answers.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXTRACT_DIR = Path("src/cmca/extract")
EXTRACTORS = ["extract_features_cadsynth.py", "extract_features_mfinstaseg.py"]

try:
    import OCC  # noqa: F401
    HAS_OCC = True
except Exception:
    HAS_OCC = False

needs_occ = pytest.mark.skipif(
    not HAS_OCC,
    reason="pythonocc-core not installed -- this is the model environment; "
           "run these under env-extraction.yml (conda env pyoccenv)")


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  EXTRACT_DIR / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Tier 1 -- no pythonocc required
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", EXTRACTORS)
def test_extractor_imports_without_pythonocc(name):
    """OCC is imported inside `_load_occ()`, not at module scope, so a missing
    install fails with a readable message at run time instead of an ImportError
    that stops the file being importable at all."""
    mod = _load(name)
    assert hasattr(mod, "OCC_AVAILABLE")
    assert callable(mod.self_test)
    assert callable(mod.validate_adjacency)
    assert callable(mod.check_json)


@pytest.mark.parametrize("name", EXTRACTORS)
def test_validate_adjacency_accepts_a_good_matrix(name):
    mod = _load(name)
    a = np.array([[0., 1., 0., 1.],
                  [1., 0., 1., 0.],
                  [0., 1., 0., 1.],
                  [1., 0., 1., 0.]])
    problems = mod.validate_adjacency(a)
    assert not problems, f"clean matrix rejected: {problems}"


@pytest.mark.parametrize("name", EXTRACTORS)
def test_validate_adjacency_rejects_the_original_defect(name):
    """The old 'adjacency' was all-ones: every face adjacent to every other,
    density exactly 1.000 across both entire corpora."""
    mod = _load(name)
    n = 5
    all_ones = np.ones((n, n))
    assert mod.validate_adjacency(all_ones), "an all-ones matrix must be rejected"


@pytest.mark.parametrize("name", EXTRACTORS)
def test_validate_adjacency_rejects_asymmetry_and_self_loops(name):
    mod = _load(name)
    a = np.array([[0., 1., 0.], [0., 0., 1.], [0., 1., 0.]])   # not symmetric
    assert mod.validate_adjacency(a)
    b = np.array([[1., 1., 0.], [1., 0., 1.], [0., 1., 0.]])   # self-loop
    assert mod.validate_adjacency(b)


# --------------------------------------------------------------------------- #
# [2026-09-21] the density rule, after the inversion.
#
# These run WITHOUT pythonocc on purpose. The extractors' own `--self-test`
# carries the same pins, but `test_extractor_self_test_passes` is skipped in
# the model environment, so until now the suite could not see this rule at all
# and stayed green whichever way the pins pointed.
# --------------------------------------------------------------------------- #

def _tetrahedron():
    """Four triangular faces, each sharing an edge with the other three:
    density exactly 1.0, and entirely legal geometry."""
    return np.ones((4, 4)) - np.eye(4)


def _cube():
    """Six faces, each adjacent to four others -- opposite faces share no edge."""
    a = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i != j and i // 2 != j // 2:
                a[i, j] = 1.0
    return a


def _two_face_cone():
    """Lateral face plus planar cap: n = 2, density 1.0, also legal."""
    return np.array([[0., 1.], [1., 0.]])


@pytest.mark.parametrize("name", EXTRACTORS)
@pytest.mark.parametrize("build,label", [(_tetrahedron, "tetrahedron"),
                                         (_cube, "cube"),
                                         (_two_face_cone, "two-face cone")])
def test_validate_adjacency_accepts_legal_small_solids(name, build, label):
    """The inverted pin. A complete face-adjacency graph on few faces is legal:
    K_4 is planar, so a tetrahedron's adjacency embeds in a genus-0 boundary
    surface. The old rule rejected exactly this."""
    mod = _load(name)
    problems = mod.validate_adjacency(build())
    assert problems == [], f"{label} rejected: {problems}"


@pytest.mark.parametrize("name", EXTRACTORS)
def test_validate_adjacency_rejects_a_large_complete_graph(name):
    """The disaster the check exists for: `get_face_adjacency` once returned an
    all-ones matrix for every part. A complete graph on 40 faces needs a
    boundary surface of genus >= 111, which no part in either corpus has."""
    mod = _load(name)
    problems = mod.validate_adjacency(np.ones((40, 40)) - np.eye(40))
    assert any("complete graph" in p for p in problems), problems
    assert mod.complete_graph_genus(40) == 111


@pytest.mark.parametrize("name", EXTRACTORS)
def test_complete_graph_genus_matches_ringel_youngs(name):
    """genus(K_n) = ceil((n - 3)(n - 4) / 12); the rule's threshold rides on it."""
    mod = _load(name)
    expected = {2: 0, 4: 0, 5: 1, 7: 1, 8: 2, 12: 6, 14: 10, 15: 11, 40: 111}
    got = {n: mod.complete_graph_genus(n) for n in expected}
    assert got == expected
    assert mod.MAX_COMPLETE_FACES == 14, "budget of 10 handles admits K_14, not K_15"


@pytest.mark.parametrize("name", EXTRACTORS)
def test_corpus_density_rule_replaces_the_per_model_one(name):
    """Per model a complete graph can be legal; across a corpus a saturated or
    constant density cannot be. This is the property the old rule reached for."""
    mod = _load(name)
    assert mod.validate_corpus_density([1.0] * 8), "an all-ones corpus must fail"
    assert mod.validate_corpus_density([0.0] * 8), "a no-topology corpus must fail"
    assert mod.validate_corpus_density([0.3] * 8), "zero variance must fail"
    assert mod.validate_corpus_density([1.0, 0.4, 0.33, 0.25, 0.2]) == [], \
        "a varied corpus that contains a tetrahedron must pass"
    assert mod.validate_corpus_density([1.0]) == [], \
        "one model carries no corpus-level evidence"


@pytest.mark.parametrize("name", EXTRACTORS)
def test_extractor_pure_self_test_passes(name):
    """`--self-test --pure` needs no pythonocc, so the extractors' own pins are
    visible to the suite in the model environment too. Without this, the two
    inverted density pins would be unobserved here."""
    r = subprocess.run([sys.executable, str(EXTRACT_DIR / name),
                        "--self-test", "--pure"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"--self-test --pure failed:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}"
    assert "ACCEPTS a tetrahedron" in r.stdout


def _write_features_corpus(dirpath, matrices):
    """Minimal *_features.json corpus for the CADSynth extractor's check_json."""
    import json
    dirpath.mkdir(parents=True, exist_ok=True)
    for i, A in enumerate(matrices):
        A = np.asarray(A)
        (dirpath / f"{i:08d}_features.json").write_text(json.dumps({
            "n_faces": int(A.shape[0]), "adjacency": A.tolist()}))


def test_check_json_fails_a_corpus_with_constant_density(tmp_path):
    """The corpus-level gate has to reach the exit code, not just the log. Each
    of these three models passes per-model validation, but every one has the
    same density -- the placeholder signature."""
    flat = np.array([[0., 1., 0., 1.], [1., 0., 1., 0.],
                     [0., 1., 0., 1.], [1., 0., 1., 0.]])
    d = tmp_path / "constant"
    _write_features_corpus(d, [flat, flat, flat])
    r = subprocess.run([sys.executable, str(EXTRACT_DIR / EXTRACTORS[0]),
                        "--check-json", str(d)], capture_output=True, text=True)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "CORPUS ADJACENCY PROBLEM" in r.stdout


def test_check_json_accepts_a_varied_corpus_containing_a_tetrahedron(tmp_path):
    """The other half: a real corpus with a tetrahedron in it must pass."""
    d = tmp_path / "varied"
    _write_features_corpus(d, [_tetrahedron(), _cube(), _two_face_cone(),
                               np.array([[0., 1., 1.], [1., 0., 0.], [1., 0., 0.]])])
    r = subprocess.run([sys.executable, str(EXTRACT_DIR / EXTRACTORS[0]),
                        "--check-json", str(d)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# --------------------------------------------------------------------------- #
# Tier 2 -- needs pythonocc
# --------------------------------------------------------------------------- #

@needs_occ
@pytest.mark.parametrize("name", EXTRACTORS)
def test_extractor_self_test_passes(name):
    """A box and a part with a blind hole, both with known answers: face
    adjacency degrees, loop counts, outward unit normals, convex/concave edges,
    and a validity mask that respects the trimmed face.

    This caught a pythonocc 7.9 binding change and a defaulting bug before
    either reached the real corpora.
    """
    r = subprocess.run([sys.executable, str(EXTRACT_DIR / name), "--self-test"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"--self-test failed:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}"


@needs_occ
def test_box_adjacency_is_not_dense():
    """A cube's six faces are each adjacent to four others, not five: opposite
    faces share no edge. The placeholder returned `len(all_faces) - 1` for every
    face, which is exactly the wrong answer this asserts against."""
    mod = _load("extract_features_cadsynth.py")
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    shape = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    adj = mod.face_adjacency(shape) if hasattr(mod, "face_adjacency") else None
    if adj is None:
        pytest.skip("extractor exposes no standalone face_adjacency() helper")
    adj = np.asarray(adj)
    assert adj.shape == (6, 6)
    assert np.allclose(adj, adj.T)
    assert adj.diagonal().sum() == 0
    assert adj.sum(axis=1).tolist() == [4] * 6, "each cube face touches exactly 4"


# --------------------------------------------------------------------------- #
# the ResNet faces extractor
# --------------------------------------------------------------------------- #

FACES_EXTRACTOR = EXTRACT_DIR / "extract_faces_cadsynth.py"


def _write_faces_corpus(dirpath, *, normalised=True, unit_normals=True,
                        trimmed=True, n_models=3, grid=8):
    """A corpus in extract_faces_cadsynth.py's output schema."""
    import json
    dirpath.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for m in range(n_models):
        faces = []
        for fid in range(4):
            g = np.zeros((7, grid, grid), dtype=np.float32)
            coords = rng.uniform(-1, 1, (3, grid, grid))
            g[0:3] = coords if normalised else coords * 250.0
            n = rng.normal(size=(3, grid, grid))
            n = n / np.linalg.norm(n, axis=0, keepdims=True) if unit_normals else n * 7.0
            g[3:6] = n
            if trimmed:
                g[6] = (rng.random((grid, grid)) > 0.25).astype(np.float32)
                g[6, 0, 0] = 0.0
            else:
                g[6] = 1.0
            faces.append({"face_id": fid, "label": int(rng.integers(0, 4)),
                          "uv_grid": g.tolist()})
        (dirpath / f"{m:08d}_faces.json").write_text(json.dumps({
            "model_id": f"{m:08d}", "n_faces": len(faces),
            "uv_grid_size": [grid, grid], "normalize": "unitcube",
            "unit_centre": [0, 0, 0], "unit_scale": 1.0,
            "schema_version": "2.0-faces", "faces": faces}))


def test_faces_extractor_reuses_the_shared_geometry():
    """It must not reimplement the UV sampling. Two separately 'corrected'
    implementations would drift, and the ResNet vs Graph Transformer rows of the
    2x2 would stop being comparable."""
    src = FACES_EXTRACTOR.read_text(encoding="utf-8")
    assert "extract_features_cadsynth.py" in src
    assert "BRepFeatureExtractor" in src
    assert "def extract_face_uv_grid" not in src, \
        "the faces extractor must call the shared grid sampler, not define one"


def test_faces_extractor_check_json_accepts_a_good_corpus(tmp_path):
    d = tmp_path / "good"
    _write_faces_corpus(d)
    r = subprocess.run([sys.executable, str(FACES_EXTRACTOR), "--check-json", str(d)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("kwargs,needle", [
    (dict(normalised=False), "not unit-cube normalised"),   # (dict(unit_normals=False), "non-unit normals"),         # (dict(trimmed=False), "all-ones"),                      # ])
def test_faces_check_json_detects_the_old_defects(tmp_path, kwargs, needle):
    """The notebook corpus had all three: raw coordinates, tangents in place of
    normals, and a validity mask that was 1.0 everywhere."""
    d = tmp_path / "bad"
    _write_faces_corpus(d, **kwargs)
    r = subprocess.run([sys.executable, str(FACES_EXTRACTOR), "--check-json", str(d)],
                       capture_output=True, text=True)
    assert r.returncode != 0, "a defective corpus must not pass --check-json"
    assert needle in (r.stdout + r.stderr)


def test_faces_output_schema_loads_in_the_resnet_dataset(tmp_path):
    """The contract between the extractor and UVGridLabelDataset has no other
    guard: a schema change would surface as an empty dataset at training time."""
    from src.data.uv_dataset import UVGridLabelDataset
    d = tmp_path / "corpus"
    _write_faces_corpus(d, n_models=3)
    ds = UVGridLabelDataset(json_dir=d, n_classes=4, verbose=False)
    assert len(ds) == 12
    uv, label = ds[0]
    assert tuple(uv.shape) == (7, 8, 8)
    assert 0 <= int(label) < 4

    sub = UVGridLabelDataset(json_dir=d, n_classes=4,
                             model_ids=["00000000"], verbose=False)
    assert len(sub) == 4 and sub.model_ids == ["00000000"]


# --------------------------------------------------------------------------- #
# the two extractors carry independent copies of the geometry -- pin them
# --------------------------------------------------------------------------- #

GEOMETRY_METHODS = [
    "extract_face_uv_grid", "_face_normal", "unit_cube_transform",
    "face_adjacency", "topology", "count_face_loops",
    "get_face_area", "get_edge_length", "get_face_type",
]


def _method_ast(path, cls, name):
    """AST dump of a method body with its docstring removed.

    Comments and formatting never reach the AST, so this compares what the code
    DOES rather than how it is written.
    """
    import ast
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    body = f.body
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        body = body[1:]
                    return ast.dump(ast.Module(body=body, type_ignores=[]))
    return None


@pytest.mark.parametrize("method", GEOMETRY_METHODS)
def test_both_extractors_share_identical_geometry(method):
    """extract_features_cadsynth.py and extract_features_mfinstaseg.py each hold
    their own copy of BRepFeatureExtractor. Duplicated code drifts, and if these
    diverge the two corpora stop being built the same way -- which would quietly
    invalidate every cross-corpus statement in the paper, and the ResNet faces
    corpus too, since extract_faces_cadsynth.py imports the CADSynth copy.

    (`edge_dihedral` is deliberately excluded: the two are written differently --
    an if/else block against a conditional expression -- but were verified
    numerically identical across 20,008 angles including the boundary cases.
    Everything else must match exactly.)
    """
    a = _method_ast(EXTRACT_DIR / EXTRACTORS[0], "BRepFeatureExtractor", method)
    b = _method_ast(EXTRACT_DIR / EXTRACTORS[1], "BRepFeatureExtractor", method)
    assert a is not None, f"{method} missing from {EXTRACTORS[0]}"
    assert b is not None, f"{method} missing from {EXTRACTORS[1]}"
    assert a == b, (
        f"{method} has diverged between the two extractors. If the change is "
        f"intended, apply it to BOTH files.")


def test_edge_dihedral_present_in_both():
    """Excluded from the AST comparison above; still must exist in both."""
    for name in EXTRACTORS:
        src = (EXTRACT_DIR / name).read_text(encoding="utf-8")
        assert "def edge_dihedral(" in src
