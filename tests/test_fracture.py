"""Headless tests for the fracture generation library.

Everything here is pure data: no Panda3D, no Bullet, no window. These tests
are the acceptance evidence for node 9, so the assertions are numeric, not
visual, and deliberately strict.
"""

from __future__ import annotations

import ast
import math
import os
import random
import statistics
import time

import pytest

from game import fracture as F
from game.fracture import Block, StructureSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Summed chunk volume must match the source structure this closely.
#: Half-space splitting is exact, so the only error is float round-off; the
#: stated contract tolerance is 0.5%, which we beat by many orders.
VOLUME_TOLERANCE = 0.005
#: A point may be claimed by at most one chunk. Points within this distance of
#: a shared face are ambiguous and excluded from the overlap count.
OVERLAP_SURFACE_EPS = 1e-6
#: Fraction of sampled interior points allowed to fall in 2+ chunks.
OVERLAP_TOLERANCE = 0.0
#: Generation must finish inside this many seconds per structure.
GEN_TIME_BOUND_S = 1.0

SEED = 20260914


@pytest.fixture(scope="module")
def specs():
    return F.default_specs()


@pytest.fixture(scope="module")
def results(specs):
    return {s.name: F.fracture(s, SEED) for s in specs}


# ------------------------------------------------------------------ hygiene
def test_module_imports_no_engine_libraries():
    """The whole point of node 9: this library must stay engine-free."""
    with open(os.path.join(ROOT, "game", "fracture.py")) as fh:
        tree = ast.parse(fh.read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    for name in names:
        low = name.lower()
        assert "panda3d" not in low, f"fracture.py imports {name}"
        assert "bullet" not in low, f"fracture.py imports {name}"
        assert not low.startswith("direct."), f"fracture.py imports {name}"


def test_four_archetypes_exist(specs):
    kinds = {s.kind for s in specs}
    assert kinds == {"tower", "slab", "arch", "cluster"}
    # And they really are different shapes, not the same box renamed.
    footprints = set()
    for s in specs:
        lo, hi = s.bounds
        footprints.add((round(hi[0] - lo[0], 3), round(hi[1] - lo[1], 3),
                        round(hi[2] - lo[2], 3)))
    assert len(footprints) == 4, f"archetypes share a bounding box: {footprints}"


# --------------------------------------------------------------- determinism
def test_same_spec_and_seed_is_byte_identical(specs):
    for spec in specs:
        a = F.fracture(spec, 4242).serialize()
        b = F.fracture(spec, 4242).serialize()
        assert a == b, f"{spec.name} is not reproducible under a fixed seed"


def test_determinism_survives_a_fresh_interpreter(specs):
    """Guards against PYTHONHASHSEED / dict-ordering leaks into the RNG."""
    import subprocess
    import sys

    code = (
        "import hashlib;from game import fracture as F;"
        "s=[x for x in F.default_specs() if x.name=='arch'][0];"
        "print(hashlib.sha256(F.fracture(s,99).serialize()).hexdigest())"
    )
    digests = set()
    for hashseed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=env,
            capture_output=True, text=True, check=True,
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1, f"digest varies with PYTHONHASHSEED: {digests}"


def test_different_seeds_give_different_fractures(specs):
    for spec in specs:
        a = F.fracture(spec, 1)
        b = F.fracture(spec, 2)
        assert a.serialize() != b.serialize(), f"{spec.name} ignores its seed"
        # Not merely different bytes - a visibly different pattern: the chunk
        # centroids must genuinely move, not just get relabelled.
        ca = set(tuple(round(v, 6) for v in c.center) for c in a.chunks)
        cb = set(tuple(round(v, 6) for v in c.center) for c in b.chunks)
        shared = len(ca.intersection(cb))
        assert shared < 0.25 * min(len(ca), len(cb)), (
            f"{spec.name}: seeds 1 and 2 share {shared} identical chunk centres"
        )


def test_seed_variation_is_broad_not_a_one_off():
    spec = F.arch_spec()
    digests = {F.fracture(spec, s).serialize() for s in range(12)}
    assert len(digests) == 12, "some seeds collapse onto the same fracture"


# --------------------------------------------------------- volume + geometry
def test_volume_is_conserved_within_tolerance(results):
    for name, r in results.items():
        assert r.volume_error <= VOLUME_TOLERANCE, (
            f"{name}: chunk volume {r.chunk_volume:.4f} vs source "
            f"{r.source_volume:.4f} (rel err {r.volume_error:.3e})"
        )


def test_mass_follows_volume_times_density(specs, results):
    for spec in specs:
        r = results[spec.name]
        for c in r.chunks:
            assert c.mass == pytest.approx(c.volume * spec.density, rel=1e-12)
        assert r.total_mass == pytest.approx(
            r.chunk_volume * spec.density, rel=1e-9
        )


def test_no_chunk_is_degenerate(results):
    for name, r in results.items():
        for c in r.chunks:
            assert c.volume > 0.0, f"{name} chunk {c.index} has volume {c.volume}"
            assert len(c.vertices) >= 4, f"{name} chunk {c.index} is not a solid"
            assert len(c.faces) >= 4, f"{name} chunk {c.index} has <4 faces"
            assert all(math.isfinite(v) for v in c.center)
            assert all(math.isfinite(x) for v in c.vertices for x in v)
            assert min(c.half_extents) > 1e-9, (
                f"{name} chunk {c.index} is flat: {c.half_extents}"
            )


def test_chunk_meshes_are_closed_polyhedra(results):
    """Euler characteristic V - E + F == 2 for every chunk.

    This is the strong form: it fails if a clip ever produced a hole, a
    duplicated face or a dangling edge.
    """
    for name, r in results.items():
        for c in r.chunks:
            v, e, f = len(c.vertices), len(c.edges), len(c.faces)
            assert v - e + f == 2, (
                f"{name} chunk {c.index}: V={v} E={e} F={f} (V-E+F={v - e + f})"
            )


def test_every_chunk_exposes_a_usable_edge_list(results):
    """The locked art direction draws chunk EDGES as glowing wireframe."""
    for name, r in results.items():
        for c in r.chunks:
            assert c.edges, f"{name} chunk {c.index} has no edges"
            assert len(set(c.edges)) == len(c.edges), "duplicate edges"
            for a, b in c.edges:
                assert a != b
                assert 0 <= a < len(c.vertices) and 0 <= b < len(c.vertices)
                seg = F._length(F._sub(c.vertices[a], c.vertices[b]))
                assert seg > 1e-9, f"{name} chunk {c.index} has a zero-length edge"
            # Every edge is shared by exactly two faces on a closed solid.
            counts = {}
            for face in c.faces:
                m = len(face)
                for i in range(m):
                    p, q = face[i], face[(i + 1) % m]
                    key = (p, q) if p < q else (q, p)
                    counts[key] = counts.get(key, 0) + 1
            assert set(counts.values()) == {2}, (
                f"{name} chunk {c.index} has non-manifold edges"
            )


def test_local_vertices_are_centred_on_the_centroid(results):
    for name, r in results.items():
        for c in r.chunks:
            faces = [[c.vertices[i] for i in f] for f in c.faces]
            centroid, vol = F.poly_centroid_volume(faces)
            assert vol == pytest.approx(c.volume, rel=1e-9)
            assert F._length(centroid) < 1e-7, (
                f"{name} chunk {c.index} local mesh is off-centre by "
                f"{F._length(centroid):.3e}"
            )


# ------------------------------------------------------------- containment
def test_every_chunk_lies_inside_the_structure_bounds(specs, results):
    pad = 1e-6
    for spec in specs:
        r = results[spec.name]
        lo, hi = spec.bounds
        for c in r.chunks:
            for p in c.world_vertices():
                for i in range(3):
                    assert lo[i] - pad <= p[i] <= hi[i] + pad, (
                        f"{spec.name} chunk {c.index} vertex {p} escapes "
                        f"bounds {lo}..{hi}"
                    )


def test_every_chunk_lies_inside_its_source_block(specs, results):
    """Tighter than the bounding box: the arch's hole must stay a hole."""
    pad = 1e-6
    for spec in specs:
        by_name = {b.name: b for b in spec.blocks}
        for c in results[spec.name].chunks:
            blo, bhi = by_name[c.source_block].bounds
            for p in c.world_vertices():
                for i in range(3):
                    assert blo[i] - pad <= p[i] <= bhi[i] + pad, (
                        f"{spec.name} chunk {c.index} leaks out of block "
                        f"{c.source_block}"
                    )


def test_arch_opening_stays_empty(results):
    """No chunk may occupy the gap under the arch's lintel."""
    r = results["arch"]
    # A point well inside the archway: between the legs, below the lintel.
    for probe in ((0.0, 0.0, 9.0), (3.0, 1.0, 12.0), (-4.0, -1.0, 5.0)):
        for c in r.chunks:
            assert not c.contains_point(probe), (
                f"chunk {c.index} fills the archway at {probe}"
            )


# --------------------------------------------------------------- non-overlap
def _sample_points(spec, rng, n):
    lo, hi = spec.bounds
    return [
        (rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]),
         rng.uniform(lo[2], hi[2]))
        for _ in range(n)
    ]


def test_chunks_do_not_overlap(specs, results):
    """Monte-Carlo occupancy: no interior point may sit in two chunks.

    Points lying within OVERLAP_SURFACE_EPS of a chunk face are shared-boundary
    cases and are excluded; anything deeper than that is a real overlap.
    """
    rng = random.Random(1234)
    for spec in specs:
        r = results[spec.name]
        overlaps = 0
        interior_hits = 0
        for p in _sample_points(spec, rng, 900):
            owners = [c for c in r.chunks
                      if c.contains_point(p, eps=-OVERLAP_SURFACE_EPS)]
            if owners:
                interior_hits += 1
            if len(owners) > 1:
                overlaps += 1
        assert interior_hits > 50, (
            f"{spec.name}: only {interior_hits} sample points landed in any "
            "chunk - the test is not actually exercising the solid"
        )
        rate = overlaps / interior_hits
        assert rate <= OVERLAP_TOLERANCE, (
            f"{spec.name}: {overlaps}/{interior_hits} interior points fall in "
            f"2+ chunks (rate {rate:.4f})"
        )


def test_chunks_cover_the_solid(specs, results):
    """The dual of non-overlap: points inside a source block must be claimed."""
    rng = random.Random(99)
    for spec in specs:
        r = results[spec.name]
        missed = 0
        tested = 0
        for block in spec.blocks:
            blo, bhi = block.bounds
            for _ in range(80):
                # Sample away from the block skin so boundary noise cannot
                # masquerade as a coverage hole.
                p = tuple(
                    rng.uniform(blo[i] + 0.02 * (bhi[i] - blo[i]),
                                bhi[i] - 0.02 * (bhi[i] - blo[i]))
                    for i in range(3)
                )
                tested += 1
                if not any(c.contains_point(p, eps=1e-7) for c in r.chunks):
                    missed += 1
        assert missed == 0, (
            f"{spec.name}: {missed}/{tested} interior points are in no chunk"
        )


# ------------------------------------------------------------ size variety
def test_chunk_sizes_span_at_least_an_order_of_magnitude(results):
    for name, r in results.items():
        vols = r.volumes()
        ratio = vols[-1] / vols[0]
        assert ratio >= 10.0, (
            f"{name}: largest/smallest chunk volume is only {ratio:.2f}"
        )


def test_size_distribution_populates_every_band(results):
    """A uniform voxel grid would dump everything into one band."""
    for name, r in results.items():
        bands = F.size_bands(r, bands=3)
        assert all(b > 0 for b in bands), f"{name}: empty size band {bands}"
        assert min(bands) >= 0.05 * len(r), (
            f"{name}: size bands are lopsided {bands} over {len(r)} chunks"
        )


def test_chunks_are_not_all_near_cubic(results):
    """Aspect ratio spread - the shard/slab requirement, asserted numerically."""
    for name, r in results.items():
        ars = sorted(c.aspect_ratio for c in r.chunks)
        med = statistics.median(ars)
        elongated = sum(1 for a in ars if a >= 3.0) / len(ars)
        near_cubic = sum(1 for a in ars if a < 1.5) / len(ars)
        assert med >= 1.8, f"{name}: median aspect ratio {med:.2f} is cube-like"
        assert elongated >= 0.20, (
            f"{name}: only {elongated:.0%} of chunks are elongated (>=3:1)"
        )
        assert near_cubic <= 0.45, (
            f"{name}: {near_cubic:.0%} of chunks are near-cubic"
        )
        assert ars[-1] >= 6.0, f"{name}: no real shards, max aspect {ars[-1]:.2f}"


def test_shapes_are_not_all_boxes(results):
    """Oblique cuts must actually produce non-box polyhedra."""
    for name, r in results.items():
        non_box = sum(1 for c in r.chunks
                      if len(c.vertices) != 8 or len(c.faces) != 6)
        assert non_box >= 0.15 * len(r), (
            f"{name}: only {non_box}/{len(r)} chunks are non-box polyhedra"
        )


# ------------------------------------------------------------ budget + perf
def test_chunk_count_respects_the_budget(specs):
    for spec in specs:
        for budget in (12, 50, 128, spec.max_chunks):
            r = F.fracture(F.with_budget(spec, budget), SEED)
            assert len(r) <= budget, (
                f"{spec.name} produced {len(r)} chunks for budget {budget}"
            )


def test_budget_is_actually_used_not_just_respected(specs):
    """A budget you never fill is a bug, not a safety feature."""
    for spec in specs:
        r = F.fracture(spec, SEED)
        assert len(r) >= 0.9 * spec.max_chunks, (
            f"{spec.name} used only {len(r)}/{spec.max_chunks} of its budget"
        )


def test_a_big_tower_yields_hundreds_of_chunks_not_thousands():
    r = F.fracture(F.tower_spec(), SEED)
    assert 200 <= len(r) <= 999, f"tower produced {len(r)} chunks"


def test_budget_below_block_count_is_rejected():
    spec = F.cluster_spec()
    with pytest.raises(ValueError):
        F.fracture(F.with_budget(spec, 2), SEED)


def test_every_block_contributes_chunks(specs, results):
    for spec in specs:
        produced = {c.source_block for c in results[spec.name].chunks}
        expected = {b.name for b in spec.blocks}
        assert produced == expected, (
            f"{spec.name}: blocks missing from the fracture: "
            f"{expected - produced}"
        )


def test_generation_is_fast_enough_for_load_time(specs):
    for spec in specs:
        best = math.inf
        for _ in range(3):
            t0 = time.perf_counter()
            F.fracture(spec, SEED)
            best = min(best, time.perf_counter() - t0)
        assert best < GEN_TIME_BOUND_S, (
            f"{spec.name} took {best:.3f}s to fracture (bound "
            f"{GEN_TIME_BOUND_S}s)"
        )


def test_whole_course_pregenerates_quickly(specs):
    """Load-time budget: a dozen structures must not stall the level load."""
    t0 = time.perf_counter()
    for i in range(12):
        F.fracture(specs[i % len(specs)], SEED + i)
    elapsed = time.perf_counter() - t0
    assert elapsed < 6.0, f"12 structures took {elapsed:.2f}s to pre-generate"


# ------------------------------------------------------------- launch hints
def test_launch_directions_point_outward_and_are_unit_length(specs, results):
    for spec in specs:
        origin = spec.resolved_fracture_origin()
        r = results[spec.name]
        agreeing = 0
        for c in r.chunks:
            assert F._length(c.launch_dir) == pytest.approx(1.0, abs=1e-9)
            radial = F._sub(c.center, origin)
            if F._length(radial) > 1e-6:
                assert F._dot(c.launch_dir, radial) > 0.0, (
                    f"{spec.name} chunk {c.index} launches inward"
                )
                agreeing += 1
        assert agreeing >= 0.95 * len(r)


def test_fracture_origin_is_configurable():
    base = F.slab_spec()
    moved = StructureSpec(
        name=base.name, kind=base.kind, blocks=base.blocks,
        fracture_origin=(base.bounds[1][0], 0.0, 0.0),
    )
    r = F.fracture(moved, SEED)
    # Blast from the +X edge: almost everything must be pushed toward -X.
    pushed = sum(1 for c in r.chunks if c.launch_dir[0] < 0.0)
    assert pushed >= 0.9 * len(r), (
        f"only {pushed}/{len(r)} chunks fly away from a +X edge blast"
    )


# ------------------------------------------------------- arbitrary new specs
def test_a_hand_written_spec_works():
    """The spec dataclass must serve structures beyond the four archetypes."""
    spec = StructureSpec(
        name="stub_wall",
        kind="wall",
        blocks=(Block("w", (0.0, 0.0, 1.5), (9.0, 0.35, 1.5)),),
        max_chunks=64,
        density=1800.0,
    )
    r = F.fracture(spec, 5)
    assert 0 < len(r) <= 64
    assert r.volume_error <= VOLUME_TOLERANCE
    assert r.volumes()[-1] / r.volumes()[0] >= 10.0


# ===================================================================
# Regression + contract-hardening tests added during independent
# re-validation of node 9. Every one of these was written because the
# original suite passed while the guarantee underneath it did not hold.
# ===================================================================

#: Exact pairwise chunk overlap allowed. Half-space splitting is exact, so the
#: only correct answer is "none": touching faces are not overlap (see
#: convex_pair_overlap's eps handling).
EXACT_OVERLAP_PAIRS_ALLOWED = 0


def test_source_blocks_never_interpenetrate(specs):
    """REGRESSION: cluster blocks c0/c5 and c1/c5 used to overlap by 6.5 m^3.

    Interpenetrating source blocks are a double fault: the shared volume is
    counted twice in ``spec.volume`` (so "volume conservation" conserves the
    wrong number), and the chunks carved from those blocks genuinely occupy the
    same space. Monte-Carlo sampling over a sparse bounding box missed it.
    """
    for spec in specs:
        bad = spec.overlapping_block_pairs()
        assert bad == [], (
            f"{spec.name}: source blocks interpenetrate: "
            + ", ".join(f"{a}<->{b} ({v:.4f} m^3)" for a, b, v in bad)
        )


def test_structure_volume_is_not_double_counted(specs):
    """spec.volume must equal the true occupied volume, not the sum of
    possibly-overlapping block volumes."""
    for spec in specs:
        double_counted = sum(v for _, _, v in spec.overlapping_block_pairs())
        assert double_counted == 0.0, (
            f"{spec.name}: {double_counted:.4f} m^3 of structure volume is "
            "counted twice"
        )


def test_validate_rejects_interpenetrating_blocks():
    """The validator must actually fire - a guard that never triggers is not a
    guard. This rebuilds the exact bug that shipped."""
    bad = StructureSpec(
        name="interpenetrating",
        kind="cluster",
        blocks=(
            Block("a", (0.0, 0.0, 2.0), (3.0, 3.0, 2.0)),
            Block("b", (2.0, 0.0, 2.0), (3.0, 3.0, 2.0)),  # overlaps a
        ),
        max_chunks=32,
    )
    assert bad.overlapping_block_pairs(), "test fixture does not actually overlap"
    with pytest.raises(ValueError, match="interpenetrat"):
        F.fracture(bad, SEED)


def test_touching_blocks_are_allowed():
    """Face-to-face contact is legal; only shared *volume* is a fault."""
    ok = StructureSpec(
        name="touching",
        kind="cluster",
        blocks=(
            Block("a", (0.0, 0.0, 2.0), (2.0, 2.0, 2.0)),
            Block("b", (4.0, 0.0, 2.0), (2.0, 2.0, 2.0)),  # touches at x=2
        ),
        max_chunks=32,
    )
    assert ok.overlapping_block_pairs() == []
    r = F.fracture(ok, SEED)
    assert len(r) > 0
    assert r.volume_error <= VOLUME_TOLERANCE


def test_chunks_are_provably_disjoint_by_separating_axis(specs):
    """Exact geometric non-overlap, not Monte-Carlo.

    Sampling can miss a thin interpenetration; a separating-axis test cannot.
    Checked against every chunk pair whose bounding spheres could touch.
    """
    for spec in specs:
        r = F.fracture(spec, SEED)
        chunks = r.chunks
        radius = [max(F._length(v) for v in c.vertices) for c in chunks]
        overlaps = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                d = F._length(F._sub(chunks[i].center, chunks[j].center))
                if d >= radius[i] + radius[j]:
                    continue  # bounding spheres disjoint - cannot overlap
                if F.convex_pair_overlap(chunks[i], chunks[j]):
                    overlaps.append((chunks[i].index, chunks[j].index))
        assert len(overlaps) <= EXACT_OVERLAP_PAIRS_ALLOWED, (
            f"{spec.name}: {len(overlaps)} chunk pairs interpenetrate, "
            f"e.g. {overlaps[:5]}"
        )


def test_separating_axis_check_detects_a_real_overlap():
    """Guard the guard: convex_pair_overlap must return True for solids that
    genuinely intersect, or the test above proves nothing."""
    a = F._emit_chunk(0, "a", F.box_faces((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                      F.tower_spec(), (0.0, 0.0, 0.0))
    hit = F._emit_chunk(1, "b", F.box_faces((0.5, 0.0, 0.0), (1.0, 1.0, 1.0)),
                        F.tower_spec(), (0.0, 0.0, 0.0))
    miss = F._emit_chunk(2, "c", F.box_faces((5.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                         F.tower_spec(), (0.0, 0.0, 0.0))
    touch = F._emit_chunk(3, "d", F.box_faces((2.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                          F.tower_spec(), (0.0, 0.0, 0.0))
    assert F.convex_pair_overlap(a, hit) is True, "failed to detect real overlap"
    assert F.convex_pair_overlap(a, miss) is False, "false positive on disjoint"
    assert F.convex_pair_overlap(a, touch) is False, "touching counted as overlap"


def test_overlap_sampling_actually_exercises_the_solid(specs):
    """The old non-overlap test sampled a sparse bounding box and only landed
    ~230 points inside the cluster. Sample inside the blocks instead."""
    rng = random.Random(777)
    for spec in specs:
        r = F.fracture(spec, SEED)
        overlaps = interior = 0
        for block in spec.blocks:
            blo, bhi = block.bounds
            for _ in range(250):
                p = tuple(rng.uniform(blo[i], bhi[i]) for i in range(3))
                owners = sum(1 for c in r.chunks
                             if c.contains_point(p, eps=-OVERLAP_SURFACE_EPS))
                if owners:
                    interior += 1
                if owners > 1:
                    overlaps += 1
        assert interior >= 200 * len(spec.blocks), (
            f"{spec.name}: only {interior} points landed in a chunk"
        )
        assert overlaps == 0, (
            f"{spec.name}: {overlaps}/{interior} interior points in 2+ chunks"
        )


def test_module_runs_with_engine_imports_unavailable():
    """CONTRACT CLAUSE 7, properly proven.

    The existing test only greps the AST for banned names. This runs the real
    generator in a subprocess where importing panda3d / direct / numpy raises,
    so the module is proven to import AND run with no engine present.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "class Blocker:\n"
        "    BANNED = ('panda3d', 'direct', 'numpy')\n"
        "    def find_module(self, name, path=None):\n"
        "        return self.find_spec(name, path)\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in self.BANNED:\n"
        "            raise ImportError('blocked engine dependency: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from game import fracture as F\n"
        "r = F.fracture(F.tower_spec(), 7)\n"
        "assert len(r) > 0 and r.volume_error < 1e-6\n"
        "leaked = [m for m in sys.modules "
        "          if m.split('.')[0] in ('panda3d', 'direct', 'numpy')]\n"
        "assert not leaked, leaked\n"
        "print('PURE', len(r))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT,
        capture_output=True, text=True,
    )
    assert out.returncode == 0, (
        f"fracture failed without engine libs:\n{out.stdout}\n{out.stderr}"
    )
    assert out.stdout.startswith("PURE"), out.stdout


def test_generation_scales_to_a_whole_level_under_budget():
    """CONTRACT CLAUSE 6: declared wall-clock bound on the largest archetype."""
    biggest = max(F.default_specs(), key=lambda s: s.max_chunks)
    t0 = time.perf_counter()
    r = F.fracture(biggest, SEED)
    elapsed = time.perf_counter() - t0
    assert elapsed < GEN_TIME_BOUND_S, (
        f"largest archetype {biggest.name} took {elapsed:.3f}s "
        f"(bound {GEN_TIME_BOUND_S}s)"
    )
    assert len(r) == biggest.max_chunks


def test_variety_holds_across_many_seeds_not_just_one(specs):
    """Guards against assertions tuned to a single lucky seed."""
    for spec in specs:
        for seed in range(6):
            r = F.fracture(spec, seed)
            vols = r.volumes()
            ars = sorted(c.aspect_ratio for c in r.chunks)
            assert vols[-1] / vols[0] >= 10.0, (
                f"{spec.name} seed {seed}: volume span only "
                f"{vols[-1] / vols[0]:.1f}x"
            )
            assert statistics.median(ars) >= 1.8, (
                f"{spec.name} seed {seed}: median aspect {statistics.median(ars):.2f}"
            )
            assert ars[-1] >= 6.0, f"{spec.name} seed {seed}: no shards"
            assert r.volume_error <= VOLUME_TOLERANCE


def test_budget_and_conservation_hold_across_many_seeds(specs):
    for spec in specs:
        for seed in range(6):
            r = F.fracture(spec, seed)
            assert len(r) <= spec.max_chunks
            assert r.volume_error <= VOLUME_TOLERANCE
            assert r.total_mass == pytest.approx(
                spec.volume * spec.density, rel=1e-6
            )
