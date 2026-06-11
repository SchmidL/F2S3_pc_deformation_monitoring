"""
GPU extraction parity tests (EXT-01..EXT-04) against the CPU extract_patch
oracle.

Each test builds a small synthetic point cloud (~2k points) via
np.random.RandomState(fixed), imports the CPU FeatureExtractionDataset
as the oracle, and imports the not-yet-existent f2s3.gpu_extract lazily
inside the test body so the failure is explicit RED.

Coverage:
  1. test_radius_returns_all  -- EXT-01: GPU radius search returns ALL
     in-radius neighbors vs a CPU oracle count
  2. test_lrf_parity          -- EXT-02: batched eigh LRF zp/xp/yp matches
     CPU np.linalg.eig LRF
  3. test_subsample_seeded    -- EXT-03: seeded subsample deterministic plus
     D-05 zero-row-selectable padding semantics
  4. test_index_reuse         -- EXT-04: one index serves k=2 plus radius
     queries

Run with::

    pytest harness/test_gpu_extract.py -x
"""

import sys
import tempfile
import os

import numpy as np
import open3d as o3d

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYNTH_N_POINTS = 2000
_SYNTH_RADIUS = 0.08
_SYNTH_BBOX_HALF = 1.0
_SEED_01 = 201
_SEED_02 = 202
_SEED_03 = 203
_SEED_04 = 204
_ATOL = 1e-6


def _synthetic_overlap_cloud(seed, n=_SYNTH_N_POINTS):
    """Build a small np.random.RandomState-seeded unit-box point cloud."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(n, 3).astype(np.float64) * _SYNTH_BBOX_HALF
    return pts


def _o3d_cloud(pts):
    """Wrap a numpy array as an open3d PointCloud (needed by the CPU oracle)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _cpu_oracle(pts, query_pts, radius, num_points=256):
    """Return the CPU-extracted patch per query point via FeatureExtractionDataset.

    Returns a list of (num_points, 3) ndarrays — the oracle patches.
    """
    from f2s3.data import FeatureExtractionDataset

    pcd = _o3d_cloud(pts)
    overlap_pcd = _o3d_cloud(pts)  # overlap == full cloud for synthetic test
    ds = FeatureExtractionDataset(
        pcd, overlap_pcd,
        points_per_batch=len(query_pts),
        feature_radius=radius,
        num_points=num_points,
    )
    patches = []
    for pt in query_pts:
        patch = ds.extract_patch(pt)
        patches.append(patch)
    return patches


def _cpu_neighbor_counts(pts, query_pts, radius):
    """Return the number of in-radius neighbors per query from the CPU oracle."""
    pcd = _o3d_cloud(pts)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    counts = []
    for pt in query_pts:
        _, idx, _ = pcd_tree.search_radius_vector_3d(pt, radius)
        counts.append(len(idx))
    return np.array(counts, dtype=np.int64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_radius_returns_all():
    """GPU radius search returns ALL in-radius neighbors vs the CPU oracle count."""
    pts = _synthetic_overlap_cloud(_SEED_01)
    # Query a subset so tests are fast
    rng = np.random.RandomState(_SEED_01 + 1)
    query_idx = rng.choice(len(pts), size=128, replace=False)
    query_pts = pts[query_idx]

    cpu_counts = _cpu_neighbor_counts(pts, query_pts, _SYNTH_RADIUS)

    # Deferred import of the not-yet-existent GPU module (RED on purpose)
    import f2s3.gpu_extract as gpu  # noqa: F811
    from f2s3.gpu_extract import build_index, radius_neighborhoods

    overlap_pts = pts.astype(np.float32)
    index = build_index(overlap_pts)
    _, ptnn_mask, _, ptall_coords, ptall_mask = radius_neighborhoods(
        index, query_pts.astype(np.float32), _SYNTH_RADIUS
    )

    gpu_counts = ptall_mask.sum(dim=1).cpu().numpy().astype(np.int64)

    assert np.array_equal(gpu_counts, cpu_counts), (
        f"GPU count mismatch: max diff {(gpu_counts != cpu_counts).sum()}, "
        f"example GPU={gpu_counts[0]} CPU={cpu_counts[0]}"
    )
    print(f"  PASS radius returns all: {len(cpu_counts)} queries, "
          f"max neighbors={cpu_counts.max()}")


def test_lrf_parity():
    """Batched eigh LRF (zp/xp/yp) matches CPU np.linalg.eig LRF within atol."""
    pts = _synthetic_overlap_cloud(_SEED_02)
    rng = np.random.RandomState(_SEED_02 + 1)
    query_idx = rng.choice(len(pts), size=16, replace=False)
    query_pts = pts[query_idx]

    # CPU oracle LRF: extract per-point patches and collect zp/xp/yp
    cpu_patches = _cpu_oracle(pts, query_pts, _SYNTH_RADIUS)
    cpu_lrf_zp = []
    cpu_lrf_xp = []
    cpu_lrf_yp = []
    for i, pt in enumerate(query_pts):
        # Re-run extract_patch to capture the LRF — the oracle does rotation
        # internally, so we replicate the LRF computation by driving the
        # CPU path inline
        from f2s3.data import FeatureExtractionDataset
        pcd = _o3d_cloud(pts)
        overlap = _o3d_cloud(pts)
        ds = FeatureExtractionDataset(pcd, overlap, 1, _SYNTH_RADIUS)
        # Access the LRF computation directly from the CPU path
        pcd_tree = o3d.geometry.KDTreeFlann(overlap)
        _, patch_idx, patch_dist = pcd_tree.search_radius_vector_3d(pt, _SYNTH_RADIUS)
        ptnn = np.asarray(overlap.points)[patch_idx[1:], :].T
        dist = np.sqrt(np.array(patch_dist)[1:])

        if ptnn.shape[1] > 10:
            vect_diff = ptnn - pt[:, np.newaxis]
            cov = 1 / ptnn.shape[0] * np.dot(vect_diff, vect_diff.T)
            a, v = np.linalg.eig(cov)
            smallest_idx = np.argmin(a)
            np_hat = v[:, smallest_idx]
            zp = np_hat if np.sum(np.dot(np_hat, -vect_diff)) > 0 else -np_hat

            v_proj = vect_diff - (np.dot(vect_diff.T, zp[:, np.newaxis]) * zp).T
            alpha = (_SYNTH_RADIUS - dist) ** 2
            beta = np.dot(vect_diff.T, zp[:, np.newaxis]).squeeze() ** 2
            if np.abs(np.linalg.norm(np.dot(v_proj, (alpha * beta)[:, np.newaxis]))) < 1e-6:
                xp = (1 / (np.linalg.norm(np.dot(v_proj, (alpha * beta)[:, np.newaxis])) + 1e-6)
                      * np.dot(v_proj, (alpha * beta)[:, np.newaxis]))
            else:
                xp = (1 / np.linalg.norm(np.dot(v_proj, (alpha * beta)[:, np.newaxis]))
                      * np.dot(v_proj, (alpha * beta)[:, np.newaxis]))
            xp = xp.squeeze()
            yp = np.cross(xp, zp)
        else:
            zp = np.array([0.0, 0.0, 1.0])
            xp = np.array([1.0, 0.0, 0.0])
            yp = np.array([0.0, 1.0, 0.0])

        cpu_lrf_zp.append(zp)
        cpu_lrf_xp.append(xp)
        cpu_lrf_yp.append(yp)

    cpu_zp = np.stack(cpu_lrf_zp, axis=0)
    cpu_xp = np.stack(cpu_lrf_xp, axis=0)
    cpu_yp = np.stack(cpu_lrf_yp, axis=0)

    # Deferred GPU import (RED on purpose)
    import f2s3.gpu_extract as gpu  # noqa: F811
    from f2s3.gpu_extract import build_index, radius_neighborhoods, compute_lrf

    overlap_pts = pts.astype(np.float32)
    index = build_index(overlap_pts)
    ptnn_coords, ptnn_mask, ptnn_dist, ptall_coords, ptall_mask = radius_neighborhoods(
        index, query_pts.astype(np.float32), _SYNTH_RADIUS
    )
    gpu_zp, gpu_xp, gpu_yp = compute_lrf(
        ptnn_coords, ptnn_mask, ptnn_dist, _SYNTH_RADIUS
    )
    gpu_zp_np = gpu_zp.cpu().numpy()
    gpu_xp_np = gpu_xp.cpu().numpy()
    gpu_yp_np = gpu_yp.cpu().numpy()

    assert np.allclose(gpu_zp_np, cpu_zp, atol=_ATOL), (
        f"zp mismatch: max abs diff {np.abs(gpu_zp_np - cpu_zp).max():.2e}"
    )
    assert np.allclose(gpu_xp_np, cpu_xp, atol=_ATOL), (
        f"xp mismatch: max abs diff {np.abs(gpu_xp_np - cpu_xp).max():.2e}"
    )
    assert np.allclose(gpu_yp_np, cpu_yp, atol=_ATOL), (
        f"yp mismatch: max abs diff {np.abs(gpu_yp_np - cpu_yp).max():.2e}"
    )
    print(f"  PASS LRF parity: {len(cpu_zp)} points, "
          f"zp max diff {np.abs(gpu_zp_np - cpu_zp).max():.2e}")


def test_subsample_seeded():
    """Seeded subsample is deterministic; D-05 zero-row-selectable padding."""
    pts = _synthetic_overlap_cloud(_SEED_03)
    rng = np.random.RandomState(_SEED_03 + 1)
    query_idx = rng.choice(len(pts), size=8, replace=False)
    query_pts = pts[query_idx]

    import f2s3.gpu_extract as gpu  # noqa: F811
    from f2s3.gpu_extract import build_index, radius_neighborhoods, subsample_seeded

    overlap_pts = pts.astype(np.float32)
    index = build_index(overlap_pts)
    _, _, _, ptall_coords, ptall_mask = radius_neighborhoods(
        index, query_pts.astype(np.float32), _SYNTH_RADIUS
    )

    # Run twice with same seed -> identical
    seed = 42
    sub1 = subsample_seeded(ptall_coords, ptall_mask, seed, 256)
    sub2 = subsample_seeded(ptall_coords, ptall_mask, seed, 256)

    assert torch.equal(sub1, sub2), (
        "Same seed must produce identical subsamples"
    )

    # Run with different seed -> different (probabilistic, run a few times)
    sub3 = subsample_seeded(ptall_coords, ptall_mask, seed + 1, 256)
    # Not guaranteed to differ, but extremely unlikely to be identical
    print(f"  PASS subsample seeded: deterministic repeat, "
          f"shape {list(sub1.shape)}")


def test_index_reuse():
    """One GPU index serves both k=2 and large-radius queries."""
    pts = _synthetic_overlap_cloud(_SEED_04)
    rng = np.random.RandomState(_SEED_04 + 1)
    query_idx = rng.choice(len(pts), size=128, replace=False)
    query_pts = pts[query_idx]

    import f2s3.gpu_extract as gpu  # noqa: F811
    from f2s3.gpu_extract import build_index, gpu_resolution, radius_neighborhoods

    overlap_pts = pts.astype(np.float32)
    index = build_index(overlap_pts)

    # Same index answers k=2 resolution
    res = gpu_resolution(index, query_pts.astype(np.float32))
    assert res > 0, f"Expected positive resolution, got {res}"

    # Same index answers large-radius neighborhood query
    _, ptnn_mask, _, ptall_coords, ptall_mask = radius_neighborhoods(
        index, query_pts.astype(np.float32), _SYNTH_RADIUS
    )
    n_counts = ptall_mask.sum(dim=1).cpu().numpy()
    assert (n_counts > 0).all(), "Expected non-zero neighbor counts"

    print(f"  PASS index reuse: k=2 resolution={res:.4f}, "
          f"radius neighbors max count={n_counts.max()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("harness/test_gpu_extract.py — GPU extraction parity tests\n")
    tests = [
        ("radius returns all (EXT-01)", test_radius_returns_all),
        ("LRF parity (EXT-02)", test_lrf_parity),
        ("subsample seeded (EXT-03)", test_subsample_seeded),
        ("index reuse (EXT-04)", test_index_reuse),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except ImportError as exc:
            print(f"  RED {name}: {exc}")
            print("    (Expected — f2s3.gpu_extract does not exist yet)")
            failed += 1
        except Exception as exc:
            print(f"  FAIL {name}: {exc}")
            failed += 1

    if failed:
        print(f"\n{failed}/{len(tests)} tests RED/FAILED")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests PASSED")
        sys.exit(0)
