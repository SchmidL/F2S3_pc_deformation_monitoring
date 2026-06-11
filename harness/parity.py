"""
Numerical-parity comparator for the F2S3 harness (HARN-02).

Source-coordinate-keyed displacement-parity check and descriptor drift
diagnostic.  Compares a candidate run's displacement field against the
baseline, keyed on source coordinates (never row index — Pitfall 1).
The headline gate is ``max_abs``; RMS + percentiles + descriptor drift
are reported as diagnostics only (D-05 / D-06).
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def load_disp(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a 7-column displacement file written by the pipeline.

    The pipeline writes::

        np.savetxt(path,
            np.concatenate((filtered_results, filtered_magnitudes.reshape(-1, 1)), axis=1))

    where ``filtered_results`` is ``(N, 6)`` (cols 0:3 src xyz, cols 3:6 tgt
    xyz) and ``filtered_magnitudes`` is ``(N,)`` appended as col 6.

    Args:
        path: Path to the ``displacement_magnitude_tile_*.txt`` file.

    Returns:
        A ``(source_coords, magnitudes)`` tuple where:
        - *source_coords* is ``(N, 3)`` float64 — the stable KEY (cols 0:3).
        - *magnitudes* is ``(N,)`` float64 — displacement magnitude (col 6).
    """
    a = np.loadtxt(path)  # (N, 7): [src_xyz(0:3) | tgt_xyz(3:6) | magnitude(6)]
    return a[:, 0:3], a[:, 6]


def displacement_parity(
    base_path: Path,
    cand_path: Path,
    match_tol: float = 1e-6,
) -> Dict[str, float]:
    """Compare a candidate displacement field to the baseline.

    Matches baseline rows to candidate rows by **source point coordinates**
    (cols 0:3 of the 7-column output) using a sklearn KD-tree — NEVER by row
    index (Pitfall 1).  The displacement output is the inlier-filtered subset
    so row count and order legitimately differ between runs.

    Args:
        base_path: Path to the baseline
            ``displacement_magnitude_tile_*.txt``.
        cand_path: Path to the candidate
            ``displacement_magnitude_tile_*.txt``.
        match_tol: Maximum Euclidean distance in source-coordinate space for
            two points to be considered the same physical point.  Default
            ``1e-6`` is appropriate for metre-scale coordinates.

    Returns:
        Dict with keys:
        - ``max_abs`` (float) — **headline gate** (D-03): maximum absolute
          displacement-magnitude difference across matched points.  ``nan``
          if no matches.
        - ``rms`` (float) — root-mean-square of the magnitude differences
          (D-05 diagnostic).
        - ``p50``, ``p95``, ``p99`` (float) — percentiles of the magnitude
          differences (D-05 diagnostics).  ``nan`` if no matches.
        - ``matched_fraction`` (float) — fraction of baseline points that
          found a candidate match within *match_tol* (coverage diagnostic).
        - ``n_baseline`` (int) — number of rows in the baseline file.
        - ``n_candidate`` (int) — number of rows in the candidate file.
    """
    src_b, mag_b = load_disp(base_path)
    src_c, mag_c = load_disp(cand_path)

    n_base = int(src_b.shape[0])
    n_cand = int(src_c.shape[0])

    if n_base == 0 or n_cand == 0:
        logging.warning(
            "displacement_parity: baseline has %d rows, candidate has %d rows — "
            "returning nan metrics.",
            n_base,
            n_cand,
        )
        return _empty_result(n_base, n_cand)

    # Fit KD-tree on candidate source coordinates, query with baseline.
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    nn.fit(src_c)
    dist, idx = nn.kneighbors(src_b)  # dist: (n_base, 1), idx: (n_base, 1)

    matched = dist.reshape(-1) <= match_tol
    n_matched = int(matched.sum())

    if n_matched == 0:
        logging.warning(
            "displacement_parity: 0 of %d baseline points matched "
            "(match_tol=%.2e) — returning nan metrics.",
            n_base,
            match_tol,
        )
        return _empty_result(n_base, n_cand)

    # Compute magnitude deltas for matched points only.
    delta = np.abs(mag_b[matched] - mag_c[idx.reshape(-1)[matched]])

    return {
        "max_abs": float(delta.max()),
        "rms": float(np.sqrt(np.mean(delta ** 2))),
        "p50": float(np.percentile(delta, 50)),
        "p95": float(np.percentile(delta, 95)),
        "p99": float(np.percentile(delta, 99)),
        "matched_fraction": float(matched.mean()),
        "n_baseline": n_base,
        "n_candidate": n_cand,
    }


def _empty_result(n_base: int, n_cand: int) -> Dict[str, float]:
    """Return a result dict with ``nan`` metrics for empty-match scenarios."""
    return {
        "max_abs": float("nan"),
        "rms": float("nan"),
        "p50": float("nan"),
        "p95": float("nan"),
        "p99": float("nan"),
        "matched_fraction": 0.0,
        "n_baseline": n_base,
        "n_candidate": n_cand,
    }


def descriptor_drift(
    base_npz: Path,
    cand_npz: Path,
) -> Dict[str, float]:
    """Compute descriptor drift between baseline and candidate ``.npz`` files.

    Diagnostic only (D-06) — **never a gate**.  The stochastic
    ``np.random.choice`` subsample in ``extract_patch`` (``data.py:95``) makes
    descriptors differ run-to-run with zero code change.

    Args:
        base_npz: Path to the baseline ``features_tile_*.npz``.
        cand_npz: Path to the candidate ``features_tile_*.npz``.

    Returns:
        Dict with keys:
        - ``cos_mean``, ``cos_min`` (float) — cosine similarity statistics
          over the ``feat_s`` vectors (unit vectors, range [-1, 1]).
        - ``l2_mean``, ``l2_max`` (float) — L2-distance statistics over the
          ``feat_s`` vectors.
        All values are ``nan`` if either file is missing or empty.
    """
    b = np.load(base_npz)
    c = np.load(cand_npz)

    fb = b["feat_s"]  # (N, 64), L2-normalized by the model
    fc = c["feat_s"]  # (M, 64)

    n = min(fb.shape[0], fc.shape[0])
    if n == 0:
        logging.warning("descriptor_drift: empty feature arrays — returning nan.")
        return {
            "cos_mean": float("nan"),
            "cos_min": float("nan"),
            "l2_mean": float("nan"),
            "l2_max": float("nan"),
        }

    # Cosine similarity (unit vectors — dot product is cosine sim).
    cos = np.sum(fb[:n] * fc[:n], axis=1)
    l2 = np.linalg.norm(fb[:n] - fc[:n], axis=1)

    return {
        "cos_mean": float(cos.mean()),
        "cos_min": float(cos.min()),
        "l2_mean": float(l2.mean()),
        "l2_max": float(l2.max()),
    }
