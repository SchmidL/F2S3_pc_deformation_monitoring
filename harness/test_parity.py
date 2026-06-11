"""
Verification tests for displacement parity and descriptor drift (HARN-02).

These tests exercise ``harness/parity.py`` — the source-coordinate-keyed
comparator and descriptor diagnostic.  Run with::

    python harness/test_parity.py

Each test is a zero-dependency assertion; failures raise ``AssertionError``
with diagnostic context.

Coverage — every behavior case from Plan 01-02 Task 1:
  1. Identical displacement arrays -> max_abs==0, matched_fraction==1.0
  2. Shuffled candidate rows -> still max_abs==0 (source-xzy key, not row index)
  3. +0.1 magnitude offset -> max_abs ~= 0.1
  4. Empty overlap -> metrics are nan, no crash
  5. Descriptor drift on identical feat_s -> cos_mean ~= 1.0, l2_max ~= 0.0
"""

import sys
import tempfile
import os

import numpy as np

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_identical_displacement():
    """Two identical 7-column displacement arrays -> max_abs==0, matched_fraction==1.0."""
    import parity; displacement_parity = parity.displacement_parity

    src = np.random.RandomState(42).rand(50, 3)
    base = np.hstack([
        src,
        src + 0.5,
        np.linalg.norm(0.5 * np.ones((50, 3)), axis=1, keepdims=True),
    ])
    b = tempfile.mktemp(suffix=".txt")
    c = tempfile.mktemp(suffix=".txt")
    np.savetxt(b, base)
    np.savetxt(c, base)

    r = displacement_parity(b, c)
    assert r["max_abs"] < 1e-9, f"Expected max_abs=0, got {r['max_abs']}"
    assert abs(r["matched_fraction"] - 1.0) < 1e-9, (
        f"Expected matched_fraction=1.0, got {r['matched_fraction']}"
    )
    assert r["n_baseline"] == 50
    assert r["n_candidate"] == 50
    print(f"  PASS identical: max_abs={r['max_abs']:.2e}, matched={r['matched_fraction']}")


def test_shuffled_candidate():
    """Shuffled candidate rows -> still max_abs==0 (source-xyz key, NOT row index)."""
    import parity; displacement_parity = parity.displacement_parity

    src = np.random.RandomState(43).rand(50, 3)
    base = np.hstack([
        src,
        src + 0.5,
        np.linalg.norm(0.5 * np.ones((50, 3)), axis=1, keepdims=True),
    ])
    perm = np.random.RandomState(44).permutation(50)
    b = tempfile.mktemp(suffix=".txt")
    c = tempfile.mktemp(suffix=".txt")
    np.savetxt(b, base)
    np.savetxt(c, base[perm])

    r = displacement_parity(b, c)
    assert r["max_abs"] < 1e-9, (
        f"Shuffled rows should NOT change max_abs (source-xyz key), got {r['max_abs']}"
    )
    assert abs(r["matched_fraction"] - 1.0) < 1e-9, (
        f"All points should match via KD-tree, got matched_fraction={r['matched_fraction']}"
    )
    print(f"  PASS shuffled: max_abs={r['max_abs']:.2e}, matched={r['matched_fraction']}")


def test_offset_magnitude():
    """+0.1 offset at every matched point -> max_abs ~= 0.1."""
    import parity; displacement_parity = parity.displacement_parity

    src = np.random.RandomState(45).rand(50, 3)
    base = np.hstack([
        src,
        src + 0.5,
        np.linalg.norm(0.5 * np.ones((50, 3)), axis=1, keepdims=True),
    ])
    offset = base.copy()
    offset[:, 6] += 0.1
    b = tempfile.mktemp(suffix=".txt")
    c = tempfile.mktemp(suffix=".txt")
    np.savetxt(b, base)
    np.savetxt(c, offset)

    r = displacement_parity(b, c)
    assert abs(r["max_abs"] - 0.1) < 1e-6, (
        f"Expected max_abs ~= 0.1, got {r['max_abs']}"
    )
    assert abs(r["rms"] - 0.1) < 1e-6, (
        f"Expected rms ~= 0.1, got {r['rms']}"
    )
    print(f"  PASS offset: max_abs={r['max_abs']:.6f}, rms={r['rms']:.6f}")


def test_descriptor_drift_identical():
    """Descriptor drift on identical feat_s -> cos_mean ~= 1.0, l2_max ~= 0.0."""
    import parity; descriptor_drift = parity.descriptor_drift

    feat = np.random.RandomState(46).rand(30, 64).astype(np.float32)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True)  # L2-normalize

    n1 = tempfile.mktemp(suffix=".npz")
    n2 = tempfile.mktemp(suffix=".npz")
    np.savez(n1, feat_s=feat, feat_t=feat)
    np.savez(n2, feat_s=feat, feat_t=feat)

    d = descriptor_drift(n1, n2)
    assert abs(d["cos_mean"] - 1.0) < 1e-5, (
        f"Expected cos_mean ~= 1.0, got {d['cos_mean']}"
    )
    assert d["l2_max"] < 1e-4, (
        f"Expected l2_max ~= 0.0, got {d['l2_max']}"
    )
    assert d["cos_min"] > 0.9999, (
        f"Expected cos_min ~= 1.0, got {d['cos_min']}"
    )
    print(f"  PASS descriptor drift: cos_mean={d['cos_mean']:.6f}, l2_max={d['l2_max']:.2e}")


def test_empty_overlap():
    """Empty overlap -> metrics are float('nan'), no crash."""
    import parity; displacement_parity = parity.displacement_parity

    # Two completely different source-point sets (no spatial overlap)
    src1 = np.random.RandomState(47).rand(20, 3) * 1.0   # [0,1)
    src2 = np.random.RandomState(48).rand(20, 3) * 1.0 + 100.0  # [100,101)
    base = np.hstack([src1, src1 + 0.3, np.ones((20, 1)) * 0.3])
    cand = np.hstack([src2, src2 + 0.3, np.ones((20, 1)) * 0.3])
    b = tempfile.mktemp(suffix=".txt")
    c = tempfile.mktemp(suffix=".txt")
    np.savetxt(b, base)
    np.savetxt(c, cand)

    r = displacement_parity(b, c, match_tol=1e-6)
    assert np.isnan(r["max_abs"]), f"Expected nan max_abs, got {r['max_abs']}"
    assert np.isnan(r["rms"]), f"Expected nan rms, got {r['rms']}"
    assert r["matched_fraction"] == 0.0, (
        f"Expected matched_fraction=0, got {r['matched_fraction']}"
    )
    print(f"  PASS empty overlap: max_abs={r['max_abs']}, matched={r['matched_fraction']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("harness/test_parity.py — displacement parity & descriptor drift tests\n")
    tests = [
        ("identical displacement", test_identical_displacement),
        ("shuffled candidate (source-xyz key)", test_shuffled_candidate),
        ("+0.1 magnitude offset", test_offset_magnitude),
        ("empty overlap (nan safety)", test_empty_overlap),
        ("descriptor drift on identical feat_s", test_descriptor_drift_identical),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            print(f"  FAIL {name}: {exc}")
            failed += 1

    if failed:
        print(f"\n{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests PASSED")
        sys.exit(0)
