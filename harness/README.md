# F2S3 Regression Harness — Gate Contract (HARN-05)

**This is the named regression gate that Phases 2–7 invoke by name.**  It
compares a candidate pipeline run against the FP32 baseline and exits with a
well-defined code.  The contract documented here is the stable interface that
every optimisation phase depends on.

---

## Quick start

```bash
# 1. Capture the FP32 baseline (once; ~30 min on the _small pair):
python harness/run_harness.py capture --out baseline/

# 2. Run the parity gate against a candidate run:
python harness/run_harness.py check --baseline baseline/ --against candidate_run/
```

---

## Single-command invocation (HARN-05)

Every later phase invokes the gate with the same command:

```bash
python harness/run_harness.py check \
    --baseline baseline/ \
    --against <candidate_run_dir>/
```

**The `check` subcommand:**
1. Resolves the displacement `.txt` files from both directories (glob
   `output/results/displacement_magnitude_tile_*.txt`).
2. Applies the Pitfall 4 guard: raises a hard error if files are missing or
   empty (the pipeline swallows per-tile exceptions silently).
3. Matches displacement rows by **source-point coordinates** via a
   `sklearn.neighbors.NearestNeighbors` KD-tree — this is **order-invariant**
   and handles the variable-length, variable-order inlier subset (Pitfall 1).
4. Computes `max_abs` (the headline gate metric, D-03), plus
   RMS / p50 / p95 / p99 as diagnostics (D-05).
5. Computes **descriptor drift** (cosine similarity + L2 distance over the
   64-dim `feat_s` vectors) as a diagnostic only — **never a gate** (D-06).
6. Computes per-stage **timing deltas** against the baseline `timings.json`.
7. Prints a structured report to stdout.

---

## Exit-code contract

The exit code encodes the parity verdict.  This is the **only** signal Phases
2–7 should depend on for automation:

| Condition | Verdict | Exit code | Meaning |
|-----------|---------|-----------|---------|
| `max_abs ≤ WARN` | **PASS** | `0` | Displacement field matches baseline within the WARN band.  Optimisation is safe to land. |
| `WARN < max_abs ≤ FAIL` | **WARN** | `0` (flagged) | Displacement drift within the WARN band — surfaced but does not auto-block.  "Otherwise we have to look at it again" (D-04). |
| `max_abs > FAIL` | **FAIL** | `1` (non-zero) | Displacement exceeds the FAIL threshold.  **Hard stop** — the optimisation changes the output beyond the acceptable tolerance. |

**Design note:** Both PASS and WARN exit 0 so CI pipelines are not blocked by a
WARN verdict.  The WARN is visible in the stdout report and logged at
`WARNING` level.  FAIL exits non-zero so shell `&&` chains and CI `set -e`
catch it.

### Customising the bands

```bash
# Tighter bands for a high-confidence check:
python harness/run_harness.py check \
    --baseline baseline/ --against candidate/ \
    --warn 0.005 --fail 0.02

# Disable the FAIL gate (diagnostic-only, always exits 0):
python harness/run_harness.py check \
    --baseline baseline/ --against candidate/ \
    --fail 999
```

---

## WARN / FAIL defaults and noise-floor calibration (D-08)

| Parameter | Default | Units | Basis |
|-----------|---------|-------|-------|
| `--warn` | **0.01** | metres (1 cm) | Surface drift that warrants human review (D-04). |
| `--fail` | **0.05** | metres (5 cm) | Hard stop — output is materially different (D-04). |

### Noise-floor justification

The shipped defaults are **calibrated above the measured noise floor** (D-08).
The natural run-to-run variation of the unseeded FP32 baseline was measured as:

| Metric | Value |
|--------|-------|
| Number of unseeded captures (N) | 5 |
| Measured noise floor (max-abs across all 10 pairs) | **276.7 m** |
| Shipped WARN band | 0.01 m (1 cm) |
| WARN / noise-floor ratio | 0.000036× |
| D-08 self-check | **FAILED — see investigation below** |

> **⚠️ D-08 self-check FAILED (2026-06-11):** The measured noise floor
> (276.7 m) is **four orders of magnitude above** the shipped 1 cm WARN band.
> This is **not** numerical noise — it is **correspondence instability**
> caused by `np.random.choice` in `extract_patch` (`data.py:95`).  Different
> random subsamples → different descriptors → HNSW matches different target
> points → displacement magnitudes differ by up to 276 m for the same source
> point.
>
> **The cm-level max-abs displacement-difference bands are valid ONLY for
> seeded within-implementation comparisons** (e.g. Phase 1.1 TF32 vs FP32 with
> `--seed`).  For cross-implementation changes (GPU extraction, FP16), the
> gate metric must account for correspondence instability — either via
> statistical distribution comparison over many unseeded runs or by redefining
> the gate on a more stable intermediate (e.g. descriptor cosine similarity
> before correspondence search, or matched-fraction of correspondences).
>
> The shipped `--warn 0.01` and `--fail 0.05` defaults remain as the
> documented contract; operators and Phase 2–7 implementors must use
> `--seed` for within-implementation gates and re-evaluate the gate
> metric for cross-implementation changes.

### Coordinate units (A1 sanity-check)

The `.ply` header does not declare a unit.  The displacement magnitude
distribution (median ~0.5 m in the `TCH_nochange` reference run) and the
coordinate range (~2300–2600 m in a localised coordinate system) are
**consistent with metres**.  The 1 cm / 5 cm bands are therefore meaningful.

If the data were in millimetres or another unit, the numeric bands (0.01 /
0.05) would need to be re-derived.  The bands are configurable via `--warn`
and `--fail` for exactly this reason.

---

## Reference pair (D-01, D-02)

The gate runs exclusively on the **`_small` epoch pair**:

| Role | File | Points |
|------|------|--------|
| Source (epoch 1) | `data/20210924_THC_centered_small.ply` | ~737,977 |
| Target (epoch 2) | `data/20220831_THC_centered_small.ply` | ~723,597 |

**Small-only gate (D-02):** No second-tier check on `_sample` or full pairs.
The `_small` pair is cheap enough (~30 min) to re-run after every optimisation,
so the gate actually gets used.

---

## Baseline policy

- **`baseline/` is gitignored.**  The baseline is regenerable with a single
  `capture` command and is not committed to version control.
- **Regenerate on demand:** `python harness/run_harness.py capture --out baseline/`
- **Provenance:** Every capture writes `command_line_args.txt` (the resolved
  `F2S3RunSettings`) and `env.json` (torch / CUDA / open3d versions) alongside
  the displacement and descriptor artifacts.
- **Descriptor storage:** Features are written as `.npz` (`feat_s` / `feat_t`,
  64-dim L2-normalised) to `<run_dir>/features/`.  Descriptors require
  `--save_interim` (the harness sets this automatically for `capture`).

---

## D-07: Statistical parity rationale

**Parity is treated as inherently statistical — the noise-floor band, not
bit-exactness, is the foundation of the gate.**

The unseeded FP32 baseline does **not** reproduce itself bit-for-bit because:

1. `extract_patch` (`data.py:95`) calls `np.random.choice(..., replace=False)`
   per point to subsample the neighbourhood to 256 points — unseeded, different
   every run.
2. `_median_nn_resolution` (`core.py:88`) subsamples query points with
   `np.random.choice` — also unseeded.
3. The per-point LRF eigendecomposition (`np.linalg.eig` on the 3×3 covariance
   matrix at `data.py:51`) is numerically sensitive to the subsampled
   neighbourhood — different subsamples → slightly different eigenvectors →
   slightly different rotated patches → slightly different descriptors →
   slightly different correspondences → different points survive the
   `> 0.99999` inlier filter.

**Later phases amplify this:** Phase 3 swaps `np.random.choice` for a GPU
permutation; Phase 1.1 introduces FP16 rounding; `torch.compile` fuses ops
differently.  No optimised run will reproduce the baseline bit-for-bit.

**Seeding (D-09) is a secondary aid, not a correctness guarantee.**  Opt-in
deterministic seeding (`--seed`) makes two runs of the *same* implementation
reproducible, which is useful for debugging, but it cannot paper over the
structural numeric differences that optimisations introduce.  The gate's
tolerance bands — calibrated above the measured noise floor — are the only
reliable contract across implementations.

---

## Recorded toolchain

Captured in `env.json` by the harness and reproduced here for Phases 3/5
library selection (Pitfall 5):

| Component | Version | Relevance |
|-----------|---------|-----------|
| **torch** | **2.7.1+cu118** | CUDA 11.8, NOT the CUDA 12.6/cu126 toolchain in the Dockerfile.  **Phases 3/5 must select GPU neighbour-search and ANN libraries that match cu118, or the toolchain must be upgraded first.** |
| CUDA | 11.8 | |
| open3d | 0.19.0 | |
| numpy | 2.1.2 | |
| GPU driver | 595.79 | RTX 3090 Ti |
| f2s3 | 0.0.post60 | |

---

## Directory layout (after capture)

```
baseline/                               # gitignored
├── command_line_args.txt               # resolved F2S3RunSettings
├── env.json                            # torch/cuda/open3d versions
├── timings.json                        # per-stage seconds (HARN-03)
├── output/results/
│   └── displacement_magnitude_tile_000.txt   # 7-col: src_xyz|tgt_xyz|mag
├── features/
│   └── features_tile_000_*.npz         # feat_s, feat_t (64-dim, L2-normed)
└── tiled_data/                         # pc_tiling output (start_from_tiled_data)
```

---

*HARN-05 gate contract — Phase 1, Plan 3 — June 2026*
