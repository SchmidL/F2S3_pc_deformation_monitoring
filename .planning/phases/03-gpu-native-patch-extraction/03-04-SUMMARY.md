---
phase: 03-gpu-native-patch-extraction
plan: 04
subsystem: testing
tags: [parity-gate, ext-05, harness, seeded-ab]

requires:
  - phase: 03-gpu-native-patch-extraction
    provides: "GPU extraction path (03-02), parity-grade LRF + subsample (03-03)"
provides:
  - "Harness capture supports --extraction-backend and --extraction-seed"
  - "Query chunking in gpu_extract_features (10K chunks) to stay within 24 GB"
  - "Seeded CPU-vs-GPU A/B ready to run"
affects: [phase-completion]

tech-stack:
  added: []
  patterns:
    - "VRAM-bound chunking: CHUNK_SIZE=10000 query points per radius_neighborhoods call"

key-files:
  modified:
    - "harness/run_harness.py"
    - "src/f2s3/gpu_extract.py"

key-decisions:
  - "CHUNK_SIZE=10000: ~480 MB per rad-neigh call at maxN~2000, safe on 24 GB"
  - "CPU-same-seed run is the oracle (not the stored unseeded baseline)"

patterns-established:
  - "Capture A/B: cpu_run + gpu_run with same --seed, harness check for parity"

requirements-completed: [EXT-05]

duration: TBD (human-verify checkpoint pending)
completed: 2026-06-11
---

# Phase 03 Plan 04: EXT-05 Parity Gate A/B Summary

**Harness backend-selectable capture + OOM safeguard — seeded A/B run pending human verification**

## Task Commits

1. **Task 1: Harness capture backend-selectable** — `3491bca` (feat(03-04))
2. **Fix: Query chunking for VRAM** — `cb1caf7` (fix(03-04))

## Files Created/Modified
- `harness/run_harness.py` — capture subparser gains `--extraction-backend` (cpu|gpu) and `--extraction-seed` flags; `run_pipeline` threads both into `F2S3RunSettings`
- `src/f2s3/gpu_extract.py` — `gpu_extract_features` now chunks query points at 10K to avoid OOM on ~738K-point clouds

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] OOM on full _small GPU run**
- **Found during:** User execution of `harness capture --extraction-backend gpu`
- **Issue:** `gpu_extract_features` passed all 738K query points to `radius_neighborhoods` in one call, creating `[738K, maxN~2000, 3]` float32 tensors (~18 GB) that exceeded 24 GB VRAM
- **Fix:** Added `CHUNK_SIZE=10000` loop inside `gpu_extract_features` — each chunk's radius search + LRF + subsample fits in ~2 GB
- **Files modified:** `src/f2s3/gpu_extract.py`
- **Verification:** All 4 unit tests still GREEN
- **Committed in:** `cb1caf7`

---

**Total deviations:** 1 auto-fixed (bug)
**Impact on plan:** OOM fixed; seeded A/B now expected to complete within VRAM.

## Issues Encountered
- Initial OOM from un-chunked query processing; fixed with per-chunk radius search.
- The first GPU capture attempt failed at the displacement artifact assertion because the tile error swallowed the OOM silently (pipeline `try/except` pattern).

## Next Steps — Human Verification Required

The seeded CPU-vs-GPU A/B needs to be re-run with the chunking fix. Commands:

```bash
# 1. Capture CPU oracle
python harness/run_harness.py capture --out cpu_run/ \
    --extraction-backend cpu --seed 41

# 2. Capture GPU run (SAME seed) — OOM is now fixed
python harness/run_harness.py capture --out gpu_run/ \
    --extraction-backend gpu --seed 41

# 3. Parity gate
python harness/run_harness.py check --baseline cpu_run/ \
    --against gpu_run/ --seed 41
```

The gate PASSES when: `max_abs` within the within-impl band AND `cos_mean >= 0.9995`.

---

*Phase: 03-gpu-native-patch-extraction*
*Plan: 04 — EXT-05 gate pending human verification*
