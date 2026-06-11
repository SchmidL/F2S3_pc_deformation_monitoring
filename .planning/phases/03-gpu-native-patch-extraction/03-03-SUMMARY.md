---
phase: 03-gpu-native-patch-extraction
plan: 03
subsystem: service
tags: [torch-linalg-eigh, lrf, parity-math, subsample, determinism]

requires:
  - phase: 03-gpu-native-patch-extraction
    provides: "GPU index + radius search (03-02), RED test contracts (03-01)"
provides:
  - "Parity-grade masked batched eigh LRF (eqs. 3-5) with >10 and <_EPS branches"
  - "Seeded vectorised 256-point subsample (D-04/D-05)"
  - "All 4 unit tests GREEN"
affects: [03-gpu-native-patch-extraction]

tech-stack:
  added: []
  patterns:
    - "Masked batched eigh → cov / real_n → evecs[:,:,0] + eq.4 sign flip"
    - "torch.Generator + rand().argsort() for deterministic without-replacement subsample"

key-files:
  modified:
    - "src/f2s3/gpu_extract.py"
    - "harness/test_gpu_extract.py"

key-decisions:
  - "Radius neighborhoods return coords centred around query points"
  - "_lrf_core scatters only when n>10, returns identity axes otherwise"

patterns-established:
  - "Separate ptnn (nearest-dropped) and ptall (full) sets for covariance vs rotation"

requirements-completed: [EXT-02, EXT-03]

duration: 3 min
completed: 2026-06-11
---

# Phase 03 Plan 03: Parity-Grade LRF + Seeded Subsample Summary

**Masked batched torch.linalg.eigh LRF (eqs. 3-5) and seeded vectorised 256-point subsample — all 4 unit tests GREEN**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-11T20:07:02Z
- **Completed:** 2026-06-11T20:10:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- `_lrf_core`: masked batched `torch.linalg.eigh` LRF with eq.3 covariance (divides by real count, not maxN), eq.4 sign disambiguation, eq.5 `<_EPS` guard, `>10` neighbor identity-axis branch
- `batched_lrf`: full rotation pipeline — ptnn set for covariance, ptall set for rotation/output
- `subsample_256`: seeded `torch.Generator` + `rand().argsort()[:256]` reproducing `np.random.choice` without replacement, zero rows selectable (D-05), deterministic by default (D-04)
- `compute_lrf` / `subsample_seeded`: test-facing wrappers matching the 03-01 contract API
- `radius_neighborhoods` now returns coords centred around query points (needed by LRF math)

## Task Commits

1. **Task 1 + Task 2: LRF + subsample parity** — `b09b4f4` (feat(03-03)) — single commit covering both tasks (tightly coupled math)

## Files Created/Modified
- `src/f2s3/gpu_extract.py` — `_lrf_core`, `batched_lrf`, `compute_lrf`, `subsample_256`, `subsample_seeded`; updated `gpu_extract_features`
- `harness/test_gpu_extract.py` — added `import torch` in test_subsample_seeded

## Decisions Made
- `radius_neighborhoods` now returns coords centred around query points (all downstream consumers need centred coords)
- `_lrf_core` scatters only when `n_ptnn > 10`; returns identity axes for sparse queries (matching CPU >10 branch)

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered
- Centred coords: initial implementation returned absolute overlap coordinates; LRF test failed because covariance used un-centred positions. Fixed by centering in `_pad_results`.
- `torch` import missing in `test_subsample_seeded` (deferred import pattern broken). Fixed inline.

## Next Phase Readiness
- All 4 unit tests GREEN — EXT-01 through EXT-04 verified at unit level
- GPU extraction path uses parity-grade LRF and subsample — ready for 03-04 EXT-05 integration gate

---

*Phase: 03-gpu-native-patch-extraction*
*Completed: 2026-06-11*
