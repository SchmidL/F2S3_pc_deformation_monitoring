---
phase: 03-gpu-native-patch-extraction
plan: 02
subsystem: service
tags: [open3d-nns, dlpack, gpu, radius-search, feature-extraction]

requires:
  - phase: 03-gpu-native-patch-extraction
    provides: "RED test scaffold (03-01), extraction_backend/extraction_seed settings"
provides:
  - "src/f2s3/gpu_extract.py — GPU index + CSR radius search + padded coords/mask"
  - "compute_local_features GPU branch wired behind extraction_backend='gpu'"
  - "GREEN test_radius_returns_all and test_index_reuse"
affects: [03-gpu-native-patch-extraction]

tech-stack:
  added: []
  patterns:
    - "Open3D-tensor NearestNeighborSearch + DLPack zero-copy bridge"
    - "CSR ragged-to-padded scatter with validity mask"
    - "GpuNeighborIndex container for NNS + positions"

key-files:
  created:
    - "src/f2s3/gpu_extract.py"
  modified:
    - "src/f2s3/core.py"
    - "harness/test_gpu_extract.py"

key-decisions:
  - "GpuNeighborIndex container carries both NNS and DLPack positions (pybind attrs rejected)"
  - "CSR→padded uses repeat_interleave + cat(arange) scatter — simpler than pre-filtered flat arrays"

patterns-established:
  - "GPU extraction: build_index → radius_neighborhoods → padded coords+mask+sqrt-dist"

requirements-completed: [EXT-01, EXT-04]

duration: 4 min
completed: 2026-06-11
---

# Phase 03 Plan 02: First GPU Extraction Slice Summary

**GPU index + radius search with padded coords/validity mask, k=2 resolution reuse, and end-to-end GPU branch**

## Performance

- **Duration:** 4 min
- **Started:** 2026-06-11T20:01:56Z
- **Completed:** 2026-06-11T20:06:27Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- `src/f2s3/gpu_extract.py` with `build_index`, `radius_neighborhoods`, `gpu_resolution`, and thin `gpu_extract_features` — Open3D-tensor NNS on CUDA:0, DLPack zero-copy bridge to torch
- `radius_neighborhoods` returns both nearest-dropped ptnn and full ptall sets with padding/validity masks (D-03/EXT-01)
- `gpu_resolution` reuses the same index for k=2 median NN (EXT-04)
- `compute_local_features` branches on `extraction_backend`: cpu path verbatim, gpu path wired end-to-end
- `test_radius_returns_all` and `test_index_reuse` GREEN; `test_lrf_parity` and `test_subsample_seeded` RED (intended — 03-03 fills in)

## Task Commits

1. **Task 1: GPU index + radius search module** — `e8581c5` (feat(03-02))
2. **Task 2: Branch compute_local_features + wire GPU path** — `5ca0eeb` (feat(03-02))

## Files Created/Modified
- `src/f2s3/gpu_extract.py` — GpuNeighborIndex, build_index, radius_neighborhoods, gpu_resolution, gpu_extract_features (thin placeholder)
- `src/f2s3/core.py` — `from . import gpu_extract`, extraction_backend branch in compute_local_features
- `harness/test_gpu_extract.py` — tests from 03-01 now exercise the real module

## Decisions Made
- Pybind NearestNeighborSearch rejects attribute assignment; used GpuNeighborIndex container class
- CSR→padded scatter uses repeat_interleave + cat(arange) for source index construction (simpler than pre-filtered flat arrays, avoids index out-of-bounds)

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered
- Initial _pad_to_max had index out-of-bounds CUDA error due to incorrect flat-array filtering; replaced with simpler scatter approach using repeat_interleave
- Pybind C++ object rejected `._positions` attribute assignment; wrapped in GpuNeighborIndex container

## Next Phase Readiness
- Two GREEN tests (EXT-01, EXT-04) — GPU radius search and index reuse verified
- Two RED tests (EXT-02, EXT-03) — ready for 03-03 LRF + subsample parity
- GPU path runs end-to-end (thin LRF/subsample placeholder)

---

*Phase: 03-gpu-native-patch-extraction*
*Completed: 2026-06-11*
