---
phase: 03-gpu-native-patch-extraction
plan: 01
subsystem: testing
tags: [pytest, dataclass, argparse, parity-gate, validation-contract]

requires:
  - phase: 01-baseline-parity-profiling-harness
    provides: "harness/run_harness.py check subcommand (HARN-05 gate)"
provides:
  - "harness/test_gpu_extract.py — 4 RED unit parity tests (EXT-01..EXT-04)"
  - "F2S3RunSettings.extraction_backend + .extraction_seed with validation"
  - "--extraction_backend / --extraction_seed CLI flags"
  - "03-EXT05-GATE.md — binding seeded A/B gate contract"
affects: [03-gpu-native-patch-extraction]

tech-stack:
  added: [pytest]
  patterns:
    - "RED contract-first test scaffold (test_parity.py analog)"
    - "typed dataclass field + __post_init__ ValueValidation + argparse.SUPPRESS"

key-files:
  created:
    - "harness/test_gpu_extract.py"
    - ".planning/phases/03-gpu-native-patch-extraction/03-EXT05-GATE.md"
  modified:
    - "src/f2s3/core.py"
    - "src/f2s3/cli.py"

key-decisions:
  - "EXT-05 gate: seeded CPU-vs-GPU A/B with max_abs within-implementation band + cos_mean >= 0.9995 floor"
  - "extraction_backend defaults to gpu (D-02a); cpu is opt-in fallback"
  - "extraction_seed defaults to 41 (D-04 seeded-by-default)"

patterns-established:
  - "RED contract-first test scaffold: deferred imports, np.random.RandomState seeds, __main__ PASS/FAIL loop"
  - "New run option: dataclass field + __post_init__ ValueError + argparse.SUPPRESS + args.* threading"

requirements-completed: [EXT-01, EXT-02, EXT-03, EXT-04, EXT-05]

duration: 3 min
completed: 2026-06-11
---

# Phase 03 Plan 01: Wave 0 Validation Contract Summary

**RED parity test scaffold with 4 named tests, extraction_backend/extraction_seed settings, and binding EXT-05 gate-metric contract**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-11T19:57:19Z
- **Completed:** 2026-06-11T20:01:06Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created `harness/test_gpu_extract.py` with exactly four tests (EXT-01..EXT-04) mirroring `harness/test_parity.py` conventions — all RED due to missing `f2s3.gpu_extract` (intended contract state)
- Added `extraction_backend` (str, default "gpu") and `extraction_seed` (int, default 41) to `F2S3RunSettings` with `ValueError` validation and `argparse.SUPPRESS` CLI flags
- Wrote `03-EXT05-GATE.md` documenting the binding EXT-05 gate: seeded CPU-vs-GPU A/B with displacement `max_abs` within-implementation band AND descriptor `cos_mean >= 0.9995`
- Threaded both new fields onto `PointCloudTile` so `compute_local_features` can branch in 03-02

## Task Commits

1. **Task 1: RED parity test scaffold** — `609333b` (test(03-01))
2. **Task 2: Settings + CLI + EXT-05 gate** — `328807c` (feat(03-01))

## Files Created/Modified
- `harness/test_gpu_extract.py` — 4 unit parity tests against CPU `extract_patch` oracle (EXT-01..EXT-04)
- `src/f2s3/core.py` — `extraction_backend` + `extraction_seed` fields, `__post_init__` validation, `PointCloudTile` threading
- `src/f2s3/cli.py` — `--extraction_backend` + `--extraction_seed` flags with `default=argparse.SUPPRESS`
- `.planning/phases/03-gpu-native-patch-extraction/03-EXT05-GATE.md` — binding seeded A/B gate contract

## Decisions Made
- EXT-05 gate metric: `max_abs` within Phase 1 within-implementation band + descriptor `cos_mean >= 0.9995` (binding floor, resolves RESEARCH Open Question 1)
- Default extraction backend is GPU (`extraction_backend="gpu"`), CPU retained as opt-in fallback (D-02a)
- Seed default is 41 (D-04 seeded-by-default)

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered
- Conda `f2s3` env had a stale installed copy of the package; resolved by `pip install -e .` to pick up source edits
- `.planning/` directory is gitignored; used `git add -f` for the EXT-05 gate doc (expected behavior)

## Next Phase Readiness
- All four tests are RED (collectable, fail only on `ModuleNotFoundError: f2s3.gpu_extract`) — ready for 03-02 GREEN
- Settings surface and CLI flags in place for 03-02 to branch on
- EXT-05 gate contract documented and binding — 03-04 knows exactly what to prove

---

*Phase: 03-gpu-native-patch-extraction*
*Completed: 2026-06-11*
