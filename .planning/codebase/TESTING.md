# Testing Patterns

**Analysis Date:** 2026-06-11

> **Status: No automated test suite exists.** This document records the current
> (absent) testing state and the conventions that should be adopted when tests are
> introduced. There are no test files, no test framework configured, and no CI test
> job in this repository as of the analysis date.

## Current State

- **No test files.** A repo-wide search for `*test*.py`, `*spec*.py`, and `conftest.py` under `src/` returns nothing. The only matches are GSD tooling docs under `.claude/` and `.opencode/`, not project tests.
- **No test framework declared.** `pyproject.toml` lists no `pytest`/`unittest` dependency and defines no `[tool.pytest.ini_options]`. There is no `pytest.ini`, `tox.ini`, `setup.cfg`, or `vitest`/`jest` config.
- **No coverage tooling.** No `coverage`, `pytest-cov`, or coverage config present.
- **No CI test stage.** The two GitHub Actions workflows are `release-please.yml` (release automation) and `docker-ghcr.yml` (Docker image build/push on tags). Neither runs tests. See `.github/workflows/`.
- **Ad-hoc verification only:** `src/f2s3/filtering_model.py:97-103` contains an `if __name__ == '__main__':` block that instantiates `FilteringNetwork` and prints its trainable parameter count — a manual smoke check, not an automated test.
- **Runtime contract checks:** A single `assert x.dim() == 4 and x.shape[1] == 1` guards tensor shape in `src/f2s3/filtering_model.py:51`. This is an in-production invariant, not a test.

## Recommended Test Framework

When adding tests, use **pytest** (the de-facto standard for scientific Python and the framework assumed by the GSD `add-tests` workflow). Suggested setup:

**Add to `pyproject.toml`:**
```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

**Run Commands (target):**
```bash
pytest                       # Run all tests
pytest -k <expr>             # Run a subset by name
pytest --cov=f2s3            # Run with coverage
pytest -q tests/test_utils.py  # Run one module
```

## Recommended Test File Organization

**Location:** A top-level `tests/` directory (separate from `src/`), mirroring the package layout:
```
tests/
├── conftest.py              # Shared fixtures (synthetic point clouds, tmp dirs)
├── test_utils.py            # Pure-math helpers in src/f2s3/utils.py
├── test_core_settings.py    # F2S3RunSettings validation
├── test_data.py             # FeatureExtractionDataset patch extraction
└── test_models.py           # Model forward shape contracts (GPU-gated)
```

**Naming:** `test_<module>.py` files; `test_<behavior>()` functions; group with classes only when helpful.

## Highest-Value Targets (where to start)

These are the most testable units because they are pure / deterministic and have clear input-output contracts:

1. **`src/f2s3/utils.py`** — pure numeric functions, ideal first tests:
   - `transform_point_cloud(x1, R, t)` — verify identity transform is a no-op; verify NumPy and torch branches agree.
   - `kabsch_transformation_estimation(x1, x2, weights, ...)` — apply a known rotation+translation to points, confirm recovered `R`/`t` match within tolerance; confirm SVD-failure fallback returns identity (`utils.py:93-102`).
   - `compute_c2c(source_pc, target_pc)` — distances to a known nearest neighbor.
2. **`src/f2s3/core.py` — `F2S3RunSettings.__post_init__`** — validation logic is fully unit-testable without a GPU:
   - Missing source/target cloud raises `FileNotFoundError` (`core.py:129-132`).
   - Bad tiled-data dir raises `NotADirectoryError` (`core.py:134`).
   - Negative `voxel_grid_size`/`max_disp_magnitude`/`neighborhood_radius` and `batch_size < 1` raise `ValueError` (`core.py:157-166`).
   - Results/tiled dirs are created; `command_line_args.txt` is written.
3. **`src/f2s3/core.py` — `_radius_from_resolution`, `_median_nn_resolution`** — deterministic given a fixed RNG seed and small synthetic clouds.
4. **`src/f2s3/data.py` — `FeatureExtractionDataset.extract_patch` / `__len__`** — verify output shape is `(num_points, 3)` and the zero-padding branch for small patches (`data.py:91-93`).

## Mocking Strategy (recommended)

There is no existing mocking pattern to follow. When tests are added:

- **Mock heavy I/O and external native deps** rather than invoking them: `open3d.io.read_point_cloud`, and the git-dependency modules `pc_tiling.pc_tiling` / `supervoxel.supervoxel`. Use `unittest.mock.patch` or `monkeypatch`.
- **Do NOT mock the math.** Test `utils.py` functions against real NumPy/torch — that is the behavior under test.
- **Gate GPU tests:** Model forward passes call `.cuda()` unconditionally (`core.py:339`, `descriptor_model.py:48`). Skip these when CUDA is unavailable:
  ```python
  import torch, pytest
  pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
  ```
  Consider refactoring hard-coded `.cuda()` calls to a configurable device to make models CPU-testable.
- **Pretrained weights:** `feature_based_deformation_analysis` loads `model_best.pth` / `model_best.pt` from package data (`core.py:600-614`). Integration tests needing these should mark as slow/optional.

## Fixtures & Test Data (recommended)

- Generate synthetic point clouds in fixtures (e.g. random points + a known rigid transform) rather than committing large `.ply` files.
- Use `tmp_path` / `tmp_path_factory` for `results_dir` and `tiled_data` so `F2S3RunSettings` directory creation and file writes are isolated.
- Seed RNG (`numpy.random.seed`) in fixtures because `extract_patch` uses `np.random.choice` (`data.py:95`) and `_median_nn_resolution` can subsample query points (`core.py:88`).
- Real sample data currently lives in `data/` at the repo root (not wired into any test).

## Coverage

- **Current coverage: 0% (no tests).**
- **No coverage target enforced.**
- Recommended initial focus: 100% on `src/f2s3/utils.py` and `F2S3RunSettings` validation, since these are pure and high-risk for silent numerical/validation errors.

## Known Test Gaps & Risks

- **Latent bug, untested:** `FeatureExtractionDataset.reset_seed` calls `logging.info(...)` but `data.py` does not import `logging`, and references `self.randng` which is never defined (`data.py:104-106`). A test exercising this path would immediately surface the `NameError`/`AttributeError`.
- **Unused/broken import untested:** `from plotly.graph_objs.icicle import Pathbar` (`core.py:35`) would fail at import time if `plotly` is not installed — an import smoke test (`import f2s3.core`) would catch this.
- **Broad exception swallowing untested:** the per-tile `try/except Exception` (`core.py:692`) hides failures; tests should assert specific tiles raise/are skipped as expected.
- **No regression baseline:** the pipeline produces `.txt` displacement files; no golden-file comparison exists to detect numerical regressions from changes like the recent TF32 enablement (`core.py:596-598`).

## Test Types (target taxonomy)

- **Unit tests:** `utils.py` math, `F2S3RunSettings` validation, `extract_patch` shapes. No GPU, fast, deterministic.
- **Integration tests:** end-to-end `feature_based_deformation_analysis` on a tiny synthetic tile with real (or stubbed) models — slow, GPU-gated, mark `@pytest.mark.slow`.
- **E2E / CLI tests:** invoke `f2s3.cli:main` with `argparse` args against a temp workspace.

---

*Testing analysis: 2026-06-11*
