# Coding Conventions

**Analysis Date:** 2026-06-11

This is a Python 3.11+ scientific computing package (`f2s3`) for deep-learning-based
deformation analysis of point clouds (PyTorch + Open3D + NumPy). Conventions below are
inferred from the actual source in `src/f2s3/`. There is **no enforced linter or formatter**
(black/flake8 are listed but commented out in `pyproject.toml`), so conventions are
established by existing code rather than tooling.

## Naming Patterns

**Files:**
- Lowercase, single word or `snake_case` module names: `core.py`, `utils.py`, `data.py`, `cli.py`, `descriptor_model.py`, `filtering_model.py`.
- Model definition modules are suffixed `_model`: `descriptor_model.py`, `filtering_model.py`.

**Functions:**
- `snake_case` for all functions: `transform_point_cloud`, `compute_c2c`, `kabsch_transformation_estimation` (`src/f2s3/utils.py`).
- Module-private helpers are prefixed with a single underscore: `_radius_from_resolution`, `_median_nn_resolution` (`src/f2s3/core.py:48`, `:61`).
- Instance-private methods also use a leading underscore: `_compute_resolution`, `_forward` (`src/f2s3/core.py:569`, `src/f2s3/descriptor_model.py:88`).

**Variables:**
- `snake_case` for locals and attributes: `median_resolution`, `supervoxel_radius`, `local_features_s` (`src/f2s3/core.py`).
- Source/target pairs use `_s` / `_t` suffixes consistently: `pcd_s`/`pcd_t`, `path_s`/`path_t`, `local_features_s`/`local_features_t` (`src/f2s3/core.py:208`).
- Short math-domain names mirror the underlying math/paper notation (acceptable in numeric code): `M`, `efC`, `efS` for HNSW params (`src/f2s3/core.py:386`); `x1`, `x2`, `R`, `t`, `u`, `s`, `v`, `cov_mat` in `kabsch_transformation_estimation` (`src/f2s3/utils.py:43`).
- Module-level constants are `UPPER_SNAKE` with a leading underscore when private: `_EPS = 1e-6` (`src/f2s3/data.py:6`).

**Types / Classes:**
- `PascalCase` for classes: `F2S3RunSettings`, `PointCloudTile` (`src/f2s3/core.py`), `FeatureExtractionDataset` (`src/f2s3/data.py`).
- Neural-network modules keep their original research/paper names: `STN3d`, `PointNetFeature`, `PointCN`, `FilteringNetwork`.

## Code Style

**Formatting:**
- No formatter is active. `black` is mentioned only in a commented-out `[project.optional-dependencies]` block in `pyproject.toml`.
- 4-space indentation throughout.
- Indentation/alignment is **inconsistent**: some multi-line calls align arguments under the opening paren (`src/f2s3/core.py:273`), others use hanging indents. Spacing around `=` in kwargs is inconsistent (e.g. `normalize_w = True, eps = 1e-7` in `src/f2s3/utils.py:43` vs `eps=1e-3` in `src/f2s3/filtering_model.py:13`).
- Line length is not enforced; many lines exceed 100 chars (e.g. log/format strings in `src/f2s3/core.py:295`, `:699`).
- **Convention to follow for new code:** Use `snake_case`, 4-space indent, no space around `=` in keyword arguments, and keep lines under ~100 chars. Prefer matching the cleaner style in `core.py` over the older `utils.py`/`filtering_model.py` style.

**Linting:**
- None configured. `flake8` appears only in a commented-out dependency block.

## Import Organization

**Order (as observed, not strictly grouped):**
1. Python standard library: `os`, `glob`, `logging`, `time`, `gc`, `re`, `from pathlib import Path`, `from typing import Optional`, `from dataclasses import dataclass`.
2. Third-party: `numpy as np`, `torch`, `open3d as o3d`, `coloredlogs`, `hnswlib`, `tqdm`, `from sklearn.neighbors import NearestNeighbors`.
3. First-party relative imports: `from .descriptor_model import PointNetFeature`, `from .data import FeatureExtractionDataset`, `from .utils import transform_point_cloud, compute_c2c`.
4. External git-dependency packages: `from pc_tiling import pc_tiling`, `from supervoxel import supervoxel`.

**Notes:**
- Grouping is loose and not blank-line separated everywhere (see `src/f2s3/core.py:19-45`).
- Intra-package imports always use **explicit relative imports** (`from .core import ...`, `from .utils import ...`).
- There is a stray/unused import: `from plotly.graph_objs.icicle import Pathbar` (`src/f2s3/core.py:35`) — do not imitate; remove such imports.
- `data.py` references `logging` in `reset_seed` (`src/f2s3/data.py:105`) without importing it — a latent bug; always import what you use.

## Error Handling

**Patterns:**
- Input validation is centralized in the `F2S3RunSettings` dataclass `__post_init__` (`src/f2s3/core.py:126`), raising specific built-in exceptions with f-string messages:
  - `FileNotFoundError` for missing input clouds.
  - `NotADirectoryError` for a bad tiled-data path.
  - `ValueError` for invalid numeric arguments (negative sizes, non-positive batch size).
- The main per-tile loop wraps processing in a broad `try/except Exception as e` and logs `logging.error(f"Error processing tile {tile_nr}: {e}")`, then continues to the next tile (`src/f2s3/core.py:692`). This favors pipeline robustness (one bad tile does not abort the run) over fail-fast.
- Numerical guards use try/except around `torch.svd` to fall back to identity rotation when SVD fails to converge (`src/f2s3/utils.py:93-102`).
- `assert` is used for tensor-shape contracts in model code: `assert x.dim() == 4 and x.shape[1] == 1` (`src/f2s3/filtering_model.py:51`).
- **Convention:** Raise specific built-in exceptions with descriptive f-string messages for user/config errors; use broad `try/except` + log + continue only for per-item loop resilience.

## Logging

**Framework:** Python standard `logging`, configured via `coloredlogs.install(level='INFO', ...)` inside `F2S3RunSettings.__post_init__` (`src/f2s3/core.py:170`).

**Patterns:**
- Calls use the root logger directly: `logging.info(...)`, `logging.warning(...)`, `logging.error(...)` (not a module-scoped `logger = logging.getLogger(__name__)`).
- Most informational logging is gated behind `if self.verbose:` (`src/f2s3/core.py:312`, `:393`).
- Timing is logged frequently: capture `start_time = time.time()` then log elapsed seconds with `'... {:.2f} s'.format(...)`.
- **Inconsistency:** mix of `.format()` and f-strings for log messages. Prefer f-strings for new code: `logging.info(f"... {value:.2f} s")`.
- **Convention:** Use module-level `logging` calls; gate verbose detail behind a `verbose` flag; always log elapsed time for long-running stages.

## Comments

**When to Comment:**
- Explain non-obvious algorithmic steps and reference the paper equation: `# eq. 3`, `# eq. 4`, `# e.q. 5` (`src/f2s3/data.py:47-62`).
- Document tuning parameters and their tradeoffs inline (e.g. HNSW recall vs. runtime at `src/f2s3/core.py:384`; TF32 / cudnn.benchmark rationale at `src/f2s3/core.py:593-598`).
- Commented-out legacy code blocks exist (`src/f2s3/core.py:218-222`, `:619-621`). **Do not add new commented-out code**; delete dead code instead.

**Docstrings:**
- Google-style docstrings with `Args:` / `Returns:` sections are the norm for functions and public methods (`src/f2s3/utils.py:6`, `src/f2s3/core.py:48`, `:61`, `:189`).
- Module-level docstring describes the pipeline and cites references (`src/f2s3/core.py:1-17`).
- Quality is uneven: several typos in docstrings (`teh`, `Eucledean`, `Returns_`, `ectraction`) and array shapes annotated as `[n,3]`. New docstrings should follow the Google style and fix typos.
- No type-annotation-driven docs; types are described in prose inside the docstring (e.g. `(torch array)`, `(np array)`).

## Type Hints

- Type hints are **partial**, not comprehensive.
- The dataclass `F2S3RunSettings` is fully typed, including `Optional[Path]` and defaults (`src/f2s3/core.py:108-124`).
- Function signatures use hints selectively: `def __init__(self, path_s: Path, path_t: Path, tile_id: str, ...)` (`src/f2s3/core.py:189`), `def feature_based_deformation_analysis(args: F2S3RunSettings)` (`src/f2s3/core.py:582`).
- Most numeric helper functions in `utils.py` and `data.py` are **unannotated**, relying on docstrings for type info.
- **Convention:** Type-hint public function signatures and dataclass fields; use `Optional[...]` from `typing` and `pathlib.Path` for filesystem paths.

## Function & Method Design

**Size:** Functions are generally medium-to-large. `filter_correspondences` (`src/f2s3/core.py:430-567`) and `feature_based_deformation_analysis` (`:582-701`) are long, branchy procedures — acceptable for the pipeline orchestration but candidates for decomposition.

**Parameters:**
- Defaults are provided generously for tunable scientific parameters: `kabsch_transformation_estimation(x1, x2, weights=None, normalize_w=True, eps=1e-7, best_k=0, w_threshold=0)` (`src/f2s3/utils.py:43`).
- The CLI uses `argparse.SUPPRESS` defaults so the dataclass owns the real default values (`src/f2s3/cli.py:23-35`).

**Return Values:**
- Functions return plain tuples of tensors/arrays (`kabsch_transformation_estimation` returns 4-tuple, `src/f2s3/utils.py:116`).
- Model forwards return tuples whose arity varies by flag (`src/f2s3/descriptor_model.py:115-131`) — caller unpacks with `_` placeholders (`src/f2s3/core.py:340`).
- Structured results use plain `dict` with string keys: `filter_input` returns `{'scores', 'rot_est', 'trans_est', 'robust_estimate'}` (`src/f2s3/filtering_model.py:84-90`).

## Module & Package Design

**Exports:**
- The package `__init__.py` declares an explicit `__all__ = ["__version__", "cli", "core"]` and imports the version + submodules (`src/f2s3/__init__.py`).
- Version is derived dynamically by `setuptools_scm` into the generated `src/f2s3/_version.py` (do not edit by hand).

**Entry point:**
- CLI entry is `f2s3.cli:main` registered as a console script in `pyproject.toml` (`[project.scripts]`).
- `main()` builds an `argparse` parser, constructs `F2S3RunSettings(**vars(args))`, and runs the pipeline inside `torch.no_grad()` (`src/f2s3/cli.py:54-61`).

**Barrel Files:** None beyond `__init__.py`. No re-export hub modules.

## Configuration & Settings

- All run configuration flows through the single `F2S3RunSettings` dataclass (`src/f2s3/core.py:108`). It validates inputs, creates output/tiled-data directories, configures logging, and persists the resolved arguments to `{results_dir}/command_line_args.txt` via a custom `__repr__` (`src/f2s3/core.py:180`).
- **Convention:** Add new run-time options as typed fields on `F2S3RunSettings` with a corresponding `argparse` flag (using `default=argparse.SUPPRESS`) in `cli.py`, and validate them in `__post_init__`.

## Device / Tensor Conventions

- CUDA is assumed available; tensors are moved with `.cuda()` directly (not a configurable `device`) — e.g. `batch_feat = batch.squeeze(0).cuda()` (`src/f2s3/core.py:339`), `feature_descriptor.cuda()` (`src/f2s3/core.py:608`).
- Inference is wrapped in `torch.no_grad()` at the entry point and models are set to `.eval()` after loading weights (`src/f2s3/core.py:609`, `:614`).
- GPU hygiene: explicit `torch.cuda.empty_cache()` and `gc.collect()` after heavy stages (`src/f2s3/core.py:358-359`, `:468-469`).
- Performance backends are enabled globally at pipeline start: TF32 matmul/cudnn and `cudnn.benchmark` (`src/f2s3/core.py:596-598`).

---

*Convention analysis: 2026-06-11*
