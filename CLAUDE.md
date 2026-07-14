
## Project

**F2S3 Performance & Throughput Optimization**

A runtime-optimization effort on **F2S3**, a CUDA/PyTorch CLI pipeline that estimates a 3D displacement field between two point-cloud epochs using deep-learned local features and supervoxel-constrained outlier filtering. This project makes the pipeline materially faster on a single **NVIDIA RTX 3090 Ti** (Ampere, 24 GB) — by actually using the GPU it has and moving CPU-bound work onto it — **without changing the displacement output beyond a validated numerical tolerance**. It is for the maintainers/operators running F2S3 on large epoch pairs.

**Core Value:** Reduce end-to-end wall-clock runtime of the F2S3 pipeline on a single RTX 3090 Ti, while keeping the resulting descriptors and displacement field numerically equivalent (within a validated tolerance) to the current FP32 baseline. **Speed is worthless here if it silently changes a sub-millimeter deformation measurement.**

### Constraints

- **Correctness**: Every change must be validated for numerical parity of descriptors and the final displacement field against the FP32 baseline — sub-millimeter metrology tolerance — using the regression harness. Parity gate is non-negotiable.
- **Hardware**: Single RTX 3090 Ti (24 GB, Ampere). GPU required; no CPU fallback path is added.
- **Dependencies**: GPU neighbor-search libs (FRNN / torch_cluster / PyTorch3D) and FAISS-GPU/cuVS add build complexity and must match the CUDA 12.6 / PyTorch cu126 toolchain. C++ extensions (`pc_tiling`, `supervoxel`) are git-pinned native deps.
- **Sequence**: Harness before optimizations; quick wins before refactors; profile between tiers (don't optimize blind).
- **Tech stack**: Python ≥3.11, PyTorch (CUDA cu126), Open3D ~0.15, hnswlib, scikit-learn.

## Technology Stack

## Languages

- Python >=3.11 - All application logic (`src/f2s3/*.py`). The Docker runtime ships Python 3 from Ubuntu 24.04; compiled bytecode present is `cpython-312`.
- C++ - Tiling and supervoxel segmentation. Not vendored in this repo; provided by external Git dependencies `pc_tiling` and `supervoxel`, built at install time via CMake + SWIG bindings.
- CUDA - GPU kernels are consumed through PyTorch (`.cuda()` calls in `src/f2s3/core.py`); no hand-written `.cu` files live in this repository.

## Runtime

- CPython >=3.11 (declared in `pyproject.toml`, `requires-python = ">=3.11"`)
- NVIDIA GPU with CUDA support is required at runtime (`src/f2s3/core.py` calls `.cuda()` unconditionally)
- CUDA 12.6 toolchain (Docker base image `nvidia/cuda:12.6.3-cudnn-*-ubuntu24.04`, PyTorch wheels from `https://download.pytorch.org/whl/cu126`)
- pip (used in `Dockerfile` and `docs/installation.md`)
- Lockfile: missing - dependencies are unpinned in `pyproject.toml` (only `open3d ~=0.15` and pinned Git tags are version-constrained)

## Frameworks

- PyTorch (`torch`) - Deep learning runtime for feature descriptor and outlier-filtering networks (`src/f2s3/descriptor_model.py`, `src/f2s3/filtering_model.py`, `src/f2s3/core.py`). Installed as a CUDA build (`torch`, `torchvision`, `torchaudio`).
- Open3D (`open3d ~=0.15`) - Point cloud IO (`.ply`) and geometry operations (`src/f2s3/data.py`, `src/f2s3/core.py`)
- None detected - No test framework configured, no `tests/` directory, no test files.
- setuptools >=64 + setuptools_scm >=8 + wheel - Build backend (`pyproject.toml` `[build-system]`)
- cmake + swig - Build-time requirements for the C++ extension dependencies (`pyproject.toml` `[build-system].requires`, `docs/installation.md`)
- release-please - Automated versioning/changelog (`release-please-config.json`, `.release-please-manifest.json`, `.github/workflows/release-please.yml`)

## Key Dependencies

- `torch` / `torchvision` / `torchaudio` (CUDA cu126 build) - Core inference engine
- `open3d ~=0.15` - Point cloud reading/writing and geometry
- `hnswlib` - Fast approximate nearest-neighbor search in feature space (`src/f2s3/core.py`, `hnswlib.Index(space='l2', dim=64)`)
- `scikit-learn` - `NearestNeighbors` for spatial queries (`src/f2s3/core.py`, `src/f2s3/utils.py`)
- `pc_tiling @ git+https://github.com/gseg-ethz/pc_tiling.git@v0.1.2` - C++ point-cloud tiling (`src/f2s3/core.py`)
- `supervoxel @ git+https://github.com/gseg-ethz/supervoxel.git@v0.1.2` - C++ supervoxel segmentation (`src/f2s3/core.py`)
- `numpy` - Array math throughout
- `tqdm` - Progress bars (`src/f2s3/core.py`)
- `coloredlogs` - Colorized logging setup (`src/f2s3/core.py`)
- `glob2` / `glob` - File discovery
- `pybind11` - C++ binding support (transitive build aid for extensions)
- `plotly` - Imported in `src/f2s3/core.py` (`from plotly.graph_objs.icicle import Pathbar`); not listed in `pyproject.toml` dependencies (undeclared, likely pulled in transitively)

## Configuration

- No application-level config files or `.env` usage - all runtime options are passed via CLI flags (`src/f2s3/cli.py`).
- Build-time environment variables: `SETUPTOOLS_SCM_PRETEND_VERSION` (Dockerfile), `TORCH_INDEX_URL`, `F2S3_VERSION` (Docker build args), `F2S3_IMAGE` (`docker-compose.yaml`).
- PyTorch performance toggles set in code: `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True` (`src/f2s3/core.py` ~line 596).
- `pyproject.toml` - Project metadata, dependencies, setuptools_scm version config, console script entry point (`f2s3 = "f2s3.cli:main"`)
- `setup.py` - Minimal shim calling `setup()` for setuptools_scm
- `Dockerfile` - Multi-stage CUDA build (builder compiles wheels, runtime installs from wheels)
- Version is generated into `src/f2s3/_version.py` by setuptools_scm (not tracked).

## Platform Requirements

- Ubuntu 20.04 / 22.04 / 24.04 LTS (native or WSL2), per `README.md`
- System packages: `python3-dev`, `swig`, `cmake`, `libpcl-dev` (`docs/installation.md`)
- NVIDIA GPU with CUDA support + matching CUDA toolkit
- Docker image `ghcr.io/gseg-ethz/f2s3_pc_deformation_monitoring` (CUDA 12.6 runtime, Ubuntu 24.04, `linux/amd64`)
- Requires `--gpus all` (NVIDIA Container Toolkit); `docker-compose.yaml` sets `gpus: all`
- Distributed as a CLI entrypoint (`ENTRYPOINT ["f2s3"]`)

## Conventions

## Naming Patterns

- Lowercase, single word or `snake_case` module names: `core.py`, `utils.py`, `data.py`, `cli.py`, `descriptor_model.py`, `filtering_model.py`.
- Model definition modules are suffixed `_model`: `descriptor_model.py`, `filtering_model.py`.
- `snake_case` for all functions: `transform_point_cloud`, `compute_c2c`, `kabsch_transformation_estimation` (`src/f2s3/utils.py`).
- Module-private helpers are prefixed with a single underscore: `_radius_from_resolution`, `_median_nn_resolution` (`src/f2s3/core.py:48`, `:61`).
- Instance-private methods also use a leading underscore: `_compute_resolution`, `_forward` (`src/f2s3/core.py:569`, `src/f2s3/descriptor_model.py:88`).
- `snake_case` for locals and attributes: `median_resolution`, `supervoxel_radius`, `local_features_s` (`src/f2s3/core.py`).
- Source/target pairs use `_s` / `_t` suffixes consistently: `pcd_s`/`pcd_t`, `path_s`/`path_t`, `local_features_s`/`local_features_t` (`src/f2s3/core.py:208`).
- Short math-domain names mirror the underlying math/paper notation (acceptable in numeric code): `M`, `efC`, `efS` for HNSW params (`src/f2s3/core.py:386`); `x1`, `x2`, `R`, `t`, `u`, `s`, `v`, `cov_mat` in `kabsch_transformation_estimation` (`src/f2s3/utils.py:43`).
- Module-level constants are `UPPER_SNAKE` with a leading underscore when private: `_EPS = 1e-6` (`src/f2s3/data.py:6`).
- `PascalCase` for classes: `F2S3RunSettings`, `PointCloudTile` (`src/f2s3/core.py`), `FeatureExtractionDataset` (`src/f2s3/data.py`).
- Neural-network modules keep their original research/paper names: `STN3d`, `PointNetFeature`, `PointCN`, `FilteringNetwork`.

## Code Style

- No formatter is active. `black` is mentioned only in a commented-out `[project.optional-dependencies]` block in `pyproject.toml`.
- 4-space indentation throughout.
- Indentation/alignment is **inconsistent**: some multi-line calls align arguments under the opening paren (`src/f2s3/core.py:273`), others use hanging indents. Spacing around `=` in kwargs is inconsistent (e.g. `normalize_w = True, eps = 1e-7` in `src/f2s3/utils.py:43` vs `eps=1e-3` in `src/f2s3/filtering_model.py:13`).
- Line length is not enforced; many lines exceed 100 chars (e.g. log/format strings in `src/f2s3/core.py:295`, `:699`).
- **Convention to follow for new code:** Use `snake_case`, 4-space indent, no space around `=` in keyword arguments, and keep lines under ~100 chars. Prefer matching the cleaner style in `core.py` over the older `utils.py`/`filtering_model.py` style.
- None configured. `flake8` appears only in a commented-out dependency block.

## Import Organization

- Grouping is loose and not blank-line separated everywhere (see `src/f2s3/core.py:19-45`).
- Intra-package imports always use **explicit relative imports** (`from .core import ...`, `from .utils import ...`).
- There is a stray/unused import: `from plotly.graph_objs.icicle import Pathbar` (`src/f2s3/core.py:35`) — do not imitate; remove such imports.
- `data.py` references `logging` in `reset_seed` (`src/f2s3/data.py:105`) without importing it — a latent bug; always import what you use.

## Error Handling

- Input validation is centralized in the `F2S3RunSettings` dataclass `__post_init__` (`src/f2s3/core.py:126`), raising specific built-in exceptions with f-string messages:
- The main per-tile loop wraps processing in a broad `try/except Exception as e` and logs `logging.error(f"Error processing tile {tile_nr}: {e}")`, then continues to the next tile (`src/f2s3/core.py:692`). This favors pipeline robustness (one bad tile does not abort the run) over fail-fast.
- Numerical guards use try/except around `torch.svd` to fall back to identity rotation when SVD fails to converge (`src/f2s3/utils.py:93-102`).
- `assert` is used for tensor-shape contracts in model code: `assert x.dim() == 4 and x.shape[1] == 1` (`src/f2s3/filtering_model.py:51`).
- **Convention:** Raise specific built-in exceptions with descriptive f-string messages for user/config errors; use broad `try/except` + log + continue only for per-item loop resilience.

## Logging

- Calls use the root logger directly: `logging.info(...)`, `logging.warning(...)`, `logging.error(...)` (not a module-scoped `logger = logging.getLogger(__name__)`).
- Most informational logging is gated behind `if self.verbose:` (`src/f2s3/core.py:312`, `:393`).
- Timing is logged frequently: capture `start_time = time.time()` then log elapsed seconds with `'... {:.2f} s'.format(...)`.
- **Inconsistency:** mix of `.format()` and f-strings for log messages. Prefer f-strings for new code: `logging.info(f"... {value:.2f} s")`.
- **Convention:** Use module-level `logging` calls; gate verbose detail behind a `verbose` flag; always log elapsed time for long-running stages.

## Comments

- Explain non-obvious algorithmic steps and reference the paper equation: `# eq. 3`, `# eq. 4`, `# e.q. 5` (`src/f2s3/data.py:47-62`).
- Document tuning parameters and their tradeoffs inline (e.g. HNSW recall vs. runtime at `src/f2s3/core.py:384`; TF32 / cudnn.benchmark rationale at `src/f2s3/core.py:593-598`).
- Commented-out legacy code blocks exist (`src/f2s3/core.py:218-222`, `:619-621`). **Do not add new commented-out code**; delete dead code instead.
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

- Defaults are provided generously for tunable scientific parameters: `kabsch_transformation_estimation(x1, x2, weights=None, normalize_w=True, eps=1e-7, best_k=0, w_threshold=0)` (`src/f2s3/utils.py:43`).
- The CLI uses `argparse.SUPPRESS` defaults so the dataclass owns the real default values (`src/f2s3/cli.py:23-35`).
- Functions return plain tuples of tensors/arrays (`kabsch_transformation_estimation` returns 4-tuple, `src/f2s3/utils.py:116`).
- Model forwards return tuples whose arity varies by flag (`src/f2s3/descriptor_model.py:115-131`) — caller unpacks with `_` placeholders (`src/f2s3/core.py:340`).
- Structured results use plain `dict` with string keys: `filter_input` returns `{'scores', 'rot_est', 'trans_est', 'robust_estimate'}` (`src/f2s3/filtering_model.py:84-90`).

## Module & Package Design

- The package `__init__.py` declares an explicit `__all__ = ["__version__", "cli", "core"]` and imports the version + submodules (`src/f2s3/__init__.py`).
- Version is derived dynamically by `setuptools_scm` into the generated `src/f2s3/_version.py` (do not edit by hand).
- CLI entry is `f2s3.cli:main` registered as a console script in `pyproject.toml` (`[project.scripts]`).
- `main()` builds an `argparse` parser, constructs `F2S3RunSettings(**vars(args))`, and runs the pipeline inside `torch.no_grad()` (`src/f2s3/cli.py:54-61`).

## Configuration & Settings

- All run configuration flows through the single `F2S3RunSettings` dataclass (`src/f2s3/core.py:108`). It validates inputs, creates output/tiled-data directories, configures logging, and persists the resolved arguments to `{results_dir}/command_line_args.txt` via a custom `__repr__` (`src/f2s3/core.py:180`).
- **Convention:** Add new run-time options as typed fields on `F2S3RunSettings` with a corresponding `argparse` flag (using `default=argparse.SUPPRESS`) in `cli.py`, and validate them in `__post_init__`.

## Device / Tensor Conventions

- CUDA is assumed available; tensors are moved with `.cuda()` directly (not a configurable `device`) — e.g. `batch_feat = batch.squeeze(0).cuda()` (`src/f2s3/core.py:339`), `feature_descriptor.cuda()` (`src/f2s3/core.py:608`).
- Inference is wrapped in `torch.no_grad()` at the entry point and models are set to `.eval()` after loading weights (`src/f2s3/core.py:609`, `:614`).
- GPU hygiene: explicit `torch.cuda.empty_cache()` and `gc.collect()` after heavy stages (`src/f2s3/core.py:358-359`, `:468-469`).
- Performance backends are enabled globally at pipeline start: TF32 matmul/cudnn and `cudnn.benchmark` (`src/f2s3/core.py:596-598`).

## Architecture

## System Overview

```text

```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| CLI parser | Parse argv, build `F2S3RunSettings`, wrap run in `torch.no_grad()` | `src/f2s3/cli.py` |
| `F2S3RunSettings` | Validate paths/args, create result dirs, configure logging, persist args | `src/f2s3/core.py:108` |
| `feature_based_deformation_analysis` | Load models, tile clouds, loop tiles, aggregate timing | `src/f2s3/core.py:582` |
| `PointCloudTile` | Per-tile orchestration of features → supervoxels → correspondences → filtering | `src/f2s3/core.py:184` |
| `PointNetFeature` | Deep local 3D feature descriptor (DIP/PointNet, 64-dim) | `src/f2s3/descriptor_model.py:56` |
| `STN3d` | Spatial transformer (T-Net) for input alignment | `src/f2s3/descriptor_model.py:5` |
| `FilteringNetwork` | Outlier-detection network producing per-correspondence inlier weights | `src/f2s3/filtering_model.py:27` |
| `FeatureExtractionDataset` | Builds normalized local patches around each point for the descriptor | `src/f2s3/data.py:8` |
| `utils` | Kabsch transform estimation, residuals, point transforms, C2C distance | `src/f2s3/utils.py` |
| `pc_tiling` | External C++ extension: voxel-grid filter + overlapping tiling | external dep (`pc_tiling`) |
| `supervoxel` | External C++ extension: boundary-preserving supervoxel segmentation | external dep (`supervoxel`) |

## Pattern Overview

- Single-process, single-GPU, no concurrency between tiles (tiles processed serially in a `for` loop).
- Stateful pipeline object: `PointCloudTile` accumulates results as instance attributes (`self.local_features_s`, `self.supervoxels`, `self.correspondences`) consumed by later methods.
- Strict method ordering: `compute_local_features` → `compute_supervoxels` → `compute_correspondences` → `filter_correspondences`. Calling out of order breaks because attributes won't exist.
- Heavy reliance on external C++ extensions (`pc_tiling`, `supervoxel`) and native libraries (Open3D, hnswlib, PyTorch/CUDA).
- File-system as the inter-stage boundary: tiling writes `.ply` tiles to disk; the pipeline reads them back per tile; final outputs are `.txt` files.

## Layers

- Purpose: Translate command-line arguments into a validated settings object.
- Location: `src/f2s3/cli.py`, `F2S3RunSettings` in `src/f2s3/core.py:108`
- Contains: argparse setup, dataclass validation in `__post_init__`, logger setup, dir creation.
- Depends on: `core` module.
- Used by: shell entry point `f2s3` (`pyproject.toml` `[project.scripts]`).
- Purpose: Drive the whole analysis — model loading, tiling, per-tile loop.
- Location: `feature_based_deformation_analysis` in `src/f2s3/core.py:582`
- Contains: pretrained model loading, TF32/cuDNN backend flags, overlap/radius derivation, tile iteration with per-tile try/except.
- Depends on: `PointCloudTile`, model classes, `pc_tiling`.
- Used by: `cli.main`.
- Purpose: Run the four-stage deformation analysis for one tile.
- Location: `PointCloudTile` in `src/f2s3/core.py:184`
- Contains: feature extraction, supervoxel computation, ANN correspondence search, outlier filtering, output writing.
- Depends on: descriptor model, filtering model, dataset, `supervoxel`, `hnswlib`, `utils`.
- Used by: orchestration loop.
- Purpose: Neural network definitions (loaded with pretrained weights, used in eval mode).
- Location: `src/f2s3/descriptor_model.py`, `src/f2s3/filtering_model.py`
- Contains: PyTorch `nn.Module` subclasses.
- Depends on: `torch`, `utils.kabsch_transformation_estimation` (filtering model only).
- Used by: orchestration (instantiation/weight loading) and `PointCloudTile` (inference).
- Purpose: Geometric math and patch-construction helpers.
- Location: `src/f2s3/utils.py`, `src/f2s3/data.py`
- Depends on: `torch`, `numpy`, `sklearn`, `open3d`.
- Used by: models and `PointCloudTile`.

## Data Flow

### Primary Request Path (full run from raw clouds)

### Pre-Tiled Path (`--start_from_tiled_data`)

- No global mutable state in the package's own code beyond PyTorch backend flags (`torch.backends.*`) set once per run.
- Per-tile state lives entirely on the `PointCloudTile` instance and is discarded when the loop advances; `gc.collect()` and `torch.cuda.empty_cache()` are called explicitly between stages.

## Key Abstractions

- Purpose: Encapsulates a single tile's full deformation-analysis lifecycle.
- Examples: `src/f2s3/core.py:184`
- Pattern: Stateful sequential processor; methods mutate `self` and must be called in fixed order.
- Purpose: Single typed source of truth for all run configuration.
- Examples: `src/f2s3/core.py:108`
- Pattern: Validated config object — all validation/side effects in `__post_init__`.
- Purpose: On-the-fly construction of normalized local patches per point, batched for GPU.
- Examples: `src/f2s3/data.py:8`
- Pattern: Each `__getitem__` returns a whole batch (`points_per_batch` patches); DataLoader `batch_size` is fixed to 1.

## Entry Points

- Location: `src/f2s3/cli.py:8` (`main`), registered in `pyproject.toml` `[project.scripts]`.
- Triggers: user shell invocation `f2s3 --source_cloud ... --target_cloud ...`.
- Responsibilities: parse args, build settings, launch analysis under `no_grad`.
- Location: `Dockerfile:72`
- Triggers: container run.
- Responsibilities: invokes the same console script inside the CUDA runtime image.

## Architectural Constraints

- **GPU required:** All model inference calls `.cuda()` unconditionally (`src/f2s3/core.py:339, 349, 450`); there is no CPU fallback. An NVIDIA GPU with CUDA is mandatory.
- **Threading:** Tiles processed strictly serially. Parallelism exists only inside (a) the PyTorch `DataLoader` (`num_workers=6`, `src/f2s3/core.py:322`) and (b) hnswlib (`num_threads=16`, `src/f2s3/core.py:391`). These worker counts are hard-coded.
- **Global backend flags:** `torch.backends.cuda.matmul.allow_tf32`, `cudnn.allow_tf32`, and `cudnn.benchmark` are set globally at run start (`src/f2s3/core.py:596-598`) — process-wide side effects.
- **Method ordering coupling:** `PointCloudTile` methods depend on attributes set by prior methods; there is no guard if called out of order.
- **External C++ extensions:** `pc_tiling` and `supervoxel` are git-pinned native deps (`pyproject.toml:41-42`); the pipeline cannot run without them compiled.
- **Filesystem coupling:** Stages communicate partly via on-disk `.ply` tiles and an `overlap/` subfolder convention (`src/f2s3/core.py:211-216`).

## Anti-Patterns

### Broad per-tile exception swallowing

### Sequential model loop instead of batched inference

### In-place mutation of shared arrays

## Error Handling

- Config errors raise typed exceptions: `FileNotFoundError`, `NotADirectoryError`, `ValueError` (`src/f2s3/core.py:129-166`).
- Per-tile errors logged via `logging.error` and skipped (`src/f2s3/core.py:692-693`).
- Missing target tile logged as a warning and skipped (`src/f2s3/core.py:694-695`).
- Kabsch SVD non-convergence handled by returning identity rotation / zero translation (`src/f2s3/utils.py:93-102`).

## Cross-Cutting Concerns

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
