<!-- refreshed: 2026-06-11 -->
# Architecture

**Analysis Date:** 2026-06-11

## System Overview

F2S3 is a CLI-driven, single-process pipeline that estimates a 3D displacement vector field between two point cloud epochs (source and target) using deep-learned local features and supervoxel-constrained outlier filtering. The pipeline runs tile-by-tile on a single CUDA GPU.

```text
┌─────────────────────────────────────────────────────────────┐
│                       CLI Entry Layer                        │
│   argparse + F2S3RunSettings validation                      │
│   `src/f2s3/cli.py`            `src/f2s3/core.py`            │
└──────────────────────────────┬──────────────────────────────┘
                               │  F2S3RunSettings
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Orchestration / Pipeline                    │
│   feature_based_deformation_analysis()                       │
│   - loads pretrained models                                  │
│   - tiles clouds (pc_tiling C++ ext)                         │
│   - loops over tiles                                         │
│   `src/f2s3/core.py:582`                                     │
└──────────────────────────────┬──────────────────────────────┘
                               │  per-tile
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   PointCloudTile (per tile)                  │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ local        │ supervoxels  │ correspond-  │ filter         │
│ features     │ (C++ ext)    │ ences (HNSW) │ correspondences│
│ `core.py:299`│ `core.py:244`│ `core.py:377`│ `core.py:430`  │
└──────┬───────┴──────┬───────┴──────┬───────┴───────┬────────┘
       ▼              ▼              ▼               ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐
│ Descriptor   │ │supervoxel│ │ hnswlib  │ │ Filtering        │
│ model        │ │ C++ ext  │ │ ANN      │ │ network +        │
│ PointNet     │ │          │ │ index    │ │ Kabsch (utils)   │
│ descriptor_  │ │          │ │          │ │ filtering_model, │
│ model.py     │ │          │ │          │ │ utils.py         │
└──────────────┘ └──────────┘ └──────────┘ └──────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Output: per-tile displacement magnitude .txt files          │
│  `{results_dir}/output/results/displacement_magnitude_*.txt` │
└─────────────────────────────────────────────────────────────┘
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

**Overall:** Sequential batch-processing pipeline (ETL-style) with a per-tile object that owns each stage's state.

**Key Characteristics:**
- Single-process, single-GPU, no concurrency between tiles (tiles processed serially in a `for` loop).
- Stateful pipeline object: `PointCloudTile` accumulates results as instance attributes (`self.local_features_s`, `self.supervoxels`, `self.correspondences`) consumed by later methods.
- Strict method ordering: `compute_local_features` → `compute_supervoxels` → `compute_correspondences` → `filter_correspondences`. Calling out of order breaks because attributes won't exist.
- Heavy reliance on external C++ extensions (`pc_tiling`, `supervoxel`) and native libraries (Open3D, hnswlib, PyTorch/CUDA).
- File-system as the inter-stage boundary: tiling writes `.ply` tiles to disk; the pipeline reads them back per tile; final outputs are `.txt` files.

## Layers

**CLI / Configuration Layer:**
- Purpose: Translate command-line arguments into a validated settings object.
- Location: `src/f2s3/cli.py`, `F2S3RunSettings` in `src/f2s3/core.py:108`
- Contains: argparse setup, dataclass validation in `__post_init__`, logger setup, dir creation.
- Depends on: `core` module.
- Used by: shell entry point `f2s3` (`pyproject.toml` `[project.scripts]`).

**Orchestration Layer:**
- Purpose: Drive the whole analysis — model loading, tiling, per-tile loop.
- Location: `feature_based_deformation_analysis` in `src/f2s3/core.py:582`
- Contains: pretrained model loading, TF32/cuDNN backend flags, overlap/radius derivation, tile iteration with per-tile try/except.
- Depends on: `PointCloudTile`, model classes, `pc_tiling`.
- Used by: `cli.main`.

**Per-Tile Processing Layer:**
- Purpose: Run the four-stage deformation analysis for one tile.
- Location: `PointCloudTile` in `src/f2s3/core.py:184`
- Contains: feature extraction, supervoxel computation, ANN correspondence search, outlier filtering, output writing.
- Depends on: descriptor model, filtering model, dataset, `supervoxel`, `hnswlib`, `utils`.
- Used by: orchestration loop.

**Model Layer:**
- Purpose: Neural network definitions (loaded with pretrained weights, used in eval mode).
- Location: `src/f2s3/descriptor_model.py`, `src/f2s3/filtering_model.py`
- Contains: PyTorch `nn.Module` subclasses.
- Depends on: `torch`, `utils.kabsch_transformation_estimation` (filtering model only).
- Used by: orchestration (instantiation/weight loading) and `PointCloudTile` (inference).

**Utility / Data Layer:**
- Purpose: Geometric math and patch-construction helpers.
- Location: `src/f2s3/utils.py`, `src/f2s3/data.py`
- Depends on: `torch`, `numpy`, `sklearn`, `open3d`.
- Used by: models and `PointCloudTile`.

## Data Flow

### Primary Request Path (full run from raw clouds)

1. CLI parses args and constructs `F2S3RunSettings` (`src/f2s3/cli.py:54-56`); validation/dir setup happens in `F2S3RunSettings.__post_init__` (`src/f2s3/core.py:126`).
2. Run is wrapped in `torch.no_grad()` and `feature_based_deformation_analysis` is invoked (`src/f2s3/cli.py:60-61`).
3. TF32/cuDNN backends enabled, pretrained descriptor + filtering models loaded to CUDA in eval mode (`src/f2s3/core.py:596-614`).
4. Tile overlap derived from user radius or global median NN resolution (`src/f2s3/core.py:628-644`).
5. `pc_tiling.tile_point_clouds` writes overlapping source/target `.ply` tiles to `args.tiled_data` (`src/f2s3/core.py:647`).
6. Tile list globbed and sorted; loop over each `source_tile_*` / matching `target_tile_*` (`src/f2s3/core.py:661-695`).
7. Per tile, `PointCloudTile` is constructed (reads tile + overlap clouds, computes median resolution) (`src/f2s3/core.py:677`).
8. `compute_local_features` → batches patches through descriptor on GPU (`src/f2s3/core.py:680`, `data.py:20`).
9. `compute_supervoxels` → boundary-preserving segmentation via `supervoxel` C++ ext (`src/f2s3/core.py:681`).
10. `compute_correspondences` → hnswlib L2 ANN search in 64-dim feature space (`src/f2s3/core.py:687`).
11. `filter_correspondences` → per-supervoxel filtering network + Kabsch, writes `.txt` results (`src/f2s3/core.py:688`).

### Pre-Tiled Path (`--start_from_tiled_data`)

1. CLI sets `start_from_tiled_data=True`; source/target clouds ignored, `tiled_data` dir required (`src/f2s3/core.py:134`).
2. Tiling step skipped; loop reads existing tiles directly (`src/f2s3/core.py:616, 661`).

**State Management:**
- No global mutable state in the package's own code beyond PyTorch backend flags (`torch.backends.*`) set once per run.
- Per-tile state lives entirely on the `PointCloudTile` instance and is discarded when the loop advances; `gc.collect()` and `torch.cuda.empty_cache()` are called explicitly between stages.

## Key Abstractions

**`PointCloudTile`:**
- Purpose: Encapsulates a single tile's full deformation-analysis lifecycle.
- Examples: `src/f2s3/core.py:184`
- Pattern: Stateful sequential processor; methods mutate `self` and must be called in fixed order.

**`F2S3RunSettings` (dataclass):**
- Purpose: Single typed source of truth for all run configuration.
- Examples: `src/f2s3/core.py:108`
- Pattern: Validated config object — all validation/side effects in `__post_init__`.

**`FeatureExtractionDataset` (torch Dataset):**
- Purpose: On-the-fly construction of normalized local patches per point, batched for GPU.
- Examples: `src/f2s3/data.py:8`
- Pattern: Each `__getitem__` returns a whole batch (`points_per_batch` patches); DataLoader `batch_size` is fixed to 1.

## Entry Points

**`f2s3` console script:**
- Location: `src/f2s3/cli.py:8` (`main`), registered in `pyproject.toml` `[project.scripts]`.
- Triggers: user shell invocation `f2s3 --source_cloud ... --target_cloud ...`.
- Responsibilities: parse args, build settings, launch analysis under `no_grad`.

**Docker `ENTRYPOINT ["f2s3"]`:**
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

**What happens:** The tile loop wraps each tile in `try/except Exception` and only logs the error, then continues (`src/f2s3/core.py:692-693`).
**Why it's wrong:** Real failures (OOM, corrupt data, model errors) are hidden as a single log line; partial/empty result sets can be produced silently.
**Do this instead:** Catch specific, recoverable exceptions; let unexpected ones surface, or record failed tiles explicitly for a post-run summary.

### Sequential model loop instead of batched inference

**What happens:** `compute_local_features` loops over the source dataloader and then the target dataloader separately, appending to Python lists (`src/f2s3/core.py:338-352`).
**Why it's wrong:** Two near-identical inference loops duplicate logic and miss opportunities to overlap source/target work.
**Do this instead:** Factor a single `_extract(dataloader)` helper and call it for source and target.

### In-place mutation of shared arrays

**What happens:** `filter_correspondences` mutates `supervoxel_coords` / `supervoxel_data` in place when refining (`src/f2s3/core.py:453-459`), and reuses `save_coords` for both raw and C2C paths.
**Why it's wrong:** Aliasing between `supervoxel_data` and `supervoxel_coords` makes the data flow hard to reason about and risks subtle corruption.
**Do this instead:** Copy before mutating refined coordinates; keep raw and derived arrays separate.

## Error Handling

**Strategy:** Fail-fast on configuration (raise in `F2S3RunSettings.__post_init__`), best-effort on per-tile processing (log and continue).

**Patterns:**
- Config errors raise typed exceptions: `FileNotFoundError`, `NotADirectoryError`, `ValueError` (`src/f2s3/core.py:129-166`).
- Per-tile errors logged via `logging.error` and skipped (`src/f2s3/core.py:692-693`).
- Missing target tile logged as a warning and skipped (`src/f2s3/core.py:694-695`).
- Kabsch SVD non-convergence handled by returning identity rotation / zero translation (`src/f2s3/utils.py:93-102`).

## Cross-Cutting Concerns

**Logging:** Python `logging` + `coloredlogs` installed in `F2S3RunSettings.__post_init__` (`src/f2s3/core.py:170-172`); INFO-level messages throughout, gated by `args.verbose` for detail and `tqdm` progress bars.
**Validation:** Centralized in `F2S3RunSettings.__post_init__`; no per-stage re-validation.
**Authentication:** Not applicable (offline CLI tool).
**Configuration persistence:** Effective settings written to `{results_dir}/command_line_args.txt` at startup (`src/f2s3/core.py:174-178`).
**Memory hygiene:** Explicit `gc.collect()` and `torch.cuda.empty_cache()` after heavy stages (`src/f2s3/core.py:297, 358-359, 468-469`).

---

*Architecture analysis: 2026-06-11*
