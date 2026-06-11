# Codebase Structure

**Analysis Date:** 2026-06-11

## Directory Layout

```
F2S3/
├── src/f2s3/                  # Python package (the actual application)
│   ├── __init__.py           # Exposes __version__, cli, core
│   ├── _version.py           # setuptools_scm-generated version (do not edit)
│   ├── cli.py                # argparse entry point (console script `f2s3`)
│   ├── core.py               # Pipeline orchestration + PointCloudTile + settings
│   ├── descriptor_model.py   # PointNet/DIP feature descriptor network
│   ├── filtering_model.py    # Outlier-filtering network
│   ├── data.py               # FeatureExtractionDataset (patch construction)
│   ├── utils.py              # Kabsch, residuals, transforms, C2C
│   └── pretrained_models/    # Bundled model weights (package data)
│       ├── feature_descriptor/model_best.pth
│       └── outlier_filtering/model_best.pt
├── docs/                     # Installation & development docs
│   ├── installation.md
│   ├── development.md
│   └── assets/F2S3_pipeline.PNG
├── data/                     # Sample data folder (mostly gitignored content)
│   └── _sample_folder/
├── results/                  # Local experiment outputs (gitignored; benchmarking runs)
├── build/                    # setuptools build artifacts (generated; not source)
├── _old/                     # Legacy scripts kept for reference (install_pcl.sh)
├── f2s3.egg-info/            # setuptools metadata (generated)
├── .github/workflows/        # CI: docker-ghcr.yml, release-please.yml
├── Dockerfile                # Multi-stage CUDA build/runtime image
├── docker-compose.yaml       # Compose service definition
├── pyproject.toml            # Build system, deps, scripts, version config
├── setup.py                  # Thin shim for setuptools
├── release-please-config.json + .release-please-manifest.json  # Release automation
├── CHANGELOG.md
├── README.md
└── LICENSE
```

## Directory Purposes

**`src/f2s3/`:**
- Purpose: The entire importable/installable application.
- Contains: 7 hand-written Python modules + generated `_version.py` + bundled model weights.
- Key files: `core.py` (701 lines, the pipeline), `cli.py` (entry point).

**`src/f2s3/pretrained_models/`:**
- Purpose: Ship pretrained network weights inside the wheel.
- Contains: `.pth` (descriptor) and `.pt` (filtering) checkpoints, declared as package data in `pyproject.toml:51-56`.
- Key files: `feature_descriptor/model_best.pth`, `outlier_filtering/model_best.pt`.

**`docs/`:**
- Purpose: User and developer documentation referenced from README.
- Key files: `installation.md`, `development.md`.

**`results/`:**
- Purpose: Output of local runs/benchmarks (e.g., `TCH_original`, `TCH_nochange`, `TCH_improvements_withPara`). Each holds `command_line_args.txt`, `tiled_data/` (`.ply` tiles), and `output/`.
- Generated: Yes. Committed: No (gitignored; present only in working tree).

**`build/`, `f2s3.egg-info/`:**
- Purpose: setuptools build/metadata artifacts. Generated. Not source of truth — do not edit.

**`_old/`:**
- Purpose: Archived legacy scripts (`scripts/install_pcl.sh`). Reference only; not part of the package.

## Key File Locations

**Entry Points:**
- `src/f2s3/cli.py`: `main()` — console script `f2s3`, registered in `pyproject.toml:82-83`.
- `Dockerfile:72`: `ENTRYPOINT ["f2s3"]`.

**Configuration:**
- `pyproject.toml`: dependencies, build backend, package data, version scheme, console script.
- `setup.py`: minimal setuptools shim.
- `F2S3RunSettings` (`src/f2s3/core.py:108`): runtime configuration dataclass.
- `.git_archival.txt`, `.gitattributes`: support setuptools_scm versioning from archives.

**Core Logic:**
- `src/f2s3/core.py`: orchestration (`feature_based_deformation_analysis`, `src/f2s3/core.py:582`) and per-tile processor (`PointCloudTile`, `src/f2s3/core.py:184`).
- `src/f2s3/descriptor_model.py`, `src/f2s3/filtering_model.py`: neural networks.
- `src/f2s3/data.py`: patch dataset.
- `src/f2s3/utils.py`: geometry/math helpers.

**Testing:**
- None present. No `tests/` directory, no `*_test.py` / `test_*.py` files, no test runner configured in `pyproject.toml`.

## Naming Conventions

**Files:**
- Module files: lowercase `snake_case.py` (`descriptor_model.py`, `filtering_model.py`).
- Domain-suffixed names: `*_model.py` for network definitions.

**Directories:**
- Package and source dirs: lowercase (`src/f2s3`, `pretrained_models`).
- Leading underscore for non-package / sample / archived dirs (`_old`, `_sample_folder`).

**Classes:**
- `PascalCase` (`PointCloudTile`, `F2S3RunSettings`, `PointNetFeature`, `FilteringNetwork`, `FeatureExtractionDataset`, `STN3d`).

**Functions:**
- `snake_case` (`feature_based_deformation_analysis`, `compute_local_features`, `kabsch_transformation_estimation`).
- Private/helper functions prefixed with `_` (`_radius_from_resolution`, `_median_nn_resolution`, `_compute_resolution`).

**Generated tile files (on disk):**
- `source_tile_{NNN}.ply` / `target_tile_{NNN}.ply`, with overlap clouds under an `overlap/` subfolder named `*_overlap.ply` (`src/f2s3/core.py:211-216, 661, 673`).

## Where to Add New Code

**New pipeline stage:**
- Add a method to `PointCloudTile` in `src/f2s3/core.py`, then call it in the correct order within the tile loop (`src/f2s3/core.py:680-688`). Preserve the feature → supervoxel → correspondence → filter ordering.

**New neural network / model:**
- New module `src/f2s3/<name>_model.py` defining `nn.Module` subclass(es). Load weights in `feature_based_deformation_analysis` (`src/f2s3/core.py:600-614`) and add the checkpoint under `src/f2s3/pretrained_models/` plus a `package-data` entry in `pyproject.toml:51-56`.

**New CLI option:**
- Add the `parser.add_argument(...)` in `src/f2s3/cli.py` using `default=argparse.SUPPRESS` (so dataclass defaults apply), then add the corresponding field + validation to `F2S3RunSettings` (`src/f2s3/core.py:108-166`).

**New geometric/math helper:**
- Add to `src/f2s3/utils.py` as a `snake_case` function with a NumPy-style docstring.

**New patch/data transformation:**
- Extend `FeatureExtractionDataset` in `src/f2s3/data.py`.

**Shared dependency:**
- Add to `dependencies` in `pyproject.toml:29-43`. Git-pinned native deps follow the `pkg @ git+https://...@vX.Y.Z` pattern (see `pc_tiling`, `supervoxel`).

## Special Directories

**`src/f2s3/pretrained_models/`:**
- Purpose: bundled model weights. Generated: No (provided artifacts). Committed: Yes (declared as package data).

**`results/`:**
- Purpose: experiment outputs. Generated: Yes. Committed: No (gitignored).

**`build/` and `f2s3.egg-info/`:**
- Purpose: build outputs/metadata. Generated: Yes. Committed: No.

**`_old/`:**
- Purpose: legacy reference scripts. Generated: No. Committed: Yes (but not part of the package).

---

*Structure analysis: 2026-06-11*
