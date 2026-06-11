# Technology Stack

**Analysis Date:** 2026-06-11

## Languages

**Primary:**
- Python >=3.11 - All application logic (`src/f2s3/*.py`). The Docker runtime ships Python 3 from Ubuntu 24.04; compiled bytecode present is `cpython-312`.

**Secondary:**
- C++ - Tiling and supervoxel segmentation. Not vendored in this repo; provided by external Git dependencies `pc_tiling` and `supervoxel`, built at install time via CMake + SWIG bindings.
- CUDA - GPU kernels are consumed through PyTorch (`.cuda()` calls in `src/f2s3/core.py`); no hand-written `.cu` files live in this repository.

## Runtime

**Environment:**
- CPython >=3.11 (declared in `pyproject.toml`, `requires-python = ">=3.11"`)
- NVIDIA GPU with CUDA support is required at runtime (`src/f2s3/core.py` calls `.cuda()` unconditionally)
- CUDA 12.6 toolchain (Docker base image `nvidia/cuda:12.6.3-cudnn-*-ubuntu24.04`, PyTorch wheels from `https://download.pytorch.org/whl/cu126`)

**Package Manager:**
- pip (used in `Dockerfile` and `docs/installation.md`)
- Lockfile: missing - dependencies are unpinned in `pyproject.toml` (only `open3d ~=0.15` and pinned Git tags are version-constrained)

## Frameworks

**Core:**
- PyTorch (`torch`) - Deep learning runtime for feature descriptor and outlier-filtering networks (`src/f2s3/descriptor_model.py`, `src/f2s3/filtering_model.py`, `src/f2s3/core.py`). Installed as a CUDA build (`torch`, `torchvision`, `torchaudio`).
- Open3D (`open3d ~=0.15`) - Point cloud IO (`.ply`) and geometry operations (`src/f2s3/data.py`, `src/f2s3/core.py`)

**Testing:**
- None detected - No test framework configured, no `tests/` directory, no test files.

**Build/Dev:**
- setuptools >=64 + setuptools_scm >=8 + wheel - Build backend (`pyproject.toml` `[build-system]`)
- cmake + swig - Build-time requirements for the C++ extension dependencies (`pyproject.toml` `[build-system].requires`, `docs/installation.md`)
- release-please - Automated versioning/changelog (`release-please-config.json`, `.release-please-manifest.json`, `.github/workflows/release-please.yml`)

## Key Dependencies

**Critical:**
- `torch` / `torchvision` / `torchaudio` (CUDA cu126 build) - Core inference engine
- `open3d ~=0.15` - Point cloud reading/writing and geometry
- `hnswlib` - Fast approximate nearest-neighbor search in feature space (`src/f2s3/core.py`, `hnswlib.Index(space='l2', dim=64)`)
- `scikit-learn` - `NearestNeighbors` for spatial queries (`src/f2s3/core.py`, `src/f2s3/utils.py`)
- `pc_tiling @ git+https://github.com/gseg-ethz/pc_tiling.git@v0.1.2` - C++ point-cloud tiling (`src/f2s3/core.py`)
- `supervoxel @ git+https://github.com/gseg-ethz/supervoxel.git@v0.1.2` - C++ supervoxel segmentation (`src/f2s3/core.py`)

**Infrastructure:**
- `numpy` - Array math throughout
- `tqdm` - Progress bars (`src/f2s3/core.py`)
- `coloredlogs` - Colorized logging setup (`src/f2s3/core.py`)
- `glob2` / `glob` - File discovery
- `pybind11` - C++ binding support (transitive build aid for extensions)
- `plotly` - Imported in `src/f2s3/core.py` (`from plotly.graph_objs.icicle import Pathbar`); not listed in `pyproject.toml` dependencies (undeclared, likely pulled in transitively)

## Configuration

**Environment:**
- No application-level config files or `.env` usage - all runtime options are passed via CLI flags (`src/f2s3/cli.py`).
- Build-time environment variables: `SETUPTOOLS_SCM_PRETEND_VERSION` (Dockerfile), `TORCH_INDEX_URL`, `F2S3_VERSION` (Docker build args), `F2S3_IMAGE` (`docker-compose.yaml`).
- PyTorch performance toggles set in code: `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True` (`src/f2s3/core.py` ~line 596).

**Build:**
- `pyproject.toml` - Project metadata, dependencies, setuptools_scm version config, console script entry point (`f2s3 = "f2s3.cli:main"`)
- `setup.py` - Minimal shim calling `setup()` for setuptools_scm
- `Dockerfile` - Multi-stage CUDA build (builder compiles wheels, runtime installs from wheels)
- Version is generated into `src/f2s3/_version.py` by setuptools_scm (not tracked).

## Platform Requirements

**Development:**
- Ubuntu 20.04 / 22.04 / 24.04 LTS (native or WSL2), per `README.md`
- System packages: `python3-dev`, `swig`, `cmake`, `libpcl-dev` (`docs/installation.md`)
- NVIDIA GPU with CUDA support + matching CUDA toolkit

**Production:**
- Docker image `ghcr.io/gseg-ethz/f2s3_pc_deformation_monitoring` (CUDA 12.6 runtime, Ubuntu 24.04, `linux/amd64`)
- Requires `--gpus all` (NVIDIA Container Toolkit); `docker-compose.yaml` sets `gpus: all`
- Distributed as a CLI entrypoint (`ENTRYPOINT ["f2s3"]`)

---

*Stack analysis: 2026-06-11*
