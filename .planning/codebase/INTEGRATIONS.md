# External Integrations

**Analysis Date:** 2026-06-11

## APIs & External Services

F2S3 is a self-contained, offline command-line tool. It makes **no runtime calls to external APIs or hosted services**. All processing runs locally on the GPU against on-disk point cloud files.

**Build-time / distribution services only:**
- **PyTorch wheel index** - `https://download.pytorch.org/whl/cu126`
  - Used at install/build time to fetch CUDA-enabled `torch`/`torchvision`/`torchaudio` (`Dockerfile`, `docker-compose.yaml`, `.github/workflows/docker-ghcr.yml`)
  - Auth: none
- **GitHub (source dependencies)** - `pc_tiling` and `supervoxel` installed via `git+https://github.com/gseg-ethz/...` (`pyproject.toml`)
  - Auth: none (public repos)
- **GitHub Container Registry (GHCR)** - `ghcr.io/gseg-ethz/f2s3_pc_deformation_monitoring`
  - Image publication target (`docker-compose.yaml`, `.github/workflows/docker-ghcr.yml`)
  - Auth: `${{ secrets.GITHUB_TOKEN }}` in CI

## Data Storage

**Databases:**
- None - No database of any kind. State lives entirely in the filesystem.

**File Storage:**
- Local filesystem only
  - Input/output point clouds in `.ply` format, read/written via Open3D `o3d.io.read_point_cloud` / `write_point_cloud` (`src/f2s3/core.py`, `src/f2s3/data.py`)
  - Sample data under `data/` (e.g. `data/20210924_THC_centered.ply`, `data/_sample_folder`)
  - Results written to a user-specified `--results_dir`; tiled intermediate data under `--tiled_data` (`src/f2s3/cli.py`)
  - In Docker, `./data` is bind-mounted to `/workspace/data` (`docker-compose.yaml`)

**Caching:**
- None (process-local only; `torch.cuda.empty_cache()` is GPU memory management, not a cache service)

## Authentication & Identity

**Auth Provider:**
- Not applicable - No users, sessions, or auth. CLI tool run directly by the operator.

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry/Rollbar/etc.)

**Logs:**
- Local stdout/stderr via Python `logging` configured with `coloredlogs` (`src/f2s3/core.py`). `--verbose` CLI flag increases detail (`src/f2s3/cli.py`).

## CI/CD & Deployment

**Hosting:**
- Container image distributed via GHCR; executed on any CUDA-capable host with the NVIDIA Container Toolkit.

**CI Pipeline:**
- GitHub Actions:
  - `.github/workflows/docker-ghcr.yml` - Builds and pushes the Docker image on `v*.*.*` tags and manual dispatch (`docker/build-push-action@v6`, GHA build cache)
  - `.github/workflows/release-please.yml` - Automated release/changelog management (release-please)

## Environment Configuration

**Required env vars:**
- None required at application runtime (all options are CLI flags).
- Build/deploy-time only: `F2S3_IMAGE`, `TORCH_INDEX_URL`, `F2S3_VERSION` (`docker-compose.yaml`), `SETUPTOOLS_SCM_PRETEND_VERSION` (`Dockerfile`).

**Secrets location:**
- No application secrets. CI uses the automatically provided `GITHUB_TOKEN` for GHCR login. No `.env` file is present or read by the application.

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Bundled Models (not network integrations)

- Pretrained weights ship inside the package and are loaded from package resources, not downloaded:
  - `src/f2s3/pretrained_models/feature_descriptor/model_best.pth`
  - `src/f2s3/pretrained_models/outlier_filtering/model_best.pt`
  - Loaded via `importlib.resources` in `src/f2s3/core.py`.

---

*Integration audit: 2026-06-11*
