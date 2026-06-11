# Codebase Concerns

**Analysis Date:** 2026-06-11

This is a CUDA/PyTorch deep-learning pipeline (`f2s3`) for point-cloud deformation
analysis. The package is small (~1,270 lines of Python across `src/f2s3/`) but
carries meaningful debt: a 700-line monolithic `core.py`, no automated tests, no
linting, and several latent correctness bugs. The repo is on the
`compute-performance-improvement` branch with recent commits explicitly noting
"no improvements yet".

## Tech Debt

**Monolithic `core.py` (701 lines):**
- Issue: `src/f2s3/core.py` mixes the run-settings dataclass, the `PointCloudTile`
  domain class (supervoxels, features, correspondences, filtering, I/O), free
  helper functions, and the top-level orchestration loop. `PointCloudTile.filter_correspondences`
  alone is ~140 lines with deeply nested branching over `refine_results`,
  `max_disp_magnitude`, `filter_median_magnitude`, and `fill_gaps_c2c`.
- Files: `src/f2s3/core.py` (esp. lines 430-567, `filter_correspondences`)
- Impact: Hard to test, hard to reason about output-path side effects, easy to
  introduce regressions when changing one flag's behavior.
- Fix approach: Split into `settings.py` (dataclass), `tile.py` (`PointCloudTile`),
  `pipeline.py` (orchestration); extract the four output-writing branches in
  `filter_correspondences` into named helper methods.

**Unused / accidental import of `plotly`:**
- Issue: `from plotly.graph_objs.icicle import Pathbar` at `src/f2s3/core.py:35`
  is never used. `plotly` is not declared in `pyproject.toml` dependencies, so a
  clean install will raise `ModuleNotFoundError` at import time.
- Files: `src/f2s3/core.py:35`, `pyproject.toml:29-43`
- Impact: Import-time crash for any environment without plotly installed; masks a
  missing-dependency problem.
- Fix approach: Delete the import line.

**Commented-out dead code left in place:**
- Issue: Several blocks of replaced logic remain as comments:
  `src/f2s3/core.py:218-222` (old overlap-path loading),
  `src/f2s3/core.py:619-621` (old `resave_point_cloud` call),
  `src/f2s3/core.py:662` (old glob).
- Files: `src/f2s3/core.py`
- Impact: Noise; ambiguity about which path is authoritative.
- Fix approach: Remove dead comments; rely on git history.

**Hardcoded "magic" constants scattered through logic:**
- Issue: Tuning constants are inlined rather than named/configurable: HNSW params
  `M=12, efC=300, efS=300`, `num_threads=16` (`src/f2s3/core.py:386-391`);
  supervoxel size threshold `> 10` (`core.py:287`); minimum-points heuristic
  `> 10` in patch extraction (`src/f2s3/data.py:43`); score threshold `> 0.99999`
  (`core.py:463`); residual gate `< 0.5` and `>= 5` inliers
  (`src/f2s3/filtering_model.py:76`); `30 * median_mag` outlier cutoff
  (`core.py:526`); `n_normals = 30` (`core.py:242`); DataLoader `num_workers=6`
  (`core.py:322-324`).
- Files: `src/f2s3/core.py`, `src/f2s3/data.py`, `src/f2s3/filtering_model.py`
- Impact: Behavior is opaque and not reproducible/tunable without code edits.
- Fix approach: Promote to named module-level constants or settings fields with
  comments explaining provenance.

**Mixed path APIs (`os.path` vs `pathlib`):**
- Issue: `F2S3RunSettings` and orchestration use `pathlib.Path`, but
  `PointCloudTile` save logic uses `os.path.join`/`os.makedirs` throughout
  (`core.py:368-373`, `420-425`, `490-564`).
- Files: `src/f2s3/core.py`
- Impact: Inconsistent style, harder to follow, mixing `str` and `Path`.
- Fix approach: Standardize on `pathlib`.

**Placeholder project metadata:**
- Issue: `pyproject.toml` still has template values: `description = "Description
  of ModuleA"`, `keywords = ["ONE", "TWO"]`, `Homepage = "https://example.com"`,
  `Repository = "https://github.com/me/spam.git"`, and the `"Private :: Do Not
  Upload"` classifier marked "REMOVE BEFORE FLIGHT".
- Files: `pyproject.toml:14-25,68-73`
- Impact: Package cannot be published; misleading metadata.
- Fix approach: Fill in real metadata before any release.

**Stale `_old/` and `build/` directories in working tree:**
- Issue: `_old/scripts/install_pcl.sh` is tracked; a `build/` directory exists in
  the working tree (build artifacts). `f2s3.egg-info/` is also present.
- Files: `_old/`, `build/`, `f2s3.egg-info/`
- Impact: Repo clutter; stale install instructions may be followed by mistake.
- Fix approach: Remove `_old/`; ensure `build/` and egg-info are gitignored.

## Known Bugs

**Feature-extractor call expects wrong number of return values:**
- Symptoms: `batch_feat, _, _ = self.feature_extractor(batch_feat)` unpacks
  exactly three values (`core.py:340,350`). `PointNetFeature.forward` returns
  three values only on the `xp.nelement()==0` path with `tnet=True`
  (`descriptor_model.py:115-123`). Because `PointNetFeature()` is constructed with
  default `tnet=True` (`core.py:606`) and called with a single arg, this currently
  works — but it is fragile: any change to `tnet` or call signature silently
  breaks unpacking. The branch returning 5 values (`l2norm=True, tnet=True`
  through `_forward`) would raise `ValueError`.
- Files: `src/f2s3/core.py:340,350`, `src/f2s3/descriptor_model.py:88-131`
- Trigger: Constructing `PointNetFeature(tnet=False)` or otherwise hitting a
  non-3-tuple return path.
- Workaround: None; the contract is only correct for the current default config.

**`reset_seed` references undefined `logging` and `self.randng`:**
- Symptoms: `FeatureExtractionDataset.reset_seed` calls `logging.info(...)` but
  `data.py` never imports `logging`, and `self.randng` is never defined.
- Files: `src/f2s3/data.py:104-106`
- Trigger: Any call to `reset_seed()` raises `NameError`/`AttributeError`.
- Workaround: Method appears unused (dead code); remove it or fix imports.

**In-place mutation of `correspondences` during refinement:**
- Symptoms: In `filter_correspondences`, `supervoxel_coords = supervoxel_data`
  is a view/alias of `self.correspondences[supervoxel, :]`, and the refine branch
  writes `supervoxel_coords[:, 3:6] = transform_point_cloud(...)`
  (`core.py:453-459`). This mutates `supervoxel_data` (and potentially the slice
  of `self.correspondences`) in place. Meanwhile `save_coords.append(supervoxel_data)`
  appends the mutated array. Intent vs. effect is ambiguous.
- Files: `src/f2s3/core.py:445-466`
- Trigger: `--refine_results` with a robust estimate.
- Workaround: Copy `supervoxel_data` before mutation if independence is intended.

**`compute_c2c` index alignment assumption in fill-gaps path:**
- Symptoms: `c2c_displacements[inlier_idx] = filtered_magnitudes` assumes
  `inlier_idx` indexes into the flattened `save_coords` order produced by
  concatenating per-supervoxel slices, but `filtered_magnitudes` has already been
  re-filtered by `max_disp_magnitude` and possibly `mag_inlier`
  (`core.py:507-512,547,561`). The `max_disp_magnitude` branch reassigns
  `inlier_idx = inlier_idx[max_magnitude_idx]` but the non-`filter_median_magnitude`
  `fill_gaps_c2c` branch uses that reduced `inlier_idx` with full `filtered_magnitudes`
  only in some orderings — the interaction across the three flags is hard to verify.
- Files: `src/f2s3/core.py:506-564`
- Trigger: Combinations of `--max_disp_magnitude`, `--filter_median_magnitude`,
  `--fill_gaps_c2c`.
- Workaround: None; needs unit tests over flag combinations.

**Patch normal-frame branch only handles the "enough points" case correctly:**
- Symptoms: In `extract_patch`, when `ptall.shape[1] <= 10` the local reference
  frame (`lRg`, transform `T`) is skipped and only normalization is applied
  (`data.py:85-89`); the EPS guard at `data.py:63-66` is logically inverted-looking
  (the `< _EPS` branch divides by `norm + _EPS` while the else divides by `norm`).
- Files: `src/f2s3/data.py:43-97`
- Trigger: Sparse patches near tile edges.
- Workaround: None; verify the small-patch and EPS branches against the reference
  paper (eq. 5).

## Security Considerations

**`torch.load` on bundled model files:**
- Risk: `torch.load(f_m)` / `torch.load(o_m)` (`core.py:607,612`) use default
  (unsafe) unpickling. The files are package-internal pretrained weights, so risk
  is low, but `torch.load` without `weights_only=True` can execute arbitrary code
  if a malicious file is ever substituted.
- Files: `src/f2s3/core.py:600-613`
- Current mitigation: Models ship inside the package
  (`src/f2s3/pretrained_models/`).
- Recommendations: Pass `weights_only=True` to both `torch.load` calls.

**Unpinned, partly Git-sourced dependencies:**
- Risk: All PyPI deps are unpinned (`torch`, `open3d ~=0.15`, `numpy`, etc.) and
  two are pulled from GitHub tags (`pc_tiling @ ...@v0.1.2`,
  `supervoxel @ ...@v0.1.2`). A tag can move or a transitive release can break the
  build silently.
- Files: `pyproject.toml:29-43`
- Current mitigation: Git deps are tag-pinned; Docker pins CUDA/Torch index URL.
- Recommendations: Add a lockfile or pin minimum/maximum versions; prefer commit
  SHAs over movable tags for Git deps.

## Performance Bottlenecks

**Per-tile, per-cloud full nearest-neighbor resolution computation:**
- Problem: `_median_nn_resolution` builds a `kd_tree` over the full point cloud
  and queries every point by default for both source and target, once per tile
  (`core.py:81-105`, called from `_compute_resolution` at `core.py:569-579`).
  `max_query_points` subsampling is only used for the global pre-tiling estimate
  (`core.py:636-637`), not per tile.
- Files: `src/f2s3/core.py:61-105,239,569-579`
- Cause: No `max_query_points` cap on the per-tile call.
- Improvement path: Pass a `max_query_points` cap to the per-tile resolution call.

**Repeated CPU<->GPU transfers and full-list accumulation in feature extraction:**
- Problem: `compute_local_features` accumulates all per-batch GPU tensors in a
  Python list then `torch.cat`s them (`core.py:329-356`); correspondence search
  immediately moves them back to CPU via `.cpu().numpy()` (`core.py:403,407`).
  Large tiles hold full feature tensors in GPU memory before transfer.
- Files: `src/f2s3/core.py:329-412`
- Cause: Batched accumulation strategy; HNSW runs on CPU.
- Improvement path: Stream features to CPU/pinned memory incrementally; consider
  building the HNSW index as features arrive.

**`np.linalg.eig` on covariance matrix in inner patch loop:**
- Problem: `extract_patch` runs `np.linalg.eig` per point per patch on the CPU
  (`data.py:51`), inside a Python `for pt in batch_data` loop (`data.py:28-30`).
  For million-point clouds this is a dominant cost.
- Files: `src/f2s3/data.py:28-97`
- Cause: Per-point Python loop with NumPy eig; `eig` (not `eigh`) used on a
  symmetric covariance.
- Improvement path: Use `np.linalg.eigh` (symmetric, faster/stabler); vectorize
  patch extraction; rely on the 6 DataLoader workers (already set).

## Fragile Areas

**`filter_correspondences` flag matrix:**
- Files: `src/f2s3/core.py:430-567`
- Why fragile: Four boolean flags produce overlapping output directories and
  re-index arrays multiple times; no tests guard the combinations.
- Safe modification: Change one branch at a time; add a regression test capturing
  output for each flag combination before refactoring.
- Test coverage: None.

**Catch-all tile error handling hides failures:**
- Files: `src/f2s3/core.py:692-693`
- Why fragile: `except Exception as e: logging.error(...)` swallows every
  per-tile error and continues. A systematic bug (e.g., shape mismatch) would
  silently produce zero output for all tiles while the run "succeeds".
- Safe modification: Narrow the exception types, or re-raise after N consecutive
  failures; surface a non-zero exit code if all tiles fail.
- Test coverage: None.

**Implicit CUDA requirement with no CPU fallback:**
- Files: `src/f2s3/core.py:339,349,450,457,608,613`; `descriptor_model.py:47-48`
- Why fragile: Hardcoded `.cuda()` calls everywhere; no `device` abstraction. The
  tool cannot run on CPU-only machines and fails late at runtime, not at startup.
- Safe modification: Introduce a `device` setting; check `torch.cuda.is_available()`
  in `F2S3RunSettings.__post_init__` and fail fast with a clear message.
- Test coverage: None.

**Global mutation of Torch backend flags:**
- Files: `src/f2s3/core.py:596-598`
- Why fragile: `feature_based_deformation_analysis` flips `allow_tf32` and
  `cudnn.benchmark` as global process state. TF32 was enabled in the latest commit
  but its commit message says "no improvements yet"; TF32 slightly changes numeric
  results, which matters for a metrology/deformation tool.
- Safe modification: Make TF32/benchmark opt-in via a settings flag; document the
  accuracy implication for sub-millimeter displacement measurement.
- Test coverage: None.

## Scaling Limits

**Whole-cloud loads into RAM:**
- Current capacity: `o3d.io.read_point_cloud` loads full source and target clouds
  into memory for the global resolution estimate (`core.py:634-635`) and each tile
  loads source/target plus their overlap clouds (`core.py:208-216`).
- Limit: Bounded by host RAM; very large epochs may OOM before tiling.
- Scaling path: Stream/chunk the global estimate (it already subsamples queries,
  but still loads the full cloud); release clouds promptly (the `del`+`gc.collect`
  at `core.py:639-640` is good — extend the pattern).

**HNSW index `max_elements` fixed to target size, single index per tile:**
- Current capacity: `max_elements=self.local_features_t.shape[0]`
  (`core.py:400`), 64-dim L2.
- Limit: Per-tile; very large tiles inflate index build memory on CPU.
- Scaling path: Already mitigated by tiling; tune `max_points_per_tile`.

## Dependencies at Risk

**`open3d ~=0.15`:**
- Risk: Pinned to an old major series (`~=0.15`, current upstream is much newer);
  KDTree and IO APIs used (`data.py:17`, `core.py:208-216`) may differ in newer
  releases, blocking upgrades.
- Impact: Stuck on old Open3D; potential incompatibility with newer NumPy/Python.
- Migration plan: Test against a current Open3D and update the constraint.

**`pc_tiling` / `supervoxel` Git dependencies:**
- Risk: External Git packages from `gseg-ethz` pinned to `v0.1.2` tags
  (`pyproject.toml:41-42`); availability depends on those repos remaining public
  and the tags remaining stable. They are compiled (CMake/SWIG) extensions.
- Impact: Build failure if repos move/disappear or tags are re-pointed.
- Migration plan: Vendor or mirror; pin to commit SHAs.

## Missing Critical Features

**No CPU/device selection:**
- Problem: Hardcoded `.cuda()` (see Fragile Areas); no way to run without an
  NVIDIA GPU.
- Blocks: CI testing, developer machines without CUDA, debugging.

**No structured output / result aggregation:**
- Problem: Each tile writes its own `displacement_magnitude_tile_*.txt`
  (`core.py:503,535,549,563`); there is no step that merges per-tile outputs into
  a single result cloud.
- Blocks: Downstream consumption requires manual concatenation of many text files.

## Test Coverage Gaps

**No automated tests at all:**
- What's not tested: The entire package. There is no `tests/` directory, no
  pytest/CI test job (`.github/workflows/` contains only `docker-ghcr.yml` and
  `release-please.yml`), and no linter/formatter config (no ruff/flake8/black/
  pre-commit). The only executable check is the `__main__` block in
  `filtering_model.py:97-103` that just prints a parameter count.
- Files: entire `src/f2s3/`
- Risk: Every bug listed above is undetectable in CI. Refactoring the monolithic
  `core.py` is high-risk without a safety net. Numeric changes (TF32, eig vs eigh)
  cannot be validated against a baseline.
- Priority: High.

**Untested numeric kernels:**
- What's not tested: `kabsch_transformation_estimation` (SVD with a silent
  `except Exception` fallback that returns identity, `utils.py:93-102`),
  `transformation_residuals`, `compute_c2c`, and the patch-frame math in
  `extract_patch`.
- Files: `src/f2s3/utils.py`, `src/f2s3/data.py`
- Risk: The Kabsch `except` swallows SVD non-convergence and returns identity with
  `valid_gradient=True` — a silently wrong rigid transform.
- Priority: High.

---

*Concerns audit: 2026-06-11*
