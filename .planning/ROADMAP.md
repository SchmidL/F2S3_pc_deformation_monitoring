# Roadmap: F2S3 Performance & Throughput Optimization

## Overview

This is a brownfield runtime-optimization effort on the existing F2S3 CUDA/PyTorch CLI pipeline, sequenced exactly as the source plan dictates: §0 (harness) → §1 (quick wins) → re-profile → §2 (extraction) → §3 (filtering) → §4 (ANN) → §5 (I/O) → §6 (compilation), ordered easy→hard. Phase 1 builds the baseline + numerical-parity + per-stage timing harness that is the non-negotiable regression gate for every optimization tier that follows. Each later phase is an optimization tier delivering a measurable, parity-validated speedup of the working pipeline: it must show a runtime reduction on representative data AND pass the Phase 1 parity harness against the FP32 baseline. We re-profile after every tier to confirm where the bottleneck moved before committing the next, larger refactor. Phase 3 (GPU-native patch extraction) is expected to be the dominant win if profiling confirms the pipeline is CPU-bound in per-point extraction.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Baseline, Parity & Profiling Harness** - Reproducible FP32 baseline + numerical-parity check + per-stage timing + profiling runbook; the regression gate for every later phase (completed 2026-06-11)
- [ ] **Phase 2: Quick Wins / Ampere Utilization** - TF32/AMP precision setting, big descriptor batch + wired batch_size, DataLoader tuning, removed hot-loop empty_cache/gc, eig→eigh + torch.linalg.svd
- [x] **Phase 3: GPU-Native Patch Extraction** - GPU radius search, batched on-GPU eigh LRF, vectorized subsample, shared neighbor structure; the highest-impact refactor (completed 2026-06-11)
- [ ] **Phase 4: Batched Supervoxel Filtering** - GPU-resident correspondences, bucket+pad batching, batched Kabsch; collapses the per-supervoxel micro-launch loop
- [ ] **Phase 5: Correspondence Search / GPU ANN** - On-device features + GPU ANN / tiled exact L2, removing the GPU→CPU feature dump
- [ ] **Phase 6: I/O & Orchestration** - Eliminate redundant disk reads, pipeline tiles across CUDA streams, right-size tiles for 24 GB
- [ ] **Phase 7: Compilation & Precision Polish** - Opt-in torch.compile with fixed shapes + optional FP16-resident features

## Phase Details

### Phase 1: Baseline, Parity & Profiling Harness

**Goal**: Establish a reproducible FP32 baseline plus a single-command numerical-parity + per-stage timing harness, and a profiling runbook, so every later optimization can be gated on "faster AND numerically equivalent"
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: HARN-01, HARN-02, HARN-03, HARN-04, HARN-05
**Success Criteria** (what must be TRUE):

  1. A single repeatable command captures the FP32 baseline (descriptors + displacement field) on a representative tile/epoch pair and stores it as the reference artifact
  2. The harness compares a new run's descriptors and final displacement field against the baseline within a configurable tolerance and reports an unambiguous pass/fail
  3. The harness emits comparable per-stage timings (features / supervoxels / correspondences / filtering) across runs, building on the existing `--verbose` output
  4. A documented profiling runbook (py-spy, nvidia-smi dmon/nvtop, torch.profiler) attributes cost across CPU vs GPU vs CPU↔GPU transfer and is runnable against a tile
  5. The harness is wired as the regression gate that every later phase invokes by name

**Plans**: 3 plansPlans:
**Wave 1**

- [x] 01-01-PLAN.md — Capture FP32 baseline + per-stage timings on the _small pair (HARN-01, HARN-03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — Source-keyed parity check + exit-code gate (HARN-02, HARN-05)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-03-PLAN.md — Profiling runbook + noise-floor calibration + gate README contract (HARN-04, HARN-05)

Notes: This phase produces no speedup itself — it is the instrument. Its parity check and timing report are the acceptance gate for Phases 2-7. The §0 profile captured here establishes the initial bottleneck attribution; the reframing (CPU/transfer-bound, dominant cost in `extract_patch`) should be confirmed on real data here before committing the §2/§3 refactors.

### Phase 2: Quick Wins / Ampere Utilization

**Goal**: Make the existing pipeline materially faster by actually using the Ampere GPU it has — tensor-core precision, large batches, a tuned DataLoader, and cheap numerical fixes — with no structural change
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: QW-01, QW-02, QW-03, QW-04, QW-05
**Success Criteria** (what must be TRUE):

  1. A `precision` setting (`fp32 | tf32 | fp16`) controls TF32 math and optional AMP autocast on GPU forwards, defaulting to current FP32 behavior until opted in
  2. `F2S3RunSettings.batch_size` is wired through to feature extraction (no hardcoded 1000) and the descriptor batch is a configurable, VRAM-tuned value; DataLoader exposes `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`; hot-loop `empty_cache()` is removed and `gc.collect()` thinned to tile boundaries; symmetric eig uses `np.linalg.eigh` and Kabsch uses `torch.linalg.svd`
  3. The pipeline shows a measurable end-to-end wall-clock reduction on representative data versus the FP32 baseline (Phase 1 timing report)
  4. The Phase 1 numerical-parity harness passes against the FP32 baseline at every enabled precision tier (including the validated FP16 path)

**Plans**: TBD

Notes: Lowest-effort, lowest-risk tier (XS items) — done first. Re-profile after this tier (§0 re-run): if extraction still dominates, the ceiling is set by Phase 3, which is where the next effort goes.

### Phase 3: GPU-Native Patch Extraction

**Goal**: Move the per-point patch extraction (CPU KD-tree radius search + per-point eigendecomposition + subsample) onto the GPU as batched tensor ops, eliminating the dominant CPU bottleneck
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: EXT-01, EXT-02, EXT-03, EXT-04, EXT-05
**Success Criteria** (what must be TRUE):

  1. Per-point radius/neighbor search runs on the GPU (library selected by a planning spike: FRNN / torch_cluster / PyTorch3D / Open3D-tensor), replacing the Open3D per-point CPU KD-tree
  2. The LRF eigendecomposition is batched on-GPU via `torch.linalg.eigh` over stacked 3×3 covariances with vectorized LRF math, the 256-point subsample is vectorized with optional deterministic seeding, and the neighbor structure is computed once per cloud and reused across resolution estimate / descriptor LRF / supervoxels where radii align
  3. The feature-extraction stage shows a large measurable runtime reduction on representative data (expected dominant win if Phase 1 profiling confirmed CPU-bound extraction)
  4. GPU-native extraction reproduces baseline descriptors within parity tolerance — the Phase 1 parity harness passes against the FP32 baseline

**Plans**: 4 plans

**Wave 0** *(test scaffolding + settings contract + gate decision)*

- [x] 03-01-PLAN.md — RED parity-test scaffold + extraction_backend/extraction_seed settings + EXT-05 gate-metric decision (EXT-01..EXT-05)

**Wave 1** *(blocked on Wave 0)*

- [x] 03-02-PLAN.md — GPU index per cloud + radius search -> padded coords + mask, wired end-to-end behind extraction_backend=gpu (EXT-01, EXT-04)

**Wave 2** *(blocked on Wave 1)*

- [x] 03-03-PLAN.md — Masked batched eigh LRF + seeded vectorized subsample (parity-grade) (EXT-02, EXT-03)

**Wave 3** *(blocked on Wave 2)*

- [x] 03-04-PLAN.md — Seeded CPU-vs-GPU A/B parity gate + descriptor cos floor + speedup sign-off (EXT-05)

Notes: Highest-impact, largest refactor (L effort, medium parity risk). The neighbor-search library is chosen by a spike during planning, not pre-committed. Re-profile after this tier — the bottleneck is expected to move off the CPU; whatever surfaces next dictates the relative payoff of Phases 4-6.

### Phase 4: Batched Supervoxel Filtering

**Goal**: Collapse the per-supervoxel filtering loop (thousands of tiny host→device transfers and batch-of-1 forwards) into a few large batched GPU ops with correspondences held resident on-device
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: FILT-01, FILT-02, FILT-03
**Success Criteria** (what must be TRUE):

  1. `self.correspondences` is held resident on the GPU and supervoxels are sliced on-device, eliminating per-supervoxel host→device copies
  2. Supervoxels are bucketed by size and padded with a validity mask so the filter forward and weighted Kabsch run as a few batched ops instead of thousands of single launches
  3. The filtering stage shows a measurable runtime reduction on representative data versus the Phase 1 baseline timing
  4. Batched filtering reproduces the baseline displacement field within parity tolerance — the Phase 1 parity harness passes against the FP32 baseline

**Plans**: TBD

Notes: M effort, high impact — turns death-by-tiny-launches into a few large GPU ops. Re-profile after this tier to confirm where the remaining wall-clock sits before deciding the priority of Phases 5-6.

### Phase 5: Correspondence Search / GPU ANN

**Goal**: Keep features on-device and run the correspondence search on the GPU, removing the GPU→CPU feature dump and CPU-only HNSW round-trip
**Mode:** mvp
**Depends on**: Phase 4
**Requirements**: ANN-01, ANN-02
**Success Criteria** (what must be TRUE):

  1. Features stay on-device and correspondence search uses a GPU ANN or tiled exact L2 (FAISS-GPU or cuVS), removing the `.cpu().numpy()` feature dump
  2. The correspondence-search stage shows a measurable runtime reduction on representative data versus the Phase 1 baseline timing
  3. GPU search reproduces baseline correspondences within parity tolerance — the Phase 1 parity harness passes against the FP32 baseline

**Plans**: TBD

Notes: M effort, situational — HNSW at 1M points may not be the bottleneck; confirm against the latest re-profile before investing. Value depends on where Phases 3-4 left the bottleneck.

### Phase 6: I/O & Orchestration

**Goal**: Hide I/O and CPU-prep latency behind GPU compute by removing redundant disk reads, pipelining tiles, and right-sizing tiles for 24 GB
**Mode:** mvp
**Depends on**: Phase 5
**Requirements**: IO-01, IO-02, IO-03
**Success Criteria** (what must be TRUE):

  1. Redundant disk reads are eliminated — `compute_supervoxels` reuses the resident source cloud instead of re-reading the tile from disk
  2. Tile processing is pipelined so tile N's GPU work overlaps tile N+1's disk load + CPU patch prep (producer queue or two CUDA streams), and `max_points_per_tile` is tunable and right-sized for 24 GB with documented host-RAM / ANN memory headroom
  3. The pipeline shows a measurable end-to-end runtime reduction on a multi-tile representative run versus the Phase 1 baseline timing
  4. The Phase 1 parity harness passes against the FP32 baseline (orchestration changes must not alter the displacement output)

**Plans**: TBD

Notes: S–M effort. Pipelining benefits scale with the number of tiles; re-profile a multi-tile run to confirm the overlap actually hides the latency it targets.

### Phase 7: Compilation & Precision Polish

**Goal**: Squeeze the final modest gains by optionally compiling the descriptor/filter with fixed shapes and optionally holding features in FP16 on-device — both opt-in and parity-validated
**Mode:** mvp
**Depends on**: Phase 6
**Requirements**: COMP-01, COMP-02
**Success Criteria** (what must be TRUE):

  1. The descriptor/filter are optionally `torch.compile`d with fixed/padded shapes and a warmup pass (opt-in, no recompile thrash on the steady-state batch)
  2. An optional FP16-resident-features setting halves on-device memory/bandwidth and is parity-validated
  3. With the polish enabled, the pipeline shows a measurable (modest) runtime reduction versus the Phase 6 state, and no regression versus the Phase 1 baseline
  4. The Phase 1 parity harness passes against the FP32 baseline with both options enabled

**Plans**: TBD

Notes: M effort, modest impact — last, per the source sequence. `torch.compile` and `cudnn.benchmark` assume stable shapes, so this depends on the fixed/padded batch shapes established in earlier tiers.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Baseline, Parity & Profiling Harness | 3/3 | Complete    | 2026-06-11 |
| 2. Quick Wins / Ampere Utilization | 0/TBD | Not started | - |
| 3. GPU-Native Patch Extraction | 4/4 | Complete   | 2026-06-11 |
| 4. Batched Supervoxel Filtering | 0/TBD | Not started | - |
| 5. Correspondence Search / GPU ANN | 0/TBD | Not started | - |
| 6. I/O & Orchestration | 0/TBD | Not started | - |
| 7. Compilation & Precision Polish | 0/TBD | Not started | - |
