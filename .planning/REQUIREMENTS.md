# Requirements: F2S3 Performance & Throughput Optimization

**Defined:** 2026-06-11
**Core Value:** Reduce end-to-end wall-clock runtime on a single RTX 3090 Ti while keeping descriptors and the displacement field numerically equivalent (within validated tolerance) to the FP32 baseline.

> **Global acceptance gate:** Every optimization requirement (QW/EXT/FILT/ANN/IO/COMP) is "done" only when it (a) demonstrably reduces runtime on representative data and (b) passes the numerical-parity harness (HARN) against the FP32 baseline. Speed without parity is a failure.

## v1 Requirements

Requirements for this milestone. Source: `_input/F2S3_performance_optimization.md`. Each maps to exactly one phase.

### Harness — Baseline, Parity & Profiling (§0 + foundation)

- [x] **HARN-01**: Capture a reproducible FP32 baseline (descriptors + displacement field) on a representative tile/epoch pair, stored as the reference artifact
- [x] **HARN-02**: Numerical-parity check compares a run's descriptors and final displacement field against the baseline within a configurable tolerance and reports pass/fail
- [x] **HARN-03**: Per-stage timing is captured comparably across runs (features / supervoxels / correspondences / filtering), building on the existing `--verbose` timings
- [x] **HARN-04**: Profiling runbook + tooling wired (py-spy, `nvidia-smi dmon`/nvtop, `torch.profiler`) to attribute cost across CPU vs GPU vs CPU↔GPU transfer
- [x] **HARN-05**: The harness runs as a single repeatable command and serves as the regression gate invoked by every later phase

### Quick Wins (§1 — XS effort, low risk)

- [ ] **QW-01**: A `precision` setting (`fp32 | tf32 | fp16`) controls Ampere TF32 math and optional AMP autocast on GPU forwards; default preserves current numerical behavior until opted in
- [ ] **QW-02**: `F2S3RunSettings.batch_size` is actually wired into feature extraction (no hardcoded 1000) and the descriptor batch is raised to a configurable, VRAM-tuned value
- [ ] **QW-03**: DataLoader exposes configurable `num_workers`, `pin_memory`, `persistent_workers`, and `prefetch_factor`
- [ ] **QW-04**: Hot-loop `torch.cuda.empty_cache()` calls are removed and `gc.collect()` is thinned to (at most) true tile boundaries
- [ ] **QW-05**: Symmetric eigendecomposition uses `np.linalg.eigh`; Kabsch uses `torch.linalg.svd`

### GPU-Native Patch Extraction (§2 — L effort, highest impact)

- [x] **EXT-01**: Per-point radius/neighbor search runs on the GPU (library selected by spike: FRNN / torch_cluster / PyTorch3D / Open3D-tensor), replacing the Open3D per-point CPU KD-tree
- [x] **EXT-02**: The LRF eigendecomposition is batched on-GPU via `torch.linalg.eigh` over stacked 3×3 covariances, with the LRF math vectorized
- [x] **EXT-03**: The 256-point subsample is vectorized (no per-point `np.random.choice`) with an option for deterministic seeding for the harness
- [x] **EXT-04**: The neighbor structure is computed once per cloud and reused across resolution estimate, descriptor LRF, and supervoxels where radii align
- [x] **EXT-05**: GPU-native extraction reproduces baseline descriptors within parity tolerance (HARN-02)

### Batched Supervoxel Filtering (§3 — M effort, high impact)

- [ ] **FILT-01**: `self.correspondences` is held resident on the GPU and supervoxels are sliced on-device, eliminating per-supervoxel host→device copies
- [ ] **FILT-02**: Supervoxels are bucketed by size and padded with a validity mask; the filter forward and weighted Kabsch run as a few batched ops instead of thousands of single launches
- [ ] **FILT-03**: Batched filtering reproduces the baseline displacement field within parity tolerance (HARN-02)

### Correspondence Search / GPU ANN (§4 — M effort, situational)

- [ ] **ANN-01**: Features stay on-device and correspondence search uses a GPU ANN or tiled exact L2 (FAISS-GPU or cuVS), removing the `.cpu().numpy()` feature dump
- [ ] **ANN-02**: GPU search reproduces baseline correspondences within parity tolerance (HARN-02)

### I/O & Orchestration (§5 — S–M effort)

- [ ] **IO-01**: Redundant disk reads are eliminated — `compute_supervoxels` reuses the resident source cloud instead of re-reading the tile from disk
- [ ] **IO-02**: Tile processing is pipelined so tile N's GPU work overlaps tile N+1's disk load + CPU patch prep (producer queue or two CUDA streams)
- [ ] **IO-03**: `max_points_per_tile` is tunable and right-sized for 24 GB, with documented host-RAM / ANN memory headroom

### Compilation & Precision Polish (§6 — M effort, modest)

- [ ] **COMP-01**: The descriptor/filter are optionally `torch.compile`d with fixed/padded shapes and a warmup pass (opt-in)
- [ ] **COMP-02**: An optional FP16-resident-features setting halves on-device memory/bandwidth, parity-validated

## v2 Requirements

Deferred. Tracked but not in this milestone.

### Algorithm Quality (separate plan)

- **ALGO-01**: Spatial gating of correspondences (shrinks ANN to a few-metre ball)
- **ALGO-02**: Multi-radius descriptors (64 → 192 dim)
- **ALGO-03**: GeDi descriptor swap / distilled-GeDi variant
- **ALGO-04**: Confidence-channel and top-k filtering changes

### Robustness / Maintainability

- **ROBUST-01**: Device abstraction + CPU fallback (`torch.cuda.is_available()` fail-fast)
- **ROBUST-02**: Structured per-run output aggregation (merge per-tile `.txt` into one result cloud)

## Out of Scope

Explicitly excluded for this milestone.

| Feature | Reason |
|---------|--------|
| Algorithm-quality changes (spatial gating, multi-radius, GeDi, confidence channels) | Covered by the separate algorithm-upgrade plan; this milestone is runtime/throughput only |
| Multi-GPU / data-parallel scaling | Target is a single RTX 3090 Ti; no second GPU to exploit |
| CPU-only fallback / device abstraction | Known gap, but not a performance goal here (deferred to v2 ROBUST-01) |
| `core.py` decomposition, packaging/metadata cleanup, linter/CI setup | Not runtime wins; only touched where a perf change requires it |
| Output result aggregation | Convenience feature, not a speed win (deferred to v2 ROBUST-02) |

## Traceability

Intended phase mapping (finalized by roadmap creation).

| Requirement | Phase | Status |
|-------------|-------|--------|
| HARN-01 | Phase 1 | Roadmapped |
| HARN-02 | Phase 1 | Roadmapped |
| HARN-03 | Phase 1 | Roadmapped |
| HARN-04 | Phase 1 | Roadmapped |
| HARN-05 | Phase 1 | Roadmapped |
| QW-01 | Phase 2 | Roadmapped |
| QW-02 | Phase 2 | Roadmapped |
| QW-03 | Phase 2 | Roadmapped |
| QW-04 | Phase 2 | Roadmapped |
| QW-05 | Phase 2 | Roadmapped |
| EXT-01 | Phase 3 | Roadmapped |
| EXT-02 | Phase 3 | Roadmapped |
| EXT-03 | Phase 3 | Roadmapped |
| EXT-04 | Phase 3 | Roadmapped |
| EXT-05 | Phase 3 | Roadmapped |
| FILT-01 | Phase 4 | Roadmapped |
| FILT-02 | Phase 4 | Roadmapped |
| FILT-03 | Phase 4 | Roadmapped |
| ANN-01 | Phase 5 | Roadmapped |
| ANN-02 | Phase 5 | Roadmapped |
| IO-01 | Phase 6 | Roadmapped |
| IO-02 | Phase 6 | Roadmapped |
| IO-03 | Phase 6 | Roadmapped |
| COMP-01 | Phase 7 | Roadmapped |
| COMP-02 | Phase 7 | Roadmapped |

**Coverage:**

- v1 requirements: 25 total
- Mapped to phases: 25
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-11*
*Last updated: 2026-06-11 after roadmap creation (25/25 mapped, 0 unmapped)*
