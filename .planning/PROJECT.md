# F2S3 Performance & Throughput Optimization

## What This Is

A runtime-optimization effort on **F2S3**, a CUDA/PyTorch CLI pipeline that estimates a 3D displacement field between two point-cloud epochs using deep-learned local features and supervoxel-constrained outlier filtering. This project makes the pipeline materially faster on a single **NVIDIA RTX 3090 Ti** (Ampere, 24 GB) — by actually using the GPU it has and moving CPU-bound work onto it — **without changing the displacement output beyond a validated numerical tolerance**. It is for the maintainers/operators running F2S3 on large epoch pairs.

## Core Value

Reduce end-to-end wall-clock runtime of the F2S3 pipeline on a single RTX 3090 Ti, while keeping the resulting descriptors and displacement field numerically equivalent (within a validated tolerance) to the current FP32 baseline. **Speed is worthless here if it silently changes a sub-millimeter deformation measurement.**

## Requirements

### Validated

<!-- Existing capabilities inferred from the codebase map — already shipped and relied upon. -->

- ✓ Tile-by-tile deformation analysis pipeline (features → supervoxels → correspondences → filtering) — existing
- ✓ DIP/PointNet 64-dim local feature descriptor with pretrained weights — existing
- ✓ Supervoxel-constrained filtering network + weighted Kabsch outlier rejection — existing
- ✓ CLI + `F2S3RunSettings` config object; `--start_from_tiled_data` fast path — existing
- ✓ Runs entirely under `torch.no_grad()` (no autograd overhead to reclaim) — existing
- ✓ C++ extensions for tiling (`pc_tiling`) and supervoxel segmentation (`supervoxel`) — existing

### Active

<!-- This milestone. Speed-only; no algorithm-quality changes. Each is a hypothesis until shipped + parity-validated. -->

**Foundation (must come first):**
- [ ] Establish a reproducible baseline + numerical-parity + per-stage timing regression harness, used as the gate for every optimization below

**§1 Quick wins (XS, low risk):**
- [ ] Opt-in precision setting (`fp32 | tf32 | fp16`) enabling Ampere tensor-core math (TF32) and optional AMP autocast
- [ ] Wire `batch_size` through to feature extraction and raise descriptor batch (1000 → 10k–50k, VRAM-tuned)
- [ ] Tune DataLoader (configurable `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`)
- [ ] Remove hot-loop `empty_cache()` / thin out `gc.collect()`
- [ ] `np.linalg.eig` → `eigh`; `torch.svd` → `torch.linalg.svd`

**§2 GPU-native patch extraction (L, highest impact):**
- [ ] Move per-point radius/neighbor search to the GPU (FRNN / torch_cluster / PyTorch3D / Open3D-tensor — chosen by spike)
- [ ] Batch the LRF eigendecomposition on-GPU (`torch.linalg.eigh` over stacked 3×3 covariances) + vectorize LRF math
- [ ] Vectorize the 256-point subsample (replace per-point `np.random.choice`), deterministic seeding for the harness
- [ ] Compute neighbor structure once per cloud and share across resolution estimate / descriptor LRF / supervoxels where radii align

**§3 Batched supervoxel filtering (M, high impact):**
- [ ] Keep `self.correspondences` resident on GPU; slice supervoxels on-device (no per-SV host→device copies)
- [ ] Bucket supervoxels by size + pad with validity mask; run batched filter forwards + batched Kabsch instead of thousands of single launches

**§4 Correspondence search / GPU ANN (M, situational):**
- [ ] Keep features on-device and run GPU ANN / tiled exact L2 (FAISS-GPU or cuVS), removing the `.cpu().numpy()` feature dump

**§5 I/O & orchestration (S–M):**
- [ ] Eliminate redundant disk reads (avoid re-reading the source tile in `compute_supervoxels`; reuse resident clouds)
- [ ] Pipeline tiles — overlap tile N GPU work with tile N+1 disk load + CPU prep (producer queue / two CUDA streams)
- [ ] Right-size tiles for 24 GB (raise `max_points_per_tile` where host RAM / ANN memory allows)

**§6 Compilation & precision polish (M, modest):**
- [ ] `torch.compile` the descriptor/filter with fixed/padded shapes + warmup
- [ ] Optional FP16-resident features to halve on-device memory/bandwidth

### Out of Scope

- Algorithm-quality changes (spatial gating, multi-radius descriptors, GeDi swap, confidence channels) — covered by a **separate** algorithm-upgrade plan; this milestone is runtime/throughput only
- Multi-GPU / data-parallel scaling — target hardware is a single RTX 3090 Ti; no second GPU to exploit
- CPU-only fallback / device abstraction — a known codebase gap, but not a performance goal for this milestone
- Broader refactors from the codebase concerns (splitting `core.py`, fixing placeholder metadata, packaging) — not runtime wins; only touched where a perf change requires it

## Context

- **Hardware target:** single NVIDIA RTX 3090 Ti (Ampere GA102, 24 GB GDDR6X, TF32 + FP16/BF16 tensor cores). The 24 GB is an asset for *batching and on-device residency*, not because the model is large (it's tiny).
- **The reframing that drives everything:** the dominant cost is almost certainly **CPU + CPU↔GPU transfer**, not GPU FLOPs. Prime suspects: per-point patch extraction (`data.py::extract_patch` — Open3D KD-tree + `np.linalg.eig` per point, ~2M calls/tile-pair), the per-supervoxel filtering loop (death-by-tiny-launches), and the GPU→CPU feature dump for CPU HNSW. The GPU descriptor forward is under-fed (FP32, hardcoded batch of 1000).
- **Codebase state:** ~1,270 lines of Python; monolithic 700-line `core.py`; **zero automated tests, no linter/CI test job**. Recent branch commits (`compute-performance-improvement`) added TF32 / changed batch default but are marked "no improvements yet" — i.e., landed but unvalidated.
- **Source of truth:** the user's plan `_input/F2S3_performance_optimization.md` (effort ratings XS→L, prioritized roadmap table, suggested sequence). Codebase map in `.planning/codebase/`.
- **Profile-first discipline:** impact estimates are qualitative until confirmed on real data/hardware (§0). Re-profile after each tier.

## Constraints

- **Correctness**: Every change must be validated for numerical parity of descriptors and the final displacement field against the FP32 baseline — sub-millimeter metrology tolerance — using the regression harness. Parity gate is non-negotiable.
- **Hardware**: Single RTX 3090 Ti (24 GB, Ampere). GPU required; no CPU fallback path is added.
- **Dependencies**: GPU neighbor-search libs (FRNN / torch_cluster / PyTorch3D) and FAISS-GPU/cuVS add build complexity and must match the CUDA 12.6 / PyTorch cu126 toolchain. C++ extensions (`pc_tiling`, `supervoxel`) are git-pinned native deps.
- **Sequence**: Harness before optimizations; quick wins before refactors; profile between tiers (don't optimize blind).
- **Tech stack**: Python ≥3.11, PyTorch (CUDA cu126), Open3D ~0.15, hnswlib, scikit-learn.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build the baseline + parity/timing harness as Phase 1 before any optimization | Zero tests exist; numeric changes (TF32/FP16, eig→eigh) can silently corrupt a metrology output. Need a regression gate first. | — Pending |
| Full plan scope (§0–§6) in this milestone | User wants the complete runtime-optimization roadmap, not a partial pass. | — Pending |
| GPU neighbor-search library chosen by spike, not pre-committed | §2.2 lists 4 viable options (FRNN / torch_cluster / PyTorch3D / Open3D-tensor); pick on build feasibility + numerical parity for our data. | — Pending |
| Precision is opt-in via a `precision` setting, A/B-able | FP16 changes numerics; must be reversible and comparable against FP32 baseline. | — Pending |
| Profile between every tier (§0 re-run) | Reframing (CPU/transfer-bound) is inferred from code; confirm on real data before committing the large §2/§3 refactors. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-11 after initialization*
