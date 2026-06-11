# F2S3 Profiling Runbook (HARN-04)

**Purpose:** Attribute wall-clock cost across CPU vs GPU vs CPU↔GPU transfer for
a single F2S3 tile, and reproduce the measurement on demand.  Every tier in the
optimisation roadmap (§1–§7 of the source plan) must be re-profiled after its
changes land — this runbook is the authoritative how-to.

All commands assume the `f2s3` conda environment is active and the working
directory is the repository root.

---

## 1. Per-stage wall-clock timing (built-in, zero install)

The pipeline already logs per-stage elapsed seconds when `--verbose` is set.  The
harness `capture` and `profile` subcommands automatically enable verbose mode and
harvest these timings into `timings.json`.

```bash
# Run the harness capture on the _small pair — timings.json is written alongside
# the displacement and descriptor artifacts.
python harness/run_harness.py capture --out baseline/

# Inspect the per-stage breakdown:
cat baseline/timings.json
```

**Expected keys:** `features`, `supervoxels`, `correspondences`, `filtering`.
Each value is a float in seconds.

**Interpretation:** If `features` dominates (e.g. 10–12 min for the `_small` pair
on the RTX 3090 Ti), the pipeline is **CPU-bound during patch extraction** — the
primary optimisation target for Phase 2 (GPU-native patch extraction) and
Phase 3 (GPU neighbour search).

---

## 2. GPU utilisation timeline (`nvidia-smi dmon`)

`nvidia-smi` ships with the NVIDIA driver and is present at `/usr/bin/nvidia-smi`.

```bash
# Start the GPU monitor in a separate terminal BEFORE launching the pipeline.
# The '-s u' flag polls utilisation (sm, mem, enc, dec) every second.
nvidia-smi dmon -s u -d 1 -o TD > gpu_util.csv

# In another terminal, run the pipeline:
python harness/run_harness.py capture --out baseline/

# Stop nvidia-smi with Ctrl-C when the pipeline finishes.
```

**Interpretation:** If the `sm` (streaming-multiprocessor) column stays below
20–30 % during feature extraction, the GPU is **starved** — CPU patch extraction
cannot feed it fast enough.  This confirms the thesis that the 3090 Ti is the
*least*-used resource until the CPU-bound stages are moved onto it.

---

## 3. Python CPU hotspots (`py-spy`)

`py-spy` is a sampling profiler that attaches to a running Python process without
code changes or restarts.  It is **not embedded in any script** — the operator
runs it manually.

**Caveat — ptrace permissions:** On some Linux hosts `py-spy` requires elevated
privileges to attach to a process.  If you see a `ptrace` error, either run with
`sudo` or grant `CAP_SYS_PTRACE`:

```bash
# Option A — sudo (quick, less granular):
sudo py-spy top --pid <PID>

# Option B — capability (more secure, persistent across reboots):
sudo setcap cap_sys_ptrace=eip $(which py-spy)
```

**NEVER embed `sudo` in a harness script.**  The operator runs `py-spy` knowingly
in a separate terminal.

### 3a. Live top view

```bash
# In one terminal, start the pipeline:
python harness/run_harness.py capture --out baseline/

# In another terminal, find the PID and attach:
pgrep -f run_harness    # or:  ps aux | grep f2s3
py-spy top --pid <PID>
```

Press `?` inside `py-spy top` for navigation help.  The flame-graph view (`f`)
is the most informative.

### 3b. Recording a flame-graph for offline analysis

```bash
py-spy record -o profile.svg --pid <PID> --duration 60
# Open profile.svg in a browser.
```

**What to look for:** Functions consuming high CPU self-time.  The prime suspects
are `extract_patch` (`data.py`), `np.linalg.eig` (LRF eigendecomposition), and
Open3D `KDTreeFlann.search_radius_vector_3d` (per-point radius search).  If any
of these dominate, the CPU-bound extraction thesis (§2 of the source plan) is
confirmed.

---

## 4. GPU kernel-level attribution (`torch.profiler`)

The harness ships with a `profile` subcommand that wraps the `_small` pipeline
run in `torch.profiler` and prints a CPU + CUDA attribution table.

```bash
# Run the full pipeline under torch.profiler (captures every PyTorch op).
# This takes ~30 min on the _small pair — same as a normal capture.
python harness/run_harness.py profile --out profile_out/

# The profiler table is printed to stdout after the run and looks like:
#
#  ---------------------------------  ------------  ------------  ------------
#                              Name    Self CPU %      Self CPU   CPU total %
#  ---------------------------------  ------------  ------------  ------------
#               compute_local_features         ...           ...           ...
#            filter_correspondences            ...           ...           ...
#  ---------------------------------  ------------  ------------  ------------
```

### 4a. Targeted profiling of specific stages

The `profile_wrapper.py` module provides `profile_stages(tile)` for profiling
a single `PointCloudTile` instance.  Use it in a script or interactive session
when you need to zoom into `compute_local_features` or `filter_correspondences`
in isolation:

```python
from harness.profile_wrapper import profile_stages
# After obtaining a PointCloudTile instance `tile` (e.g. from the pipeline loop):
prof = profile_stages(tile)
prof.export_chrome_trace("trace.json")   # open in chrome://tracing
```

### 4b. Chrome trace export

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    run_pipeline(...)

prof.export_chrome_trace("chrome_trace.json")
# Open chrome://tracing in Chrome and load the file.
```

---

## 5. One-time noise-floor measurement (D-08)

The parity gate is only trustworthy when its WARN/FAIL thresholds sit **above**
the natural run-to-run variation of an unseeded baseline.  This procedure
measures that variation once and calibrates the shipped defaults.

### Procedure

1. **Run N unseeded captures** (N = 5; ~30 min each, ~2.5 h total):

   ```bash
   for i in $(seq 1 5); do
       echo "=== Capture $i/5 ==="
       python harness/run_harness.py capture --out baseline_noise_$i/
   done
   ```

2. **Compare every pair** using the harness parity check (Plan 01-02):

   ```bash
   for i in $(seq 1 5); do
       for j in $(seq $((i+1)) 5); do
           echo "=== baseline_noise_$i vs baseline_noise_$j ==="
           python harness/run_harness.py check \
               --baseline baseline_noise_$i/ \
               --against baseline_noise_$j/ \
               --warn 999 --fail 999   # disable bands, just get the max_abs
       done
   done
   ```

3. **Record the maximum max_abs** across all 10 pairs — this is the **measured
   noise floor**.

4. **D-08 self-check:** If the measured floor **approaches or exceeds 0.01 m
   (1 cm)**, the 1 cm WARN band is invalid — the gate would fire on natural
   variation.  In that case, **raise the WARN band above the floor** (at least
   3× the measured floor, or investigate the source of high variance).

### Measured noise floor

**Measured:** 2026-06-11 on the `_small` pair (RTX 3090 Ti, torch 2.7.1+cu118).

| Metric | Value |
|--------|-------|
| N (number of unseeded captures) | 5 |
| Inlier counts (min … max) | 71,905 … 72,723 |
| Measured noise floor (max-abs across all 10 pairs) | **276.7 m** |
| Shipped WARN default | 0.01 m (1 cm) |
| Shipped FAIL default | 0.05 m (5 cm) |
| D-08 self-check | **FAILED — noise floor (276.7 m) ≫ WARN band (0.01 m)** |

### Investigation (D-08 self-check failure)

The measured 276.7 m noise floor is **not** numerical floating-point noise — it
is **correspondence instability** caused by the stochastic per-point subsample
in `extract_patch` (`data.py:95`, `np.random.choice`).  Different random
subsamples produce different local-feature descriptors → HNSW finds different
nearest neighbours in feature space → different target points are matched for
the same source point → displacement magnitudes differ by up to 276 m between
runs.

This is the *designed* statistical nature of parity (D-07): no two unseeded
runs produce the same correspondences, so comparing displacement magnitudes
point-by-point across unseeded runs yields enormous differences.

**Resolution:** The 1 cm / 5 cm max-abs displacement-difference bands are only
meaningful when comparing runs that share the **same random state** (seeded
reproducibility, D-09) or when the comparison metric is redefined to account
for correspondence instability (e.g. statistical distribution comparison over
many unseeded runs).  The current gate contract (HARN-05) with cm-level bands
on max-abs displacement difference is **valid for seeded within-implementation
comparisons** (e.g. Phase 1.1 TF32 vs FP32 with same seed) but **not for
unseeded cross-run comparisons**.

The shipped `--warn 0.01` and `--fail 0.05` defaults remain in place as the
documented contract; operators must use `--seed` when comparing within one
implementation, and Phases 2–7 should re-evaluate the gate metric before
committing cross-implementation changes (GPU extraction, FP16, etc.).

---

## 6. Attribution interpretation guide

| Profile signature | Diagnosis | Action |
|-------------------|-----------|--------|
| High CPU self-time, low GPU utilisation during feature extraction | CPU-bound patch extraction (expected) | Proceed to Phase 2 (GPU-native extraction) |
| Many small CUDA kernel launches, GPU idle between them | Per-supervoxel micro-launches in `filter_correspondences` | Proceed to Phase 3 (batched filtering) |
| High `cudaMemcpy` time, moderate GPU time | Host↔device transfer bottleneck | Keep data on GPU (Phase 4 correspondences, Phase 3 features) |
| High GPU utilisation, low CPU wait time | GPU compute-bound (good — unlikely on current code) | Proceed to Phase 6 (torch.compile, precision) |
| `features` time drops after Phase 2/3, `filtering` time unchanged | Extraction bottleneck removed — filtering is the new ceiling | Proceed to Phase 3 (batching). **Re-profile after every tier.** |

**Key rule from §0 of the source plan:** Re-profile after every optimisation tier
before committing the next, larger refactor.  Confirm where the bottleneck moved
on real data.

---

## 7. Recorded toolchain

Recorded at harness creation time (June 2026) and written to `env.json` on every
`capture`/`profile` run:

| Component | Version | Notes |
|-----------|---------|-------|
| torch | 2.7.1+cu118 | **CUDA 11.8**, not the CUDA 12.6/cu126 documented in the Dockerfile.  This gap matters for Phases 3/5 GPU-library selection. |
| CUDA (runtime) | 11.8 | |
| open3d | 0.19.0 | Newer than the `~0.15` spec in CLAUDE.md |
| numpy | 2.1.2 | |
| GPU | NVIDIA GeForce RTX 3090 Ti | Driver 595.79 |
| py-spy | 0.4.2 | Sampling profiler (installed into `f2s3` env) |
| f2s3 | 0.0.post60 | Built from source |

---

## 8. Optional / advanced tools

| Tool | Status | Command |
|------|--------|---------|
| **nvtop** | Not installed | `sudo apt install nvtop` — interactive GPU monitor (nicer than `nvidia-smi dmon`). |
| **Nsight Systems** | Not verified | `nsys profile python harness/run_harness.py ...` — kernel/transfer/idle-gap timeline.  Useful for the Phase 3 supervoxel micro-launch investigation.  Requires CUDA toolkit install. |

Neither `nvtop` nor Nsight Systems is required for HARN-04 acceptance.
`py-spy` + `nvidia-smi dmon` + `torch.profiler` satisfy the requirement.

---

*HARN-04 runbook — Phase 1, Plan 3 — June 2026*
