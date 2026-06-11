# EXT-05 Parity Gate: GPU vs CPU Extraction (Seeded A/B)

**Status:** ✅ decided  
**Source:** resolves RESEARCH Open Question 1  
**Date:** 2026-06-11  

## Gate Decision

The EXT-05 gate is a **seeded CPU-vs-GPU within-implementation A/B**
comparison. Both backends run on the same `_small` epoch pair with the same
`extraction_seed`, and the results are compared via the Phase 1 harness. This
is NOT a comparison against the stored unseeded FP32 baseline (the ~277 m
cross-implementation noise floor makes that meaningless — Phase 1 D-07/D-08).

## Procedure

```bash
# 1. Capture the CPU run (same seed, extraction_backend=cpu)
python harness/run_harness.py capture --out cpu_run/ --seed 41

# 2. Capture the GPU run (same seed, extraction_backend=gpu)
python harness/run_harness.py capture --out gpu_run/ --seed 41

# 3. Check GPU against CPU (seeded within-implementation A/B)
python harness/run_harness.py check --baseline cpu_run/ --against gpu_run/ --seed 41
```

The CPU-same-seed run is the **oracle** (not the stored unseeded baseline).

## Passing Criteria (BINDING)

The EXT-05 gate **PASSES** when ALL of the following hold:

1. **Displacement `max_abs`** is within the Phase 1 within-implementation band
   (currently ≤ 5 cm per the Phase 1 D-04 cm-level tolerance; the exact band
   is set by the harness `--threshold` default).

2. **Descriptor `cos_mean` ≥ 0.9995** — cosine similarity over the 64-dim
   descriptor vectors from `parity.descriptor_drift`. This is the **binding
   floor** recorded here as a concrete number (not an "e.g." or
   recommendation). The LRF math is `eigh`-equivalent to the CPU `eig` path
   within `atol` 1e-6 (RESEARCH Pattern 2, Assumption A5), so under the same
   `extraction_seed` the per-point descriptors are numerically near-identical
   and `cos_mean` should sit essentially at 1.0. The 0.9995 floor leaves
   headroom for FP32 reduction-order noise on the batched `eigh` and the
   vectorized `alpha`/`beta` weighting (eqs. 3-5).

## Rationale

- **Within-implementation comparison only:** The unseeded FP32 baseline stored
  from Phase 1 uses a different RNG state for `np.random.choice`, so
  descriptor-by-descriptor comparison is inherently noisy. A deterministic
  default seed (D-04) makes within-implementation parity reproducible and
  meaningful.

- **Seeded gate, not statistical:** The cross-implementation noise floor is
  ~277 m (Phase 1 Harness Decision D-07), which dwarfs the expected GPU-shape
  parity. The seeded A/B eliminates the noise floor entirely by using the
  same sequence at every nondeterministic fork, leaving only the real
  algorithm-shape differences.

- **`cos_mean ≥ 0.9995` is the fine-grained check:** Even if displacement
  passes the `max_abs` band, descriptor drift reveals subtle LRF/eigh bugs
  that would otherwise land undetectably in a ~5 cm displacement gate.

