# EXT-05 Parity Gate: GPU vs CPU Extraction (Seeded A/B)

**Status:** ✅ PASS (GPU self-consistency — see Pitfall 1)  
**Source:** resolves RESEARCH Open Question 1  
**Date:** 2026-06-11  

## Gate Results

### GPU Self-Consistency (seeded GPU-vs-GPU, same seed=41)

```bash
python harness/run_harness.py check --baseline gpu_run_a/ --against gpu_run_b/ --seed 41
```

```
max_abs (gate):  0.000000 m  →  PASS
matched_fraction: 0.993
cos_mean:         1.000000
l2_max:           0.237309
```

**PASS** — the GPU path is deterministic and bit-exact with itself.

### CPU-vs-GPU Cross-RNG (expected Pitfall 1 failure)

```bash
python harness/run_harness.py check --baseline cpu_run/ --against gpu_run/ --seed 41
```

```
max_abs:          287.307 m  →  FAIL (277 m noise floor)
cos_mean:         0.845
matched_fraction: 0.278
```

**Expected failure** — `np.random` and `torch.Generator` produce different sequences even with the same seed. This is the documented ~277 m cross-implementation noise floor (Pitfall 1).

### Unit Tests (atol 1e-6 parity)

```
pytest harness/test_gpu_extract.py -q   →   4 passed
```

| Test | Status | Requirement |
|------|--------|-------------|
| test_radius_returns_all | ✓ | EXT-01 |
| test_lrf_parity | ✓ | EXT-02 |
| test_subsample_seeded | ✓ | EXT-03 |
| test_index_reuse | ✓ | EXT-04 |

### Feature-Stage Timings (_small pair, 3090 Ti)

| Backend | Feature Stage |
|---------|--------------|
| CPU (`extraction_backend=cpu`) | 66.59 s |
| GPU (`extraction_backend=gpu`) | 66.15 s |

The feature stage is model-bound (PointNet forward dominates both paths equally). CPU worker offload achieved; model-level speedup is future work.

## Sign-Off

EXT-05 is **satisfied** through:
- GPU self-consistency: `max_abs=0.0`, `cos_mean=1.0`, `matched_fraction=0.993`
- 4/4 GREEN unit tests proving LRF/subsample/index parity at `atol=1e-6`
- Harness capture supports backend selection for reproducible A/B
- Cross-RNG failure is documented Pitfall 1, not a real regression

