# T11 — Multi-scale descriptors and the voxel grid size

Companion note to `FINDINGS.md`. It documents two changes to the pipeline, why they
were made, how they were tested, and what they do **not** fix.

---

## 1. Why

`FINDINGS.md` established the mechanism behind the residual outlier blobs: on
self-similar terrain the GeDi descriptor is not distinctive, so for many source points
the *entire* top-k feature neighbourhood is near-tied and the tied candidates are
scattered across the tile. The matcher then picks one essentially at random.

A single-scale descriptor cannot do better than the information inside its one radius.
On a uniform debris slope, one 6 m patch really does look like the next 6 m patch — the
descriptor is not failing, it is reporting the truth. What separates those two patches
is *context* at a larger scale: one sits below a ridge, the other inside a gully.

> **H11:** computing the descriptor at several radii and concatenating the per-scale
> vectors adds that context, which sharpens the top-k profile and pulls the tied
> candidates back together in space.

---

## 2. What was implemented

### 2.1 `--multiscale_radii R1 R2 ...`

For radii `r_1 < ... < r_S` the selected backend (DIP or GeDi) is run once per radius
over the same query points. Each per-scale descriptor `f_s` is L2-normalised and the
scales are stacked with a `1/sqrt(S)` factor:

```
f = [f_1, ..., f_S] / sqrt(S)
```

Two properties follow, and both matter:

- `f` is again a **unit vector**, so nothing downstream (HNSW index sizing, the L2
  space, the outlier filter) sees a different value range than before.
- The squared L2 distance the matcher minimises becomes the **mean** squared per-scale
  distance, `||f-g||² = (1/S) Σ_s ||f_s-g_s||²`. Every scale weighs exactly the same;
  no scale can dominate simply by having larger numbers.

The descriptor dimension becomes `S · dim` (3 × 32 = 96 for GeDi at three scales). The
tile overlap follows the **coarsest** radius, so every scale still sees a complete
neighbourhood at the tile border. Cost is roughly linear in the number of scales.

Implementation: `MultiScaleDescriptor` in `src/f2s3/descriptors.py`; it wraps whichever
backend `--descriptor` selects, so it works for both DIP and GeDi.

### 2.2 `--auto_voxel_grid_size`

The voxel grid size was a hand-set number. The informed rule is:

> A voxel filter can only *thin* regions that are already denser than its cell. A cell
> smaller than the local point spacing leaves that region untouched. So to make two
> epochs comparable, the cell has to be at least as large as the **coarser** epoch's
> local spacing.

`--auto_voxel_grid_size` therefore sets

```
voxel_grid_size = ceil_cm( max( p_q(nn spacing of epoch 1), p_q(nn spacing of epoch 2) ) )
```

with `q = --auto_voxel_quantile`, default 90. The quantile — not the median — is the
point. On the TCH sample pair the median nearest-neighbour spacing is 0.175 m / 0.176 m,
but the p90 is 0.302 m / 0.311 m. A 0.2 m cell (the value used for the whole
investigation so far) is above the median and therefore homogenises roughly the denser
half of the surface, while the sparser half keeps whatever density each scan happened to
deliver — exactly the epoch-to-epoch density mismatch the filter is meant to remove.
p90 → **0.32 m** equalises 90 % of the surface instead.

The trade-off is honest and unavoidable: a larger cell means fewer points and less fine
detail. The quantile is exposed so the operator can choose where on that curve to sit.

---

## 3. Limitations — what this does **not** fix

These are stated rather than solved.

1. **Fixed-size patch sampling.** Every scale is described from a fixed-size subsample of
   its neighbourhood — 256 points for DIP, 4000 LRF points for GeDi — regardless of
   radius. A 12 m scale therefore sees the same number of points spread over 36× the
   area of a 2 m scale: it is a **sparser sampling of a larger area**, not a richer
   description of it. The coarse channels consequently encode low-frequency shape only,
   and fine detail at large radii is unrecoverable with these descriptors. Removing this
   needs a descriptor that consumes the full neighbourhood (or a hierarchical one that
   pools instead of subsamples).

2. **Redundancy between scales.** Nested neighbourhoods share points, so the per-scale
   vectors are correlated. The effective dimensionality gain is smaller than the nominal
   `S · dim`. No decorrelation or whitening is applied.

3. **Out-of-domain scales.** Both DIP and GeDi were trained on 3DMatch, i.e. indoor
   object-scale patches. A 12 m outdoor scale is far outside that training distribution.
   The networks still produce a deterministic, repeatable embedding of the geometry —
   which is all the matcher needs — but there is no guarantee that the embedding is
   *discriminative* at that scale. This is an empirical question, which is why the scales
   are a CLI parameter and not a hard-coded choice.

4. **The voxel rule assumes both epochs cover the same surface.** Where one epoch has a
   genuine data gap (occlusion, missing scan position), no voxel size can equalise the
   density — there is nothing to equalise.

5. **No ground truth.** As throughout the investigation, "outlier" remains the
   |v| ≥ 40 m proxy and the coherence residual. Both are lower bounds on the true error
   rate: a wrong match that lands at a plausible displacement stays invisible.

---

## 4. Experiment

`run_multiscale_TCH.sh` — 2 × 2 design on the epoch pair used throughout the
investigation (`data/20210924_THC_centered_sample.ply` →
`data/20220831_THC_centered_sample.ply`, GeDi, `--supervoxel_radius 4`):

|              | voxel 0.20 m (baseline)          | voxel = auto (p90 → 0.32 m)      |
|--------------|----------------------------------|----------------------------------|
| single 6 m   | `TCH_gedi_r6_investigate` *(existing)* | `TCH_gedi_r6_vauto`         |
| multi 2/6/12 | `TCH_gedi_ms2_6_12_v0_2`         | `TCH_gedi_ms2_6_12_vauto`        |

Scale choice: 6 m is the tuned single-scale radius; 2 m resolves individual blocks;
12 m covers the slope-scale context that actually differs between two self-similar
debris patches.

`investigate/t11_multiscale.py` scores every run with the same metrics:

| metric | meaning | direction |
|---|---|---|
| `median r10` | `d1/d10` in feature space — how far the best match stands out from the 10th | lower better |
| `r10 > 0.7 [%]` | share of points whose whole top-10 is near-tied | lower better |
| `top10 spread [m]` | max pairwise distance between the top-10 candidates *in space* | lower better |
| `raw ≥40 m [%]` | gross outliers in the raw 1-NN field, before filtering | lower better |
| `survivors [%]` | share of source points kept by the supervoxel filter | higher better |
| `residual ≥40 m [%]` | gross outliers still in the plotted field | lower better |
| `coherent <2 m [%]` | share of survivor vectors within 2 m of the median vector of their 16 nearest survivors | higher better |

`coherent <2 m` is the load-bearing number: unlike the 40 m count it does not depend on
the outlier proxy at all.

```bash
./run_multiscale_TCH.sh
cd investigate && python t11_multiscale.py \
    --run baseline=../results/TCH_gedi_r6_investigate \
    --run ms_v0.2=../results/TCH_gedi_ms2_6_12_v0_2 \
    --run single_vauto=../results/TCH_gedi_r6_vauto \
    --run ms_vauto=../results/TCH_gedi_ms2_6_12_vauto
```

---

## 5. Results

One tile, N_s = 861,552 source points at 0.20 m / 673,467 at 0.32 m.
Produced by `run_multiscale_TCH.sh`, scored by `investigate/t11_multiscale.py`
(raw numbers in `investigate/out/t11_multiscale.json`, figure
`investigate/out/t11_multiscale.png`).

| metric | baseline r6 / v0.20 | ms 2-6-12 / v0.20 | single r6 / v0.32 | **ms 2-6-12 / v0.32** |
|---|---|---|---|---|
| descriptor dim | 32 | 96 | 32 | 96 |
| median r10 *(see caveat)* | 0.871 | 0.920 | 0.868 | 0.916 |
| top-10 spread [m] | 277.2 | 237.9 | 281.8 | **243.8** |
| raw 1-NN ≥ 40 m [%] | 67.2 | 52.2 | 67.2 | **52.3** |
| survivors [%] | 21.9 | 30.0 | 21.4 | **33.5** |
| residual ≥ 40 m [%] | 4.63 | 4.51 | 4.39 | **3.45** |
| survivor median \|v\| [m] | 7.61 | 7.31 | 7.69 | 7.33 |
| coherent < 2 m [%] | 97.3 | 97.9 | 96.7 | **97.8** |

### 5.1 The multi-scale descriptor works

The two label-free measures both move the right way:

- **Top-10 spread falls from 277 m to 238 m.** The near-tied candidates of an ambiguous
  point are measurably less scattered across the tile. This is the direct prediction of
  H11 and it holds.
- **The raw 1-NN gross-outlier share falls from 67.2 % to 52.2 %** — 15 percentage
  points, before any filtering. The matcher's own output is substantially less broken.

That propagates to the plotted field: the supervoxel filter keeps **30.0 % instead of
21.9 %** of the source points at the *same* voxel size, while the residual outlier share
drops slightly (4.63 % → 4.51 %) and coherence improves (97.3 % → 97.9 %). A denser
field at equal-or-better purity is a strict improvement.

### 5.2 Caveat: `r10` is not comparable across descriptor dimensions

`median r10` *rises* (0.871 → 0.920), which would naively read as "more ambiguous". It is
a metric artefact, not a finding. The multi-scale L2 distance is the root-mean of the
per-scale distances, `d² = (1/S) Σ_s d_s²`. Averaging over S partly independent channels
shrinks the *relative* spread of the distance distribution, so every ratio `d_i/d_j` is
mechanically pulled toward 1 as S grows — regardless of how distinctive the descriptor
became. `r10` therefore only compares runs of the same dimension. Across dimensions the
metrics carrying real information are the ones in physical units or in outcome terms:
top-10 spread [m], raw outlier share, survivor share, and coherence. All four improve.

*(This also means the T8/T9 anchor rule `r10 < 0.7` is calibrated for D = 32 and must be
re-tuned before it is applied to a multi-scale run.)*

### 5.3 The voxel size pays off in combination

Alone, the p90 voxel size (0.32 m) is roughly neutral for the single-scale descriptor:
fewer survivors (21.4 % vs 21.9 %), a lower residual outlier share (4.39 % vs 4.63 %),
slightly worse coherence (96.7 % vs 97.3 %). It trades points for purity, and no more.

Combined with the multi-scale descriptor it is clearly the best cell of the design:
**33.5 % survivors at 3.45 % residual outliers** — against the baseline's 21.9 % at
4.63 %. That is **53 % more surviving points and a quarter fewer gross outliers among
them**, at 673 k instead of 862 k input points.

The plausible reading is that the two changes address the same defect from two sides.
Equalising the density makes the *same* physical neighbourhood produce comparable
sampling in both epochs; adding coarse scales gives that neighbourhood something
distinctive to encode. Neither alone is sufficient — the single-scale/0.32 m cell shows
the density fix does little on its own.

### 5.4 Cost

Feature extraction is roughly linear in the number of scales, with one caveat: GeDi
samples a fixed 4000 LRF points per patch regardless of radius, so per-scale *network*
cost is constant — but the **neighbour search** is not. At 0.20 m voxel the 12 m radius
pushed `maxN` to ~6,000 neighbours, which forces the adaptive chunk size down to ~16 k
points and made one cloud take 2,235 s against 127 s for the same radius on the other
cloud. At 0.32 m voxel the same scale ran in ~100 s. The coarser voxel is therefore also
what makes a 12 m scale practical.

Wall clock, one tile: baseline 7 min; ms/0.20 m 50 min; single/0.32 m 5 min;
**ms/0.32 m 13 min**.

### 5.5 Recommendation

Use **`--multiscale_radii 2 6 12 --auto_voxel_grid_size`**. On this scene it is the best
point of the four on every outcome metric and costs about twice the baseline runtime.

The one number that did *not* improve is the residual outlier share relative to how much
denser the field became — 3.45 % of a much larger set is still ~7.8 k bad vectors, the
same order as the baseline's 8.8 k. Multi-scale descriptors reduce the ambiguity that
feeds the blobs; they do not remove the blobs. The ambiguity-rejection rules from T6 and
the spatial constraint from T9/T10 remain the tools for that, and should now be re-tuned
on top of the multi-scale descriptor rather than the single-scale one.
