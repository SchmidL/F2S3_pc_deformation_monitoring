# Feature-space ambiguity in GeDi — findings

Run: `results/TCH_gedi_r6_investigate`, TCH sample pair (20210924 → 20220831),
`--voxel_grid_size 0.2 --neighborhood_radius 6 --supervoxel_radius 4 --descriptor gedi`.
One tile: 861,552 source points, 775,358 target points, GeDi D=32.

## 0. A correction to the brief

The handoff's §8 assert (`feat_s.shape[0] == corr.shape[0] == out.shape[0]`) is **wrong**,
and quietly so. `filter_correspondences` writes only the points that survive the supervoxel
filter, reordered by supervoxel (`save_coords[inlier_idx]`), so the output txt has 189,037
rows against 861,552 source points. Anything that joined metrics to the txt by row index
would have silently compared unrelated points.

We therefore recompute the raw 1-NN magnitude from `corr` (which *is* row-aligned with
`feat_s`) and remap the txt back onto source rows by exact 6-D coordinate match — verified
to zero magnitude error. That yields two distinct fields:

| field | points | median \|v\| | ≥ 40 m |
|---|---|---|---|
| raw 1-NN (what the matcher produced) | 861,552 | 96.1 m | **67.2 %** |
| after supervoxel filter (what you plot) | 189,034 (21.9 %) | 7.61 m | 4.63 % |

The second row reproduces the reported ≈7.5 m / ≈4.6 %, so the run is faithful.

**The headline is the first row.** The matcher's own output is two-thirds gross outliers.
The supervoxel filter is not polishing a good field — it is rescuing a broken one, and the
4.6 % residual blobs are what leaks through.

## 1. Is it descriptor ambiguity? Yes — but not *collapse*

**T1 is a clean negative.** There is no descriptor collapse: not one source point has a
near-identical twin (median distance to the nearest *other* source feature is 0.21; zero
points have a neighbour within eps = 0.021). GeDi descriptors are individually distinct.
`n_eps` has AUC exactly 0.500 — it predicts nothing, because it is identically zero.

So the failure is **not** "the descriptor emits the same vector everywhere". It is that the
descriptors, while distinct, are not *separated enough to rank correctly*:

- **T2**: the top-20 distance profile is nearly flat for outliers. Mean `r2` is 0.919 for
  trusted points vs 0.951 for outliers — the 2nd-best match is only ~5-8 % worse than the
  best, for everybody.
- **T3**: at eps = 0.1 the median ambiguity depth is 4, and **69 % of points have depth > 2**.
  A Lowe ratio test compares d1 against d2 only; by construction it cannot see a 4-way tie.
- **T4**: the top-10 candidates of a typical source point are scattered **277 m (median)
  across a 479 m tile**. The correct partner is not being ranked above candidates hundreds
  of metres away.

Caveat: `depth_0.2` has median 20 = k, i.e. it is **censored** at the top-20 cutoff and
under-measures the true depth. Re-run with k=100 if the exact value matters.

## 2. Does depth beat ratio? Yes, clearly

AUC against the outlier flag (higher = better separation):

| metric | raw target | residual target |
|---|---|---|
| **r10 (d1/d10)** | **0.781** | **0.797** |
| flatness | 0.781 | 0.797 |
| depth_0.2 | 0.765 | 0.772 |
| depth_0.1 | 0.712 | 0.699 |
| d1 (raw match distance) | 0.680 | 0.791 |
| depth_0.05 | 0.641 | 0.629 |
| **r2 (Lowe, d1/d2)** | **0.640** | **0.636** |
| spread (top-10) | 0.629 | 0.607 |
| n_eps | 0.500 | 0.500 |

`r10` (0.797) beats the Lowe ratio `r2` (0.636) decisively. The ambiguity is deeper than the
2nd neighbour, exactly as hypothesised. (`flatness` ties `r10` because it is the monotone
transform `1/r10 − 1` — a sanity check on the AUC code, not an independent signal.)

## 3. The AUC ranking is a trap — read the Pareto instead

**AUC picked the wrong filter.** Spread ranks second-*worst* by AUC (0.607), yet it is by far
the most precise filter in the regime we actually operate in. From T6, acting on the raw 1-NN
field alone:

| filter | keeps (of raw) | outliers among kept |
|---|---|---|
| none | 100 % | 67.2 % |
| mutual NN | 23.2 % | 50.2 % |
| depth_0.1 ≤ 2 | 30.7 % | 46.3 % |
| ratio < 0.80 | 2.0 % | 14.6 % |
| **spread < 10 m** | **3.0 %** | **2.4 %** |

At *comparable retention*, spread is **6× cleaner than the tightest ratio test, and denser**.
It takes the raw field from 67.2 % outliers to 2.4 %.

Why AUC misses this: AUC scores *global ranking* across the whole distribution. Spread is
useless over most of its range (nearly everything has a huge spread — the descriptor is
globally non-distinctive, so even trusted points average 220 m) but the small-spread tail is
almost pure. It is a low-recall, very-high-precision filter, and AUC averages that virtue away.
The brief's decision rule (§6.2, "prefer whichever metric has higher AUC") would have selected
`r10` and discarded the best filter available.

**Mutual NN is the other casualty.** It is popular, and it is nearly worthless here: on its own
it still leaves 50.2 % outliers. Ambiguity this deep defeats a symmetry check, because the
wrong match is wrong in both directions.

## 4. Choosing a filter (T6 Pareto, composed with the supervoxel filter)

Density is quoted against all source points, so 21.9 % = today's baseline.

| filter | kept % | outlier % | median \|v\| |
|---|---|---|---|
| none (current 1-NN) | 21.9 | 4.63 | 7.61 |
| depth_0.1 ≤ 3 | 16.0 | 2.73 | 7.61 |
| depth_0.1 ≤ 2 | 12.8 | 2.42 | 7.62 |
| ratio < 0.95 | 12.5 | 3.10 | 7.62 |
| mutual NN | 8.6 | 1.54 | 7.49 |
| mutual & depth ≤ 3 | 7.6 | 1.24 | 7.53 |
| **spread < 10 m** | **2.1** | **0.47** | **7.32** |
| mutual & ratio<0.90 & spread<10 m | 0.5 | 0.13 | 7.31 |

Note `depth_0.1 ≤ 2` vs `ratio < 0.95`: at the *same* density (12.8 % vs 12.5 %), depth gives
2.42 % outliers where ratio gives 3.10 %. Like-for-like, depth wins — the T5 conclusion holds.

No filter is both denser and cleaner than the baseline; every one trades density for purity.
Which knee to take depends on what happens next:

- **If you densify by interpolation** (the plan in §6.4), precision dominates and recall is
  cheap to recover. Take **`spread < 10 m`**: 18.5 k points at 0.47 % outliers — a clean
  skeleton to interpolate a smooth glacier-flow field from.
- **If you want to keep the field mostly as-is**, take **`depth_0.1 ≤ 3`**: 73 % of current
  density, 41 % fewer outliers, median unchanged.
- **Balanced**: `mutual & depth ≤ 3` — 7.6 % density, 1.24 % outliers.

Reassuringly, the median displacement is stable (7.3–7.6 m) under every filter. The central
estimate is not in question; only the blobs are.

## 5. What this means for the descriptor

The mechanism is confirmed, but the honest reading is stronger than "add a ratio test":
**GeDi at r=6 m is not distinctive enough on this ice to rank the correct partner first**, and
no rejection test can conjure information the descriptor did not encode. Every filter above
buys purity by *throwing away* points, because the ranking itself is not trustworthy.

Filtering is the right short-term fix. The longer-term fix is a descriptor that separates —
a larger/multi-scale patch radius, or geometric verification (rigid consistency among a
supervoxel's matches) that uses *spatial* evidence rather than feature distance. T4 is the
hint: spatial coherence of the candidate set carries information that feature distance does not.

## Reproducing

```bash
cd investigate
python t2_cross_profile.py   # builds the top-20 cross-kNN cache; T3-T6 reuse it
python t3_depth.py
python t4_spread.py
python t1_collapse.py        # slowest: 861k x 861k self-search
python t5_predict.py
python t6_filters.py
```
Outputs (PNG + CSV/NPZ per task) land in `investigate/out/`.
