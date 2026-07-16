# Feature-space ambiguity in the F2S3/GeDi correspondence search — investigation report

**Scene:** rocky landslide with moraine-like structures ("TCH"), two TLS epochs
2021-09-24 → 2022-08-31 (~11 months apart), sample clouds.
**Run:** `results/TCH_gedi_r6_investigate`, branch `feature_descriptor_gedi`, configuration
`--voxel_grid_size 0.2 --neighborhood_radius 6 --supervoxel_radius 4 --descriptor gedi
--save_interim`. One tile: N_s = 861,552 source points, N_t = 775,358 target points,
GeDi descriptor dimension D = 32.

---

## 1. Motivation and problem statement

F2S3 [1] estimates a 3D displacement vector field between two point-cloud epochs by
(i) computing a deep local feature descriptor per point — here GeDi [2] — (ii) assigning
each source point its **single nearest neighbour** in feature space (1-NN, approximate
search via HNSW [3]), and (iii) rejecting outliers by enforcing local rigid consistency
inside supervoxels.

After parameter tuning, the displacement field on this scene is largely coherent
(median |v| ≈ 7.6 m), but ~4.6 % of the **plotted** vectors are gross outliers: vectors
of tens to hundreds of metres that point "all over the place", although the supervoxel
filter should have removed them. They are not isolated: they form **spatially clustered
blobs** in visually low-texture regions (uniform debris/moraine surfaces).

The matching step contains **no ambiguity rejection** of any kind — every source point
receives a partner, however unreliable. Two classical remedies were proposed: Lowe's
ratio test [4] and a mutual-nearest-neighbour check. Both examine only the top-1/top-2
matches. Before implementing either, this investigation tests a sharper hypothesis:

> **H:** On self-similar terrain the GeDi descriptor is not distinctive: for many source
> points the *entire* top-k feature neighbourhood — not just the 2nd neighbour — is
> near-tied, and the near-tied candidates are scattered across the tile. That ambiguity
> is the mechanism behind the surviving outlier blobs, and a top-2 ratio test can only
> partially capture it.

From H we derive five questions, each answered by one task:

| # | Question | Task |
|---|---|---|
| Q1 | Are the descriptors themselves degenerate (many near-identical vectors)? | T1 |
| Q2 | How sharply does the best match stand out from the runners-up? | T2, T3 |
| Q3 | Where do the near-tied candidates sit *in space*? | T4 |
| Q4 | Which ambiguity measure best predicts the outliers? | T5 |
| Q5 | Which rejection rule yields the best density/purity trade-off? | T6 |
| — | Are the answers robust to the outlier-label threshold? | T7 |

---

## 2. Data, labels, and terminology

### 2.1 Point sets and fields

`--save_interim` exports, per tile: the feature matrices `feat_s [N_s×D]`,
`feat_t [N_t×D]`, the point coordinates `xyz_s`, `xyz_t` (added by commit `642bbcf` for
this investigation), the raw 1-NN correspondences `corr [N_s×6]`
(source xyz | matched target xyz), and the final filtered field
`displacement_magnitude_tile_*.txt`.

Two displacement fields must be distinguished:

- **Raw 1-NN field** — the matcher's own output before any filtering: one vector per
  source point, magnitude `|v_i| = ‖corr_i[3:6] − corr_i[0:3]‖`. This is the field the
  descriptor is directly responsible for.
- **Survivor field** — the subset kept by the supervoxel filter; this is what is
  written to the output txt and what the operator plots.

**A correction discovered during validation:** the output txt is *not* row-aligned with
the feature matrices, contrary to the investigation brief. `filter_correspondences`
(`core.py`) writes only the surviving points, reordered by supervoxel — 189,037 rows
against 861,552 source points. Any analysis joining metrics to the txt by row index
would silently compare unrelated points. We therefore recompute all magnitudes from
`corr` (which *is* row-aligned with `feat_s` by construction) and recover the survivor
set by exact 6-D coordinate matching of txt rows into `corr` rows, verified to zero
magnitude error on all 189,037 rows (6 collide with the 66 duplicated source
coordinates and resolve via the full 6-D key).

| field | points | median \|v\| | ≥ 40 m |
|---|---|---|---|
| raw 1-NN | 861,552 (100 %) | 96.1 m | **67.2 %** |
| survivors | 189,034 (21.9 %) | 7.61 m | 4.63 % |

The survivor row reproduces the operator-reported ≈7.5 m / ≈4.6 %, confirming the run
is faithful. The raw row is the first substantive finding: **the matcher's own output is
two-thirds gross outliers.** The supervoxel filter is not polishing a good field — it is
rescuing a broken one, and the 4.6 % blobs are the leakage.

### 2.2 The outlier label (proxy, not ground truth)

No independent reference exists for this scene — no GNSS, no manually verified
correspondences, no identified stable areas. All "outlier" labels in this report are
therefore a **proxy**: a correspondence is labelled a gross outlier when |v| ≥ θ, with
θ = 40 m as the working value. Two observations make this proxy defensible:

1. **The survivor magnitude distribution is strongly bimodal.** 93.3 % of survivors lie
   below 15 m (the coherent-motion mode); 3.3 % lie beyond 100 m (the scattered far
   tail); only **0.66 % of survivors fall anywhere between 25 m and 60 m**. Any
   threshold inside that valley labels nearly the same set of points.

2. **The conclusions are threshold-invariant (T7, §5.6).** Recomputing every ranking at
   θ ∈ {20, 25, 30, 40, 60} m changes no ordering — neither of the metrics nor of the
   filters.

What the proxy *cannot* do is validate individual vectors: a wrong match that lands,
by chance, at a plausible displacement is invisible to it. All purity numbers below
are therefore lower bounds on the true error rate (see Limitations, §7).

Recurring point groups:

| group | definition | size |
|---|---|---|
| raw outlier | \|v\| ≥ 40 m, any source point | 578,677 (67.2 % of N_s) |
| survivor | kept by the supervoxel filter | 189,034 (21.9 % of N_s) |
| trusted | survivor **and** \|v\| < 15 m | 176,435 |
| residual outlier | survivor **and** \|v\| ≥ 40 m — the blobs in the plotted field | 8,756 (4.63 % of survivors) |
| **baseline** | the current pipeline output = the survivor field: 21.9 % density, 4.63 % outliers | — |

### 2.3 Does the exact kNN speak for the pipeline's HNSW matcher?

All analyses below use an exact (brute-force, GPU-chunked) k-nearest-neighbour search,
while F2S3 uses approximate HNSW (`hnswlib`, M=12, efC=efS=300). If the two disagreed
materially, our metrics would describe a different matcher than the one producing the
field. Verification on this tile: the HNSW-chosen partner equals the exact 1-NN for
**99.960 %** of source points (343 disagreements out of 861,552); in every disagreement
the HNSW pick still lies within the exact top-20, with a median feature-distance excess
of 1.5 % (p95: 8.6 %). This confirms the ">99 % NN recall" noted in `core.py` and makes
exact-kNN-based metrics valid statements about the production matcher.

---

## 3. Metric definitions

Notation: for source point i, let d₁ ≤ d₂ ≤ … ≤ d₂₀ be its sorted L2 distances to the
20 nearest target features (computed once in T2, cached, reused by all later tasks).

**Lowe ratio** — `r2 = d₁/d₂` [4]. Values near 1 mean the runner-up is nearly as good
as the best match; Lowe's classical criterion keeps a match only if r2 < ~0.8. It sees
exactly one competitor.

**Deep ratio and flatness** — `r10 = d₁/d₁₀` and `flatness = (d₁₀ − d₁)/d₁`. These are
**ad-hoc extensions defined for this investigation** (specified in the handoff brief);
we claim no literature source. They generalise the ratio idea to the 10th competitor.
Note `flatness = 1/r10 − 1`, i.e. the two are monotone transforms of each other and
therefore identical under any rank-based evaluation — we report both only as a
computational cross-check.

**Ambiguity depth** — `depth_ε = #{ j ≤ 20 : d_j ≤ (1+ε)·d₁ }`, ε ∈ {0.05, 0.1, 0.2}.
Interpretation of ε: a *relative distance tolerance*. depth_0.1 counts how many target
candidates are within 10 % of the best match's distance — how many are "effectively
tied for first place". depth = 1 means the winner stands alone; depth ≫ 2 is precisely
the situation a top-2 ratio test cannot represent. The three ε values probe how strict
"tied" is taken to be; there is no canonical choice, which is why we report all three
and let T5/T7 judge.

**Spatial spread** — map the top-10 candidates to their coordinates
`{xyz_t[I_j] : j = 1…10}` and take the **maximum pairwise Euclidean distance** among
those 10 points (a std-radius variant was computed as well and behaves identically).
Spread is the only *geometric* measure: it asks not "how close are the runner-up
distances" but "where would the alternative matches put this point".

**T1 self-similarity** — `d_self1(i)` = distance from feat_s[i] to its nearest *other*
source feature; `n_eps(i)` = number of other source features within
ε₁ = 0.1 · median(d_self1). This ε₁ is **not** the depth-ε above: it is an absolute
feature-space radius, deliberately tied to the data scale ("one tenth of the typical
nearest-neighbour distance"), so that n_eps counts *near-duplicates* rather than merely
close descriptors. The factor 0.1 is a convention from the brief; since the result is
identically zero (§5.1), no tuning of it would change the conclusion.

**AUC** — Area Under the ROC (Receiver Operating Characteristic) Curve, the standard
threshold-free score for how well a scalar metric separates two classes. Equivalent
formulation: AUC = P(metric of a random outlier > metric of a random non-outlier).
AUC = 0.5 means the metric carries no information about the label; AUC = 1.0 means
perfect separation. We compute it rank-based with tie correction. There is **no
relation between AUC and the ε parameters** — ε shapes a metric's values; AUC then
scores those values against the outlier labels of §2.2.

---

## 4. Expected outcomes under H

Stating expectations before results (per good practice — deviations are flagged in §5):

- **E1 (T1):** if GeDi degenerates on self-similar surfaces, large regions should show
  d_self1 ≈ 0 / n_eps ≫ 0 ("descriptor collapse").
- **E2 (T2):** trusted points should show a distance profile that rises steeply after
  d₁ (distinct winner); residual outliers a **flat** profile (all 20 near-equal). In an
  ideal world the d₁-vs-r2 distribution would be concentrated at r2 ≪ 1 — a clear
  winner for every point; ambiguity manifests as mass piling up against r2 → 1.
- **E3 (T3):** residual outliers should have depth ≫ 2 far more often than trusted
  points; a substantial fraction of all points at depth > 2 would prove a top-2 test
  structurally blind here.
- **E4 (T4):** trusted points → small spread (candidates cluster at the true partner);
  ambiguous points → spread approaching the tile diameter.
- **E5 (T5):** under H, depth/spread should outrank r2 in AUC.
- **E6 (T6):** a depth- or spread-based rejection should reach a better density/purity
  operating point than the ratio test.

---

## 5. Results

### 5.1 Q1 — No descriptor collapse (T1, expectation E1 **not** met)

Median d_self1 = 0.211; ε₁ = 0.021; **n_eps = 0 for every one of the 861,552 source
points** — not a single descriptor has even one near-duplicate among its own epoch.
GeDi does not collapse on this terrain; each descriptor is individually distinct.

This negative sharpens H rather than refuting it: the failure mode is not "the
descriptor emits the same vector everywhere" but "the descriptors, while distinct, do
not place the correct partner *first* by a reliable margin" (§5.2–5.3). It also
predicts, correctly, that n_eps carries no information (AUC exactly 0.500 in T5).

### 5.2 Q2a — The winner barely stands out (T2)

Group means: r2 = 0.919 (trusted) vs 0.951 (residual outliers); r10 = 0.766 vs 0.858.
Against E2: even for *trusted* points the runner-up is only ~8 % worse than the winner —
far from the ideal-world picture — and the mean top-20 profile of residual outliers is
visibly flatter than that of trusted points. Two consequences:

1. Classical Lowe filtering (r2 < 0.8) would discard the overwhelming majority of
   *correct* matches on this scene (see T6: it keeps 2.0 % of points).
2. The r2 separation between the groups (0.919 vs 0.951) is small; the r10 separation
   (0.766 vs 0.858) is nearly three times larger — first evidence that the ambiguity
   signal lives deeper than rank 2. (The r2/r10/flatness values are visualised in the
   T5 percentile plot; the T2 figure shows the d₁-vs-r2 density and the group profiles.)

### 5.3 Q2b — The ambiguity is deep (T3, E3 met)

At ε = 0.1, the median ambiguity depth is **4**, and **69.3 % of all source points have
depth > 2**; group means are 2.92 (trusted) vs 5.81 (residual outliers). For over
two-thirds of the points, the situation a top-2 ratio test models — one winner, one
challenger — simply does not exist; there is a *cluster* of challengers.

Caveat: at ε = 0.2 the median depth equals 20 = k, i.e. the value is **right-censored**
by the top-20 search. True depths at loose tolerance are larger than reported; k = 100
would be needed to measure them (does not affect any ranking below, which use ε = 0.1).

### 5.4 Q3 — The near-ties are scattered across the whole tile (T4, E4 partially met)

Median spread of the top-10 candidates: **277 m on a tile with a 479 m diagonal**;
p90 = 343 m. The alternative matches of a typical point are not local jitter around the
true partner — they are distributed over the entire scene, which is exactly the
mechanism that turns a near-tie into a hundreds-of-metres blunder.

Deviation from E4: *trusted* points also show large mean spread (220 m vs 262 m for
residual outliers). Global candidate scatter is near-universal; what distinguishes
reliable points is the **small-spread tail** (spread < 10–20 m), not a low average.
This asymmetry explains spread's paradoxical behaviour in T5 vs T6 below.

### 5.5 Q4 — Which metric predicts the outliers? (T5)

AUC of each metric against the outlier label, for both targets (raw = all points,
label |v| ≥ 40 m; residual = survivors only, same label):

| metric | raw target | residual target |
|---|---|---|
| r10 (= flatness, monotone) | 0.781 | **0.797** |
| d₁ (raw match distance) | 0.680 | 0.791 |
| depth_0.2 | 0.765 | 0.772 |
| depth_0.1 | 0.712 | 0.699 |
| depth_0.05 | 0.641 | 0.629 |
| **r2 (Lowe)** | 0.640 | **0.636** |
| spread (top-10) | 0.629 | 0.607 |
| n_eps (T1) | 0.500 | 0.500 |

E5 is met for the *depth direction* of H: **r10 (0.797) decisively outranks r2
(0.636)** — the discriminative information sits between ranks 2 and 10, invisible to a
top-2 test. Two unexpected results:

- **d₁ alone is nearly as predictive as r10 on the residual target** (0.791): among
  survivors, a large absolute match distance is itself a strong warning sign.
- **spread scores second-worst** — see §5.6 for why this AUC value misleads.

### 5.6 Q5 — Filter simulation and the Pareto view (T6)

Each candidate rejection rule is a boolean mask over source points, so the resulting
field can be evaluated offline. Because a feature-space filter would run *before* the
(unchanged) supervoxel filter, the operating point of a rule is the **intersection** of
both filters, quoted as: retained density (% of all N_s source points; baseline
21.9 %) and outlier rate among the kept points (θ = 40 m proxy).

**Pareto terminology:** rule A *dominates* rule B if A is both denser and cleaner. The
*Pareto frontier* is the set of non-dominated rules — the menu of rational choices; the
"knee" is the frontier point where giving up more density stops buying meaningful
purity. Full table in `out/t6_pareto_tile_000.csv`; key rows:

| filter | kept % of N_s | outlier % among kept | median \|v\| | outlier % acting alone on raw field |
|---|---|---|---|---|
| none — baseline | 21.9 | 4.63 | 7.61 | 67.2 |
| depth_0.1 ≤ 3 | 16.0 | 2.73 | 7.61 | 49.8 |
| depth_0.1 ≤ 2 | 12.8 | 2.42 | 7.62 | 46.3 |
| ratio r2 < 0.95 | 12.5 | 3.10 | 7.62 | 54.8 |
| mutual NN | 8.6 | 1.54 | 7.49 | 50.2 |
| ratio r2 < 0.80 (Lowe's value) | 1.4 | 0.67 | 7.78 | 14.6 |
| **spread < 10 m** | **2.1** | **0.47** | **7.32** | **2.4** |
| mutual ∧ r2<0.9 ∧ spread<10 m | 0.5 | 0.13 | 7.31 | 0.6 |

Findings, expectation E6 against observation:

1. **Like-for-like, depth beats ratio** — at matched density (12.8 % vs 12.5 %),
   depth ≤ 2 yields 2.42 % outliers vs 3.10 % for r2 < 0.95. The T5 conclusion carries
   through to operating points. E6 confirmed for depth-vs-ratio.
2. **Spread is the best single filter by a wide margin**, despite its poor AUC. Acting
   alone on the raw field it cuts 67.2 % → 2.4 % outliers while keeping 3.0 % of points
   — at *comparable retention*, ~6× cleaner than the tightest ratio test. The
   resolution of the T5/T6 contradiction: AUC scores a metric's ranking across its
   *whole* range, but spread is uninformative over most of its range (§5.4: nearly
   everything scatters widely) and nearly pure in its low tail. A low-recall /
   high-precision signal loses at AUC and wins as a thresholded filter. **Methodological
   consequence: selecting a filter by AUC alone — as the brief's decision rule
   proposed — would have picked r10 and discarded the strongest filter.**
3. **Mutual NN alone is insufficient** (50.2 % outliers on the raw field): when the
   descriptor cannot rank the true partner first, the wrong assignment is often
   symmetric, so the mutuality check passes.
4. Median |v| is stable (7.3–7.8 m) under every rule — the central motion estimate was
   never in question; the filters differ only in how much of the tail they admit.
5. No rule dominates the baseline: **every gain in purity costs density.** The choice
   of knee depends on the downstream use (§8).

### 5.7 Robustness to the label threshold (T7)

All of the above at θ ∈ {20, 25, 30, 40, 60} m:

| θ [m] | raw outlier % | residual outlier % | AUC r2 | AUC r10 | AUC depth_0.1 | AUC spread | AUC d₁ |
|---|---|---|---|---|---|---|---|
| 20 | 71.5 | 5.51 | 0.629 | 0.779 | 0.687 | 0.591 | 0.784 |
| 25 | 70.4 | 5.02 | 0.635 | 0.794 | 0.697 | 0.600 | 0.794 |
| 30 | 69.4 | 4.93 | 0.635 | 0.793 | 0.696 | 0.601 | 0.793 |
| 40 | 67.2 | 4.63 | 0.636 | 0.797 | 0.699 | 0.607 | 0.791 |
| 60 | 61.8 | 4.35 | 0.636 | 0.796 | 0.699 | 0.613 | 0.788 |

Filter outlier rates move in parallel (e.g. spread < 10 m: 0.76 → 0.43 % from θ = 20 to
60 m) and **no ordering — of metrics or of filters — changes anywhere in the range**.
Combined with the bimodality of §2.2, the proxy label is as solid as it can be made
without ground truth.

---

## 6. Synthesis

**H is confirmed in its mechanism, refuted in one detail, and exceeded in one respect.**

- Confirmed: the surviving outlier blobs are caused by feature-space ambiguity that is
  *deeper than the 2nd neighbour* (median 4-way ties; 69 % of points beyond the reach
  of a top-2 test; r10 ≫ r2 in predictive power), and the near-tied candidates scatter
  across the full tile, which is what converts ambiguity into large blunders.
- Refuted detail: there is **no descriptor collapse** (T1). The descriptors are
  distinct; they are insufficiently *separated* to rank reliably. No rejection test can
  recover information the descriptor did not encode — every filter buys purity with
  density.
- Exceeded: the raw 1-NN field is 67 % gross outliers. The problem is not a blemish on
  a good matching stage; the supervoxel filter is currently carrying the pipeline.

## 7. Limitations

1. **Proxy labels.** All purity numbers rest on |v| ≥ θ. Wrong matches with plausible
   magnitudes are uncounted; true purity is somewhat worse than reported. Mitigated by
   bimodality (§2.2) and threshold invariance (§5.7); removable only with reference
   data (GNSS / stable areas / manual verification).
2. **One tile, one scene, one descriptor configuration.** Everything above describes
   TCH sample data at r = 6 m with GeDi-32. Generalisation to the full-resolution
   clouds, other scenes, or DIP-64 is untested (the sample was chosen because it
   reproduces the production outlier pattern).
3. **k-censoring.** depth_0.2 is right-censored at k = 20 (§5.3).
4. **Filter thresholds are not optimised.** T6 evaluates the grid from the brief
   (ratio ∈ {0.8…0.95}, depth ≤ {2,3,5}, spread < {5,10,20} m); the true Pareto
   frontier may lie between grid points.
5. **The spread filter needs the top-10 search** (and mutual needs a reverse search):
   adopting it in the pipeline changes the correspondence stage from k=1 to k≥10
   HNSW queries plus a k=1 reverse query — a runtime cost not yet measured.

## 8. Recommendations

1. **Short term — implement a rejection filter in `compute_correspondences`,**
   opt-in via CLI flags, choosing the knee by downstream use:
   - densification planned (RBF interpolation / `--fill_gaps_c2c`): take
     **spread < 10 m** — 18.5 k points at 0.47 % outliers is a clean skeleton, and
     recall is cheap to recover by interpolation;
   - field to be used as-is: take **depth_0.1 ≤ 3** — 73 % of current density with
     41 % fewer outliers and unchanged median.
2. **Medium term — exploit spatial evidence, not only feature distance.** T4 is the
   pointer: the geometric coherence of the candidate set carries information the
   feature distances do not. Candidates: geometric verification of supervoxel match
   sets, or matching with spatial regularisation.
3. **Longer term — a more distinctive descriptor** on this terrain type (multi-scale /
   larger support, or fine-tuning on the scene class), since §6 shows filtering cannot
   add information, only remove points.
4. **For validation:** identify stable areas (bedrock) in the full scene, or a handful
   of manually verified correspondences, to replace the proxy label in the next
   iteration.

## 9. Reproducibility

```bash
cd investigate
python t2_cross_profile.py            # builds the top-20 cross-kNN cache (T3-T7 reuse it)
python t3_depth.py
python t4_spread.py
python t1_collapse.py                 # slowest: 861k x 861k self-search
python t5_predict.py
python t6_filters.py
python t7_threshold_sensitivity.py
```

Each task writes a `.png` (figures) and `.csv`/`.npz` (per-point metrics, keyed by
source row) to `investigate/out/`. `common.py` holds the loader (with the §2.1
alignment correction and its asserts), the chunked exact kNN, and the tie-corrected
AUC. Interim data comes from re-running the production configuration with
`--save_interim` (plus commit `642bbcf`, which adds `xyz_s`/`xyz_t` to the features
archive — required by T4/T6).

## References

[1] Gojcic, Z., et al.: *F2S3: Robustified determination of 3D displacement vector
fields using deep learning.* Journal of Applied Geodesy, 2020.
[2] Poiesi, F., Boscaini, D.: *Learning general and distinctive 3D local deep
descriptors for point cloud registration.* IEEE T-PAMI, 2022.
[3] Malkov, Y. A., Yashunin, D. A.: *Efficient and robust approximate nearest neighbor
search using hierarchical navigable small world graphs.* IEEE T-PAMI, 2018.
[4] Lowe, D. G.: *Distinctive image features from scale-invariant keypoints.* IJCV
60(2), 2004. (Origin of the ratio test, Sec. 7.1.)
