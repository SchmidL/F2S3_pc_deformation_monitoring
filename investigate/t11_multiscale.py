"""T11 - does a multi-scale descriptor reduce feature-space ambiguity?

Hypothesis
----------
The investigation (FINDINGS.md) traced the residual outlier blobs to descriptor
ambiguity: on self-similar terrain the *entire* top-k feature neighbourhood of a
source point is near-tied, and the tied candidates are scattered across the tile.
A single-scale descriptor can only see what is inside its one radius, so on a
uniform debris slope it has nothing to distinguish one patch from another.

    H11: computing the descriptor at several radii and concatenating the
         per-scale vectors adds context that is not self-similar at the fine
         scale, which sharpens the top-k profile and pulls the tied candidates
         back together in space.

This script compares two or more finished F2S3 runs on the SAME epoch pair -
normally one single-scale baseline and one `--multiscale_radii` run - on metrics
that are computed the same way for every run.

Metrics
-------
Descriptor side (independent of any outlier label):

    r10 = d1/d10   ratio of the 1st to the 10th feature distance. 1.0 = the top-10
                   are perfectly tied, small = the best match stands out.
                   Reported as the median and as the ambiguous fraction r10 > 0.7
                   (the anchor threshold used in T8/T9).
                   CAVEAT - r10 is only comparable between runs of the SAME descriptor
                   dimension. A multi-scale L2 distance is the root-mean of the
                   per-scale distances, d^2 = (1/S) sum_s d_s^2, and averaging over S
                   partly independent channels shrinks the relative spread of the
                   distance distribution. Every ratio d_i/d_j is therefore pulled
                   mechanically toward 1 as S grows, whatever the descriptor's real
                   distinctiveness. Compare across dimensions with `spread`, the raw
                   outlier share and the coherence instead.
    spread         max pairwise distance between the top-10 candidates IN SPACE.
                   Large = the tied candidates sit all over the tile, so picking
                   the wrong one costs tens of metres.

Field side:

    raw 1-NN       median |v| and the gross-outlier share (|v| >= 40 m) over every
                   source point, before the supervoxel filter.
    survivors      density (share of source points kept by the filter), residual
                   outlier share, median |v| - this is the field the operator plots.
    coherence      median residual of each survivor vector to the median vector of
                   its 16 nearest survivors, and the share below 2 m. This is the
                   honest quality measure: unlike the 40 m count it does not
                   depend on the outlier proxy at all.

Usage
-----
    python t11_multiscale.py \
        --run baseline=../results/TCH_gedi_r6_investigate \
        --run multiscale=../results/TCH_gedi_ms2_6_12_investigate

Both runs must have been produced with `--save_interim`.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from common import OUT_DIR, OUTLIER_M, Tile, ensure_out, knn, tile_ids

TOPK = 10          # the top-k window the ambiguity metrics are measured over
R10_AMBIG = 0.7    # r10 above this counts as "ambiguous" (T8/T9 anchor rule)
KNN_COH = 16       # neighbours used for the coherence residual
COH_OK = 2.0       # m, a vector counts as coherent below this residual
KNN_CHUNK = 2048   # cdist rows per chunk; 2048 x ~800k floats fits in 24 GB


def cross_knn(tile, label, k=20):
    """kNN from feat_s into feat_t, cached per run label.

    ``common.cross_knn_cached`` keys its cache on the tile id alone, which would
    collide across runs of the same tile, so T11 keeps its own cache.
    """
    ensure_out()
    path = os.path.join(OUT_DIR, f't11_crossknn_{label}_tile_{tile.tile_id}_k{k}.npz')
    if os.path.exists(path):
        z = np.load(path)
        return z['Dk'], z['Ik']
    Dk, Ik = knn(tile.feat_s, tile.feat_t, k=k, chunk=KNN_CHUNK)
    np.savez_compressed(path, Dk=Dk, Ik=Ik)
    return Dk, Ik


def candidate_spread(cand):
    """Max pairwise distance inside each candidate set. cand: [N, TOPK, 3]."""
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    c = torch.as_tensor(cand, dtype=torch.float32, device=device)
    out = torch.empty(c.shape[0])
    for s in range(0, c.shape[0], 8192):
        b = c[s:s + 8192]
        out[s:s + 8192] = torch.cdist(b, b).amax(dim=(1, 2)).cpu()
    return out.numpy()


def coherence(xyz, vec):
    """Residual of every vector to the median vector of its KNN_COH neighbours.

    A displacement field of a moving rock mass is locally smooth, so a vector that
    disagrees with its neighbours is wrong regardless of its magnitude. The query
    point is included in its own neighbourhood (as in T10) so the measure is a
    deviation-from-local-consensus, not a leave-one-out prediction error.
    """
    if xyz.shape[0] <= KNN_COH:
        return np.full(xyz.shape[0], np.nan)
    _, idx = cKDTree(xyz).query(xyz, k=KNN_COH)
    local_median = np.median(vec[idx], axis=1)
    return np.linalg.norm(vec - local_median, axis=1)


def analyse(label, results_dir):
    """All T11 metrics for one finished run, per tile."""
    rows = []
    for tid in tile_ids(results_dir):
        tile = Tile(tid, results_dir=results_dir)
        Dk, Ik = cross_knn(tile, label, k=20)

        r10 = Dk[:, 0] / np.maximum(Dk[:, TOPK - 1], 1e-12)
        spread = (candidate_spread(tile.xyz_t[Ik[:, :TOPK]])
                  if tile.xyz_t is not None else np.full(tile.n_src, np.nan))

        vec = tile.corr[:, 3:6] - tile.corr[:, 0:3]
        surv = tile.survived
        res = coherence(tile.xyz_s[surv], vec[surv])

        rows.append(dict(
            run=label, tile=tid, dim=int(tile.feat_s.shape[1]),
            n_src=int(tile.n_src), n_tgt=int(tile.n_tgt),
            # descriptor distinctiveness
            r10_median=float(np.median(r10)),
            r10_ambiguous_pct=float(100 * (r10 > R10_AMBIG).mean()),
            spread_median_m=float(np.nanmedian(spread)),
            # raw 1-NN field
            raw_mag_median_m=float(np.median(tile.mag)),
            raw_outlier_pct=float(100 * (tile.mag >= OUTLIER_M).mean()),
            # survivor field
            survivor_pct=float(100 * surv.mean()),
            survivor_mag_median_m=float(np.median(tile.mag[surv])),
            residual_outlier_pct=float(100 * (tile.mag[surv] >= OUTLIER_M).mean()),
            # coherence of the plotted field
            coh_residual_median_m=float(np.nanmedian(res)),
            coh_ok_pct=float(100 * np.nanmean(res < COH_OK)),
        ))
        print(f'  {label} tile {tid}: D={rows[-1]["dim"]} '
              f'r10={rows[-1]["r10_median"]:.3f} '
              f'raw_out={rows[-1]["raw_outlier_pct"]:.1f}% '
              f'surv={rows[-1]["survivor_pct"]:.1f}% '
              f'res_out={rows[-1]["residual_outlier_pct"]:.2f}% '
              f'coh_ok={rows[-1]["coh_ok_pct"]:.1f}%')
    return rows


COLUMNS = [
    ('dim', 'D', '{:d}'),
    ('r10_median', 'median r10', '{:.3f}'),
    ('r10_ambiguous_pct', 'r10>0.7 [%]', '{:.1f}'),
    ('spread_median_m', 'top10 spread [m]', '{:.1f}'),
    ('raw_outlier_pct', 'raw >=40m [%]', '{:.1f}'),
    ('survivor_pct', 'survivors [%]', '{:.1f}'),
    ('residual_outlier_pct', 'residual >=40m [%]', '{:.2f}'),
    ('survivor_mag_median_m', 'survivor |v| [m]', '{:.2f}'),
    ('coh_residual_median_m', 'coh residual [m]', '{:.2f}'),
    ('coh_ok_pct', 'coherent <2m [%]', '{:.1f}'),
]


def print_table(rows):
    runs = []
    for r in rows:
        if r['run'] not in runs:
            runs.append(r['run'])
    width = max(len(h) for _, h, _ in COLUMNS) + 2
    header = 'metric'.ljust(width) + ''.join(r.rjust(16) for r in runs)
    print('\n' + header)
    print('-' * len(header))
    for key, head, fmt in COLUMNS:
        line = head.ljust(width)
        for run in runs:
            vals = [r[key] for r in rows if r['run'] == run]
            line += fmt.format(np.mean(vals) if key != 'dim' else int(vals[0])).rjust(16)
        print(line)
    print()


def plot(rows, path):
    runs = []
    for r in rows:
        if r['run'] not in runs:
            runs.append(r['run'])
    panels = [('r10_median', 'median $r_{10}=d_1/d_{10}$\n(lower = more distinctive)'),
              ('spread_median_m', 'median top-10 spread [m]\n(lower = tied candidates closer)'),
              ('raw_outlier_pct', 'raw 1-NN $|v|\\geq$40 m [%]\n(lower = better)'),
              ('residual_outlier_pct', 'residual $|v|\\geq$40 m [%]\n(lower = better)'),
              ('survivor_pct', 'survivors [%]\n(higher = denser field)'),
              ('coh_ok_pct', 'coherent <2 m [%]\n(higher = better)')]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    x = np.arange(len(runs))
    for ax, (key, title) in zip(axes.ravel(), panels):
        vals = [np.mean([r[key] for r in rows if r['run'] == run]) for run in runs]
        ax.bar(x, vals, color=['#4c72b0', '#dd8452', '#55a868', '#c44e52'][:len(runs)])
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=15, ha='right', fontsize=8)
        ax.set_title(title, fontsize=9)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'figure -> {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, metavar='LABEL=PATH',
                    help='a finished F2S3 results dir to include, repeatable')
    args = ap.parse_args()

    ensure_out()
    rows = []
    for spec in args.run:
        label, _, path = spec.partition('=')
        if not path:
            raise SystemExit(f'--run needs LABEL=PATH, got {spec!r}')
        print(f'analysing {label} ({path})')
        rows += analyse(label, path)

    print_table(rows)
    with open(os.path.join(OUT_DIR, 't11_multiscale.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    plot(rows, os.path.join(OUT_DIR, 't11_multiscale.png'))


if __name__ == '__main__':
    main()
