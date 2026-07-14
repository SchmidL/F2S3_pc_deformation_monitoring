"""T2 - Cross distance profile (source -> target feature space).

Computes and caches the top-k=20 feature matches from feat_s into feat_t.
Everything downstream (T3, T4, T5, T6) reuses that cache.

Metrics per source point:
    d1        distance to the best target feature
    r2        d1 / d2        (the classical Lowe ratio)
    r10       d1 / d10       (does the ambiguity go deeper than the 2nd NN?)
    flatness  (d10 - d1) / d1

The key plot is the mean distance profile d1..d20 for the trusted group
(|v| < 15 m) vs the gross-outlier group (|v| >= 40 m). A flat profile in the
outlier group means all 20 candidates are near-tied - which a k=2 ratio test
can only partly see.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import INLIER_M, OUT_DIR, OUTLIER_M, Tile, cross_knn_cached, ensure_out, tile_ids

K = 20


def main():
    ensure_out()
    for tid in tile_ids():
        tile = Tile(tid)
        print(tile)
        Dk, Ik = cross_knn_cached(tile, k=K)

        d1 = Dk[:, 0]
        eps = 1e-12
        r2 = d1 / (Dk[:, 1] + eps)
        r10 = d1 / (Dk[:, 9] + eps)
        flatness = (Dk[:, 9] - d1) / (d1 + eps)

        np.savez_compressed(os.path.join(OUT_DIR, f't2_tile_{tid}.npz'),
                            d1=d1, r2=r2, r10=r10, flatness=flatness)

        # compare the field the user actually plots: survivors of the supervoxel filter,
        # split into the trusted ones and the residual blobs
        inl, out = tile.trusted, tile.residual_outlier
        print(f'  trusted (survived, |v|<{INLIER_M:g}m): {inl.sum()} pts   '
              f'residual outliers (survived, |v|>={OUTLIER_M:g}m): {out.sum()} pts')
        print(f'  r2   mean: inliers {r2[inl].mean():.3f}  outliers {r2[out].mean():.3f}')
        print(f'  r10  mean: inliers {r10[inl].mean():.3f}  outliers {r10[out].mean():.3f}')

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))

        h = ax[0].hist2d(d1, r2, bins=[120, 120], cmap='viridis',
                         norm=matplotlib.colors.LogNorm())
        ax[0].set_xlabel('d1 (distance to best target feature)')
        ax[0].set_ylabel('r2 = d1/d2  (Lowe ratio)')
        ax[0].set_title(f'T2 match quality, tile {tid}')
        fig.colorbar(h[3], ax=ax[0], label='count')

        ranks = np.arange(1, K + 1)
        for mask, lab, col in ((inl, f'trusted (|v| < {INLIER_M:g} m)', 'seagreen'),
                               (out, f'residual outlier (|v| >= {OUTLIER_M:g} m)', 'crimson'),
                               (~tile.survived, 'rejected by supervoxel filter', 'slategrey')):
            if mask.sum() == 0:
                continue
            prof = Dk[mask].mean(axis=0)
            sd = Dk[mask].std(axis=0)
            ax[1].plot(ranks, prof, '-o', ms=3, color=col, label=lab)
            ax[1].fill_between(ranks, prof - sd, prof + sd, color=col, alpha=0.15)
        ax[1].set_xlabel('feature-space neighbour rank')
        ax[1].set_ylabel('feature distance')
        ax[1].set_title('T2 mean top-20 distance profile')
        ax[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f't2_profile_tile_{tid}.png'), dpi=130)
        plt.close(fig)
        print(f'  -> {OUT_DIR}/t2_profile_tile_{tid}.png')


if __name__ == '__main__':
    main()
