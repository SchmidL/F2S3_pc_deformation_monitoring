"""T3 - Ambiguity depth: how many target candidates are near-tied with the best?

    depth_eps(i) = #{ j <= k : Dk[i,j] <= (1+eps) * Dk[i,0] }

depth = 1 means the best match stands alone. depth >> 2 means a whole cluster of
target features is effectively tied for first place - exactly what a k=2 ratio
test cannot see, because it only ever compares d1 against d2.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, Tile, cross_knn_cached, ensure_out, tile_ids

K = 20
EPSILONS = (0.05, 0.10, 0.20)


def depths(Dk, eps):
    """#neighbours within (1+eps) of the best distance, counting the best itself."""
    return (Dk <= (1.0 + eps) * Dk[:, [0]]).sum(axis=1).astype(np.int32)


def main():
    ensure_out()
    for tid in tile_ids():
        tile = Tile(tid)
        print(tile)
        Dk, _ = cross_knn_cached(tile, k=K)

        d = {f'depth_{e:g}': depths(Dk, e) for e in EPSILONS}
        np.savez_compressed(os.path.join(OUT_DIR, f't3_tile_{tid}.npz'), **d)

        inl, out = tile.is_inlier, tile.is_outlier
        for e in EPSILONS:
            v = d[f'depth_{e:g}']
            print(f'  eps={e:.2f}: median {np.median(v):.0f}  '
                  f'frac depth>2 {np.mean(v > 2):.1%}  |  '
                  f'mean inlier {v[inl].mean():.2f}  mean outlier {v[out].mean():.2f}')

        fig, ax = plt.subplots(1, 3, figsize=(17, 5))

        for e in EPSILONS:
            v = d[f'depth_{e:g}']
            ax[0].hist(v, bins=np.arange(1, K + 2) - 0.5, histtype='step', lw=2,
                       label=f'eps={e:g}')
        ax[0].set_yscale('log')
        ax[0].set_xlabel('ambiguity depth (# near-tied target candidates)')
        ax[0].set_ylabel('count (log)')
        ax[0].set_title(f'T3 ambiguity depth, tile {tid}')
        ax[0].legend()

        main_eps = 0.10
        v = d[f'depth_{main_eps:g}']
        for mask, lab, col in ((inl, 'trusted (|v|<15 m)', 'seagreen'),
                               (out, 'gross outlier (|v|>=40 m)', 'crimson')):
            if mask.sum() == 0:
                continue
            ax[1].hist(v[mask], bins=np.arange(1, K + 2) - 0.5, histtype='step', lw=2,
                       density=True, color=col, label=lab)
        ax[1].set_xlabel(f'depth (eps={main_eps:g})')
        ax[1].set_ylabel('density')
        ax[1].set_title('T3 depth: trusted vs outlier')
        ax[1].legend()

        sc = ax[2].scatter(tile.xyz_s[:, 0], tile.xyz_s[:, 1], c=v, s=2, cmap='inferno',
                           vmin=1, vmax=max(2, np.percentile(v, 99)))
        ax[2].set_aspect('equal')
        ax[2].set_xlabel('x [m]')
        ax[2].set_ylabel('y [m]')
        ax[2].set_title(f'T3 depth map (eps={main_eps:g})')
        fig.colorbar(sc, ax=ax[2], label='depth')

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f't3_depth_tile_{tid}.png'), dpi=130)
        plt.close(fig)
        print(f'  -> {OUT_DIR}/t3_depth_tile_{tid}.png')


if __name__ == '__main__':
    main()
