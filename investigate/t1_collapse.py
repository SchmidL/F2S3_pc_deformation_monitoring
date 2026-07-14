"""T1 - Descriptor collapse: how self-similar are the SOURCE descriptors?

For each source feature we find its nearest *other* source feature. If GeDi
collapses on self-similar ice, large connected regions carry near-identical
descriptors, and no target-side test can rescue them.

Metrics per source point:
    d_self1  distance to the nearest other source feature
    n_eps    number of source features within eps = 0.1 * median(d_self1)
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import OUT_DIR, Tile, ensure_out, knn, tile_ids


def run_tile(tile):
    # k=2: the first neighbour is the point itself (distance 0), so take the second.
    D, _ = knn(tile.feat_s, tile.feat_s, k=2)
    d_self1 = D[:, 1]

    eps = 0.1 * float(np.median(d_self1))

    # n_eps: count source features within eps. Chunked radius count, no full matrix.
    q = torch.as_tensor(tile.feat_s, dtype=torch.float32,
                        device='cuda' if torch.cuda.is_available() else 'cpu')
    n_eps = np.empty(tile.n_src, dtype=np.int32)
    chunk = 4096
    for s in range(0, q.shape[0], chunk):
        dist = torch.cdist(q[s:s + chunk], q)
        # subtract 1 to exclude the self-match
        n_eps[s:s + chunk] = ((dist <= eps).sum(dim=1) - 1).cpu().numpy()

    return d_self1, n_eps, eps


def main():
    ensure_out()
    for tid in tile_ids():
        tile = Tile(tid)
        print(tile)
        d_self1, n_eps, eps = run_tile(tile)

        np.savez_compressed(os.path.join(OUT_DIR, f't1_tile_{tid}.npz'),
                            d_self1=d_self1, n_eps=n_eps, eps=eps)

        frac_collapsed = float((n_eps >= 5).mean())
        print(f'  median d_self1 = {np.median(d_self1):.4f}   eps = {eps:.4f}')
        print(f'  n_eps: median {np.median(n_eps):.0f}  p90 {np.percentile(n_eps, 90):.0f}  '
              f'max {n_eps.max()}')
        print(f'  fraction of source points with >=5 near-identical twins: {frac_collapsed:.1%}')

        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        ax[0].hist(d_self1, bins=120, color='steelblue')
        ax[0].axvline(eps, color='crimson', ls='--', label=f'eps={eps:.3f}')
        ax[0].set_xlabel('distance to nearest other source feature')
        ax[0].set_ylabel('count')
        ax[0].set_title(f'T1 self-similarity of source descriptors (tile {tid})')
        ax[0].legend()

        ax[1].hist(n_eps, bins=np.arange(0, min(n_eps.max(), 60) + 2) - 0.5, color='indianred')
        ax[1].set_yscale('log')
        ax[1].set_xlabel(f'# source features within eps ({eps:.3f})')
        ax[1].set_ylabel('count (log)')
        ax[1].set_title('T1 collapse multiplicity')

        sc = ax[2].scatter(tile.xyz_s[:, 0], tile.xyz_s[:, 1], c=n_eps, s=2,
                           cmap='inferno', vmin=0, vmax=np.percentile(n_eps, 99))
        ax[2].set_aspect('equal')
        ax[2].set_xlabel('x [m]')
        ax[2].set_ylabel('y [m]')
        ax[2].set_title('T1 collapse map (n_eps)')
        fig.colorbar(sc, ax=ax[2], label='n_eps')

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f't1_collapse_tile_{tid}.png'), dpi=130)
        plt.close(fig)
        print(f'  -> {OUT_DIR}/t1_collapse_tile_{tid}.png')


if __name__ == '__main__':
    main()
