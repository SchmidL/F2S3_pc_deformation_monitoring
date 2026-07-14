"""T4 - Spatial spread of the top-k feature matches (the decisive test).

Map the top-10 feature matches back to target coordinates via xyz_t[Ik].

    distinctive point  -> its 10 best candidates sit close together, near the
                          true partner. Small spread.
    ambiguous  point  -> its 10 best candidates are scattered right across the
                          tile. Spread approaches the tile diameter.

Spread is a *geometric* statement about ambiguity, whereas r2/depth are purely
feature-space statements. This is the metric that most directly explains why a
1-NN matcher lands in a wrong blob far from the true partner.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import OUT_DIR, Tile, cross_knn_cached, ensure_out, tile_ids

K = 20
TOPK = 10  # spread is measured over the top-10 candidates


def spreads(cand):
    """cand: [N, TOPK, 3]. Returns max pairwise distance and std-radius per point."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    c = torch.as_tensor(cand, dtype=torch.float32, device=device)
    mx = torch.empty(c.shape[0])
    sd = torch.empty(c.shape[0])
    chunk = 8192
    for s in range(0, c.shape[0], chunk):
        b = c[s:s + chunk]                              # [B, TOPK, 3]
        pw = torch.cdist(b, b)                          # [B, TOPK, TOPK]
        mx[s:s + chunk] = pw.amax(dim=(1, 2)).cpu()
        centre = b.mean(dim=1, keepdim=True)
        sd[s:s + chunk] = (b - centre).norm(dim=2).mean(dim=1).cpu()
    return mx.numpy(), sd.numpy()


def fan_plot(ax, tile, Ik, idx, title, col):
    src = tile.xyz_s[idx]
    tgt = tile.xyz_t[Ik[idx, :TOPK]]
    ax.scatter(tile.xyz_s[::20, 0], tile.xyz_s[::20, 1], s=1, c='0.85', zorder=0)
    for t in tgt:
        ax.plot([src[0], t[0]], [src[1], t[1]], '-', color=col, lw=1, alpha=0.8, zorder=1)
    ax.scatter(tgt[:, 0], tgt[:, 1], s=28, c=col, marker='o', zorder=2, label='top-10 candidates')
    ax.scatter([src[0]], [src[1]], s=70, c='black', marker='*', zorder=3, label='source point')
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(fontsize=8, loc='best')


def main():
    ensure_out()
    for tid in tile_ids():
        tile = Tile(tid)
        print(tile)
        if tile.xyz_t is None:
            print('  xyz_t missing in the features archive - re-run with the patched core.py.')
            continue

        _, Ik = cross_knn_cached(tile, k=K)
        cand = tile.xyz_t[Ik[:, :TOPK]]                 # [Ns, TOPK, 3]
        spread_max, spread_std = spreads(cand)

        np.savez_compressed(os.path.join(OUT_DIR, f't4_tile_{tid}.npz'),
                            spread=spread_max, spread_std=spread_std)

        extent = np.linalg.norm(tile.xyz_t.max(axis=0) - tile.xyz_t.min(axis=0))
        inl, out = tile.trusted, tile.residual_outlier
        print(f'  tile diagonal ~ {extent:.1f} m')
        print(f'  spread  median {np.median(spread_max):.1f} m   p90 {np.percentile(spread_max, 90):.1f} m')
        print(f'  spread  mean: trusted {spread_max[inl].mean():.1f} m   '
              f'residual outlier {spread_max[out].mean():.1f} m')

        fig = plt.figure(figsize=(17, 10))
        ax0 = fig.add_subplot(2, 2, 1)
        ax0.hist(spread_max, bins=120, color='steelblue')
        ax0.axvline(extent, color='crimson', ls='--', label=f'tile diagonal {extent:.0f} m')
        ax0.set_yscale('log')
        ax0.set_xlabel('spread of top-10 target candidates [m]')
        ax0.set_ylabel('count (log)')
        ax0.set_title(f'T4 spatial spread, tile {tid}')
        ax0.legend()

        ax1 = fig.add_subplot(2, 2, 2)
        sc = ax1.scatter(tile.xyz_s[:, 0], tile.xyz_s[:, 1], c=spread_max, s=2, cmap='inferno',
                         vmin=0, vmax=np.percentile(spread_max, 99))
        ax1.set_aspect('equal')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.set_title('T4 spread map')
        fig.colorbar(sc, ax=ax1, label='spread [m]')

        # one clearly distinctive and one clearly ambiguous example, side by side
        order = np.argsort(spread_max)
        fan_plot(fig.add_subplot(2, 2, 3), tile, Ik, int(order[len(order) // 100]),
                 f'distinctive: spread {spread_max[order[len(order)//100]]:.1f} m', 'seagreen')
        fan_plot(fig.add_subplot(2, 2, 4), tile, Ik, int(order[-1]),
                 f'ambiguous: spread {spread_max[order[-1]]:.1f} m', 'crimson')

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f't4_spread_tile_{tid}.png'), dpi=130)
        plt.close(fig)
        print(f'  -> {OUT_DIR}/t4_spread_tile_{tid}.png')


if __name__ == '__main__':
    main()
