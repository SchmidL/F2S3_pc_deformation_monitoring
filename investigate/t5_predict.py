"""T5 - Do the ambiguity metrics actually predict the outliers?

Joins the per-point metrics from T2/T3/T4 with the gross-outlier flag
(|v| >= 40 m) taken from the output txt, which is row-aligned with feat_s.

For each metric we report:
    AUC             separability of outlier vs non-outlier (0.5 = useless)
    outlier rate    as a function of the metric's percentile

The whole point: if AUC(depth) or AUC(spread) clearly beats AUC(r2), then the
ambiguity is deeper than the 2nd nearest neighbour, and a plain Lowe ratio test
is the wrong tool. Metrics are sign-flipped where needed so that "higher score
= more likely an outlier".
"""

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, OUTLIER_M, Tile, auc, ensure_out, tile_ids


def load_metrics(tid):
    """All per-point metrics, oriented so that a HIGH value means MORE ambiguous."""
    t2 = np.load(os.path.join(OUT_DIR, f't2_tile_{tid}.npz'))
    t3 = np.load(os.path.join(OUT_DIR, f't3_tile_{tid}.npz'))
    m = {
        'r2 (Lowe, d1/d2)': t2['r2'],
        'r10 (d1/d10)': t2['r10'],
        'flatness': -t2['flatness'],   # flat profile => low flatness => ambiguous
        'depth_0.05': t3['depth_0.05'].astype(float),
        'depth_0.1': t3['depth_0.1'].astype(float),
        'depth_0.2': t3['depth_0.2'].astype(float),
        'd1 (raw match dist)': t2['d1'],
    }
    p4 = os.path.join(OUT_DIR, f't4_tile_{tid}.npz')
    if os.path.exists(p4):
        m['spread (top-10, m)'] = np.load(p4)['spread']
    p1 = os.path.join(OUT_DIR, f't1_tile_{tid}.npz')
    if os.path.exists(p1):
        m['n_eps (source collapse)'] = np.load(p1)['n_eps'].astype(float)
    return m


def main():
    ensure_out()
    for tid in tile_ids():
        tile = Tile(tid)
        metrics = load_metrics(tid)
        y = tile.is_outlier
        print(f'{tile}\n  outliers (|v|>={OUTLIER_M:g} m): {y.sum()} / {len(y)} ({y.mean():.2%})')

        rows = []
        for name, v in metrics.items():
            a = auc(v, y)
            # a metric that predicts the *absence* of outliers is just as useful,
            # so report the strength of separation either way
            rows.append((name, a, max(a, 1 - a)))
        rows.sort(key=lambda r: -r[2])

        csv_path = os.path.join(OUT_DIR, f't5_auc_tile_{tid}.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['metric', 'auc', 'separability'])
            for name, a, sep in rows:
                w.writerow([name, f'{a:.4f}', f'{sep:.4f}'])

        print(f'  {"metric":<26} {"AUC":>7} {"separability":>13}')
        for name, a, sep in rows:
            print(f'  {name:<26} {a:>7.3f} {sep:>13.3f}')

        # outlier rate vs metric percentile
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
        pct = np.arange(0, 100, 5)
        for name, v in metrics.items():
            edges = np.percentile(v, np.append(pct, 100))
            rate = []
            for i in range(len(pct)):
                lo, hi = edges[i], edges[i + 1]
                m = (v >= lo) & (v <= hi) if i == len(pct) - 1 else (v >= lo) & (v < hi)
                rate.append(y[m].mean() if m.sum() else np.nan)
            ax[0].plot(pct + 2.5, np.array(rate) * 100, '-o', ms=3, label=name)
        ax[0].axhline(y.mean() * 100, color='k', ls=':', label='base rate')
        ax[0].set_xlabel('metric percentile (higher = more ambiguous)')
        ax[0].set_ylabel(f'outlier rate [%]  (|v| >= {OUTLIER_M:g} m)')
        ax[0].set_title(f'T5 outlier rate vs metric percentile, tile {tid}')
        ax[0].legend(fontsize=8)

        names = [r[0] for r in rows][::-1]
        seps = [r[2] for r in rows][::-1]
        cols = ['seagreen' if s > 0.7 else 'goldenrod' if s > 0.6 else 'indianred' for s in seps]
        ax[1].barh(names, seps, color=cols)
        ax[1].axvline(0.5, color='k', ls=':', label='no separation')
        ax[1].set_xlim(0.5, 1.0)
        ax[1].set_xlabel('separability (AUC, orientation-corrected)')
        ax[1].set_title('T5 which metric best separates the outliers?')
        ax[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f't5_predict_tile_{tid}.png'), dpi=130)
        plt.close(fig)
        print(f'  -> {OUT_DIR}/t5_predict_tile_{tid}.png  and  {csv_path}')


if __name__ == '__main__':
    main()
