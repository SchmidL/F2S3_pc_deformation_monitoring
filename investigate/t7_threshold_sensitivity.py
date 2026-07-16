"""T7 - Sensitivity of the conclusions to the outlier-label threshold.

Every result in this investigation rests on a proxy label: a correspondence is
called a gross outlier when its displacement magnitude |v| exceeds a threshold.
The 40 m value came from visual inspection of the field, not from a physical
bound, and no independent reference (GNSS, stable areas, validated field) exists
for this scene. This task therefore answers two questions:

1. Does the magnitude distribution itself support a threshold? If the survivors'
   |v| histogram is bimodal - a coherent-motion mode well below and a scattered
   tail well above - any threshold inside the gap yields the same labels, and the
   proxy is defensible. If the distribution is unimodal with a smooth tail, the
   label (and everything built on it) is soft.

2. Do the rankings that drive the conclusions change with the threshold? We
   recompute, for |v|_out in {20, 25, 30, 40, 60} m:
     - the raw and residual base rates,
     - the AUC of each T5 metric,
     - the survivor outlier rate of the key T6 filters.
   If the metric ordering (r10 > r2, etc.) and the filter ordering (spread
   cleanest, mutual weakest) are stable across thresholds, the conclusions do
   not hinge on the choice of 40 m.
"""

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, Tile, auc, cross_knn_cached, ensure_out, knn, tile_ids

THRESHOLDS = (20.0, 25.0, 30.0, 40.0, 60.0)
K = 20


def load_metrics(tid):
    t2 = np.load(os.path.join(OUT_DIR, f't2_tile_{tid}.npz'))
    t3 = np.load(os.path.join(OUT_DIR, f't3_tile_{tid}.npz'))
    t4 = np.load(os.path.join(OUT_DIR, f't4_tile_{tid}.npz'))
    return {
        'r2': t2['r2'], 'r10': t2['r10'], 'd1': t2['d1'],
        'depth_0.1': t3['depth_0.1'].astype(float),
        'spread': t4['spread'],
    }


def main():
    ensure_out()
    for tid in tile_ids():
        tile = Tile(tid)
        print(tile)
        m = load_metrics(tid)
        Dk, Ik = cross_knn_cached(tile, k=K)

        _, Ik_back = knn(tile.feat_t, tile.feat_s, k=1)
        filters = {
            'ratio<0.95': m['r2'] < 0.95,
            'mutual NN': Ik_back[Ik[:, 0], 0] == np.arange(tile.n_src),
            'depth<=3': m['depth_0.1'] <= 3,
            'spread<10m': m['spread'] < 10,
        }

        rows = []
        for thr in THRESHOLDS:
            out_raw = tile.mag >= thr
            surv = tile.survived
            row = {
                'threshold_m': thr,
                'raw_outlier_pct': 100 * out_raw.mean(),
                'residual_outlier_pct': 100 * out_raw[surv].mean(),
            }
            for name, v in m.items():
                row[f'auc_{name}'] = auc(v[surv], out_raw[surv])
            for name, keep in filters.items():
                final = keep & surv
                row[f'filter_{name}_outlier_pct'] = (
                    100 * out_raw[final].mean() if final.any() else float('nan'))
            rows.append(row)

        csv_path = os.path.join(OUT_DIR, f't7_sensitivity_tile_{tid}.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v)
                            for k, v in r.items()})

        print(f'  {"thr":>4} {"raw%":>6} {"res%":>6} | AUC: '
              f'{"r2":>6} {"r10":>6} {"depth":>6} {"spread":>6} {"d1":>6}')
        for r in rows:
            print(f'  {r["threshold_m"]:>4.0f} {r["raw_outlier_pct"]:>6.2f} '
                  f'{r["residual_outlier_pct"]:>6.2f} |      '
                  f'{r["auc_r2"]:>6.3f} {r["auc_r10"]:>6.3f} '
                  f'{r["auc_depth_0.1"]:>6.3f} {r["auc_spread"]:>6.3f} {r["auc_d1"]:>6.3f}')
        print(f'  filter outlier% at each threshold:')
        for name in filters:
            vals = ' '.join(f'{r[f"filter_{name}_outlier_pct"]:>6.2f}' for r in rows)
            print(f'    {name:<12} {vals}')

        fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

        # 1. magnitude distributions: is there a gap that justifies a threshold at all?
        bins = np.logspace(np.log10(0.05), np.log10(500), 160)
        ax[0].hist(np.clip(tile.mag, 0.05, 500), bins=bins, color='0.7',
                   label='raw 1-NN (all points)')
        ax[0].hist(np.clip(tile.mag[tile.survived], 0.05, 500), bins=bins,
                   color='steelblue', label='supervoxel survivors')
        for thr in THRESHOLDS:
            ax[0].axvline(thr, color='crimson', ls=':', lw=0.8)
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xlabel('|v| [m]')
        ax[0].set_ylabel('count')
        ax[0].set_title(f'T7 magnitude distribution, tile {tid}\n(dotted: tested thresholds)')
        ax[0].legend()

        thrs = [r['threshold_m'] for r in rows]
        for name, lab in (('r2', 'r2 (Lowe)'), ('r10', 'r10'), ('depth_0.1', 'depth_0.1'),
                          ('spread', 'spread'), ('d1', 'd1')):
            ax[1].plot(thrs, [r[f'auc_{name}'] for r in rows], '-o', ms=4, label=lab)
        ax[1].axhline(0.5, color='k', ls=':')
        ax[1].set_xlabel('outlier threshold [m]')
        ax[1].set_ylabel('AUC (residual target)')
        ax[1].set_title('T7 metric AUC vs threshold')
        ax[1].legend(fontsize=8)

        for name in filters:
            ax[2].plot(thrs, [r[f'filter_{name}_outlier_pct'] for r in rows],
                       '-o', ms=4, label=name)
        ax[2].plot(thrs, [r['residual_outlier_pct'] for r in rows], '-k', lw=2,
                   label='no filter (baseline)')
        ax[2].set_xlabel('outlier threshold [m]')
        ax[2].set_ylabel('outlier rate among survivors [%]')
        ax[2].set_title('T7 filter performance vs threshold')
        ax[2].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f't7_sensitivity_tile_{tid}.png'), dpi=130)
        plt.close(fig)
        print(f'  -> {OUT_DIR}/t7_sensitivity_tile_{tid}.png  and  {csv_path}')


if __name__ == '__main__':
    main()
