"""T6 - Offline filter simulation: retained density vs outlier rate.

Every candidate filter is a boolean mask over the source points, so we can
evaluate it directly against the already-computed field - no F2S3 re-run.

    retained density = % of source points that survive the filter
    outlier rate     = % of the SURVIVORS whose |v| >= 40 m

The deliverable is a Pareto plot. The knee is the filter that hits the outlier
target at the highest retained density; the holes it leaves get densified
afterwards (--fill_gaps_c2c or an RBF interpolation of the trusted vectors).
"""

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, OUTLIER_M, Tile, cross_knn_cached, ensure_out, knn, tile_ids

K = 20


def build_filters(tile, Dk, Ik):
    """name -> (family, boolean keep-mask)."""
    t2 = np.load(os.path.join(OUT_DIR, f't2_tile_{tile.tile_id}.npz'))
    t3 = np.load(os.path.join(OUT_DIR, f't3_tile_{tile.tile_id}.npz'))
    r2 = t2['r2']
    depth = t3['depth_0.1']

    filters = {'none (current 1-NN)': ('baseline', np.ones(tile.n_src, dtype=bool))}

    for thr in (0.80, 0.85, 0.90, 0.95):
        filters[f'ratio < {thr:.2f}'] = ('ratio', r2 < thr)

    # mutual NN: i survives if its best target maps back to i
    _, Ik_back = knn(tile.feat_t, tile.feat_s, k=1)
    mutual = Ik_back[Ik[:, 0], 0] == np.arange(tile.n_src)
    filters['mutual NN'] = ('mutual', mutual)

    for kk in (2, 3, 5):
        filters[f'depth_0.1 <= {kk}'] = ('depth', depth <= kk)

    spread = None
    p4 = os.path.join(OUT_DIR, f't4_tile_{tile.tile_id}.npz')
    if os.path.exists(p4):
        spread = np.load(p4)['spread']
        for dd in (5, 10, 20):
            filters[f'spread < {dd} m'] = ('spread', spread < dd)

    # combinations
    filters['mutual & ratio<0.90'] = ('combo', mutual & (r2 < 0.90))
    filters['mutual & depth<=3'] = ('combo', mutual & (depth <= 3))
    if spread is not None:
        filters['mutual & spread<10 m'] = ('combo', mutual & (spread < 10))
        filters['ratio<0.90 & spread<10 m'] = ('combo', (r2 < 0.90) & (spread < 10))
        filters['mutual & ratio<0.90 & spread<10 m'] = (
            'combo', mutual & (r2 < 0.90) & (spread < 10))
    return filters


def main():
    ensure_out()
    for tid in tile_ids():
        tile = Tile(tid)
        print(tile)
        Dk, Ik = cross_knn_cached(tile, k=K)
        filters = build_filters(tile, Dk, Ik)
        y = tile.mag >= OUTLIER_M

        rows = []
        for name, (fam, keep) in filters.items():
            # The supervoxel filter is not going away: a feature-space filter would sit
            # in FRONT of it, so what the user ends up plotting is the intersection.
            # Density is quoted against all source points, so it is comparable to the
            # 21.9 % the supervoxel filter alone retains.
            final = keep & tile.survived
            n = int(final.sum())
            dens = 100.0 * n / tile.n_src
            orate = 100.0 * y[final].mean() if n else float('nan')
            med = float(np.median(tile.mag[final])) if n else float('nan')
            # feature filter acting alone on the raw 1-NN field, for reference
            n_raw = int(keep.sum())
            orate_raw = 100.0 * tile.is_outlier[keep].mean() if n_raw else float('nan')
            rows.append(dict(filter=name, family=fam, retained_pct=dens, n_kept=n,
                             outlier_pct=orate, median_mag=med,
                             raw_retained_pct=100.0 * n_raw / tile.n_src,
                             raw_outlier_pct=orate_raw))

        csv_path = os.path.join(OUT_DIR, f't6_pareto_tile_{tid}.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f'{v:.3f}' if isinstance(v, float) else v)
                            for k, v in r.items()})

        print('  (filter AND supervoxel filter; density vs all source points)')
        print(f'  {"filter":<34} {"kept %":>7} {"outlier %":>10} {"med |v|":>9} '
              f'{"raw out %":>10}')
        for r in sorted(rows, key=lambda r: r['outlier_pct']):
            print(f'  {r["filter"]:<34} {r["retained_pct"]:>7.1f} '
                  f'{r["outlier_pct"]:>10.2f} {r["median_mag"]:>9.2f} '
                  f'{r["raw_outlier_pct"]:>10.1f}')

        fig, ax = plt.subplots(figsize=(10, 7))
        cols = {'baseline': 'black', 'ratio': 'tab:blue', 'mutual': 'tab:orange',
                'depth': 'tab:red', 'spread': 'tab:green', 'combo': 'tab:purple'}
        seen = set()
        for r in rows:
            fam = r['family']
            ax.scatter(r['retained_pct'], r['outlier_pct'], s=90, color=cols[fam],
                       marker='*' if fam == 'baseline' else 'o',
                       label=fam if fam not in seen else None, zorder=3)
            seen.add(fam)
            ax.annotate(r['filter'], (r['retained_pct'], r['outlier_pct']),
                        textcoords='offset points', xytext=(6, 4), fontsize=7)

        # Pareto frontier: no other filter is both denser and cleaner
        pts = sorted(rows, key=lambda r: -r['retained_pct'])
        front, best = [], float('inf')
        for r in pts:
            if r['outlier_pct'] < best:
                front.append(r)
                best = r['outlier_pct']
        front.sort(key=lambda r: r['retained_pct'])
        ax.plot([r['retained_pct'] for r in front], [r['outlier_pct'] for r in front],
                '--', color='0.5', lw=1, zorder=1, label='Pareto frontier')

        ax.set_xlabel('retained density [% of source points]')
        ax.set_ylabel(f'outlier rate among survivors [%]  (|v| >= {OUTLIER_M:g} m)')
        ax.set_title(f'T6 dense vs clean: filter Pareto, tile {tid}')
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f't6_pareto_tile_{tid}.png'), dpi=130)
        plt.close(fig)
        print(f'  -> {OUT_DIR}/t6_pareto_tile_{tid}.png  and  {csv_path}')


if __name__ == '__main__':
    main()
