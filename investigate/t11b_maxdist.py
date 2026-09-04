"""T11b - score the plotted (survivor) field of several runs from the output txt alone.

Why a second script. `t11_multiscale.py` needs the `--save_interim` feature dumps to
measure descriptor-side quantities (r10, top-10 spread). Those dumps are ~600 MB per
multi-scale run, and they are *identical* between a run with and without
`--max_search_distance`: the constraint acts after feature extraction, on the candidate
list, not on the descriptors. So for the spatial-constraint comparison the dumps carry
no information and are not written.

What is left is the field the operator actually plots, and it is fully described by
`output/results/displacement_magnitude_tile_*.txt` (source xyz | target xyz | |v|). This
script scores exactly that, so a `--save_interim`-less run can still be compared.

Metrics (all on the survivor field):

    survivors      number of points the supervoxel filter kept. Against a stated
                   `--n_source`, also the share of the source cloud.
    median |v|     median displacement magnitude.
    >= 40 m [%]    gross-outlier share under the FINDINGS.md proxy.
    coherence      residual of each vector to the median vector of its 16 nearest
                   survivors, reported as the median residual and the share below 2 m.
                   Label-free: it does not use the 40 m threshold at all.
    coherent pts   survivors x coherent share - the count of vectors that are both kept
                   and locally consistent. This is the number to compare across runs,
                   because density and purity move in opposite directions and a
                   percentage alone can be gamed by throwing points away.

Usage
-----
    python t11b_maxdist.py \
        --run ms_vauto=../results/TCH_gedi_ms2_6_12_vauto \
        --run ms_vauto_d40=../results/TCH_gedi_ms2_6_12_vauto_d40
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from common import OUTLIER_M, OUT_DIR, ensure_out

KNN_COH = 16     # neighbours used for the coherence residual
COH_OK = 2.0     # m, a vector counts as coherent below this residual


def coherence(xyz, vec):
    """Residual of every vector to the median vector of its KNN_COH neighbours."""
    if xyz.shape[0] <= KNN_COH:
        return np.full(xyz.shape[0], np.nan)
    _, idx = cKDTree(xyz).query(xyz, k=KNN_COH)
    return np.linalg.norm(vec - np.median(vec[idx], axis=1), axis=1)


def analyse(label, results_dir):
    rows = []
    pattern = os.path.join(results_dir, 'output', 'results',
                           'displacement_magnitude_tile_*.txt')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'no result txt under {pattern}')

    for path in paths:
        tid = os.path.basename(path).replace(
            'displacement_magnitude_tile_', '').replace('.txt', '')
        out = np.loadtxt(path)
        xyz, mag = out[:, 0:3], out[:, 6]
        res = coherence(xyz, out[:, 3:6] - out[:, 0:3])
        coh_ok = float(np.nanmean(res < COH_OK))

        rows.append(dict(
            run=label, tile=tid,
            survivors=int(out.shape[0]),
            mag_median_m=float(np.median(mag)),
            outlier_pct=float(100 * (mag >= OUTLIER_M).mean()),
            coh_residual_median_m=float(np.nanmedian(res)),
            coh_ok_pct=float(100 * coh_ok),
            coherent_points=int(round(out.shape[0] * coh_ok)),
        ))
        r = rows[-1]
        print(f'  {label} tile {tid}: survivors={r["survivors"]} '
              f'|v|={r["mag_median_m"]:.2f} m out={r["outlier_pct"]:.2f}% '
              f'coh_ok={r["coh_ok_pct"]:.1f}% coherent={r["coherent_points"]}')
    return rows


COLUMNS = [
    ('survivors', 'survivors', '{:,.0f}'),
    ('source_pct', 'of source [%]', '{:.1f}'),
    ('mag_median_m', 'median |v| [m]', '{:.2f}'),
    ('outlier_pct', '>=40 m [%]', '{:.2f}'),
    ('coh_residual_median_m', 'coh residual [m]', '{:.2f}'),
    ('coh_ok_pct', 'coherent <2 m [%]', '{:.1f}'),
    ('coherent_points', 'coherent points', '{:,.0f}'),
]


def print_table(rows):
    runs = []
    for r in rows:
        if r['run'] not in runs:
            runs.append(r['run'])
    width = max(len(h) for _, h, _ in COLUMNS) + 2
    colw = max(18, max(len(r) for r in runs) + 2)
    header = 'metric'.ljust(width) + ''.join(r.rjust(colw) for r in runs)
    print('\n' + header)
    print('-' * len(header))
    for key, head, fmt in COLUMNS:
        vals = [[r[key] for r in rows if r['run'] == run and key in r] for run in runs]
        if not any(vals):
            continue
        line = head.ljust(width)
        for v in vals:
            agg = np.sum(v) if key in ('survivors', 'coherent_points') else np.mean(v)
            line += (fmt.format(agg) if len(v) else '-').rjust(colw)
        print(line)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, metavar='LABEL=PATH')
    ap.add_argument('--n_source', type=int, default=0,
                    help='source points of the tiled cloud, to report survivors as a '
                         'share. Only meaningful when every run shares the same tiling.')
    args = ap.parse_args()

    ensure_out()
    rows = []
    for spec in args.run:
        label, _, path = spec.partition('=')
        if not path:
            raise SystemExit(f'--run needs LABEL=PATH, got {spec!r}')
        print(f'analysing {label} ({path})')
        rows += analyse(label, path)

    if args.n_source:
        for r in rows:
            r['source_pct'] = 100.0 * r['survivors'] / args.n_source

    print_table(rows)
    with open(os.path.join(OUT_DIR, 't11b_maxdist.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"json -> {os.path.join(OUT_DIR, 't11b_maxdist.json')}")


if __name__ == '__main__':
    main()
