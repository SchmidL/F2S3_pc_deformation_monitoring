"""Regenerate publication-quality figures for the paper draft (paper/main.tex).

Reads only cached artefacts in investigate/out/ (t1..t4 npz, cross-kNN cache,
t5/t6/t7 csv) plus the interim tile data; no GPU recomputation. Writes PDF
figures to investigate/paper/figures/.

Differences to the exploratory task figures in investigate/out/:
  - degenerate panels dropped (T1 n_eps bar/map: identically zero)
  - no task-number titles; consistent serif style, sized for \textwidth
  - annotations carry the headline numbers so figures stand alone
"""

import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir))

from common import Tile, tile_ids, OUT_DIR, OUTLIER_M, INLIER_M  # noqa: E402

FIG_DIR = os.path.join(_HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
})

C_TRUST = '#1b7837'    # green
C_OUT = '#c51b2f'      # red
C_REJ = '#5b6a79'      # slate grey
C_RAW = '#9e9e9e'
C_SURV = '#2c7fb8'

tid = tile_ids()[0]
tile = Tile(tid)
print(tile)

t1 = np.load(os.path.join(OUT_DIR, f't1_tile_{tid}.npz'))
t2 = np.load(os.path.join(OUT_DIR, f't2_tile_{tid}.npz'))
t3 = np.load(os.path.join(OUT_DIR, f't3_tile_{tid}.npz'))
t4 = np.load(os.path.join(OUT_DIR, f't4_tile_{tid}.npz'))
ck = np.load(os.path.join(OUT_DIR, f'crossknn_tile_{tid}_k20.npz'))
Dk, Ik = ck['Dk'], ck['Ik']

trusted = tile.trusted
resid = tile.residual_outlier
rejected = ~tile.survived


def save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p)
    plt.close(fig)
    print('wrote', p)


# ---------------------------------------------------------------- fig 1: |v| distribution
fig, ax = plt.subplots(figsize=(6.3, 2.6), layout='constrained')
bins = np.logspace(np.log10(0.05), np.log10(tile.mag.max() * 1.05), 120)
ax.hist(np.clip(tile.mag, bins[0], None), bins=bins, color=C_RAW,
        label=f'raw 1-NN field (all {tile.n_src:,} points)')
ax.hist(np.clip(tile.mag[tile.survived], bins[0], None), bins=bins, color=C_SURV,
        alpha=0.9, label=f'supervoxel survivors ({tile.survived.sum():,} points)')
ax.axvspan(25, 60, color='0.85', zorder=0)
for th in (20, 25, 30, 40, 60):
    ax.axvline(th, color=C_OUT, ls=':', lw=0.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(top=4e5)
ax.set_xlabel(r'displacement magnitude $|v|$ [m]')
ax.set_ylabel('count')
ax.text(38, 1.1e5, 'tested label\nthresholds $\\theta$', color=C_OUT, ha='center',
        va='top', fontsize=8)
ax.legend(loc='upper left')
save(fig, 'fig_magnitude.pdf')

# ---------------------------------------------------------------- fig 2: T1 self-similarity
d_self1, eps1 = t1['d_self1'], float(t1['eps'])
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.hist(d_self1, bins=120, color=C_SURV)
ax.axvline(eps1, color=C_OUT, ls='--', lw=1.2,
           label=fr'$\varepsilon_1 = 0.1\,\mathrm{{med}}(d^{{self}}_1) = {eps1:.3f}$')
ax.axvline(np.median(d_self1), color='k', ls=':', lw=1.0,
           label=fr'median $d^{{self}}_1 = {np.median(d_self1):.3f}$')
ax.set_xlabel('distance to nearest other source descriptor $d^{self}_1$')
ax.set_ylabel('count')
ax.text(0.97, 0.55, f'$n_\\varepsilon = 0$ for all\n{tile.n_src:,} descriptors:\nno near-duplicates',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round', fc='white', ec='0.6'))
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
save(fig, 'fig_t1_selfsim.pdf')

# ---------------------------------------------------------------- fig 3: T2 match quality
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), layout='constrained')
ax = axes[0]
h = ax.hist2d(t2['d1'], t2['r2'], bins=120, norm=matplotlib.colors.LogNorm(),
              cmap='viridis', rasterized=True)
fig.colorbar(h[3], ax=ax, label='count')
ax.set_xlabel('$d_1$ (distance to best target descriptor)')
ax.set_ylabel('$r_2 = d_1/d_2$  (Lowe ratio)')
ax.axhline(0.8, color=C_OUT, ls='--', lw=1.0)
ax.text(0.645, 0.62, "Lowe's criterion\n$r_2 < 0.8$", color=C_OUT, fontsize=7.5,
        ha='right', bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.8))
ax.set_title('(a) most mass piles up at $r_2 \\to 1$', fontsize=9)

ax = axes[1]
ranks = np.arange(1, 21)
for mask, color, label in ((trusted, C_TRUST, f'trusted (survived, $|v|<{INLIER_M:.0f}$ m)'),
                           (resid, C_OUT, f'residual outlier (survived, $|v|\\geq{OUTLIER_M:.0f}$ m)'),
                           (rejected, C_REJ, 'rejected by supervoxel filter')):
    sub = Dk[mask]
    mean = sub.mean(axis=0)
    q1, q3 = np.percentile(sub, [25, 75], axis=0)
    ax.plot(ranks, mean, color=color, marker='.', ms=4, lw=1.2, label=label)
    ax.fill_between(ranks, q1, q3, color=color, alpha=0.13, lw=0)
ax.set_xlabel('feature-space neighbour rank $j$')
ax.set_ylabel('mean distance $d_j$ (IQR shaded)')
ax.set_xticks([1, 5, 10, 15, 20])
ax.legend(loc='lower right', fontsize=7)
ax.set_title('(b) outlier profiles are flatter', fontsize=9)
save(fig, 'fig_t2_profile.pdf')

# ---------------------------------------------------------------- fig 4: T3 ambiguity depth
d01 = t3['depth_0.1']
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), layout='constrained')
ax = axes[0]
b = np.arange(0.5, 21.5)
for key, color in (('depth_0.05', '#4575b4'), ('depth_0.1', '#e08214'), ('depth_0.2', '#31a354')):
    ax.hist(t3[key], bins=b, histtype='step', lw=1.4, color=color,
            label=fr'$\varepsilon={key.split("_")[1]}$')
ax.set_yscale('log')
ax.axvline(20, color='k', ls=':', lw=1.0)
ax.text(18.7, 1.5e2, 'right-censored\nat $k=20$', ha='right', fontsize=7.5,
        bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.8))
ax.set_xlabel(r'ambiguity depth$_\varepsilon$ (near-tied candidates)')
ax.set_ylabel('count')
ax.legend(loc='lower left')
ax.set_title('(a) all source points', fontsize=9)

ax = axes[1]
for mask, color, label in ((trusted, C_TRUST, 'trusted'),
                           (resid, C_OUT, 'residual outlier')):
    ax.hist(d01[mask], bins=b, density=True, histtype='step', lw=1.6, color=color, label=label)
ax.axvline(np.median(d01), color='k', ls=':', lw=1.0,
           label=f'overall median = {int(np.median(d01))}')
frac = (d01 > 2).mean() * 100
ax.text(0.97, 0.45, f'{frac:.1f}% of all points\nhave depth > 2',
        transform=ax.transAxes, ha='right', fontsize=8,
        bbox=dict(boxstyle='round', fc='white', ec='0.6'))
ax.set_xlabel(r'depth$_{0.1}$')
ax.set_ylabel('density')
ax.legend(loc='upper right', fontsize=7.5)
ax.set_title('(b) survivors, by group', fontsize=9)
save(fig, 'fig_t3_depth.pdf')

# ---------------------------------------------------------------- fig 5: T4 spatial spread
spread = t4['spread']
diag = np.linalg.norm(tile.xyz_t.max(axis=0) - tile.xyz_t.min(axis=0))
rng = np.random.default_rng(0)
bg = tile.xyz_t[rng.choice(tile.n_tgt, 30000, replace=False)]

i_lo = int(np.argmin(np.where(spread > 0, spread, np.inf)))
i_hi = int(np.argmax(spread))

fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4), layout='constrained',
                         width_ratios=[1.5, 1, 1])
ax = axes[0]
ax.hist(spread, bins=100, color=C_SURV)
ax.set_yscale('log')
ax.set_ylim(top=3e6)
ax.axvline(diag, color=C_OUT, ls='--', lw=1.0, label=f'tile diagonal {diag:.0f} m')
ax.axvline(10, color=C_TRUST, ls='--', lw=1.0, label='filter threshold 10 m')
ax.set_xlabel('spread of top-10 candidates [m]')
ax.set_ylabel('count')
ax.legend(fontsize=7, loc='upper left')
ax.set_title(f'(a) median {np.median(spread):.0f} m', fontsize=9)

for ax, idx, color, ttl in ((axes[1], i_lo, C_TRUST, '(b)'),
                            (axes[2], i_hi, C_OUT, '(c)')):
    ax.scatter(bg[:, 0], bg[:, 1], s=0.3, color='0.85', rasterized=True)
    cand = tile.xyz_t[Ik[idx, :10]]
    src = tile.xyz_s[idx]
    for c in cand:
        ax.plot([src[0], c[0]], [src[1], c[1]], color=color, lw=0.7, alpha=0.8)
    ax.scatter(cand[:, 0], cand[:, 1], s=10, color=color, zorder=3)
    ax.scatter([src[0]], [src[1]], marker='*', s=60, color='k', zorder=4)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_title(f'{ttl}: spread {spread[idx]:.1f} m', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
save(fig, 'fig_t4_spread.pdf')

# ---------------------------------------------------------------- fig 6: T5 AUC ranking
rows = list(csv.DictReader(open(os.path.join(OUT_DIR, f't5_auc_tile_{tid}.csv'))))
res = {r['metric']: float(r['auc']) for r in rows if r['target'].startswith('residual')}
raw = {r['metric']: float(r['auc']) for r in rows if r['target'].startswith('raw')}
drop = {'flatness'}  # monotone transform of r10 -> identical AUC
names = [m for m in sorted(res, key=res.get) if m not in drop]
lab = {'r10 (d1/d10)': '$r_{10} = d_1/d_{10}$',
       'd1 (raw match dist)': '$d_1$ (match distance)',
       'depth_0.2': r'depth$_{0.2}$', 'depth_0.1': r'depth$_{0.1}$',
       'depth_0.05': r'depth$_{0.05}$',
       'r2 (Lowe, d1/d2)': '$r_2$ (Lowe ratio)',
       'spread (top-10, m)': 'spread (top-10)',
       'n_eps (source collapse)': r'$n_\varepsilon$ (T1 collapse)'}

fig, ax = plt.subplots(figsize=(4.8, 2.7), layout='constrained')
y = np.arange(len(names))
ax.barh(y + 0.2, [res[m] for m in names], height=0.38, color=C_SURV,
        label='residual target (survivors)')
ax.barh(y - 0.2, [raw[m] for m in names], height=0.38, color=C_RAW,
        label='raw target (all points)')
ax.set_yticks(y, [lab.get(m, m) for m in names])
ax.axvline(0.5, color='k', ls=':', lw=1.0)
ax.set_xlim(0.45, 0.85)
ax.set_xlabel('AUC (0.5 = no separation)')
ax.legend(loc='lower right', fontsize=7.5)
ax.grid(axis='y', alpha=0)
save(fig, 'fig_t5_auc.pdf')

# ---------------------------------------------------------------- fig 7: T6 Pareto
rows = list(csv.DictReader(open(os.path.join(OUT_DIR, f't6_pareto_tile_{tid}.csv'))))
fam_color = {'baseline': 'k', 'ratio': '#4575b4', 'mutual': '#e08214',
             'depth': '#c51b2f', 'spread': '#1b7837', 'combo': '#8073ac'}
fig, ax = plt.subplots(figsize=(6.3, 3.6), layout='constrained')
pts = []
for r in rows:
    x, yv = float(r['retained_pct']), float(r['outlier_pct'])
    pts.append((x, yv, r['filter'], r['family']))
    if r['family'] == 'baseline':
        ax.scatter(x, yv, marker='*', s=140, color='k', zorder=5)
    else:
        ax.scatter(x, yv, s=36, color=fam_color[r['family']], zorder=4)
# pareto frontier: sort by density desc, keep running-min outlier
frontier = []
best = np.inf
for x, yv, name, fam in sorted(pts, key=lambda p: -p[0]):
    if yv < best:
        frontier.append((x, yv))
        best = yv
fx, fy = zip(*frontier)
ax.plot(fx, fy, ls='--', color='0.5', lw=1.0, zorder=1, label='Pareto frontier')
ann = {'none (current 1-NN)': (-0.8, -0.25, 'right'),
       'depth_0.1 <= 3': (0.3, 0.1, 'left'),
       'depth_0.1 <= 2': (0.3, 0.08, 'left'),
       'ratio < 0.95': (0.3, 0.05, 'left'),
       'mutual NN': (0.3, 0.05, 'left'),
       'ratio < 0.80': (0.3, 0.05, 'left'),
       'spread < 10 m': (0.35, 0.12, 'left'),
       'mutual & ratio<0.90 & spread<10 m': (0.3, 0.14, 'left')}
for x, yv, name, fam in pts:
    if name in ann:
        dx, dy, ha = ann[name]
        txt = 'baseline (no feature filter)' if name.startswith('none') else name
        ax.annotate(txt, (x, yv), (x + dx, yv + dy), fontsize=7.5, ha=ha)
handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=f)
           for f, c in fam_color.items() if f != 'baseline']
handles.insert(0, plt.Line2D([], [], marker='*', ms=12, ls='', color='k', label='baseline'))
handles.append(plt.Line2D([], [], ls='--', color='0.5', label='Pareto frontier'))
ax.legend(handles=handles, loc='upper left', fontsize=7.5)
ax.set_xlabel(r'retained density [% of all source points $N_s$]')
ax.set_ylabel(r'outliers among kept survivors [%]  ($|v|\geq 40$ m)')
ax.set_xlim(-0.5, 24)
ax.set_ylim(-0.15, 5.2)
save(fig, 'fig_t6_pareto.pdf')

# ---------------------------------------------------------------- fig 8: T7 sensitivity
rows = list(csv.DictReader(open(os.path.join(OUT_DIR, f't7_sensitivity_tile_{tid}.csv'))))
th = [float(r['threshold_m']) for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), layout='constrained')
ax = axes[0]
for key, label, color in (('auc_r10', '$r_{10}$', '#e08214'),
                          ('auc_d1', '$d_1$', '#8073ac'),
                          ('auc_depth_0.1', r'depth$_{0.1}$', '#31a354'),
                          ('auc_r2', '$r_2$ (Lowe)', '#4575b4'),
                          ('auc_spread', 'spread', '#c51b2f')):
    ax.plot(th, [float(r[key]) for r in rows], marker='o', ms=3, color=color, label=label)
ax.axhline(0.5, color='k', ls=':', lw=0.8)
ax.set_ylim(0.44, 0.85)
ax.set_xlabel(r'label threshold $\theta$ [m]')
ax.set_ylabel('AUC (residual target)')
ax.legend(fontsize=7, ncol=3, loc='lower right')
ax.set_title('(a) metric ranking is $\\theta$-invariant', fontsize=9)

ax = axes[1]
for key, label, color in (('residual_outlier_pct', 'no filter (baseline)', 'k'),
                          ('filter_ratio<0.95_outlier_pct', 'ratio $<0.95$', '#4575b4'),
                          ('filter_depth<=3_outlier_pct', r'depth$_{0.1}\leq 3$', '#31a354'),
                          ('filter_mutual NN_outlier_pct', 'mutual NN', '#e08214'),
                          ('filter_spread<10m_outlier_pct', 'spread $<10$ m', '#c51b2f')):
    ax.plot(th, [float(r[key]) for r in rows], marker='o', ms=3, color=color, label=label)
ax.set_xlabel(r'label threshold $\theta$ [m]')
ax.set_ylabel('outliers among kept [%]')
ax.legend(fontsize=7)
ax.set_title('(b) filter ordering is $\\theta$-invariant', fontsize=9)
save(fig, 'fig_t7_sensitivity.pdf')

print('done')
