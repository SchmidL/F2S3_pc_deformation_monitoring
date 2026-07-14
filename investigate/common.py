"""Shared loading / kNN helpers for the feature-space ambiguity investigation.

All tasks operate on the interim data written by `f2s3 --save_interim`:
    features/features_tile_{T}_{R}.npz        feat_s, feat_t, xyz_s, xyz_t
    correspondences/correspondences_tile_{T}.npz   corr [Ns,6]
    output/results/displacement_magnitude_tile_{T}.txt   [Nk,7]

IMPORTANT - row alignment. feat_s, xyz_s and corr are row-aligned per source
point. The output txt is NOT: `filter_correspondences` writes only the points
that survived the supervoxel filter, reordered by supervoxel
(core.py: save_coords[inlier_idx]). For the sample tile that is 189k of 861k
source points. So we do not read magnitudes from the txt at all - we recompute
the raw 1-NN magnitude straight from `corr`, which is row-aligned by
construction, and separately remap the txt back onto source rows (exact 6-D
coordinate match) to recover which points survived the filter.

That gives two distinct fields, and the distinction matters:

    mag        raw 1-NN displacement, every source point. What the *matcher*
               produced, before any filtering. This is what descriptor
               ambiguity acts on directly.
    survived   the subset the supervoxel filter kept - the field actually
               plotted, ~7.6 m median with ~4.6 % residual gross outliers.
"""

import glob
import os
import re

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.environ.get(
    'F2S3_RESULTS', os.path.join(_HERE, os.pardir, 'results', 'TCH_gedi_r6_investigate'))
# anchored to this file, so the outputs land in investigate/out no matter where
# the scripts are invoked from
OUT_DIR = os.environ.get('F2S3_INVEST_OUT', os.path.join(_HERE, 'out'))

# A displacement of at least this magnitude is treated as a gross outlier.
OUTLIER_M = 40.0
# Vectors below this are treated as the trusted "inlier" reference group.
INLIER_M = 15.0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR


def tile_ids(results_dir=RESULTS_DIR):
    """Tile ids present in the features dir, in sorted order."""
    ids = []
    for p in glob.glob(os.path.join(results_dir, 'features', 'features_tile_*.npz')):
        m = re.match(r'features_tile_(.+)_[\d.]+\.npz$', os.path.basename(p))
        if m:
            ids.append(m.group(1))
    return sorted(ids)


class Tile:
    """One tile's interim data, with the row-alignment contract asserted."""

    def __init__(self, tile_id, results_dir=RESULTS_DIR):
        self.tile_id = tile_id
        self.results_dir = results_dir

        feat_paths = glob.glob(os.path.join(
            results_dir, 'features', f'features_tile_{tile_id}_*.npz'))
        if not feat_paths:
            raise FileNotFoundError(f'no features archive for tile {tile_id} in {results_dir}')
        F = np.load(feat_paths[0])

        self.feat_s = F['feat_s'].astype(np.float32)
        self.feat_t = F['feat_t'].astype(np.float32)
        self.xyz_t = F['xyz_t'].astype(np.float32) if 'xyz_t' in F else None

        self.corr = np.load(os.path.join(
            results_dir, 'correspondences',
            f'correspondences_tile_{tile_id}.npz'))['corr']
        self.out = np.loadtxt(os.path.join(
            results_dir, 'output', 'results',
            f'displacement_magnitude_tile_{tile_id}.txt'))

        self.xyz_s = self.corr[:, :3].astype(np.float32)

        n = self.feat_s.shape[0]
        assert self.corr.shape[0] == n, 'corr rows != feat_s rows'
        if self.xyz_t is not None:
            assert self.xyz_t.shape[0] == self.feat_t.shape[0], 'xyz_t rows != feat_t rows'
        # the 1-NN target coords in corr must be rows of xyz_t
        assert np.allclose(self.corr[:, :3], self.xyz_s, atol=1e-5)

        # raw 1-NN displacement, row-aligned with feat_s (float64: metre-scale values,
        # and the remap assert below compares against the float64 txt)
        self.mag = np.linalg.norm(self.corr[:, 3:6] - self.corr[:, 0:3], axis=1)

        self.survived = self._remap_survivors()

    def _remap_survivors(self):
        """Boolean mask over source rows: did this point survive the supervoxel filter?

        The txt rows are exact copies of `corr` rows, so a 6-D exact match recovers
        the source index. (Matching on source xyz alone is not safe: 66 source points
        in the sample tile share coordinates with another point.)
        """
        def key(a):
            a = np.ascontiguousarray(a, dtype=np.float64)
            return a.view([('', a.dtype)] * a.shape[1]).ravel()

        kc, ko = key(self.corr[:, :6]), key(self.out[:, :6])
        order = np.argsort(kc)
        pos = np.searchsorted(kc[order], ko)
        hit = order[np.clip(pos, 0, len(kc) - 1)]
        ok = kc[hit] == ko
        assert ok.all(), 'some output rows did not match a correspondence row'
        # magnitudes must agree exactly, which proves the remap is right
        assert np.abs(self.mag[hit] - self.out[:, 6]).max() < 1e-6, 'remap magnitude mismatch'
        m = np.zeros(self.n_src, dtype=bool)
        m[hit] = True
        return m

    @property
    def n_src(self):
        return self.feat_s.shape[0]

    @property
    def n_tgt(self):
        return self.feat_t.shape[0]

    @property
    def is_outlier(self):
        """Gross outlier in the RAW 1-NN field (before the supervoxel filter)."""
        return self.mag >= OUTLIER_M

    @property
    def is_inlier(self):
        return self.mag < INLIER_M

    @property
    def trusted(self):
        """Survived the supervoxel filter AND has a plausible magnitude."""
        return self.survived & (self.mag < INLIER_M)

    @property
    def residual_outlier(self):
        """Survived the supervoxel filter and is STILL a gross outlier.

        These are the clustered blobs that remain in the plotted field.
        """
        return self.survived & (self.mag >= OUTLIER_M)

    def __repr__(self):
        return (f'Tile({self.tile_id}: Ns={self.n_src} Nt={self.n_tgt} '
                f'D={self.feat_s.shape[1]} xyz_t={"yes" if self.xyz_t is not None else "no"} '
                f'survived={self.survived.sum()} ({self.survived.mean():.1%}))')


def knn(query, db, k, chunk=8192, device=DEVICE):
    """Exact kNN by chunked cdist. Returns (distances, indices), both [Nq, k].

    The full Nq x Ndb distance matrix is never materialised.
    """
    q = torch.as_tensor(query, dtype=torch.float32, device=device)
    d = torch.as_tensor(db, dtype=torch.float32, device=device)
    D = torch.empty((q.shape[0], k), dtype=torch.float32)
    I = torch.empty((q.shape[0], k), dtype=torch.long)
    for s in range(0, q.shape[0], chunk):
        dist = torch.cdist(q[s:s + chunk], d)
        dk, ik = dist.topk(k, largest=False, dim=1)
        D[s:s + chunk] = dk.cpu()
        I[s:s + chunk] = ik.cpu()
    return D.numpy(), I.numpy()


def cross_knn_cached(tile, k=20):
    """kNN from feat_s into feat_t, cached to disk (T2 computes it, T3-T6 reuse)."""
    ensure_out()
    path = os.path.join(OUT_DIR, f'crossknn_tile_{tile.tile_id}_k{k}.npz')
    if os.path.exists(path):
        z = np.load(path)
        return z['Dk'], z['Ik']
    Dk, Ik = knn(tile.feat_s, tile.feat_t, k=k)
    np.savez_compressed(path, Dk=Dk, Ik=Ik)
    return Dk, Ik


def auc(score, label):
    """AUC of `score` as a predictor of the boolean `label`, via rank statistics.

    0.5 = no separation, 1.0 = perfect (high score => label true).
    """
    label = np.asarray(label, dtype=bool)
    n_pos, n_neg = label.sum(), (~label).sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties so tied scores cannot fake separation
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return (ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
