"""Shared loading / kNN helpers for the feature-space ambiguity investigation.

All tasks operate on the interim data written by `f2s3 --save_interim`:
    features/features_tile_{T}_{R}.npz        feat_s, feat_t, xyz_s, xyz_t
    correspondences/correspondences_tile_{T}.npz   corr [Ns,6]
    output/results/displacement_magnitude_tile_{T}.txt   [Ns,7]

Row i of feat_s / corr / the output txt all refer to the same source point.
"""

import glob
import os
import re

import numpy as np
import torch

RESULTS_DIR = os.environ.get('F2S3_RESULTS', 'results/TCH_gedi_r6_investigate')
OUT_DIR = os.environ.get('F2S3_INVEST_OUT', 'investigate/out')

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
        self.mag = self.out[:, 6].astype(np.float32)

        n = self.feat_s.shape[0]
        assert self.corr.shape[0] == n, 'corr rows != feat_s rows'
        assert self.out.shape[0] == n, 'output rows != feat_s rows'
        assert np.allclose(self.corr[:, :3], self.out[:, :3], atol=1e-3), \
            'source coords in corr and output txt disagree'
        if self.xyz_t is not None:
            assert self.xyz_t.shape[0] == self.feat_t.shape[0], 'xyz_t rows != feat_t rows'

    @property
    def n_src(self):
        return self.feat_s.shape[0]

    @property
    def n_tgt(self):
        return self.feat_t.shape[0]

    @property
    def is_outlier(self):
        return self.mag >= OUTLIER_M

    @property
    def is_inlier(self):
        return self.mag < INLIER_M

    def __repr__(self):
        return (f'Tile({self.tile_id}: Ns={self.n_src} Nt={self.n_tgt} '
                f'D={self.feat_s.shape[1]} xyz_t={"yes" if self.xyz_t is not None else "no"})')


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
