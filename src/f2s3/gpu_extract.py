import logging
import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c
from torch.utils.dlpack import from_dlpack

_EPS = 1e-6


class GpuNeighborIndex:
    """Thin container that holds an Open3D NearestNeighborSearch together
    with the overlap positions (DLPack→torch) needed for coordinate
    scattering inside `radius_neighborhoods`.
    """

    __slots__ = ("nns", "positions")

    def __init__(self, nns, positions):
        self.nns = nns
        self.positions = positions  # [N, 3] float32 torch tensor on CUDA


def build_index(overlap_positions):
    """Build a GPU NearestNeighborSearch index over the overlap cloud.

    Args:
        overlap_positions: (N, 3) float32 numpy array of overlap-cloud points.

    Returns:
        ``GpuNeighborIndex`` whose ``.nns`` is the Open3D index and
        ``.positions`` is the DLPack→torch tensor on CUDA.
    """
    dev = o3c.Device("CUDA:0")
    pos = o3c.Tensor(
        overlap_positions.astype(np.float32).copy(), dtype=o3c.float32, device=dev
    )
    nns = o3d.core.nns.NearestNeighborSearch(pos)
    nns.knn_index()
    positions = from_dlpack(pos.to_dlpack())
    return GpuNeighborIndex(nns, positions)


def gpu_resolution(gpu_index, query_positions):
    """Estimate median k=2 nearest-neighbour distance on the GPU.

    Replaces ``_median_nn_resolution`` for the GPU path (EXT-04:
    same index that serves the radius search).

    Args:
        gpu_index: GpuNeighborIndex returned by `build_index`.
        query_positions: (Q, 3) float32 numpy array.

    Returns:
        float – median of the 2nd-column NN distance (sqrt of the
        squared distance returned by Open3D).
    """
    dev = o3c.Device("CUDA:0")
    q = o3c.Tensor(
        query_positions.astype(np.float32).copy(), dtype=o3c.float32, device=dev
    )
    dist, _ = gpu_index.nns.knn_search(q, 2)
    t_dist = from_dlpack(dist.to_dlpack())
    median_dist = t_dist[:, 1].sqrt().median().item()
    return median_dist


def radius_neighborhoods(gpu_index, query_positions, feature_radius):
    """GPU fixed-radius search → two padded neighbour sets + validity masks.

    Builds a ``fixed_radius_index`` on the same index object, then
    converts the CSR output into padded ``[Q, maxN, 3]`` tensors with
    boolean validity masks.  Returns **two** distinct sets mirroring
    the CPU ``data.py::extract_patch`` semantics:

    1. **ptnn** (nearest-dropped) — index‑0 excluded; feeds the
       covariance (eq. 3) and eq. 5 ``alpha`` distances.
    2. **ptall** (full in-radius) — index‑0 *kept*; feeds the LRF
       rotation and the 256‑point subsample.

    Args:
        gpu_index: GpuNeighborIndex returned by `build_index`.
        query_positions: (Q, 3) float32 numpy array.
        feature_radius: float, search radius.

    Returns:
        tuple of five torch tensors on CUDA:
        - ptnn_coords   [Q, maxN_nn, 3]  float32  (nearest dropped)
        - ptnn_mask     [Q, maxN_nn]      bool
        - ptnn_sqrt_dist [Q, maxN_nn]     float32  sqrt(squared dist)
        - ptall_coords  [Q, maxN_all, 3]  float32  (nearest kept)
        - ptall_mask    [Q, maxN_all]     bool
    """
    nns = gpu_index.nns
    dev = o3c.Device("CUDA:0")
    q = o3c.Tensor(
        query_positions.astype(np.float32).copy(), dtype=o3c.float32, device=dev
    )

    nns.fixed_radius_index(feature_radius)
    idx, dist, splits = nns.fixed_radius_search(q, feature_radius)

    t_idx = from_dlpack(idx.to_dlpack()).long()
    t_dist = from_dlpack(dist.to_dlpack())
    t_splits = from_dlpack(splits.to_dlpack()).long()
    overlap = gpu_index.positions  # [N, 3] float32, cuda

    counts = t_splits[1:] - t_splits[:-1]
    Q = int(counts.shape[0])

    # Center query points for relative coords
    q_t = from_dlpack(q.to_dlpack())  # [Q, 3] float32, cuda

    def _pad_results(keep_first):
        """Pad ragged CSR into [Q, maxN, 3] coords + [Q, maxN] mask + dist."""
        if keep_first:
            c = counts
            start_ofs = t_splits[:-1]
        else:
            c = (counts - 1).clamp(min=0)
            start_ofs = t_splits[:-1] + 1  # skip the nearest/self point

        max_nn = int(c.max())
        if max_nn == 0:
            return (_empty_padded(Q), _empty_mask(Q),
                    torch.zeros(Q, 0, dtype=torch.float32, device=overlap.device))

        mask = torch.arange(max_nn, device=c.device).unsqueeze(0) < c.unsqueeze(1)
        coords = torch.zeros(Q, max_nn, 3, dtype=torch.float32, device=overlap.device)
        sqrt_dist = torch.zeros(Q, max_nn, dtype=torch.float32, device=overlap.device)
        if not torch.any(c > 0):
            return coords, mask, sqrt_dist

        # Build flat source and destination indices for a single scatter
        # For each query q, neighbours at CSR[s[q] .. s[q]+n[q]-1] → row q, cols 0..n[q]-1
        nz_mask = c > 0
        nz_c = c[nz_mask]
        nz_start = start_ofs[nz_mask]  # flat CSR start per non-empty query
        nz_q = torch.arange(Q, device=c.device)[nz_mask]

        total = int(nz_c.sum().item())
        # Broadcast each query row to the number of neighbours it has
        q_rep = torch.repeat_interleave(nz_q, nz_c)
        # Column offset: 0, 1, 2, ..., n-1 for each query, concatenated
        col_rep = torch.cat([torch.arange(n.item(), device=c.device) for n in nz_c])
        # Source flat CSR index: s[q] + 0, s[q] + 1, ..., s[q] + n-1
        src_rep = torch.cat([
            torch.arange(s.item(), s.item() + n.item(), device=c.device)
            for s, n in zip(nz_start, nz_c)
        ])

        coords[q_rep, col_rep] = overlap[t_idx[src_rep]] - q_t[q_rep]
        sqrt_dist[q_rep, col_rep] = t_dist[src_rep].sqrt()
        return coords, mask, sqrt_dist

    ptnn_coords, ptnn_mask, ptnn_sqrt_dist = _pad_results(keep_first=False)
    ptall_coords, ptall_mask, _ = _pad_results(keep_first=True)

    return ptnn_coords, ptnn_mask, ptnn_sqrt_dist, ptall_coords, ptall_mask


def _empty_padded(Q):
    """Return an empty [Q, 0, 3] padded tensor."""
    return torch.zeros(Q, 0, 3, dtype=torch.float32, device="cuda:0")


def _empty_mask(Q):
    """Return an empty [Q, 0] bool mask."""
    return torch.zeros(Q, 0, dtype=torch.bool, device="cuda:0")


def batched_lrf(ptnn_coords, ptnn_mask, ptnn_sqrt_dist, ptall_coords,
                ptall_mask, q_pt, feature_radius):
    """Masked batched ``torch.linalg.eigh`` LRF (eqs. 3–5).

    Computes a 3×3 covariance over the **nearest-dropped** ``ptnn`` set
    (mirroring CPU ``points[patch_idx[1:]]`` / ``patch_dist[1:]``),
    holds the vectorized eq. 4 sign disambiguation and eq. 5 in-plane
    axis, then rotates the **full** ``ptall`` set (mirroring CPU
    ``points[patch_idx]``, index 0 kept) into the LRF.

    Args:
        ptnn_coords:   [Q, maxN_nn, 3]   neighbour coords (nearest dropped)
        ptnn_mask:     [Q, maxN_nn]      bool validity mask
        ptnn_sqrt_dist: [Q, maxN_nn]     sqrt squared distance
        ptall_coords:  [Q, maxN_all, 3]  full-in-radius coords (nearest kept)
        ptall_mask:    [Q, maxN_all]     bool validity mask
        q_pt:          [Q, 3]            query-point positions
        feature_radius: float

    Returns:
        patches: [Q, maxN_all, 3]   rotated by lRg, divided by feature_radius
    """
    zp, xp, yp = _lrf_core(ptnn_coords, ptnn_mask, ptnn_sqrt_dist, feature_radius)

    n_ptall = ptall_mask.sum(dim=1)

    # Build lRg per query: [Q, 3, 3] as [xp, yp, zp]^T
    lRg = torch.stack([xp, yp, zp], dim=2)

    # Rotate the FULL ptall set into the LRF (coords already centered)
    ptall_rot = torch.bmm(lRg.transpose(1, 2), ptall_coords.transpose(1, 2))
    ptall_rot = ptall_rot.transpose(1, 2) / feature_radius

    # >10-neighbor branch: <=10 → skip rotation (CPU else-branch)
    few_mask = n_ptall <= 10
    no_lrf = ptall_coords / feature_radius
    result = torch.where(few_mask[:, None, None], no_lrf, ptall_rot)
    result = result * ptall_mask[..., None]

    return result, zp, xp, yp


def subsample_256(patches, seed):
    """Seeded vectorised 256-point subsample without replacement.

    Reproduces ``np.random.choice(replace=False)`` over a zero-padded
    array (zero rows are eligible, D-05).  Deterministic by default
    (D-04) — a local ``torch.Generator`` is seeded from ``seed``.

    Args:
        patches: [Q, P, 3] float32 tensor (already rotated + normalised,
                 may be zero-padded to ≥ 256 rows).
        seed: int for the local Generator.

    Returns:
        selected: [Q, 256, 3] float32 tensor (deterministic for same seed).
    """
    P = patches.shape[1]
    device = patches.device

    # Pad to at least 256 if narrower (D-05 sparse regime)
    if P < 256:
        z = torch.zeros(patches.shape[0], 256 - P, 3, dtype=patches.dtype,
                        device=device)
        patches = torch.cat([patches, z], dim=1)
        P = 256

    g = torch.Generator(device=device).manual_seed(seed)
    keys = torch.rand(patches.shape[0], P, generator=g, device=device)
    perm = keys.argsort(dim=1)[:, :256]
    selected = torch.gather(patches, 1,
                            perm[..., None].expand(-1, -1, 3))
    return selected


def gpu_extract_features(
    query_pts_t, gpu_index, feature_radius, feature_extractor,
    batch_size, seed, verbose=False
):
    """End-to-end GPU extraction returning (Q, feat_dim) local features.

    Runs the full parity-grade pipeline: radius search → batched masked
    ``eigh`` LRF (eqs. 3–5) → seeded vectorised subsample (256 points) →
    descriptor model forward.  Chunks query points to stay within 24 GB
    VRAM.

    Args:
        query_pts_t: (Q, 3) float32 torch tensor on CUDA.
        gpu_index: GpuNeighborIndex returned by `build_index`.
        feature_radius: float, search radius.
        feature_extractor: PointNetFeature model in eval mode (on CUDA).
        batch_size: int, batch size fed to the descriptor model.
        seed: int, seed for subsample determinism.
        verbose: bool.

    Returns:
        local_features: (Q, feat_dim) float32 torch tensor on CUDA.
    """
    Q = query_pts_t.shape[0]
    if Q == 0:
        return torch.zeros(0, 64, device=query_pts_t.device)

    # VRAM-bound chunk size: radius_neighborhoods produces [chunk, maxN, 3]
    # float32 tensors; with maxN up to ~2000 on the _small pair at
    # r≈2.96, a chunk of 20000 points needs ~20000*2000*3*4 = 480 MB.
    # Two such sets (ptnn + ptall) + NNS scratch ~= 2 GB/chunk.
    # Twenty-four GB / 2 GB → safe at 10000 chunks (verified on 3090 Ti).
    CHUNK_SIZE = 10000

    local_features = []
    for chunk_start in range(0, Q, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, Q)
        chunk_pts_t = query_pts_t[chunk_start:chunk_end]
        chunk_pts_np = chunk_pts_t.cpu().numpy().astype(np.float32)

        ptnn_coords, ptnn_mask, ptnn_sqrt_dist, ptall_coords, ptall_mask = (
            radius_neighborhoods(gpu_index, chunk_pts_np, feature_radius)
        )

        patches, _, _, _ = batched_lrf(
            ptnn_coords, ptnn_mask, ptnn_sqrt_dist, ptall_coords, ptall_mask,
            chunk_pts_t, feature_radius
        )

        patches_256 = subsample_256(patches, seed)

        # Feed through descriptor model in batches
        chunk_Q = chunk_end - chunk_start
        for start in range(0, chunk_Q, batch_size):
            end = min(start + batch_size, chunk_Q)
            batch_patches = patches_256[start:end, :, :]
            batch_feat, _, _ = feature_extractor(batch_patches)
            local_features.append(batch_feat)

        if verbose:
            logging.info(
                f"  GPU extraction chunk {chunk_start//CHUNK_SIZE + 1}: "
                f"{(chunk_end - chunk_start)//1000}k points processed"
            )

    if local_features:
        local_features = torch.cat(local_features, 0)
    else:
        local_features = torch.zeros(0, 64, device=query_pts_t.device)

    return local_features


# ---------------------------------------------------------------------------
# Test-facing thin wrappers (plan 03-01 contract names)
# ---------------------------------------------------------------------------

def compute_lrf(ptnn_coords, ptnn_mask, ptnn_sqrt_dist, feature_radius):
    """Computes only the LRF vectors (zp, xp, yp) from the nearest-dropped set.

    Thin alias around ``_lrf_core`` that the test contract expects.
    Used by ``test_lrf_parity`` (EXT-02).
    """
    return _lrf_core(ptnn_coords, ptnn_mask, ptnn_sqrt_dist, feature_radius)


def subsample_seeded(patches, mask, seed, target_size=256):
    """Seeded vectorised subsample matching the test contract.

    Thin alias around ``subsample_256`` that the test contract expects.
    Used by ``test_subsample_seeded`` (EXT-03).
    """
    return subsample_256(patches, seed)


# Extract the LRF-only core so compute_lrf can call it without the full
# rotation path needed for gpu_extract_features.
def _lrf_core(ptnn_coords, ptnn_mask, ptnn_sqrt_dist, feature_radius):
    """Core LRF computation — returns (zp, xp, yp) only.

    Reproduces the CPU ``>10`` neighbor branch (data.py:43): when a query
    has ≤ 10 real neighbours the LRF defaults to identity axes.
    """
    Q = ptnn_coords.shape[0]
    device = ptnn_coords.device
    n_ptnn = ptnn_mask.sum(dim=1).clamp(min=1)

    # Default (identity) LRF for queries with ≤10 neighbours
    zp0 = torch.tensor([0.0, 0.0, 1.0], device=device).expand(Q, 3)
    xp0 = torch.tensor([1.0, 0.0, 0.0], device=device).expand(Q, 3)
    yp0 = torch.tensor([0.0, 1.0, 0.0], device=device).expand(Q, 3)

    has_neighbors = n_ptnn > 10
    if not has_neighbors.any():
        return zp0, xp0, yp0

    # Only compute on queries with enough neighbours
    q_keep = has_neighbors
    idx_keep = q_keep.nonzero(as_tuple=True)[0]

    sub_coords = ptnn_coords[idx_keep]
    sub_mask = ptnn_mask[idx_keep]
    sub_dist = ptnn_sqrt_dist[idx_keep]
    sub_n = n_ptnn[idx_keep]

    # eq.3 masked covariance
    vect_diff = sub_coords * sub_mask[..., None]
    cov = (torch.einsum('qni,qnj->qij', vect_diff, vect_diff)
           / sub_n[..., None, None])
    evals, evecs = torch.linalg.eigh(cov)
    np_hat = evecs[:, :, 0]
    flip = (torch.einsum('qi,qni->q', np_hat, -vect_diff)) > 0
    sub_zp = torch.where(flip[:, None], np_hat, -np_hat)

    # eq.5 in-plane axis xp
    v_proj = vect_diff - (torch.einsum('qni,qj->qni', vect_diff, sub_zp)
                          * sub_zp[:, None, :])
    alpha = (feature_radius - sub_dist) ** 2
    beta = (torch.einsum('qni,qi->qn', vect_diff, sub_zp)).square()
    weighted = torch.einsum('qni,qn->qi', v_proj, alpha * beta)
    w_norm = weighted.norm(dim=1)
    safe_norm = torch.where(w_norm < _EPS, w_norm + _EPS, w_norm)
    sub_xp = weighted / safe_norm[:, None]
    sub_xp = sub_xp[:, :3]
    sub_zp = sub_zp[:, :3]
    sub_yp = torch.linalg.cross(sub_xp, sub_zp)

    # Scatter back into Q-sized tensors
    zp = zp0.clone()
    xp = xp0.clone()
    yp = yp0.clone()
    zp[idx_keep] = sub_zp
    xp[idx_keep] = sub_xp
    yp[idx_keep] = sub_yp

    return zp, xp, yp
