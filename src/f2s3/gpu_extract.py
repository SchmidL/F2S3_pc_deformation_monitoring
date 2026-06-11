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

        coords[q_rep, col_rep] = overlap[t_idx[src_rep]]
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


def gpu_extract_features(
    query_pts_t, gpu_index, feature_radius, feature_extractor,
    batch_size, seed, verbose=False
):
    """Thin end-to-end GPU extraction returning (B, 3, 256) patch tensor.

    This is a placeholder LRF/subsample path for plan 03-02: it runs
    the radius search on-device but uses an identity LRF + uniform
    subsample so the slice runs end-to-end.  Parity-grade LRF and
    seeded subsample are 03-03's job.

    Args:
        query_pts_t: (Q, 3) float32 torch tensor on CUDA.
        gpu_index: GpuNeighborIndex returned by `build_index`.
        feature_radius: float, search radius.
        feature_extractor: PointNetFeature model in eval mode (on CUDA).
        batch_size: int, batch size fed to the descriptor model.
        seed: int, seed for subsample determinism (respected in 03-03).
        verbose: bool.

    Returns:
        local_features: (Q, feat_dim) float32 torch tensor on CUDA.
    """
    Q = query_pts_t.shape[0]
    if Q == 0:
        return torch.zeros(0, feature_extractor.final_mlp[-1].out_features
                           if hasattr(feature_extractor, 'final_mlp')
                           else 64, device=query_pts_t.device)

    query_np = query_pts_t.cpu().numpy().astype(np.float32)
    _, _, _, ptall_coords, ptall_mask = radius_neighborhoods(
        gpu_index, query_np, feature_radius
    )
    # ptall_coords: [Q, maxN_all, 3], ptall_mask: [Q, maxN_all]
    max_nn = ptall_coords.shape[1]

    # Thin subsample: take first 256 real neighbours + pad with zeros
    n_real = ptall_mask.sum(dim=1).long()
    patches = torch.zeros(Q, 256, 3, dtype=torch.float32, device=ptall_coords.device)
    for i in range(min(Q, 256)):  # safety cap — full vectorize in 03-03
        n = int(min(n_real[i].item(), 256))
        if n > 0:
            patches[i, :n] = ptall_coords[i, :n]
        # remainder stays zero (D-05 zero-pad)

    # Normalise by feature_radius (unchanged scale contract)
    patches = patches / feature_radius

    # Feed through descriptor model in batches
    feat_dim = 64  # default; overridden below
    local_features = []
    for start in range(0, Q, batch_size):
        end = min(start + batch_size, Q)
        batch_patches = patches[start:end, :, :].unsqueeze(0)  # [1, B, 256, 3]
        batch_feat = batch_patches.squeeze(0).cuda()  # [B, 256, 3]
        batch_feat, _, _ = feature_extractor(batch_feat)
        local_features.append(batch_feat)

    if local_features:
        local_features = torch.cat(local_features, 0)
    else:
        local_features = torch.zeros(0, feat_dim, device=query_pts_t.device)

    return local_features
