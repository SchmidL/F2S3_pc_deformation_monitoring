"""Switchable local-feature descriptor backends.

The pipeline (``PointCloudTile.compute_local_features``) delegates per-cloud
feature extraction to a ``Descriptor`` object so the descriptor can be selected at
runtime via ``--descriptor``:

- ``DipDescriptor`` wraps the original DIP ``PointNetFeature`` (64-d) and the existing
  cpu/gpu extraction paths. This is the default and is behaviourally unchanged.
- ``GediDescriptor`` wraps the vendored GeDi network (32-d). Its heavy native
  dependency (``pointnet2_ops``) is imported lazily, so a DIP-only run never needs it.

Both expose ``extract_features(query_pcd, search_pcd, neighborhood_radius) -> Tensor``
returning an ``[N, dim]`` tensor (DIP on CUDA, GeDi on CPU; callers move as needed) and
a ``dim`` attribute used to size the correspondence index.
"""
from importlib import resources
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .descriptor_model import PointNetFeature
from .data import FeatureExtractionDataset
from . import gpu_extract


class DipDescriptor:
    """DIP local descriptor (`PointNetFeature`, 64-d). Default backend.

    Holds the same cpu/gpu extraction logic that previously lived inline in
    ``compute_local_features`` so behaviour (and DIP parity) is preserved.
    """

    name = "dip"
    dim = 64

    def __init__(self, model, extraction_backend, batch_size, extraction_seed, verbose):
        self.model = model
        self.extraction_backend = extraction_backend
        self.batch_size = batch_size
        self.extraction_seed = extraction_seed
        self.verbose = verbose

    def extract_features(self, query_pcd, search_pcd, neighborhood_radius):
        if self.extraction_backend == "cpu":
            return self._extract_cpu(query_pcd, search_pcd, neighborhood_radius)
        return self._extract_gpu(query_pcd, search_pcd, neighborhood_radius)

    def _extract_cpu(self, query_pcd, search_pcd, neighborhood_radius):
        # The FeatureExtractionDataset/DataLoader workers apply only to this cpu backend.
        dataset = FeatureExtractionDataset(
            query_pcd, search_pcd, self.batch_size, neighborhood_radius)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)
        if self.verbose:
            dataloader = tqdm(dataloader, ncols=90)
        features = []
        for batch in dataloader:
            batch_feat = batch.squeeze(0).cuda()
            batch_feat, _, _ = self.model(batch_feat)
            features.append(batch_feat)
        return torch.cat(features, 0)

    def _extract_gpu(self, query_pcd, search_pcd, neighborhood_radius):
        # The GPU backend builds its patches on-device via Open3D NNS and does NOT
        # use the DataLoader workers.
        search_np = np.asarray(search_pcd.points).astype(np.float32)
        gpu_idx = gpu_extract.build_index(search_np, neighborhood_radius)
        query_np = np.asarray(query_pcd.points).astype(np.float32)
        query_t = torch.from_numpy(query_np).cuda()
        return gpu_extract.gpu_extract_features(
            query_t, gpu_idx, neighborhood_radius, self.model, self.batch_size,
            self.extraction_seed, verbose=self.verbose)


def _gedi_compute_gpu(gedi, query_pts, search_pts):
    """GPU replacement for ``GeDi.compute`` using F2S3's GPU radius search.

    Upstream ``GeDi.compute`` runs a CPU ``cKDTree`` search plus a per-point Python
    gather loop and a CPU LRF SVD — together the bulk of runtime, with the GPU idle.
    This keeps the exact GeDi network + LRF math but builds neighbourhoods with
    F2S3's GPU fixed-radius search (``gpu_extract.radius_neighborhoods``) and runs
    everything (patch sampling, LRF, network) on the GPU, chunked to bound VRAM.

    ``radius_neighborhoods`` returns the in-radius neighbour coords already centred
    on the query (``neighbour - query``); GeDi's LRF centres internally by subtracting
    its ``x0`` argument, so we pass ``x0 = 0`` — identical to feeding raw coords with
    the real query centre. Patch points are sampled uniformly **with replacement**
    from each query's in-radius set (statistically equivalent to upstream's
    all+replica padding in the point-starved F2S3 regime; GeDi is not parity-gated).

    Args:
        gedi: a vendored ``GeDi`` instance (network + LRF already on CUDA).
        query_pts: (N, 3) float32 numpy array of descriptor centres.
        search_pts: (M, 3) float32 numpy array of the cloud to search.

    Returns:
        (N, gedi.dim) numpy array of unit-normalised descriptors.
    """
    device = "cuda"
    spb = gedi.samples_per_batch
    sppl = gedi.samples_per_patch_lrf
    r_lrf = gedi.r_lrf
    N = query_pts.shape[0]
    # VRAM-bound: radius_neighborhoods builds [chunk, maxN, 3] (see gpu_extract).
    CHUNK = 60000

    gpu_index = gpu_extract.build_index(search_pts, r_lrf)
    query_t = torch.from_numpy(query_pts).float().to(device)

    desc = np.empty((N, gedi.dim))
    for c0 in range(0, N, CHUNK):
        c1 = min(c0 + CHUNK, N)
        # ptall = full in-radius set (nearest kept), coords centred on the query.
        _, _, _, ptall_coords, ptall_mask = gpu_extract.radius_neighborhoods(
            gpu_index, query_t[c0:c1], r_lrf)
        counts = ptall_mask.sum(dim=1).clamp(min=1)                 # [chunk]

        for b0 in range(0, c1 - c0, spb):
            b1 = min(b0 + spb, c1 - c0)
            coords = ptall_coords[b0:b1]                            # [b, maxN, 3] centred
            cnt = counts[b0:b1]                                     # [b]
            # Uniform-with-replacement sample positions in [0, cnt) per query.
            samp = (torch.rand(b1 - b0, sppl, device=device) * cnt[:, None]).long()
            x_c = torch.gather(coords, 1, samp[..., None].expand(-1, -1, 3))  # [b, sppl, 3]
            x = x_c.transpose(1, 2).contiguous()                   # [b, 3, sppl]

            # coords are already centred -> pass zero query centres to the LRF.
            x0 = torch.zeros(b1 - b0, 3, device=device)
            patch = gedi.lrf(x0, x)                                 # GPU LRF -> [b, 3, out]
            with torch.no_grad():
                f = gedi.gedi_net(patch)
            desc[c0 + b0:c0 + b1] = f.cpu().numpy()

    return desc


class GediDescriptor:
    """GeDi local descriptor (vendored, 32-d).

    The GeDi network owns its own neighbour search + LRF + PointNet++ sampling.
    ``r_lrf`` defaults to the per-tile ``neighborhood_radius`` (the same scale DIP
    searches over) unless a fixed value is supplied via ``--gedi_r_lrf``. The loaded
    network is reused across tiles; only ``r_lrf`` is updated per call. Extraction uses
    the GPU-vectorised :func:`_gedi_compute_gpu` path.
    """

    name = "gedi"

    def __init__(self, checkpoint_path, fixed_r_lrf=0.0, samples_per_batch=500,
                 samples_per_patch_lrf=4000, samples_per_patch_out=512, dim=32,
                 verbose=False):
        # Lazy import: pulls in pointnet2_ops / torchgeometry only when GeDi is used.
        from .gedi import GeDi

        self.dim = dim
        self.fixed_r_lrf = fixed_r_lrf
        self.verbose = verbose
        config = {
            'dim': dim,
            'samples_per_batch': samples_per_batch,
            'samples_per_patch_lrf': samples_per_patch_lrf,
            'samples_per_patch_out': samples_per_patch_out,
            # placeholder; overwritten per call in extract_features
            'r_lrf': fixed_r_lrf if fixed_r_lrf > 0.0 else 1.0,
            'fchkpt_gedi_net': str(checkpoint_path),
        }
        self.gedi = GeDi(config)
        # Run the LRF SVD on the GPU (see _gedi_compute_gpu).
        self.gedi.lrf.device = "cuda"

    def extract_features(self, query_pcd, search_pcd, neighborhood_radius):
        r_lrf = self.fixed_r_lrf if self.fixed_r_lrf > 0.0 else neighborhood_radius
        self.gedi.r_lrf = r_lrf
        self.gedi.lrf.r_lrf = r_lrf

        query_pts = np.asarray(query_pcd.points).astype(np.float32)
        search_pts = np.asarray(search_pcd.points).astype(np.float32)
        desc = _gedi_compute_gpu(self.gedi, query_pts, search_pts)  # np [N, dim]
        return torch.from_numpy(desc).float()


def build_descriptor(args):
    """Construct the descriptor backend selected by ``args.descriptor``.

    Args:
        args (F2S3RunSettings): resolved run settings.

    Returns:
        A descriptor exposing ``extract_features(...)`` and ``dim``.
    """
    if args.descriptor == "gedi":
        with resources.path('f2s3.pretrained_models.gedi', 'chkpt.tar') as ckpt:
            ckpt = Path(ckpt)
        return GediDescriptor(ckpt, fixed_r_lrf=args.gedi_r_lrf, verbose=args.verbose)

    # default: DIP
    with resources.path('f2s3.pretrained_models.feature_descriptor', 'model_best.pth') as f_m:
        f_m = Path(f_m)
    model = PointNetFeature()
    model.load_state_dict(torch.load(f_m))
    model.cuda()
    model.eval()
    return DipDescriptor(
        model, args.extraction_backend, args.batch_size, args.extraction_seed, args.verbose)
