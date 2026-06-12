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


class GediDescriptor:
    """GeDi local descriptor (vendored, 32-d).

    The GeDi network owns its own neighbour search + LRF + PointNet++ sampling via
    ``GeDi.compute(pts, pcd)``. ``r_lrf`` defaults to the per-tile
    ``neighborhood_radius`` (the same scale DIP searches over) unless a fixed value
    is supplied via ``--gedi_r_lrf``. The loaded network is reused across tiles; only
    ``r_lrf`` is updated per call.
    """

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

    def extract_features(self, query_pcd, search_pcd, neighborhood_radius):
        r_lrf = self.fixed_r_lrf if self.fixed_r_lrf > 0.0 else neighborhood_radius
        self.gedi.r_lrf = r_lrf
        self.gedi.lrf.r_lrf = r_lrf

        pts = torch.from_numpy(np.asarray(query_pcd.points)).float()
        pcd = torch.from_numpy(np.asarray(search_pcd.points)).float()
        desc = self.gedi.compute(pts=pts, pcd=pcd)  # np [N, dim]
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
