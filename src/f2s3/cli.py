import argparse
from pathlib import Path

import torch

from .core import F2S3RunSettings, feature_based_deformation_analysis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_cloud', type=Path,
                        help='Path to the source point cloud (ignored if ``--start_from_tiled_data``).')
    parser.add_argument('-t', '--target_cloud', type=Path,
                        help='Path to the target point cloud (ignored if ``--start_from_tiled_data``).')
    parser.add_argument("--start_from_tiled_data", action="store_true",
                        help="Start processing from pre-tiled data (needs to be done with ``pc_tiling``)")
    parser.add_argument("--tiled_data", type=Path,
                        help="Used to store (and retrieve in case of ``--start_from_tiled_data``) tiled data. "
                             "If not set, then ``{results_dir}/tiled_data`` (in case of ``--start_from_tiled_data`` "
                             "the path has to be explicitly set).")

    parser.add_argument('-r', '--results_dir', type=str, help='Path to the root results directory', default="")

    parser.add_argument('--max_points_per_tile', type=int, default=argparse.SUPPRESS,
        help='Maximum number of points per each tile. Should be decreased if the program runs OOM.')
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS,
        help='Number of point patches pushed to the GPU per batch during local feature extraction '
             '(default 10000). Increase to improve GPU occupancy, decrease if running OOM.')

    parser.add_argument('--voxel_grid_size', type=float, default=argparse.SUPPRESS,
        help='Target spacing for for voxel grid filter. Used to make the point density more uniform (No Filter if 0)')
    parser.add_argument('--max_disp_magnitude', type=float, default=argparse.SUPPRESS,
        help='Maximum displacement magnitude, all points with higher magnitude will be filtered out (no filtering if 0)')
    parser.add_argument('--max_search_distance', type=float, default=argparse.SUPPRESS,
        help='Maximum plausible displacement in meters. Constrains the correspondence SEARCH: only '
             'target points within this distance of the source point are accepted as a match, so an '
             'ambiguous point can still be matched via a lower-ranked feature candidate. Source '
             'points without any plausible candidate are dropped. This differs from '
             '--max_disp_magnitude, which only deletes points from the finished result. '
             'No spatial constraint if 0 (default).')
    parser.add_argument('--search_candidates', type=int, default=argparse.SUPPRESS,
        help='Number of feature-space candidates inspected per source point before giving up on it '
             '(default 20). Only used when --max_search_distance > 0. The default follows the '
             'ambiguity investigation, where the median number of near-tied candidates was 4 and '
             'the measurement was right-censored at 20.')
    parser.add_argument('--neighborhood_radius', type=float, default=argparse.SUPPRESS,
        help='Neighborhood radius for local feature extraction. Overrides the default '
             'sqrt(3) * 10 * median_resolution (auto-derived if 0)')
    parser.add_argument('--multiscale_radii', type=float, nargs='+', default=argparse.SUPPRESS,
        help='Two or more neighborhood radii in meters, e.g. "--multiscale_radii 2 6 12". The '
             'descriptor is computed once per radius and the per-scale vectors are L2-normalised '
             'and concatenated (scaled by 1/sqrt(S)), so the correspondence search sees a '
             'S*dim-dimensional descriptor in which every scale weighs the same. Coarse scales add '
             'context that breaks ties the fine scale cannot break on self-similar terrain. '
             'Supersedes --neighborhood_radius for the feature extraction; the tile overlap follows '
             'the coarsest radius. Cost is roughly linear in the number of scales. LIMITATION: each '
             'scale is described from a fixed-size subsample of its neighborhood (256 points for '
             'DIP, 4000 LRF points for GeDi), so a coarse scale is a sparser sampling of a larger '
             'area, not a richer description of it.')
    parser.add_argument('--supervoxel_radius', type=float, default=argparse.SUPPRESS,
        help='Fixed supervoxel radius in meters (unit of local-consistency filtering). '
             'Overrides the default max(sqrt(3)*10*median_resolution, voxel_grid_size). '
             'Derived automatically if 0.')
    parser.add_argument('--auto_voxel_grid_size', action='store_true', default=argparse.SUPPRESS,
        help='Derive --voxel_grid_size from the data instead of setting it by hand. The cell size '
             'is the maximum over the two epochs of the --auto_voxel_quantile percentile of the '
             'nearest-neighbor point spacing, rounded up to the centimeter. Rationale: a voxel '
             'filter can only thin regions that are denser than its cell, so the cell has to be at '
             'least as large as the coarser epoch spacing for the two epochs to end up with a '
             'comparable density. Cannot be combined with --start_from_tiled_data.')
    parser.add_argument('--auto_voxel_quantile', type=float, default=argparse.SUPPRESS,
        help='Percentile of the nearest-neighbor spacing used by --auto_voxel_grid_size '
             '(default 90). 50 gives the median spacing and equalises only half of the surface; '
             '90 equalises the density over 90 %% of it at the price of a coarser cloud.')
    parser.add_argument('--extraction_backend', type=str, default=argparse.SUPPRESS,
        help='Extraction backend for local feature patch construction: "gpu" (default) '
             'or "cpu" (the original DataLoader/CPU-extract path). '
             'The gpu backend constructs patches on-device and does NOT use the '
             'DataLoader workers; --Q W-02/--Q W-03 knobs apply only to the cpu backend.')
    parser.add_argument('--extraction_seed', type=int, default=argparse.SUPPRESS,
        help='Seed for the GPU subsample (default 41). Deterministic by default. '
             'Set to a non-negative integer.')
    parser.add_argument('--descriptor', type=str, default=argparse.SUPPRESS,
        help='Local feature descriptor backend: "dip" (default, 64-d) or "gedi" (32-d). '
             'The gedi backend requires the pointnet2_ops and torchgeometry packages and '
             'ignores --extraction_backend (GeDi builds its own patches internally).')
    parser.add_argument('--gedi_r_lrf', type=float, default=argparse.SUPPRESS,
        help='LRF / patch radius for the GeDi descriptor. If 0 (default), the pipeline '
             'neighborhood_radius is used. Only applies to --descriptor gedi. GeDi samples '
             '4000 points per patch; if your cloud is sparse relative to this radius the '
             'patches get heavily up-sampled (replicated), which weakens the descriptors. '
             'If GeDi matches look weak, increase this so each patch holds more real points.')
    parser.add_argument(
        '--save_interim', action='store_true',
        help='Save interim results such as pointwise feature desriptors and supervoxels.')
    parser.add_argument(
        '--save_supervoxels', action='store_true',
        help='Save the per-point supervoxel assignment of every source tile to '
             '<results_dir>/supervoxels/supervoxel_idx_tile_<tile>.npz (xyz + label + radius). '
             'Unlike --save_interim this writes only the supervoxel mapping, not the (much larger) '
             'feature descriptor and correspondence dumps. The label array is row-aligned with '
             'tiled_data/source_tile_<tile>.ply and lets downstream tools join results back to the '
             'supervoxel they came from.')
    parser.add_argument(
        '--verbose', action='store_true', help='Print out more details about the process to the command line')
    parser.add_argument(
        '--filter_median_magnitude',
        action='store_true',
        help='filter the remaining outliers by 30x median magnitude (improves visualizations where large displacements do not dominate)')
    parser.add_argument(
        '--refine_results',
        action='store_true',
        help='refine correspondence based displacement vectors with the estimate transformation parameters')
    parser.add_argument(
        '--fill_gaps_c2c',
        action='store_true',
        help='estimate the displacement with c2c for the points that were filtered by the outlier filtering algorithm')

    args = parser.parse_args()

    run_settings = F2S3RunSettings(**vars(args))



    with torch.no_grad():
        feature_based_deformation_analysis(run_settings)


if __name__ == '__main__':
    main()
