#!/usr/bin/env bash
# T11b - multi-scale descriptor combined with the T9/T10 spatial constraint.
#
# --max_search_distance 40 m: only target points within 40 m of the source point are
# accepted, so an ambiguous point falls back to a lower-ranked but spatially plausible
# candidate instead of being sent across the tile. 40 m is well above the T9 calibration
# (21 m from 1.5 x p99 of the anchor magnitudes) and equals the |v| >= 40 m outlier proxy,
# so it is the loosest constraint that can still remove a "gross outlier" by definition.
#
# Both cells are run so the descriptor effect and the constraint effect can be separated:
#
#            no constraint                     --max_search_distance 40
#   single   results/TCH_gedi_r6_vauto         results/TCH_gedi_r6_vauto_d40
#   multi    results/TCH_gedi_ms2_6_12_vauto   results/TCH_gedi_ms2_6_12_vauto_d40
#
# The tiles of the two unconstrained runs are reused (--start_from_tiled_data), so the
# constrained runs see byte-identical input and the comparison isolates the constraint.
# --save_interim is deliberately NOT set: the descriptors would be identical to the
# unconstrained runs (the constraint acts after feature extraction) and cost ~1.1 GB.
set -euo pipefail

COMMON="--supervoxel_radius 4 --descriptor gedi --verbose"

f2s3 --start_from_tiled_data --tiled_data results/TCH_gedi_r6_vauto/tiled_data \
     -r results/TCH_gedi_r6_vauto_d40 \
     --neighborhood_radius 6 --max_search_distance 40 $COMMON

f2s3 --start_from_tiled_data --tiled_data results/TCH_gedi_ms2_6_12_vauto/tiled_data \
     -r results/TCH_gedi_ms2_6_12_vauto_d40 \
     --multiscale_radii 2 6 12 --max_search_distance 40 $COMMON
