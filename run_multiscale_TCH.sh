#!/usr/bin/env bash
# T11 experiment - multi-scale descriptor and voxel size, on the TCH sample pair.
#
# 2x2 design on the same epoch pair used for the whole investigation:
#
#            voxel 0.20 m (baseline)      voxel = auto (p90 spacing)
#   single   results/TCH_gedi_r6_investigate    results/TCH_gedi_r6_vauto
#   multi    results/TCH_gedi_ms2_6_12_v0_2     results/TCH_gedi_ms2_6_12_vauto
#
# The single/0.20 cell is the run the investigation report is built on and already
# exists, so it is not recomputed here.
#
# Scale choice: 2 / 6 / 12 m. 6 m is the tuned single-scale radius; 2 m resolves
# individual blocks; 12 m covers the slope-scale context (gully, ridge, slope break)
# that is what actually differs between two self-similar debris patches.
#
# Voxel choice: --auto_voxel_grid_size sets the cell to the max over both epochs of
# the p90 nearest-neighbor spacing (0.30 m / 0.31 m here -> 0.32 m). A voxel filter
# only thins regions denser than its cell, so 0.20 m leaves the sparser half of the
# surface untouched and the two epochs keep different local densities.
set -euo pipefail

SRC=data/20210924_THC_centered_sample.ply
TGT=data/20220831_THC_centered_sample.ply
COMMON="--supervoxel_radius 4 --descriptor gedi --save_interim --verbose"

f2s3 --source_cloud $SRC --target_cloud $TGT -r results/TCH_gedi_ms2_6_12_v0_2 \
     --voxel_grid_size 0.2 --multiscale_radii 2 6 12 $COMMON

f2s3 --source_cloud $SRC --target_cloud $TGT -r results/TCH_gedi_r6_vauto \
     --auto_voxel_grid_size --neighborhood_radius 6 $COMMON

f2s3 --source_cloud $SRC --target_cloud $TGT -r results/TCH_gedi_ms2_6_12_vauto \
     --auto_voxel_grid_size --multiscale_radii 2 6 12 $COMMON
