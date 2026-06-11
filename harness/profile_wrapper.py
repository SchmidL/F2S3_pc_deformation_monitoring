"""
Thin ``torch.profiler`` context around the two heaviest ``PointCloudTile``
stage methods — ``compute_local_features`` and ``filter_correspondences``.

Used by the ``profile`` subcommand and the profiling runbook (HARN-04) to
attribute wall-clock cost across CPU vs GPU vs CPU↔GPU transfer.

Uses ``torch.profiler`` (NOT the deprecated ``torch.autograd.profiler``).
"""
import logging
from typing import List, Optional, Union

from torch.profiler import ProfilerActivity, profile, record_function


def profile_stages(
    tile: "PointCloudTile",
    activities: Optional[List[ProfilerActivity]] = None,
    row_limit: int = 20,
    sort_by: str = "cuda_time_total",
) -> profile:
    """Wrap ``compute_local_features`` and ``filter_correspondences`` in
    a ``torch.profiler`` context with ``record_function`` labels.

    Calls each method inside a labelled ``record_function`` block so the
    printed table clearly separates the two stages.

    Args:
        tile: A ``PointCloudTile`` instance whose stage methods to profile.
            Clouds must already be loaded, and ``compute_supervoxels``
            must have been called before ``filter_correspondences``.
        activities: Profiler activities to record.  Defaults to
            ``[ProfilerActivity.CPU, ProfilerActivity.CUDA]``.
        row_limit: Maximum rows in the printed ``key_averages`` table.
        sort_by: Column to sort the table by (default: ``cuda_time_total``).

    Returns:
        The ``torch.profiler.profile`` instance, so callers can inspect
        events, export a Chrome trace, or print additional breakdowns.
    """
    if activities is None:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    logging.info(
        "Starting torch.profiler session — activities=%s",
        [a.name for a in activities],
    )
    logging.info("Recording stage: compute_local_features")
    logging.info("Recording stage: filter_correspondences")

    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("compute_local_features"):
            tile.compute_local_features()
        with record_function("filter_correspondences"):
            tile.filter_correspondences()

    table = prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    logging.info("Profiler attribution table follows.\n%s", table)
    return prof


def profile_pipeline(
    run_fn,
    activities: Optional[List[ProfilerActivity]] = None,
    row_limit: int = 20,
    sort_by: str = "cuda_time_total",
) -> profile:
    """Wrap an entire pipeline invocation in a ``torch.profiler`` context.

    This is the generic entry point the ``profile`` subcommand uses when
    it cannot (or does not need to) extract a single ``PointCloudTile``
    from the orchestration loop.  The full pipeline — including model
    loading, tiling, and all stage methods — runs under the profiler.

    Args:
        run_fn: A zero-argument callable that invokes the pipeline
            (e.g. ``lambda: run_pipeline(...)``).
        activities: Profiler activities (default: CPU + CUDA).
        row_limit: Maximum rows in the printed table.
        sort_by: Column to sort by.

    Returns:
        The ``torch.profiler.profile`` instance.
    """
    if activities is None:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    logging.info(
        "Starting torch.profiler session (full pipeline) — activities=%s",
        [a.name for a in activities],
    )

    with profile(activities=activities, record_shapes=True) as prof:
        run_fn()

    table = prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    logging.info("Profiler attribution table follows.\n%s", table)
    return prof
