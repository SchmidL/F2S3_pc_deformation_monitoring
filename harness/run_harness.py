"""
F2S3 regression harness — single-command entry point.

Provides ``capture`` (baseline capture), ``check`` (parity gate), and
``profile`` (profiling runbook dispatch) subcommands.  This is the named
gate that Phases 2–7 invoke by name (HARN-05).
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

from f2s3.core import F2S3RunSettings, feature_based_deformation_analysis

# ---------------------------------------------------------------------------
# Fixed reference pair (D-01, D-02 — small-only gate)
# ---------------------------------------------------------------------------
SMALL_S = "data/20210924_THC_centered_small.ply"
SMALL_T = "data/20220831_THC_centered_small.ply"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_pipeline(
    results_dir: Path,
    source: str,
    target: str,
    save_interim: bool = True,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> None:
    """Run the real F2S3 pipeline in-process on a single tile-pair.

    Mirrors ``src/f2s3/cli.py:main`` — imports and calls
    ``feature_based_deformation_analysis`` under ``torch.no_grad()`` so
    the harness measures the exact path Phases 2–7 optimize.

    Args:
        results_dir: Root results directory (becomes the baseline dir).
        source: Path to the source ``.ply`` cloud.
        target: Path to the target ``.ply`` cloud.
        save_interim: Write feature-descriptor ``.npz`` and other interim
            files.  Must be ``True`` for descriptor parity diagnostics.
        verbose: Enable per-stage timing log lines consumed by HARN-03.
        seed: If not ``None``, seed numpy + torch RNGs before the run
            (D-09 opt-in determinism, OFF by default).
    """
    if seed is not None:
        import numpy as np

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logging.info(f"Seeded numpy + torch RNGs with seed={seed} (D-09 opt-in)")

    settings = F2S3RunSettings(
        results_dir=results_dir,
        source_cloud=Path(source),
        target_cloud=Path(target),
        save_interim=save_interim,
        verbose=verbose,
    )

    with torch.no_grad():
        feature_based_deformation_analysis(settings)


def _record_env_json(out_dir: Path) -> None:
    """Write provenance ``env.json`` to *out_dir*.

    Records the actual torch / CUDA / Open3D toolchain so the cu118-vs-cu126
    gap is visible when Phases 3/5 select GPU libraries (Pitfall 5).
    """
    import open3d as o3d

    env = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "open3d": o3d.__version__,
    }
    path = out_dir / "env.json"
    with open(path, "w") as f:
        json.dump(env, f, indent=2, sort_keys=True)
    logging.info(f"Provenance recorded to {path}")


def _assert_displacement_artifact_exists(out_dir: Path) -> None:
    """Guard against a silently-failed pipeline run (Pitfall 4).

    The pipeline swallows per-tile exceptions (``core.py:692-693``) — a
    missing or empty displacement file is a hard harness error, distinct
    from a future parity FAIL.
    """
    pattern = str(out_dir / "output" / "results" / "displacement_magnitude_tile_*.txt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No displacement artifact found matching {pattern}. "
            f"The pipeline may have silently failed — check the log output above."
        )
    for path in matches:
        if os.path.getsize(path) == 0:
            raise RuntimeError(
                f"Displacement artifact {path} exists but is empty. "
                f"Pipeline produced no output — check the log output above."
            )
    logging.info(
        f"Artifact-existence guard passed: {len(matches)} displacement file(s) found."
    )


# ---------------------------------------------------------------------------
# Subcommand: capture
# ---------------------------------------------------------------------------

def _add_capture_subparser(subparsers) -> None:
    """Register the ``capture`` subcommand."""
    parser_capture = subparsers.add_parser(
        "capture",
        help="Capture FP32 baseline artifacts on the _small pair (HARN-01).",
    )
    parser_capture.add_argument(
        "--out",
        type=Path,
        default=Path("baseline"),
        help="Output directory for the baseline artifact set (default: baseline/).",
    )
    parser_capture.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Opt-in deterministic seed for numpy+torch RNGs (D-09, OFF by default).",
    )
    parser_capture.set_defaults(func=_cmd_capture)


def _cmd_capture(args) -> None:
    """Execute the capture subcommand."""
    out_dir: Path = args.out
    seed: Optional[int] = args.seed

    logging.info(f"Capture: baseline -> {out_dir.resolve()}")
    logging.info(f"  source: {SMALL_S}")
    logging.info(f"  target: {SMALL_T}")
    if seed is not None:
        logging.info(f"  seed:   {seed}")

    run_pipeline(
        results_dir=out_dir,
        source=SMALL_S,
        target=SMALL_T,
        save_interim=True,   # required for descriptor .npz (Pitfall 3)
        verbose=True,        # required for per-stage timing logs (HARN-03)
        seed=seed,
    )

    _record_env_json(out_dir)
    _assert_displacement_artifact_exists(out_dir)

    logging.info(f"Baseline captured successfully in {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Subcommand stubs (fleshed out in Plans 02 / 03)
# ---------------------------------------------------------------------------

def _add_check_subparser(subparsers) -> None:
    """Register the ``check`` subcommand (stub — Plan 02)."""
    parser_check = subparsers.add_parser(
        "check",
        help="Run parity gate against baseline (Plan 02).",
    )
    parser_check.set_defaults(func=_cmd_stub("check", "Plan 02"))


def _add_profile_subparser(subparsers) -> None:
    """Register the ``profile`` subcommand (stub — Plan 03)."""
    parser_profile = subparsers.add_parser(
        "profile",
        help="Profiling runbook dispatch (Plan 03).",
    )
    parser_profile.set_defaults(func=_cmd_stub("profile", "Plan 03"))


def _cmd_stub(name: str, planned_in: str):
    """Return a stub handler that informs the user the subcommand is pending."""

    def _stub(args) -> None:
        logging.warning(
            f"'{name}' subcommand is not yet implemented — planned in {planned_in}."
        )

    return _stub


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="F2S3 regression harness — baseline capture, parity gate, profiling."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_capture_subparser(subparsers)
    _add_check_subparser(subparsers)
    _add_profile_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
