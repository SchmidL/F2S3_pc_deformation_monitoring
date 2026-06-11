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

import numpy as np
import torch

from f2s3.core import F2S3RunSettings, feature_based_deformation_analysis

# Intra-harness imports — harness/ is its own directory, imported directly
# when run_harness.py is invoked as a script.
import parity
import profile_wrapper
import timing

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
    log_lines: Optional[list] = None,
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
        log_lines: If provided, a list that will be populated with
            formatted log records from the pipeline run (for HARN-03
            timing harvest).  The capture handler is installed *after*
            ``F2S3RunSettings`` configures logging so it survives
            ``coloredlogs.install()``.
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

    # Install log-capture handler AFTER coloredlogs.install() so it
    # survives the handler reset in F2S3RunSettings.__post_init__.
    capture_handler = None
    if log_lines is not None:
        capture_handler = timing._CaptureHandler(log_lines)
        capture_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(capture_handler)

    try:
        with torch.no_grad():
            feature_based_deformation_analysis(settings)
    finally:
        if capture_handler is not None:
            logging.getLogger().removeHandler(capture_handler)


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

    # Capture per-stage --verbose log lines for HARN-03 timing harvest.
    log_lines: list = []

    run_pipeline(
        results_dir=out_dir,
        source=SMALL_S,
        target=SMALL_T,
        save_interim=True,   # required for descriptor .npz (Pitfall 3)
        verbose=True,        # required for per-stage timing logs (HARN-03)
        seed=seed,
        log_lines=log_lines,
    )

    _record_env_json(out_dir)
    _assert_displacement_artifact_exists(out_dir)

    # Harvest stage timings and persist as a JSON sidecar (HARN-03).
    stage_timings = timing.parse_stage_timings(log_lines)
    timing.write_timings(stage_timings, out_dir)

    logging.info(f"Baseline captured successfully in {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Subcommand: check (parity gate — HARN-02/05)
# ---------------------------------------------------------------------------

def _add_check_subparser(subparsers) -> None:
    """Register the ``check`` subcommand (parity gate)."""
    parser_check = subparsers.add_parser(
        "check",
        help="Run parity gate against baseline (HARN-02/05).",
    )
    parser_check.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to the baseline run directory.",
    )
    parser_check.add_argument(
        "--against",
        type=Path,
        default=None,
        help="Path to an existing candidate run directory to compare.",
    )
    parser_check.add_argument(
        "--warn",
        type=float,
        default=0.01,
        help="WARN threshold in metres (default: 0.01 = 1 cm, D-04).",
    )
    parser_check.add_argument(
        "--fail",
        type=float,
        default=0.05,
        help="FAIL threshold in metres (default: 0.05 = 5 cm, D-04).",
    )
    parser_check.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Opt-in deterministic seed for numpy+torch RNGs (D-09, OFF by default).",
    )
    parser_check.set_defaults(func=_cmd_check)


def _resolve_disp_files(run_dir: Path, label: str) -> list:
    """Glob and validate displacement files for *run_dir*.

    Args:
        run_dir: Root directory of a pipeline run.
        label: Human-readable label for error messages (e.g. ``"baseline"``).

    Returns:
        Sorted list of matching ``.txt`` paths.

    Raises:
        FileNotFoundError: No displacement files found.
        RuntimeError: A matched file is empty.
    """
    pattern = str(run_dir / "output" / "results" / "displacement_magnitude_tile_*.txt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No displacement files found in {label} — "
            f"expected pattern: {pattern}. "
            f"The pipeline may have silently failed (Pitfall 4)."
        )
    for path in matches:
        if os.path.getsize(path) == 0:
            raise RuntimeError(
                f"Displacement file is empty in {label}: {path} "
                f"(Pitfall 4 — pipeline produced no output)."
            )
    logging.info(f"{label}: {len(matches)} displacement file(s) found.")
    return matches


def _resolve_feat_files(run_dir: Path, label: str) -> list:
    """Glob descriptor ``.npz`` files from the **features** subdirectory.

    The descriptor ``.npz`` is written to ``<run_dir>/features/`` (NOT
    ``output/results/``).  See ``core.py:367-375``.

    Args:
        run_dir: Root directory of a pipeline run.
        label: Human-readable label for log messages.

    Returns:
        Sorted list of matching ``.npz`` paths (may be empty — Pitfall 3).
    """
    pattern = str(run_dir / "features" / "features_tile_*.npz")
    matches = sorted(glob.glob(pattern))
    if matches:
        logging.info(f"{label}: {len(matches)} descriptor .npz file(s) found.")
    else:
        logging.warning(
            f"{label}: no descriptor .npz files found at {pattern} — "
            f"descriptors require --save_interim (Pitfall 3)."
        )
    return matches


def _load_timings_json(run_dir: Path, label: str) -> Optional[dict]:
    """Load ``timings.json`` from *run_dir* if it exists.

    Returns:
        The parsed dict, or ``None`` if the file is absent or unparseable.
    """
    path = run_dir / "timings.json"
    if not path.is_file():
        logging.info(f"{label}: no timings.json found — skipping timing delta.")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning(f"{label}: failed to load {path}: {exc}")
        return None


def _print_check_report(
    parity_result: dict,
    verdict: str,
    desc_diag: Optional[dict],
    timing_delta: Optional[dict],
    warn: float,
    fail: float,
) -> None:
    """Print a structured parity-check report to stdout."""
    lines = []
    lines.append("=" * 60)
    lines.append("  F2S3 Parity Check — Regression Gate (HARN-02/05)")
    lines.append("=" * 60)
    lines.append("")

    # Headline
    max_abs = parity_result.get("max_abs", float("nan"))
    if np.isnan(max_abs):
        lines.append(f"  max_abs (gate):  nan  ->  {verdict}")
    else:
        lines.append(
            f"  max_abs (gate):  {max_abs:.6f} m  ->  {verdict}"
        )
    lines.append(f"  bands: WARN > {warn:.4f} m, FAIL > {fail:.4f} m")
    lines.append("")

    # Displacement diagnostics
    lines.append("  --- Displacement Diagnostics (D-05) ---")
    lines.append(f"  rms:              {parity_result.get('rms', 'nan')}")
    lines.append(f"  p50:              {parity_result.get('p50', 'nan')}")
    lines.append(f"  p95:              {parity_result.get('p95', 'nan')}")
    lines.append(f"  p99:              {parity_result.get('p99', 'nan')}")
    lines.append(f"  matched_fraction: {parity_result.get('matched_fraction', 'nan')}")
    lines.append(
        f"  n_baseline:       {parity_result.get('n_baseline', 'nan')}"
    )
    lines.append(
        f"  n_candidate:      {parity_result.get('n_candidate', 'nan')}"
    )
    lines.append("")

    # Descriptor diagnostic
    lines.append("  --- Descriptor Drift (D-06, diagnostic only) ---")
    if desc_diag is not None:
        lines.append(f"  cos_mean:  {desc_diag.get('cos_mean', 'nan'):.6f}")
        lines.append(f"  cos_min:   {desc_diag.get('cos_min', 'nan'):.6f}")
        lines.append(f"  l2_mean:   {desc_diag.get('l2_mean', 'nan'):.6f}")
        lines.append(f"  l2_max:    {desc_diag.get('l2_max', 'nan'):.6f}")
    else:
        lines.append("  (skipped — no descriptor .npz files found)")
    lines.append("")

    # Timing deltas
    lines.append("  --- Timing Delta vs Baseline (HARN-03) ---")
    if timing_delta is not None:
        for stage, delta_s in timing_delta.items():
            if delta_s is not None:
                sign = "+" if delta_s >= 0 else ""
                lines.append(f"  {stage:18s}: {sign}{delta_s:+.2f} s")
            else:
                lines.append(f"  {stage:18s}: (not available)")
    else:
        lines.append("  (skipped — no timings.json found)")
    lines.append("")

    # Units note (A1)
    lines.append("  Note: bands assume coordinates are in metres (A1).")
    lines.append(f"  If units differ, adjust --warn / --fail accordingly.")
    lines.append("=" * 60)

    print("\n".join(lines))


def _cmd_check(args) -> None:
    """Execute the check subcommand — the regression gate.

    Exit-code contract (HARN-05):
    - PASS / WARN → exit 0
    - FAIL       → exit non-zero (``sys.exit(1)``)
    """
    baseline_dir: Path = args.baseline
    against_dir: Optional[Path] = args.against
    warn: float = args.warn
    fail: float = args.fail
    seed: Optional[int] = args.seed

    # Validate paths (Security V5 / T-02-01)
    if not baseline_dir.is_dir():
        raise FileNotFoundError(
            f"Baseline directory not found: {baseline_dir.resolve()}"
        )
    if against_dir is None:
        raise ValueError(
            "--against is required (specify a candidate run directory to compare)."
        )
    if not against_dir.is_dir():
        raise FileNotFoundError(
            f"Candidate directory not found: {against_dir.resolve()}"
        )

    logging.info(f"check: baseline = {baseline_dir.resolve()}")
    logging.info(f"check: against  = {against_dir.resolve()}")
    logging.info(f"check: WARN > {warn:.4f} m, FAIL > {fail:.4f} m")
    if seed is not None:
        logging.info(f"check: seed = {seed}")

    # Resolve displacement files (Pitfall 4 guard)
    base_disp_files = _resolve_disp_files(baseline_dir, "baseline")
    cand_disp_files = _resolve_disp_files(against_dir, "candidate")

    # Compare the first (and for _small, only) tile
    base_disp = Path(base_disp_files[0])
    cand_disp = Path(cand_disp_files[0])
    logging.info(f"  baseline disp: {base_disp}")
    logging.info(f"  candidate disp: {cand_disp}")

    # --- Apply parity gate (D-03, D-05) ---
    r = parity.displacement_parity(base_disp, cand_disp)
    max_abs = r["max_abs"]

    # Determine verdict from band contract (D-04)
    if np.isnan(max_abs):
        verdict = "FAIL"
    elif max_abs <= warn:
        verdict = "PASS"
    elif max_abs <= fail:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    # --- Descriptor diagnostic (D-06, diagnostic only) ---
    base_feat_files = _resolve_feat_files(baseline_dir, "baseline")
    cand_feat_files = _resolve_feat_files(against_dir, "candidate")
    desc_diag = None
    if base_feat_files and cand_feat_files:
        desc_diag = parity.descriptor_drift(
            Path(base_feat_files[0]), Path(cand_feat_files[0])
        )

    # --- Timing deltas ---
    base_timing = _load_timings_json(baseline_dir, "baseline")
    cand_timing = _load_timings_json(against_dir, "candidate")
    timing_delta = None
    if base_timing is not None and cand_timing is not None:
        timing_delta = {}
        for stage in ["features", "supervoxels", "correspondences", "filtering"]:
            bt = base_timing.get(stage)
            ct = cand_timing.get(stage)
            timing_delta[stage] = (ct - bt) if (bt is not None and ct is not None) else None

    # --- Print structured report ---
    _print_check_report(r, verdict, desc_diag, timing_delta, warn, fail)

    # --- Exit-code contract (HARN-05) ---
    if verdict == "FAIL":
        logging.error(f"check: FAIL — max_abs={max_abs} > fail={fail}")
        sys.exit(1)

    if verdict == "WARN":
        logging.warning(
            f"check: WARN — max_abs={max_abs:.6f} in ({warn}, {fail}] — "
            f"surface drift without auto-blocking (D-04)."
        )
    else:
        logging.info(f"check: PASS — max_abs={max_abs:.6f} <= warn={warn}")

    # PASS and WARN exit 0 (implicit)
    return 0



# ---------------------------------------------------------------------------
# Subcommand: profile (HARN-04 profiling runbook dispatch)
# ---------------------------------------------------------------------------

def _add_profile_subparser(subparsers) -> None:
    """Register the ``profile`` subcommand (torch.profiler wrapper)."""
    parser_profile = subparsers.add_parser(
        "profile",
        help="Run the _small pair under torch.profiler and print a CPU+CUDA "
        "attribution table (HARN-04).",
    )
    parser_profile.add_argument(
        "--out",
        type=Path,
        default=Path("profile_out"),
        help="Output directory for the profiled pipeline artifacts "
        "(default: profile_out/).",
    )
    parser_profile.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Opt-in deterministic seed for reproducibility (D-09).",
    )
    parser_profile.add_argument(
        "--row-limit",
        type=int,
        default=20,
        help="Maximum rows in the printed profiler table (default: 20).",
    )
    parser_profile.set_defaults(func=_cmd_profile)


def _cmd_profile(args) -> None:
    """Execute the profile subcommand — runs the pipeline under torch.profiler.

    Wraps the full pipeline invocation on the ``_small`` pair in a
    ``torch.profiler.profile`` context so the printed ``key_averages``
    table attributes time across CPU ops, CUDA kernels, and host↔device
    transfers.  This satisfies HARN-04's "non-empty attribution table"
    requirement.

    High CPU self-time + low CUDA time on the feature-extraction stage
    confirms the CPU-bound extraction thesis (§2 of the source plan);
    the interpretation rule is documented in ``PROFILING_RUNBOOK.md``.
    """
    out_dir: Path = args.out
    seed: int = args.seed
    row_limit: int = args.row_limit

    logging.info(f"Profile: pipeline -> {out_dir.resolve()}")
    logging.info(f"  source: {SMALL_S}")
    logging.info(f"  target: {SMALL_T}")
    if seed is not None:
        logging.info(f"  seed:   {seed}")

    # Run the full pipeline under torch.profiler so every PyTorch op —
    # including the two heaviest stages (compute_local_features and
    # filter_correspondences) — is captured with CPU + CUDA attribution.
    profile_wrapper.profile_pipeline(
        run_fn=lambda: run_pipeline(
            results_dir=out_dir,
            source=SMALL_S,
            target=SMALL_T,
            save_interim=False,  # skip .npz to keep the profile lean
            verbose=True,        # per-stage timing logs
            seed=seed,
        ),
        row_limit=row_limit,
    )

    _record_env_json(out_dir)
    _assert_displacement_artifact_exists(out_dir)

    logging.info(f"Profile completed — artifacts in {out_dir.resolve()}")
    logging.info(
        "Interpretation: high CPU self-time + low CUDA time on "
        "feature-extraction confirms CPU-bound extraction.  "
        "See PROFILING_RUNBOOK.md for the full attribution guide."
    )


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
