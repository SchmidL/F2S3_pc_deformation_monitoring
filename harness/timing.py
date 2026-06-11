"""
Per-stage timing harvest for the F2S3 harness (HARN-03).

Parses the existing ``--verbose`` stage log lines emitted by the pipeline
(see ``src/f2s3/core.py``) and serializes them to a ``timings.json`` sidecar
that is comparable across runs.

.. note::

    **Assumption A4:** The log strings are the parse contract.  If stage log
    messages change in later phases the regexes here must be updated — they
    are kept in one place (this module) so that is easy.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Regex patterns — one per stage, extracted from the exact log strings in
# src/f2s3/core.py.  Keep in sync with the pipeline.
# ---------------------------------------------------------------------------
_STAGE_PATTERNS: Dict[str, str] = {
    # core.py:295  '{} supervoxels computed in: {:.2f} s ...'
    "supervoxels": r"(\d+)\s+supervoxels\s+computed\s+in:\s+([\d.]+)\s+s",
    # core.py:364  'Local features were computed in {:.2f} s'
    "features": r"Local\s+features\s+were\s+computed\s+in\s+([\d.]+)\s+s",
    # core.py:416  'Correspondence estimation in the feature space completed in {:.2f} s'
    "correspondences": r"Correspondence\s+estimation\s+in\s+the\s+feature\s+space\s+completed\s+in\s+([\d.]+)\s+s",
    # core.py:567  'Outlier detection step completed in {:.2f} s'
    "filtering": r"Outlier\s+detection\s+step\s+completed\s+in\s+([\d.]+)\s+s",
}

# Pre-compile for speed (called once per run, but good practice).
_COMPILED_PATTERNS: Dict[str, re.Pattern] = {
    key: re.compile(pattern) for key, pattern in _STAGE_PATTERNS.items()
}


def parse_stage_timings(log_lines: List[str]) -> Dict[str, Optional[float]]:
    """Extract per-stage elapsed seconds from *log_lines*.

    Each line is tested against the four known stage patterns.  The first
    match for a given stage wins (later duplicates are ignored — typical
    for a single-tile run).

    Args:
        log_lines: Log messages captured during a pipeline run
            (e.g. from a ``logging.Handler`` attached to the root logger).

    Returns:
        Dict with keys ``features``, ``supervoxels``, ``correspondences``,
        ``filtering``.  Values are ``float`` seconds or ``None`` if the
        stage line was absent from the captured logs.
    """
    timings: Dict[str, Optional[float]] = {
        "features": None,
        "supervoxels": None,
        "correspondences": None,
        "filtering": None,
    }

    for line in log_lines:
        for stage, pattern in _COMPILED_PATTERNS.items():
            if timings[stage] is not None:
                continue  # already captured — skip
            match = pattern.search(line)
            if match:
                # The float is always the *last* capture group for all
                # four patterns (single float for features/correspondences/
                # filtering; group 2 for supervoxels which also captures
                # the count in group 1).
                timings[stage] = float(match.groups()[-1])

    return timings


def write_timings(timings: Dict[str, Any], out_dir: Path) -> None:
    """Write *timings* dict to ``timings.json`` inside *out_dir*.

    Args:
        timings: The dictionary returned by :func:`parse_stage_timings`
            (or any dict with at least the four stage keys).
        out_dir: Directory to write ``timings.json`` into (created if
            it does not exist).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "timings.json"
    with open(path, "w") as f:
        json.dump(timings, f, indent=2, sort_keys=True)
    logging.info(f"Per-stage timings written to {path}")


# ---------------------------------------------------------------------------
# Log-capture handler — used by run_harness.py to intercept the pipeline's
# --verbose output without modifying any pipeline stage.
# ---------------------------------------------------------------------------

class _CaptureHandler(logging.Handler):
    """A ``logging.Handler`` that appends formatted records to a list."""

    def __init__(self, target: List[str]):
        super().__init__()
        self._target = target

    def emit(self, record: logging.LogRecord) -> None:
        self._target.append(self.format(record))


def capture_log_lines() -> tuple:
    """Install a list-appending handler on the root logger.

    Returns:
        A ``(lines_list, handler)`` tuple.  Caller must remove the handler
        after the pipeline run and pass ``lines_list`` to
        :func:`parse_stage_timings`.
    """
    lines: List[str] = []
    handler = _CaptureHandler(lines)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    return lines, handler
