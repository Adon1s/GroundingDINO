"""
tools/renovation_estimate_v4.py

Passthrough scaffold for the v4 renovation estimate.

PR 1 of a multi-PR refactor that moves the estimator from a flat line-item
view (v3) to a room-aware, package-aware rehab model (v4). This module
currently emits the same totals as v3 with the v4 version string and empty
placeholders for room_surrogates, packages, reconciliation, and provenance.
Real v4 logic (room surrogates, package inference, reconciliation) lands in
later PRs and will populate those fields.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional


def compute_renovation_estimate_v4(
    *,
    v3_estimate: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build the v4 passthrough scaffold from a precomputed v3 estimate."""
    if v3_estimate is None:
        return None
    v4 = copy.deepcopy(v3_estimate)
    v4["version"] = "renovation_estimate_v4"
    v4["room_surrogates"] = []
    v4["packages"] = []
    v4["reconciliation"] = {}
    v4["provenance"] = {
        "mode": "passthrough_scaffold",
        "derived_from": "renovation_estimate_v3",
        "v4_phases_applied": [],
    }
    return v4
