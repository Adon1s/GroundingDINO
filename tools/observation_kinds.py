"""Observation-kind ontology v2 — the single authoritative kind enum.

Dependency-free by contract: classifier (Pass 2c), catalog validator,
embeddings index, and Pass 2d resolver all import from here so the
three-kind vocabulary cannot drift between them.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Observation-kind ontology v2 (Pass 2c semantic contract)
# ─────────────────────────────────────────────────────────────────────────────
# defect        — expected function, safety, integrity, or protection has FAILED
# degradation   — functional but visibly deteriorated (wear/fading/weathering)
# modernization — functional and acceptably maintained but dated/basic
# Every other kind of text lands in the excluded lane with a closed-enum reason.
# Historical artifacts without an ontology_version are legacy_v1 and are never
# reinterpreted (see tools/pipeline_common.py).

ONTOLOGY_VERSION = "observation-kind-v2"

OBSERVATION_KINDS = frozenset({"defect", "degradation", "modernization"})

EXCLUSION_REASONS = frozenset({
    "good_condition",
    "neutral_presence",
    "advice_or_process",
    "unsupported_or_speculative",
    "measurement_overlay",
    "not_renovation_related",
})

# v1 catalog vocabulary (tools/issue_catalog.json). Retired for observations by
# Task 1; still valid for the shipped v1 catalog until the Task 3 cutover.
LEGACY_CATALOG_KINDS = frozenset({"defect", "upgrade"})
