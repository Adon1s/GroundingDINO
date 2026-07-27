"""Reference authoring: YAML draft in, canonical JSON out.

Annotation is hand work across hundreds of findings, and JSON cannot carry a
comment. So the authored artifact is ``reference.draft.yaml`` — commented,
diffable, the git source of truth — and the harness compiles it to a canonical
``reference.json`` that the evaluator reads.

Compilation **materializes every default explicitly**. A sealed reference must
not depend on an implicit default, because changing that default later would
retroactively alter truth that a human already reviewed and a baseline was
already accepted against.

The template is deliberately blank-first: it contains the schema, the controlled
vocabularies, and the listing's own photo keys, and no model output whatsoever.
Prefilling from a model would let that model define the universe of findings the
gold set can contain. Reconciliation against a source-blinded union is a later,
separate step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from tools.benchmarking.schemas import (
    REFERENCE_SCHEMA_VERSION,
    ValidationResult,
    validate_reference,
)
from tools.benchmarking.vocabulary import (
    COVERAGE_DOMAIN_KEYS,
    VALID_ACTIONABILITY,
    VALID_CATALOG_STATUSES,
    VALID_PACKAGE_DECISIONS,
    VALID_PRESENCE,
    VALID_REPORTING_EXPECTATIONS,
    VALID_VISUAL_SUFFICIENCY,
)
from tools.comparison_common import ComparisonError, atomic_json

DRAFT_FILENAME = "reference.draft.yaml"
COMPILED_FILENAME = "reference.json"

# Core-field defaults, materialized at compile time. Chosen so the common case
# ("this is really here and a good analyzer should report it") needs no typing.
_FINDING_DEFAULTS: Dict[str, Any] = {
    "presence": "present",
    "reporting_expectation": "required",
    "catalog_status": "matched",
    "visual_sufficiency": "sufficient",
    "critical": False,
}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def derive_actionability(catalog_item_ids: List[str], vocabulary: Dict[str, Any]) -> Optional[str]:
    """Actionability implied by the matched catalog items.

    Returns None when the items disagree, so the annotator is asked rather than
    silently given the first item's answer — a finding that maps to both a repair
    and a modernization item is a mapping decision a human should make.
    """
    table: Dict[str, str] = vocabulary.get("actionability_by_catalog_item") or {}
    values = {table[item_id] for item_id in catalog_item_ids if item_id in table}
    return values.pop() if len(values) == 1 else None


def normalize(raw: Dict[str, Any], *, listing: Dict[str, Any],
              vocabulary: Dict[str, Any]) -> Dict[str, Any]:
    """Turn an authored draft into a canonical reference with no implicit values."""
    if not isinstance(raw, dict):
        raise ComparisonError("reference draft must be a YAML mapping")

    reference: Dict[str, Any] = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "listing_id": str(listing.get("listing_id") or ""),
        "dataset_version": str(listing.get("dataset_version") or ""),
        "tier": raw.get("tier"),
        "review_status": raw.get("review_status", "draft"),
        "annotation_phase": raw.get("annotation_phase", 1),
        "reference_coverage": raw.get("reference_coverage", "targeted"),
        "vocabulary_fingerprint": str(vocabulary.get("fingerprint") or ""),
    }

    coverage_domain = raw.get("coverage_domain")
    if isinstance(coverage_domain, dict):
        reference["coverage_domain"] = {
            key: coverage_domain.get(key) for key in COVERAGE_DOMAIN_KEYS
            if key in coverage_domain
        }
    elif reference["reference_coverage"] == "exhaustive":
        reference["coverage_domain"] = {}

    reference["rooms"] = [
        {"room_id": str(room.get("room_id") or ""), "room_type": str(room.get("room_type") or "")}
        for room in _as_list(raw.get("rooms")) if isinstance(room, dict)
    ]

    reference["photo_expectations"] = [
        _normalize_photo_expectation(entry)
        for entry in _as_list(raw.get("photo_expectations")) if isinstance(entry, dict)
    ]

    reference["findings"] = [
        _normalize_finding(finding, vocabulary)
        for finding in _as_list(raw.get("findings")) if isinstance(finding, dict)
    ]

    components = _as_list(raw.get("billable_components"))
    if components:
        reference["billable_components"] = [
            _normalize_component(component) for component in components
            if isinstance(component, dict)
        ]

    packages = _as_list(raw.get("packages"))
    if packages:
        reference["packages"] = [
            _normalize_package(package) for package in packages if isinstance(package, dict)
        ]

    if isinstance(raw.get("scope"), dict):
        reference["scope"] = {
            "scope_band": raw["scope"].get("scope_band"),
            "band_confidence": raw["scope"].get("band_confidence"),
        }
    if isinstance(raw.get("cost"), dict):
        cost = raw["cost"]
        reference["cost"] = {
            "expected_band": cost.get("expected_band"),
            "confidence": cost.get("confidence"),
            "basis": cost.get("basis"),
        }
    return reference


def _normalize_photo_expectation(entry: Dict[str, Any]) -> Dict[str, Any]:
    indeterminate = bool(entry.get("indeterminate", False))
    visible = [str(v) for v in _as_list(entry.get("visible_room_ids"))]
    primary = entry.get("primary_room_id")
    primary = str(primary) if primary else None
    # A photo whose primary room is named but not listed as visible is almost
    # always an authoring slip rather than a claim; fill it rather than fail.
    if primary and primary not in visible:
        visible.insert(0, primary)
    return {
        "photo_key": str(entry.get("photo_key") or ""),
        "primary_room_id": primary,
        "visible_room_ids": visible,
        "expected_group": (str(entry["expected_group"])
                           if entry.get("expected_group") else None),
        "accepted_scene_ids": [str(s) for s in _as_list(entry.get("accepted_scene_ids"))],
        "indeterminate": indeterminate,
    }


def _normalize_finding(finding: Dict[str, Any], vocabulary: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = dict(_FINDING_DEFAULTS)
    normalized.update({
        "id": str(finding.get("id") or ""),
        "description": str(finding.get("description") or ""),
        "room_id": str(finding.get("room_id") or ""),
        "catalog_item_ids": [str(i) for i in _as_list(finding.get("catalog_item_ids"))],
        "aliases": [str(a) for a in _as_list(finding.get("aliases"))],
        "evidence": [
            {"photo_key": str(occurrence.get("photo_key") or ""),
             **({"note": str(occurrence["note"])} if occurrence.get("note") else {})}
            for occurrence in _as_list(finding.get("evidence"))
            if isinstance(occurrence, dict)
        ],
    })
    for key in _FINDING_DEFAULTS:
        if finding.get(key) is not None:
            normalized[key] = finding[key]

    if finding.get("billable_component_key"):
        normalized["billable_component_key"] = str(finding["billable_component_key"])

    # Derived rather than typed, so actionability costs the annotator nothing on
    # the ~95% of findings that map cleanly to a catalog item.
    actionability = finding.get("actionability")
    if not actionability and normalized["catalog_status"] != "missing_catalog_item":
        actionability = derive_actionability(normalized["catalog_item_ids"], vocabulary)
    if actionability:
        normalized["actionability"] = str(actionability)
    return normalized


def _normalize_component(component: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "key": str(component.get("key") or ""),
        "room_id": str(component.get("room_id") or ""),
        "label": str(component.get("label") or ""),
        "finding_ids": [str(f) for f in _as_list(component.get("finding_ids"))],
        "expected_units": component.get("expected_units", 1),
    }
    if component.get("expected_cost_band") is not None:
        normalized["expected_cost_band"] = component["expected_cost_band"]
    return normalized


def _normalize_package(package: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "package_id": str(package.get("package_id") or ""),
        "package_type": str(package.get("package_type") or ""),
        "room_id": str(package.get("room_id") or ""),
        "decision": package.get("decision"),
        "tier": package.get("tier"),
        "confirmed_finding_ids": [str(f) for f in _as_list(package.get("confirmed_finding_ids"))],
        "component_keys": [str(k) for k in _as_list(package.get("component_keys"))],
        "reason": str(package.get("reason") or ""),
    }


# ---------------------------------------------------------------------------
# Load / compile / check
# ---------------------------------------------------------------------------

def load_draft(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ComparisonError(f"reference draft not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise ComparisonError(f"reference draft is not valid YAML: {path}: {exc}") from exc
    if raw is None:
        raise ComparisonError(f"reference draft is empty: {path}")
    if not isinstance(raw, dict):
        raise ComparisonError(f"reference draft must be a mapping, got "
                              f"{type(raw).__name__}: {path}")
    return raw


def check(draft_path: Path, *, listing: Dict[str, Any],
          vocabulary: Dict[str, Any]) -> Tuple[Dict[str, Any], ValidationResult]:
    """Normalize and validate a draft without writing anything."""
    raw = load_draft(draft_path)
    reference = normalize(raw, listing=listing, vocabulary=vocabulary)
    return reference, validate_reference(reference, listing=listing, vocabulary=vocabulary)


def compile_draft(draft_path: Path, *, listing: Dict[str, Any], vocabulary: Dict[str, Any],
                  output_path: Optional[Path] = None) -> Path:
    """Compile a validated draft to canonical JSON. Refuses to emit on any error."""
    reference, result = check(draft_path, listing=listing, vocabulary=vocabulary)
    if not result.ok:
        lines = [f"{draft_path}: {len(result.errors)} validation error(s)"]
        lines.extend(f"  - {message}" for message in result.errors)
        raise ComparisonError("\n".join(lines))
    destination = output_path or draft_path.with_name(COMPILED_FILENAME)
    atomic_json(destination, reference)
    return destination


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

def _vocab_comment(label: str, values: Any) -> str:
    return f"#   {label}: {', '.join(sorted(str(v) for v in values))}"


def template(listing: Dict[str, Any], vocabulary: Dict[str, Any], *, tier: str = "gold") -> str:
    """Render a blank-first draft with every photo pre-listed.

    Photo keys come from the frozen dataset, not from any model. Pre-listing them
    is what stops a photo from being silently skipped during annotation.
    """
    listing_id = str(listing.get("listing_id") or "")
    photos = sorted(
        (p for p in (listing.get("photos") or []) if isinstance(p, dict)),
        key=lambda p: p.get("order", 0),
    )
    scene_ids = sorted((vocabulary.get("scene_ids") or {}))
    groups = sorted(vocabulary.get("ui_scene_groups") or ())
    bands = sorted(vocabulary.get("rehab_scope_bands") or ())

    lines: List[str] = [
        f"# Reference draft for {listing_id} (tier: {tier}).",
        "#",
        "# PHASE 1 is blind: work only from the photos and the frozen listing metadata.",
        "# Do not look at any model output. Use `benchmark catalog search <query>` to find",
        "# catalog items; the catalog is not model output, so consulting it is safe. If",
        "# nothing fits, set catalog_status: missing_catalog_item and leave",
        "# catalog_item_ids empty -- that records a catalog gap instead of pretending the",
        "# nearest item is right.",
        "#",
        "# Compile with:  benchmark reference compile --listing " + listing_id,
        "",
        f"tier: {tier}",
        "review_status: draft            # draft | reviewed  (reviewed requires all 4 phases)",
        "annotation_phase: 1             # 1 inventory, 2 packages, 3 reconciliation, 4 cost",
        "reference_coverage: targeted    # targeted | exhaustive",
        "",
        "# Required when reference_coverage is exhaustive. Photographs cannot establish",
        "# hidden electrical, plumbing, foundation, or HVAC condition -- leave",
        "# hidden_system_conditions false and record those as presence: indeterminate.",
        "# coverage_domain:",
    ]
    lines.extend(
        f"#   {key}: {'false' if key == 'hidden_system_conditions' else 'true'}"
        for key in COVERAGE_DOMAIN_KEYS
    )
    lines.extend([
        "",
        "# ---------------------------------------------------------------------------",
        "# Physical rooms. One entry per distinct physical space, however many photos",
        "# show it. room_type is a scene id:",
        f"#   {', '.join(scene_ids)}",
        "# ---------------------------------------------------------------------------",
        "rooms: []",
        "#  - room_id: kitchen-1",
        "#    room_type: kitchen",
        "",
        "# ---------------------------------------------------------------------------",
        "# Per-photo scene truth. Separate from rooms because one photo can show",
        "# several spaces (open kitchen/living, a hallway view, a vanity vs a closet).",
        f"#   expected_group: {', '.join(groups)}",
        "# accepted_scene_ids may span groups for an open-plan photo; scoring credits",
        "# any of them. Set indeterminate: true when the photo has no defensible scene.",
        "# ---------------------------------------------------------------------------",
        "photo_expectations:",
    ])
    for photo in photos:
        lines.extend([
            f"  - photo_key: {photo.get('filename')}",
            "    primary_room_id:",
            "    visible_room_ids: []",
            "    expected_group:",
            "    accepted_scene_ids: []",
            "    indeterminate: false",
        ])
    if not photos:
        lines.append("  []")

    lines.extend([
        "",
        "# ---------------------------------------------------------------------------",
        "# Findings. ONE entry per physical condition, however many photos show it --",
        "# list each photo under evidence instead of repeating the finding.",
        "#",
        "# Core fields (everything else is derived or defaulted):",
        "#   id, description, room_id, evidence, catalog_item_ids",
        "#",
        "# Defaults applied at compile time, so only override the exceptions:",
        "#   presence: present   reporting_expectation: required",
        "#   catalog_status: matched   visual_sufficiency: sufficient   critical: false",
        "#",
        _vocab_comment("presence", VALID_PRESENCE),
        _vocab_comment("reporting_expectation", VALID_REPORTING_EXPECTATIONS),
        _vocab_comment("visual_sufficiency", VALID_VISUAL_SUFFICIENCY),
        _vocab_comment("catalog_status", VALID_CATALOG_STATUSES),
        _vocab_comment("actionability", VALID_ACTIONABILITY),
        "#",
        "# Combination rules enforced at compile time:",
        "#   absent            -> must_not_report",
        "#   indeterminate     -> required or acceptable (never must_not_report)",
        "#   insufficient      -> presence must be indeterminate",
        "#   critical: true    -> present + required + sufficient",
        "# Allowed and useful: present + must_not_report, for a real but trivial",
        "# condition a good analyzer should stay quiet about.",
        "#",
        "# actionability is derived from the catalog item, so you only type it when",
        "# catalog_status is missing_catalog_item (or the matched items disagree).",
        "# ---------------------------------------------------------------------------",
        "findings: []",
        "#  - id: finding-001",
        "#    description: Ceiling drywall crack above the sink run",
        "#    room_id: kitchen-1",
        "#    catalog_item_ids: [damaged_drywall_or_cracks]",
        "#    evidence:",
        "#      - {photo_key: photo_020.jpg, note: primary}",
        "#      - {photo_key: photo_021.jpg}",
        "",
        "# ---------------------------------------------------------------------------",
        "# PHASE 2 and later. Leave these out entirely during phase 1.",
        "#",
        "# billable_components: one entry per unit of billable work. expected_units is",
        "# what detects repeated-evidence cost inflation -- six photos of the same dated",
        "# cabinets are one component with expected_units: 1.",
        "# ---------------------------------------------------------------------------",
        "# billable_components:",
        "#  - key: kitchen-1:ceiling-drywall",
        "#    room_id: kitchen-1",
        "#    label: Kitchen ceiling drywall repair",
        "#    finding_ids: [finding-001]",
        "#    expected_units: 1",
        "#    expected_cost_band: {low: 400, high: 900}",
        "",
        "# packages: your independent judgment, made BEFORE looking at what the",
        f"#   pipeline proposed. decision: {', '.join(sorted(VALID_PACKAGE_DECISIONS))}",
        "#   'unsupported' and 'indeterminate' require a reason.",
        "# packages:",
        "#  - package_id: pkg-001",
        "#    package_type: kitchen_modernization",
        "#    room_id: kitchen-1",
        "#    decision: unsupported",
        "#    confirmed_finding_ids: [finding-001]",
        "#    component_keys: []",
        "#    reason: Ceiling repair is real, but the visible kitchen finishes do not",
        "#      justify modernization.",
        "",
        "# PHASE 4. Market inputs live in metadata.json and must NOT be restated here.",
        f"#   scope_band: {', '.join(bands)}",
        "# scope: {scope_band: moderate, band_confidence: medium}",
        "# cost:",
        "#   expected_band: {low: 4000, high: 9000}",
        "#   confidence: medium",
        "#   basis: Given the human scope above.",
        "",
    ])
    return "\n".join(lines)


def write_template(destination: Path, listing: Dict[str, Any], vocabulary: Dict[str, Any],
                   *, tier: str = "gold", overwrite: bool = False) -> Path:
    """Write a draft template, refusing to clobber hand-authored work."""
    if destination.exists() and not overwrite:
        raise ComparisonError(
            f"{destination} already exists; refusing to overwrite annotation work. "
            f"Pass --force only if you are certain."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template(listing, vocabulary, tier=tier), encoding="utf-8")
    return destination
