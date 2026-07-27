"""Benchmark schemas: strict validation and the one reporting-category derivation.

Hand-rolled, no jsonschema dependency — matching ``tools/catalog_validation.py``.
Cross-field rules ("critical requires present + required + sufficient", "empty
catalog_item_ids iff missing_catalog_item") are more natural in plain Python
than in a schema document, and every vocabulary comes from the dataset's frozen
snapshot so validation cannot drift from the truth it is checking.

Validation reports *all* errors for a file, not the first. A reference is
hand-authored across hundreds of findings; surfacing one error per run would
make fixing a draft an afternoon of round-trips.

Validation is also **phase-aware**. Annotation happens in four passes (see
benchmarks/README.md), so a phase-1 draft with no packages is a legitimate
artifact, while a ``review_status: reviewed`` reference without them is not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tools.benchmarking.vocabulary import (
    COVERAGE_DOMAIN_KEYS,
    VALID_ACTIONABILITY,
    VALID_ANNOTATION_PHASES,
    VALID_BAND_CONFIDENCE,
    VALID_CATALOG_STATUSES,
    VALID_PACKAGE_DECISIONS,
    VALID_PRESENCE,
    VALID_REFERENCE_COVERAGE,
    VALID_REPORTING_EXPECTATIONS,
    VALID_REVIEW_STATUSES,
    VALID_TIERS,
    VALID_VISUAL_SUFFICIENCY,
)
from tools.pass_config import ALL_PASSES

MANIFEST_SCHEMA_VERSION = 1
LISTING_SCHEMA_VERSION = 1
REFERENCE_SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# Scene ids that are not physical rooms and so may not be a room_type. Kept as a
# small declared exclusion rather than an allowlist so new scenes are usable as
# room types by default.
NON_ROOM_SCENE_IDS = frozenset({"floor_plan", "aerial_view", "street_view", "unknown"})

# Only the passes that actually invoke an LLM may be routed. 1b/1c are
# compatibility stubs that construct a blank result without a model call
# (scene_classifier_orchestrator.py:532-560) and 2e is rule-based, so a model
# entry for any of them would be silently ignored at runtime — the exact class of
# config error this benchmark exists to catch.
BENCHMARK_MODEL_MAP_PASSES = ("1a", "2a", "2b", "2c", "2d", "2f")
_STUB_PASSES = {"1b": "a compatibility stub (no LLM call)",
                "1c": "a compatibility stub (no LLM call)",
                "2e": "rule-based (no LLM call)"}


@dataclass
class ValidationResult:
    """Errors block; warnings are advisory and never gate."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")

    def extend(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


# ---------------------------------------------------------------------------
# Reporting category — the single derivation point
# ---------------------------------------------------------------------------

def reporting_category(finding: Dict[str, Any]) -> str:
    """Derive the reporting category from the orthogonal truth fields.

    The four categories (required/acceptable/unsupported/indeterminate) are a
    *reporting* view; they are never stored, because storing them would conflate
    presence, reporting duty, and visual sufficiency into one enum that cannot
    express "the drywall damage is real but the kitchen modernization is not".

    The cases below are disjoint *because* ``validate_reference`` rejects the
    contradictory combinations — notably ``indeterminate + must_not_report``.
    That is why this needs no precedence rules:

      must_not_report        -> unsupported    (absent, or real-but-trivial noise)
      presence indeterminate -> indeterminate  (includes insufficient photos)
      present + required     -> required
      present + acceptable   -> acceptable
    """
    if finding.get("reporting_expectation") == "must_not_report":
        return "unsupported"
    if finding.get("presence") == "indeterminate":
        return "indeterminate"
    return "required" if finding.get("reporting_expectation") == "required" else "acceptable"


# ---------------------------------------------------------------------------
# Shared field helpers
# ---------------------------------------------------------------------------

def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_str(res: ValidationResult, where: str, obj: Dict[str, Any], key: str,
                 *, pattern: Optional[re.Pattern] = None) -> str:
    value = _str(obj.get(key))
    if not value:
        res.error(where, f"{key} is required and must be a non-empty string")
        return ""
    if pattern is not None and not pattern.match(value):
        res.error(where, f"{key} {value!r} must match {pattern.pattern}")
    return value


def _require_choice(res: ValidationResult, where: str, obj: Dict[str, Any], key: str,
                    allowed: Iterable[Any], *, default: Any = None) -> Any:
    value = obj.get(key, default)
    if isinstance(value, str):
        value = value.strip()
    if value not in set(allowed):
        res.error(where, f"{key} {obj.get(key)!r} must be one of {sorted(map(str, allowed))}")
        return None
    return value


def _require_bool(res: ValidationResult, where: str, obj: Dict[str, Any], key: str,
                  *, default: Optional[bool] = None) -> Optional[bool]:
    value = obj.get(key, default)
    if not isinstance(value, bool):
        res.error(where, f"{key} must be a boolean, got {obj.get(key)!r}")
        return None
    return value


def _require_list(res: ValidationResult, where: str, obj: Dict[str, Any], key: str,
                  *, allow_missing: bool = False) -> List[Any]:
    if key not in obj and allow_missing:
        return []
    value = obj.get(key)
    if not isinstance(value, list):
        res.error(where, f"{key} must be a list, got {type(value).__name__}")
        return []
    return value


def _require_money_range(res: ValidationResult, where: str, raw: Any, key: str) -> None:
    if not isinstance(raw, dict):
        res.error(where, f"{key} must be an object with low and high")
        return
    low, high = raw.get("low"), raw.get("high")
    if not _is_number(low) or not _is_number(high):
        res.error(where, f"{key} low and high must be numbers")
        return
    if low < 0 or high < 0:
        res.error(where, f"{key} must not be negative")
    if low > high:
        res.error(where, f"{key} low {low} exceeds high {high}")


def _check_relative_path(res: ValidationResult, where: str, value: str, key: str) -> None:
    """Shape-only path check. Filesystem containment is dataset.py's job; this
    catches an absolute or traversing path before anything touches the disk."""
    if not value:
        return
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        res.error(where, f"{key} must be relative to the dataset directory, got {value!r}")
    if ".." in normalized.split("/"):
        res.error(where, f"{key} must not traverse upward, got {value!r}")


def _duplicates(values: Sequence[Any]) -> List[Any]:
    seen: Set[Any] = set()
    dupes: List[Any] = []
    for value in values:
        if value in seen and value not in dupes:
            dupes.append(value)
        seen.add(value)
    return dupes


# ---------------------------------------------------------------------------
# Listing metadata
# ---------------------------------------------------------------------------

def validate_listing(data: Any, *, listing_id: Optional[str] = None) -> ValidationResult:
    res = ValidationResult()
    if not isinstance(data, dict):
        res.error("listing", "must be a JSON object")
        return res

    if data.get("schema_version") != LISTING_SCHEMA_VERSION:
        res.error("listing", f"schema_version must be {LISTING_SCHEMA_VERSION}, "
                             f"got {data.get('schema_version')!r}")
    actual_id = _require_str(res, "listing", data, "listing_id", pattern=_ID_PATTERN)
    if listing_id and actual_id and actual_id != listing_id:
        res.error("listing", f"listing_id {actual_id!r} does not match manifest id {listing_id!r}")
    _require_str(res, "listing", data, "dataset_version")

    for key in ("asking_price", "sqft"):
        value = data.get(key)
        if not _is_number(value) or value <= 0:
            res.error("listing", f"{key} must be a positive number, got {value!r}")
    for key in ("beds", "baths"):
        value = data.get(key)
        if value is not None and (not _is_number(value) or value < 0):
            res.error("listing", f"{key} must be a non-negative number or null, got {value!r}")

    market = data.get("market_inputs")
    if not isinstance(market, dict):
        res.error("listing.market_inputs", "must be an object; frozen market inputs are "
                                          "required so cost scoring is reproducible")
    else:
        ppsf = market.get("area_ppsf")
        if not _is_number(ppsf) or ppsf <= 0:
            res.error("listing.market_inputs", f"area_ppsf must be a positive number, got {ppsf!r}")
        _require_str(res, "listing.market_inputs", market, "source")
        sample = market.get("sample_size")
        if sample is not None and (not isinstance(sample, int) or isinstance(sample, bool) or sample < 0):
            res.error("listing.market_inputs", f"sample_size must be a non-negative integer, got {sample!r}")

    res.extend(_validate_photos(data))
    return res


def _validate_photos(data: Dict[str, Any]) -> ValidationResult:
    res = ValidationResult()
    photos = _require_list(res, "listing", data, "photos")
    if not photos:
        res.error("listing.photos", "at least one photo is required")
        return res

    filenames: List[str] = []
    orders: List[Any] = []
    by_hash: Dict[str, List[str]] = {}

    for index, photo in enumerate(photos):
        where = f"listing.photos[{index}]"
        if not isinstance(photo, dict):
            res.error(where, "must be an object")
            continue
        filename = _require_str(res, where, photo, "filename")
        if filename:
            filenames.append(filename)
            normalized = filename.replace("\\", "/")
            if "/" in normalized or normalized in (".", ".."):
                res.error(where, f"filename must be a bare basename, got {filename!r}")
        order = photo.get("order")
        if not isinstance(order, int) or isinstance(order, bool):
            res.error(where, f"order must be an integer, got {order!r}")
        else:
            orders.append(order)
        digest = _str(photo.get("sha256")).lower()
        if not _SHA256_PATTERN.match(digest):
            res.error(where, f"sha256 must be 64 lowercase hex characters, got {photo.get('sha256')!r}")
        else:
            by_hash.setdefault(digest, []).append(filename)
        size = photo.get("byte_size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            res.error(where, f"byte_size must be a positive integer, got {size!r}")

    for dupe in _duplicates(filenames):
        res.error("listing.photos", f"duplicate filename {dupe!r}")

    if orders:
        expected = list(range(1, len(photos) + 1))
        if sorted(orders) != expected:
            res.error(
                "listing.photos",
                f"order must be contiguous 1..{len(photos)} with no gaps or duplicates; "
                f"got {sorted(orders)}",
            )

    # Byte-identical photos are usually an import bug (the same file copied
    # twice), but they are also the point of a duplicate-stress slice. Requiring
    # the intent to be declared keeps the accident loud and the deliberate case
    # legal.
    declared: Dict[str, str] = {}
    for photo in photos:
        if isinstance(photo, dict):
            target = _str(photo.get("intentional_duplicate_of"))
            if target:
                declared[_str(photo.get("filename"))] = target
    for digest, names in sorted(by_hash.items()):
        if len(names) < 2:
            continue
        undeclared = [n for n in names[1:] if n not in declared]
        if undeclared:
            res.error(
                "listing.photos",
                f"photos {names} share sha256 {digest[:12]}...; declare "
                f"intentional_duplicate_of on the copies or remove them",
            )
    for source, target in sorted(declared.items()):
        if target not in filenames:
            res.error("listing.photos", f"{source!r} declares intentional_duplicate_of "
                                        f"{target!r}, which is not a photo in this listing")
        elif target == source:
            res.error("listing.photos", f"{source!r} declares itself as its own duplicate source")
    return res


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def validate_reference(
    data: Any,
    *,
    listing: Optional[Dict[str, Any]] = None,
    vocabulary: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """Validate a compiled reference against its listing and frozen vocabulary.

    ``listing`` and ``vocabulary`` are optional so a draft can be shape-checked
    before a dataset exists, but cross-reference and vocabulary errors can only
    be found when they are supplied.
    """
    res = ValidationResult()
    if not isinstance(data, dict):
        res.error("reference", "must be a JSON object")
        return res

    if data.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        res.error("reference", f"schema_version must be {REFERENCE_SCHEMA_VERSION}, "
                               f"got {data.get('schema_version')!r}")
    _require_str(res, "reference", data, "listing_id", pattern=_ID_PATTERN)
    _require_str(res, "reference", data, "dataset_version")
    _require_choice(res, "reference", data, "tier", VALID_TIERS)
    review_status = _require_choice(res, "reference", data, "review_status", VALID_REVIEW_STATUSES)
    phase = _require_choice(res, "reference", data, "annotation_phase", VALID_ANNOTATION_PHASES)
    coverage = _require_choice(res, "reference", data, "reference_coverage", VALID_REFERENCE_COVERAGE)

    if listing is not None:
        listing_id = _str(data.get("listing_id"))
        expected = _str(listing.get("listing_id"))
        if listing_id and expected and listing_id != expected:
            res.error("reference", f"listing_id {listing_id!r} does not match listing {expected!r}")

    if vocabulary is not None:
        recorded = _str(data.get("vocabulary_fingerprint"))
        actual = _str(vocabulary.get("fingerprint"))
        if recorded and actual and recorded != actual:
            res.error(
                "reference",
                f"vocabulary_fingerprint {recorded[:12]}... does not match the dataset's "
                f"frozen vocabulary {actual[:12]}...; the reference was authored against a "
                f"different vocabulary and must be re-checked, not silently accepted",
            )

    _validate_coverage_domain(res, data, coverage)

    rooms, room_ids = _validate_rooms(res, data, vocabulary)
    photo_keys = _photo_keys(listing)
    _validate_photo_expectations(res, data, room_ids, photo_keys, vocabulary, listing is not None)
    component_keys = _validate_billable_components(res, data, room_ids)
    finding_ids = _validate_findings(res, data, room_ids, photo_keys, component_keys,
                                    vocabulary, listing is not None)
    _validate_components_backref(res, data, finding_ids)
    _validate_packages(res, data, room_ids, finding_ids, component_keys, vocabulary)
    _validate_scope_and_cost(res, data, vocabulary)

    _validate_phase_completeness(res, data, phase, review_status)
    return res


def _validate_coverage_domain(res: ValidationResult, data: Dict[str, Any],
                              coverage: Optional[str]) -> None:
    domain = data.get("coverage_domain")
    if coverage != "exhaustive":
        if domain is not None and not isinstance(domain, dict):
            res.error("reference.coverage_domain", "must be an object or null")
        return
    if not isinstance(domain, dict):
        res.error("reference.coverage_domain",
                  "reference_coverage 'exhaustive' requires a coverage_domain; "
                  "'exhaustive' is meaningless without saying exhaustive over what")
        return
    for key in COVERAGE_DOMAIN_KEYS:
        if not isinstance(domain.get(key), bool):
            res.error("reference.coverage_domain", f"{key} must be a boolean")
    for key in sorted(set(domain) - set(COVERAGE_DOMAIN_KEYS)):
        res.error("reference.coverage_domain", f"unknown key {key!r}")
    if domain.get("hidden_system_conditions") is True:
        res.warn("reference.coverage_domain",
                 "hidden_system_conditions is true, but photographs cannot establish hidden "
                 "electrical, plumbing, foundation, or HVAC condition; those findings belong "
                 "to presence 'indeterminate'")


def _validate_rooms(res: ValidationResult, data: Dict[str, Any],
                    vocabulary: Optional[Dict[str, Any]]) -> Tuple[List[Any], Set[str]]:
    rooms = _require_list(res, "reference", data, "rooms")
    scene_ids = set((vocabulary or {}).get("scene_ids") or ())
    room_ids: Set[str] = set()
    for index, room in enumerate(rooms):
        where = f"reference.rooms[{index}]"
        if not isinstance(room, dict):
            res.error(where, "must be an object")
            continue
        room_id = _require_str(res, where, room, "room_id", pattern=_ID_PATTERN)
        if room_id:
            if room_id in room_ids:
                res.error(where, f"duplicate room_id {room_id!r}")
            room_ids.add(room_id)
        room_type = _require_str(res, where, room, "room_type")
        if room_type and scene_ids:
            if room_type in NON_ROOM_SCENE_IDS:
                res.error(where, f"room_type {room_type!r} is not a physical room")
            elif room_type not in scene_ids:
                res.error(where, f"room_type {room_type!r} is not a scene id in the frozen "
                                 f"vocabulary")
    return rooms, room_ids


def _photo_keys(listing: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """Photo keys in frozen manifest order, or None when no listing was given."""
    if listing is None:
        return None
    photos = listing.get("photos") or []
    ordered = sorted(
        (p for p in photos if isinstance(p, dict) and _str(p.get("filename"))),
        key=lambda p: p.get("order") if isinstance(p.get("order"), int) else 0,
    )
    return [_str(p.get("filename")) for p in ordered]


def _validate_photo_expectations(res: ValidationResult, data: Dict[str, Any],
                                 room_ids: Set[str], photo_keys: Optional[List[str]],
                                 vocabulary: Optional[Dict[str, Any]],
                                 have_listing: bool) -> None:
    entries = _require_list(res, "reference", data, "photo_expectations")
    scene_to_group: Dict[str, str] = (vocabulary or {}).get("scene_ids") or {}
    ui_groups = set((vocabulary or {}).get("ui_scene_groups") or ())

    seen: List[str] = []
    for index, entry in enumerate(entries):
        where = f"reference.photo_expectations[{index}]"
        if not isinstance(entry, dict):
            res.error(where, "must be an object")
            continue
        photo_key = _require_str(res, where, entry, "photo_key")
        if photo_key:
            seen.append(photo_key)
            if photo_keys is not None and photo_key not in photo_keys:
                res.error(where, f"photo_key {photo_key!r} is not a photo in this listing")

        indeterminate = _require_bool(res, where, entry, "indeterminate", default=False)

        visible = _require_list(res, where, entry, "visible_room_ids", allow_missing=True)
        visible_ids = [_str(v) for v in visible]
        for room_id in visible_ids:
            if room_id and room_ids and room_id not in room_ids:
                res.error(where, f"visible_room_ids entry {room_id!r} is not a declared room")
        for dupe in _duplicates(visible_ids):
            res.error(where, f"duplicate visible_room_ids entry {dupe!r}")

        primary = _str(entry.get("primary_room_id"))
        if primary:
            if room_ids and primary not in room_ids:
                res.error(where, f"primary_room_id {primary!r} is not a declared room")
            # Unconditional: the compiler fills this, but a hand-edited
            # reference.json must not be able to name a primary room that the
            # photo does not show.
            if primary not in visible_ids:
                res.error(where, f"primary_room_id {primary!r} must also appear in "
                                 f"visible_room_ids")
        elif not indeterminate:
            res.error(where, "primary_room_id is required unless indeterminate is true")

        accepted = [_str(s) for s in _require_list(res, where, entry, "accepted_scene_ids",
                                                   allow_missing=True)]
        expected_group = _str(entry.get("expected_group"))
        if indeterminate:
            continue
        if not accepted:
            res.error(where, "accepted_scene_ids is required unless indeterminate is true")
        for scene in accepted:
            if scene and scene_to_group and scene not in scene_to_group:
                res.error(where, f"accepted_scene_ids entry {scene!r} is not a scene id in the "
                                 f"frozen vocabulary")
        for dupe in _duplicates(accepted):
            res.error(where, f"duplicate accepted_scene_ids entry {dupe!r}")

        if not expected_group:
            res.error(where, "expected_group is required unless indeterminate is true")
        elif ui_groups and expected_group not in ui_groups:
            res.error(where, f"expected_group {expected_group!r} is not a UI scene group")
        elif scene_to_group and accepted:
            groups = {scene_to_group[s] for s in accepted if s in scene_to_group}
            if groups and expected_group not in groups:
                res.error(where, f"expected_group {expected_group!r} is not the group of any "
                                 f"accepted scene id (groups: {sorted(groups)})")
            if len(groups) > 1:
                # Legitimate for open-plan and multi-space photos; the metric
                # credits a prediction matching any accepted id's group.
                res.warn(where, f"accepted_scene_ids span multiple groups {sorted(groups)}; "
                                f"group accuracy will credit any of them")

    for dupe in _duplicates(seen):
        res.error("reference.photo_expectations", f"duplicate entry for photo {dupe!r}")

    if photo_keys is not None and have_listing:
        missing = [k for k in photo_keys if k not in set(seen)]
        if missing:
            res.error(
                "reference.photo_expectations",
                f"every photo needs exactly one entry; missing {len(missing)}: "
                f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}",
            )


def _validate_billable_components(res: ValidationResult, data: Dict[str, Any],
                                  room_ids: Set[str]) -> Set[str]:
    components = _require_list(res, "reference", data, "billable_components", allow_missing=True)
    keys: Set[str] = set()
    for index, component in enumerate(components):
        where = f"reference.billable_components[{index}]"
        if not isinstance(component, dict):
            res.error(where, "must be an object")
            continue
        key = _require_str(res, where, component, "key")
        if key:
            if key in keys:
                res.error(where, f"duplicate component key {key!r}")
            keys.add(key)
        room_id = _str(component.get("room_id"))
        if room_id and room_ids and room_id not in room_ids:
            res.error(where, f"room_id {room_id!r} is not a declared room")
        _require_str(res, where, component, "label")

        units = component.get("expected_units")
        if not isinstance(units, int) or isinstance(units, bool) or units < 1:
            res.error(where, f"expected_units must be an integer >= 1, got {units!r}; it is "
                             f"what detects repeated-evidence cost inflation")
        if component.get("expected_cost_band") is not None:
            _require_money_range(res, where, component.get("expected_cost_band"),
                                 "expected_cost_band")
    return keys


def _validate_findings(res: ValidationResult, data: Dict[str, Any], room_ids: Set[str],
                       photo_keys: Optional[List[str]], component_keys: Set[str],
                       vocabulary: Optional[Dict[str, Any]], have_listing: bool) -> Set[str]:
    findings = _require_list(res, "reference", data, "findings")
    catalog_items = set((vocabulary or {}).get("catalog_items") or ())
    ids: Set[str] = set()

    for index, finding in enumerate(findings):
        where = f"reference.findings[{index}]"
        if not isinstance(finding, dict):
            res.error(where, "must be an object")
            continue

        finding_id = _require_str(res, where, finding, "id", pattern=_ID_PATTERN)
        if finding_id:
            if finding_id in ids:
                res.error(where, f"duplicate finding id {finding_id!r}")
            ids.add(finding_id)
        _require_str(res, where, finding, "description")

        presence = _require_choice(res, where, finding, "presence", VALID_PRESENCE)
        expectation = _require_choice(res, where, finding, "reporting_expectation",
                                      VALID_REPORTING_EXPECTATIONS)
        sufficiency = _require_choice(res, where, finding, "visual_sufficiency",
                                      VALID_VISUAL_SUFFICIENCY, default="sufficient")
        catalog_status = _require_choice(res, where, finding, "catalog_status",
                                         VALID_CATALOG_STATUSES)
        critical = _require_bool(res, where, finding, "critical", default=False)

        room_id = _str(finding.get("room_id"))
        if not room_id:
            res.error(where, "room_id is required")
        elif room_ids and room_id not in room_ids:
            res.error(where, f"room_id {room_id!r} is not a declared room")

        item_ids = [_str(i) for i in _require_list(res, where, finding, "catalog_item_ids",
                                                   allow_missing=True)]
        if catalog_status == "missing_catalog_item":
            if item_ids:
                res.error(where, "catalog_item_ids must be empty when catalog_status is "
                                 "'missing_catalog_item'")
        elif catalog_status is not None:
            if not item_ids:
                res.error(where, f"catalog_item_ids is required when catalog_status is "
                                 f"{catalog_status!r}")
            for item_id in item_ids:
                if item_id and catalog_items and item_id not in catalog_items:
                    res.error(where, f"catalog_item_ids entry {item_id!r} is not in the frozen "
                                     f"vocabulary")
            for dupe in _duplicates(item_ids):
                res.error(where, f"duplicate catalog_item_ids entry {dupe!r}")

        # actionability is derived from the catalog item when one matched, so it
        # is only hand-authored where the catalog has nothing to say.
        actionability = finding.get("actionability")
        if catalog_status == "missing_catalog_item":
            if actionability is None:
                res.error(where, "actionability is required when catalog_status is "
                                 "'missing_catalog_item' (it cannot be derived without a "
                                 "catalog item)")
            else:
                _require_choice(res, where, finding, "actionability", VALID_ACTIONABILITY)
        elif actionability is not None:
            _require_choice(res, where, finding, "actionability", VALID_ACTIONABILITY)

        aliases = _require_list(res, where, finding, "aliases", allow_missing=True)
        for alias_index, alias in enumerate(aliases):
            if not _str(alias):
                res.error(where, f"aliases[{alias_index}] must be a non-empty string")

        component_key = _str(finding.get("billable_component_key"))
        if component_key and component_keys and component_key not in component_keys:
            res.error(where, f"billable_component_key {component_key!r} is not a declared "
                             f"billable component")

        evidence_photos = _validate_evidence(res, where, finding, photo_keys, have_listing)
        if presence == "present" and not evidence_photos:
            res.error(where, "presence 'present' requires at least one evidence occurrence")

        _validate_finding_combination(res, where, presence, expectation, sufficiency, critical)

    return ids


def _validate_evidence(res: ValidationResult, where: str, finding: Dict[str, Any],
                       photo_keys: Optional[List[str]], have_listing: bool) -> List[str]:
    evidence = _require_list(res, where, finding, "evidence", allow_missing=True)
    keys: List[str] = []
    for index, occurrence in enumerate(evidence):
        sub = f"{where}.evidence[{index}]"
        if not isinstance(occurrence, dict):
            res.error(sub, "must be an object with a photo_key")
            continue
        photo_key = _require_str(res, sub, occurrence, "photo_key")
        if not photo_key:
            continue
        keys.append(photo_key)
        if photo_keys is not None and have_listing and photo_key not in photo_keys:
            res.error(sub, f"photo_key {photo_key!r} is not a photo in this listing")
        note = occurrence.get("note")
        if note is not None and not isinstance(note, str):
            res.error(sub, "note must be a string when present")
    for dupe in _duplicates(keys):
        res.error(where, f"duplicate evidence photo_key {dupe!r}; one occurrence per photo")
    return keys


def _validate_finding_combination(res: ValidationResult, where: str, presence: Optional[str],
                                  expectation: Optional[str], sufficiency: Optional[str],
                                  critical: Optional[bool]) -> None:
    """Reject contradictory field combinations so reporting_category stays disjoint.

    Note what is deliberately *allowed*: ``present`` + ``must_not_report`` records
    a real, visible, but trivial condition that a good analyzer should stay quiet
    about — the mechanism for scoring noise. And ``indeterminate`` + ``required``
    records a condition photos cannot settle that should still be flagged for
    inspection, which is a core product case.
    """
    if presence == "absent" and expectation not in (None, "must_not_report"):
        res.error(where, f"presence 'absent' requires reporting_expectation 'must_not_report', "
                         f"got {expectation!r}")
    if presence == "indeterminate" and expectation == "must_not_report":
        res.error(where, "presence 'indeterminate' cannot be 'must_not_report': you cannot "
                         "assert a model must stay silent about something you could not "
                         "determine. Use 'acceptable'.")
    # Subsumes "present permits sufficient or limited only": presence 'present'
    # with 'insufficient' is caught here, so a separate check would only ever
    # double-report the same violation.
    if sufficiency == "insufficient" and presence not in (None, "indeterminate"):
        res.error(where, f"visual_sufficiency 'insufficient' requires presence 'indeterminate', "
                         f"got {presence!r}. A condition you can see but cannot size is "
                         f"'present' + 'limited'.")
    if critical:
        if presence != "present" or expectation != "required" or sufficiency != "sufficient":
            res.error(
                where,
                "critical requires presence 'present', reporting_expectation 'required', and "
                f"visual_sufficiency 'sufficient'; got {presence!r}/{expectation!r}/"
                f"{sufficiency!r}. Critical findings are a hard release gate, so they must be "
                f"unambiguous.",
            )


def _validate_components_backref(res: ValidationResult, data: Dict[str, Any],
                                 finding_ids: Set[str]) -> None:
    """The finding -> component link must be symmetric in both directions, or a
    component's expected_units would be compared against the wrong evidence."""
    components = data.get("billable_components") or []
    findings = data.get("findings") or []
    by_finding: Dict[str, str] = {
        _str(f.get("id")): _str(f.get("billable_component_key"))
        for f in findings if isinstance(f, dict) and _str(f.get("id"))
    }
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            continue
        where = f"reference.billable_components[{index}]"
        key = _str(component.get("key"))
        listed = [_str(f) for f in (component.get("finding_ids") or [])]
        if not listed:
            res.error(where, "finding_ids must list at least one finding")
        for finding_id in listed:
            if finding_id and finding_ids and finding_id not in finding_ids:
                res.error(where, f"finding_ids entry {finding_id!r} is not a declared finding")
            elif by_finding.get(finding_id, "") != key:
                res.error(where, f"finding {finding_id!r} does not point back at this component "
                                 f"({key!r}); the link must be symmetric")
        for dupe in _duplicates(listed):
            res.error(where, f"duplicate finding_ids entry {dupe!r}")

    declared_keys = {_str(c.get("key")) for c in components if isinstance(c, dict)}
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        key = _str(finding.get("billable_component_key"))
        finding_id = _str(finding.get("id"))
        if not key or key not in declared_keys:
            continue
        component = next((c for c in components
                          if isinstance(c, dict) and _str(c.get("key")) == key), None)
        if component is not None:
            listed = [_str(f) for f in (component.get("finding_ids") or [])]
            if finding_id not in listed:
                res.error(f"reference.findings[{finding_id}]",
                          f"points at component {key!r}, which does not list it in finding_ids")


def _validate_packages(res: ValidationResult, data: Dict[str, Any], room_ids: Set[str],
                       finding_ids: Set[str], component_keys: Set[str],
                       vocabulary: Optional[Dict[str, Any]]) -> None:
    packages = _require_list(res, "reference", data, "packages", allow_missing=True)
    valid_types = set((vocabulary or {}).get("package_types") or ())
    seen: Set[str] = set()

    for index, package in enumerate(packages):
        where = f"reference.packages[{index}]"
        if not isinstance(package, dict):
            res.error(where, "must be an object")
            continue
        package_id = _require_str(res, where, package, "package_id", pattern=_ID_PATTERN)
        if package_id:
            if package_id in seen:
                res.error(where, f"duplicate package_id {package_id!r}")
            seen.add(package_id)

        package_type = _require_str(res, where, package, "package_type")
        if package_type and valid_types and package_type not in valid_types:
            res.error(where, f"package_type {package_type!r} is not in the frozen vocabulary")

        decision = _require_choice(res, where, package, "decision", VALID_PACKAGE_DECISIONS)
        room_id = _str(package.get("room_id"))
        if room_id and room_ids and room_id not in room_ids:
            res.error(where, f"room_id {room_id!r} is not a declared room")

        confirmed = [_str(f) for f in _require_list(res, where, package, "confirmed_finding_ids",
                                                     allow_missing=True)]
        for finding_id in confirmed:
            if finding_id and finding_ids and finding_id not in finding_ids:
                res.error(where, f"confirmed_finding_ids entry {finding_id!r} is not a declared "
                                 f"finding")
        for dupe in _duplicates(confirmed):
            res.error(where, f"duplicate confirmed_finding_ids entry {dupe!r}")

        keys = [_str(k) for k in _require_list(res, where, package, "component_keys",
                                                allow_missing=True)]
        for key in keys:
            if key and component_keys and key not in component_keys:
                res.error(where, f"component_keys entry {key!r} is not a declared component")
        for dupe in _duplicates(keys):
            res.error(where, f"duplicate component_keys entry {dupe!r}")

        # A rejection without a stated reason is unauditable: it is exactly the
        # judgment L4 scores a model against.
        if decision in ("unsupported", "indeterminate") and not _str(package.get("reason")):
            res.error(where, f"decision {decision!r} requires a reason explaining why")
        if decision == "expected" and not confirmed:
            res.error(where, "decision 'expected' requires at least one confirmed_finding_ids "
                             "entry")


def _validate_scope_and_cost(res: ValidationResult, data: Dict[str, Any],
                             vocabulary: Optional[Dict[str, Any]]) -> None:
    scope = data.get("scope")
    bands = set((vocabulary or {}).get("rehab_scope_bands") or ())
    if scope is not None:
        if not isinstance(scope, dict):
            res.error("reference.scope", "must be an object or null")
        else:
            band = _str(scope.get("scope_band"))
            if not band:
                res.error("reference.scope", "scope_band is required")
            elif bands and band not in bands:
                res.error("reference.scope", f"scope_band {band!r} is not in the frozen "
                                             f"vocabulary {sorted(bands)}")
            _require_choice(res, "reference.scope", scope, "band_confidence", VALID_BAND_CONFIDENCE)

    cost = data.get("cost")
    if cost is None:
        return
    if not isinstance(cost, dict):
        res.error("reference.cost", "must be an object or null")
        return
    _require_money_range(res, "reference.cost", cost.get("expected_band"), "expected_band")
    _require_choice(res, "reference.cost", cost, "confidence", VALID_BAND_CONFIDENCE)
    if not _str(cost.get("basis")):
        res.error("reference.cost", "basis is required; it records what scope the band prices")
    # Market inputs are frozen in metadata.json and must not be restated here:
    # two copies would eventually disagree and silently change cost scoring.
    for forbidden in ("market_assumptions", "market_assumption_override", "market_inputs"):
        if forbidden in cost:
            res.error("reference.cost", f"{forbidden} must not appear in the reference; market "
                                        f"inputs live only in metadata.json. If a frozen input is "
                                        f"wrong, fix the metadata and bump the dataset version.")


def _validate_phase_completeness(res: ValidationResult, data: Dict[str, Any],
                                 phase: Optional[int], review_status: Optional[str]) -> None:
    """Phase-aware completeness. A phase-1 draft is a legitimate artifact; a
    reviewed reference missing package or cost truth is not."""
    has_packages = bool(data.get("packages"))
    has_components = bool(data.get("billable_components"))
    has_cost = isinstance(data.get("cost"), dict)
    has_scope = isinstance(data.get("scope"), dict)

    if phase is not None and phase >= 2:
        if not has_packages:
            res.error("reference", f"annotation_phase {phase} requires packages (phase 2 is "
                                   f"blind package judgment)")
        if not has_components:
            res.error("reference", f"annotation_phase {phase} requires billable_components")
    if phase is not None and phase >= 4:
        if not has_cost:
            res.error("reference", f"annotation_phase {phase} requires cost (phase 4 is cost "
                                   f"review)")
        if not has_scope:
            res.error("reference", f"annotation_phase {phase} requires scope")

    if review_status in ("reviewed", "sealed"):
        missing = [name for name, present in (
            ("packages", has_packages), ("billable_components", has_components),
            ("scope", has_scope), ("cost", has_cost),
        ) if not present]
        if missing:
            res.error("reference", f"review_status {review_status!r} requires all annotation "
                                   f"phases complete; missing {', '.join(missing)}")
        if not data.get("findings"):
            res.error("reference", f"review_status {review_status!r} requires at least one finding")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def validate_manifest(data: Any) -> ValidationResult:
    res = ValidationResult()
    if not isinstance(data, dict):
        res.error("manifest", "must be a JSON object")
        return res

    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        res.error("manifest", f"schema_version must be {MANIFEST_SCHEMA_VERSION}, "
                              f"got {data.get('schema_version')!r}")
    _require_str(res, "manifest", data, "dataset_id", pattern=_ID_PATTERN)
    _require_str(res, "manifest", data, "dataset_version")
    state = _require_choice(res, "manifest", data, "state", ("draft", "sealed"))

    # A draft dataset legitimately has no listings yet: `import` is the command
    # that adds the first one, so requiring one here would make the dataset
    # unloadable before it could ever be populated. Sealing is what makes it
    # mandatory.
    listings = _require_list(res, "manifest", data, "listings")

    ids: Set[str] = set()
    for index, listing in enumerate(listings):
        where = f"manifest.listings[{index}]"
        if not isinstance(listing, dict):
            res.error(where, "must be an object")
            continue
        listing_id = _require_str(res, where, listing, "id", pattern=_ID_PATTERN)
        if listing_id:
            if listing_id in ids:
                res.error(where, f"duplicate listing id {listing_id!r}")
            ids.add(listing_id)
        _require_choice(res, where, listing, "tier", VALID_TIERS)

        for key in ("metadata_path", "reference_path"):
            value = _require_str(res, where, listing, key)
            _check_relative_path(res, where, value, key)

        slices = _require_list(res, where, listing, "slices", allow_missing=True)
        slice_names = [_str(s) for s in slices]
        for slice_index, name in enumerate(slice_names):
            if not name:
                res.error(where, f"slices[{slice_index}] must be a non-empty string")
            elif not _ID_PATTERN.match(name):
                res.error(where, f"slice {name!r} must match {_ID_PATTERN.pattern}")
        for dupe in _duplicates(slice_names):
            res.error(where, f"duplicate slice {dupe!r}")

    if state == "sealed":
        if not listings:
            res.error("manifest.listings", "a sealed dataset requires at least one listing")
        for key in ("dataset_fingerprint", "vocabulary_fingerprint"):
            if not _str(data.get(key)):
                res.error("manifest", f"a sealed dataset requires {key}")
    return res


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def validate_config(data: Any) -> ValidationResult:
    res = ValidationResult()
    if not isinstance(data, dict):
        res.error("config", "must be a JSON object")
        return res

    if data.get("schema_version") != CONFIG_SCHEMA_VERSION:
        res.error("config", f"schema_version must be {CONFIG_SCHEMA_VERSION}, "
                            f"got {data.get('schema_version')!r}")
    _require_str(res, "config", data, "name")

    axes = data.get("axes")
    if not isinstance(axes, dict) or not axes:
        res.error("config.axes", "must be a non-empty object; declaring the axes is what lets a "
                                 "comparison refuse to attribute causality across two changes")
    else:
        for key, value in sorted(axes.items()):
            if not _str(key) or not _str(value):
                res.error("config.axes", f"axis {key!r} must map a non-empty name to a "
                                         f"non-empty value")

    pipeline = data.get("pipeline")
    if not isinstance(pipeline, dict):
        res.error("config.pipeline", "must be an object")
    else:
        toggles = pipeline.get("pass_toggles")
        if not isinstance(toggles, dict) or not toggles:
            res.error("config.pipeline.pass_toggles", "must be a non-empty object of "
                                                      "{pass: bool}")
        else:
            for key in sorted(toggles):
                if key not in ALL_PASSES:
                    res.error("config.pipeline.pass_toggles",
                              f"unknown pass {key!r}; expected one of {list(ALL_PASSES)}")
                elif not isinstance(toggles[key], bool):
                    res.error("config.pipeline.pass_toggles",
                              f"{key} must be a boolean, got {toggles[key]!r}")
        concurrency = pipeline.get("concurrency")
        if concurrency is not None and (not isinstance(concurrency, int)
                                        or isinstance(concurrency, bool) or concurrency < 1):
            res.error("config.pipeline", f"concurrency must be a positive integer, "
                                         f"got {concurrency!r}")

    _validate_model_map(res, data)

    repetitions = data.get("repetitions", 1)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        res.error("config", f"repetitions must be an integer >= 1, got {repetitions!r}")

    thresholds = data.get("regression_thresholds")
    if thresholds is not None:
        if not isinstance(thresholds, dict):
            res.error("config.regression_thresholds", "must be an object")
        else:
            for key, value in sorted(thresholds.items()):
                if not _is_number(value) or value < 0:
                    res.error("config.regression_thresholds",
                              f"{key} must be a non-negative number, got {value!r}")

    evaluation = data.get("evaluation")
    if evaluation is not None and not isinstance(evaluation, dict):
        res.error("config.evaluation", "must be an object")
    elif isinstance(evaluation, dict):
        threshold = evaluation.get("semantic_threshold")
        if threshold is not None and (not _is_number(threshold) or not 0 < threshold <= 1):
            res.error("config.evaluation", f"semantic_threshold must be in (0, 1], "
                                           f"got {threshold!r}")
        if "gate_on_judge" in evaluation and not isinstance(evaluation["gate_on_judge"], bool):
            res.error("config.evaluation", "gate_on_judge must be a boolean")
    return res


def _validate_model_map(res: ValidationResult, data: Dict[str, Any]) -> None:
    model_map = data.get("model_map")
    if not isinstance(model_map, dict) or not model_map:
        res.error("config.model_map", "must be a non-empty object of {pass: model spec}")
        return

    toggles = (data.get("pipeline") or {}).get("pass_toggles") or {}
    for key in sorted(model_map):
        where = f"config.model_map.{key}"
        if key in _STUB_PASSES:
            res.error(where, f"pass {key} is {_STUB_PASSES[key]}, so a model entry would be "
                             f"silently ignored at runtime; remove it")
            continue
        if key not in BENCHMARK_MODEL_MAP_PASSES:
            res.error(where, f"unknown pass {key!r}; routable passes are "
                             f"{list(BENCHMARK_MODEL_MAP_PASSES)}")
            continue
        spec = model_map[key]
        if not isinstance(spec, dict):
            res.error(where, "must be an object with provider and model")
            continue
        _require_str(res, where, spec, "provider")
        _require_str(res, where, spec, "model")
        for optional in ("reasoning_effort", "verbosity"):
            if optional in spec and not _str(spec.get(optional)):
                res.error(where, f"{optional} must be a non-empty string when present")

    # An enabled routable pass with no model entry would run on whatever the
    # profile defaults to, which is precisely the unverified routing the
    # benchmark is built to prevent.
    for key in BENCHMARK_MODEL_MAP_PASSES:
        if toggles.get(key, True) and key not in model_map:
            res.error("config.model_map", f"pass {key} is enabled but has no model entry; "
                                          f"routing would fall back to the profile default")
