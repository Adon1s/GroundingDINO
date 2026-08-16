"""Startup-time runtime for the renovation architecture.

The selector (pipeline_config.RENOVATION_ARCHITECTURE_MODE) is resolved at
import; this module holds the heavier per-process state that needs the loaded
catalog. Entry points call initialize_renovation_architecture() after catalog
load and before models load / worker ready, so an invalid catalog — or a
shadow mode with no Terra model — fails startup instead of a listing.

The shadow writer seam is defensive about an uninitialized runtime (an entry
point that never wired init, e.g. audit_runner): it receives a valid private
`failed` envelope rather than an exception. Since Session 2 the initialized
shadow path runs the full condition review; a typed operational failure
inside it becomes a `failed` envelope whose reason is the failure-taxonomy
category, and never a semantic verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    ENVELOPE_SCHEMA_VERSION,
    EstimateProvenance,
    POLICY_VERSIONS,
    PROJECTION_VERSION,
    RenovationEstimateEnvelope,
    REQUIRED_KIND_ONTOLOGY_SELECTOR,
)
from tools.renovation_architecture.ids import make_estimate_id


class RenovationArchitectureInitError(RuntimeError):
    """Startup-time failure: the selected mode cannot run on this catalog."""


@dataclass(frozen=True)
class RenovationArchitectureRuntime:
    mode: str
    projection: Mapping[str, Any]
    # The raw newest catalog: the Session 4 candidate builder feeds it to the
    # legacy inference primitives (package_affinity blocks, base costs). The
    # projection stays the versioned contract; both come from the same file,
    # pinned by catalog_sha256.
    catalog: Mapping[str, Any]
    catalog_sha256: str
    projection_fingerprint: str
    kind_ontology_selector: str
    terra_model: str
    terra_max_output_tokens: int
    sol_model: str
    sol_max_output_tokens: int


_RUNTIME: Optional[RenovationArchitectureRuntime] = None


def initialize_renovation_architecture(
    *,
    mode: str,
    catalog: Mapping[str, Any],
    catalog_path: Path,
    kind_ontology_version: str,
    terra_model: str = "",
    terra_max_output_tokens: int = 8192,
    sol_model: str = "",
    sol_max_output_tokens: int = 8192,
) -> None:
    """Build the immutable per-process runtime. No-op in current mode."""
    global _RUNTIME
    if mode == "current":
        _RUNTIME = None
        return
    if mode != "shadow":
        raise RenovationArchitectureInitError(
            f"RENOVATION_ARCHITECTURE_MODE={mode!r} cannot initialize — only "
            "'current' and 'shadow' are runnable before Session 6"
        )
    if kind_ontology_version != REQUIRED_KIND_ONTOLOGY_SELECTOR:
        raise RenovationArchitectureInitError(
            f"shadow mode requires KIND_ONTOLOGY_VERSION="
            f"{REQUIRED_KIND_ONTOLOGY_SELECTOR}, got {kind_ontology_version!r}"
        )
    if not (terra_model or "").strip():
        raise RenovationArchitectureInitError(
            "shadow mode requires a Terra model: set RENOVATION_TERRA_MODEL "
            "or OPENAI_MODEL"
        )
    if not isinstance(terra_max_output_tokens, int) or terra_max_output_tokens <= 0:
        raise RenovationArchitectureInitError(
            "RENOVATION_TERRA_MAX_OUTPUT_TOKENS must be a positive integer, "
            f"got {terra_max_output_tokens!r}"
        )
    if not (sol_model or "").strip():
        raise RenovationArchitectureInitError(
            "shadow mode requires a Sol model: set RENOVATION_SOL_MODEL "
            "or OPENAI_MODEL"
        )
    if not isinstance(sol_max_output_tokens, int) or sol_max_output_tokens <= 0:
        raise RenovationArchitectureInitError(
            "RENOVATION_SOL_MAX_OUTPUT_TOKENS must be a positive integer, "
            f"got {sol_max_output_tokens!r}"
        )
    # Imported here so current-mode startup never pays for the projection's
    # dependency chain (catalog validation, rehab_packages).
    from tools.renovation_architecture.catalog_projection import (
        build_renovation_catalog_projection,
    )

    try:
        projection = build_renovation_catalog_projection(
            catalog, catalog_path=Path(catalog_path)
        )
    except Exception as exc:
        raise RenovationArchitectureInitError(
            f"catalog projection build failed: {exc}"
        ) from exc
    _RUNTIME = RenovationArchitectureRuntime(
        mode=mode,
        projection=projection,
        catalog=catalog,
        catalog_sha256=projection["catalog_sha256"],
        projection_fingerprint=projection["fingerprint"],
        kind_ontology_selector=kind_ontology_version,
        terra_model=terra_model.strip(),
        terra_max_output_tokens=terra_max_output_tokens,
        sol_model=sol_model.strip(),
        sol_max_output_tokens=sol_max_output_tokens,
    )


def get_runtime() -> Optional[RenovationArchitectureRuntime]:
    return _RUNTIME


def reset_runtime_for_tests() -> None:
    global _RUNTIME
    _RUNTIME = None


def _make_provenance(
    runtime: Optional[RenovationArchitectureRuntime],
    *,
    property_key: str,
    run_id: str,
    created_at: str,
    source_artifact: str,
) -> EstimateProvenance:
    if runtime is None:
        return EstimateProvenance(
            schema_version=CONTRACTS_SCHEMA_VERSION,
            architecture_mode="shadow",
            contracts_schema_version=CONTRACTS_SCHEMA_VERSION,
            projection_version=PROJECTION_VERSION,
            catalog_version=None,
            catalog_ontology_version=None,
            catalog_sha256=None,
            projection_fingerprint=None,
            kind_ontology_selector=None,
            policy_versions=dict(POLICY_VERSIONS),
            property_key=property_key,
            source_run_id=run_id,
            source_artifact=source_artifact,
            created_at=created_at,
        )
    return EstimateProvenance(
        schema_version=CONTRACTS_SCHEMA_VERSION,
        architecture_mode=runtime.mode,
        contracts_schema_version=CONTRACTS_SCHEMA_VERSION,
        projection_version=PROJECTION_VERSION,
        catalog_version=runtime.projection["catalog_version"],
        catalog_ontology_version=runtime.projection["catalog_ontology_version"],
        catalog_sha256=runtime.catalog_sha256,
        projection_fingerprint=runtime.projection_fingerprint,
        kind_ontology_selector=runtime.kind_ontology_selector,
        policy_versions=dict(POLICY_VERSIONS),
        property_key=property_key,
        source_run_id=run_id,
        source_artifact=source_artifact,
        created_at=created_at,
    )


def build_shadow_envelope(
    *,
    property_key: str,
    run_id: str,
    created_at: str,
    source_artifact: str,
    issues_flat: Optional[List[Dict[str, Any]]] = None,
    photos: Optional[Mapping[str, Any]] = None,
    property_metadata: Optional[Dict[str, Any]] = None,
    photo_key_to_path: Optional[Mapping[str, Path]] = None,
    vlm_client: Any = None,
    api_key: str = "",
    artifacts_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """The shadow output: a package_review_complete envelope (Session 2
    condition review + Session 3 work derivation + Session 4 deterministic
    candidates and bounded Sol review), or a valid failed envelope
    (uninitialized runtime, or a typed operational failure mapped to its
    failure-taxonomy category). Completed Terra checkpoints and any parsed
    Sol checkpoint survive a downstream failure — the deterministic stages
    are recomputed on retry, and the model calls replay from checkpoints
    without new spend."""
    runtime = _RUNTIME
    if runtime is None:
        envelope = RenovationEstimateEnvelope(
            schema_version=ENVELOPE_SCHEMA_VERSION,
            estimate_id=make_estimate_id(
                property_key=property_key,
                source_run_id=run_id,
                catalog_sha256="",
                projection_fingerprint="",
            ),
            state="failed",
            reason="runtime_not_initialized",
            error_detail=None,
            provenance=_make_provenance(
                None,
                property_key=property_key,
                run_id=run_id,
                created_at=created_at,
                source_artifact=source_artifact,
            ),
            result=None,
        )
        return envelope.to_dict()

    estimate_id = make_estimate_id(
        property_key=property_key,
        source_run_id=run_id,
        catalog_sha256=runtime.catalog_sha256,
        projection_fingerprint=runtime.projection_fingerprint,
    )
    provenance = _make_provenance(
        runtime,
        property_key=property_key,
        run_id=run_id,
        created_at=created_at,
        source_artifact=source_artifact,
    )
    # Imported here so current mode never pays for the review pipeline chain.
    from tools.renovation_architecture.package_candidates import (
        build_package_candidates,
    )
    from tools.renovation_architecture.review_pipeline import run_condition_review
    from tools.renovation_architecture.sol_review import run_package_review
    from tools.renovation_architecture.work_items import derive_standalone_estimate

    try:
        review_result = run_condition_review(
            runtime=runtime,
            estimate_id=estimate_id,
            property_key=property_key,
            source_run_id=run_id,
            created_at=created_at,
            issues_flat=issues_flat or [],
            photos=photos or {},
            property_metadata=property_metadata,
            photo_key_to_path=photo_key_to_path or {},
            vlm_client=vlm_client,
            api_key=api_key,
            artifacts_root=Path(artifacts_root) if artifacts_root else None,
        )
        standalone_result = derive_standalone_estimate(
            review_result=review_result,
            projection=runtime.projection,
            property_metadata=property_metadata,
            estimate_id=estimate_id,
        )
        candidates = build_package_candidates(
            standalone_result=standalone_result,
            projection=runtime.projection,
            catalog=runtime.catalog,
            estimate_id=estimate_id,
        )
        result = run_package_review(
            runtime=runtime,
            estimate_id=estimate_id,
            property_key=property_key,
            source_run_id=run_id,
            created_at=created_at,
            standalone_result=standalone_result,
            package_candidates=candidates,
            vlm_client=vlm_client,
            api_key=api_key,
            artifacts_root=Path(artifacts_root) if artifacts_root else None,
        )
    except Exception as exc:
        from tools.failure_taxonomy import classify_failure

        # PassExecutionError carries its own pass/stage/provider/model, which
        # win inside classify_failure — the kwargs only cover raw exceptions.
        descriptor = classify_failure(
            exc, pass_key="terra_review", stage="request",
            provider="openai", model=runtime.terra_model,
        )
        envelope = RenovationEstimateEnvelope(
            schema_version=ENVELOPE_SCHEMA_VERSION,
            estimate_id=estimate_id,
            state="failed",
            reason=descriptor.category,
            error_detail=str(exc),
            provenance=provenance,
            result=None,
        )
        return envelope.to_dict()

    envelope = RenovationEstimateEnvelope(
        schema_version=ENVELOPE_SCHEMA_VERSION,
        estimate_id=estimate_id,
        state="package_review_complete",
        reason=None,
        error_detail=None,
        provenance=provenance,
        result=result,
    )
    return envelope.to_dict()
