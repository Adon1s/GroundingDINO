"""Renovation architecture v1: contracts, validators, and estimator runtime.

catalog_projection is deliberately not re-exported — it drags the catalog
validation and rehab_packages dependency chain, which current-mode startup
must never pay for. Import it as a submodule where needed.
"""
from tools.renovation_architecture.contracts import (
    CONTRACTS_SCHEMA_VERSION,
    ENVELOPE_SCHEMA_VERSION,
    PROJECTION_VERSION,
    SCAFFOLD_REASON,
    SHADOW_DEBUG_KEY,
    TERMINAL_ROUTE_POLICY_VERSION,
)
from tools.renovation_architecture.runtime import (
    RenovationArchitectureInitError,
    build_estimate_envelope,
    build_shadow_envelope,
    get_runtime,
    initialize_renovation_architecture,
    reset_runtime_for_tests,
)
from tools.renovation_architecture.validators import (
    ValidationResult,
    validate_complete_result,
    validate_condition_review_result,
    validate_envelope,
    validate_package_review_result,
    validate_standalone_estimate_result,
)

__all__ = [
    "CONTRACTS_SCHEMA_VERSION",
    "ENVELOPE_SCHEMA_VERSION",
    "PROJECTION_VERSION",
    "SCAFFOLD_REASON",
    "SHADOW_DEBUG_KEY",
    "TERMINAL_ROUTE_POLICY_VERSION",
    "RenovationArchitectureInitError",
    "ValidationResult",
    "build_estimate_envelope",
    "build_shadow_envelope",
    "get_runtime",
    "initialize_renovation_architecture",
    "reset_runtime_for_tests",
    "validate_complete_result",
    "validate_condition_review_result",
    "validate_envelope",
    "validate_package_review_result",
    "validate_standalone_estimate_result",
]
