"""Benchmark CLI: dataset freezing and reference authoring.

    python -m tools.benchmark validate  --dataset renovation-v1
    python -m tools.benchmark import    --dataset renovation-v1 --listing hsv-001 \
                                       --property-key redfin_125970550
    python -m tools.benchmark reference template --dataset renovation-v1 --listing hsv-001
    python -m tools.benchmark reference check    --dataset renovation-v1 --listing hsv-001
    python -m tools.benchmark reference compile  --dataset renovation-v1 --listing hsv-001
    python -m tools.benchmark catalog search "water stain"
    python -m tools.benchmark vocabulary diff --dataset renovation-v1
    python -m tools.benchmark seal      --dataset renovation-v1

Exit codes: 0 ok, 1 configuration/integrity failure. Run execution and
evaluation land in later sessions; this session covers everything needed to
freeze inputs and author human truth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.benchmarking import catalog_index, dataset as ds, reference_authoring as authoring
from tools.benchmarking import vocabulary as vocab_mod
from tools.benchmarking.schemas import ValidationResult
from tools.comparison_common import ComparisonError

DATASETS_ROOT = PROJECT_ROOT / "benchmarks" / "datasets"

# The frontend owns listing photos. Read-only: nothing here writes to that repo.
FRONTEND_ROOT = Path(r"C:\Users\Steven\IntelliJProjects\realtorvision")


def _load_catalog() -> Dict[str, Any]:
    from tools import pipeline_config as cfg
    from tools.artifact_writers import load_issue_catalog
    return load_issue_catalog(Path(cfg.ISSUE_CATALOG_PATH))


def _dataset_path(args: argparse.Namespace) -> Path:
    root = Path(args.datasets_root).resolve() if args.datasets_root else DATASETS_ROOT
    path = ds.dataset_dir(root, args.dataset)
    if not path.is_dir():
        raise ComparisonError(f"dataset not found: {path}")
    return path


def _vocabulary(dataset_path: Path) -> Dict[str, Any]:
    """The dataset's frozen vocabulary, or a live snapshot for a draft dataset.

    A draft has nothing frozen yet, so authoring against the live vocabulary is
    correct there — sealing is the moment it gets written down.
    """
    frozen = ds.load_frozen_vocabulary(dataset_path)
    return frozen if frozen is not None else vocab_mod.snapshot(_load_catalog())


def _report(result: ValidationResult, label: str) -> int:
    for message in result.warnings:
        print(f"warning: {message}")
    if result.ok:
        print(f"{label}: OK" + (f" ({len(result.warnings)} warning(s))"
                                if result.warnings else ""))
        return 0
    print(f"{label}: {len(result.errors)} error(s)", file=sys.stderr)
    for message in result.errors:
        print(f"  - {message}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    dataset_path = _dataset_path(args)
    result = ds.validate_dataset(
        dataset_path,
        listing_ids=[args.listing] if args.listing else None,
        check_photo_bytes=not args.skip_photo_bytes,
    )
    return _report(result, f"dataset {args.dataset}")


def cmd_import(args: argparse.Namespace) -> int:
    dataset_path = _dataset_path(args)
    source = (Path(args.source).resolve() if args.source
              else FRONTEND_ROOT / "public" / "images" / "properties" / args.property_key)
    listing = ds.import_listing(
        dataset_path,
        listing_id=args.listing,
        source_dir=source,
        tier=args.tier,
        slices=args.slice or (),
        property_metadata=_scrape_metadata(args.property_key) if args.property_key else None,
        market_inputs={"area_ppsf": args.area_ppsf, "source": args.market_source,
                       "sample_size": args.market_sample_size} if args.area_ppsf else None,
    )
    print(f"imported {args.listing}: {len(listing['photos'])} photos from {source}")
    print("Next: fill market_inputs in metadata.json if you did not pass --area-ppsf, then")
    print(f"      python -m tools.benchmark reference template --dataset {args.dataset} "
          f"--listing {args.listing}")
    return 0


def _scrape_metadata(property_key: str) -> Dict[str, Any]:
    """Best-effort listing facts from the frontend's scrape output.

    Returns {} rather than failing: metadata can be filled by hand, and an
    import that dies because a scrape file moved would be worse than one that
    leaves fields blank for validation to catch.
    """
    import json

    for candidate in (
        FRONTEND_ROOT / "property_scrapes" / f"{property_key}.json",
        FRONTEND_ROOT / "scrapes" / f"{property_key}.json",
    ):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8-sig"))
                if isinstance(data, dict):
                    print(f"read listing metadata from {candidate}")
                    return data
            except (OSError, ValueError) as exc:
                print(f"warning: could not read {candidate}: {exc}")
    print(f"warning: no scrape metadata found for {property_key}; fill metadata.json by hand")
    return {}


def cmd_reference(args: argparse.Namespace) -> int:
    dataset_path = _dataset_path(args)
    manifest = ds.load_manifest(dataset_path)
    listing = ds.load_listing(dataset_path, manifest, args.listing)
    vocabulary = _vocabulary(dataset_path)
    listing_path = ds.listing_dir(dataset_path, args.listing)
    draft_path = listing_path / authoring.DRAFT_FILENAME

    if args.reference_action == "template":
        ds.require_unsealed(manifest, f"writing a reference template for {args.listing!r}")
        written = authoring.write_template(draft_path, listing, vocabulary,
                                          tier=args.tier, overwrite=args.force)
        print(f"wrote {written}")
        print(f"{len(listing.get('photos') or [])} photos pre-listed under photo_expectations.")
        return 0

    if args.reference_action == "check":
        _, result = authoring.check(draft_path, listing=listing, vocabulary=vocabulary)
        return _report(result, f"reference {args.listing}")

    ds.require_unsealed(manifest, f"compiling a reference for {args.listing!r}")
    written = authoring.compile_draft(draft_path, listing=listing, vocabulary=vocabulary)
    print(f"compiled {draft_path.name} -> {written}")
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    results = catalog_index.search(
        _load_catalog(), args.query,
        scene_group=args.scene_group, kind=args.kind, limit=args.limit,
    )
    print(catalog_index.format_results(results))
    return 0


def cmd_vocabulary(args: argparse.Namespace) -> int:
    dataset_path = _dataset_path(args)
    frozen = ds.load_frozen_vocabulary(dataset_path)
    if frozen is None:
        print(f"dataset {args.dataset} has no frozen vocabulary yet (not sealed); "
              f"nothing to diff against")
        return 0
    changes = vocab_mod.diff(frozen, issue_catalog=_load_catalog())
    lines = vocab_mod.summarize_diff(changes)
    if not lines:
        print("frozen vocabulary matches live code exactly")
        return 0
    print(f"vocabulary drift since sealing ({'compatible' if changes['compatible'] else 'INCOMPATIBLE'}):")
    for line in lines:
        print(f"  {line}")
    if not changes["compatible"]:
        print("\nRemovals and regroupings invalidate existing truth. The evaluator maps live "
              "output into the frozen vocabulary; review these before trusting a comparison.")
    return 0


def cmd_seal(args: argparse.Namespace) -> int:
    dataset_path = _dataset_path(args)
    manifest = ds.seal(dataset_path, _load_catalog())
    print(f"sealed {args.dataset}")
    print(f"  vocabulary_fingerprint {manifest['vocabulary_fingerprint'][:16]}...")
    print(f"  dataset_fingerprint    {manifest['dataset_fingerprint'][:16]}...")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-root", help="override benchmarks/datasets (tests use this)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_dataset_arg(sub: argparse.ArgumentParser, *, required: bool = True) -> None:
        sub.add_argument("--dataset", required=required, help="dataset version directory")

    validate = subparsers.add_parser("validate", help="validate manifest, listings, references, photos")
    add_dataset_arg(validate)
    validate.add_argument("--listing", help="validate one listing only")
    validate.add_argument("--skip-photo-bytes", action="store_true",
                          help="skip re-hashing photo bytes (faster; weaker)")
    validate.set_defaults(func=cmd_validate)

    importer = subparsers.add_parser("import", help="freeze a listing's photos and metadata")
    add_dataset_arg(importer)
    importer.add_argument("--listing", required=True, help="benchmark listing id, e.g. hsv-001")
    importer.add_argument("--property-key", help="production property key, e.g. redfin_125970550")
    importer.add_argument("--source", help="explicit source image directory")
    importer.add_argument("--tier", default="gold", choices=("gold", "silver"))
    importer.add_argument("--slice", action="append", help="repeatable slice name")
    importer.add_argument("--area-ppsf", type=float, help="frozen area price per sqft")
    importer.add_argument("--market-source", default="manual", help="provenance of --area-ppsf")
    importer.add_argument("--market-sample-size", type=int, default=0)
    importer.set_defaults(func=cmd_import)

    reference = subparsers.add_parser("reference", help="author, check, and compile references")
    add_dataset_arg(reference)
    reference.add_argument("reference_action", choices=("template", "check", "compile"))
    reference.add_argument("--listing", required=True)
    reference.add_argument("--tier", default="gold", choices=("gold", "silver"))
    reference.add_argument("--force", action="store_true",
                           help="overwrite an existing draft (destroys annotation work)")
    reference.set_defaults(func=cmd_reference)

    catalog = subparsers.add_parser("catalog", help="search the issue catalog while annotating")
    catalog.add_argument("catalog_action", choices=("search",))
    catalog.add_argument("query", nargs="?", default="")
    catalog.add_argument("--scene-group")
    catalog.add_argument("--kind", choices=("defect", "upgrade"))
    catalog.add_argument("--limit", type=int, default=25)
    catalog.set_defaults(func=cmd_catalog)

    vocabulary = subparsers.add_parser("vocabulary", help="compare frozen vocabulary to live code")
    add_dataset_arg(vocabulary)
    vocabulary.add_argument("vocabulary_action", choices=("diff",))
    vocabulary.set_defaults(func=cmd_vocabulary)

    seal = subparsers.add_parser("seal", help="freeze the vocabulary and make the dataset immutable")
    add_dataset_arg(seal)
    seal.set_defaults(func=cmd_seal)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ComparisonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
