"""Build reports/adjudication_queue.json — the v1.1 label repair queue.

Design: docs/DESIGN_label_v1_1_adjudication.md (schema and mappings §§3-5 are
pre-committed there; this script only implements them).

Three arms over the FROZEN review evidence (read-only, never modified):

  phase 1  blind    55 canary `supported_billed` cards, no prior labels shown
  phase 2  re-ask   the 46 re-tag cards, prior verdict + re-tag answer shown
  phase 3  blind    24 production `supported_billed` cards

plus ~12% invisible repeats inside the blind arms, placed >= 15 positions after
their origin, to measure the reviewer's own consistency — the one noise floor
in this program that has never been measured.

Phase order matters: the blind arm runs before the reviewer re-opens cards he
has strong opinions about, and after phases 1+2 the canary label set is
complete, which is what unblocks Session F. Stopping between phases always
leaves a coherent artifact.

Blindness is enforced, not assumed: `--check` re-serialises every blind card
and fails if any v1 verdict, re-tag answer, stratum name or withheld meta key
survives into what the browser can see. The v1 linkage lives in a top-level
`provenance` block, which `review_server.py` never sends.

Run:
  .venv\\Scripts\\python.exe scripts\\build_adjudication_queue.py --check
  .venv\\Scripts\\python.exe scripts\\review_server.py --queue reports\\adjudication_queue.json --verdicts reports\\adjudication_verdicts.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import label_schema as ls  # noqa: E402
from tools import review_cards as rc  # noqa: E402

QUEUE = ROOT / "reports" / "review_queue.json"
VERDICTS = ROOT / "reports" / "review_verdicts.jsonl"
RETAG_QUEUE = ROOT / "reports" / "retag_queue.json"
RETAG_VERDICTS = ROOT / "reports" / "retag_verdicts.jsonl"
OUT = ROOT / "reports" / "adjudication_queue.json"

# The frozen evidence this repair is defined against.
FROZEN = {
    "review_queue.json": "8512b87c2af18d1104069c15a5eda977d5fe79eb736f9aedd976d174f89d318a",
    "review_verdicts.jsonl": "0c7ca6a8b55d6dac69021a00e81315be1b15cef68ffd59fbe2973cb895a8d70f",
    "retag_queue.json": "aac934b72102fe91be1d89a249474ea05d1fba54994cba532bb748e230c4f78a",
    "retag_verdicts.jsonl": "a038029811c9fafe78f6e2cd3d2456d26ad6499018a217ff66ecb4c13901a4a7",
}

EXPECTED = {"blind_canary": 55, "reask": 46, "blind_production": 24}

REPEAT_RATE = 0.12
REPEAT_GAP = 15          # a repeat lands at least this many cards after its origin
ORDER_SALT = "adjudication-order-v1.1"
REPEAT_SALT = "adjudication-repeat-v1.1"

VERDICT_KEYS = {k: f"{slug}: {ls.ADJUDICATION_HELP[slug]}"
                for k, slug in ls.ADJUDICATION_KEYS.items()}

# Anything that would tell a blind reviewer what he decided last time.
# Field names are matched as substrings; vocabulary words are matched only as
# whole JSON values, so prose like "uniform fading" in a claim cannot trip it.
FORBIDDEN_SUBSTRINGS = (
    "terra_claim_", "terra_evidence_", "terra_verdict", "replica_terra",
    "second_opinion", "disposition", "accepted_for_work", "prior_verdict",
    "prior_note", "retag_group",
)
# Matched as a complete JSON string value (`"dirA"`), so an observation that
# happens to contain the word cannot trip it.
FORBIDDEN_EXACT_VALUES = (
    "dirA", "dirB", "uniform", "terra_flip", "p6_forced_single",
    "p1_package", "p3_bathroom",
    "supported", "unsupported", "overstated", "cannot_assess",
)
# Re-tag answers are stored as `slug: help text`, so match the slug prefix.
# None of these collide with the v1.1 vocabulary in ADJUDICATION_KEYS.
FORBIDDEN_VALUE_PREFIXES = (
    "mechanism_only", "wholly_false", "too_trivial", "cannot_tell",
    "wording_blocked", "terra_miss", "borderline",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mint(origin_card_id: str, role: str) -> str:
    digest = hashlib.sha1(f"{origin_card_id}:{role}".encode("utf-8")).hexdigest()
    return f"adj_{digest[:12]}"


def order_key(card_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{card_id}".encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ selection

def select(cards: List[Dict[str, Any]], done: Dict[str, Dict[str, Any]],
           retag_ids: set) -> Dict[str, List[Dict[str, Any]]]:
    """Blind arms = `supported_billed` by the v1 rule; re-ask arm = the re-tag set."""
    out: Dict[str, List[Dict[str, Any]]] = {
        "blind_canary": [], "blind_production": [], "reask": []}
    for card in cards:
        if card.get("kind") != "condition":
            continue
        cid = card["card_id"]
        rec = done.get(cid)
        if not rec:
            continue
        if cid in retag_ids:
            out["reask"].append(card)
            continue
        klass = ls.classify_v1(card.get("strata") or [],
                               (card.get("meta") or {}).get("accepted"),
                               rec.get("verdict"))
        if klass != "supported_billed":
            continue
        arm = ("blind_canary" if card.get("source") == "canary"
               else "blind_production")
        out[arm].append(card)
    return out


# --------------------------------------------------------------- card shaping

def base_card(card: Dict[str, Any], adj_id: str, phase: int) -> Dict[str, Any]:
    """The reviewer-visible card. Everything v1 decided is stripped here."""
    meta = {k: v for k, v in (card.get("meta") or {}).items()
            if k not in rc.BLIND_META}
    return {
        "card_id": adj_id,
        "kind": "adjudication",          # not `condition`: hides the a-e tags
        "phase": phase,
        "source": card.get("source"),
        "property_key": card.get("property_key"),
        "run_id": card.get("run_id"),
        "address": card.get("address"),
        "title": card.get("title"),
        "claim": {
            "catalog_claim": (card.get("claim") or {}).get("catalog_claim"),
            "observations": list((card.get("claim") or {}).get("observations") or []),
        },
        "strips": card.get("strips") or [],
        "meta": meta,
        "verdict_options": list(VERDICT_KEYS.values()),
        "verdict_keys": dict(VERDICT_KEYS),
        "tags": {},
        "strata": [],                    # real strata live in `provenance`
        "reveal": [],
        "hidden": None,
        "legacy_item_id": None,
    }


def blind_card(card: Dict[str, Any], adj_id: str, phase: int) -> Dict[str, Any]:
    out = base_card(card, adj_id, phase)
    out["reveal"] = [{
        "label": "v1.1 adjudication",
        "text": "recorded. Prior labels are withheld by design on this arm.",
    }]
    return out


def reask_card(card: Dict[str, Any], adj_id: str, phase: int,
               v1: Dict[str, Any], retag: Optional[Dict[str, Any]]
               ) -> Dict[str, Any]:
    """Non-blind by design: both earlier answers are the reviewer's own."""
    out = base_card(card, adj_id, phase)
    v1_verdict = str(v1.get("verdict") or "").replace("terra_claim_", "").replace("terra_evidence_", "")
    v1_note = (v1.get("notes") or "").strip()
    lines = [f">> v1 call: {v1_verdict}" + (f' — "{v1_note}"' if v1_note else " (no note)")]
    if retag:
        rt_slug = ls.slug_of(retag.get("verdict")) or "?"
        rt_note = (retag.get("notes") or "").strip()
        lines.append(f">> re-tag call: {rt_slug}" + (f' — "{rt_note}"' if rt_note else " (no note)"))
    lines.append(">> now: separate the two axes — is the claim true AS WORDED, "
                 "and is the work worth doing?")
    out["title"] = f"[re-ask] {card.get('title')}"
    out["claim"]["observations"] = lines + out["claim"]["observations"]
    out["reveal"] = [{"label": "v1 verdict", "text": v1_verdict + (f" — {v1_note}" if v1_note else "")}]
    if retag:
        out["reveal"].append({"label": "re-tag answer",
                              "text": str(retag.get("verdict"))})
    return out


# ------------------------------------------------------------------- ordering

def hash_order(primaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strata-blind presentation order: stable, reproducible, uncorrelated."""
    return sorted(primaries, key=lambda c: order_key(c["card_id"], ORDER_SALT))


def pick_repeats(ordered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Which cards to ask twice.

    Only the prefix that still leaves room for a full REPEAT_GAP is eligible —
    a repeat crammed two cards behind its origin measures recall, not
    consistency. The prefix is a hash order, so restricting to it does not
    correlate the repeat sample with anything on the card.
    """
    n = round(len(ordered) * REPEAT_RATE)
    eligible = ordered[:max(0, len(ordered) - REPEAT_GAP)]
    return sorted(eligible, key=lambda c: order_key(c["card_id"], REPEAT_SALT))[:n]


def insert_repeats(ordered: List[Dict[str, Any]],
                   repeats: List[Tuple[Dict[str, Any], Dict[str, Any]]]
                   ) -> List[Dict[str, Any]]:
    """Each repeat lands exactly REPEAT_GAP cards after its origin."""
    out = list(ordered)
    for origin, repeat in repeats:
        idx = next(i for i, c in enumerate(out) if c["card_id"] == origin["card_id"])
        out.insert(idx + REPEAT_GAP, repeat)
    return out


# --------------------------------------------------------------- blind checks

def leaks(card: Dict[str, Any]) -> List[str]:
    """What a blind card would still tell the reviewer about his v1 answer.

    Checks exactly what the server can send: `blind_card()` before the verdict
    and `reveal_payload()` after it.
    """
    served = json.dumps(
        {"public": rc.blind_card(card), "reveal": rc.reveal_payload(card)},
        ensure_ascii=False,
    )
    # Photo paths are served as opaque tokens, and carry property keys/run ids
    # that are legitimately on the card anyway.
    for strip in (card.get("strips") or []):
        for photo in (strip.get("photos") or []):
            if photo.get("path"):
                served = served.replace(json.dumps(str(photo["path"]))[1:-1], "")
    found = {bad for bad in FORBIDDEN_SUBSTRINGS if bad in served}
    found |= {v for v in FORBIDDEN_EXACT_VALUES if f'"{v}"' in served}
    found |= {v for v in FORBIDDEN_VALUE_PREFIXES if f'"{v}:' in served}
    return sorted(found)


# ----------------------------------------------------------------------- main

def build(review_queue: Dict[str, Any], done: Dict[str, Dict[str, Any]],
          retag_queue: Dict[str, Any], retag_done: Dict[str, Dict[str, Any]],
          source_hashes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    retag_order = {c["card_id"]: i for i, c in enumerate(retag_queue["cards"])}
    picked = select(review_queue["cards"], done, set(retag_order))

    provenance: Dict[str, Dict[str, Any]] = {}
    phases: Dict[int, List[Dict[str, Any]]] = {}

    def record(origin: Dict[str, Any], adj_id: str, role: str, arm: str) -> None:
        cid = origin["card_id"]
        rec = done.get(cid) or {}
        rt = retag_done.get(cid) or {}
        meta = origin.get("meta") or {}
        provenance[adj_id] = {
            "origin_card_id": cid,
            "role": role,
            "arm": arm,
            "source": origin.get("source"),
            "property_key": origin.get("property_key"),
            "condition_id": meta.get("condition_id"),
            "catalog_item_id": meta.get("catalog_item_id"),
            "catalog_kind": meta.get("catalog_kind"),
            "strata": list(origin.get("strata") or []),
            "accepted": bool(meta.get("accepted")),
            "stored_terra_verdict": meta.get("terra_verdict"),
            "v1_verdict": rec.get("verdict"),
            "v1_note": (rec.get("notes") or None),
            "v1_class": ls.classify_v1(origin.get("strata") or [],
                                       meta.get("accepted"), rec.get("verdict")),
            "retag_answer": ls.slug_of(rt.get("verdict")) if rt else None,
            "retag_note": (rt.get("notes") or None) if rt else None,
        }

    for arm, phase in (("blind_canary", 1), ("reask", 2), ("blind_production", 3)):
        originals = picked[arm]
        if arm == "reask":
            # Keep the re-tag queue's own order: group A then group B.
            originals = sorted(originals,
                               key=lambda c: retag_order[c["card_id"]])
            cards = []
            for origin in originals:
                adj_id = mint(origin["card_id"], "primary")
                record(origin, adj_id, "primary", arm)
                cards.append(reask_card(origin, adj_id, phase,
                                        done[origin["card_id"]],
                                        retag_done.get(origin["card_id"])))
            phases[phase] = cards
            continue

        primaries = []
        by_adj: Dict[str, Dict[str, Any]] = {}
        for origin in originals:
            adj_id = mint(origin["card_id"], "primary")
            record(origin, adj_id, "primary", arm)
            primaries.append(blind_card(origin, adj_id, phase))
            by_adj[adj_id] = origin
        ordered = hash_order(primaries)
        repeats = []
        for card in pick_repeats(ordered):
            origin = by_adj[card["card_id"]]
            adj_id = mint(origin["card_id"], "repeat")
            record(origin, adj_id, "repeat", arm)
            provenance[adj_id]["repeat_of"] = card["card_id"]
            repeats.append((card, blind_card(origin, adj_id, phase)))
        phases[phase] = insert_repeats(ordered, repeats)

    cards = [c for phase in sorted(phases) for c in phases[phase]]
    return {
        "schema_version": 1,
        "label_version": ls.LABEL_VERSION,
        "generated_for": "v1.1 label repair — docs/DESIGN_label_v1_1_adjudication.md",
        "order": "hash",
        "sources": {name: {"sha256": sha,
                           "frozen_match": sha == FROZEN.get(name)}
                    for name, sha in sorted((source_hashes or {}).items())},
        "schema": {
            "keys": VERDICT_KEYS,
            "claim_axis": ls.CLAIM_AXIS,
            "work_axis": ls.WORK_AXIS,
        },
        "arms": {
            arm: {
                "phase": phase,
                "primaries": sum(1 for p in provenance.values()
                                 if p["arm"] == arm and p["role"] == "primary"),
                "repeats": sum(1 for p in provenance.values()
                               if p["arm"] == arm and p["role"] == "repeat"),
            }
            for arm, phase in (("blind_canary", 1), ("reask", 2),
                               ("blind_production", 3))
        },
        "provenance": provenance,
        "cards": cards,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=QUEUE)
    ap.add_argument("--verdicts", type=Path, default=VERDICTS)
    ap.add_argument("--retag-queue", type=Path, default=RETAG_QUEUE)
    ap.add_argument("--retag-verdicts", type=Path, default=RETAG_VERDICTS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="fail on drifted inputs, wrong arm sizes, or blindness leaks")
    args = ap.parse_args(argv)

    paths = {"review_queue.json": args.queue, "review_verdicts.jsonl": args.verdicts,
             "retag_queue.json": args.retag_queue,
             "retag_verdicts.jsonl": args.retag_verdicts}
    ok = True
    hashes: Dict[str, str] = {}
    for name, path in paths.items():
        hashes[name] = sha256(path)
        match = hashes[name] == FROZEN[name]
        ok &= match
        print(f"{name:24s} {hashes[name][:12]}  "
              f"{'== frozen' if match else '!! DRIFTED from the frozen evidence'}")

    review_queue = json.loads(args.queue.read_text(encoding="utf-8"))
    retag_queue = json.loads(args.retag_queue.read_text(encoding="utf-8"))
    payload = build(review_queue, rc.latest_verdicts(args.verdicts),
                    retag_queue, rc.latest_verdicts(args.retag_verdicts),
                    source_hashes=hashes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote {args.out}  cards={len(payload['cards'])}")

    for arm, info in payload["arms"].items():
        expected = EXPECTED[arm]
        good = info["primaries"] == expected
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] phase {info['phase']} {arm:18s} "
              f"{info['primaries']} primaries (expected {expected}) "
              f"+ {info['repeats']} repeats")

    ids = [c["card_id"] for c in payload["cards"]]
    unique = len(set(ids)) == len(ids)
    ok &= unique
    print(f"  [{'ok' if unique else 'FAIL'}] unique card ids: {len(set(ids))}/{len(ids)}")

    blind = [c for c in payload["cards"] if c["phase"] in (1, 3)]
    leaked = {c["card_id"]: bad for c in blind if (bad := leaks(c))}
    ok &= not leaked
    print(f"  [{'ok' if not leaked else 'FAIL'}] blindness: {len(blind)} blind cards, "
          f"{len(leaked)} leaking")
    for cid, bad in list(leaked.items())[:5]:
        print(f"         {cid}: {bad}")

    gaps = repeat_gaps(payload)
    tight = [g for g in gaps if g < REPEAT_GAP]
    ok &= not tight
    print(f"  [{'ok' if not tight else 'FAIL'}] repeat spacing: min gap "
          f"{min(gaps) if gaps else '—'} (need >= {REPEAT_GAP})")

    print("\nnext:\n  .venv\\Scripts\\python.exe scripts\\review_server.py "
          "--queue reports\\adjudication_queue.json "
          "--verdicts reports\\adjudication_verdicts.jsonl")
    if args.check and not ok:
        return 1
    return 0


def repeat_gaps(payload: Dict[str, Any]) -> List[int]:
    pos = {c["card_id"]: i for i, c in enumerate(payload["cards"])}
    out = []
    for adj_id, prov in payload["provenance"].items():
        origin = prov.get("repeat_of")
        if origin and origin in pos and adj_id in pos:
            out.append(pos[adj_id] - pos[origin])
    return out


if __name__ == "__main__":
    sys.exit(main())
