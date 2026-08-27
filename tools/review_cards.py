"""Review cards for the manual v5 output-quality review (zero provider calls).

Builds one generic card queue from canary (shadow-mode) and production
(new-mode) artifacts:

  condition cards  the claim Terra was asked to verify + the photos it saw
  package cards    v4 Pass 2f rejected the package, v5 Sol decided it (P1)
  bathroom cards   listings with >= 2 bathroom surrogates (P3)

Strata (dirA / dirB / terra_flip / uniform / p1_package / p3_bathroom /
p6_forced_single) are *sampling labels*: the human judges the claim against the
photo; nothing about Pass 2f correctness is derived from a condition verdict.
See docs/DESIGN_review_method.md. The Session 9 builder and its outputs are
read (legacy ids) but never modified.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.compare_renovation_architecture_cutover import _load_latest_artifacts
from tools.pass_2f_artifact_inputs import photo_key_to_path
from tools.renovation_architecture.contracts import SHADOW_DEBUG_KEY
from tools.renovation_architecture.terra_review import OBSERVATION_MAX_CHARS, _claim_text

ROOT = Path(__file__).resolve().parents[1]
CANARY_ROOT = ROOT / "artifacts_canary" / "renovation_session9_20260818"
PROD_ROOT = Path("C:/Users/Steven/IntelliJProjects/renointel-prod/artifacts")
FE_DB = Path("C:/Users/Steven/IntelliJProjects/renointel-prod/prisma/dev.db")
CATALOG_PATH = ROOT / "tools" / "issue_catalog_kind_v2.json"
PACKETS_PATH = ROOT / "reports" / "session9_decision_packets.json"
DEFAULT_SINCE = "20260821_230000"
DEFAULT_UNIFORM_RATE = 0.07
RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{8}$")
MIN_EVIDENCE_PX = 500  # a photo whose short side is below this is a thumbnail, unusable as evidence

CONDITION_VERDICTS = ["terra_claim_supported", "terra_claim_unsupported",
                      "terra_claim_overstated", "terra_evidence_inconclusive"]
PACKAGE_VERDICTS = ["package_warranted", "not_warranted", "unsure"]
BATHROOM_BILLING = ["per_bathroom", "once", "unsure"]
ERROR_TAGS = {"a": "stain_or_colour_read_as_wear", "b": "style_claim_on_ordinary_finish",
              "c": "wrong_object_or_room", "d": "real_but_overstated", "e": "other"}
PHASE_OF = {"dirA": 1, "terra_flip": 1, "uniform": 1, "dirB": 2,
            "p1_package": 3, "p3_bathroom": 3, "p6_forced_single": 3}
STRATUM_ORDER = ["dirA", "terra_flip", "uniform", "dirB", "p1_package", "p3_bathroom", "p6_forced_single"]
SECOND_OPINION = {"agree": "2f_agreed", "A": "2f_objected", "mixed": "2f_mixed",
                  "B": "2f_confirmed", "none": "none"}
BLIND_META = ("terra_verdict", "replica_terra_verdict", "direction", "second_opinion",
              "accepted", "disposition")


# --------------------------------------------------------------------------- loading

def v5_result(art: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The complete v5 result: root placement (new mode) or analysis_debug (shadow)."""
    if not isinstance(art, dict):
        return None
    env = art.get("renovation_estimate_v5")
    if not isinstance(env, dict):
        env = (art.get("analysis_debug") or {}).get(SHADOW_DEBUG_KEY)
    if isinstance(env, dict) and env.get("state") == "complete" and isinstance(env.get("result"), dict):
        return env["result"]
    return None


def iter_runs(root: Path, since: Optional[str] = DEFAULT_SINCE) -> List[Tuple[str, str, Path, Dict[str, Any]]]:
    """Newest complete-v5 run per property under a production root.

    Filters run-dir names (``YYYYMMDD_HHMMSS_hex8`` and ``>= since``) *before*
    loading JSON so the 1,200+ pre-cutover runs are never read."""
    out: List[Tuple[str, str, Path, Dict[str, Any]]] = []
    root = Path(root)
    if not root.is_dir():
        return out
    for prop_dir in sorted(root.iterdir()):
        if not prop_dir.is_dir() or prop_dir.name.startswith("."):
            continue
        for run_dir in sorted(prop_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or not RUN_DIR_RE.match(run_dir.name):
                continue
            if since and run_dir.name < since:
                continue
            path = run_dir / "photo_intel_debug.json"
            if not path.is_file():
                continue
            try:
                art = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if v5_result(art) is None:
                continue
            prop = str(((art.get("property") or {}).get("property_key")) or prop_dir.name)
            out.append((prop, run_dir.name, path, art))
            break
    return out


def load_canary(root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """run_1 / run_2 ``{property_key: (path, artifact)}`` via the cutover comparator's loader."""
    root = Path(root)
    run1 = _load_latest_artifacts(root / "run_1" / "candidate") if (root / "run_1").is_dir() else {}
    run2 = _load_latest_artifacts(root / "run_2" / "candidate") if (root / "run_2").is_dir() else {}
    return run1, run2


def claim_texts(catalog_path: Path = CATALOG_PATH) -> Dict[str, str]:
    """catalog_item_id -> the claim text Terra is shown (atomic_claim subject + state)."""
    cat = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    items = cat.get("items") if isinstance(cat, dict) else cat
    return {it["id"]: _claim_text(it, it["id"]) for it in items or [] if isinstance(it, dict) and it.get("id")}


def load_packets(path: Path = PACKETS_PATH) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def legacy_ids(packets: Dict[str, Any]) -> Dict[Tuple[str, ...], str]:
    """Session 9 ids (C###/P##/B##/P6 review ids) keyed by exact joins."""
    m: Dict[Tuple[str, ...], str] = {}
    for c in packets.get("p2_contradictions") or []:
        m[("condition", c["property"], c["condition_id"])] = c["item_id"]
    for p in packets.get("p1_packages") or []:
        m[("package", p["property"], p["package"])] = p["item_id"]
    for b in packets.get("p3_multi_bath") or []:
        m[("bathroom", b["property"])] = b["item_id"]
    for t in packets.get("p6_tier2_zero_items") or []:
        m[("p6", t["property"], t["item"])] = t["review_id"]
    return m


def addresses(db_path: Path, keys: Iterable[str]) -> Dict[str, str]:
    """property_key -> 'street, city ST zip' from the frontend SQLite (read-only; optional)."""
    db_path = Path(db_path)
    wanted = set(keys)
    if not db_path.is_file() or not wanted:
        return {}
    try:
        con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        rows = con.execute("SELECT propertyKey, address, city, state, zipCode FROM Property").fetchall()
    except sqlite3.Error:
        return {}
    finally:
        con.close()
    out = {}
    for key, street, city, state, zipcode in rows:
        if key in wanted:
            tail = " ".join(x for x in (state, zipcode) if x)
            out[key] = ", ".join(x for x in (street, city, tail) if x)
    return out


# --------------------------------------------------------------------------- joins

def index_result(res: Dict[str, Any]) -> Dict[str, Any]:
    idx: Dict[str, Any] = {
        "conds": {c["condition_id"]: c for c in res.get("observed_conditions") or []},
        "revs": {r["condition_id"]: r for r in res.get("condition_reviews") or []},
        "disps": {d["condition_id"]: d for d in res.get("condition_dispositions") or []},
        "evs": {e["condition_id"]: e for e in res.get("evidence_facts") or []},
        "calls": {c["call_id"]: c for c in res.get("terra_calls") or [] if c.get("call_id")},
        "works": {w["work_item_id"]: w for w in res.get("work_items") or []},
        "work_by_cond": {},
        "cond_by_issue": {},
    }
    for w in res.get("work_items") or []:
        if w.get("status") == "active":
            for cid in w.get("condition_ids") or []:
                idx["work_by_cond"][cid] = w
    for c in idx["conds"].values():
        for iid in c.get("issue_ids") or []:
            idx["cond_by_issue"][iid] = c
    return idx


def join_2f(v4: Dict[str, Any], idx: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """v4 Pass 2f issue-level verdicts folded onto v5 conditions.

    condition_id -> {rows: [(issue_id, verdict, package_id)], packages: [..],
    evidence: [summary..]}. Same dedup as the Session 9 builder: (issue_id, package_id)
    over ``package_candidates + packages``."""
    out: Dict[str, Dict[str, Any]] = {}
    seen = set()
    for pk in (v4.get("package_candidates") or []) + (v4.get("packages") or []):
        raw = pk.get("raw_pass_2f_response")
        try:
            j = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            j = {}
        summary = j.get("evidence_summary") or pk.get("evidence_summary")
        for verdict, ids in (("confirmed", pk.get("confirmed_issue_ids") or []),
                             ("rejected", pk.get("rejected_issue_ids") or [])):
            for iid in ids:
                key = (iid, pk.get("package_id"))
                if key in seen:
                    continue
                seen.add(key)
                c = idx["cond_by_issue"].get(iid)
                if not c:
                    continue
                rec = out.setdefault(c["condition_id"], {"rows": [], "packages": [], "evidence": []})
                rec["rows"].append((iid, verdict, pk.get("package_id")))
                if pk.get("package_id") not in rec["packages"]:
                    rec["packages"].append(pk.get("package_id"))
                if summary and summary not in rec["evidence"]:
                    rec["evidence"].append(summary)
    return out


def direction(terra_verdict: Optional[str], f2_verdicts: Iterable[str]) -> str:
    """A = Terra supported / 2f rejected; B = 2f confirmed / Terra not; agree; mixed; none."""
    vs = set(f2_verdicts)
    if not vs or terra_verdict not in ("supported", "unsupported", "cannot_assess"):
        return "none"
    if terra_verdict == "supported":
        return "agree" if vs == {"confirmed"} else "A" if vs == {"rejected"} else "mixed"
    return "B" if "confirmed" in vs else "agree"


def terra_flips(res1: Dict[str, Any], res2: Optional[Dict[str, Any]]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """(run_1 condition_id, verdict_1, verdict_2) where the same (item, unit, photo-set)
    key — unique within each run — got different Terra verdicts (Session 9 proxy rule)."""
    if not res2:
        return []

    def by_key(res):
        evs = {e["condition_id"]: e for e in res.get("evidence_facts") or []}
        revs = {r["condition_id"]: r for r in res.get("condition_reviews") or []}
        cnt, val = Counter(), {}
        for c in res.get("observed_conditions") or []:
            k = (c["catalog_item_id"], c["estimate_unit_id"],
                 tuple(sorted(evs.get(c["condition_id"], {}).get("photo_keys") or [])))
            cnt[k] += 1
            val[k] = (c["condition_id"], revs.get(c["condition_id"], {}).get("verdict"))
        return {k: v for k, v in val.items() if cnt[k] == 1}

    k1, k2 = by_key(res1), by_key(res2)
    return [(k1[k][0], k1[k][1], k2[k][1]) for k in sorted(set(k1) & set(k2)) if k1[k][1] != k2[k][1]]


def terra_batch(idx: Dict[str, Any], cid: str) -> Tuple[Optional[int], Optional[int]]:
    """(conditions in the Terra call that reviewed cid, distinct images sent in that call)."""
    call = idx["calls"].get((idx["revs"].get(cid) or {}).get("terra_call_id"))
    if not call:
        return None, None
    cids = call.get("condition_ids") or []
    imgs = set()
    for c in cids:
        ev = idx["evs"].get(c) or {}
        imgs.update(ev.get("representative_photo_keys") or ev.get("photo_keys") or [])
    return len(cids), len(imgs)


# --------------------------------------------------------------------------- cards

def photo_dims(paths: Dict[str, Path]) -> Dict[str, Optional[Tuple[int, int]]]:
    """photo_key -> (width, height); None when the file is missing or undecodable."""
    try:
        from PIL import Image
    except ImportError:
        return {k: None for k in paths}
    out: Dict[str, Optional[Tuple[int, int]]] = {}
    for k, p in paths.items():
        try:
            with Image.open(p) as im:
                out[k] = (int(im.width), int(im.height))
        except Exception:
            out[k] = None
    return out


def all_low_res(dims: Dict[str, Optional[Tuple[int, int]]], keys: Iterable[str]) -> bool:
    """True when every known-size photo in keys is a thumbnail (short side < MIN_EVIDENCE_PX).

    Cards whose entire evidence passes this are excluded from the queue: a human cannot
    judge the claim, and the model was shown the same thumbnail (see
    docs/HANDOFF_thumbnail_photo_ingest_fix.md). Unknown sizes never exclude."""
    known = [dims.get(k) for k in keys if dims.get(k)]
    return bool(known) and all(min(wh) < MIN_EVIDENCE_PX for wh in known)


def card_id(*parts: Any) -> str:
    return "rc_" + hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]


def uniform_included(cid: str, rate: float) -> bool:
    """Stateless hash-threshold sampling: monotone in rate, independent across cards."""
    h = int(hashlib.sha256(cid.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10000) < int(round(rate * 10000))


def _photos(paths: Dict[str, Path], keys: Iterable[str],
            dims: Optional[Dict[str, Optional[Tuple[int, int]]]] = None) -> List[Dict[str, Any]]:
    return [{"key": k, "path": str(paths.get(k) or ""),
             "wh": list((dims or {}).get(k) or []) or None} for k in keys]


def _base(kind: str, source: str, prop: str, run_id: str, key: str) -> Dict[str, Any]:
    return {"card_id": card_id(source, prop, run_id, kind, key), "kind": kind, "phase": None,
            "strata": [], "source": source, "property_key": prop, "run_id": run_id, "address": None,
            "legacy_item_id": None}


def condition_card(*, source: str, prop: str, run_id: str, idx: Dict[str, Any], cid: str,
                   claims: Dict[str, str], paths: Dict[str, Path], f2: Dict[str, Dict[str, Any]],
                   dims: Optional[Dict[str, Any]] = None,
                   replica: Optional[Tuple[Optional[str], Optional[str]]] = None) -> Dict[str, Any]:
    c = idx["conds"][cid]
    r = idx["revs"].get(cid) or {}
    ev = idx["evs"].get(cid) or {}
    d = idx["disps"].get(cid) or {}
    w = idx["work_by_cond"].get(cid)
    all_keys = list(ev.get("photo_keys") or [])
    rep = list(ev.get("representative_photo_keys") or all_keys)
    dups = [k for k in all_keys if k not in rep]
    obs = sorted({str(x.get("observation") or "")[:OBSERVATION_MAX_CHARS]
                  for x in ev.get("evidence_refs") or [] if x.get("observation")})
    f2rec = f2.get(cid) or {}
    f2v = sorted({row[1] for row in f2rec.get("rows", [])})
    dir_ = direction(r.get("verdict"), f2v)
    n_batch, n_imgs = terra_batch(idx, cid)
    card = _base("condition", source, prop, run_id, cid)
    card.update({
        "title": f"{c.get('catalog_item_id')} @ {c.get('estimate_unit_id')}",
        "claim": {"catalog_claim": claims.get(c.get("catalog_item_id"), c.get("catalog_item_id")),
                  "observations": obs},
        "strips": [{"label": "evidence photos (as sent to Terra)", "photos": _photos(paths, rep, dims)}]
                  + ([{"label": "near-duplicates not sent", "muted": True, "photos": _photos(paths, dups, dims)}] if dups else []),
        "meta": {"catalog_item_id": c.get("catalog_item_id"), "catalog_kind": c.get("catalog_kind"),
                 "scene_group": c.get("scene_group"), "estimate_unit_id": c.get("estimate_unit_id"),
                 "condition_id": cid, "photo_count": int(ev.get("distinct_photo_count") or len(all_keys) or 0),
                 "terra_batch_conditions": n_batch, "terra_batch_images": n_imgs,
                 "terra_verdict": r.get("verdict"),
                 "replica_terra_verdict": replica[1] if replica else None,
                 "direction": dir_, "second_opinion": SECOND_OPINION[dir_],
                 "accepted": r.get("verdict") == "supported" and d.get("disposition") == "accepted_for_work",
                 "disposition": d.get("disposition")},
        "reveal": [{"label": "Terra verdict", "text": f"{r.get('verdict')} — {r.get('rationale') or ''}"}]
                  + ([{"label": "Terra verdict, replica 2 (same photos)", "text": str(replica[1])}] if replica else [])
                  + ([{"label": "Pass 2f (v4, package-level)",
                       "text": f"{'/'.join(f2v)} in {', '.join(str(p) for p in f2rec.get('packages') or [])}: "
                               + " | ".join(f2rec.get("evidence") or [])}] if f2rec else
                     [{"label": "Pass 2f (v4, package-level)", "text": "no opinion — this issue never reached a 2f package candidate"}])
                  + [{"label": "v5 disposition", "text": f"{d.get('disposition')} ({d.get('reason_code')})"
                                                         + (f"; work {w.get('action_code')}" if w else "")}],
        "hidden": {"low": w.get("low"), "high": w.get("high")} if w else None,
        "verdict_options": list(CONDITION_VERDICTS),
        "verdict_keys": {str(i + 1): v for i, v in enumerate(CONDITION_VERDICTS)},
        "tags": dict(ERROR_TAGS),
    })
    return card


def _child_lines(idx: Dict[str, Any], cand: Dict[str, Any], paths: Dict[str, Path], dims=None):
    lines, strips = [], []
    drivers = set(cand.get("driver_work_item_ids") or [])
    for wid in cand.get("child_work_item_ids") or []:
        w = idx["works"].get(wid) or {}
        role = "driver" if wid in drivers else "support"
        lines.append(f"{role}: {w.get('action_code')} — {', '.join(w.get('catalog_item_ids') or [])}")
        for cid in w.get("condition_ids") or []:
            cc = idx["conds"].get(cid) or {}
            ev = idx["evs"].get(cid) or {}
            keys = ev.get("representative_photo_keys") or ev.get("photo_keys") or []
            strips.append({"label": f"{role} · {cc.get('catalog_item_id')}", "photos": _photos(paths, keys, dims)})
    return lines, strips


def package_card(*, source: str, prop: str, run_id: str, idx: Dict[str, Any], v4_pk: Dict[str, Any],
                 cand: Dict[str, Any], res: Dict[str, Any], paths: Dict[str, Path],
                 dims: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """v4 2f rejected this package; v5 built the same (type, unit) candidate and Sol decided it."""
    decs = {d["package_candidate_id"]: d for d in res.get("package_decisions") or []}
    apps = {a["package_candidate_id"]: a for a in res.get("package_applications") or []}
    d = decs.get(cand.get("package_candidate_id")) or {}
    a = apps.get(cand.get("package_candidate_id")) or {}
    raw = v4_pk.get("raw_pass_2f_response")
    try:
        j = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        j = {}
    ptype, unit = cand.get("package_type"), cand.get("estimate_unit_id") or ""
    lines, strips = _child_lines(idx, cand, paths, dims)
    review_keys = v4_pk.get("review_photo_keys") or []
    card = _base("package", source, prop, run_id, f"{ptype}|{unit}")
    card.update({
        "title": f"package {ptype} @ {unit}",
        "claim": {"catalog_claim": f"Is a {ptype} package warranted for {unit}, given these children?",
                  "observations": lines},
        "strips": ([{"label": "photos Pass 2f reviewed for the package", "photos": _photos(paths, review_keys, dims)}] if review_keys else []) + strips,
        "meta": {"package_type": ptype, "estimate_unit_id": unit, "child_count": len(cand.get("child_work_item_ids") or []),
                 "v5_tier": cand.get("pricing_tier"), "v5_sol_decision": d.get("decision"),
                 "v5_status": a.get("status"), "v4_2f_status": v4_pk.get("verification_status")},
        "reveal": [{"label": "Sol (v5) decision", "text": f"{d.get('decision')} — {d.get('rationale') or ''}"},
                   {"label": "Pass 2f (v4) package verdict", "text": f"{v4_pk.get('verification_status')} — {j.get('evidence_summary') or v4_pk.get('evidence_summary') or ''}"},
                   {"label": "v5 application", "text": f"{a.get('status')} ({a.get('reason_code')})"}],
        "hidden": {"low": a.get("effective_low") if a else cand.get("low"), "high": a.get("effective_high") if a else cand.get("high")},
        "verdict_options": list(PACKAGE_VERDICTS),
        "verdict_keys": {str(i + 1): v for i, v in enumerate(PACKAGE_VERDICTS)},
        "tags": {},
    })
    return card


def bathroom_card(*, source: str, prop: str, run_id: str, art: Dict[str, Any], idx: Dict[str, Any],
                  res: Dict[str, Any], surr: List[Dict[str, Any]], paths: Dict[str, Path],
                  dims: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    v4 = art.get("renovation_estimate_v4") or {}
    exp = v4.get("bathroom_expansion_audit") or {}
    meta = art.get("property") or {}
    bath_units = sorted({c.get("estimate_unit_id") for c in idx["conds"].values() if "bath" in str(c.get("estimate_unit_id") or "")})
    v5bm = [c for c in res.get("package_candidates") or [] if c.get("package_type") == "bathroom_modernization"]
    v4bm = [p for p in v4.get("packages") or [] if p.get("package_type") == "bathroom_modernization"]
    card = _base("bathroom", source, prop, run_id, "bathrooms")
    card.update({
        "title": f"bathrooms — listing says {meta.get('baths') or meta.get('bath_count')} bath(s)",
        "claim": {"catalog_claim": "How many distinct bathrooms do these photo groups show, and should bathroom work bill per bathroom or once?",
                  "observations": [f"v4 surrogates: {len(surr)}", f"v5 bathroom units: {', '.join(bath_units) or 'none'}"]},
        "strips": [{"label": f"surrogate {s.get('room_surrogate_id')}", "photos": _photos(paths, s.get("photo_keys") or [], dims)} for s in surr],
        "meta": {"listing_baths": meta.get("baths") or meta.get("bath_count"), "surrogates": len(surr),
                 "v5_bath_units": bath_units, "v4_expanded": exp.get("expanded")},
        "reveal": [{"label": "v4 expansion", "text": f"expanded={exp.get('expanded')} qualifying={exp.get('qualifying_surrogate_ids')} fallback={exp.get('fallback_reason')}; {len(v4bm)} bathroom_modernization package(s)"},
                   {"label": "v5", "text": f"{len(v5bm)} bathroom_modernization candidate(s) on units {bath_units}"}],
        "hidden": {"low": sum(p.get("cost_low") or 0 for p in v4bm), "high": sum(p.get("cost_high") or 0 for p in v4bm)},
        "verdict_options": list(BATHROOM_BILLING),
        "verdict_keys": {"p": "per_bathroom", "o": "once", "u": "unsure"},
        "tags": {},
    })
    return card


# --------------------------------------------------------------------------- queue

def _bath_surrogates(v4: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in v4.get("room_surrogates") or []
            if str(s.get("scene_group")) == "bathroom" or "bath" in str(s.get("room_surrogate_id"))]


def build_queue_from_listings(listings: List[Dict[str, Any]], *, uniform_rate: float = DEFAULT_UNIFORM_RATE,
                              packets: Optional[Dict[str, Any]] = None, claims: Optional[Dict[str, str]] = None,
                              address_map: Optional[Dict[str, str]] = None, order: str = "hash") -> Dict[str, Any]:
    """listings: [{source, property_key, run_id, artifact, replica_artifact?}] -> queue dict."""
    packets = packets or {}
    claims = claims if claims is not None else claim_texts()
    legacy = legacy_ids(packets)
    p6 = {(t["property"], t["item"]) for t in packets.get("p6_tier2_zero_items") or []}
    cards: List[Dict[str, Any]] = []
    per_source: Dict[str, Dict[str, Any]] = {}
    for L in listings:
        source, prop, run_id, art = L["source"], L["property_key"], L["run_id"], L["artifact"]
        res = v5_result(art)
        if res is None:
            continue
        st = per_source.setdefault(source, {"listings": 0, "accepted": 0, "accepted_dirA": 0,
                                            "strata": Counter(), "low_res": Counter()})
        st["listings"] += 1
        idx = index_result(res)
        v4 = art.get("renovation_estimate_v4") or {}
        f2 = join_2f(v4, idx)
        paths = photo_key_to_path(art)
        dims = photo_dims(paths)
        flips = {cid: (v1, v2) for cid, v1, v2 in terra_flips(res, v5_result(L.get("replica_artifact")))}
        for cid, c in idx["conds"].items():
            r = idx["revs"].get(cid) or {}
            d = idx["disps"].get(cid) or {}
            dir_ = direction(r.get("verdict"), {row[1] for row in (f2.get(cid) or {}).get("rows", [])})
            accepted = r.get("verdict") == "supported" and d.get("disposition") == "accepted_for_work"
            if all_low_res(dims, (idx["evs"].get(cid) or {}).get("photo_keys") or []):
                st["low_res"]["conditions"] += 1
                if accepted:
                    st["low_res"]["accepted"] += 1  # kept out of the accepted denominators too
                continue
            strata = []
            if dir_ in ("A", "mixed"):
                strata.append("dirA")
            elif dir_ == "B":
                strata.append("dirB")
            if cid in flips:
                strata.append("terra_flip")
            item_unit = f"{c.get('catalog_item_id')}|{c.get('estimate_unit_id')}"
            if (prop, item_unit) in p6:
                strata.append("p6_forced_single")
            if accepted:
                st["accepted"] += 1
                if dir_ in ("A", "mixed"):
                    st["accepted_dirA"] += 1
                elif uniform_included(card_id(source, prop, run_id, "condition", cid), uniform_rate):
                    strata.append("uniform")
            if not strata:
                continue
            card = condition_card(source=source, prop=prop, run_id=run_id, idx=idx, cid=cid, claims=claims,
                                  paths=paths, f2=f2, dims=dims, replica=flips.get(cid))
            card["strata"] = strata
            card["legacy_item_id"] = legacy.get(("condition", prop, cid)) or legacy.get(("p6", prop, item_unit))
            cards.append(card)
        for pk in v4.get("package_candidates") or []:
            if pk.get("verification_status") != "rejected":
                continue
            ptype, unit = pk.get("package_type"), pk.get("estimate_unit_id") or ""
            cands = [x for x in res.get("package_candidates") or [] if x.get("package_type") == ptype and x.get("estimate_unit_id") == unit]
            if not cands:
                continue
            card = package_card(source=source, prop=prop, run_id=run_id, idx=idx, v4_pk=pk, cand=cands[0], res=res, paths=paths, dims=dims)
            if all_low_res(dims, [p["key"] for s in card["strips"] for p in s["photos"]]):
                st["low_res"]["packages"] += 1
                continue
            card["strata"] = ["p1_package"]
            card["legacy_item_id"] = legacy.get(("package", prop, f"{ptype}|{unit}"))
            cards.append(card)
        surr = _bath_surrogates(v4)
        if len(surr) >= 2:
            card = bathroom_card(source=source, prop=prop, run_id=run_id, art=art, idx=idx, res=res, surr=surr, paths=paths, dims=dims)
            if all_low_res(dims, [p["key"] for s in card["strips"] for p in s["photos"]]):
                st["low_res"]["bathrooms"] += 1
                continue
            card["strata"] = ["p3_bathroom"]
            card["legacy_item_id"] = legacy.get(("bathroom", prop))
            cards.append(card)
    for card in cards:
        card["phase"] = min(PHASE_OF[s] for s in card["strata"])
        card["address"] = (address_map or {}).get(card["property_key"])
        for s in card["strata"]:
            per_source[card["source"]]["strata"][s] += 1
    cards = order_cards(cards, order)
    meta = {src: {"listings": v["listings"], "accepted": v["accepted"], "accepted_dirA": v["accepted_dirA"],
                  "accepted_non_dirA": v["accepted"] - v["accepted_dirA"], "strata": dict(v["strata"]),
                  "low_res_excluded": dict(v["low_res"])}
            for src, v in per_source.items()}
    return {"schema_version": 1, "uniform_rate": uniform_rate, "order": order, "meta": meta, "cards": cards}


def order_cards(cards: List[Dict[str, Any]], order: str = "hash") -> List[Dict[str, Any]]:
    """Phase first; inside a phase either hash order (blind interleave) or stratum/item order."""
    if order == "tier":
        pri = {s: i for i, s in enumerate(STRATUM_ORDER)}
        return sorted(cards, key=lambda c: (c["phase"], min(pri[s] for s in c["strata"]),
                                            str(c.get("meta", {}).get("catalog_item_id") or ""), c["property_key"], c["card_id"]))
    return sorted(cards, key=lambda c: (c["phase"], hashlib.sha256(c["card_id"].encode("utf-8")).hexdigest()))


def build_queue(*, canary_root: Optional[Path] = CANARY_ROOT, prod_root: Optional[Path] = PROD_ROOT,
                since: Optional[str] = DEFAULT_SINCE, uniform_rate: float = DEFAULT_UNIFORM_RATE,
                packets_path: Optional[Path] = PACKETS_PATH, fe_db: Optional[Path] = FE_DB,
                order: str = "hash") -> Dict[str, Any]:
    listings: List[Dict[str, Any]] = []
    if canary_root:
        run1, run2 = load_canary(Path(canary_root))
        for prop, (path, art) in sorted(run1.items()):
            listings.append({"source": "canary", "property_key": prop, "run_id": path.parent.name,
                             "artifact": art, "replica_artifact": (run2.get(prop) or (None, None))[1]})
    if prod_root:
        for prop, run_id, _path, art in iter_runs(Path(prod_root), since):
            listings.append({"source": "production", "property_key": prop, "run_id": run_id, "artifact": art})
    addr = addresses(Path(fe_db), [L["property_key"] for L in listings]) if fe_db else {}
    packets = load_packets(Path(packets_path)) if packets_path else {}
    q = build_queue_from_listings(listings, uniform_rate=uniform_rate, packets=packets, address_map=addr, order=order)
    q["generated_from"] = {"canary_root": str(canary_root) if canary_root else None,
                           "prod_root": str(prod_root) if prod_root else None, "since": since}
    return q


# --------------------------------------------------------------------------- verdict store + blind view

def latest_verdicts(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """card_id -> latest record; a record with verdict null (undo) removes the card."""
    out: Dict[str, Dict[str, Any]] = {}
    path = Path(jsonl_path)
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        cid = rec.get("card_id")
        if not cid:
            continue
        if rec.get("verdict") is None:
            out.pop(cid, None)
        else:
            out[cid] = rec
    return out


def blind_card(card: Dict[str, Any]) -> Dict[str, Any]:
    """What the review page may show before a verdict: no model text, no strata, no prices."""
    keep = ("card_id", "kind", "phase", "source", "property_key", "run_id", "address", "title",
            "claim", "strips", "verdict_options", "verdict_keys", "tags")
    out = {k: copy.deepcopy(card.get(k)) for k in keep}  # deep: callers rewrite photo paths into urls
    out["meta"] = {k: v for k, v in (card.get("meta") or {}).items() if k not in BLIND_META}
    return out


def reveal_payload(card: Dict[str, Any]) -> Dict[str, Any]:
    return {"card_id": card["card_id"], "reveal": card.get("reveal") or [], "hidden": card.get("hidden"),
            "strata": card.get("strata") or [], "legacy_item_id": card.get("legacy_item_id"),
            "meta": card.get("meta") or {}}
