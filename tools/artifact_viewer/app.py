import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image

# =============================================================================
# Prompt constants (copied from your scene_classifier_passes.py)
# =============================================================================

PASS_1A_SYSTEM_PROMPT = """You are a real estate image classifier. Your task is to identify the scene type shown in a property photo.

Classify the image into exactly ONE of these categories:
- exterior_front: Front view of the property
- exterior_back: Back/rear view of the property
- exterior_side: Side view of the property
- living_room: Living room or family room
- kitchen: Kitchen area
- bedroom: Bedroom
- bathroom: Bathroom (full or half)
- dining_room: Dining room or eating area
- basement: Basement or cellar
- attic: Attic space
- garage: Garage (interior or exterior)
- yard: Yard, garden, or outdoor space
- pool: Pool or spa area
- roof: Roof view
- hvac: HVAC equipment, water heater, electrical panel
- other: Any other space not listed

Respond with ONLY a JSON object:
{
  "scene": "<category>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}"""
PASS_1A_USER_PROMPT = "Classify the scene type in this real estate photo."

PASS_1B_SYSTEM_PROMPT = (
    "What positive features or upgrades do you see in this photo that a realtor might want to highlight? "
    "Only mention things that are clearly visible. There might not be any positives at all"
)
PASS_1B_USER_PROMPT_TEMPLATE = """Write positives/inventory notes for this {scene} photo in this exact format:

IMPRESSION: <1-2 conservative sentences>
SUMMARY: <1 factual sentence describing what's visible>
FEATURES:
- <feature>
- <feature>
(If none, write FEATURES: - none)
"""

PASS_1C_SYSTEM_PROMPT_TEMPLATE = """You convert FREEFORM positives/inventory notes about a real estate photo into STRICT JSON.

INPUT NOTES:
---
{positives_notes}
---

Rules:
- Use ONLY what is stated in the notes. Do not add new features or claims.
- Keep language conservative and factual.
- If the notes do not contain enough information for overall_impression or image_summary, output "" for that field (do not infer).
- If the notes indicate there are no positives, output an empty notable_features list.
- notable_features must be a list of short strings (2–10 words each), deduplicated.

Output JSON only (no markdown):
{{
  "overall_impression": "<1-2 conservative sentences>",
  "image_summary": "<1-2 factual sentences>",
  "notable_features": ["<feature1>", "<feature2>", ...]
}}
"""
PASS_1C_USER_PROMPT = "Convert the notes into the JSON format."

PASS_2A_SYSTEM_PROMPT = "What issues do you see that a realtor might want to know about? Dont be overdramatic. There might not be any issues at all"
PASS_2A_USER_PROMPT = "Analyze this image for any issues, defects, or concerns."

PASS_2B_SYSTEM_PROMPT_TEMPLATE = """You convert noisy property-photo notes into STRICT JSON. Be conservative and high-signal.

Scene: {scene}
Catalog IDs: {catalog_list_str}

Notes:
{freeform_notes}

Catalog reference:
{catalog_text}

Rules:
- Output ONLY the most important visible defects.
- Do NOT turn "not visible" into "missing" (e.g., smoke detector not seen ≠ missing).
- Do NOT speculate. If wording is "may/might/could/possible" without a clearly described visible defect → present="uncertain", severity="none".
- If notes say no issues → issues_natural_language = [] and all catalog_flags present="no", severity="none".

Output JSON only (no markdown):
{{
  "issues_natural_language": [
    {{
      "description": "calm, factual, only what is visibly described in the notes",
      "rough_category": "cosmetic|moisture|structure|systems|exterior|opportunity",
      "location_hint": "where"
    }}
  ],
  "catalog_flags": {{
    "<issue_id>": {{
      "present": "yes|no|uncertain",
      "severity": "none|minor_repair|moderate_repair|full_replacement",
      "evidence": "short quote or empty"
    }}
  }}
}}

Catalog_flags requirements:
- Include EVERY issue_id from {catalog_list_str}.
- If present != "yes" → severity MUST be "none".
- Only use moderate_repair/full_replacement when notes clearly imply substantial work.
"""
PASS_2B_USER_PROMPT = "Convert the notes into the JSON format."

PASS_3_SYSTEM_PROMPT_TEMPLATE = """You generate object-detection keywords for a real estate photo using ONLY provided extracted facts.

Scene: {scene}

Positives (visible features):
{notable_features_json}

Issues (visible concerns):
{issues_json}

Rules:
- Keywords must be visually detectable objects/materials (1–4 words).
- No speculation and no code/safety claims.
- Deduplicate and keep high-signal. Max 20 keywords.

Output JSON only:
{{
  "keywords": ["<keyword1>", "<keyword2>", ...],
  "categories": {{
    "structural": ["<kw>", ...],
    "fixtures": ["<kw>", ...],
    "condition": ["<kw>", ...],
    "style": ["<kw>", ...]
  }}
}}
"""
PASS_3_USER_PROMPT = "Generate detection keywords."

PASS_4_SYSTEM_PROMPT = """You are a real estate investment analyst synthesizing property photo notes.

You will be given:
- POSITIVES NOTES: freeform positives/inventory notes from multiple photos
- ISSUES NOTES: freeform issues/concerns notes from multiple photos

Rules:
- Use ONLY what is explicitly stated in the notes. Do not add new features, issues, or assumptions.
- Keep it balanced: strengths + risks.
- Be conservative; avoid strong claims unless clearly supported by the notes.

Respond with ONLY a JSON object:
{
  "property_summary": "<2-3 sentence investment-focused summary grounded in the notes>",
  "investment_considerations": ["<fact-based point1>", "<fact-based point2>", ...],
  "estimated_condition": "excellent|good|fair|poor",
  "confidence": <0.0-1.0>
}
"""

PASS_4A_SYSTEM_PROMPT = """You generate conservative room-group summaries for a property photo analysis.

You will receive per-photo extracted facts (scene, positives, issues).
Rules:
- Be factual, conservative, and brief.
- Do NOT add new issues or features.
- If there is no evidence for a room group, output an empty string for that group.

Return ONLY JSON:
{
  "room_summaries": {
    "kitchen": "<1-3 sentences or ''>",
    "bathroom": "<1-3 sentences or ''>",
    "bedroom": "<1-3 sentences or ''>",
    "living_areas": "<1-3 sentences or ''>",
    "utility": "<1-3 sentences or ''>",
    "exterior": "<1-3 sentences or ''>",
    "other": "<1-3 sentences or ''>"
  }
}
"""

PASS_4B_SYSTEM_PROMPT = """You generate concise UI card fields for a property analysis.

Rules:
- Conservative, buyer/investor-friendly.
- Use ONLY what is provided. Do not invent issues/features.
- Keep it short.

Return ONLY JSON:
{
  "overall_condition": "excellent|good|fair|poor",
  "overall_summary": "<Provide about a paragraph summarizng the >",
  "investment_verdict": "buy|maybe|pass",
  "investment_rationale": "<1-3 sentences>",
  "renovation_scope": "light|moderate|heavy",
  "renovation_priorities": ["<short>", "..."],
  "risk_flags": ["<short>", "..."],
  "deferred_maintenance": ["<short>", "..."]
}
"""

# =============================================================================
# Helpers
# =============================================================================

KNOWN_META_KEYS = {
    "run_id", "job_id", "property_key", "timestamp", "created_at", "artifacts_dir",
    "detection_backend", "analysis_profile", "used_pass_architecture", "pass_toggles",
    "model_overrides", "model", "gpt_model", "prompt_version", "scene_policy_version",
}
PROPERTY_SECTION_KEYS = {"property_pass4", "property_pass4a", "property_pass4b", "property_summary", "renovation_needs"}


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_read(path: str) -> Optional[str]:
    try:
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return None


def is_rollup_group(v: Any) -> bool:
    return isinstance(v, dict) and "scenes_included" in v and "image_keys" in v


def get_scene(photo: dict) -> str:
    s = (photo.get("scene") or "").strip()
    if s:
        return s
    # try pass 1a
    p = (photo.get("passes") or {}).get("1a") or {}
    return str(p.get("scene") or "property")


def get_issues_nl(photo: dict) -> List[dict]:
    # prefer flattened
    issues = photo.get("issues_natural_language")
    if isinstance(issues, list):
        return issues
    # fallback: pass 2b
    p2b = (photo.get("passes") or {}).get("2b") or {}
    issues = p2b.get("issues_natural_language")
    return issues if isinstance(issues, list) else []


def get_catalog_flags(photo: dict) -> Dict[str, Any]:
    cf = photo.get("catalog_flags")
    if isinstance(cf, dict) and cf:
        return cf
    p2b = (photo.get("passes") or {}).get("2b") or {}
    cf = p2b.get("catalog_flags")
    return cf if isinstance(cf, dict) else {}


def build_catalog_from_issue_catalog(issue_catalog: dict) -> Tuple[List[dict], List[str], str]:
    defect_items = (issue_catalog or {}).get("defect_issues", []) or []
    opp_items = (issue_catalog or {}).get("opportunity_flags", []) or []

    all_items: List[dict] = []
    for x in list(defect_items) + list(opp_items):
        if isinstance(x, dict) and x.get("id"):
            all_items.append(x)

    ids = [i["id"] for i in all_items]
    catalog_text = "\n".join(f"- {i.get('id')}: {i.get('name', '')}" for i in all_items)
    return all_items, ids, catalog_text


def render_prompts_for_pass(
    pass_id: str,
    run: dict,
    photo_key: Optional[str],
    issue_catalog: Optional[dict],
) -> Tuple[str, str, str]:
    """
    Returns (system_prompt, user_prompt, notes_about_reconstruction)
    """
    notes = []
    photos = run.get("photos") or {}
    photo = photos.get(photo_key) if photo_key else None

    if pass_id == "1a":
        return PASS_1A_SYSTEM_PROMPT, PASS_1A_USER_PROMPT, ""

    if pass_id == "1b":
        scene = get_scene(photo or {})
        user = PASS_1B_USER_PROMPT_TEMPLATE.format(scene=scene)
        return PASS_1B_SYSTEM_PROMPT, user, ""

    if pass_id == "1c":
        pos = ((photo or {}).get("positives_notes") or "").strip()
        sys = PASS_1C_SYSTEM_PROMPT_TEMPLATE.format(positives_notes=pos or "(no positives_notes captured)")
        return sys, PASS_1C_USER_PROMPT, ""

    if pass_id == "2a":
        return PASS_2A_SYSTEM_PROMPT, PASS_2A_USER_PROMPT, ""

    if pass_id == "2b":
        scene = get_scene(photo or {})
        freeform = ((photo or {}).get("issues_notes") or "").strip()

        # Catalog rendering:
        catalog_ids: List[str] = []
        catalog_text: str = ""
        if issue_catalog:
            _, catalog_ids, catalog_text = build_catalog_from_issue_catalog(issue_catalog)
        else:
            # best-effort: infer from output flags (works well when captured)
            flags = get_catalog_flags(photo or {})
            catalog_ids = sorted(list(flags.keys()))
            catalog_text = "\n".join(f"- {i}: (name unavailable in viewer)" for i in catalog_ids)
            notes.append("Catalog names not available (pass --issue-catalog to render full catalog_text).")

        catalog_list_str = ", ".join(catalog_ids) if catalog_ids else "(none)"

        sys = PASS_2B_SYSTEM_PROMPT_TEMPLATE.format(
            scene=scene,
            catalog_list_str=catalog_list_str,
            freeform_notes=freeform or "(no notes provided)",
            catalog_text=catalog_text or "(no catalog provided)",
        )
        return sys, PASS_2B_USER_PROMPT, " ".join(notes)

    if pass_id == "3":
        scene = get_scene(photo or {})
        notable = (photo or {}).get("notable_features") or []
        issues = get_issues_nl(photo or {})

        sys = PASS_3_SYSTEM_PROMPT_TEMPLATE.format(
            scene=scene,
            notable_features_json=json.dumps(notable, ensure_ascii=False),
            issues_json=json.dumps(issues, ensure_ascii=False),
        )
        return sys, PASS_3_USER_PROMPT, ""

    # property-level “passes”
    if pass_id == "4":
        # Best-effort: reconstruct from photo_intel photos in key order
        photo_keys = sorted(list((run.get("photos") or {}).keys()))
        positives_blocks = []
        issues_blocks = []
        for k in photo_keys[:20]:
            p = (run.get("photos") or {}).get(k) or {}
            scene = get_scene(p)
            pos = (p.get("positives_notes") or "").strip()
            neg = (p.get("issues_notes") or "").strip()
            if pos:
                positives_blocks.append(f"- {k} ({scene}): {pos}")
            if neg:
                issues_blocks.append(f"- {k} ({scene}): {neg}")

        user = (
            "POSITIVES NOTES:\n---\n"
            + "\n".join(positives_blocks)
            + "\n---\n\nISSUES NOTES:\n---\n"
            + "\n".join(issues_blocks)
            + "\n---\n"
            + f"\nTotal images analyzed: {len(run.get('photos') or {})}"
        )
        notes.append("Reconstructed prompt: ordering may differ from runtime (dict/list ordering).")
        return PASS_4_SYSTEM_PROMPT, user, " ".join(notes)

    if pass_id == "4a":
        # Reconstruct the user payload that your pass4a builds
        SCENE_GROUPS_UI = {
            "kitchen": ["kitchen", "pantry"],
            "bathroom": ["bathroom"],
            "bedroom": ["bedroom", "closet"],
            "living_areas": ["living_room", "dining_room", "home_office", "hallway", "stairway"],
            "utility": ["laundry_room", "basement", "attic", "garage", "hvac"],
            "exterior": ["exterior_front", "exterior_back", "exterior_side", "yard", "patio", "deck", "balcony", "driveway", "pool", "garden"],
            "other": ["roof", "other", "unknown", "floor_plan", "aerial_view", "street_view"],
        }
        scene_to_group = {}
        for g, scenes in SCENE_GROUPS_UI.items():
            for s in scenes:
                scene_to_group[s] = g

        groups = {k: [] for k in SCENE_GROUPS_UI.keys()}
        issues_by_category: Dict[str, int] = {}
        total_issues_found = 0

        for k in sorted(list((run.get("photos") or {}).keys())):
            p = (run.get("photos") or {}).get(k) or {}
            scene = str(p.get("scene") or "unknown").strip()
            group = scene_to_group.get(scene, "other")

            pos = str(p.get("positives_notes") or "").strip()
            neg = str(p.get("issues_notes") or "").strip()

            if pos:
                groups[group].append(f"- {k} ({scene}) POS: {pos}")
            if neg:
                groups[group].append(f"- {k} ({scene}) ISSUES: {neg}")

            # deterministic counts from issues_natural_language
            for it in get_issues_nl(p):
                total_issues_found += 1
                cat = str(it.get("rough_category") or "other").strip() or "other"
                issues_by_category[cat] = issues_by_category.get(cat, 0) + 1

        for g in groups:
            groups[g] = groups[g][:30]

        user_payload = {
            "scene_counts": {},  # not in artifact unless you store it; safe empty
            "grouped_notes": groups,
            "total_images_analyzed": len(run.get("photos") or {}),
            "total_issues_found": total_issues_found,
            "issues_by_category": issues_by_category,
        }
        return PASS_4A_SYSTEM_PROMPT, json.dumps(user_payload, ensure_ascii=False), "Reconstructed payload; scene_counts omitted (not stored)."

    if pass_id == "4b":
        # Use what’s already in property_pass4a if present, else compute minimal
        p4a = run.get("property_pass4a") or {}
        room_summaries = p4a.get("room_summaries") or {}
        issues_by_category = p4a.get("issues_by_category") or {}
        total_issues_found = int(p4a.get("total_issues_found") or 0)
        total_images = len(run.get("photos") or {})

        user_payload = {
            "room_summaries": room_summaries,
            "total_issues_found": total_issues_found,
            "total_images_analyzed": total_images,
            "issues_by_category": issues_by_category,
        }
        return PASS_4B_SYSTEM_PROMPT, json.dumps(user_payload, ensure_ascii=False), ""

    return "", "", "Unknown pass id"


def coverage(photo: dict) -> dict:
    passes_run = set(photo.get("passes_run") or [])
    passes = photo.get("passes") or {}
    present = set(passes.keys())
    missing = sorted(passes_run - present)
    extra = sorted(present - passes_run)
    empty = sorted([k for k, blk in passes.items() if blk in (None, {}, [])])
    return {
        "passes_run": sorted(passes_run),
        "passes_present": sorted(present),
        "missing_blocks": missing,
        "extra_blocks": extra,
        "empty_blocks": empty,
        "error": photo.get("error"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo-intel", required=True)
    ap.add_argument("--property-summary")
    ap.add_argument("--issue-catalog")
    args, _unknown = ap.parse_known_args()

    run = load_json(args.photo_intel)
    prop_summary = load_json(args.property_summary) if args.property_summary else None
    issue_catalog = load_json(args.issue_catalog) if args.issue_catalog else None

    photos = run.get("photos") or {}
    rollups = {k: v for k, v in run.items() if k not in KNOWN_META_KEYS and k not in PROPERTY_SECTION_KEYS and k != "photos" and is_rollup_group(v)}
    property_sections = {k: run.get(k) for k in PROPERTY_SECTION_KEYS if run.get(k) is not None}

    st.set_page_config(layout="wide", page_title="RealtorVision Artifact Viewer")
    st.title("RealtorVision Artifact Viewer (with prompts)")

    # Sidebar filters
    st.sidebar.header("Photos")
    only_issues = st.sidebar.checkbox("Only photos with issues_natural_language", value=False)
    only_missing_passes = st.sidebar.checkbox("Only photos missing pass blocks", value=False)
    only_errors = st.sidebar.checkbox("Only photos with error != null", value=False)
    search = st.sidebar.text_input("Search in selected photo JSON", "")

    photo_names = sorted(photos.keys())
    filtered = []
    for name in photo_names:
        p = photos[name]
        if only_issues and not get_issues_nl(p):
            continue
        if only_missing_passes and not coverage(p)["missing_blocks"]:
            continue
        if only_errors and not p.get("error"):
            continue
        filtered.append(name)

    st.sidebar.caption(f"{len(filtered)} / {len(photo_names)} photos shown")
    sel = st.sidebar.selectbox("Select photo", filtered if filtered else photo_names)

    colL, colR = st.columns([1, 2])

    p = photos.get(sel) if sel else None
    if not p:
        st.warning("No photo selected.")
        return

    with colL:
        st.subheader(sel)
        img_path = p.get("image_path")
        if img_path and Path(img_path).exists():
            st.image(Image.open(img_path), use_container_width=True)
        else:
            st.info("Image not found at image_path.")

        st.markdown("### Quick fields")
        st.write(f"**scene:** {get_scene(p)}")
        st.write(f"**models_used:** {p.get('models_used')}")
        st.write(f"**processing_time:** {p.get('processing_time')}")

        st.markdown("### Coverage")
        st.json(coverage(p))

    with colR:
        tabs = st.tabs(["Overview", "Pass Viewer (with prompt)", "Raw Photo JSON", "Rollups", "Property Sections", "property_summary.json"])

        with tabs[0]:
            st.write(f"**overall_impression:** {p.get('overall_impression')}")
            st.write(f"**image_summary:** {p.get('image_summary')}")
            st.write("**notable_features:**")
            st.write(p.get("notable_features") or [])

            st.markdown("### positives_notes (1b)")
            st.code((p.get("positives_notes") or "").strip(), language="markdown")

            st.markdown("### issues_notes (2a)")
            st.code((p.get("issues_notes") or "").strip(), language="markdown")

            st.markdown("### issues_natural_language (2b)")
            st.json(get_issues_nl(p))

            st.markdown("### catalog_flags (2b)")
            st.json(get_catalog_flags(p))

        with tabs[1]:
            pass_id = st.selectbox("Pass", ["1a", "1b", "1c", "2a", "2b", "3", "4", "4a", "4b"])
            sys_p, usr_p, recon_notes = render_prompts_for_pass(pass_id, run, sel, issue_catalog)

            st.markdown("### System prompt")
            st.code(sys_p or "(not available)", language="markdown")

            st.markdown("### User prompt")
            st.code(usr_p or "(not available)", language="markdown")

            if recon_notes:
                st.info(recon_notes)

            st.markdown("### Stored output (what your artifact captured)")
            if pass_id == "1a":
                st.json((p.get("passes") or {}).get("1a") or {"scene": p.get("scene")})
            elif pass_id == "1b":
                st.code((p.get("positives_notes") or "").strip(), language="markdown")
            elif pass_id == "1c":
                # Structured positives
                out = {
                    "overall_impression": p.get("overall_impression"),
                    "image_summary": p.get("image_summary"),
                    "notable_features": p.get("notable_features"),
                }
                st.json(out)
            elif pass_id == "2a":
                st.code((p.get("issues_notes") or "").strip(), language="markdown")
            elif pass_id == "2b":
                st.json({
                    "issues_natural_language": get_issues_nl(p),
                    "catalog_flags": get_catalog_flags(p),
                })
            elif pass_id == "3":
                st.json({
                    "keywords": p.get("keywords") or [],
                    "passes.3": (p.get("passes") or {}).get("3"),
                })
            elif pass_id == "4":
                st.json(run.get("property_pass4") or run.get("property_summary") or {})
            elif pass_id == "4a":
                st.json(run.get("property_pass4a") or {})
            elif pass_id == "4b":
                st.json(run.get("property_pass4b") or {})

            if search:
                blob = json.dumps(p, ensure_ascii=False, indent=2)
                if search.lower() not in blob.lower():
                    st.warning(f"Search '{search}' not found in selected photo JSON.")

        with tabs[2]:
            st.json(p)

        with tabs[3]:
            if not rollups:
                st.info("No rollup groups found.")
            else:
                grp = st.selectbox("Rollup group", sorted(rollups.keys()))
                st.json(rollups[grp])

        with tabs[4]:
            if not property_sections:
                st.info("No property sections found in photo_intel.json.")
            else:
                sec = st.selectbox("Property section", sorted(property_sections.keys()))
                st.json(property_sections[sec])

        with tabs[5]:
            if prop_summary is None:
                st.info("No property_summary.json loaded.")
            else:
                st.json(prop_summary)


if __name__ == "__main__":
    main()
