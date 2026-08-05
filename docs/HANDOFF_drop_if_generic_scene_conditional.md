# Handoff: scene-conditional `drop_if_generic`

## Why

Today's `drop_if_generic` flag at Pass 2e ([tools/scene_classifier_passes.py:1509-1513](../tools/scene_classifier_passes.py#L1509)) is a **global, scene-unconditional kill switch** — any issue whose resolved catalog item has the flag is suppressed from `verified_issues` regardless of scene or whether a more-specific item also matched.

This makes the flag unsafe to apply to cross-room generics like `visible_mold_or_mildew` or `older_flooring_style`, because suppression cascades to all their scene groups (bedroom mold, garage flooring, etc.), not just the scenes where a competing specific catalog item exists.

The desired semantics:

> Drop the generic only when a more-specific catalog item also resolved within the same photo/scene.

Then `drop_if_generic: true` on `visible_mold_or_mildew` would let bathroom-specific mold items win in bathrooms while still letting the generic fire in bedrooms / garages / utility rooms where no specific exists.

## Current state (as of 2026-05-21)

- [tools/scene_classifier_passes.py:440-452](../tools/scene_classifier_passes.py#L440) — `is_generic_resolution_candidate`, `prioritize_resolution_candidates`. The flag is consulted during **Pass 2c/2d retrieval scoring**; specific candidates are preferred over generic at scoring time. Working as intended.
- [tools/scene_classifier_passes.py:1509-1513](../tools/scene_classifier_passes.py#L1509) — `_2e_policy_reason` Gate 1. **Unconditional kill switch at Pass 2e. This is the problem site.**
- [tools/issue_catalog.json](../tools/issue_catalog.json) — the four catalog generics that were briefly flagged `drop_if_generic: true` have been rolled back to `false` (see `reactive-sparking-llama` plan, Step 1). The 5 pre-existing `drop_if_generic: true` items predate this work (tier=`optional`, mostly cosmetic) and were not in scope for the rollback.

## Catalog items that should be re-flagged once the implementation is scene-conditional

Once the scene-conditional behavior ships, you can safely re-apply `drop_if_generic: true` to these (review each before flipping):

| Catalog ID | Scene groups | Why flagging it makes sense | Risk if mis-flagged |
|---|---|---|---|
| `outdated_bathroom_finishes` | bathroom only | Bathroom-specific upgrade items now cover the dated-finish space; the generic should defer to them. Lowest risk to re-flag first since the scene group is bathroom-only. | If bathroom-specifics don't fully cover an observation, the generic suppression leaves a gap. |
| `visible_mold_or_mildew` | kitchen, bathroom, bedroom, living_areas, utility, exterior, other, pool | Bathroom-specific mold catalog item exists; generic should defer in bathrooms but still fire in bedrooms / utility. | Without scene-conditional behavior, suppresses mold detection in 7 non-bath scenes. |
| `older_flooring_style` | kitchen, bathroom, bedroom, living_areas, utility | Defer to room-specific flooring catalog items when they exist; generic still useful for "older flooring everywhere" calls. | Massive — losing this globally kills older-flooring detection. |
| `plumbing_fixture_leaking_stained` | kitchen, bathroom, utility, pool | Defer to bathroom-specific plumbing items in bathroom scenes; still useful in kitchen / utility / pool. | Significant — loses plumbing-leak detection in 3 non-bath scenes. |

## Desired implementation sketch

**Option A — Track competing specifics through to Pass 2e (recommended)**:

1. At Pass 2d resolution time, when an observation has multiple plausible matches, record whether a non-generic (`is_generic_resolution_candidate == False`) candidate was retrieved alongside the chosen generic. Attach a `competing_specific_resolved: bool` field to the issue dict before passing it to Pass 2e.
2. In `_2e_policy_reason` Gate 1, change to:

   ```python
   if meta.get("drop_if_generic") and issue.get("competing_specific_resolved"):
       return "drop_if_generic"
   ```

3. The flag now drops only when there's a competing specific in scope; otherwise the generic fires normally.

**Open question**: "competing specific" needs careful definition. Same photo? Same photo + same observation cluster? Same scene type? The right answer depends on how Pass 2d organizes its work — probably "same photo / same observation cluster" but this needs research at implementation time.

**Option B — Compute competing-specific at Pass 2e from the matched_issues list**:

Pass 2e already has the full `matched_issues` list before policy gating. It could compute per-issue whether any other matched issue from the same photo resolved to a non-generic item.

- Simpler in plumbing (no Pass 2d changes), but expensive: O(n²) over matched issues per gate evaluation. Cache the per-photo specific-resolved set up front to bring it down to O(n).

**Option C — Push the dedup into Pass 2d**:

Don't even let generic candidates resolve when a specific is in scope for the same observation. Cleaner semantically but invasive — requires reshaping Pass 2d's candidate selection logic.

Pick A or B. Avoid C unless retrieval needs reshaping for other reasons.

## Risks

- Tests that assert the current unconditional behavior will need updating. Find them with: `rg "drop_if_generic" tests/`. There may also be implicit assumptions in orchestration tests.
- Properties with both bathroom-specific and generic catalog items resolving for the same observation will see the generic drop where they previously co-fired. That's the intended behavior, but worth verifying with a live audit comparison (before/after).
- Watch for orchestration code that reads `verified_issues` and expects no `drop_if_generic` items. If anything was implicitly relying on the unconditional drop, it'll now see those items in non-bathroom scenes.

## Verification plan

1. **Unit tests**:
   - A non-bathroom mold finding (e.g., bedroom) with `visible_mold_or_mildew` resolved and NO competing specific → survives Pass 2e
   - Same finding in a bathroom with `bathroom_mold_<specific>` also resolved → suppressed
   - A bathroom mold finding with `visible_mold_or_mildew` resolved but no bathroom-specific catalog match → survives (degenerate but correct)
2. **Live audit** on a bedroom-mold property and a bathroom-mold property; diff the `final_issues` before/after the implementation.
3. **Re-apply `drop_if_generic: true`** to whichever of the 4 catalog items it's actually safe for (probably `outdated_bathroom_finishes` first, since it's bathroom-only).
4. **No-regression check** on properties that exercise the 5 existing `drop_if_generic: true` items (tier=optional items at JSON lines 3332, 3369, 3542, 3706, 4376).

## Files (most likely to touch)

- [tools/scene_classifier_passes.py](../tools/scene_classifier_passes.py) — main change site (Gate 1 of `_2e_policy_reason`, plus possibly Pass 2d resolution to attach `competing_specific_resolved`)
- [tools/scene_classifier_orchestrator.py](../tools/scene_classifier_orchestrator.py) — if the competing-specific tracking needs to flow through orchestration (it does in Option A)
- `tests/test_scene_classifier_passes*.py` — existing `drop_if_generic` assertions to update; new conditional-behavior tests to add
- [tools/issue_catalog.json](../tools/issue_catalog.json) — re-enable `drop_if_generic: true` on appropriate items once the implementation lands

## Out of scope for this follow-up

- New cross-room generics
- `defaultHidden` semantics (separate flag, separate purpose)
- Pass 2c/2d candidate scoring logic (only Pass 2e gate behavior changes)
- Adding new bathroom or kitchen-specific catalog items to fill gaps left by generics

## Origin

This handoff was extracted from the `reactive-sparking-llama` plan (see `~/.claude/plans/reactive-sparking-llama.md`) where the catalog rollback was sequenced before the rehab_packages.py refactor. The rollback explanation in that plan's Context section provides additional background.
