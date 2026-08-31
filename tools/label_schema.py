"""The v1.1 label schema: two axes, their classes, and the scoring policy.

Single source of truth for `scripts/build_adjudication_queue.py`,
`scripts/build_labels_v1_1.py` and `scripts/score_redecide_variants.py`.
Every table here is pre-committed in `docs/DESIGN_label_v1_1_adjudication.md`
(§§3-5), written before any adjudication verdict existed. Change a table only
by changing the design doc in the same commit.

Why two axes: the v1 review recorded one label for three different judgments —
is the condition there, is the exact claim accurate, is the work worth doing.
The frozen notes show all three colliding ("slightly dated but not worth
replacing" was recorded as `unsupported`). v1.1 separates them so Session F
tunes wording against wording and severity against severity.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

LABEL_VERSION = "v1.1"

# --------------------------------------------------------------------- schema

# key -> slug. One keypress per card; the axes below decode it.
ADJUDICATION_KEYS: Dict[str, str] = {
    "1": "exact_and_warranted",
    "2": "misnamed_but_warranted",
    "3": "exact_but_trivial",
    "4": "misnamed_and_trivial",
    "5": "wrong_object_or_place",
    "6": "absent",
    "7": "inconclusive",
}

# Shown to the reviewer next to each key.
ADJUDICATION_HELP: Dict[str, str] = {
    "exact_and_warranted": "the claim is true as worded, and work is warranted",
    "misnamed_but_warranted": "a real problem is there, this claim names it wrong (stains called scuffs)",
    "exact_but_trivial": "the claim is true, but it is not worth billing",
    "misnamed_and_trivial": "something is there, named wrong, and not worth billing",
    "wrong_object_or_place": "the problem is real somewhere, but not this object/room",
    "absent": "nothing of this kind is there at all",
    "inconclusive": "the evidence cannot settle it",
}

CLAIM_AXIS: Dict[str, str] = {
    "exact_and_warranted": "exact",
    "misnamed_but_warranted": "misnamed",
    "exact_but_trivial": "exact",
    "misnamed_and_trivial": "misnamed",
    "wrong_object_or_place": "absent",
    "absent": "absent",
    "inconclusive": "inconclusive",
}

WORK_AXIS: Dict[str, str] = {
    "exact_and_warranted": "warranted",
    "misnamed_but_warranted": "warranted",
    "exact_but_trivial": "trivial",
    "misnamed_and_trivial": "trivial",
    "wrong_object_or_place": "none",
    "absent": "none",
    "inconclusive": "inconclusive",
}

# Lossy, for side-by-side reporting only — never feed back into a v1.1 result.
V1_PROJECTION: Dict[str, str] = {
    "exact": "terra_claim_supported",
    "misnamed": "terra_claim_overstated",
    "absent": "terra_claim_unsupported",
    "inconclusive": "terra_evidence_inconclusive",
}

# What each re-tag answer implies. Validation only: the 46 re-tag cards are
# re-asked under this schema, never translated. Disagreement is a finding.
RETAG_EXPECTATION: Dict[str, Dict[str, str]] = {
    "wholly_false": {"claim": "absent"},
    "mechanism_only": {"claim": "misnamed"},
    "too_trivial": {"work": "trivial"},
    "cannot_tell": {"claim": "inconclusive"},
    "terra_miss": {"claim": "exact"},
    "wording_blocked": {"claim": "misnamed"},
    "borderline": {},
}


def slug_of(verdict: Optional[str]) -> Optional[str]:
    """`mechanism_only: a real problem...` -> `mechanism_only`.

    The review page renders the stored verdict string as its own button label,
    so both vocabularies store `slug: help text` and recover the slug here.
    """
    if not verdict:
        return None
    return str(verdict).split(":", 1)[0].strip()


retag_slug = slug_of  # the re-tag vocabulary uses the same `slug: help` shape


# -------------------------------------------------------------------- classes

BILLED_CLASSES = ("hard_false_billed", "supported_billed", "misnamed_billed",
                  "trivial_billed")
DIRB_CLASSES = ("dirB_recovery", "dirB_wording_recovery", "dirB_trivial",
                "dirB_terra_correct")
V1_1_CLASSES = BILLED_CLASSES + DIRB_CLASSES

# The v1 classes, kept so a v1-labelled run scores exactly as it did before.
V1_CLASSES = ("hard_false_billed", "supported_billed", "overstated_billed",
              "dirB_recovery")


def is_billed(strata: Sequence[str], accepted: Any) -> bool:
    return bool(accepted) and bool(set(strata or ()) & {"dirA", "uniform"})


def classify_v1(strata: Sequence[str], accepted: Any,
                verdict: Optional[str]) -> str:
    """The frozen v1 rule, mirrored from the Session B scorer."""
    if "dirB" in set(strata or ()):
        return ("dirB_recovery" if verdict == "terra_claim_supported"
                else "dirB_other")
    if is_billed(strata, accepted):
        return {
            "terra_claim_unsupported": "hard_false_billed",
            "terra_claim_supported": "supported_billed",
            "terra_claim_overstated": "overstated_billed",
        }.get(verdict or "", "other_billed")
    return "out_of_scope"


def classify_v1_1(strata: Sequence[str], accepted: Any,
                  slug: Optional[str]) -> str:
    """Design doc §4. `excluded` drops the card from the scored population."""
    claim = CLAIM_AXIS.get(slug or "")
    work = WORK_AXIS.get(slug or "")
    if claim is None:
        return "unlabelled"
    if claim == "inconclusive":
        return "excluded"
    dirb = "dirB" in set(strata or ())
    if dirb:
        if claim == "absent":
            return "dirB_terra_correct"      # Terra was right to reject it
        if work == "trivial":
            return "dirB_trivial"
        return ("dirB_recovery" if claim == "exact"
                else "dirB_wording_recovery")
    if not is_billed(strata, accepted):
        return "out_of_scope"
    if claim == "absent":
        return "hard_false_billed"
    if work == "trivial":
        return "trivial_billed"              # severity lever, not wording
    return ("supported_billed" if claim == "exact" else "misnamed_billed")


# -------------------------------------------------------------- scoring policy

# class -> (event, outcome when the arm keeps the claim text,
#                  outcome when the arm rewrites/coarsens the claim text)
#
# event `flip_away` fires when the arm's verdict is no longer `supported`;
# `flip_to` fires when it becomes `supported`.
#
# The two arm-dependent rows are the point of the split. A `misnamed` claim is
# literally wrong but sits on a real condition: an arm that leaves the wording
# alone gets no credit for keeping it (dropping it is a cost — the fix is
# wording), while an arm that coarsens the text is being asked about a
# different, now-true claim, so acceptance is the intended behaviour.
SCORING: Dict[str, Tuple[str, str, str]] = {
    "hard_false_billed":     ("flip_away", "win", "win"),
    "supported_billed":      ("flip_away", "loss", "loss"),
    "misnamed_billed":       ("flip_away", "cost", "neutral"),
    "trivial_billed":        ("flip_away", "neutral", "neutral"),
    "dirB_recovery":         ("flip_to", "win", "win"),
    "dirB_wording_recovery": ("flip_to", "neutral", "win"),
    "dirB_trivial":          ("flip_to", "neutral", "neutral"),
    "dirB_terra_correct":    ("flip_to", "loss", "loss"),
}

# The v1 policy, unchanged, so `--labels` absent reproduces Session B exactly.
SCORING_V1: Dict[str, Tuple[str, str, str]] = {
    "hard_false_billed":  ("flip_away", "win", "win"),
    "supported_billed":   ("flip_away", "loss", "loss"),
    "overstated_billed":  ("flip_away", "cost", "cost"),
    "dirB_recovery":      ("flip_to", "win", "win"),
}


def outcome_for(policy: Mapping[str, Tuple[str, str, str]], klass: str,
                new_verdict: Optional[str], *, claim_text_changed: bool
                ) -> Optional[str]:
    """`win` / `loss` / `cost` / `neutral`, or None when the event didn't fire."""
    rule = policy.get(klass)
    if rule is None or new_verdict is None:
        return None
    event, static_outcome, coarsened_outcome = rule
    fired = ((event == "flip_away" and new_verdict != "supported")
             or (event == "flip_to" and new_verdict == "supported"))
    if not fired:
        return None
    return coarsened_outcome if claim_text_changed else static_outcome
