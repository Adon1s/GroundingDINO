"""
renovation_costs.py
-------------------

Centralized renovation cost assumptions for RealtorVision.

Each entry in RENOVATION_COST_TABLE maps an issue_id from issue_catalog.json
to a dict of severity-band -> (low_cost, high_cost) in USD.

Severity bands are aligned with SEVERITY_RANK in `auto_analyzer.py`:

    - "minor_repair"      : localized / small-scope work
    - "moderate_repair"   : multiple areas or mid-sized project
    - "full_replacement"  : whole-system or major renovation

These are intentionally approximate, meant for investor-level “order of
magnitude” budgeting, *not* for quoting real jobs. Adjust for your
region and property type.
"""

from __future__ import annotations
from typing import Dict, Tuple

Currency = float
CostRange = Tuple[Currency, Currency]

# Optional helper mapping from catalog numeric severity (1–4) to
# our repair severity bands. You can use this in your scene classifier
# when you start emitting per-flag "severity" in photo_intel.
CATALOG_SEVERITY_TO_REPAIR = {
    1: "minor_repair",
    2: "minor_repair",
    3: "moderate_repair",
    4: "full_replacement",
}

# Main lookup table: issue_id -> severity_band -> (low, high) in USD
RENOVATION_COST_TABLE: Dict[str, Dict[str, CostRange]] = {
    # ───────────────────────────── MOISTURE / INTERIOR ENVELOPE ─────────────────────────────
    "water_stain_ceiling": {
        "minor_repair":     (150, 350),     # Spot stain, patch + paint small area
        "moderate_repair":  (350, 1_200),   # Multiple stains / partial ceiling section + cause fix
        "full_replacement": (1_200, 3_500), # Large area, extensive drywall + paint + possible leak tracing
    },
    "visible_mold_or_mildew": {
        "minor_repair":     (400, 1_500),   # Localized surface remediation (bath/kitchen corner, closet)
        "moderate_repair":  (1_500, 5_000), # Several rooms or one larger space
        "full_replacement": (5_000, 15_000) # Professional remediation, containment, rebuild finishes
    },

    # ───────────────────────────── INTERIOR FINISHES ─────────────────────────────
    "peeling_or_discolored_paint": {
        "minor_repair":     (200, 800),     # Single room walls/ceiling refresh
        "moderate_repair":  (800, 2_500),   # Several rooms
        "full_replacement": (2_500, 6_000), # Whole interior repaint (avg SF home)
    },
    "damaged_drywall_or_cracks": {
        "minor_repair":     (200, 600),     # Patch a few cracks/holes + touch-up
        "moderate_repair":  (600, 2_000),   # Multiple rooms / larger repairs
        "full_replacement": (2_000, 6_000), # Widespread replacement + full repaint sections
    },
    "ceiling_cracks_or_sagging": {
        "minor_repair":     (500, 2_500),   # Limited section, reinforcement + finish repair
        "moderate_repair":  (2_500, 8_000), # Larger area, possible framing evaluation
        "full_replacement": (8_000, 20_000) # Structural correction + large-scale ceiling replacement
    },

    # ───────────────────────────── FLOORING (INTERIOR) ─────────────────────────────
    "worn_or_stained_carpet": {
        "minor_repair":     (300, 1_200),   # One bedroom or small area
        "moderate_repair":  (1_200, 3_500), # Several rooms
        "full_replacement": (3_500, 7_000), # Whole house carpet replacement
    },
    "scratched_or_damaged_flooring": {
        "minor_repair":     (300, 1_500),   # Spot repairs / refinishing small zone
        "moderate_repair":  (1_500, 5_000), # Main living areas
        "full_replacement": (5_000, 12_000) # Whole-house hard flooring redo
    },
    "trip_hazard_or_unlevel_floor": {
        "minor_repair":     (400, 2_000),   # Threshold fixes, small leveling/transition
        "moderate_repair":  (2_000, 8_000), # Significant subfloor/leveling in parts of home
        "full_replacement": (8_000, 25_000) # Major structural/leveling + flooring redo
    },

    # ───────────────────────────── EXTERIOR ENVELOPE ─────────────────────────────
    "damaged_or_rotted_siding_or_trim": {
        "minor_repair":     (500, 2_500),   # Replace localized boards/trim + spot paint
        "moderate_repair":  (2_500, 8_000), # One elevation / significant sections
        "full_replacement": (8_000, 25_000) # Whole-house reside (mid-size home)
    },
    "damaged_or_aged_roof_shingles": {
        "minor_repair":     (500, 2_500),   # Small patch / a slope, tune-up
        "moderate_repair":  (2_500, 10_000),# Partial reroof
        "full_replacement": (10_000, 25_000) # Full reroof, asphalt shingles
    },
    "damaged_or_unsafe_deck_or_porch": {
        "minor_repair":     (500, 2_500),   # Replace rail sections/boards, stabilize stairs
        "moderate_repair":  (2_500, 8_000), # Rebuild large sections
        "full_replacement": (8_000, 20_000) # Full new deck/porch
    },
    "broken_or_fogged_windows": {
        "minor_repair":     (300, 1_200),   # Single-unit glass/IGU replacement
        "moderate_repair":  (1_200, 4_500), # Several units
        "full_replacement": (4_500, 15_000) # Most/all windows in a typical home
    },

    # ───────────────────────────── KITCHEN / BATH FINISHES ─────────────────────────────
    "outdated_or_damaged_cabinets": {
        "minor_repair":     (500, 3_000),   # Door/drawer repair, paint, new hardware
        "moderate_repair":  (3_000, 10_000),# Partial replacement + refinish
        "full_replacement": (10_000, 30_000) # Full new cabinets (kitchen + some baths)
    },
    "countertop_damage": {
        "minor_repair":     (400, 1_500),   # Spot repair / small section replacement
        "moderate_repair":  (1_500, 4_000), # Replace kitchen counters only
        "full_replacement": (4_000, 10_000) # Kitchen + baths w/ mid-grade stone
    },
    "tile_or_grout_damage": {
        "minor_repair":     (300, 1_200),   # Re-grout/re-tile small area
        "moderate_repair":  (1_200, 4_000), # Entire tub surround or kitchen floor
        "full_replacement": (4_000, 10_000) # Multiple rooms / large areas
    },
    "stained_or_damaged_bath_fixtures": {
        "minor_repair":     (200, 1_000),   # Replace single toilet/sink or refinish tub
        "moderate_repair":  (1_000, 4_000), # Replace multiple fixtures in a bath
        "full_replacement": (4_000, 10_000) # Full bath fixture overhaul (no layout move)
    },

    # ───────────────────────────── SYSTEMS (APPLIANCE / PLUMBING / ELECTRICAL / PEST) ─────────────────────────────
    "appliance_damage_or_missing": {
        "minor_repair":     (500, 2_500),   # One or two mid-grade appliances
        "moderate_repair":  (2_500, 6_000), # Full basic kitchen suite
        "full_replacement": (6_000, 12_000) # Premium suite / multiple areas (kitchen + laundry)
    },
    "plumbing_fixture_leaking_stained": {
        "minor_repair":     (200, 1_000),   # Fix a few leaks, replace a faucet/angle stop
        "moderate_repair":  (1_000, 4_000), # Multiple fixtures, some drain work
        "full_replacement": (4_000, 12_000) # Re-pipe sections, many fixtures, possible wall opens
    },
    "visible_electrical_risks": {
        "minor_repair":     (300, 1_500),   # Correct exposed wiring / add covers / minor rewiring
        "moderate_repair":  (1_500, 6_000), # Multiple circuits, panel clean-up, GFCI/AFI upgrades
        "full_replacement": (6_000, 20_000) # Service upgrade, extensive rewiring
    },
    "pest_or_rodent_evidence": {
        "minor_repair":     (200, 800),     # Basic treatment + sealing entry points
        "moderate_repair":  (800, 2_500),   # Ongoing treatments, some material replacement
        "full_replacement": (2_500, 7_000)  # Major infestation, cleanup, insulation/finish replacement
    },

    # ───────────────────────────── SAFETY / STAIRS / RAILINGS ─────────────────────────────
    # NOTE: your issue_catalog currently *omits* an "id" for this item.
    # You should update issue_catalog.json to include:
    #   "id": "missing_or_damaged_handrails"
    "missing_or_damaged_handrails": {
        "minor_repair":     (200, 800),     # Add/repair a short rail section
        "moderate_repair":  (800, 2_500),   # Multiple stair runs / deck rail sections
        "full_replacement": (2_500, 6_000)  # Full stair/deck railing replacement
    },

    # ───────────────────────────── BASEMENT / GARAGE / FOUNDATION / GRADING ─────────────────────────────
    "garage_or_basement_damage": {
        "minor_repair":     (800, 4_000),   # Patch cracks, seal, minor moisture-mitigation
        "moderate_repair":  (4_000, 15_000),# Larger crack injection, drainage work, structural shoring
        "full_replacement": (15_000, 40_000)# Major structural repair / extensive waterproofing
    },
    "major_foundation_or_settlement_signs": {
        "minor_repair":     (2_000, 15_000),# Limited piering, slabjacking, or localized repair
        "moderate_repair":  (15_000, 40_000),# More extensive system of piers/beams/braces
        "full_replacement": (40_000, 100_000)# Major foundation replacement / rebuild portions of structure
    },
    "standing_water_or_poor_grading": {
        "minor_repair":     (500, 2_500),   # Add downspout extensions, minor grading
        "moderate_repair":  (2_500, 8_000), # Regrade around home, add drains/sump
        "full_replacement": (8_000, 25_000) # Major drainage system, retaining walls etc.
    },

    # ───────────────────────────── SITE / EXTERIOR SURFACES ─────────────────────────────
    "driveway_or_walkway_cracking": {
        "minor_repair":     (400, 2_000),   # Crack fill, patch sections
        "moderate_repair":  (2_000, 8_000), # Replace significant sections
        "full_replacement": (8_000, 20_000) # Full driveway/walkway replacement
    },
    "clogged_or_damaged_gutters": {
        "minor_repair":     (200, 800),     # Clean, reattach small sections
        "moderate_repair":  (800, 2_500),   # Replace runs, add/downsize/downspouts
        "full_replacement": (2_500, 6_000)  # Full gutter system replacement
    },
    "trees_or_vegetation_too_close": {
        "minor_repair":     (300, 1_500),   # Trim/clear smaller vegetation near structure
        "moderate_repair":  (1_500, 5_000), # Remove 1–2 moderate trees / heavy pruning
        "full_replacement": (5_000, 15_000) # Multiple large trees close to structure
    },

    # ───────────────────────────── OPPORTUNITY FLAGS (VALUE-ADD RENOS) ─────────────────────────────
    "outdated_kitchen_finishes": {
        "minor_repair":     (2_000, 8_000),  # Paint cabinets, update hardware/backsplash, minor counters
        "moderate_repair":  (8_000, 20_000), # Mid-grade cosmetic remodel (no layout change)
        "full_replacement": (20_000, 60_000) # High-end or full gut kitchen
    },
    "outdated_bathroom_finishes": {
        "minor_repair":     (1_500, 6_000),  # New vanity, fixtures, minor tile updates
        "moderate_repair":  (6_000, 15_000), # Full bath cosmetic remodel
        "full_replacement": (15_000, 35_000) # High-end / layout change, new tub/shower systems
    },
    "older_flooring_style": {
        "minor_repair":     (800, 3_000),    # Replace flooring in a couple of rooms
        "moderate_repair":  (3_000, 8_000),  # Main living areas
        "full_replacement": (8_000, 20_000)  # Whole-house flooring modernization
    },
    "dated_lighting_fixtures": {
        "minor_repair":     (250, 1_500),    # Swap a few key fixtures
        "moderate_repair":  (1_500, 5_000),  # Most rooms incl. exterior lights
        "full_replacement": (5_000, 12_000)  # Higher-end packages / extensive rewiring for new layouts
    },
    "paint_refresh_recommended": {
        "minor_repair":     (500, 2_500),    # A couple of rooms
        "moderate_repair":  (2_500, 7_000),  # Majority of interior
        "full_replacement": (7_000, 15_000)  # Whole interior (and possibly some exterior trims)
    },
    "landscape_improvement_needed": {
        "minor_repair":     (500, 2_500),    # Clean-up, mulch, simple plantings
        "moderate_repair":  (2_500, 8_000),  # Add beds, small hardscape, irrigation tweaks
        "full_replacement": (8_000, 20_000)  # Larger-scale landscape redesign
    },
    "curb_appeal_upgrade": {
        "minor_repair":     (300, 1_500),    # Paint door, new hardware, small accents
        "moderate_repair":  (1_500, 5_000),  # New walkway/steps, porch upgrades, better lighting
        "full_replacement": (5_000, 15_000)  # Significant front facade improvements
    },
    "unfinished_basement_present": {
        "minor_repair":     (2_000, 15_000), # Partial finishing / basic storage upgrade
        "moderate_repair":  (15_000, 40_000),# Finish a good portion with basic materials
        "full_replacement": (40_000, 90_000) # Full basement finish w/ rooms, bath, nicer materials
    },
    "staging_or_decluttering_opportunity": {
        "minor_repair":     (150, 800),      # DIY decluttering, minor storage solutions
        "moderate_repair":  (800, 2_500),    # Professional organizer / partial staging
        "full_replacement": (2_500, 6_000)   # Full professional staging for the home
    },
}
