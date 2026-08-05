# kind-ontology-v2 gold-label review

All 132 cases need your approval before the dev/holdout sets freeze.
Labels follow the approved ontology: **defect** = function/safety/integrity/protection
has FAILED; **degradation** = functional but visibly deteriorated; **modernization** =
functional and maintained but dated/basic. Excluded lanes get a reason, not a kind.

`FLAG` marks cases I consider borderline — read those rationales first.
To change a label, edit the `gold` field in benchmarks/kind-ontology-v2/cases_dev.json
or cases_holdout.json (case_id is the key), or just tell me the case_id and the fix.

## Atomic cases (90)

| id | slice | scene | text | gold kind | rationale | src | flag |
|---|---|---|---|---|---|---|---|
| def-001 | dev | living_room | Multiple visible cracks run across the ceiling. | **defect** | Structural surface integrity has failed (cracks). | corpus |  |
| def-002 | dev | exterior_front | A downspout is disconnected and terminates at the foundation. | **defect** | Asserted failure of a protective drainage component. | synthetic |  |
| def-003 | dev | basement | Exposed wiring runs along the ceiling beams without visible protection or conduit. | **defect** | Unsafe condition (exposed wiring). | corpus |  |
| def-004 | dev | basement | Multiple drop-ceiling tiles are missing or sagging, exposing framing and wiring. | **defect** | Missing/failed required parts, exposure. | corpus |  |
| def-005 | dev | exterior_back | The existing concrete patio pad is cracked, with sections heaved and a raised trip edge. | **defect** | Flatwork rule: heaving/raised trip edge = defect (approved in review). | corpus-adapted |  |
| def-006 | dev | bathroom | Some tiles above the vanity appear removed or damaged, exposing the substrate. | **defect** | Missing/damaged tile exposing substrate = failed protection. | corpus |  |
| def-007 | dev | bathroom | Visible staining and mold are present around the tub surround area. | **defect** | Mold is a failure/health condition; it dominates the staining detail. | corpus | FLAG |
| def-008 | dev | bedroom | A large patched or damaged area is visible on the right wall. | **defect** | Visible damage to wall surface. | corpus |  |
| def-009 | dev | exterior_front | The concrete driveway is cracked with uneven joints. | **degradation** | Flatwork rule (approved in review): cracking/minor unevenness = degradation; heaving/trip edge = defect. | corpus-adapted |  |
| def-010 | dev | exterior_back | Deck boards are rotted through near the stairs. | **defect** | Rot = material failure. | synthetic |  |
| def-011 | dev | exterior_front | The roof is missing several shingles above the garage. | **defect** | Missing required weather-protection parts. | synthetic |  |
| def-012 | dev | living_room | Water stains are visible on the ceiling below the bathroom. | **defect** | Ceiling water stain evidences moisture intrusion (failed protection) — NOT cosmetic staining. | synthetic | FLAG |
| def-013 | dev | exterior_back | A window pane in the rear door is cracked. | **defect** | Broken glazing. | synthetic |  |
| def-014 | dev | bathroom | The vanity base under the bathroom sink is water-stained and swollen. | **defect** | Visible moisture damage (swelling = material failure). Replaced an unverifiable 'leaking faucet' claim in review. | synthetic |  |
| def-015 | dev | yard | A section of the wooden privacy fence has collapsed. | **defect** | Structural collapse. | synthetic |  |
| def-016 | dev | exterior_front | The screen door is torn and hanging off its lower hinge. | **defect** | Torn + detached = functional failure. | corpus-adapted |  |
| def-017 | dev | exterior_front | Gutters are sagging and pulling away from the fascia along the front roofline. | **defect** | Attached component failing (sagging/detaching). | synthetic |  |
| def-018 | dev | basement | An open or missing cover is visible at a ceiling vent or access point. | **defect** | Missing required cover. | corpus |  |
| def-019 | dev | garage | The garage door is dented and does not close fully at one corner. | **defect** | Stated functional failure (does not close). | synthetic |  |
| def-020 | dev | exterior_side | Mortar joints in the chimney are crumbling near the top. | **defect** | Masonry integrity failure. | synthetic |  |
| def-021 | dev | bedroom | The subfloor is exposed where flooring has been stripped and damaged. | **defect** | Failed/absent floor covering exposing subfloor. | corpus-adapted |  |
| def-022 | holdout | exterior_back | The railing on the back steps is broken and unstable. | **defect** | Safety component failure. | synthetic |  |
| def-023 | holdout | living_room | A recessed lighting fixture is missing from its hole in the ceiling. | **defect** | Missing required installed part. | corpus-adapted |  |
| def-024 | holdout | yard | The patio slab is severely cracked and heaved in places. | **defect** | Severe cracking with displacement. | corpus-adapted |  |
| def-025 | holdout | bathroom | Black mold and mildew are visible along the lower shower wall. | **defect** | Mold growth = failure condition. | synthetic |  |
| def-026 | holdout | exterior_back | The rear soffit shows rot where the boards meet the gutter. | **defect** | Rot = material failure. | synthetic |  |
| def-027 | holdout | exterior_side | An exterior outlet cover is missing, leaving the receptacle exposed to weather. | **defect** | Missing protective component, unsafe exposure. | synthetic |  |
| def-028 | holdout | bedroom | A hole in the drywall exposes framing beside the closet door. | **defect** | Wall integrity failure. | synthetic |  |
| def-029 | holdout | exterior_back | Several deck balusters are missing along the left side. | **defect** | Missing safety components. | synthetic |  |
| def-030 | holdout | bathroom | The bathroom floor tile is cracked and lifting near the toilet. | **defect** | Cracked + lifting = failed floor covering. | synthetic |  |
| deg-001 | dev | exterior_side | Wood siding is faded and weathered but remains intact. | **degradation** | Visible deterioration, explicitly intact. | synthetic |  |
| deg-002 | dev | bedroom | The carpet is visibly worn with heavy traffic patterns near the center and entryways. | **degradation** | Wear without failure. | corpus |  |
| deg-003 | dev | exterior_front | The roof shingles appear aged and weathered. | **degradation** | Weathering with no failure evidence (contrast def-011 missing shingles). | corpus | FLAG |
| deg-004 | dev | exterior_front | White trim around the door and windows looks faded or chipped. | **degradation** | Surface finish deterioration; chipping is finish-level, not structural. | corpus | FLAG |
| deg-005 | dev | kitchen | Countertop edges and seams appear worn. | **degradation** | Wear without failure. | corpus |  |
| deg-006 | dev | exterior_front | Exterior paint and trim show peeling and weathered areas. | **degradation** | Finish deterioration (peeling paint), protection not asserted failed. | corpus | FLAG |
| deg-007 | dev | living_room | The hardwood floors are worn and uneven in places near the center of the room. | **degradation** | Wear-dominant; unevenness described as wear pattern, no failure. | corpus-adapted | FLAG |
| deg-008 | dev | living_room | Hardwood flooring shows scuffs, wear, and uneven tone. | **degradation** | Surface wear. | corpus-adapted |  |
| deg-009 | dev | living_room | The brown low-pile carpet is visibly worn and stained in places. | **degradation** | Wear + cosmetic staining, same claim type. | corpus |  |
| deg-010 | dev | utility_room | Dark staining is present on the floor surface. | **degradation** | Cosmetic staining, no moisture-source assertion. | corpus |  |
| deg-011 | dev | exterior_front | Staining and white powdery residue appear on the brick near ground level. | **degradation** | Efflorescence/staining = visible deterioration; no failure demonstrated. | corpus | FLAG |
| deg-012 | dev | garage | The concrete floor shows significant staining and discoloration. | **degradation** | Cosmetic surface condition. | corpus-adapted |  |
| deg-013 | dev | exterior_front | The asphalt shingle roof shows discoloration and moss growth. | **degradation** | Aging/organic growth without asserted failure. | corpus-adapted | FLAG |
| deg-014 | dev | dining_room | An exposed beam in the corner has peeling, weathered green paint. | **degradation** | Finish deterioration on a sound element. | corpus-adapted |  |
| deg-015 | dev | exterior_back | Deck planks appear aged with gaps and discoloration. | **degradation** | Aging/wear; no rot or failure asserted (contrast def-010). | corpus-adapted | FLAG |
| deg-016 | dev | exterior_side | The wooden picket fence looks weathered and gray. | **degradation** | Weathering without failure. | corpus-adapted |  |
| deg-017 | dev | living_room | Baseboards show scuffs and chipped paint along the hallway. | **degradation** | Surface wear. | synthetic |  |
| deg-018 | dev | bathroom | The vanity cabinet doors are worn around the handles. | **degradation** | Wear without failure. | synthetic |  |
| deg-019 | dev | kitchen | The kitchen faucet is tarnished and water-spotted but functional. | **degradation** | Explicitly functional, visibly deteriorated. | synthetic |  |
| deg-020 | dev | exterior_front | The driveway surface is faded with hairline cracking throughout. | **degradation** | Surface-level fade + hairline cracks (contrast def-009 cracked/uneven joints). | corpus-adapted | FLAG |
| deg-021 | dev | bedroom | Wall paint is scuffed and dingy throughout the room. | **degradation** | Finish wear. | synthetic |  |
| deg-022 | holdout | exterior_front | The metal handrail is rusty on the surface but solidly attached. | **degradation** | Surface corrosion, explicitly sound (contrast rusted-through = defect). | synthetic |  |
| deg-023 | holdout | garage | The garage floor coating is worn through in the parking strip. | **degradation** | Coating wear (finish layer), slab itself not failed. | synthetic | FLAG |
| deg-024 | holdout | bathroom | The tub surround caulk is discolored and dingy. | **degradation** | Cosmetic deterioration of caulk, no gap/failure asserted. | synthetic |  |
| deg-025 | holdout | exterior_front | Roof shingles look worn and near the end of their life. | **degradation** | Aged but not failed — the classic v1 wear-band case (v1 catalog called this defect). | corpus | FLAG |
| deg-026 | holdout | living_room | The stair treads are worn smooth with the finish rubbed off in the middle. | **degradation** | Finish wear. | synthetic |  |
| deg-027 | holdout | kitchen | The laminate countertop is faded and scratched near the sink. | **degradation** | Wear without failure. | synthetic |  |
| deg-028 | holdout | exterior_side | Window sills are weathered with flaking paint. | **degradation** | Weathering/finish deterioration. | synthetic |  |
| deg-029 | holdout | living_room | The ceiling fan blades are discolored and dusty. | **degradation** | Cosmetic deterioration, fan works. | corpus-adapted |  |
| deg-030 | holdout | exterior_front | The front door finish is faded and worn around the handle. | **degradation** | Finish wear. | synthetic |  |
| mod-001 | dev | kitchen | Kitchen has dated oak cabinets in good repair. | **modernization** | Dated but explicitly maintained. | synthetic |  |
| mod-002 | dev | kitchen | The ceiling lights are flush-mount builder-grade fixtures. | **modernization** | Basic/low-grade fixture, functional. | corpus |  |
| mod-003 | dev | kitchen | Countertops appear to be made of laminate. | **modernization** | Low-grade material = improvement opportunity (no condition claim). | corpus | FLAG |
| mod-004 | dev | bathroom | The vanity cabinet is made of white laminate material. | **modernization** | Low-grade material. | corpus | FLAG |
| mod-005 | dev | living_room | The ceiling fan is dated in style and scale. | **modernization** | Dated style, no condition claim. | corpus |  |
| mod-006 | dev | bedroom | The room has a single dated flush-mount light fixture. | **modernization** | Dated/basic fixture. | corpus |  |
| mod-007 | dev | kitchen | Lighting includes recessed lights and basic pendants. | **modernization** | 'Basic' grades the fixture — improvement opportunity, not mere presence. | corpus | FLAG |
| mod-008 | dev | bedroom | The window has original-style trim and appears older. | **modernization** | Dated/original, no deterioration language. | corpus |  |
| mod-009 | dev | kitchen | Dark raised-panel cabinets appear dated but serviceable. | **modernization** | Dated, explicitly serviceable. | corpus |  |
| mod-010 | dev | living_room | A basic builder-grade dome light fixture is mounted on the ceiling. | **modernization** | Basic/builder-grade. | corpus |  |
| mod-011 | dev | bathroom | The bathroom mirror is a basic frameless sheet. | **modernization** | Basic grade. | corpus |  |
| mod-012 | dev | exterior_side | Windows are older single-pane units with aluminum frames. | **modernization** | Dated technology, functional. | synthetic |  |
| mod-013 | dev | kitchen | The backsplash features a dated diamond-pattern tile that is not broken. | **modernization** | Dated style, explicitly intact. | corpus-adapted |  |
| mod-014 | dev | kitchen | The kitchen appliances are white and mismatched in age. | **modernization** | Dated/mismatched, no failure. | synthetic |  |
| mod-015 | dev | living_room | Popcorn ceiling texture covers the living room ceiling. | **modernization** | Dated finish, improvement opportunity. | synthetic |  |
| mod-016 | dev | bathroom | The tub is an older almond-colored fixture in working order. | **modernization** | Dated, explicitly working. | synthetic |  |
| mod-017 | dev | exterior_front | The screen door appears dated in style. | **modernization** | Style only (contrast def-016 torn screen door). | corpus-adapted |  |
| mod-018 | dev | kitchen | The window features blinds that are functional but outdated in style. | **modernization** | Explicitly functional, outdated. | corpus-adapted |  |
| mod-019 | dev | dining_room | The brass chandelier is a traditional style from the 1990s. | **modernization** | Dated style. | corpus-adapted |  |
| mod-020 | dev | living_room | Trim consists of basic flat casing. | **modernization** | Basic grade. | corpus | FLAG |
| mod-021 | dev | exterior_front | The exterior features a basic wall sconce beside the door. | **modernization** | Basic fixture. | corpus-adapted |  |
| mod-022 | holdout | bathroom | The bathroom has a cultured marble vanity top typical of older builds. | **modernization** | Dated material, no condition claim. | synthetic |  |
| mod-023 | holdout | kitchen | The kitchen sink is a standard double-bowl stainless unit with a basic faucet. | **modernization** | Basic grade. | synthetic |  |
| mod-024 | holdout | bedroom | The bedroom closet has a single wire shelf typical of basic construction. | **modernization** | Basic construction. | synthetic |  |
| mod-025 | holdout | kitchen | Flooring is an older sheet vinyl in a dated pattern, free of damage. | **modernization** | Dated, explicitly undamaged. | synthetic |  |
| mod-026 | holdout | living_room | The ceiling fan is a basic model that appears over ten years old. | **modernization** | Basic/dated. | corpus-adapted |  |
| mod-027 | holdout | living_room | The fireplace surround is dated brass and tile but intact. | **modernization** | Dated, intact. | synthetic |  |
| mod-028 | holdout | exterior_front | The garage has a basic builder-grade door without windows. | **modernization** | Basic grade (absent windows are an optional feature). | synthetic | FLAG |
| mod-029 | holdout | bathroom | The hall bathroom features an oak vanity from the original construction. | **modernization** | Original/dated. | synthetic |  |
| mod-030 | holdout | bedroom | Interior doors are hollow-core flat panels with dated brass knobs. | **modernization** | Basic/dated. | synthetic |  |

## Excluded cases (18)

| id | slice | scene | text | exclusion reason | rationale | src | flag |
|---|---|---|---|---|---|---|---|
| exc-good-001 | dev | bedroom | Baseboards appear intact and well-maintained along the walls. | **good_condition** | Explicit good condition. | corpus |  |
| exc-good-002 | dev | bathroom | White subway tile surrounds the shower and appears generally intact. | **good_condition** | Explicit intact. | corpus |  |
| exc-good-003 | holdout | living_room | Wood-look plank flooring is in good condition. | **good_condition** | Explicit good condition. | corpus |  |
| exc-neut-001 | dev | bedroom | The room contains a ceiling fan. | **neutral_presence** | Pure presence, no grade or condition. | synthetic |  |
| exc-neut-002 | dev | living_room | Wide-plank wood-look flooring is installed. | **neutral_presence** | Presence without grade/condition (contrast mod-003 'laminate'). | corpus | FLAG |
| exc-neut-003 | holdout | bathroom | The light fixture is a multi-bulb vanity light. | **neutral_presence** | Descriptive presence only. | corpus |  |
| exc-adv-001 | dev | exterior_back | Windows should be checked for drafts, seal failure, and frame condition. | **advice_or_process** | Inspection advice, no observed condition. | corpus |  |
| exc-adv-002 | dev | bedroom | Hardwood flooring under carpet may need refinishing or replacement if worn. | **advice_or_process** | Conditional advice about an unseen surface. | corpus-adapted |  |
| exc-adv-003 | holdout | yard | Erosion control needs should be checked near the slope. | **advice_or_process** | Process language. | corpus-adapted |  |
| exc-spec-001 | dev | exterior_front | No gutters or downspouts are visibly present on the exterior. | **unsupported_or_speculative** | Absence inferred from non-visibility — the gutter absence-safety case. | corpus | FLAG |
| exc-spec-002 | dev | living_room | Exposed brick wall may have cracks, spalling, or previous water damage. | **unsupported_or_speculative** | Speculative 'may have' with no visible sign. | corpus |  |
| exc-spec-003 | holdout | exterior_front | Insulation levels are probably low in the attic and walls. | **unsupported_or_speculative** | Hidden system, no visible sign. | corpus |  |
| exc-dim-001 | dev | bedroom | Primary Bedroom 12'6 x 10' | **measurement_overlay** | Floorplan overlay text (deterministic pre-filter). | synthetic |  |
| exc-dim-002 | dev | living_room | Living Room 20 x 15 | **measurement_overlay** | Floorplan overlay text (deterministic pre-filter). | synthetic |  |
| exc-dim-003 | holdout | kitchen | Kitchen 10'x8'6" | **measurement_overlay** | Floorplan overlay text (deterministic pre-filter). | synthetic |  |
| exc-nrr-001 | dev | kitchen | A watermark reading 'Valley MLS' is visible on the image. | **not_renovation_related** | Image artifact, not the property. | corpus-adapted |  |
| exc-nrr-002 | dev | bedroom | The current furniture arrangement makes the room feel cramped. | **not_renovation_related** | Staging/furniture, not property condition. | corpus |  |
| exc-nrr-003 | holdout | living_room | A patterned rug in the foreground contrasts with the hardwood flooring. | **not_renovation_related** | Furnishing, not property condition. | corpus |  |

## Mixed decomposition cases (24)

| id | slice | scene | note | gold claims | rationale | src | flag |
|---|---|---|---|---|---|---|---|
| mix-001 | dev | exterior_back | The deck boards are weathered and rotted near the stairs. | deck/weathered → **degradation**; deck/rot|rotted → **defect** | deg+def on one component: weathering vs rot must split. | synthetic |  |
| mix-002 | dev | bedroom | The carpet is worn throughout and water-stained near the vent. | carpet/worn → **degradation**; carpet/water → **defect** | deg+def: wear vs moisture evidence. | corpus-adapted | FLAG |
| mix-003 | dev | exterior_side | Exterior paint is peeling and one siding board is cracked. | paint/peel → **degradation**; siding/crack → **defect** | deg+def on different components. | synthetic |  |
| mix-004 | dev | exterior_front | The driveway is faded and one section has heaved, leaving a raised crack across the middle. | driveway/faded → **degradation**; driveway/heav|raised → **defect** | deg+def on one component; heave/raised edge keeps the defect claim under the flatwork rule. | synthetic |  |
| mix-005 | dev | kitchen | The kitchen cabinets are worn and dated. | cabinet/worn → **degradation**; cabinet/dated → **modernization** | deg+mod on one component — the canonical wear/dated split. | synthetic |  |
| mix-006 | dev | bathroom | The bathroom vanity is an older basic unit and its top is stained. | vanity/stain → **degradation**; vanity/basic|older → **modernization** | deg+mod on one component. | synthetic |  |
| mix-007 | dev | living_room | The ceiling fan is dated in style and its blades are discolored. | fan/discolor → **degradation**; fan/dated → **modernization** | deg+mod; bundling both into one claim is the failure mode. | corpus-adapted |  |
| mix-008 | dev | exterior_front | The screen door is dated in style and torn at the bottom corner. | door/torn → **defect**; door/dated → **modernization** | def+mod on one component. | corpus-adapted |  |
| mix-009 | dev | kitchen | The kitchen has dated oak cabinets and a cracked sink basin. | cabinet/dated|oak → **modernization**; sink/crack → **defect** | def+mod on different components. 'Leaking faucet' replaced in review — 2c classifies asserted text; unverifiable claims are a 2a/2b grounding issue. | synthetic |  |
| mix-010 | dev | exterior_front | Gutters are stained along the front and one downspout is disconnected. | gutter/stain → **degradation**; downspout/disconnect → **defect** | deg+def in the gutter system. | synthetic |  |
| mix-011 | dev | living_room | The hardwood floors are scuffed and the trim is a basic flat casing. | floor/scuff → **degradation**; trim/basic → **modernization** | deg+mod on different components. | corpus-adapted |  |
| mix-012 | dev | bathroom | The bathroom has a basic frameless mirror and cracked floor tile near the tub. | mirror/basic|frameless → **modernization**; tile/crack → **defect** | def+mod on different components. | corpus-adapted |  |
| mix-013 | dev | exterior_front | The roof shingles are weathered, the gutter is sagging, and the porch light is a dated builder-grade fixture. | shingle/weather → **degradation**; gutter/sag → **defect**; light/dated|builder → **modernization** | All three kinds in one note. | synthetic |  |
| mix-014 | dev | kitchen | Kitchen cabinets are dated oak, the counter is worn at the edges, and the sink base shows water damage. | cabinet/dated|oak → **modernization**; counter/worn → **degradation**; sink/water → **defect** | All three kinds in one note. | synthetic |  |
| mix-015 | dev | bedroom | The bedroom carpet is stained, a window pane is cracked, and the light fixture is a basic dome. | carpet/stain → **degradation**; pane/crack → **defect**; fixture/basic|dome → **modernization** | All three kinds in one note. | synthetic |  |
| mix-016 | dev | exterior_back | The back fence has collapsed in one section and the patio slab is heaved and cracked. | fence/collaps → **defect**; slab/crack|heav → **defect** | Same-kind (defect) different concepts must stay separate. | corpus-adapted |  |
| mix-017 | dev | exterior_front | The siding is faded and the front walkway is stained with rust marks. | siding/faded → **degradation**; walkway/stain|rust → **degradation** | Same-kind (degradation) different concepts. | synthetic |  |
| mix-018 | dev | kitchen | The kitchen has laminate countertops and builder-grade cabinet hardware. | countertop/laminate → **modernization**; hardware/builder → **modernization** | Same-kind (modernization) different concepts. | synthetic |  |
| mix-019 | holdout | exterior_side | The wood fence is weathered gray and two pickets are missing. | fence/weathered → **degradation**; picket/missing → **defect** | deg+def. | synthetic |  |
| mix-020 | holdout | bathroom | The vanity is a dated builder-grade unit with a worn and scratched top. | vanity/dated|builder → **modernization**; vanity/worn|scratch → **degradation** | deg+mod on one component. | synthetic |  |
| mix-021 | holdout | living_room | The living room has a dated brass chandelier and a large water stain on the ceiling. | chandelier/dated|brass → **modernization**; ceiling/water → **defect** | def+mod. | synthetic |  |
| mix-022 | holdout | garage | The garage door is dented at the corner, its paint is faded, and the opener is an older basic model. | door/dent → **defect**; paint/faded → **degradation**; opener/older|basic → **modernization** | All three kinds. | synthetic |  |
| mix-023 | holdout | bathroom | The bathroom has a cracked floor tile and a leaking faucet. | tile/crack → **defect**; faucet/leak → **defect** | Same-kind (defect) different concepts. | synthetic |  |
| mix-024 | holdout | bathroom | The bathroom has an oak vanity from original construction and a basic frameless mirror. | vanity/oak|original → **modernization**; mirror/basic|frameless → **modernization** | Same-kind (modernization) different concepts. | synthetic |  |
