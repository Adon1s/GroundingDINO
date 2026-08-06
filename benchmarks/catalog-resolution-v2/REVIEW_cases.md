# REVIEW — catalog-resolution-v2 cases

Review sheet for the Pause B gate. Cases are `status: draft` until reviewed;
the runner refuses `--gates` on a draft slice.

**What to check per row:** does the observation text read like something Pass 2b
would actually emit, and is the gold item the one *you* would want it to resolve to?
Flag anything where the text could legitimately resolve elsewhere — ambiguous text
makes the recall gate meaningless.

Paired groups are the point of the benchmark: same subject, different kind. If a
pair's two texts do not read as clearly different claims, the split behind them is
probably wrong and belongs back in Pause A.

## dev (62 cases)

| case | kind | gold | FLAG | observation |
|---|---|---|---|---|
| res-app-001 _appliances-trio_ | defect | `appliances_damaged_or_missing` |  | There is no range in the kitchen, just an empty gap and a capped connection. |
| res-app-002 _appliances-trio_ | degradation | `appliances_worn_or_neglected` |  | Appliances are running but heavily greased and grimy with age. |
| res-app-003 _appliances-trio_ | modernization | `appliances_dated_or_basic` |  | Kitchen has older white appliances that work fine but look out of date. |
| res-bth-001 _bath-pair_ | defect | `bath_fixtures_chipped_or_damaged` |  | The tub has a chipped area where the enamel has broken away. |
| res-bth-002 _bath-pair_ | degradation | `bath_fixtures_stained_or_worn` |  | Tub and toilet show heavy staining and a dulled, worn finish. |
| res-cab-001 _cabinets-trio_ | defect | `cabinets_damaged_or_water_stained` |  | The cabinet door under the sink is broken off its hinge and the base panel is swollen from water. |
| res-cab-002 _cabinets-trio_ | degradation | `cabinets_worn_finish` |  | Cabinet finish is worn through around the handles and the doors look faded. |
| res-cab-003 _cabinets-trio_ | modernization | `cabinets_dated_style` |  | Kitchen has golden oak raised-panel cabinetry in sound condition. |
| res-cfn-001 _ceiling-fan-pair_ | defect | `ceiling_fan_blades_damaged` |  | One ceiling fan blade is snapped and another is drooping badly. |
| res-cfn-002 _ceiling-fan-pair_ | modernization | `ceiling_fan_blades_mismatched` |  | The ceiling fan blades do not match each other, though the fan runs. |
| res-fen-001 _fence-pair_ | defect | `fence_broken_or_leaning` |  | A section of the wood fence is leaning over with rotted posts. |
| res-fen-002 _fence-pair_ | degradation | `fence_weathered` |  | Fence boards have greyed out from sun exposure but stand straight. |
| res-flr-001 _flooring-hard-pair_ | defect | `hard_flooring_broken_or_warped` |  | Several floor tiles are cracked through and one plank has buckled upward. |
| res-flr-002 _flooring-hard-pair_ | degradation | `hard_flooring_scratched_or_worn` |  | The hardwood is intact but the finish is dull and scratched in the walkway. |
| res-lnd-001 _landscaping-pair_ | degradation | `landscaping_overgrown_or_neglected` |  | Shrubs are overgrown and the lawn is patchy and full of weeds. |
| res-lnd-002 _landscaping-pair_ | modernization | `landscaping_enhancement_opportunity` |  | The yard is tidy but bare; planting beds would lift the frontage. |
| res-pnt-001 _paint-quad_ | defect | `bathroom_paint_moisture_damage` |  | Paint above the shower is bubbling and there is a chalky white bloom on the wall. |
| res-pnt-002 _paint-quad_ | degradation | `bathroom_paint_peeling_or_worn` |  | Bathroom wall paint is flaking near the door casing with no sign of moisture. |
| res-pnt-003 _paint-quad_ | degradation | `peeling_or_discolored_paint` |  | Interior wall paint is peeling and looks aged in the hallway. |
| res-pnt-004 _paint-quad_ | modernization | `paint_refresh_recommended` |  | Walls are painted a deep personalized color that would benefit from a neutral repaint. |
| res-plm-001 _plumbing-pair_ | defect | `plumbing_fixture_leaking` |  | Water is actively dripping from the supply connection under the kitchen sink. |
| res-plm-002 _plumbing-pair_ | degradation | `plumbing_fixture_stained_or_corroded` |  | Heavy mineral crust and rust staining around the faucet base, nothing dripping. |
| res-sid-001 _siding-trio_ | defect | `damaged_or_rotted_siding_or_trim` |  | A run of siding boards is rotted soft at the bottom edge and one piece is missing. |
| res-sid-002 _siding-trio_ | degradation | `exterior_siding_discoloration_fading` |  | Siding is sound but the color has faded unevenly and looks chalky. |
| res-sid-003 _siding-trio_ | modernization | `dated_exterior_finishes` |  | The exterior color palette and trim details read dated though everything is maintained. |
| res-sof-001 _soffit-pair_ | defect | `soffit_or_porch_ceiling_failed` |  | A soffit panel is hanging loose under the eave and part of the run is missing. |
| res-sof-002 _soffit-pair_ | degradation | `soffit_or_porch_ceiling_weathered` |  | Porch ceiling boards are all in place but the finish is patchy and faded. |
| res-vct-001 _vanity-top-pair_ | degradation | `vanity_countertop_worn` |  | The cultured marble vanity top has yellowed and the surface looks dulled. |
| res-vct-002 _vanity-top-pair_ | modernization | `vanity_countertop_dated` |  | Vanity top is a clean laminate in a dated pattern. |
| res-car-001 | degradation | `worn_or_stained_carpet` |  | Carpet is matted down and stained along the main traffic path. |
| res-def-001 | defect | `broken_or_fogged_windows` |  | One window pane is cracked and another is fogged between the glass. |
| res-def-002 | defect | `tile_or_grout_damage` |  | Grout is missing between several tiles and one tile is loose. |
| res-def-003 | defect | `clogged_or_damaged_gutters` |  | The gutter is sagging away from the fascia and overflowing at the corner. |
| res-def-004 | defect | `countertop_damage` |  | The kitchen countertop has a burn mark and a chipped edge. |
| res-def-005 | defect | `damaged_drywall_or_cracks` |  | There is a hole punched through the drywall beside the door. |
| res-def-006 | defect | `trip_hazard_or_unlevel_floor` |  | The floor drops abruptly at the threshold creating a trip point. |
| res-def-007 | defect | `standing_water_or_poor_grading` |  | Water is pooling against the foundation after rain. |
| res-def-008 | defect | `missing_bathroom_tile_exposed_substrate` |  | Tile is missing from the shower wall exposing the backer board. |
| res-def-009 | defect | `missing_vanity_exposed_plumbing` |  | The vanity has been removed leaving the supply and drain lines exposed. |
| res-def-010 | defect | `tub_surround_or_shower_pan_damage` |  | The shower pan is cracked across the base. |
| res-deg-001 | degradation | `baseboard_wear_scuffs` |  | Baseboards are scuffed and nicked along the hallway. |
| res-deg-002 | degradation | `wall_scuffs_marks_or_dents` |  | Walls show scuff marks and a few small dents. |
| res-deg-003 | degradation | `deck_surface_weathering` |  | Deck boards have greyed and the surface looks weathered. |
| res-deg-004 | degradation | `patio_or_porch_surface_wear` |  | The covered patio slab coating is breaking down and worn. |
| res-deg-005 | degradation | `metal_carport_rust_corrosion` |  | Surface rust is spreading along the carport seams. |
| res-ele-001 | defect | `visible_electrical_risks` |  | Bare wiring is hanging out of an open junction box. |
| res-emp-001 | degradation (empty_filter) | `_(none)_` |  | Cabinet finish is worn through around the handles. |
| res-flr-004 | degradation | `hard_flooring_scratched_or_worn` |  | Wood floor shows surface scuffing and a worn sheen near the doorway. |
| res-inv-001 | upgrade (invalid_kind) | `_(none)_` |  | Cabinet finish is worn through around the handles. |
| res-mod-001 | modernization | `dated_or_older_windows` |  | Windows are older single-hung units that still operate. |
| res-mod-002 | modernization | `popcorn_or_acoustic_ceiling_texture` |  | The ceiling carries a sprayed popcorn texture throughout. |
| res-mod-003 | modernization | `dated_wood_paneling` |  | Walls are covered in dark wood paneling. |
| res-mod-004 | modernization | `dated_lighting_fixtures` |  | Light fixtures throughout are an older style. |
| res-mod-005 | modernization | `unfinished_basement_present` |  | The basement is unfinished open space. |
| res-moi-001 | defect | `water_stain_ceiling` |  | There is a brown ring-shaped stain spreading across the ceiling. |
| res-moi-002 | defect | `visible_mold_or_mildew` |  | Black fuzzy growth is spreading across the wall near the tub. |
| res-nm-001 | defect (no_match) | `_(none)_` |  | The smoke detector on the ceiling is chirping intermittently. |
| res-nm-002 | modernization (no_match) | `_(none)_` |  | The homeowner mentions the neighborhood has an active HOA. |
| res-pav-003 | degradation | `concrete_driveway_surface_wear` |  | Concrete driveway surface is pitted and rough from age. |
| res-str-001 | defect | `major_foundation_or_settlement_signs` |  | Large stair-step cracks run through the foundation wall. |
| res-str-002 | defect | `missing_or_damaged_handrails` |  | The stair railing is missing entirely along the open side. |
| res-str-003 | defect | `rotted_subfloor_or_structural_framing` |  | Floor framing is soaked and crumbling under the utility room. |

## holdout (60 cases)

| case | kind | gold | FLAG | observation |
|---|---|---|---|---|
| res-fan-001 _exhaust-trio_ | defect | `exhaust_fan_missing_or_broken` |  | There is no exhaust fan anywhere in this bathroom. |
| res-fan-002 _exhaust-trio_ | degradation | `exhaust_fan_grime_clogged` |  | The bathroom fan grille is packed with dust and grime. |
| res-fan-003 _exhaust-trio_ | modernization | `exhaust_fan_dated` |  | The exhaust fan cover is yellowed and dated-looking but appears to work. |
| res-vin-001 _flooring-vinyl-pair_ | defect | `vinyl_linoleum_torn_or_lifted` |  | The sheet vinyl is torn open in front of the range and curling away from the floor. |
| res-vin-002 _flooring-vinyl-pair_ | degradation | `vinyl_linoleum_worn_or_stained` |  | Linoleum is fully adhered but discolored and dingy across the room. |
| res-hdw-001 _hardware-pair_ | degradation | `door_hardware_worn` |  | The door lever finish is worn through and tarnished at the grip. |
| res-hdw-002 _hardware-pair_ | modernization | `door_hardware_dated_style` |  | Doors have shiny brass knobs that look dated but operate fine. |
| res-brk-001 _masonry-pair_ | defect | `mortar_joints_deteriorated` |  | Mortar between the bricks is crumbling out of the joints. |
| res-brk-002 _masonry-pair_ | degradation | `brick_weathered_or_discolored` |  | Brick face is stained and weathered though the joints look sound. |
| res-pav-001 _paving-pair_ | defect | `paving_heaved_or_trip_hazard` |  | A walkway slab has heaved up leaving a raised lip across the path. |
| res-pav-002 _paving-pair_ | degradation | `paving_cracked_or_settled` |  | The driveway has hairline cracking and some surface spalling, all flat. |
| res-pool-001 _pool-pair_ | defect | `inground_pool_empty_or_unserviceable` |  | The in-ground pool is drained and sitting empty. |
| res-pool-002 _pool-pair_ | degradation | `inground_pool_finishes_deteriorated` |  | Pool is full and running but the plaster is stained and the coping looks tired. |
| res-roof-001 _roofing-pair_ | defect | `roof_shingles_missing_or_curling` |  | Shingles are missing in patches and the edges are curling up along the ridge. |
| res-roof-002 _roofing-pair_ | degradation | `roof_shingles_aged_or_worn` |  | Roof surface looks weathered with green moss growth and thin granule cover. |
| res-van-001 _vanity-trio_ | defect | `vanity_damaged_or_water_stained` |  | The vanity base is swollen and delaminating where water has soaked the particle board. |
| res-van-002 _vanity-trio_ | degradation | `vanity_worn_finish` |  | Vanity cabinet finish is worn and faded but the box is solid. |
| res-van-003 _vanity-trio_ | modernization | `vanity_dated_style` |  | The bathroom vanity is an oak unit in working order that looks dated. |
| res-cab-004 | defect | `cabinets_damaged_or_water_stained` |  | Dark water staining runs down the cabinet box beside the dishwasher. |
| res-cab-005 | modernization | `cabinets_dated_style` |  | Cabinets are builder-grade flat-panel units that read dated but are intact. |
| res-car-002 | degradation | `worn_or_stained_carpet` |  | Bedroom carpet looks heavily worn with dark marks near the closet. |
| res-def-011 | defect | `missing_base_cabinets_exposed_subfloor` |  | Lower cabinets are gone, exposing bare subfloor and plumbing. |
| res-def-012 | defect | `unfinished_interior_wall_osb_exposed` |  | An interior wall is finished only in bare OSB sheathing. |
| res-def-013 | defect | `pest_or_rodent_evidence` |  | Rodent droppings and chewed insulation are visible in the corner. |
| res-def-014 | defect | `retaining_wall_failure_or_missing_section` |  | A section of the retaining wall has collapsed outward. |
| res-def-015 | defect | `bare_or_missing_finish_flooring` |  | The room is down to bare subfloor with no finish flooring. |
| res-def-016 | defect | `missing_or_damaged_caulk_at_tub_or_shower` |  | Caulk at the tub edge is cracked away and mouldy. |
| res-def-017 | defect | `damaged_or_unsafe_deck_or_porch` |  | Deck boards are rotted through and the railing is loose. |
| res-def-018 | defect | `ceiling_cracks_or_sagging` |  | The ceiling is sagging with a crack opening along the joint. |
| res-def-019 | defect | `mold_or_mildew_visible_bathroom` |  | Mildew is growing across the bathroom grout lines and ceiling. |
| res-def-020 | defect | `garage_or_basement_damage` |  | The basement wall is cracked with white efflorescence and damp patches. |
| res-deg-006 | degradation | `floor_dirty_or_heavily_soiled` |  | The floor is heavily soiled and grimy underfoot. |
| res-deg-007 | degradation | `exterior_door_paint_failure` |  | Paint on the front door is chipped and peeling at the edges. |
| res-deg-008 | degradation | `gutter_maintenance_needed` |  | Gutters look like they need cleaning out. |
| res-deg-009 | degradation | `dirty_or_grimy_window_screens` |  | Window screens are caked with dirt and debris. |
| res-deg-010 | degradation | `shed_exterior_paint_failure` |  | The shed's paint is peeling off in sheets. |
| res-ele-002 | defect | `bathroom_gfci_missing_or_damaged` |  | The outlet beside the bathroom sink has no GFCI protection. |
| res-emp-002 | defect (empty_filter) | `_(none)_` |  | Shingles are missing in patches along the ridge. |
| res-flr-003 | defect | `hard_flooring_broken_or_warped` |  | A section of laminate flooring has warped and lifted at the seam. |
| res-inv-002 | safety (invalid_kind) | `_(none)_` |  | Shingles are missing in patches along the ridge. |
| res-inv-003 | _(empty)_ (invalid_kind) | `_(none)_` |  | The vanity base is swollen from water. |
| res-lnd-003 | degradation | `yard_debris_overgrown_leaves` |  | Fallen leaves and yard debris have piled up across the lawn. |
| res-lnd-004 | modernization | `tree_stump_present_in_yard` |  | A cut tree stump is still sitting in the middle of the side yard. |
| res-mod-006 | modernization | `dated_interior_doors` |  | Interior doors are hollow-core six-panel units. |
| res-mod-007 | modernization | `staging_or_decluttering_opportunity` |  | The room is cluttered with personal belongings throughout. |
| res-mod-008 | modernization | `dated_wallpaper_present` |  | Busy patterned wallpaper covers the kitchen walls. |
| res-mod-009 | modernization | `outdated_kitchen_finishes` |  | The kitchen finishes overall feel out of date. |
| res-mod-010 | modernization | `layout_modernization_opportunity` |  | The closed-off layout could be opened up between the rooms. |
| res-moi-003 | defect | `active_water_damage_bathroom` |  | There is standing water on the bathroom floor and the drywall is wet. |
| res-nm-003 | degradation (no_match) | `_(none)_` |  | The mailbox post at the street leans slightly. |
| res-nm-004 | defect (no_match) | `_(none)_` |  | A parked vehicle blocks the view of the lower wall. |
| res-plm-003 | defect | `bathroom_plumbing_visible_issue` |  | The shutoff valve under the bathroom sink is leaking onto the cabinet floor. |
| res-pnt-005 | modernization | `bathroom_paint_refresh_recommended` |  | Bathroom paint is sound but the color reads dull and dated. |
| res-roof-003 | defect | `roofline_water_damage_suspected` |  | Brown water staining runs along the fascia below the gutter line. |
| res-roof-004 | degradation | `roof_shingles_aged_or_worn` |  | The roof is covered in leaf debris and the shingle color has faded unevenly. |
| res-sid-004 | defect | `exposed_sheathing_or_missing_siding` |  | Bare sheathing is exposed where siding panels are absent on the side wall. |
| res-str-004 | defect | `interior_wall_stripped_to_studs` |  | Drywall has been stripped off leaving bare studs and insulation exposed. |
| res-str-005 | defect | `boarded_up_entry_or_window` |  | A window opening has been boarded over with plywood. |
| res-vct-003 | defect | `vanity_countertop_damage` |  | There is a crack running through the vanity countertop beside the basin. |
| res-vin-003 | defect | `vinyl_linoleum_torn_or_lifted` |  | Resilient flooring has worn through to the backing beside the washer. |

## Legacy baseline mapping

Each resolution case carries a `legacy` block: the kind v1 Pass 2c would have
emitted (the frozen baseline put 110/110 degradation-gold decisions on
`defect_or_damage`) and the v1 catalog item it would have resolved to — the
split parent. That defines the comparable population for the no-regression gate.

