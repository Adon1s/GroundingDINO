# Pass 2d photo review — Steven's answers (2026-09-09)

Human evidence, recorded alongside existing dispositions (none replaced). Steven's words are verbatim under each card; the axis mapping (A photo truth / B sentence fidelity / C match-to-sentence) is the coordinator's reading and is marked as such. No stage is attributed here.

### C1 — row 26, "The shower trim is old."
> yes bathroom finishes description fits. the main outdated items are the sink and the bathtub. the trim is an issue but it is more minor than the 2 former.

Mapping: A = bathroom finishes true (sink, bathtub); trim true but minor. B = faithful in content, misplaced in emphasis. C = better candidate was `outdated_bathroom_finishes` (rank 1). Existing record (v1 `terra_evidence_inconclusive`) preserved. Open: whether the "trim" is plumbing trim or millwork.

### C2 — row 3, "Wood-look plank flooring has discoloration, patchy areas, and scuffs." (photo_029)
> The observation is not true. it is hallucinating because of the light entering the room from the windows. it might not be preventable this type of issue with the LLM description. Its just like how it says ceiling stain when its the light and / or shadow cast from the ceiling fan light.

Mapping: A = none. B = wrong (light artefact). C = not judgeable; the sentence itself is false. Existing record: 2026-08-28 retag on the aggregated condition, answer `mechanism_only`, note "I see stains not scuffs"; v1.1 `misnamed_but_warranted`. **Tension to resolve:** that condition spans photo_025 and photo_029; the earlier note may describe photo_025. Both records kept.

### C3 — row 16, "Wall surfaces have heavy staining and deteriorated wallboard or paneling."
> This is true there are heavy stains, deteriorated paint, etc. pretty much everything you asked is true. I would say the selected is better than the alternative

Mapping: A = multiple true: staining, deteriorated paint, deteriorated wallboard (not offered), moisture (not offered). B = faithful. C = the model's pick (`wall_scuffs_marks_or_dents`) reasonable and better than the nominee (`peeling_or_discolored_paint`). CAP-014 disposition preserved; this is evidence against the worklist's nominee on this row.

### C4 — row 18, "Bathroom walls and ceiling are stained."
> So this is one of those tricky photos where it is a shot from outside the bathroom so the bathroom is not the focus. with that being said i would not say that you can clearly see staining on the walls or ceiling. also why the heck would Wall scuffs, marks, or minor dents match to Bathroom walls and ceiling are stained ever? Thats pretty bad. I do not see peeling paint either so the alternative is also bad. the issues are real but they are mostly outside of the bathroom in this shot. since you cannot see much of the bathroom here.

Mapping: A = staining not clearly visible; no peeling paint; real issues outside the bathroom (cannot tell for the bathroom). B = unsupported by the photo. C = the model's pick unreasonable for the sentence; the nominee also poor; no listed candidate fits. Two independent problems on one row. CAP-014 disposition preserved.

### C5 — row 5, "The floor has discoloration and staining around the tub and vanity."
> Staining is not the main issue here, outdatedness is. But there is some staining around the tub and vanity as described. I will say I do like the alternative match "Worn or stained flooring; worn or stained vinyl/linoleum" more since it accurately describes the floor, not the fixtures having the staining. The fixtures are not stained they are outdated.

Mapping: A = floor staining true (minor); fixtures not stained; fixtures outdated (a modernization truth, not offered because the bullet was kinded degradation). B = faithful but secondary. C = better candidate was a flooring item; the shortcut's subject (fixtures stained) is not what the photo shows. v1.1 `misnamed_but_warranted` on the aggregated condition preserved.

### C6 — declined, "The range appears older."
> I would say the candidate is the best option here as they are clearly outdated but functional.

Mapping: A = range dated. B = faithful. C = `appliances_dated_or_basic` fit; the decline left a true, listed condition unrecorded.

### C7 — declined, "The faucet has an older style."
> I think the candidate is good here since it is clearly outdated. It is quite functional though and im not positive i would cost up the bathroom for renovation. but at the same time i wont refute terras claim.

Mapping: A = faucet dated. B = faithful. C = `outdated_bathroom_finishes` fit; decline left a true, listed condition unrecorded. Work axis: uncertain whether it warrants costing the bathroom; recorded as uncertainty, not as a verdict.

### C8 — declined, "Kitchen is functional and very dated."
> The floor, appliances, cabinets are all outdated. Functional but very outdated especially the floor. I would definitely cost it for renovation. I think general decor is okay. since the room is generally outdated. but im not sure if thats the best wording for other cases outside of this one.

Mapping: A = floor, appliances, cabinets all dated (three listed items true: `older_flooring_style` rank 4, `appliances_dated_or_basic` rank 2, `cabinets_dated_style` rank 7); general decor acceptable for this photo, doubtful as general wording. B = faithful but generic. C = listed candidates fit; the decline left true conditions unrecorded; a one-item contract could not carry three anyway.

## Clarifications (Steven, 2026-09-09)

### C2 (row 3)
> There is no scuffing on the floor anywhere that i can see. there is paint on the floor however.

Mapping: A = paint marks on the floor (contamination) true; scuffs absent; "discoloration / patchy areas" a light artefact. B = partly wrong: "scuffs" false, "paint marks" (photo_025 sentence) true. C = the pick (scratched/scuffed hard flooring) is not supported by the photo; a staining/contamination item fits the real condition. **The tension with the 2026-08-28 retag is resolved: "stains not scuffs" and this answer agree.** Preserved: v1.1 misnamed_but_warranted.

### C1 (row 26)
> So between the tub and the toilet there is some missing floor trim. The rest of the trim appears present

Mapping: A = a missing section of base trim between tub and toilet (millwork, absent, not merely old); the rest present; sink and bathtub dated. B = partly wrong: "old" for a section that is missing. C = neither listed item names missing trim; bathroom finishes still the better listed match for the photo. **Bears on CAP-013 (missing or incomplete interior base trim; deferred, schema-blocked under Q-6, evidence bar met on two properties): this would be a third property. Disposition preserved; recorded as evidence only.**

## Round 2 (Steven, 2026-09-09)

### D1 — redfin_10952874/photo_031, "Ceiling above the shower has discoloration and staining." (shortcut on "shower")
> The ceiling above the shower is slightly stained yes. the fixtures are not stained

Mapping: A = ceiling staining true (slight; moisture-type item was not offered); fixtures NOT stained. B = faithful. C = the shortcut's pick (fixtures stained) is not what the photo shows; the term fired on a location word. **Second human-confirmed location-word shortcut (with C5).**

### D2 — redfin_126418713/photo_025, "Ceiling has a heavy decorative swirl plaster texture." (declined; popcorn at rank 1)
> The ceiling has a very interesting pattern. i have never seen anything like it, its not popcorn. I honestly would not cost it. But I cant complain too much the model says outdated ceiling pattern or something. but popcorn is straight up false. I also dont mind it being ignored by the model since the room in general should not be renovated

Mapping: A = decorative swirl plaster true; popcorn false. B = faithful. C = declining was correct; the popcorn item would have been a false claim. Work axis: would not cost it. **Recorded as the CORRECT-NULL CONTROL.** CAP-016 disposition preserved (naming question; the sentence itself is fine).

### D3 — redfin_11185681/photo_027, "Painted cabinets appear older and have dated door styling." (declined; cabinets item never offered)
> Ah I see yes its true this is not a kitchen. It is a laundry room or some extra room. there is a washer and drier. yes the cabinets are outdated but its true this is a case of a scene error. an understandable one to some extent but still an error.

Mapping: A = cabinets present and dated; room is a laundry/utility room. B = faithful. C = declining was reasonable given what was offered; the true item was not in the list. **Distinguish two things:** (i) the scene classification (laundry/utility, not kitchen) was CORRECT; (ii) the loss comes from cabinets_dated_style being eligible only in kitchen scenes, a catalog scene-eligibility restriction. The error is eligibility breadth, not Pass 2c/scene and not Pass 2d. D6 (cabinets no change) preserved; recorded as evidence toward its trigger.

### D4 — redfin_80925528/photo_044, "The wall finish behind the sink appears patched." (row 21)
> Yes the claim here is correct. The wall is patched or something. I can say its missing some paint for certain

Mapping: A = patched wall true; paint missing true. B = faithful. C = the pick (wall scuffs) accepted as correct by the reviewer; peeling paint also true. Two true conditions; the worklist's "the 2a text mislabels peeling as patching" is not supported (patching is real). CAP-014 preserved.
