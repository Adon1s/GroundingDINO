# Pass 2d photo review — round 2, four cards (2026-09-09, rules v2)

Same three axes as round 1, answered separately: **A** photo truth (multi-select; "none" and "cannot tell" allowed; claims marked *(not in list)* were never offered to the model), **B** sentence fidelity, **C** match to the sentence with the photo set aside. No answer assigns a stage or closes anything; existing dispositions shown and preserved. Photos under `renointel-prod\public\images\properties\`.

Why these four: round 1's three declines were the rows a reader thought had a fitting candidate, so they cannot say whether declines the reader called *correct* are also losing conditions (D2, D3). C5 showed one shortcut firing on a location word; D1 is the other such case in the corpus. D4 is the cleanest CAP-014 sentence, with a full eight-item list and no guardrail drop, to set beside C3 and C4.

---

### D1 — `redfin_10952874/photo_031.jpg` · bathroom · shortcut on a location word
**Sentence:** "Ceiling above the shower has discoloration and staining."
**Model chose (rank 1, by lexical shortcut on "shower"; model never called):** `bath_fixtures_stained_or_worn` — *bath fixtures: heavy staining or worn finish, undamaged*.
**Also offered:** `bathroom_paint_peeling_or_worn` — *bathroom paint: peeling or worn finish without moisture evidence*; `wall_scuffs_marks_or_dents`.
**Not in list (kind defect):** `water_stain_ceiling` — *ceiling: water staining indicating moisture intrusion*.
**Existing record:** none.

- A. True of the photo (any): [ ] stained or worn tub/shower/fixtures [ ] discoloured or stained ceiling, moisture-type *(not in list)* [ ] peeling or worn paint [ ] none [ ] cannot tell
- B. Sentence fidelity: faithful / understated / overstated / wrong / cannot tell
- C. Match to the sentence: reasonable / a listed candidate was better (name it) / no listed candidate fits / cannot tell

---

### D2 — `redfin_126418713/photo_025.jpg` · declined; a reader called the decline correct
**Sentence:** "Ceiling has a heavy decorative swirl plaster texture."
**Model chose:** nothing. Rank 1 at 0.767 was `popcorn_or_acoustic_ceiling_texture` — *ceiling: sprayed or stippled popcorn/acoustic texture* — a near-shortcut (margin 0.118, no term fired).
**Existing record:** CAP-016 (swirled plaster texture) is deferred; its recorded swirl example was Terra-supported and billed, and the open question there is naming. Preserved.

- A. True of the photo (any): [ ] swirl/decorative plaster texture [ ] popcorn or acoustic spray texture [ ] a dated ceiling of some other kind [ ] none [ ] cannot tell
- B. Sentence fidelity: faithful / understated / overstated / wrong / cannot tell
- C. Match to the sentence: declining was reasonable / the popcorn item fit well enough to carry it / cannot tell
- Work axis (optional): would you cost this ceiling for work? yes / no / not sure

---

### D3 — `redfin_11185681/photo_027.jpg` · declined; the fitting item was scene-gated
**Sentence:** "Painted cabinets appear older and have dated door styling."
**Model chose:** nothing. Rank 1 was `dated_overall_decor_style` (generic, $0 display line). `cabinets_dated_style` — *cabinets: dated style or basic grade, functional* — was **never offered**: it is scoped to kitchen scenes and this photo was classified as a different scene.
**Existing record:** none; D6 leaves `cabinets_dated_style` unchanged.

- A. True of the photo (any): [ ] cabinets present and dated [ ] cabinets present, not dated [ ] no cabinets visible [ ] cannot tell
- What room is this? (short note)
- B. Sentence fidelity: faithful / understated / overstated / wrong / cannot tell
- C. Match to the sentence: declining was reasonable given what was offered / the decor item should have carried it / cannot tell

---

### D4 — `redfin_80925528/photo_044.jpg` · kitchen · worklist row 21 (CAP-014), full list, no drops
**Sentence:** "The wall finish behind the sink appears patched."
**Model chose (rank 1):** `wall_scuffs_marks_or_dents` — *walls: scuffs, marks, or minor dents from use*.
**Also offered (rank 2):** `peeling_or_discolored_paint` — *interior paint: peeling, bubbling, or visibly aged finish*.
**Not in list (kind defect):** `damaged_drywall_or_cracks` — *drywall: cracks, holes, or impact damage (poor repairs)*.
**Existing record:** none on this row; the worklist reviewer wrote that the photo shows paint peeled to bare plaster and that "patched" was the wrong word. CAP-014 disposition preserved.

- A. True of the photo (any): [ ] patch or repair compound visible *(not in list as such)* [ ] paint peeled or missing to substrate [ ] scuffs / marks / dents [ ] none [ ] cannot tell
- B. Sentence fidelity: faithful / understated / overstated / wrong / cannot tell
- C. Match to the sentence: reasonable / better candidate was `peeling_or_discolored_paint` / no listed candidate fits / cannot tell

---

After these four, no further photos are needed for Phase A. What they add: whether "correct" declines also lose conditions (D2, D3), whether the location-word shortcut generalises beyond C5 (D1), and a third CAP-014 data point on a clean sentence (D4).
