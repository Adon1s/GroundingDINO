# Pass 2d photo review — eight cards (2026-09-09, rules v2)

Photos: `C:\Users\Steven\IntelliJProjects\renointel-prod\public\images\properties\<property>\<photo>`. All eight exist.

**Three axes on every card, answered separately.**
- **A. Photo truth.** Which listed claims are true of the photo? Multi-select; "none" and "cannot tell" always allowed. Claims marked *(not in list)* were never offered to the model; they are here so photo truth is complete.
- **B. Observation fidelity.** Was the sentence the model was given faithful to the photo? faithful / understated / overstated / wrong / cannot tell.
- **C. Match to the sentence.** Judging only the sentence and the candidates offered, with the photo set aside: was the model's pick a reasonable reading? reasonable / a listed candidate was better (name it) / no listed candidate fits / cannot tell.

**Rules.** No answer assigns a stage by itself; stage attribution is a later judgment across cards, recorded separately. No card or pair of cards closes or opens the Pass 2d selection lane. Where a human disposition already exists it is shown and preserved; your answer is a new judgment recorded alongside it. "Cannot tell" is a valid final answer.

---

### C1 — `redfin_11079485/photo_005.jpg` · bathroom · row 26
**Sentence given to the model:** "The shower trim is old."
**Model chose (rank 3):** `dated_interior_trim` — claim *interior trim: plain, thin, or builder-grade trim package*.
**Also offered (rank 1):** `outdated_bathroom_finishes` — claim *bathroom finishes: dated tile, vanity, or fixtures*.
**Existing human record:** v1 verdict `terra_evidence_inconclusive` on this card; no v1.1 label. Preserved.

- A. True of the photo (any): [ ] dated plumbing trim (escutcheon, handles, spout) [ ] dated millwork (baseboard, casing) [ ] dated tile / vanity / fixtures generally [ ] none [ ] cannot tell
- B. Sentence fidelity: faithful / understated / overstated / wrong / cannot tell
- C. Match to the sentence: reasonable / better candidate was `outdated_bathroom_finishes` / no listed candidate fits / cannot tell

---

### C2 — `redfin_10952874/photo_029.jpg` · bedroom · row 3
**Sentence:** "Wood-look plank flooring has discoloration, patchy areas, and scuffs."
**Model chose (rank 3):** `hard_flooring_scratched_or_worn` — *hard flooring finish: scratches, scuffs, or worn finish on intact flooring*.
**Also offered (rank 1):** `worn_or_stained_flooring` — *flooring: wear, staining, or discoloration without structural concern*.
**Existing human record:** your 2026-08-28 retag answer `mechanism_only` with note "I see stains not scuffs" (prior verdict overstated); v1.1 `misnamed_but_warranted`. Preserved. (Sibling `photo_025.jpg` resolved the same way.)

- A. True of the photo (any): [ ] scratches / scuffs / worn finish [ ] staining / discoloration [ ] patching or repair marks [ ] none [ ] cannot tell
- B. Sentence fidelity: faithful / understated / overstated / wrong / cannot tell
- C. Match to the sentence: reasonable / better candidate was `worn_or_stained_flooring` / both fit about equally / no listed candidate fits / cannot tell

---

### C3 — `redfin_125970550/photo_018.jpg` · basement/utility · row 16
**Sentence:** "Wall surfaces have heavy staining and deteriorated wallboard or paneling."
**Model chose (rank 2):** `wall_scuffs_marks_or_dents` — *walls: scuffs, marks, or minor dents from use*.
**Also offered (rank 1):** `peeling_or_discolored_paint` — *interior paint: peeling, bubbling, or visibly aged finish*.
**Not in list (kind defect):** `damaged_drywall_or_cracks` — *drywall: cracks, holes, or impact damage*; `water_stain_ceiling` — *ceiling: water staining indicating moisture intrusion*.
**Existing human record:** none on this row. CAP-014 disposition (approvals_v2, D5) already records the scuffs-vs-paint rows as Pass 2d selection and two rows as Pass 2c kind exclusion; that disposition is preserved and this card adds evidence to it, not a re-ruling.

- A. True of the photo (any): [ ] scuffs / marks / dents from use [ ] peeling or aged paint [ ] deteriorated or damaged wallboard *(not in list)* [ ] water staining / moisture *(not in list)* [ ] none [ ] cannot tell
- B. Sentence fidelity: faithful / understated / overstated / wrong / cannot tell
- C. Match to the sentence: reasonable / better candidate was `peeling_or_discolored_paint` / no listed candidate fits / cannot tell

---

### C4 — `redfin_166147710/photo_038.jpg` · bathroom · row 18
**Sentence:** "Bathroom walls and ceiling are stained."
**Model chose (rank 3):** `wall_scuffs_marks_or_dents` — *walls: scuffs, marks, or minor dents from use*.
**Also offered:** rank 1 `bathroom_paint_peeling_or_worn` — *bathroom paint: peeling or worn finish without moisture evidence*; rank 2 `bath_fixtures_stained_or_worn` — *bath fixtures: heavy staining or worn finish*.
**Not in list (kind defect):** `water_stain_ceiling` — *ceiling: water staining indicating moisture intrusion*.
**Existing human record:** none on this row; CAP-014 disposition as in C3, preserved.

- A. True of the photo (any): [ ] scuffs / marks / dents [ ] peeling or worn paint, no moisture [ ] stained fixtures [ ] moisture staining on walls or ceiling *(not in list)* [ ] none [ ] cannot tell
- B. Sentence fidelity: faithful / understated / overstated / wrong / cannot tell
- C. Match to the sentence: reasonable / better candidate was ______ / no listed candidate fits / cannot tell

---

### C5 — `redfin_125779232/photo_007.jpg` · bathroom · row 5
**Sentence:** "The floor has discoloration and staining around the tub and vanity."
**Model chose (rank 1, by lexical shortcut on the word "tub"; the model was never called):** `bath_fixtures_stained_or_worn` — *bath fixtures: heavy staining or worn finish, undamaged*.
**Also offered:** rank 2 `worn_or_stained_flooring`; rank 4 `vinyl_linoleum_worn_or_stained` — *flooring: wear, staining, or discoloration*.
**Existing human record:** v1.1 `misnamed_but_warranted` on the aggregated condition (four sentences, this one included). Preserved. The billing was suppressed by dedup; this card is about the mechanism.

- A. True of the photo (any): [ ] stained or worn floor [ ] stained or worn tub / fixtures [ ] stained or worn vanity [ ] none [ ] cannot tell
- B. Sentence fidelity: faithful / understated / overstated / wrong / cannot tell
- C. Match to the sentence (the shortcut's pick, judged as if a reader had chosen it): reasonable / better candidate was a flooring item / no listed candidate fits / cannot tell

---

### C6 — `redfin_11185681/photo_024.jpg` · kitchen · declined
**Sentence:** "The range appears older."
**Model chose:** nothing (null).
**Offered (rank 4):** `appliances_dated_or_basic` — *kitchen appliances: dated style or basic grade, functional*.
**Existing human record:** none.

- A. True of the photo (any): [ ] range is dated or basic [ ] other appliances dated or basic [ ] none [ ] cannot tell
- B. Sentence fidelity: faithful / understated / overstated / wrong / cannot tell
- C. Match to the sentence: declining was reasonable / `appliances_dated_or_basic` fit the sentence / cannot tell

---

### C7 — `redfin_25809814/photo_027.jpg` · bathroom · declined
**Sentence:** "The faucet has an older style."
**Model chose:** nothing.
**Offered (rank 1):** `outdated_bathroom_finishes` — *bathroom finishes: dated tile, vanity, or fixtures*.
**Existing human record:** none.

- A. True of the photo (any): [ ] faucet dated [ ] vanity dated [ ] tile dated [ ] none [ ] cannot tell
- B. Sentence fidelity: faithful / understated / overstated / wrong / cannot tell
- C. Match to the sentence: declining was reasonable / `outdated_bathroom_finishes` fit the sentence / cannot tell

---

### C8 — `redfin_80925528/photo_045.jpg` · kitchen · declined
**Sentence:** "Kitchen is functional and very dated."
**Model chose:** nothing.
**Offered:** rank 1 `dated_overall_decor_style` — *overall decor: dated finish selections* (generic; routes to a $0 display line); rank 2 `appliances_dated_or_basic`; rank 7 `cabinets_dated_style` — *cabinets: dated style or basic grade, functional*.
**Existing human record:** none.

- A. True of the photo (any): [ ] cabinets dated [ ] appliances dated [ ] counters or finishes dated [ ] general decor only [ ] none [ ] cannot tell
- B. Sentence fidelity: faithful / understated / overstated / wrong / cannot tell
- C. Match to the sentence: declining was reasonable / a listed candidate fit (name it) / cannot tell

---

## What the review produces

Eight rows with human-grade answers on three separate axes, recorded alongside any existing disposition. Read together, and only together, they inform four questions, each of which remains a judgment made after the review rather than a rule fired by one card:

1. Whether any row shows a human-confirmed gap between the sentence's best listed match and the model's pick (axis C), and separately whether that gap mattered for the photo (axis A).
2. Whether the CAP-014 rows' trouble sits in the sentence (B), the offered list (C), or the photo truth the list could not express (A), as evidence toward the recorded trigger, with the disposition unchanged.
3. Whether the "tub" shortcut asserted a subject the photo does not support (A against the chosen claim).
4. Whether declines on C6–C8 left true, listed conditions unrecorded (A together with C).

Stage attribution and any change to the Phase B menu follow from the pattern across the eight, are proposed separately, and are yours to accept. The remaining seventeen null rows are not required for this review.
