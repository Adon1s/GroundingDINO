# The wrong-subject shortcut: bounded follow-up proposal, or deferral (2026-09-10)

## The defect is real and human-confirmed

The lexical shortcut can fire on a support term that names a subject the sentence does not have, bypassing the model and asserting a claim about the wrong component. Steven confirmed two instances on photos:

- C5, `redfin_125779232` photo_007: "The floor has discoloration and staining around the tub and vanity." fired on `tub$` and asserted stained bath fixtures. His judgment: the floor is stained, the fixtures are not.
- D1, `redfin_10952874` photo_031: "Ceiling above the shower has discoloration and staining." fired on `shower` and asserted the same. His judgment: the ceiling is slightly stained, the fixtures are not.

Rule E's rejection does not make this go away. Rule E was rejected because it was too broad, not because the defect was disproved.

## Why Rule E failed, precisely

Rule E asked whether any firing term matched the sentence head. That flags 25 rows, and on most of them the mismatch is harmless: "Walls have holes" fires on `hole` in the predicate and the item is right. Sending those to the model converted three plausible selections into nulls and left four of eleven consequential rows unstable. The signal was mismatch, and mismatch is common and mostly benign.

## The narrower hypothesis

The harmful cases share a stronger property than head mismatch: **the firing term names a catalog subject that is different from the subject the sentence is about, and the candidate list contains an item for the sentence's actual subject.** In C5 the sentence is about a floor, `tub$` names a fixture, and two flooring items sat at ranks 2 and 4. In D1 the sentence is about a ceiling, `shower` names a fixture, and a bathroom-paint item was present.

That is a two-sided test, not a one-sided one. Rule E asked only "did the term miss the head?". The narrower rule asks "did the term name a *competing catalog subject* while an item for the real subject was available?".

## What a bounded test would look like

- **Population.** Enumerate offline, over the 640 stored shortcut rows, every row where (a) every firing term is a bare subject noun, (b) that noun is the `atomic_claim` subject of some *other* item, and (c) an item whose claim subject matches the sentence head is present in the same candidate list. My expectation from the Rule E data is that this is well under 25 rows and includes C5 and D1; if the enumeration returns fewer than about 5 rows or misses either confirmed case, the hypothesis is dead and the work stops there.
- **Validation.** The two human-confirmed rows must move to the competing item or to null. The rows Rule E broke (caulk at the tub joints, roof-edge staining, foundation seams, the built-in under the windows) must be untouched, because none of them has a competing same-subject item in its list. The 615 rows the rule does not flag must be untouched by construction.
- **Cost.** The enumeration is free and offline. Only if it passes would any model call be needed, and then only on the flagged rows.

## Recommendation: propose, do not run now

The hypothesis is clear and the validation cases exist, which is the bar Steven set. But two things argue for recording it rather than running it in this closeout:

1. It needs a subject index the catalog does not currently expose. Deciding that a bare noun "names a catalog subject" means matching support terms against `atomic_claim.subject` across 129 items, which is new derived structure. That is design work, and Steven has excluded another open-ended design cycle from this closeout.
2. The measured billing consequence of the two confirmed cases is currently **zero**. C5's billing was suppressed by dedup collision; D1's condition is not one the reviewed record prices. The defect is a correctness and naming problem, not a dollar one, so it does not force action now.

**Deferred, with a trigger:** revisit when either a wrong-subject shortcut is found that produces a real billing, or the catalog gains a subject index for another reason. The enumeration in step one is the first thing to run when it is revisited, and it costs nothing.
