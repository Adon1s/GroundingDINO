# Pass 2d — combined check and implementation decision (2026-09-10)

Combined arm: the 25 Rule E rows resolved with the arm-N modified prompt, 5 replicas, 125 local Qwen calls, 0 failures. Rule E changes only these rows; N's effect on every other row was measured separately in the N arm.

## The hypothesis was wrong

I predicted that N would recover the rows Rule E sends to null, because S's failure mode is the null and N reduces nulls. It does not.

| Row Rule E sent to null | S alone | S + N |
|---|---|---|
| Caulk at the tub joints | null 5/5 | **null 5/5** |
| Ceiling above the shower (D1) | null 3/5 | **null 5/5** (more stably null) |
| Ceiling and wall near the tub | null 4/5 | **null 5/5** |
| Seams along the foundation | null 4/5 | **null 5/5** |
| Dark staining along the roof edge | null 3/5 | roof shingles 3/5 — recovered, still unstable |

One of five recovered, and only at three of five. Worse, the combination introduced a loss that neither change caused alone: "The built-in under the windows looks dated" held `dated_or_older_windows` under S alone and goes null 5/5 under S+N.

The two changes do not compose the way I expected. N helps a sentence that names a specific instance of a candidate's subject. The rows Rule E strands are ones where the sentence's subject has no catalog item at all, so there is no instance rule to apply.

## Net effect of shipping both

Against the shortcut baseline on those 25 rows: three stable gains (the floor sentence to `worn_or_stained_flooring` at 5/5, human-confirmed; lower cabinets to `cabinets_worn_finish`; outlets to `dated_electrical_outlets_switches`), against five stable nulls of which three are defensible removals of a wrong subject and two are outright losses, plus the new built-in loss. Roughly a wash, with more moving parts.

## Decision: implement N alone

**N.** Two of three reviewed missed matches fixed at 5/5, the third correctly left alone. Both must-stay-null controls held at 5/5, including the swirl-plaster ceiling where Steven judged the alternative claim false. Zero movement on 40 correct declines. Two of 40 non-null controls moved, at the stated limit, one gain and one loss. It is a two-line addition to `PASS_2D_USER_PROMPT_TEMPLATE`, which bumps the prompt version and therefore needs its own validation and a benchmark re-score before it ships.

**Not S.** The wrong-subject shortcut is a real, human-confirmed defect, but Rule E is the wrong instrument for it: it converts a wrong-subject selection into no selection about as often as into a right one, and the combined check removed the last reason to expect that to improve. A narrower rule scoped to the specific failure — a firing bare noun that names a *different* catalog subject than the sentence's head, rather than any head mismatch — is the shape worth designing, and that is new design work, not a re-run.

**Not the resolver.** Breached both control limits and produced a stable false claim on the row Steven pre-judged.

## Tokens

Local Qwen 412 calls / 292,482 tokens across all arms. GPT-5.6 Terra 238 calls / **162,893 of 2,500,000 authorized (6.5%)**, 2,337,107 remaining. No Terra verification calls. No production artifact touched, no catalog change, no arm R, nothing implemented.
