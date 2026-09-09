# Photo and lineage review brief — catalog audit Session 2 (separate session)

Date: 2026-09-01. This brief is self-contained. Read it, the packet it names, and the photos; nothing else is required.

## Mission

Adjudicate, one packet row at a time, whether the evidence photo(s) and the recorded pipeline lineage support a stated claim. You are the reviewer for catalog-audit Session 2; the audit itself (diagnosis, catalog proposals) happens elsewhere and must not be attempted here. Describe what is physically visible, compare it with the claim under test and with what the pipeline recorded, and record one adjudication per row.

## Authority

- You may create and edit exactly one repository file: `reports/catalog_audit_photo_review.json`.
- Do not modify any other file. In particular do not touch `reports/catalog_audit_photo_review_packet.json`, `reports/catalog_audit_proposals.json`, `reports/catalog_audit_evidence.json`, the catalogs, the migration decisions, prompts, or runtime code.
- Do not run the pipeline, the embeddings sidecar, or any model provider. Do not commit.
- Do not read `reports/catalog_audit_proposals.json` or `docs/PROPOSAL_catalog_audit_20260901.md`; the packet already carries everything you need, and those files hold hypotheses that must not steer the adjudication.

## Inputs

| Input | Path | Identity |
|---|---|---|
| Review packet | `reports/catalog_audit_photo_review_packet.json` | sha256 `7b4621dfe6c72aa42369c2f5ef7051dfecbec05b6333a80651b1b57a5b37bc80`; 223 rows, 191 photos |
| Photo root | `C:\Users\Steven\IntelliJProjects\renointel-prod\public\images\properties\<property_key>\<photo_key>` | every packet photo carries its absolute `path` and `sha256` |
| Optional background | `docs/catalog_audit_program/00_OVERALL_CONTEXT.md` sections 7 to 9 | evidence discipline only |

Verify the packet hash first (`Get-FileHash -Algorithm SHA256 reports\catalog_audit_photo_review_packet.json`). If it differs, stop and report; do not review a drifted packet.

## Packet structure

`rows[]` is the work list, sorted by `row_id`. Each row has:

- `row_id`, `row_type` (`unit`, `coverage`, or `control`), `priority` (`required` or `optional`).
- `claim_under_test`: the sentence you adjudicate against. For `unit` rows it is the cluster's claim; for `coverage` rows it is the gold finding; for `control` rows it is the catalog item's own claim text.
- `photos.photos[]`: `path`, `sha256`, `property_key`, `photo_key`. `photos.status` is `available` for every row in this packet.
- `lineage`: what the pipeline recorded. For runtime units: `cases[]` (the review-queue case: observations per issue with the frozen Pass 2d candidate list and the resolved item, `terra_verdict` and `terra_rationale`, `human_truth`, the latest attribution and its rationale, any reviewer note) and `leads[]` (a factorized-verifier lead with its `observed_description`, `derived_class`, `stored_verdict`, and candidate list). For gold units: `gold` (the finding, matching decision and note, covering condition/item if any, attribution and rationale) plus `gold_photo_resolutions` (what Pass 2d resolved on that photo).
- Unit rows also carry `unit_role` (`support`, `counterexample`, `context`), `evidence_class` (`human_review`, `gold_reference`, `model_judge`), `implicated_items`, `item_claims`, `item_claim_text`.
- Control rows carry `control_role` (`positive_use`, `agreement`, `correct_rejection`, `hallucination`), `item_id`, `human_truth`, `terra_verdict`, `reviewer_note`.

Workload: 163 required rows (55 unit, 45 control, 63 coverage) over 134 distinct photos; 60 optional rows over 57 further photos. Do every required row. Do optional rows if budget allows, required first.

## Procedure per row

1. Read the row: claim under test, item claims, lineage (observations, candidates, resolved item, Terra verdict and rationale, human truth, attribution rationale).
2. Open every photo in `photos.photos[]` with the image-reading tool at the recorded `path`. Confirm the file hash matches `sha256` (once per distinct photo is enough; record it in `photos`).
3. Write down what is physically visible that bears on the claim: surfaces, materials, mechanisms (peeling, staining, cracks, wear, dirt), objects (valance vs blinds, brick vs stone, vinyl vs plank), location and extent. Facts only; no repair advice, no cost, no diagnosis of which pipeline stage failed.
4. Compare with the claim under test and with the catalog item claim: which commitments does the image support, which does it not.
5. Compare with the lineage: does Terra's rationale describe what you see; does the Pass 2a observation match the image; is the resolved item's claim the right name for what is visible; if the candidate list contains a better-fitting item, name it.
6. Record the adjudication and a one-line reason.

## Adjudication vocabulary

| Value | Meaning |
|---|---|
| `supports_claim` | The photo(s), read together with the lineage, support the row's `claim_under_test` as stated. |
| `refuted` | The photo(s) contradict the claim under test, or the specific thing the claim asserts is not there. |
| `unclear` | The photo(s) cannot settle it (angle, resolution, occlusion, ambiguity). Use this honestly; do not guess. |
| `unavailable` | A photo could not be opened or its hash did not match. Never substitute another image. |

For `unit` rows the claim is the cluster claim: adjudicate whether this unit's photo(s) and lineage exhibit that specific situation. For `coverage` rows, adjudicate whether the gold finding is visible as stated. For `control` rows, adjudicate whether the catalog item's own claim is supported on that card's photos (for `hallucination` and `correct_rejection` controls the expected honest answer is often `refuted`; say what you see regardless).

## Rules

- Evidence classes matter downstream, not here: a `model_judge` lead is a model output. Refute it freely when the image disagrees; do not treat it as truth because it is in the packet.
- Describe, do not diagnose. Do not decide whether the catalog, Pass 2d, or Terra owns a failure. Do note plainly when Terra's rationale disagrees with what you see, or when the observation text disagrees with the image.
- Adjudicate each row independently, even when two rows share a photo. The same photo can support one claim and refute another.
- Do not skip a required row. If you cannot open its photo, answer `unavailable`.
- Save incrementally (every 10 to 20 rows) so a context reset loses nothing; on resume, read the results file and continue from the first unanswered required row.
- Do not renumber, rename, or reorder anything from the packet; copy `row_id` and `unit_key` verbatim.

## Output

Write `reports/catalog_audit_photo_review.json` (UTF-8, indent 2, sorted keys):

```json
{
  "schema_version": 1,
  "packet_sha256": "7b4621dfe6c72aa42369c2f5ef7051dfecbec05b6333a80651b1b57a5b37bc80",
  "reviewer": "claude_photo_review_session_<date>",
  "rows": [
    {
      "row_id": "<copied from the packet>",
      "unit_key": "<copied from the packet>",
      "adjudication": "supports_claim | refuted | unclear | unavailable",
      "reason": "one line",
      "what_image_shows": "physical facts that bear on the claim",
      "claim_commitments_supported": "which parts of the claim/item claim the image supports",
      "claim_commitments_not_supported": "which parts it does not",
      "lineage_note": "does Terra's rationale / the observation / the resolved item agree with the image; better-fitting candidate if any"
    }
  ],
  "photos": {
    "<property_key>/<photo_key>": {"visible_facts": "what the photo shows overall", "sha256_verified": true}
  }
}
```

Answer each row at most once. Optional rows use the same shape.

## Self-check before you finish

```bash
.venv\Scripts\python.exe -c "import json,hashlib;p=json.load(open('reports/catalog_audit_photo_review_packet.json',encoding='utf-8'));r=json.load(open('reports/catalog_audit_photo_review.json',encoding='utf-8'));ids=[x['row_id'] for x in r['rows']];req=[x['row_id'] for x in p['rows'] if x['priority']=='required'];v={'supports_claim','refuted','unclear','unavailable'};print('packet hash ok',r['packet_sha256']==hashlib.sha256(open('reports/catalog_audit_photo_review_packet.json','rb').read()).hexdigest());print('duplicates',[i for i in set(ids) if ids.count(i)>1]);print('unknown rows',[i for i in ids if i not in {x['row_id'] for x in p['rows']}]);print('required unanswered',[i for i in req if i not in ids]);print('bad vocabulary',[x['row_id'] for x in r['rows'] if x.get('adjudication') not in v]);import collections;print('counts',collections.Counter(x['adjudication'] for x in r['rows']))"
```

All four lists must be empty and the packet hash must be `True`. Report the counts by adjudication, the number of optional rows completed, and anything unexpected (photos that did not open, hash mismatches, rows whose lineage looked inconsistent with the packet).

## Stop conditions

- Packet hash mismatch: stop, report, do nothing else.
- A photo missing or with a hash mismatch: answer that row `unavailable`, continue.
- Anything that would require editing a file other than the results file: stop and report instead.

## Opening prompt for this session

> Perform the photo and lineage review for catalog-audit Session 2. Read `docs/catalog_audit_program/PHOTO_REVIEW_BRIEF_SESSION_2.md`, verify the packet hash, then work through every `required` row of `reports/catalog_audit_photo_review_packet.json` (optional rows afterwards if budget allows): open each photo at its recorded path, compare what is visible with the row's claim under test and its lineage, and record one adjudication per row in `reports/catalog_audit_photo_review.json`, saving incrementally. Do not read the proposals artifact or the proposal document, do not modify any other file, and do not diagnose which pipeline stage failed. Finish with the self-check from the brief and report counts by adjudication.
