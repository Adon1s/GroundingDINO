# First replica investigation and review packet

All 18 artifacts pass the production verifier and catalog invariants. Final acceptance is pending.

| Model | Artifact / ledger tokens | Calls | Unsettled |
| --- | --- | --- | --- |
| terra | 912131 | 148 | 0 |
| sol | 136606 | 18 | 0 |

Terra remaining in the original shared batch: 1,087,869 tokens.

## Observation selection

| Change | Observations |
| --- | --- |
| unchanged | 3501 |
| null_to_item | 42 |
| item_to_item | 70 |
| item_to_null | 55 |

These are selection changes, not counts of useful losses. All 3,668 observations were compared.

## Human-labeled warranted work still missing

| Case | Property | Original verdict | Current rationale | Observed work allowance delta |
| --- | --- | --- | --- | --- |
| rc_128caa6212b8 | redfin_11000447 | unsupported | The visible flooring is hard plank flooring, not vinyl or linoleum with torn or lifting material. | $0–$0 |
| rc_32ff789c40d0 | redfin_11079485 | supported | The shower tile is visible but does not show a clearly vintage or dated pattern. | $-256–$-1,278 |
| rc_48ea825e33f8 | redfin_10952874 | unsupported | The baseboards are visible, but no clear scuffs, paint loss, or dents are discernible. | $0–$0 |
| rc_5063066c1a57 | redfin_80925528 | unsupported | The visible chimney is stained stone masonry rather than brickwork with intact mortar. | $0–$0 |
| rc_790b583fcbf0 | redfin_80990371 | unsupported | The visible wall paint does not show clear peeling or bubbling. | $0–$0 |
| rc_bcb50df30b68 | redfin_10952874 | unsupported | The visible gray wall and ceiling paint is neutral and not clearly dated or highly personalized. | $0–$0 |
| rc_d60c090abd2c | redfin_11000447 | unsupported | The flooring is visible, but no clearly torn or lifted material is shown. | $0–$0 |
| rc_f468f4066e3f | redfin_11185681 | supported | The photographed wall surfaces do not show clear scuffs, marks, or dents. | $-33–$-168 |

Allowances are non-additive where conditions share work. Zero delta for an old miss does not value its recovery at zero.

## Material headline changes requiring final review

Every row below uses effective applied package amounts. Component deltas reconcile exactly to the headline.

### redfin_10806500

Stored run 1: $23,903–$74,107; fresh replica 1: $13,513–$51,036; stored run 2: $19,889–$57,538.

| Component | Before low/high | After low/high | Delta low/high | Application reason; Sol |
| --- | --- | --- | --- | --- |
| kitchen_modernization/kitchen_primary | $20,780–$48,486 | $10,390–$25,415 | $-10,390–$-23,071 | approved_absorbs_children; approve |

### redfin_11079485

Stored run 1: $7,896–$60,141; fresh replica 1: $5,640–$50,677; stored run 2: $15,806–$72,456.

| Component | Before low/high | After low/high | Delta low/high | Application reason; Sol |
| --- | --- | --- | --- | --- |
| bathroom_modernization/bathroom_primary | $2,184–$7,280 | $0–$0 | $-2,184–$-7,280 | absent; n/a |
| standalone | $4,620–$48,493 | $4,548–$46,309 | $-72–$-2,184 | standalone; n/a |

### redfin_25809814

Stored run 1: $33,317–$94,326; fresh replica 1: $26,132–$73,512; stored run 2: $27,940–$86,248.

| Component | Before low/high | After low/high | Delta low/high | Application reason; Sol |
| --- | --- | --- | --- | --- |
| bedroom_modernization/bedroom_2 | $4,316–$12,948 | $0–$0 | $-4,316–$-12,948 | opportunity_only_interior_modernization; approve |
| bedroom_modernization/bedroom_3 | $4,316–$12,948 | $0–$0 | $-4,316–$-12,948 | opportunity_only_interior_modernization; approve |
| standalone | $2,673–$14,048 | $4,120–$19,130 | $1,447–$5,082 | standalone; n/a |

### redfin_80925528

Stored run 1: $34,806–$133,180; fresh replica 1: $28,794–$122,899; stored run 2: $34,187–$145,436.

| Component | Before low/high | After low/high | Delta low/high | Application reason; Sol |
| --- | --- | --- | --- | --- |
| bedroom_modernization/bedroom_1 | $5,000–$15,000 | $0–$0 | $-5,000–$-15,000 | opportunity_only_interior_modernization; approve |
| kitchen_modernization/kitchen_primary | $15,000–$35,000 | $15,000–$35,667 | $0–$667 | approved_absorbs_children; approve |
| living_modernization/living_room_primary | $2,000–$10,150 | $0–$0 | $-2,000–$-10,150 | opportunity_only_interior_modernization; approve |
| living_repair/living_room_primary | $3,000–$10,000 | $3,000–$13,760 | $0–$3,760 | approved_absorbs_children; approve |
| standalone | $2,306–$37,030 | $3,294–$47,472 | $988–$10,442 | standalone; n/a |

### redfin_81000709

Stored run 1: $13,281–$52,090; fresh replica 1: $12,885–$62,076; stored run 2: $16,904–$65,478.

| Component | Before low/high | After low/high | Delta low/high | Application reason; Sol |
| --- | --- | --- | --- | --- |
| bedroom_modernization/bedroom_1 | $4,249–$12,746 | $0–$0 | $-4,249–$-12,746 | opportunity_only_interior_modernization; approve |
| bedroom_modernization/bedroom_3 | $1,275–$5,098 | $0–$0 | $-1,275–$-5,098 | absent; n/a |
| bedroom_repair/bedroom_3 | $2,124–$6,798 | $2,124–$9,103 | $0–$2,305 | approved_absorbs_children; approve |
| exterior_repair/exterior_primary | $0–$0 | $4,249–$15,295 | $4,249–$15,295 | approved_absorbs_children; approve |
| standalone | $1,810–$13,852 | $2,689–$24,082 | $879–$10,230 | standalone; n/a |

## Limits

- Stored-to-fresh differences confound catalog, temperature and model changes.
- A mechanical endpoint is not proof of the first semantic error.
- The second replica, production smoke and material price review remain pending.
