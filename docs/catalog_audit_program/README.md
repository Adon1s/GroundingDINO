# Catalog audit program — session packet

This directory is the reviewable context and handoff packet for the evidence-driven catalog audit. It turns the original project prompt, the 2026-09-01 context pack, subsequent repository fact-checking, and the agreed session structure into a bounded six-session program.

The purpose of this packet is not to pre-decide catalog changes. It gives each future Codex task enough verified context to produce its own implementation plan and execute only its assigned stage.

## Program shape

| Order | Task | Brief | Gate/output |
|---|---|---|---|
| 1 | Evidence foundation and worklist | [Session 1](01_SESSION_EVIDENCE_FOUNDATION.md) | Deterministic evidence bundle and item-family worklist |
| 2 | Semantic audit and proposal drafting | [Session 2](02_SESSION_SEMANTIC_AUDIT.md) | Reviewable proposal document; no catalog edits |
| 3 | Independent adversarial review | [Session 3](03_SESSION_ADVERSARIAL_REVIEW.md) | Challenge report; no implementation |
| — | Human disposition | [Human review gate](04_HUMAN_REVIEW_GATE.md) | Locked approval manifest |
| 4 | Approved surgical implementation | [Session 4](05_SESSION_APPROVED_IMPLEMENTATION.md) | Candidate decisions and generated artifacts |
| 5 | Deterministic and retrieval validation | [Session 5](06_SESSION_DETERMINISTIC_VALIDATION.md) | Tier 1/2 validation report and live-run manifest |
| 6 | Cost-gated live validation | [Session 6](07_SESSION_LIVE_VALIDATION.md) | Publish/no-publish recommendation |
| 7 | Bounded catalog policy checkpoint (successor to the Session 7 prose in `HANDOFF_SESSION_6.md`) | [Session 7](08_SESSION_BOUNDED_POLICY_CHECKPOINT.md) | Decision packet, exact-op approval, one accepted checkpoint |
| Optional | Publication/cutover | Not scoped in this packet | Separately authorized task only |

The sequence is intentionally mostly serial. Do not start a downstream task before its predecessor has produced and verified the required handoff. Session 3 should be a fresh task rather than a continuation of Session 2 so that its review is meaningfully independent.

File prefixes are packet order, not session numbers: the human gate is `04_`, so Sessions 4–6 live in `05_`–`07_`. Handoffs are named by session number (`HANDOFF_SESSION_<N>.md`).

## Baseline rule and the unvalidated 3.2 change

Catalog 3.2 (commit `a9ed7ad`) is implemented and generated but not yet accepted: it still owes a live run to obtain fresh Sol package decisions for the 33 candidates its re-routing changed, Steven's manual package review, and the 15% v5-versus-v4 headline-delta gate (see that commit message and `docs/FINDINGS_catalog_3_2_deferred_issues.md`). Sol is the runtime package-review model, so this is pipeline work, not a document review. Decided by Steven on 2026-09-01: it is **not** a blocker. The audit starts on the as-generated 3.2 catalog, and 3.2 acceptance is a separately owned thread that this program must neither perform nor decide. What the program owes it is data: Session 6's unmodified baseline arm is a live 3.2 run and produces exactly the package decisions and comparator inputs acceptance needs, so those outputs must be preserved and handed off rather than discarded. If 3.2 acceptance changes the decisions file mid-program, that is ordinary baseline drift under Section 5 of the [overall context](00_OVERALL_CONTEXT.md), not an expected mismatch.

## What to give each Codex task

Every task should receive:

1. [Overall program context](00_OVERALL_CONTEXT.md).
2. Its session brief from the table above.
3. The previous task's completed handoff, using [the handoff template](HANDOFF_TEMPLATE.md).
4. The machine-readable artifacts named in the session brief.
5. For Session 4 and later, the human approval or prior validation artifact required by that session.

Do not rely on the prior conversation as the only carrier of important facts. The repository documents and hashed artifacts are the durable source of truth.

## Required behavior in every session

Each session must:

- Inspect the current repository and fact-check the relevant brief before acting.
- Begin by writing or presenting a task-specific implementation plan grounded in the brief.
- State any material mismatch between the brief and the current checkout.
- Preserve frozen evidence and unrelated user changes.
- Stay within the assigned stage; discovered adjacent work goes into the handoff rather than being implemented opportunistically.
- End with a complete handoff and exact test/command results.

Official OpenAI guidance recommends lean prompts, explicit context boundaries, and one clear handoff between distinct workflows. See [OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model).

## Source precedence

When sources disagree, use this order:

1. Current repository code and tests, after verifying the expected baseline.
2. Hash-verified frozen machine artifacts.
3. Human approval manifest for post-review work.
4. [Overall program context](00_OVERALL_CONTEXT.md).
5. The stage handoff.
6. Historical planning/context documents.

The existing `docs/CONTEXT_catalog_audit_inputs_20260901.md` is retained as historical input. It contains valuable evidence pointers but also several claims corrected in the overall context. It is not the authoritative instruction document for these sessions.

## Scope ceiling

This program is intentionally narrow:

- No rebuilding the completed 2a-to-downstream trace.
- No broad pipeline redesign.
- No catalog editor or audit UI.
- No pricing-calibration project.
- No product-policy change unless separately approved.
- No catalog or migration edits before the human gate.
- No live provider runs before Tier 1/2 validation and explicit cost authorization.
- No publication or production cutover as an implied final step.

