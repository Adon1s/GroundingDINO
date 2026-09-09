# Human review gate — catalog audit

No catalog implementation begins until this gate is completed. The human reviewer owns the final disposition; Codex recommendations are inputs, not authorization.

## Inputs

- `reports/catalog_audit_proposals.json`
- Generated proposal Markdown.
- `reports/catalog_audit_adversarial_review.json`
- Adversarial review Markdown.
- `docs/catalog_audit_program/HANDOFF_SESSION_2.md`
- `docs/catalog_audit_program/HANDOFF_SESSION_3.md`
- Source evidence/photos for any disputed record.

## Review choices

Assign exactly one disposition to every proposal ID:

- `approved`
- `approved_with_modification`
- `rejected`
- `deferred`
- `reclassified_non_catalog`

For `approved_with_modification`, specify the exact semantic/diff change rather than an open-ended direction. Materially changed wording or scope must return for evidence, overlap, and adversarial checks before implementation.

## Per-proposal checklist

- [ ] I understand the observed failure and reviewed its supporting evidence.
- [ ] Successful uses, correct rejections, and counterexamples were considered.
- [ ] The evidence independence claim is credible.
- [ ] Catalog ownership is stronger than the rejected alternate stages.
- [ ] The proposed operation is supported by the migration system, or is explicitly classified as a separately scoped system gap.
- [ ] Retrieval/kind/scene/embedding risks are understood.
- [ ] Economic, work-item, package, route, and product-policy effects are understood.
- [ ] Adversarial conditions are resolved or accepted explicitly.
- [ ] The exact approved change is unambiguous.

## Approval manifest

Create `reports/catalog_audit_approvals.json` with a schema like:

```json
{
  "schema_version": "catalog-audit-approvals-v1",
  "approved_against": {
    "git_commit": "<proposal baseline commit>",
    "evidence_sha256": "<hash>",
    "proposals_sha256": "<hash>",
    "adversarial_review_sha256": "<hash>"
  },
  "reviewed_by": "<human>",
  "reviewed_at": "<timestamp>",
  "dispositions": [
    {
      "proposal_id": "<stable id>",
      "disposition": "approved|approved_with_modification|rejected|deferred|reclassified_non_catalog",
      "approved_diff": {},
      "conditions": [],
      "notes": ""
    }
  ]
}
```

Rejected, deferred, and non-catalog records remain in the manifest so Session 4 can prove that every proposal was accounted for.

## Gate outcomes

Session 4 may begin only if:

- Every proposal ID appears exactly once.
- Every approved native change has an exact approved diff.
- The proposal and review hashes match.
- No approved record depends on an unresolved migration-system gap.
- Any new schema/generator or product-policy work has separate explicit authorization.

If no proposal is approved, the catalog-audit implementation program ends successfully with a no-change decision. Do not create an empty implementation merely to continue the sequence.

