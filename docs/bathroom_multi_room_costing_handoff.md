# Bathroom Multi-Room Costing Handoff

Current behavior keeps bathroom package costing conservative: at most one costed bathroom package per package type is emitted from the merged bathroom estimate unit, and kitchens remain one-per-house.

Bathroom Pass 2F now records audit-only room-count telemetry on package verification records:

- `visible_room_count`: `one_room`, `multiple_rooms`, or `unclear`
- `visible_room_count_evidence`: brief fixed-feature basis from the reviewed photos
- `reviewed_issue_ids`: issue IDs whose evidence photos were actually reviewed

V4 also emits `bathroom_room_count_signal`. It sets `likely_multiple_visible_bathrooms` when at least one active bathroom package 2F record votes `multiple_rooms` and no bathroom package votes `one_room`. This signal is for audit-page validation and is only one gate in multi-bathroom package expansion; expansion still requires listing metadata for 2+ bathrooms and confirmed evidence refs spanning multiple deterministic bathroom surrogates.

Future multi-bathroom costing should use deterministic bathroom surrogates and listing metadata as the source of per-bathroom package scope. Treat 2F room-count telemetry as a quality signal for whether the review photos were mixed, not as the authority for room counts or cost multiplication.
