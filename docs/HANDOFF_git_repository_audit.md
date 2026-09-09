# Handoff — git repository audit and hygiene

Date prepared: 2026-09-06, by the catalog-audit human-gate session.

## Objective

Investigate the state of this git repository and decide what should be committed, what should be ignored, what should be
removed, and what should be left alone. **The decisions are yours to make in that session.** This document supplies
measurements and open questions only; it deliberately contains no recommendations, and where a fact suggests an obvious
course of action it is still recorded as a fact rather than a conclusion.

Steven's framing when commissioning this: he has historically avoided committing generated and report-shaped files on
the belief that it bloats git, and wants that strategy re-examined from evidence rather than assumed either way.

## Repository identity

- Path: `C:/Users/Steven/PycharmProjects/realtorvision-backend`
- Branch: `terra_factorized_verifier` at `9afe0fa5a0490a856abed507dccc4a02ed1de24c`, last commit 2026-09-01
- **This branch has no upstream and has never been pushed.** Nothing on it exists anywhere but this machine.
- Tracked files: 378, totalling 18 MB
- Untracked files: 9,163
- `git status --porcelain -uno` is empty (no tracked modifications)

## Measured state

### Object store

| Measure | Value |
|---|---|
| `.git` on disk | 726 MB |
| `size-pack` | 628.56 MiB across 2 packs |
| loose objects | 2,208, 89.25 MiB |
| garbage objects | 10 stray `tmp_obj_*` files, 346 KiB, reported by `git count-objects -v` |

Largest blobs anywhere in history:

| Size | Path |
|---:|---|
| 661.8 MB | `groundingdino_swint_ogc.pth` |
| 2.8 MB | `.asset/hero_figure.png` |
| 2.2 MB | `reports/catalog_audit_evidence.json` (appears twice; committed in two Session 1 commits) |
| 2.1 MB | `demo/image_editing_with_groundingdino_gligen.ipynb` |
| 2.0 MB | `benchmarks/pass2a-prompt/runs/judge_baseline_vs_checklist/judgments.json` |
| 1.7 MB | `.asset/GD_SD.png` |

`git-lfs` 3.5.1 is installed and `.gitattributes` declares LFS rules for `benchmarks/datasets/**/photos/**`, but
`.git/lfs` is empty (0 bytes) and `.gitattributes` itself is untracked.

### Top-level directories

| Directory | Size | Files | Tracked | Ignored |
|---|---:|---:|---:|---|
| `artifacts/` | 527 MB | 8,975 | 0 | no |
| `artifacts_canary/` | 470 MB | 1,297 | 0 | **yes** (`.gitignore:151`) |
| `benchmarks/` | 111 MB | 703 | 52 | no |
| `outputs/` | 11 MB | 106 | 0 | **yes** (`.gitignore:138`) |
| `reports/` | 15 MB | 45 | 20 | no |
| `reports_scratch/` | 15 MB | 44 | 0 | no |
| `tests/` | 8.7 MB | 191 | 90 | no |
| `tools/` | 6.3 MB | 189 | 98 | no |
| `docs/` | 2.9 MB | 120 | 70 | no |
| `scripts/` | 2.2 MB | 65 | 35 | no |
| `configs/` | 43 KB | 10 | 9 | no |
| `pytest_tmp/`, `pytest_tmp_compare_plan/`, `weights/` | 0 | 0 | 0 | no |

Repository root also holds 70 loose files totalling 42 MB: 65 `.json` (bias-check and catalog-audit run outputs and
their `.checkpoint.json` companions), one `.yaml`, one `.txt`, a `Dockerfile`, a file named `nul`, and a file named
`4,627`.

### Worktrees

| Path | Branch | HEAD |
|---|---|---|
| `realtorvision-backend` (main) | `terra_factorized_verifier` | `9afe0fa` |
| `.claude/worktrees/brave-merkle-effe31` | `claude/brave-merkle-effe31` | `151804e` |
| `.claude/worktrees/quirky-franklin-be57e8` | `claude/quirky-franklin-be57e8` | `495045c` |
| `rv-catalog-audit-s4` | `catalog_audit_session4` | `9afe0fa` |
| `rv-factorized-replay` | `factorized_replay_v1` | `c833da4` |
| `rv-legacy-a1972cf` | detached | `a1972cf` |

Local branches include several `claude/*` branches with no worktree attached (`claude/funny-albattani`,
`claude/hopeful-ellis`, `claude/quizzical-easley`, among others).

### Line endings

`core.autocrlf` is `true` and `core.eol` is unset. Text files are stored LF in the object database and materialise CRLF
in working trees. This was verified end to end on `reports/catalog_audit_evidence.json`: repo blob
`c4691415…`, working tree `43354104…`, and a freshly created worktree reproduces `43354104…`, which is the value pinned
throughout the catalog-audit program.

## Observations

Recorded as measurements, not as recommendations.

1. A single 661.8 MB file, `groundingdino_swint_ogc.pth`, accounts for roughly 91% of the 726 MB object store. It is
   in history, not in the working tree.
2. `artifacts_canary/` (470 MB) is ignored; `artifacts/` (527 MB, 8,975 files) is not. Both are run-output trees with
   zero tracked files.
3. Tracking is partial in most source directories: 90 of 191 files under `tests/`, 98 of 189 under `tools/`, 35 of 65
   under `scripts/`, 70 of 120 under `docs/`, 20 of 45 under `reports/`, 52 of 703 under `benchmarks/`.
4. The catalog-audit program is tracked inconsistently across its own sessions. Session 1 committed its evidence bundle
   and builder in three commits. Sessions 2 and 3 and the human gate committed nothing, so the proposal artifact, the
   adversarial review, all four handoffs, the six session briefs, the approvals manifest and the other gate artifacts
   exist only as untracked working-tree files.
5. `.gitattributes` carries the LFS rules but is itself untracked, so a clone would not receive it. `.git/lfs` is empty.
6. `git count-objects -v` reports 10 garbage `tmp_obj_*` files.
7. The branch has never been pushed, so the only copy of everything above is this machine's disk.
8. Measured compressibility of the untracked program artifacts: `reports/catalog_audit_proposals.json` is 2.0 MB on
   disk and 423 KB gzipped. The full set proposed for a hypothetical commit measures 5.9 MB on disk.

## Constraints and hazards

These are facts the session needs before running anything destructive.

- **Uncommitted work is load-bearing.** The catalog-audit gate produced `reports/catalog_audit_gate_decisions.json`,
  `catalog_audit_redraft.json`, `catalog_audit_renewed_review.json`, `catalog_audit_live_check_results.json` and
  `catalog_audit_approvals.json`. The last of these is the authorization Session 4 implements from, and the live-check
  file records 23 paid provider calls that cannot be reproduced without spending again. None is committed. A
  `git clean`, a stash, or a history rewrite could destroy them.
- **Hash pins.** The catalog-audit artifacts pin each other and their inputs by sha256 of the working-tree bytes.
  Anything that changes bytes on disk, including a line-ending normalisation, invalidates those pins and breaks the
  chain from the approvals manifest back to the evidence bundle. `docs/catalog_audit_program/HANDOFF_SESSION_GATE.md`
  lists the current values.
- **An active worktree depends on copies.** `rv-catalog-audit-s4` was created on 2026-09-06 for Session 4 and contains
  copies of the untracked program artifacts, because a worktree materialises only tracked files. It has its own
  `WORKTREE_NOTES.md`. If the tracking decision changes, that worktree may need rebuilding.
- **Five other worktrees share this object store**, so a history rewrite affects all of them.
- **Prior session guidance on stashing** (from earlier work in this repository): test paths under `tests/` are partly
  untracked, and stashing test paths has previously failed atomically and collided with a pre-existing stash.

## Questions to answer

Ordered from lowest to highest blast radius; the session decides which are in scope.

1. What is the intended tracking policy, stated explicitly? Which categories of file belong in git, which belong in
   `.gitignore`, and which should be deleted outright?
2. Should `artifacts/` be ignored, and if so, is `artifacts_canary/` already being ignored for the same reason or a
   different one?
3. What are the 70 loose files in the repository root, and does each belong in git, in `.gitignore`, or in the bin?
   Specifically, what are `nul` and `4,627`?
4. Should `reports/` be split, so that regenerable run output and durable decision records are governed differently, or
   is a single rule correct for the whole directory?
5. Should the catalog-audit program artifacts be committed, and if so, as one commit or per session? If not, what is
   the backup and durability plan for files that cannot be regenerated?
6. Is the partial tracking under `tests/`, `tools/`, `scripts/` and `benchmarks/` intentional, and if not, what should
   change?
7. Should `.gitattributes` be tracked? Is the LFS configuration working as intended given `.git/lfs` is empty?
8. What should happen to the `claude/*` branches with no worktree, and to the two `.claude/worktrees/` checkouts?
9. Should the 10 garbage objects be cleaned, and is a `git gc` or repack warranted?
10. What, if anything, should be done about `groundingdino_swint_ogc.pth` in history? Note that any answer involving
    rewriting history interacts with every item in Constraints and Hazards above.
11. Should this branch be pushed to a remote, given nothing on it is backed up?

## Suggested constraints for the session

- Verify before destroying. Take a hash inventory of every untracked file being considered for deletion, and confirm
  with Steven before removing anything that cannot be regenerated.
- Treat the catalog-audit hash chain as a correctness test: after any change, the pins in
  `docs/catalog_audit_program/HANDOFF_SESSION_GATE.md` should still verify, and
  `scripts/render_catalog_audit_proposals.py --check` and `scripts/render_catalog_audit_adversarial_review.py --check`
  should still exit 0.
- Do not rewrite history without an explicit decision from Steven that accounts for the five worktrees.

## Suggested opening prompt

> Audit this git repository and help me decide what to commit, what to ignore, what to delete, and what to leave alone.
> Read `docs/HANDOFF_git_repository_audit.md` first: it has the measurements and the open questions, and deliberately
> draws no conclusions, so the analysis is yours. Start by re-verifying its numbers, since it was written on 2026-09-06
> and the repository may have moved. Pay attention to the Constraints and Hazards section before running anything
> destructive: the catalog-audit program has uncommitted, hash-pinned artifacts including an approvals manifest and a
> live-check record of 23 paid provider calls, and an active worktree at `rv-catalog-audit-s4` depends on copies of
> them. Work through the questions in order of blast radius, propose a tracking policy with your reasoning, and check
> with me before deleting or rewriting anything.

---

## Addendum — 2026-09-08, appended by the catalog policy checkpoint session

The body above was written on 2026-09-06 and instructs the audit session to re-verify its numbers.
The repository has moved since. These are measurements and open questions in the same discipline as
the rest of the document: **no recommendations, and no decision is taken here.** Nothing in this
addendum was acted on.

### What changed since 2026-09-06

- The main checkout is unchanged: `terra_factorized_verifier` at `9afe0fa`, tracked-clean.
- The candidate worktree `rv-catalog-audit-s4` advanced from `6e67eaa` to
  **`7ddd52ab1fb198f3242d857e700cab23527327af`** on branch `catalog_audit_session4`, three new
  commits (`637a7c0`, `ea34be5`, `7ddd52a`), tracked-clean. Sessions 5, 6 and 7 all ran.
- Session 6 spent 1,149,750 provider tokens. Its raw evidence is under the ignored
  `artifacts_canary/` tree, so it is **not** recoverable from git at all.

### New untracked load-bearing records

The body notes that the catalog program's approvals manifest is uncommitted. That set has grown, and
now includes the authorization for a catalog change that IS committed:

| Untracked file | Why it is load-bearing |
|---|---|
| `reports/catalog_audit_approvals.json` | CAP-007's authorization; hash-pinned by tests and harnesses |
| `reports/catalog_audit_approvals_v2.json` | the 2026-09-08 authorization for the two ops in commit `ea34be5` |
| `docs/DECISION_RECORD_catalog_checkpoint_20260908.md` | Steven's decisions D1-D8 and the accepted losses |
| `reports/catalog_policy_checkpoint_packet.json` | the evidence those decisions rest on |
| `docs/catalog_audit_program/HANDOFF_SESSION_5.md`, `_6`, `_7` | the session chain |
| `reports/catalog_checkpoint_*.json` (7 files) | the Tier 1/2, isolation, package and cutover evidence |
| `scripts/catalog_audit_validation.py`, `catalog_audit_live_validation.py`, `render_catalog_audit_*.py` and their tests | the harnesses; all four render/validate scripts are untracked |

None of these are gitignored. They are simply not added. `reports/catalog_audit_evidence.json` **is**
tracked, so the evidence bundle is in history while the approval that gates it is not.

Count as of this addendum: **119 untracked files under `docs/`, `reports/`, `scripts/` and `tests/`.**

### A fact the body does not record: master is a clean fast-forward

| Measure | Value |
|---|---|
| `master` and `main` | both at `650a447` (2026-08-14), both tracking `origin/main` |
| `catalog_audit_session4` vs `master` | **37 commits ahead, 0 behind**; `master` is a direct ancestor |
| `git merge-base --is-ancestor master catalog_audit_session4` | true, so `--ff-only` would succeed |
| Diff | 162 files, +181,521 / -833, dominated by frozen evidence JSON |
| Other branches with commits not already contained | **only `pass_2c_kind_v3`** (2 commits) |

Consequences, recorded as facts:

1. A fast-forward would move 37 commits of tracked work onto `master`, including the catalog change
   in `ea34be5`, **without** the untracked records that authorize and explain it. The merge decision
   and the tracking-policy decision are therefore coupled; they were previously independent.
2. After a fast-forward, four tests fail in a working copy that has the untracked artifacts present:
   the three audit-renderer parity tests (they pin baseline decisions and catalog hashes that the
   branch changed) and `test_benchmark_pass2a::test_compute_pre2f_totals_force_confirms_packages`
   (CCF-8 stale id). On a fresh clone most of these **skip** instead, because they are guarded on
   untracked files existing. `master` would stop being green either way.

### Relevant to question 11 ("should this branch be pushed")

`origin` is `git@github.com:Adon1s/GroundingDINO.git` and `upstream` is
`https://github.com/IDEA-Research/GroundingDINO.git`. `master` and `main` both track `origin/main`
**on that remote**. The body records `groundingdino_swint_ogc.pth` as a 661.8 MB blob in history and
a GroundingDINO demo notebook among the largest files; the remote configuration appears to be the
same inheritance. A push of any branch here, as configured today, would publish this backend into a
GroundingDINO repository. No push has been made from any session.

### Two further questions this addendum raises

12. Should `master` be fast-forwarded to `catalog_audit_session4` before or after the tracking policy
    is decided, given consequence 1 above?
13. Should the remote be corrected, removed, or left alone, and does that change the answer to
    question 11?
