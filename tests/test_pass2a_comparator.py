"""Tests for the Pass 2a prompt comparator.

The comparator's whole claim is that it runs Pass 2a and nothing else, that a
resume never re-bills a completed call, and that identities stay hidden until
the review is done. Those three are what most of this module pins.

No markers, no plugins (there is no pytest.ini); async is driven with bare
asyncio.run(), and VLM clients are hand-rolled fakes with a `.calls` list.
"""
import asyncio
import json
import os
from pathlib import Path

import pytest

from tools.comparison_common import sha256_bytes, sha256_file
from tools.pass2a_comparator import config as cc
from tools.pass2a_comparator import review as rv
from tools.pass2a_comparator import runner
from tools.pass2a_comparator import storage as store
from tools.scene_classifier_passes import (
    PASS_2A_SYSTEM_PROMPT,
    PASS_2A_USER_PROMPT,
    run_pass_2a,
)

CANDIDATE_SYSTEM = "You are a meticulous property condition inspector."
CANDIDATE_USER = "List every visible material condition in this photo."
PHOTOS_PER_PROPERTY = {"prop_a": 7, "prop_b": 8}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeVLM:
    """Records every call. Optionally fails chosen images, keyed `<prop>/<photo>`.

    Yields to the event loop before answering so the shared runner's semaphore
    really admits `concurrency` calls at a time; without that, one coroutine
    runs to completion before the next starts and fail-fast looks stricter than
    it is.
    """

    def __init__(self, fail_on=(), raise_after=None):
        self.calls = []
        self.fail_on = set(fail_on)
        self.raise_after = raise_after

    async def analyze_image(self, **kwargs):
        await asyncio.sleep(0)
        self.calls.append(kwargs)
        path = Path(str(kwargs["image_path"]))
        key = f"{path.parent.name}/{path.name}"
        if key in self.fail_on:
            raise RuntimeError(f"provider failure on {key}")
        if self.raise_after is not None and len(self.calls) > self.raise_after:
            raise RuntimeError("quota wall")
        return f"output for {key} :: {kwargs['user_prompt'][:24]}"


class RecordingPass:
    """Stands in for a downstream pass so a test can assert it never ran."""

    def __init__(self):
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bench(tmp_path, monkeypatch):
    """A synthetic 15-photo dataset plus gold, wired into the comparator paths."""
    images_root = tmp_path / "images"
    properties = {}
    gold_photos = {}
    for property_key, count in PHOTOS_PER_PROPERTY.items():
        (images_root / property_key).mkdir(parents=True)
        entries = []
        for index in range(1, count + 1):
            photo_key = f"photo_{index:03d}.jpg"
            path = images_root / property_key / photo_key
            path.write_bytes(f"{property_key}/{photo_key} pixels".encode("utf-8"))
            entries.append({
                "photo_key": photo_key,
                "image_sha256": sha256_file(path),
                "scene": "kitchen" if index % 2 else "bathroom",
            })
            gold_photos[f"{property_key}/{photo_key}"] = [
                {"gold_id": "g1", "condition": "Flooring is worn"},
                {"gold_id": "g2", "condition": "Cabinets are dated"},
            ]
        properties[property_key] = {"photos": entries}

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "images_root": str(images_root), "properties": properties,
    }), encoding="utf-8")
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"photos": gold_photos}), encoding="utf-8")

    monkeypatch.setattr(cc, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(cc, "GOLD_PATH", gold)
    monkeypatch.setattr(cc, "BASELINES_DIR", tmp_path / "runs" / "baselines")
    monkeypatch.setattr(cc, "EXPERIMENTS_DIR", tmp_path / "runs" / "experiments")

    model_config = {
        "model": "gpt-5.6-terra", "provider": "openai", "api_key": "unit-test",
        "reasoning_effort": "low", "max_output_tokens": 8000, "max_tokens": 8000,
    }
    monkeypatch.setattr(cc, "build_model_config", lambda: dict(model_config))

    photos, root = store.load_dataset()
    return {"photos": photos, "images_root": root, "model_config": model_config,
            "tmp_path": tmp_path}


def make_experiment(bench, system=CANDIDATE_SYSTEM, user=CANDIDATE_USER):
    fingerprint = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")
    bid, _, _ = store.ensure_baseline(fingerprint)
    exp_id, exp_dir = store.create_experiment(
        system_prompt=system, user_prompt=user, display_name="unit",
        baseline_id_value=bid, fingerprint=fingerprint,
        model_config=bench["model_config"],
    )
    return exp_id, exp_dir, bid


def run(bench, exp_id, client, **kwargs):
    return runner.run_experiment(
        exp_id, client_factory=lambda _property_key: client, **kwargs
    )


# ---------------------------------------------------------------------------
# 1. Prompt forwarding - production behaviour must not move
# ---------------------------------------------------------------------------

def test_run_pass_2a_defaults_to_the_production_prompts(tmp_path):
    image = tmp_path / "p.jpg"
    image.write_bytes(b"x")
    client = FakeVLM()

    asyncio.run(run_pass_2a(image, client, {"model": "m"}))

    call = client.calls[0]
    assert call["system_prompt"] == PASS_2A_SYSTEM_PROMPT
    assert call["user_prompt"] == PASS_2A_USER_PROMPT


def test_run_pass_2a_forwards_both_overrides_verbatim(tmp_path):
    image = tmp_path / "p.jpg"
    image.write_bytes(b"x")
    client = FakeVLM()

    asyncio.run(run_pass_2a(
        image, client, {"model": "m"},
        system_prompt=CANDIDATE_SYSTEM, user_prompt=CANDIDATE_USER,
    ))

    call = client.calls[0]
    assert call["system_prompt"] == CANDIDATE_SYSTEM
    assert call["user_prompt"] == CANDIDATE_USER


def test_overriding_only_the_system_prompt_leaves_the_user_prompt_production(tmp_path):
    image = tmp_path / "p.jpg"
    image.write_bytes(b"x")
    client = FakeVLM()

    asyncio.run(run_pass_2a(image, client, {"model": "m"}, system_prompt="S"))

    assert client.calls[0]["system_prompt"] == "S"
    assert client.calls[0]["user_prompt"] == PASS_2A_USER_PROMPT


# ---------------------------------------------------------------------------
# 2-4. Call counts, isolation, invocation shape
# ---------------------------------------------------------------------------

def test_first_experiment_runs_45_baseline_and_45_candidate_calls(bench):
    exp_id, _, _ = make_experiment(bench)
    client = FakeVLM()

    results = run(bench, exp_id, client)

    assert len(client.calls) == 90
    assert results["baseline"]["called"] == 45
    assert results["candidate"]["called"] == 45
    baseline_calls = [c for c in client.calls if c["user_prompt"] == PASS_2A_USER_PROMPT]
    assert len(baseline_calls) == 45
    assert all(c["system_prompt"] == PASS_2A_SYSTEM_PROMPT for c in baseline_calls)


def test_a_second_experiment_reuses_the_baseline_and_runs_only_45(bench):
    first_id, _, _ = make_experiment(bench)
    run(bench, first_id, FakeVLM())

    second_id, _, _ = make_experiment(bench, user="A different candidate wording.")
    client = FakeVLM()
    results = run(bench, second_id, client)

    assert len(client.calls) == 45
    assert results["baseline"]["reused"] == 45
    assert results["baseline"]["called"] == 0
    assert results["candidate"]["called"] == 45


def test_no_downstream_pass_is_invoked(bench, monkeypatch):
    """Pass 1a and 2b-2f must be unreachable from the comparator."""
    from tools import scene_classifier_passes as passes

    spies = {}
    for name in ("run_pass_1a", "run_pass_2b", "run_pass_2c", "run_pass_2d"):
        if hasattr(passes, name):
            spy = RecordingPass()
            spies[name] = spy
            monkeypatch.setattr(passes, name, spy)
    assert spies, "expected downstream passes to exist to spy on"

    exp_id, _, _ = make_experiment(bench)
    run(bench, exp_id, FakeVLM())

    for name, spy in spies.items():
        assert spy.calls == [], f"{name} was invoked"


def test_every_call_carries_the_pinned_invocation(bench):
    exp_id, _, _ = make_experiment(bench)
    client = FakeVLM()

    run(bench, exp_id, client)

    for call in client.calls:
        assert call["model"] == "gpt-5.6-terra"
        assert call["reasoning_effort"] == "low"
        assert call["max_output_tokens"] == 8000
        assert call["analysis_pass"] == "Pass 2a (observations freeform)"


def test_budget_context_is_set_for_ledger_attribution(bench, monkeypatch):
    created = []

    class Client(FakeVLM):
        pass

    def factory(property_key):
        client = Client()
        client.budget_context = {
            "property_key": property_key,
            "source_run_id": "pass2a_comparator/x",
        }
        created.append(client)
        return client

    exp_id, _, _ = make_experiment(bench)
    runner.run_experiment(exp_id, client_factory=factory)

    assert created
    assert {c.budget_context["property_key"] for c in created} == set(PHOTOS_PER_PROPERTY)


def test_build_client_attaches_budget_context(monkeypatch):
    class Stub:
        pass

    monkeypatch.setattr("tools.vlm_client.create_vlm_client", lambda *a, **k: Stub())
    client = runner.build_client("pass2a_comparator/e_1", "prop_a")
    assert client.budget_context == {
        "property_key": "prop_a", "source_run_id": "pass2a_comparator/e_1",
    }


# ---------------------------------------------------------------------------
# 5. Checkpointing and resume
# ---------------------------------------------------------------------------

def test_every_successful_call_is_checkpointed(bench):
    exp_id, exp_dir, bid = make_experiment(bench)
    run(bench, exp_id, FakeVLM())

    for photo, rep in store.iter_calls(bench["photos"]):
        for base in (exp_dir, store.baseline_dir(bid)):
            record = store.load_call(store.call_path(base, photo, rep))
            assert record and record["status"] == "ok" and record["text"]


def test_resume_issues_no_calls_when_everything_is_complete(bench):
    exp_id, _, _ = make_experiment(bench)
    run(bench, exp_id, FakeVLM())

    resumed = FakeVLM()
    results = run(bench, exp_id, resumed)

    assert resumed.calls == []
    assert results["candidate"]["reused"] == 45


def test_resume_reruns_only_missing_and_failed_calls(bench):
    exp_id, exp_dir, _ = make_experiment(bench)
    photos = bench["photos"]
    # Fails in the second property of rep 1, so earlier calls really did land.
    first = FakeVLM(fail_on={"prop_b/photo_008.jpg"})

    with pytest.raises(runner.ComparatorRunError):
        run(bench, exp_id, first)

    done_before = store.baseline_progress(
        store.baseline_dir(store.load_experiment(exp_id)["baseline_id"]),
        photos, store.load_experiment(exp_id)["fingerprint"],
    )
    assert 0 < done_before < 45  # the run really did stop early

    resumed = FakeVLM()
    run(bench, exp_id, resumed)

    # Every call is accounted for exactly once across the two attempts.
    assert len(resumed.calls) == 90 - done_before
    assert store.experiment_progress(exp_dir, photos, store.load_experiment(exp_id)) == 45


def test_a_failed_call_is_never_reused(bench):
    photos = bench["photos"]
    record = {
        "status": "error", "text": "", "error": "boom",
        "system_prompt_sha256": sha256_bytes(CANDIDATE_SYSTEM.encode()),
        "user_prompt_sha256": sha256_bytes(CANDIDATE_USER.encode()),
        "image_sha256": photos[0].image_sha256,
    }
    assert not store.call_is_reusable(
        record, system_sha=record["system_prompt_sha256"],
        user_sha=record["user_prompt_sha256"], image_sha=photos[0].image_sha256,
    )


def test_a_call_from_a_different_prompt_is_never_reused(bench):
    photos = bench["photos"]
    record = {
        "status": "ok", "text": "text",
        "system_prompt_sha256": sha256_bytes(b"other"),
        "user_prompt_sha256": sha256_bytes(CANDIDATE_USER.encode()),
        "image_sha256": photos[0].image_sha256,
    }
    assert not store.call_is_reusable(
        record, system_sha=sha256_bytes(CANDIDATE_SYSTEM.encode()),
        user_sha=sha256_bytes(CANDIDATE_USER.encode()),
        image_sha=photos[0].image_sha256,
    )


def test_a_partial_checkpoint_is_treated_as_missing(bench):
    exp_id, exp_dir, _ = make_experiment(bench)
    run(bench, exp_id, FakeVLM())
    victim = store.call_path(exp_dir, bench["photos"][0], 1)
    victim.write_text("{ truncated", encoding="utf-8")

    resumed = FakeVLM()
    run(bench, exp_id, resumed)

    assert len(resumed.calls) == 1


# ---------------------------------------------------------------------------
# 6. Fingerprint invalidation
# ---------------------------------------------------------------------------

def test_fingerprint_is_stable_across_calls(bench):
    a = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")
    b = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")
    assert a == b and store.baseline_id(a) == store.baseline_id(b)


def test_fingerprint_excludes_git_head(bench):
    fingerprint = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")
    assert "git_head" not in fingerprint


def test_git_head_is_recorded_in_the_sibling_info_file(bench):
    fingerprint = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")
    _, path, _ = store.ensure_baseline(fingerprint)
    assert json.loads((path / "info.json").read_text(encoding="utf-8"))["git_head"]


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda b, m: m.setattr(cc, "MAX_OUTPUT_TOKENS", 2000), id="token-cap"),
    pytest.param(lambda b, m: m.setattr(cc, "REASONING_EFFORT", "medium"), id="reasoning"),
    pytest.param(lambda b, m: m.setattr(cc, "REPEATS", 2), id="repeats"),
    pytest.param(lambda b, m: m.setattr(cc, "COMPARATOR_SCHEMA_VERSION", "v2"), id="schema"),
])
def test_runtime_changes_invalidate_baseline_reuse(bench, monkeypatch, mutate):
    before = store.baseline_id(store.compute_fingerprint(bench["photos"], "gpt-5.6-terra"))
    mutate(bench, monkeypatch)
    after = store.baseline_id(store.compute_fingerprint(bench["photos"], "gpt-5.6-terra"))
    assert before != after


def test_a_changed_image_invalidates_baseline_reuse(bench):
    photos = bench["photos"]
    before = store.baseline_id(store.compute_fingerprint(photos, "gpt-5.6-terra"))
    swapped = [store.Photo(photos[0].property_key, photos[0].photo_key,
                           "0" * 64, photos[0].scene)] + photos[1:]
    after = store.baseline_id(store.compute_fingerprint(swapped, "gpt-5.6-terra"))
    assert before != after


def test_a_changed_pass_2a_source_invalidates_baseline_reuse(bench, monkeypatch, tmp_path):
    before = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")
    fake_source = tmp_path / "scene_classifier_passes.py"
    fake_source.write_text("# edited", encoding="utf-8")
    monkeypatch.setattr(cc, "PASS_PATH_SOURCES", (fake_source,))
    after = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")

    assert store.baseline_id(before) != store.baseline_id(after)
    assert "scene_classifier_passes_sha256" in store.fingerprint_diff(before, after)


def test_a_changed_call_path_blocks_running_an_existing_experiment(bench, monkeypatch, tmp_path):
    exp_id, _, _ = make_experiment(bench)
    fake_source = tmp_path / "vlm_client.py"
    fake_source.write_text("# edited", encoding="utf-8")
    monkeypatch.setattr(cc, "PASS_PATH_SOURCES", (fake_source,))

    with pytest.raises(runner.ComparatorRunError, match="call path changed"):
        run(bench, exp_id, FakeVLM())


def test_a_new_fingerprint_preserves_the_earlier_baseline(bench, monkeypatch):
    first = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")
    first_id, first_dir, _ = store.ensure_baseline(first)
    monkeypatch.setattr(cc, "MAX_OUTPUT_TOKENS", 4000)
    second = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")
    second_id, second_dir, created = store.ensure_baseline(second)

    assert created and second_id != first_id
    assert first_dir.is_dir() and second_dir.is_dir()
    stale = dict(store.stale_baselines(second))
    assert first_id in stale and "max_output_tokens" in stale[first_id]


# ---------------------------------------------------------------------------
# 7. Blinding and review
# ---------------------------------------------------------------------------

def test_blinding_is_deterministic_and_balanced(bench):
    photos = bench["photos"]
    first = rv.assign_sides("e_1", photos)
    assert first == rv.assign_sides("e_1", photos)
    assert first != rv.assign_sides("e_2", photos)
    assert sum(1 for v in first.values() if v == rv.CANDIDATE) == 8
    assert sum(1 for v in first.values() if v == rv.BASELINE) == 7


def test_all_repeats_of_a_photo_stay_on_one_side(bench):
    exp_id, exp_dir, bid = make_experiment(bench)
    run(bench, exp_id, FakeVLM())
    photo = bench["photos"][0]
    sides = rv.assign_sides(exp_id, bench["photos"])

    side_a, side_b = rv.side_outputs(photo, sides[photo.key],
                                     store.baseline_dir(bid), exp_dir)

    assert len(side_a) == 3 and len(side_b) == 3
    # One side is entirely candidate wording, the other entirely production.
    assert len({CANDIDATE_USER[:24] in text for text in side_a}) == 1
    assert len({CANDIDATE_USER[:24] in text for text in side_b}) == 1
    assert (CANDIDATE_USER[:24] in side_a[0]) != (CANDIDATE_USER[:24] in side_b[0])


def test_reveal_is_refused_until_every_photo_has_a_verdict(bench):
    photos = bench["photos"]
    review = rv.empty_review()
    for photo in photos[:-1]:
        rv.record_decision(review, photo.key, verdict="A better")

    assert not rv.is_complete(review, photos)
    with pytest.raises(ValueError, match="cannot reveal"):
        rv.reveal(review, photos)
    assert not rv.is_revealed(review)


def test_reveal_snapshots_the_blind_decisions(bench):
    photos = bench["photos"]
    review = rv.empty_review()
    for photo in photos:
        rv.record_decision(review, photo.key, verdict="A better")
    rv.reveal(review, photos)

    rv.record_decision(review, photos[0].key, verdict="B better")

    assert review["blind_decisions"][photos[0].key]["verdict"] == "A better"
    assert review["decisions"][photos[0].key]["verdict"] == "B better"
    assert rv.revisions(review) == [photos[0].key]


def test_revealing_twice_cannot_overwrite_the_blind_snapshot(bench):
    photos = bench["photos"]
    review = rv.empty_review()
    for photo in photos:
        rv.record_decision(review, photo.key, verdict="same")
    rv.reveal(review, photos)
    first_revealed_at = review["revealed_at"]

    rv.record_decision(review, photos[0].key, verdict="A better")
    rv.reveal(review, photos)

    assert review["revealed_at"] == first_revealed_at
    assert review["blind_decisions"][photos[0].key]["verdict"] == "same"


def test_vote_mapping_follows_the_side_assignment():
    assert rv.map_vote("A better", rv.CANDIDATE) == rv.BETTER
    assert rv.map_vote("A better", rv.BASELINE) == rv.WORSE
    assert rv.map_vote("B better", rv.CANDIDATE) == rv.WORSE
    assert rv.map_vote("B better", rv.BASELINE) == rv.BETTER
    assert rv.map_vote("same", rv.CANDIDATE) == rv.SAME
    assert rv.map_vote("unclear", rv.BASELINE) == rv.UNCLEAR


def test_tags_and_notes_persist_through_a_save_and_reload(bench):
    exp_id, exp_dir, _ = make_experiment(bench)
    photo = bench["photos"][0]
    review = rv.empty_review()
    rv.record_decision(review, photo.key, verdict="A better",
                       tags=["better coverage", "better grounding"],
                       note="caught the ceiling stain", critical_regression=True)
    rv.save_review(exp_dir, review)

    reloaded = rv.load_review(exp_dir)["decisions"][photo.key]
    assert reloaded["tags"] == ["better coverage", "better grounding"]
    assert reloaded["note"] == "caught the ceiling stain"
    assert reloaded["critical_regression"] is True


def test_unknown_verdicts_and_tags_are_rejected(bench):
    review = rv.empty_review()
    with pytest.raises(ValueError, match="unknown verdict"):
        rv.record_decision(review, "p", verdict="candidate wins")
    with pytest.raises(ValueError, match="unknown reason tags"):
        rv.record_decision(review, "p", verdict="same", tags=["vibes"])
    with pytest.raises(ValueError, match="unknown final verdict"):
        rv.record_final_verdict(review, "great")


# ---------------------------------------------------------------------------
# 8. Directional summary and report
# ---------------------------------------------------------------------------

def _review_with(photos, verdicts, critical=()):
    review = rv.empty_review()
    for photo, verdict in zip(photos, verdicts):
        rv.record_decision(review, photo.key, verdict=verdict,
                           critical_regression=photo.key in critical)
    return review


def test_directional_is_no_signal_while_reviews_are_incomplete(bench):
    photos = bench["photos"]
    sides = {p.key: rv.CANDIDATE for p in photos}
    summary = rv.summarize(_review_with(photos, ["A better"] * 3), sides, photos)
    assert summary["directional"] == rv.NO_SIGNAL
    assert summary["directional_reason"] == "reviews incomplete"


def test_directional_is_positive_when_better_exceeds_worse(bench):
    photos = bench["photos"]
    sides = {p.key: rv.CANDIDATE for p in photos}
    summary = rv.summarize(_review_with(photos, ["A better"] * 15), sides, photos)
    assert summary["directional"] == rv.POSITIVE
    assert summary["counts"][rv.BETTER] == 15


def test_directional_is_negative_when_worse_exceeds_better(bench):
    photos = bench["photos"]
    sides = {p.key: rv.CANDIDATE for p in photos}
    summary = rv.summarize(_review_with(photos, ["B better"] * 15), sides, photos)
    assert summary["directional"] == rv.NEGATIVE


def test_directional_is_no_signal_on_a_tie(bench):
    photos = bench["photos"]
    sides = {p.key: rv.CANDIDATE for p in photos}
    verdicts = ["A better"] * 7 + ["B better"] * 7 + ["same"]
    summary = rv.summarize(_review_with(photos, verdicts), sides, photos)
    assert summary["directional"] == rv.NO_SIGNAL
    assert "tied" in summary["directional_reason"]


def test_a_critical_regression_forces_negative_despite_a_winning_tally(bench):
    photos = bench["photos"]
    sides = {p.key: rv.CANDIDATE for p in photos}
    review = _review_with(photos, ["A better"] * 15, critical={photos[3].key})
    summary = rv.summarize(review, sides, photos)

    assert summary["counts"][rv.BETTER] == 15
    assert summary["directional"] == rv.NEGATIVE
    assert summary["critical_regressions"] == [photos[3].key]


def test_report_is_written_and_never_claims_production_equivalence(bench):
    exp_id, exp_dir, _ = make_experiment(bench)
    photos = bench["photos"]
    sides = rv.assign_sides(exp_id, photos)
    review = _review_with(photos, ["A better"] * 15)
    rv.reveal(review, photos)
    rv.record_final_verdict(review, "positive", "worth a production-cap run")

    report = rv.write_report(exp_dir, store.load_experiment(exp_id), review, sides, photos)

    assert report["production_equivalent"] is False
    assert "2000-token production cap" in report["production_equivalent_reason"]
    stored = json.loads((exp_dir / "report.json").read_text(encoding="utf-8"))
    assert stored["final_verdict"]["verdict"] == "positive"
    markdown = (exp_dir / "report.md").read_text(encoding="utf-8")
    assert "Not production-equivalent" in markdown
    assert CANDIDATE_USER in markdown
    assert len([line for line in markdown.splitlines() if line.startswith("| `")]) == 15


def test_report_records_post_reveal_revisions(bench):
    exp_id, exp_dir, _ = make_experiment(bench)
    photos = bench["photos"]
    sides = rv.assign_sides(exp_id, photos)
    review = _review_with(photos, ["A better"] * 15)
    rv.reveal(review, photos)
    rv.record_decision(review, photos[0].key, verdict="B better")

    report = rv.write_report(exp_dir, store.load_experiment(exp_id), review, sides, photos)

    assert report["summary"]["post_reveal_revisions"] == [photos[0].key]
    assert report["blind_decisions"][photos[0].key]["verdict"] == "A better"
    assert "revised after reveal" in (exp_dir / "report.md").read_text(
        encoding="utf-8").lower()


# ---------------------------------------------------------------------------
# 9. Failure modes
# ---------------------------------------------------------------------------

def test_a_missing_image_is_an_explicit_error(bench):
    photo = bench["photos"][0]
    store.image_path(photo, bench["images_root"]).unlink()
    with pytest.raises(store.ComparatorDataError, match="image missing"):
        store.resolve_image(photo, bench["images_root"])


def test_a_substituted_image_is_rejected_by_hash(bench):
    photo = bench["photos"][0]
    store.image_path(photo, bench["images_root"]).write_bytes(b"different pixels")
    with pytest.raises(store.ComparatorDataError, match="image hash mismatch"):
        store.resolve_image(photo, bench["images_root"])


def test_a_missing_image_stops_the_run_with_checkpoints_intact(bench):
    exp_id, _, _ = make_experiment(bench)
    store.image_path(bench["photos"][0], bench["images_root"]).unlink()

    with pytest.raises(runner.ComparatorRunError, match="Checkpoints kept"):
        run(bench, exp_id, FakeVLM())


def test_missing_gold_for_a_photo_is_an_explicit_error(bench):
    gold = store.load_gold()
    photo = bench["photos"][0]
    gold.pop(photo.key)
    with pytest.raises(store.ComparatorDataError, match="no gold entry"):
        store.gold_for(gold, photo)


def test_empty_candidate_prompts_are_refused(bench):
    fingerprint = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")
    bid, _, _ = store.ensure_baseline(fingerprint)
    for system, user in ((" ", CANDIDATE_USER), (CANDIDATE_SYSTEM, "\n")):
        with pytest.raises(store.ComparatorDataError, match="non-blank"):
            store.create_experiment(
                system_prompt=system, user_prompt=user, display_name="bad",
                baseline_id_value=bid, fingerprint=fingerprint,
                model_config=bench["model_config"],
            )


def test_a_provider_failure_stops_the_run_and_is_recorded(bench):
    exp_id, _, bid = make_experiment(bench)
    photos = bench["photos"]
    client = FakeVLM(fail_on={photos[0].key})

    with pytest.raises(runner.ComparatorRunError):
        run(bench, exp_id, client)

    record = store.load_call(store.call_path(store.baseline_dir(bid), photos[0], 1))
    assert record["status"] == "error" and "provider failure" in record["error"]


def test_a_quota_wall_does_not_burn_the_remaining_calls(bench):
    """fail_fast is why the guard is worth turning on: stop, do not spend."""
    exp_id, _, _ = make_experiment(bench)
    client = FakeVLM(raise_after=3)

    with pytest.raises(runner.ComparatorRunError):
        run(bench, exp_id, client)

    assert len(client.calls) < 45


def test_concurrent_starts_are_rejected(bench):
    exp_id, exp_dir, _ = make_experiment(bench)
    lock = runner.RunLock(exp_dir / "run.lock").acquire()
    try:
        with pytest.raises(runner.ComparatorRunError, match="already active"):
            run(bench, exp_id, FakeVLM())
    finally:
        lock.release()


def test_a_stale_lock_from_a_dead_process_is_taken_over(bench):
    exp_id, exp_dir, _ = make_experiment(bench)
    lock_path = exp_dir / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"pid": 2 ** 31 - 1}), encoding="utf-8")

    run(bench, exp_id, FakeVLM())

    assert not lock_path.exists()


def test_the_lock_is_released_after_a_failed_run(bench):
    exp_id, exp_dir, _ = make_experiment(bench)
    with pytest.raises(runner.ComparatorRunError):
        run(bench, exp_id, FakeVLM(fail_on={bench["photos"][0].key}))

    assert not (exp_dir / "run.lock").exists()


def test_baseline_only_leaves_the_candidate_untouched(bench):
    exp_id, exp_dir, _ = make_experiment(bench)
    client = FakeVLM()

    run(bench, exp_id, client, baseline_only=True)

    assert len(client.calls) == 45
    assert store.experiment_progress(exp_dir, bench["photos"],
                                     store.load_experiment(exp_id)) == 0


def test_run_baseline_works_before_any_experiment_exists(bench):
    client = FakeVLM()
    result = runner.run_baseline(client_factory=lambda _p: client)

    assert len(client.calls) == 45
    assert result["baseline"]["called"] == 45
    assert store.list_experiments() == []


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def test_cli_requires_an_experiment_unless_baseline_only(bench):
    with pytest.raises(SystemExit):
        runner.main([])


def test_cli_reports_failure_on_the_wire_rather_than_raising(bench, capsys):
    code = runner.main(["--experiment", "does_not_exist"])
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]

    assert code == 1
    assert any(e["type"] == "error" for e in events)


def test_the_stub_client_never_reaches_a_provider(bench, monkeypatch):
    monkeypatch.setenv(runner.STUB_ENV, "1")
    exp_id, exp_dir, _ = make_experiment(bench)

    def explode(*args, **kwargs):
        raise AssertionError("create_vlm_client must not be called in stub mode")

    monkeypatch.setattr("tools.vlm_client.create_vlm_client", explode)
    runner.run_experiment(exp_id)

    record = store.load_call(store.call_path(exp_dir, bench["photos"][0], 1))
    assert record["stub"] is True and "STUB OUTPUT" in record["text"]


def test_stub_evidence_can_never_be_reused_by_a_paid_run(bench, monkeypatch):
    """The mocked walkthrough must not poison a real baseline.

    Stub mode is part of the fingerprint, so stub outputs land in their own
    baseline directory and a paid run neither reuses them nor sees them as
    complete.
    """
    real = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")
    monkeypatch.setenv(cc.STUB_ENV, "1")
    stubbed = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")

    assert store.baseline_id(real) != store.baseline_id(stubbed)
    assert store.fingerprint_diff(real, stubbed) == ["stub"]

    _, stub_dir, _ = store.ensure_baseline(stubbed)
    runner.run_baseline()
    assert store.baseline_progress(stub_dir, bench["photos"], stubbed) == 45

    monkeypatch.delenv(cc.STUB_ENV)
    _, real_dir, _ = store.ensure_baseline(real)
    assert real_dir != stub_dir
    assert store.baseline_progress(real_dir, bench["photos"], real) == 0


# ---------------------------------------------------------------------------
# Budget guard - decision #2: comparator runs are metered, never silent
# ---------------------------------------------------------------------------

def test_the_guard_env_carries_a_usage_root(monkeypatch):
    """Forcing the guard on without a root raises before the first call."""
    monkeypatch.delenv("RENOVATION_TERRA_USAGE_ROOT", raising=False)
    env = cc.budget_guard_env()
    assert env["RENOVATION_VLM_BUDGET_GUARD"] == "1"
    assert env["RENOVATION_TERRA_USAGE_ROOT"] == str(cc.USAGE_ROOT)


def test_an_operator_set_usage_root_is_left_alone(monkeypatch):
    monkeypatch.setenv("RENOVATION_TERRA_USAGE_ROOT", "D:/shared")
    assert "RENOVATION_TERRA_USAGE_ROOT" not in cc.budget_guard_env()


def test_building_a_real_client_refuses_to_spend_unmetered(bench, monkeypatch):
    from tools import pipeline_config as pc

    monkeypatch.setattr(pc, "RENOVATION_VLM_BUDGET_GUARD", False)
    with pytest.raises(cc.ComparatorConfigError, match="would spend"):
        runner.resolve_client_factory("e_1", None, None)


def test_a_guard_without_a_ledger_root_is_refused(bench, monkeypatch):
    from tools import pipeline_config as pc

    monkeypatch.setattr(pc, "RENOVATION_VLM_BUDGET_GUARD", True)
    monkeypatch.setattr(pc, "RENOVATION_TERRA_USAGE_ROOT", None)
    with pytest.raises(cc.ComparatorConfigError, match="nowhere to keep"):
        runner.resolve_client_factory("e_1", None, None)


def test_the_guard_is_not_required_when_a_client_is_injected(bench, monkeypatch):
    """Tests and the stub must not need the guard - they never reach a provider."""
    from tools import pipeline_config as pc

    monkeypatch.setattr(pc, "RENOVATION_VLM_BUDGET_GUARD", False)
    client = FakeVLM()
    run(bench, make_experiment(bench)[0], client)
    assert len(client.calls) == 90


# ---------------------------------------------------------------------------
# Scene injection - {scene} from the frozen Pass 1a capture, no 1a calls
# ---------------------------------------------------------------------------

SCENE_USER = "This is a {scene}. List every visible material condition."


def test_scene_renders_from_the_frozen_manifest_value(bench):
    exp_id, _, _ = make_experiment(bench, user=SCENE_USER)
    client = FakeVLM()

    run(bench, exp_id, client)

    by_scene = {}
    for call in client.calls:
        path = Path(str(call["image_path"]))
        by_scene[f"{path.parent.name}/{path.name}"] = call["user_prompt"]
    for photo in bench["photos"]:
        rendered = by_scene[photo.key]
        if rendered.startswith("This is a "):          # the candidate side
            assert rendered == f"This is a {photo.scene}. List every visible material condition."
            assert "{scene}" not in rendered


def test_a_scene_prompt_costs_no_pass_1a_calls(bench, monkeypatch):
    from tools import scene_classifier_passes as passes

    spy = RecordingPass()
    monkeypatch.setattr(passes, "run_pass_1a_scene_type", spy)
    exp_id, _, _ = make_experiment(bench, user=SCENE_USER)
    run(bench, exp_id, FakeVLM())

    assert spy.calls == []


def test_the_baseline_is_unaffected_by_a_scene_candidate(bench):
    """Production prompts have no placeholder, so the baseline stays reusable."""
    first_id, _, _ = make_experiment(bench)
    run(bench, first_id, FakeVLM())

    second_id, _, _ = make_experiment(bench, user=SCENE_USER)
    client = FakeVLM()
    results = run(bench, second_id, client)

    assert results["baseline"]["reused"] == 45
    assert len(client.calls) == 45


def test_a_non_scene_prompt_does_not_key_reuse_on_scene(bench):
    """Regression: keying reuse on scene unconditionally orphans paid calls.

    Records written before scene existed carry no `scene` field; a prompt that
    never renders it must still reuse them.
    """
    exp_id, exp_dir, _ = make_experiment(bench)
    run(bench, exp_id, FakeVLM())
    for photo, rep in store.iter_calls(bench["photos"]):
        path = store.call_path(exp_dir, photo, rep)
        record = json.loads(path.read_text(encoding="utf-8"))
        record.pop("scene", None)
        record.pop("scene_conditional", None)
        path.write_text(json.dumps(record), encoding="utf-8")

    resumed = FakeVLM()
    run(bench, exp_id, resumed)

    assert resumed.calls == []


def test_a_scene_prompt_reruns_when_the_scene_changes(bench, monkeypatch):
    exp_id, exp_dir, _ = make_experiment(bench, user=SCENE_USER)
    run(bench, exp_id, FakeVLM())
    victim = store.call_path(exp_dir, bench["photos"][0], 1)
    record = json.loads(victim.read_text(encoding="utf-8"))
    record["scene"] = "a_different_room"
    victim.write_text(json.dumps(record), encoding="utf-8")

    resumed = FakeVLM()
    run(bench, exp_id, resumed)

    assert len(resumed.calls) == 1


def test_unsupported_placeholders_are_refused_before_any_call(bench):
    fingerprint = store.compute_fingerprint(bench["photos"], "gpt-5.6-terra")
    bid, _, _ = store.ensure_baseline(fingerprint)
    with pytest.raises(cc.ComparatorConfigError, match="unsupported placeholder"):
        store.create_experiment(
            system_prompt=CANDIDATE_SYSTEM, user_prompt="Describe the {room}.",
            display_name="bad", baseline_id_value=bid, fingerprint=fingerprint,
            model_config=bench["model_config"],
        )


def test_escaped_braces_survive_rendering():
    assert cc.render_prompt("A {scene} with {{braces}}", "kitchen") == "A kitchen with {braces}"
    assert cc.render_prompt("No placeholder {{here}}", "kitchen") == "No placeholder {{here}}"


def test_scene_conditional_is_recorded_and_reported(bench):
    exp_id, exp_dir, _ = make_experiment(bench, user=SCENE_USER)
    photos = bench["photos"]
    run(bench, exp_id, FakeVLM())
    experiment = store.load_experiment(exp_id)
    assert experiment["scene_conditional"] is True

    record = store.load_call(store.call_path(exp_dir, photos[0], 1))
    assert record["scene"] == photos[0].scene and record["scene_conditional"] is True

    review = _review_with(photos, ["A better"] * 15)
    rv.reveal(review, photos)
    report = rv.write_report(exp_dir, experiment, review,
                             rv.assign_sides(exp_id, photos), photos)

    assert report["scene_conditional"] is True
    markdown = (exp_dir / "report.md").read_text(encoding="utf-8")
    assert "Scene-conditional prompt" in markdown
    assert "never reads it" in markdown


def test_a_plain_experiment_carries_no_scene_caveat(bench):
    exp_id, exp_dir, _ = make_experiment(bench)
    photos = bench["photos"]
    review = _review_with(photos, ["same"] * 15)
    rv.reveal(review, photos)
    rv.write_report(exp_dir, store.load_experiment(exp_id), review,
                    rv.assign_sides(exp_id, photos), photos)

    assert "Scene-conditional" not in (exp_dir / "report.md").read_text(encoding="utf-8")
