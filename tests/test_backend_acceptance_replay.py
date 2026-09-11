"""Acceptance controls must fail before spending or dropping frozen inputs."""
import asyncio
import copy
import json
from pathlib import Path

import pytest

from scripts.replay_frozen_upstream_acceptance import (
    CANARY, PRODUCTION, FrozenInputGap, NoModelCalls, guard_out_root, reconstruct_photo,
)
from tools.artifact_writers import _write_json_atomic
from tools.pipeline_config import resolve_renovation_sol_model
from tools.renovation_architecture import usage_guard as ug
from tools.pass_config import SceneClassifierRunOptions, PassToggles
from tools.scene_classifier_orchestrator import SceneClassifierOrchestrator


def test_atomic_writer_repairs_lone_surrogates_preserving_unicode(tmp_path):
    path = tmp_path / 'artifact.json'
    _write_json_atomic(path, {'description': 'wear\udc8f café \ud83d\ude00 😀'})
    assert json.loads(path.read_text(encoding='utf-8')) == {'description': 'wear� café 😀 😀'}


@pytest.mark.parametrize('raw', [None, '', '  '])
def test_new_mode_requires_explicit_sol(raw):
    with pytest.raises(ValueError, match='explicit RENOVATION_SOL_MODEL'):
        resolve_renovation_sol_model(raw, openai_model='gpt-5.6-terra', mode='new')
    assert resolve_renovation_sol_model(raw, openai_model='gpt-5.6-terra') == 'gpt-5.6-terra'
    assert resolve_renovation_sol_model('gpt-5.6-sol', openai_model='', mode='new') == 'gpt-5.6-sol'


@pytest.mark.parametrize('path', [CANARY, CANARY / 'run_1', CANARY.parent, PRODUCTION, PRODUCTION / 'x'])
def test_output_guard_protects_sources(path):
    with pytest.raises(ValueError):
        guard_out_root(path)


def reserve(ledger, tokens):
    return ledger.reserve(property_key='p', source_run_id='r', estimate_unit_id='u',
                          request_fingerprint='f', tokens=tokens)


def test_batch_cap_shared_across_process_instances_and_utc_days(tmp_path, monkeypatch):
    monkeypatch.setattr(ug, '_utc_today', lambda: '2026-09-10')
    monkeypatch.setattr(ug, '_utc_now_iso', lambda: '2026-09-10T00:00:00+00:00')
    old = ug.TerraUsageLedger(tmp_path)
    reserve(old, 90)
    start = '2026-09-10T01:00:00Z'
    monkeypatch.setattr(ug, '_utc_now_iso', lambda: '2026-09-10T01:00:01+00:00')
    one = ug.TerraUsageLedger(tmp_path, batch_start=start, batch_ceiling=100)
    rid = reserve(one, 70)
    one.settle(rid, provider_total_tokens=60)
    monkeypatch.setattr(ug, '_utc_today', lambda: '2026-09-11')
    monkeypatch.setattr(ug, '_utc_now_iso', lambda: '2026-09-11T00:00:01+00:00')
    two = ug.TerraUsageLedger(tmp_path, batch_start=start, batch_ceiling=100)
    rid = reserve(two, 40)
    two.settle(rid, provider_total_tokens=None)
    assert two.spent_today() == 40
    assert one.spent_since(start) == 100
    with pytest.raises(ug.TerraBatchBudgetExceeded):
        reserve(two, 1)
    assert two.spent_since(start) == 100


def test_reconstruction_keeps_llm_null_and_detects_missing_input():
    photo = {'issues': {'matched': [{'issue_id': 'a', 'description': 'Worn floor.', 'kind': 'degradation'}]},
             '_pass_2e_telemetry': {'input_count': 1},
             'debug': {'resolved_items': [{'issue_id': 'a', 'description': 'Worn floor.',
                         'original_kind': 'degradation', 'resolved_item_id': None}]}}
    observations, resolutions = reconstruct_photo(photo, 'photo.jpg')
    assert len(observations['observations']) == 1
    assert resolutions['a']['resolved_item_id'] is None
    photo['_pass_2e_telemetry']['input_count'] = 2
    with pytest.raises(FrozenInputGap):
        reconstruct_photo(photo, 'photo.jpg')


def test_empty_frozen_passes_do_not_fall_through_to_models():
    meta = {'run_id': 'r', 'photo_key': 'p.jpg', 'pass_1a_frozen_scene': 'kitchen',
            'pass_2a_frozen_freeform': '', 'pass_2b_frozen_struct': {'observations': []},
            'pass_2c_frozen_observations': {'observations': [], 'excluded': []},
            'pass_2d_frozen_resolutions': {}}
    before = copy.deepcopy(meta)
    orchestrator = SceneClassifierOrchestrator(qwen_config={}, gpt5_config={},
        vlm_client=NoModelCalls(), catalog_items=[])
    result = asyncio.run(orchestrator.analyze_image(image_path=Path('unused.jpg'),
        options=SceneClassifierRunOptions(pipeline_mode='publish',
            toggles=PassToggles(pass_2f=False), meta=meta)))
    assert result.observations == []
    assert meta == before
    assert result.models_used['2b'] == result.models_used['2c'] == 'frozen_replay'
