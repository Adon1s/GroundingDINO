"""Score complete acceptance artifacts; preserve every reviewed population row."""
from __future__ import annotations

from collections import Counter
import contextlib
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analysis.terra_miss_mechanisms import CANARY, load_inputs, read, write, provenance, table
from scripts.analysis.pipeline_loss_ledger import trace_artifact
from scripts.verify_renovation_artifact import main as verify_artifact
from tools import review_cards as rc

SUCCESSORS = {'dated_window_treatment_valance': {'window_blinds_basic_or_plain', 'dated_window_valance_or_curtains'}}


def condition_key(condition, evidence):
    return (condition['catalog_item_id'], tuple(sorted(evidence.get('photo_keys', []))))


def match_condition(old, old_evidence, new_index):
    allowed = {old['catalog_item_id']} | SUCCESSORS.get(old['catalog_item_id'], set())
    exact, lineage = [], []
    for cid, condition in new_index['conds'].items():
        if condition['catalog_item_id'] not in allowed:
            continue
        if tuple(sorted(new_index['evs'][cid]['photo_keys'])) == tuple(sorted(old_evidence['photo_keys'])):
            exact.append(cid)
        elif set(condition['issue_ids']) & set(old['issue_ids']):
            lineage.append(cid)
    if exact:
        return exact, 'item_and_exact_evidence'
    return lineage, 'item_and_shared_issue_ids' if lineage else 'no_equivalent_condition'


def classify_case(old, old_review, old_disposition, matches, new_index, label):
    item = old['catalog_item_id']
    if item == 'dated_interior_trim' and matches and all(
            new_index['disps'][cid]['terminal_route'] == 'no_action' for cid in matches):
        return 'policy-superseded', 'D1: trim is recorded at no_action/$0'
    if not matches:
        if item == 'dated_wallpaper_present' and old.get('scene_group') == 'bathroom':
            return 'policy-superseded', 'D2: bathroom wallpaper exclusion'
        return 'lost', 'No equivalent condition under the same item or named CAP-007 successor'
    good = label.get('claim') == 'exact' and label.get('work') == 'warranted'
    if good and not any(new_index['disps'][cid]['disposition'] == 'accepted_for_work' for cid in matches):
        return 'lost', 'Human-supported warranted work remains excluded or changed route'
    if len(matches) != 1 or new_index['conds'][matches[0]]['catalog_item_id'] != item:
        return 'represented elsewhere', 'Named successor/merged representation'
    cid = matches[0]
    if new_index['revs'][cid]['verdict'] == old_review['verdict'] and new_index['disps'][cid]['terminal_route'] == old_disposition['terminal_route']:
        return 'retained', 'Same claim, verdict and route'
    return 'represented elsewhere', 'Condition remains, verdict or route changed; inspect recorded evidence'


def packages(result):
    works = {w['work_item_id']: w for w in result['work_items']}
    decisions = {d['package_candidate_id']: d for d in result['package_decisions']}
    def work_keys(ids):
        return sorted((works[i]['action_code'], works[i]['billable_unit_id'],
                       tuple(sorted(works[i]['catalog_item_ids']))) for i in ids)
    out = {}
    for candidate in result['package_candidates']:
        key = (candidate['package_type'], candidate['estimate_unit_id'])
        if key in out:
            raise ValueError(f'Non-unique package join key: {key}')
        out[key] = {'type': key[0], 'unit': key[1],
            'drivers': work_keys(candidate['driver_work_item_ids']),
            'supports': work_keys(candidate['support_work_item_ids']),
            'children': work_keys(candidate['child_work_item_ids']),
            'pricing': [candidate.get(k) for k in ('pricing_profile', 'pricing_tier', 'low', 'high')],
            'decision': decisions.get(candidate['package_candidate_id'])}
    return out


def compare_packages(old, new):
    before, after = packages(old), packages(new)
    rows = []
    for key in sorted(before.keys() | after.keys()):
        a, b = before.get(key), after.get(key)
        if a and b:
            shape = any(a[k] != b[k] for k in ('drivers', 'children', 'pricing'))
            change = 'shape' if shape else 'support_list_only' if a['supports'] != b['supports'] else 'unchanged'
        else:
            change = 'appeared' if b else 'disappeared'
        rows.append({'type': key[0], 'unit': key[1], 'change': change, 'before': a, 'after': b,
            'fresh_sol_decision': bool(b and b['decision']),
            'decision_agreement': a['decision']['decision'] == b['decision']['decision']
                if a and b and a['decision'] and b['decision'] else None})
    return rows


def score(out_root):
    baseline = read(ROOT / 'reports/pipeline_loss_ledger_20260910.json')
    _, _, _, originals = load_inputs()
    runs, run2 = rc.load_canary(CANARY)
    replicas = {}
    checks, properties, package_changes = [], [], []
    for replica in (1, 2):
        for prop, (_, source) in sorted(runs.items()):
            dest = Path(out_root) / f'replica_{replica}' / prop / source['run']['run_id']
            if not (dest / 'complete.json').exists():
                checks.append({'replica': replica, 'property_key': prop, 'passed': False, 'reason': 'not_run'})
                continue
            with contextlib.redirect_stdout(io.StringIO()) as output:
                valid = verify_artifact(['--artifact', str(dest), '--expect-mode', 'new',
                    '--expect-terra-model', 'gpt-5.6-terra', '--expect-sol-model', 'gpt-5.6-sol']) == 0
            checks.append({'replica': replica, 'property_key': prop, 'passed': valid, 'verification': output.getvalue()})
            if not valid:
                continue
            artifact = read(dest / 'photo_intel_debug.json')
            result = rc.v5_result(artifact)
            index = rc.index_result(result)
            rows, upstream, gaps = trace_artifact(artifact, 'canary')
            replicas[replica, prop] = (artifact, index, rows)
            old = rc.v5_result(source)
            headline, previous = result['totals']['headline'], old['totals']['headline']
            deltas = {k: (headline[k] - previous[k]) / previous[k] if previous[k] else None for k in ('low', 'high')}
            delta_review = any(abs(v) > .15 if v is not None else headline[k] != previous[k] for k, v in deltas.items())
            properties.append({'replica': replica, 'property_key': prop, 'headline': headline,
                'stored_run_1': previous, 'stored_run_2': rc.v5_result(run2[prop][1])['totals']['headline'],
                'relative_delta': deltas, 'requires_steven_review': delta_review,
                'terra_usage': result['terra_listing_usage'], 'sol_usage': result['sol_listing_usage'],
                'work_dedup_collisions': result['work_dedup_collisions'], 'lineage_gaps': gaps,
                'source': provenance([dest / 'photo_intel_debug.json'])[0]})
            for row in compare_packages(old, result):
                package_changes.append({'replica': replica, 'property_key': prop, **row})
            parent = [c for c in result['observed_conditions'] if c['catalog_item_id'] in SUCCESSORS]
            trim = [c for c in result['observed_conditions'] if c['catalog_item_id'] == 'dated_interior_trim']
            bad_trim = [c['condition_id'] for c in trim if index['disps'][c['condition_id']]['terminal_route'] != 'no_action'
                        or c['condition_id'] in index['work_by_cond']]
            wallpaper = [c for c in result['observed_conditions'] if c['catalog_item_id'] == 'dated_wallpaper_present'
                         and c.get('scene_group') == 'bathroom']
            checks.append({'replica': replica, 'property_key': prop, 'check': 'catalog_invariants',
                'passed': not parent and not bad_trim and not wallpaper,
                'parent_count': len(parent), 'trim_count': len(trim), 'bad_trim': bad_trim, 'bathroom_wallpaper_count': len(wallpaper)})
    dispositions = []
    for case in baseline['reviewed_cases']:
        ref = (case.get('source'), case.get('property_key'), case.get('run_id'))
        original = originals.get(ref)
        old_index = rc.index_result(rc.v5_result(original[1])) if original else None
        cid = case.get('condition_id')
        old = old_index['conds'].get(cid) if old_index else None
        for replica in (1, 2):
            record = {**case, 'replica': replica, 'replay_disposition': 'unresolved'}
            replay = replicas.get((replica, ref[1])) if ref[0] == 'canary' else None
            if not replay:
                record['replay_note'] = 'Outside the authorized 18-property replay' if ref[0] != 'canary' else 'Replica not complete'
            elif old:
                _, index, traces = replay
                matches, method = match_condition(old, old_index['evs'][cid], index)
                disposition, note = classify_case(old, old_index['revs'][cid], old_index['disps'][cid],
                    matches, index, case.get('human_label') or {})
                record.update(replay_disposition=disposition, replay_note=note, join_method=method,
                    replay_condition_ids=matches, before_review=old_index['revs'][cid],
                    before_disposition=old_index['disps'][cid], after_reviews=[index['revs'][c] for c in matches],
                    after_dispositions=[index['disps'][c] for c in matches],
                    after_work_items=[index['work_by_cond'].get(c) for c in matches],
                    after_observation_boundaries=[{'issue_id': r['issue_id'], 'boundary': r['mechanical_terminal_boundary']}
                        for r in traces if r['issue_id'] in case.get('observation_trace_ids', [])])
            else:
                _, index, traces = replay
                selected = [r for r in traces if r['issue_id'] in case.get('observation_trace_ids', [])]
                record['after_observation_boundaries'] = [{'issue_id': r['issue_id'],
                    'boundary': r['mechanical_terminal_boundary'], 'catalog_item_id': r['catalog_item_id']} for r in selected]
                record['replay_note'] = 'No original condition anchor; exact issue lineage recorded, semantic recovery needs review'
            dispositions.append(record)
    stability, flips = [], []
    for prop in sorted(runs):
        a, b = replicas.get((1, prop)), replicas.get((2, prop))
        if not a or not b:
            continue
        one = {r['issue_id']: r['catalog_item_id'] for r in a[2]}
        two = {r['issue_id']: r['catalog_item_id'] for r in b[2]}
        for iid in sorted(one.keys() | two.keys()):
            stability.append({'property_key': prop, 'issue_id': iid, 'replica_1': one.get(iid),
                              'replica_2': two.get(iid), 'equal': iid in one and iid in two and one[iid] == two[iid]})
        for cid, condition in a[1]['conds'].items():
            matches, method = match_condition(condition, a[1]['evs'][cid], b[1])
            if len(matches) == 1:
                v1, v2 = a[1]['revs'][cid]['verdict'], b[1]['revs'][matches[0]]['verdict']
                flips.append({'property_key': prop, 'catalog_item_id': condition['catalog_item_id'],
                              'replica_1': v1, 'replica_2': v2, 'flipped': v1 != v2, 'join_method': method})
    assert len(dispositions) == 2 * sum(baseline['population_counts'].values())
    report = {'schema_version': 1, 'passed': len(replicas) == 36 and all(c['passed'] for c in checks),
        'readiness': 'pending_investigation_and_production_smoke', 'model_calls': 0,
        'completed_artifacts': len(replicas), 'checks': checks, 'properties': properties,
        'reviewed_case_dispositions': dispositions, 'package_changes': package_changes,
        'pass_2d_stability': stability, 'terra_verdict_flips': flips,
        'disposition_counts': dict(Counter(r['replay_disposition'] for r in dispositions)),
        'limitations': ['Unrun and out-of-cohort cases remain unresolved, never silently removed.',
            'Per-case costs are not additive when several conditions share work or a package.',
            'Human review of material dollar deltas and the production smoke remain required.']}
    dest = ROOT / 'reports/backend_acceptance_score_20260910.json'
    write(dest, report)
    dest.with_suffix('.md').write_text('\n'.join(['# Backend acceptance investigation score', '',
        f"Completed artifacts: {len(replicas)}/36. Readiness: pending.", '',
        table(['Disposition', 'Population rows × replicas'], sorted(report['disposition_counts'].items())), '',
        'Unresolved rows include unrun replicas and reviewed production properties outside the authorized canary.', '']),
        encoding='utf-8', newline='\n')
    return report
