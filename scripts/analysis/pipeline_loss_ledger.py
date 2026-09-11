"""Observation survival census and reviewed first-loss evidence, without model calls.

Mechanical disappearance is not automatically a useful-observation loss. Human
labels and prior attributed causes stay separate from exact artifact joins.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analysis.terra_miss_mechanisms import load_inputs, read, write, provenance, table, SIX
from scripts.replay_frozen_upstream_acceptance import reconstruct_photo, FrozenInputGap, DEFAULT_OUT
from scripts.error_attribution_report import latest_verdicts
from tools import review_cards as rc


def normalized(text):
    return ' '.join(str(text or '').casefold().split()).rstrip('.')


def trace_artifact(artifact, source):
    """One row per recoverable 2c observation, plus unmatched 2b bullets.

    2b has no ids. Exact normalized text joins are explicit, and ambiguous or
    missing joins stay unknown rather than claiming that 2b/2c deleted meaning.
    """
    result = rc.v5_result(artifact) or {}
    idx = rc.index_result(result)
    flat = {name: {r['issue_id']: r for r in artifact.get(name, [])}
            for name in ('issues_flat', 'estimate_issues_flat', 'product_issues_flat', 'product_estimate_issues_flat')}
    canonical = 'product_estimate_issues_flat' if artifact.get('estimate_issues_flat') else 'product_issues_flat'
    coverage = {r['work_item_id']: r for r in result.get('coverage_ledger', [])}
    work_by_any_condition = {cid: w for w in result.get('work_items', []) for cid in w['condition_ids']}
    dedup_representatives = {wid: c['active_work_item_id'] for c in result.get('work_dedup_collisions', [])
                             for wid in c['suppressed_work_item_ids']}
    rows, gaps, upstream = [], [], []
    for key, photo in artifact['photos'].items():
        base = {'source': source, 'property_key': artifact['property']['property_key'],
                'run_id': artifact['run']['run_id'], 'photo_key': key}
        bullets = photo.get('debug', {}).get('observations_struct', {}).get('observations', [])
        try:
            frozen, resolved = reconstruct_photo(photo, key)
        except (FrozenInputGap, KeyError) as exc:
            gaps.append({**base, 'error': str(exc)})
            continue
        observed_texts = {normalized(r['description']) for r in frozen['observations']}
        exclusions = photo.get('debug', {}).get('excluded_observations')
        for i, bullet in enumerate(bullets):
            text = bullet.get('description', '') if isinstance(bullet, dict) else bullet
            if normalized(text) not in observed_texts:
                upstream.append({**base, 'bullet_index': i, 'observation': text,
                    'boundary': '2b_to_2c_no_exact_forward_match',
                    'interpretation': 'Could be non-actionable exclusion or wording change; not a confirmed useful loss.',
                    'persisted_exclusions_available': exclusions is not None})
        removed = {r['issue_id']: r for r in photo['issues'].get('removed', [])}
        suppressed = {r['issue_id']: r for r in photo['_pass_2e_telemetry'].get('suppressed_samples', [])}
        finals = {r['issue_id'] for r in photo['issues'].get('final', [])}
        for obs in frozen['observations']:
            iid = obs['issue_id']
            resolution = resolved[iid]
            item = resolution.get('resolved_item_id') if resolution else None
            cond = idx['cond_by_issue'].get(iid)
            representation_join = 'issue_id' if cond else None
            if cond is None and item:
                equivalent = [c for cid, c in idx['conds'].items() if c['catalog_item_id'] == item
                              and key in idx['evs'][cid]['photo_keys']]
                if len(equivalent) == 1:
                    cond, representation_join = equivalent[0], 'same_item_and_photo'
            cid = cond['condition_id'] if cond else None
            review, disp, work = (idx[k].get(cid) for k in ('revs', 'disps', 'work_by_cond'))
            suppressed_work = None
            if work is None and cid in work_by_any_condition:
                suppressed_work = work_by_any_condition[cid]
                active_id = dedup_representatives.get(suppressed_work['work_item_id'])
                work = idx['works'].get(active_id)
            bill = coverage.get(work['work_item_id']) if work else None
            if not item:
                boundary = '2d_null_selection' if resolution else '2d_no_resolver_record'
            elif iid not in flat[canonical] and not cond:
                raw_lane = 'estimate_issues_flat' if canonical == 'product_estimate_issues_flat' else 'issues_flat'
                boundary = 'product_filter' if iid in flat[raw_lane] else '2e_or_canonical_dedup'
            elif not cond:
                boundary = 'condition_projection'
            elif not review:
                boundary = 'terra_review_missing'
            elif review['verdict'] != 'supported':
                boundary = 'terra_' + review['verdict']
            elif disp and disp['disposition'] == 'withheld':
                boundary = 'routing_withheld'
            elif not work:
                boundary = 'routing_' + str((disp or {}).get('terminal_route', 'unknown'))
            elif not bill:
                boundary = 'package_output_missing_coverage'
            else:
                boundary = 'represented_' + bill['representation']
            matches = [i for i, b in enumerate(bullets)
                       if normalized(b.get('description', '') if isinstance(b, dict) else b) == normalized(obs['description'])]
            rows.append({**base, 'issue_id': iid, 'observation': obs['description'], 'kind': obs['kind'],
                'p2b_exact_indices': matches, 'p2c_present': True, 'catalog_item_id': item,
                'p2d_record': resolution, 'p2e_display_present': iid in finals,
                'p2e_removed': removed.get(iid), 'p2e_suppressed': suppressed.get(iid),
                'flat_lane_presence': {name: iid in entries for name, entries in flat.items()},
                'condition_id': cid, 'condition_issue_ids': cond.get('issue_ids') if cond else [],
                'representation_join': representation_join, 'suppressed_work_item': suppressed_work,
                'terra_review': review, 'disposition': disp, 'work_item': work, 'coverage': bill,
                'mechanical_terminal_boundary': boundary,
                'confirmed_useful_loss': None})
    return rows, upstream, gaps


def run(out_root=DEFAULT_OUT):
    cards, labels, review_verdicts, artifacts = load_inputs()
    queue = read(ROOT / 'reports/error_attribution_queue.json')
    gold = read(ROOT / 'reports/error_attribution_gold_cases.json')
    worklist = read(ROOT / 'reports/pass2d_worklist_reconciliation_20260910.json')['rows']
    attributed = latest_verdicts(ROOT / 'reports/error_attribution_verdicts.jsonl')
    cases = {c['case_id']: c for c in queue['cases'] + gold['cases']}
    cards_by_id = {c['card_id']: c for c in cards}
    all_rows, all_upstream, gaps, by_property = [], [], [], []
    for (source, prop, run_id), (path, artifact) in sorted(artifacts.items()):
        rows, upstream, missing = trace_artifact(artifact, source)
        all_rows.extend(rows); all_upstream.extend(upstream); gaps.extend(missing)
        by_property.append({'source': source, 'property_key': prop, 'run_id': run_id,
            'observations': len(rows), 'unmatched_2b_bullets': len(upstream),
            'terminal_boundaries': dict(Counter(r['mechanical_terminal_boundary'] for r in rows))})
    by_cond, by_issue = defaultdict(list), {}
    for row in all_rows:
        by_issue[(row['source'], row['property_key'], row['run_id'], row['issue_id'])] = row
        if row['condition_id']:
            by_cond[(row['source'], row['property_key'], row['run_id'], row['condition_id'])].append(row)
    populations = {'labels_v1_1': list(labels),
        'rc_misses': [c['case_id'] for c in queue['cases'] if c['case_id'].startswith('rc_') and c['lane'].startswith('miss_')],
        'rc_hallucinations': [c['case_id'] for c in queue['cases'] if c['case_id'].startswith('rc_') and c['lane'].startswith('halluc_')],
        'six_closeout': list(SIX), 'gold': [c['case_id'] for c in gold['cases']]}
    reviewed = []
    for population, ids in populations.items():
        for case_id in ids:
            case, card, label = cases.get(case_id, {}), cards_by_id.get(case_id, {}), labels.get(case_id, {})
            ref = case.get('run_ref') or card
            claim = case.get('v5_claim', {})
            cid = label.get('condition_id') or claim.get('condition_id') or claim.get('covering_rejected_condition_id') or card.get('condition_id')
            # Review cards store the condition id in the unblinded hidden payload.
            if not cid:
                cid = card.get('meta', {}).get('condition_id')
            key = (ref.get('source'), ref.get('property_key'), ref.get('run_id'), cid)
            traces = by_cond.get(key, [])
            if not traces and case.get('lineage'):
                traces = [by_issue[k] for iid in case['lineage'].get('issue_ids', [])
                          if (k := (*key[:3], iid)) in by_issue]
            prior = attributed.get(case_id)
            if not traces and prior:
                # Reviewed rationales sometimes preserve the exact issue id
                # when the gold case has no original condition. Restrict to
                # this exact source run and record that join's provenance.
                reviewed_ids = re.findall(r'(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])', prior.get('rationale', ''))
                traces = [by_issue[k] for iid in dict.fromkeys(reviewed_ids)
                          if (k := (*key[:3], iid)) in by_issue]
            first = ('2a' if prior and prior.get('attribution') == 'pass_2a'
                     else prior.get('first_responsible_stage') if prior else None)
            reviewed.append({'population': population, 'case_id': case_id,
                'source': key[0], 'property_key': key[1], 'run_id': key[2], 'condition_id': cid,
                'human_label': label or case.get('human_truth'), 'prior_attribution': prior,
                'first_responsible_stage': first, 'stage_evidence': 'prior_review' if first else 'unresolved',
                'observation_trace_ids': [r['issue_id'] for r in traces],
                'mechanical_terminal_boundaries': sorted({r['mechanical_terminal_boundary'] for r in traces}),
                'original_lineage': case.get('lineage'),
                'status': 'traced' if traces else 'no_existing_condition_or_unresolved_join',
                'replay_disposition': 'unresolved', 'replay_note': 'Fresh acceptance replicas have not been scored.'})
    for work in worklist:
        ref = work['case_ref']; traces = []
        card_match = re.search(r'rc_[0-9a-f]{12}', ref)
        card_id = card_match.group() if card_match else None
        if card_id in cards_by_id:
            related = next((r for r in reviewed if r['case_id'] == card_id), None)
            if not related:
                card = cards_by_id[card_id]
                related = {k: card.get(k) for k in ('source', 'property_key', 'run_id')}
        else:
            related = None
            parts = ref.split(':')
            if len(parts) == 5:
                item = by_issue.get((parts[0], parts[1], parts[2], parts[4]))
                traces = [item] if item else []
        # References have several historical formats; only exact, unique issue
        # ids are accepted. Never substitute a newest artifact or fuzzy text.
        explicit_ids = re.findall(r'(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])', ref)
        for iid in explicit_ids:
            hits = [r for k, r in by_issue.items() if k[3] == iid]
            if len(hits) == 1:
                traces = hits
        reviewed.append({'population': 'worklist', 'case_id': f"worklist_{work['worklist_index']}",
            'source_case_ref': ref, 'worklist_evidence': work,
            'source': traces[0]['source'] if traces else (related or {}).get('source'),
            'property_key': traces[0]['property_key'] if traces else (related or {}).get('property_key'),
            'run_id': traces[0]['run_id'] if traces else (related or {}).get('run_id'),
            'condition_id': traces[0]['condition_id'] if traces else (related or {}).get('condition_id'),
            'observation_trace_ids': [r['issue_id'] for r in traces] or (related or {}).get('observation_trace_ids', []),
            'mechanical_terminal_boundaries': sorted({r['mechanical_terminal_boundary'] for r in traces}) or (related or {}).get('mechanical_terminal_boundaries', []),
            'first_responsible_stage': (related or {}).get('first_responsible_stage'),
            'replay_disposition': 'unresolved', 'replay_note': 'Fresh acceptance replicas have not been scored.'})
    detail_path = Path(out_root) / 'investigation/observation_traces.json'
    write(detail_path, {'observations': all_rows, 'unmatched_2b_bullets': all_upstream, 'gaps': gaps})
    counters = {source: dict(Counter(r['mechanical_terminal_boundary'] for r in all_rows if r['source'] == source))
                for source in sorted({r['source'] for r in all_rows})}
    prior_counts = dict(Counter(r['first_responsible_stage'] or 'unresolved'
        for r in reviewed if r['population'] == 'rc_misses'))
    doc = {'schema_version': 1, 'model_calls': 0, 'scope': 'historical baseline; not fresh replay results',
        'population_counts': {p: sum(r['population'] == p for r in reviewed) for p in (*populations, 'worklist')},
        'property_counts': dict(Counter(p['source'] for p in by_property)), 'by_property': by_property,
        'mechanical_terminal_boundaries': counters, 'prior_rc_miss_attribution': prior_counts,
        'observation_detail': provenance([detail_path])[0], 'reviewed_cases': reviewed, 'reconstruction_gaps': gaps,
        'limitations': ['Unlabelled mechanical filtering is not a useful-loss rate.',
            '2a prose has no observation ids; first-loss causes use existing reviewed evidence, not lexical guesses.',
            'Old artifacts do not persist every 2c exclusion; unmatched 2b bullets are not proven deletions.',
            'Condition-based reviewed cards cannot measure observations that never reached conditions.',
            'Gold review covers two properties; cohort counts are not population recall.'],
        'sources': provenance([ROOT / ('reports/' + n) for n in ('error_attribution_queue.json',
            'error_attribution_gold_cases.json', 'error_attribution_verdicts.jsonl', 'labels_v1_1.json',
            'pass2d_worklist_reconciliation_20260910.json')] + [p for p, _ in artifacts.values()])}
    target = ROOT / 'reports/pipeline_loss_ledger_20260910.json'
    write(target, doc)
    lines = ['# Pipeline loss investigation — historical baseline', '',
        'The table counts observation endpoints, not useful observations lost. Several observations can represent one condition; package absorption preserves representation.', '',
        table(['Endpoint', *counters], [(boundary, *(counts.get(boundary, 0) for counts in counters.values()))
              for boundary in sorted({b for c in counters.values() for b in c})]), '',
        '## Reviewed causes', '',
        table(['First responsible stage from prior review', 'rc_ miss cases'], sorted(prior_counts.items())), '',
        'The final mechanical endpoint and the first semantic error are separate fields in the ledger. A mismatched catalog claim can fail Terra correctly while the useful observation was lost at 2d.', '',
        '## Coverage and limits', '', str(doc['population_counts']), '', *['- ' + x for x in doc['limitations']], '',
        f"Detailed observation traces: `{doc['observation_detail']['path']}`.", '',
        'Fresh replica dispositions remain unresolved until the replay is scored. No launch conclusion follows from this baseline census.', '']
    target.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', type=Path, default=DEFAULT_OUT)
    doc = run(parser.parse_args().out_root)
    print({'populations': doc['population_counts'], 'gaps': len(doc['reconstruction_gaps']),
           'reviewed_miss_first_stage': doc['prior_rc_miss_attribution']})
