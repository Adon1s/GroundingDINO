"""Archive the first paid replica before replica 2 starts; no model calls."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.terra_miss_mechanisms import read, write, table, provenance


def dollars(amount):
    return f"${amount['low']:,}–${amount['high']:,}"


def main():
    root = ROOT / 'artifacts_canary/backend_acceptance_20260910'
    if list((root / 'replica_2').glob('*/*/complete.json')):
        raise ValueError('Replica 1 ledger snapshot must precede paid replica 2')
    source = ROOT / 'reports/backend_acceptance_score_20260910.json'
    score, manifest = read(source), read(root / 'manifest.json')
    properties = [p for p in score['properties'] if p['replica'] == 1]
    assert len(properties) == 18
    checks = [c for c in score['checks'] if c['replica'] == 1]
    assert len(checks) == 36 and all(c['passed'] for c in checks)
    assert all(c['passed'] for c in score['work_dedup_audit'] if c['replica'] == 1)
    usage = {}
    for model in ('terra', 'sol'):
        path = Path(manifest['usage_root']) / '.renovation_architecture' / (model + '_usage.sqlite3')
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute(
                f'SELECT * FROM {model}_usage WHERE julianday(created_at) >= julianday(?)',
                (manifest['batch_start'],))]
        artifact_tokens = sum(p[model + '_usage']['total_tokens'] for p in properties)
        artifact_calls = sum(p[model + '_usage']['call_count'] for p in properties)
        debit = sum(r['debited_tokens'] for r in rows)
        unsettled = sum(r['state'] != 'settled' for r in rows)
        matched = artifact_tokens == debit == sum(r['provider_total_tokens'] or 0 for r in rows)
        assert matched and not unsettled and artifact_calls == len(rows)
        usage[model] = {'artifact_tokens': artifact_tokens, 'ledger_debit': debit,
            'calls': len(rows), 'unsettled': unsettled, 'reconciled': matched,
            'utc_days': sorted({r['utc_day'] for r in rows}), 'ledger_rows': rows}
    cases = [c for c in score['reviewed_case_dispositions'] if c['replica'] == 1]
    labels = [c for c in cases if c['population'] == 'labels_v1_1']
    missing = [c for c in labels if c.get('quality_effect') == 'warranted_work_missing']
    obs = [c for c in score['observation_changes'] if c['replica'] == 1]
    changes = Counter('unchanged' if not r['selection_changed'] else
        'null_to_item' if r['before_catalog_item_id'] is None else
        'item_to_null' if r['after_catalog_item_id'] is None else 'item_to_item' for r in obs)
    population = {}
    for name in sorted({c['population'] for c in cases}):
        group = [c for c in cases if c['population'] == name]
        population[name] = {'total': len(group),
            'dispositions': dict(Counter(c['replay_disposition'] for c in group)),
            'quality_effects': dict(Counter(c.get('quality_effect', 'unresolved') for c in group))}
    report = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'replica_1_complete_final_acceptance_pending', 'model_calls_by_this_script': 0,
        'manifest_identity': manifest['identity'], 'batch_start': manifest['batch_start'],
        'verified_artifacts': 18, 'checks': checks, 'usage': usage,
        'remaining_terra_batch_tokens': manifest['batch_terra_ceiling'] - usage['terra']['ledger_debit'],
        'population_summary': population, 'selection_changes': dict(changes),
        'before_boundaries': dict(Counter(r['before_boundary'] for r in obs)),
        'after_boundaries': dict(Counter(r['after_boundary'] for r in obs)),
        'warranted_work_missing': missing,
        'newly_missing': [c['case_id'] for c in missing if c['before_review']['verdict'] == 'supported'],
        'six_closeout': [c for c in cases if c['population'] == 'six_closeout'],
        'material_price_review': [p for p in properties if p['requires_steven_review']],
        'named_subsets': {k: [r for r in rows if r['replica'] == 1]
            for k, rows in score['named_subsets'].items()},
        'work_dedup_audit': [r for r in score['work_dedup_audit'] if r['replica'] == 1],
        'provenance': provenance([source, root / 'manifest.json']),
        'limitations': ['Stored-to-fresh differences confound catalog, temperature and model changes.',
            'A mechanical endpoint is not proof of the first semantic error.',
            'The second replica, production smoke and material price review remain pending.']}
    dest = ROOT / 'reports/backend_acceptance_replica1_packet_20260911.json'
    write(dest, report)
    lines = ['# First replica investigation and review packet', '',
        'All 18 artifacts pass the production verifier and catalog invariants. Final acceptance is pending.', '',
        table(['Model', 'Artifact / ledger tokens', 'Calls', 'Unsettled'],
            [(m, u['artifact_tokens'], u['calls'], u['unsettled']) for m, u in usage.items()]), '',
        f"Terra remaining in the original shared batch: {report['remaining_terra_batch_tokens']:,} tokens.", '',
        '## Observation selection', '', table(['Change', 'Observations'], changes.items()), '',
        'These are selection changes, not counts of useful losses. All 3,668 observations were compared.', '',
        '## Human-labeled warranted work still missing', '',
        table(['Case', 'Property', 'Original verdict', 'Current rationale', 'Observed work allowance delta'],
            [(c['case_id'], c['property_key'], c['before_review']['verdict'],
              ' / '.join(r.get('rationale', '') for r in c['after_reviews']),
              dollars(c['observed_work_allowance_delta'])) for c in missing]), '',
        'Allowances are non-additive where conditions share work. Zero delta for an old miss does not value its recovery at zero.', '',
        '## Material headline changes requiring final review', '',
        'Every row below uses effective applied package amounts. Component deltas reconcile exactly to the headline.', '']
    for p in report['material_price_review']:
        lines += [f"### {p['property_key']}", '',
            f"Stored run 1: {dollars(p['stored_run_1'])}; fresh replica 1: {dollars(p['headline'])}; stored run 2: {dollars(p['stored_run_2'])}.", '',
            table(['Component', 'Before low/high', 'After low/high', 'Delta low/high', 'Application reason; Sol'],
                [(r['component'], dollars((r['before'] or {}).get('amount', {'low': 0, 'high': 0})),
                  dollars((r['after'] or {}).get('amount', {'low': 0, 'high': 0})), dollars(r['delta']),
                  (r['after'] or {}).get('reason', 'standalone' if r['component'] == 'standalone' else 'absent') + '; ' +
                  str(((r['after'] or {}).get('sol_decision') or {}).get('decision', 'n/a')))
                 for r in p['price_components'] if any(r['delta'].values())]), '']
    lines += ['## Limits', '', *['- ' + s for s in report['limitations']], '']
    dest.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print({'artifacts': 18, 'newly_missing': report['newly_missing'],
        'price_review_properties': len(report['material_price_review'])})


if __name__ == '__main__':
    main()
