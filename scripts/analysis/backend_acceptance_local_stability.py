"""Compare verified local 2d replicas independently of paid downstream completion."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.terra_miss_mechanisms import read, write, provenance, table
from scripts.replay_frozen_upstream_acceptance import DEFAULT_OUT, load_manifest


def selections(records):
    rows = {}
    for photo in records:
        resolutions = {r['issue_id']: r for r in photo['resolutions']}
        for observation in photo['observations']:
            iid = observation['issue_id']
            if iid in rows:
                raise ValueError(f'Duplicate observation identity: {iid}')
            record = resolutions.get(iid)
            rows[iid] = {'photo_key': photo['photo_key'], 'observation': observation,
                'item': (record or {}).get('resolved_item_id'), 'record': record}
    return rows


def main():
    manifest = load_manifest(DEFAULT_OUT)
    comparisons, properties, inputs = [], [], []
    for prop in manifest['properties']:
        prepared = []
        for replica in (1, 2):
            directory = DEFAULT_OUT / f'replica_{replica}' / prop['property_key'] / prop['source_run_id']
            stamp = directory / 'prepared_complete.json'
            if not stamp.exists():
                break
            completion = read(stamp)
            if completion['manifest_identity'] != manifest['identity'] or any(
                    provenance([ROOT / f['path']])[0] != f for f in completion['files']):
                raise ValueError(f'Prepared checkpoint drift: {directory}')
            path = directory / 'resolution_record.json'
            prepared.append(selections(read(path)))
            inputs.append(path)
        if len(prepared) != 2:
            continue
        one, two = prepared
        if one.keys() != two.keys():
            raise ValueError(f'Frozen observation identity drift: {prop["property_key"]}')
        changes = 0
        for iid in sorted(one):
            a, b = one[iid], two[iid]
            if a['observation'] != b['observation'] or a['photo_key'] != b['photo_key']:
                raise ValueError(f'Frozen observation text drift: {iid}')
            changed = a['item'] != b['item']
            changes += changed
            comparisons.append({'property_key': prop['property_key'], 'issue_id': iid,
                'photo_key': a['photo_key'], 'observation': a['observation'],
                'replica_1_item': a['item'], 'replica_2_item': b['item'], 'selection_changed': changed,
                'replica_1_record': a['record'] if changed else None,
                'replica_2_record': b['record'] if changed else None})
        properties.append({'property_key': prop['property_key'], 'observations': len(one), 'changed': changes})
    reviewed = read(ROOT / 'reports/pipeline_loss_ledger_20260910.json')['reviewed_cases']
    changed_keys = {(r['property_key'], r['issue_id']) for r in comparisons if r['selection_changed']}
    affected = [{'population': c['population'], 'case_id': c['case_id'], 'property_key': c['property_key'],
        'changed_issue_ids': sorted(i for i in c.get('observation_trace_ids', [])
            if (c['property_key'], i) in changed_keys)} for c in reviewed if c.get('source') == 'canary'
        and any((c['property_key'], i) in changed_keys for i in c.get('observation_trace_ids', []))]
    summary = Counter('unchanged' if not r['selection_changed'] else
        'null_to_item' if r['replica_1_item'] is None else
        'item_to_null' if r['replica_2_item'] is None else 'item_to_item' for r in comparisons)
    report = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'model_calls': 0, 'manifest_identity': manifest['identity'], 'complete': len(properties) == 18,
        'compared_properties': len(properties), 'expected_properties': 18,
        'summary': dict(summary), 'properties': properties, 'observations': comparisons,
        'affected_reviewed_cases': affected, 'provenance': provenance(inputs),
        'limitation': 'Selection repeatability only; no claim about useful recovery or downstream Terra/Sol stability.'}
    dest = ROOT / 'reports/backend_acceptance_local_stability_20260911.json'
    write(dest, report)
    dest.with_suffix('.md').write_text('\n'.join(['# Local Pass 2d repeatability', '',
        f'Compared {len(properties)}/18 verified property pairs; {len(comparisons):,} observations.', '',
        table(['Selection change', 'Count'], summary.items()), '',
        table(['Property', 'Observations', 'Changed'],
            [(p['property_key'], p['observations'], p['changed']) for p in properties]), '',
        report['limitation'], '']), encoding='utf-8', newline='\n')
    print({'properties': len(properties), 'observations': len(comparisons), 'summary': dict(summary)})


if __name__ == '__main__':
    main()
