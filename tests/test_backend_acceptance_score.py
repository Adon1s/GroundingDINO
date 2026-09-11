from scripts.analysis.backend_acceptance_score import match_condition, classify_case, compare_packages
from scripts.analysis.pipeline_loss_ledger import trace_artifact


def test_price_components_use_application_not_sol_approval_or_candidate_allowance():
    import copy
    import pytest
    from scripts.analysis.backend_acceptance_score import compare_prices, price_components
    old = {'totals': {'standalone': {'low': 10, 'high': 20}, 'headline': {'low': 110, 'high': 220}},
        'package_candidates': [{'package_candidate_id': 'p', 'package_type': 'bedroom',
            'estimate_unit_id': 'bedroom_1', 'pricing_tier': 'full', 'low': 100, 'high': 200}],
        'package_decisions': [{'package_candidate_id': 'p', 'decision': 'approve'}],
        'package_applications': [{'package_candidate_id': 'p', 'effective_low': 100, 'effective_high': 200,
            'status': 'applied', 'reason_code': 'approved_absorbs_children', 'absorbed_work_item_ids': ['w']}]}
    new = copy.deepcopy(old)
    new['totals'] = {'standalone': {'low': 30, 'high': 60}, 'headline': {'low': 30, 'high': 60}}
    new['package_applications'][0].update(effective_low=0, effective_high=0, status='not_applied',
        reason_code='opportunity_only_interior_modernization', absorbed_work_item_ids=[])
    rows = compare_prices(old, new)
    assert sum(r['delta']['low'] for r in rows) == -80
    assert rows[0]['after']['sol_decision']['decision'] == 'approve'
    new['totals']['headline']['low'] = 31
    with pytest.raises(ValueError, match='reconcile'):
        price_components(new)


def test_condition_join_survives_new_catalog_ids_and_names_successors():
    old = {'catalog_item_id': 'dated_window_treatment_valance', 'issue_ids': ['i']}
    index = {'conds': {'new_id': {'catalog_item_id': 'dated_window_valance_or_curtains', 'issue_ids': ['i']}},
             'evs': {'new_id': {'photo_keys': ['p2', 'p1']}}}
    assert match_condition(old, {'photo_keys': ['p1', 'p2']}, index) == (['new_id'], 'item_and_exact_evidence')
    index['conds']['new_id']['catalog_item_id'] = 'unrelated'
    assert match_condition(old, {'photo_keys': ['p1', 'p2']}, index)[0] == []


def test_same_unsupported_verdict_is_still_a_loss_for_warranted_work():
    old = {'catalog_item_id': 'paint'}
    review = {'verdict': 'unsupported'}
    disp = {'terminal_route': 'work', 'disposition': 'excluded'}
    index = {'conds': {'c': old}, 'revs': {'c': review}, 'disps': {'c': disp}}
    disposition, _ = classify_case(old, review, disp, ['c'], index, {'claim': 'exact', 'work': 'warranted'})
    assert disposition == 'lost'


def test_intentional_trim_policy_is_not_reported_as_work_loss():
    old = {'catalog_item_id': 'dated_interior_trim'}
    index = {'disps': {'c': {'terminal_route': 'no_action'}}}
    assert classify_case(old, {}, {}, ['c'], index, {'claim': 'exact', 'work': 'warranted'})[0] == 'policy-superseded'


def test_unknown_gold_without_condition_does_not_join_by_photo_alone():
    old = {'catalog_item_id': 'paint', 'issue_ids': ['one']}
    index = {'conds': {'c': {'catalog_item_id': 'floor', 'issue_ids': ['one']}},
             'evs': {'c': {'photo_keys': ['p']}}}
    assert match_condition(old, {'photo_keys': ['p']}, index)[0] == []


def test_cap007_blinds_no_action_is_policy_even_when_old_label_warranted_work():
    old = {'catalog_item_id': 'dated_window_treatment_valance'}
    index = {'conds': {'c': {'catalog_item_id': 'window_blinds_basic_or_plain'}},
             'disps': {'c': {'terminal_route': 'no_action'}}}
    assert classify_case(old, {}, {}, ['c'], index, {'claim': 'exact', 'work': 'warranted'})[0] == 'policy-superseded'


def test_catalog_support_change_is_not_package_shape_until_work_or_price_changes():
    import copy
    original = {'work_items': [{'work_item_id': 'w', 'action_code': 'CABINET_REPLACE',
        'billable_unit_id': 'kitchen', 'catalog_item_ids': ['dated', 'worn']}],
        'package_decisions': [], 'package_candidates': [{'package_candidate_id': 'p',
        'package_type': 'kitchen', 'estimate_unit_id': 'kitchen', 'driver_work_item_ids': ['w'],
        'child_work_item_ids': ['w'], 'support_work_item_ids': [], 'low': 100, 'high': 200}]}
    current = copy.deepcopy(original)
    current['work_items'][0]['catalog_item_ids'] = ['dated']
    assert compare_packages(original, current)[0]['change'] == 'support_list_only'
    current['package_candidates'][0]['low'] = 50
    assert compare_packages(original, current)[0]['change'] == 'shape'
