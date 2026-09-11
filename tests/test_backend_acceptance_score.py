from scripts.analysis.backend_acceptance_score import match_condition, classify_case
from scripts.analysis.pipeline_loss_ledger import trace_artifact


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
