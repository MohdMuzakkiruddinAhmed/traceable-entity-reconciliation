"""Independently recompute included summary metrics and check manuscript files.

Requires pypdf for page/text checks. No source acquisition or model execution.
"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def close(a, b):
    return abs(float(a) - float(b)) < 1e-12


def rows(name):
    with (DATA / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


queries = rows('query_controls.csv')
lookup = {(r['dataset'], r['seed'], r['method'], r['query']): r for r in queries}
check('192_unique_method_query_cells', len(queries) == len(lookup) == 192)
for i, r in enumerate(queries):
    expected, actual, tp = [int(r[k]) for k in ('expected_answers', 'actual_answers', 'tp')]
    check(f'query_{i}_f1', close(r['f1'], 2 * tp / (expected + actual)))
    check(f'query_{i}_precision', close(r['precision'], tp / actual if actual else 0.))
    check(f'query_{i}_recall', close(r['recall'], tp / expected))
    check(f'query_{i}_parity', r['SQL_KG_exact'] == 'True')
    if r['query'] in ('Q2', 'Q4'):
        check(f'query_{i}_local_exact', float(r['f1']) == 1.)
paired = rows('query_paired_differences.csv')
check('96_unique_paired_cells', len(paired) == len({(r['dataset'], r['seed'], r['query']) for r in paired}) == 96)
for i, r in enumerate(paired):
    a = lookup[(r['dataset'], r['seed'], 'B2Attributes', r['query'])]
    c = lookup[(r['dataset'], r['seed'], 'B5Graph', r['query'])]
    check(f'pair_{i}_values', close(a['f1'], r['B2Attributes']) and close(c['f1'], r['B5Graph']))
    check(f'pair_{i}_difference', close(r['delta_f1'], float(c['f1']) - float(a['f1'])))
means = rows('query_paired_means.csv')
check('32_query_means', len(means) == 32)
for i, r in enumerate(means):
    subset = [s for s in paired if s['dataset'] == r['dataset'] and s['query'] == r['query']]
    check(f'mean_{i}_three_seeds', {s['seed'] for s in subset} == {'17', '29', '43'})
    for field in ('B2Attributes', 'B5Graph', 'delta_f1'):
        check(f'mean_{i}_{field}', close(r[field], math.fsum(float(s[field]) for s in subset) / 3))
identity = rows('identity_controls.csv')
check('24_identity_maps', len(identity) == 24)
for i, r in enumerate(identity):
    accepted, false, true, expected = [int(r[k]) for k in ('accepted_links', 'false_merges', 'true_accepted', 'test_groups')]
    check(f'identity_{i}_counts', accepted == false + true)
    check(f'identity_{i}_precision', close(r['precision'], true / accepted))
    check(f'identity_{i}_recall', close(r['recall'], true / expected))
    check(f'identity_{i}_f1', close(r['f1'], 2 * true / (accepted + expected)))
    check(f'identity_{i}_Q6_equal', close(r['f1'], lookup[(r['dataset'], r['seed'], r['method'], 'Q6')]['f1']))
for i, r in enumerate(rows('identity_totals.csv')):
    subset = [s for s in identity if s['dataset'] == r['dataset'] and s['method'] == r['method']]
    for field in ('accepted_links', 'false_merges', 'true_accepted', 'test_groups'):
        check(f'identity_total_{i}_{field}', int(r[field]) == sum(int(s[field]) for s in subset))
    check(f'identity_total_{i}_precision', close(r['precision'], int(r['true_accepted']) / int(r['accepted_links'])))
    check(f'identity_total_{i}_recall', close(r['recall'], int(r['true_accepted']) / int(r['test_groups'])))
gate = rows('gate_policy_diagnostic.csv')
for i, r in enumerate(gate):
    check(f'gate_{i}_all_zero', int(r['gate_zero_pairs']) == int(r['n_pairs']) and float(r['score_max_difference_from_attributes']) == 0.)
    check(f'gate_{i}_end_f1', close(r['end_to_end_f1'], 2 * int(r['tp']) / (int(r['tp']) + int(r['fp']) + 276)))
audit = rows('wdc_split_audit.csv')
check('27_WDC_cells', len(audit) == 27)
check('18_flagged_WDC_cells', sum(int(r['train_test_offer_overlap']) > 0 or int(r['validation_test_offer_overlap']) > 0 for r in audit) == 18)
check('nine_fully_unseen_disjoint', sum(r['unseen_percent'] == '100' and int(r['train_test_offer_overlap']) == int(r['validation_test_offer_overlap']) == 0 for r in audit) == 9)
for i, r in enumerate(audit):
    check(f'audit_{i}_same_attributes', r['train_test_offer_overlap'] == r['train_test_identical_offer_attributes'] and r['validation_test_offer_overlap'] == r['validation_test_identical_offer_attributes'])
    check(f'audit_{i}_all_pairs', int(r['test_pairs']) == 4500 and 0 < int(r['offer_disjoint_test_pairs']) <= 4500)
    check(f'audit_{i}_hashes', all(re.fullmatch('[0-9a-f]{64}', r[k]) for k in ('train_decompressed_sha256', 'validation_decompressed_sha256', 'test_decompressed_sha256')))
manifest = json.loads((DATA / 'verification.json').read_text())
check('748_execution_checks', manifest['status'] == 'passed' and manifest['checks_passed'] == len(manifest['checks']) == 748)
check('execution_checks_all_pass', all(c['passed'] for c in manifest['checks']))
for name, h in manifest['output_sha256'].items():
    check('saved_output_hash:' + name, sha(DATA / name) == h)
check('control_script_hash', sha(ROOT / 'analysis/run_controls.py') == manifest['script_sha256'])
xml = ET.parse(DATA / 'tests.xml').getroot()
suites = [xml] if xml.tag == 'testsuite' else list(xml)
check('36_tests_pass', sum(int(s.get('tests', 0)) for s in suites) == 36 and all(int(s.get('failures', 0)) + int(s.get('errors', 0)) + int(s.get('skipped', 0)) == 0 for s in suites))
tex = (ROOT / 'main.tex').read_text(encoding='utf-8')
query_names = {'Q1': 'Shared tags', 'Q3': 'Attribution', 'Q5': 'Transaction links', 'Q6': 'Identities', 'Q7': 'Tag totals', 'Q8': 'Lineage'}
for q, name in query_names.items():
    cells = []
    for dataset in ('WWI', 'Olist', 'DataCo', 'BPI2019'):
        r = next(r for r in means if r['dataset'] == dataset and r['query'] == q)
        cells.append(f"{float(r['B2Attributes']):.3f}".lstrip('0') + ' / ' + f"{float(r['B5Graph']):.3f}".lstrip('0'))
    check('main_query_table:' + q, q + ': ' + name + ' & ' + ' & '.join(cells) in tex)
pdf_info = {}
for name, expected in (('main', 10), ('supplement', 7)):
    path = ROOT / (name + '.pdf')
    reader = PdfReader(path)
    content = '\n'.join(p.extract_text() for p in reader.pages)
    check(name + '_page_count', len(reader.pages) == expected)
    check(name + '_no_unresolved_tokens', '??' not in content and '[?]' not in content)
    check(name + '_GitHub_placeholder', 'repository link to be added' in content)
    check(name + '_query_evidence', '192' in content and '748' in content)
    check(name + '_author', 'Muzakkiruddin Ahmed Mohammed' in content and 'John R Talburt' in content)
    pdf_info[name] = {'pages': len(reader.pages), 'sha256': sha(path)}
for name in ('main', 'supplement'):
    log = ROOT / 'build' / (name + '.log')
    if log.exists():
        t = log.read_text(errors='replace')
        check(name + '_resolved_typesetting', not any(marker in t for marker in ('Overfull ', 'There were undefined references', 'There were undefined citations')))
report = {'status': 'passed', 'checks_passed': len(checks), 'checks': checks, 'pdfs': pdf_info,
          'project_tests': 36, 'analysis_execution_checks': 748, 'query_cells': 192, 'wdc_cells': 27,
          'no_retraining': True, 'GitHub_link': 'author placeholder retained'}
(ROOT / 'verification').mkdir(exist_ok=True)
(ROOT / 'verification/manuscript_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: report[k] for k in ('status', 'checks_passed', 'pdfs')}))
