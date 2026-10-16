"""Serial post-run repair, cost studies, verification, reporting and packaging."""
import csv
from datetime import datetime,timezone
import subprocess
import sys
from kgerp.io import ROOT,read_json,write_json,digest


def run(script,*arguments):
    print('Finalization: '+script,flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*arguments],cwd=ROOT,check=True)


def main():
    out=ROOT/'results/all-datasets-v1'
    if read_json(out/'manifest.json')['status']!='completed':raise ValueError('Main five-dataset execution must finish first')
    run('repair_olist_lineage.py')
    run('profile_all_datasets.py')
    subprocess.run([sys.executable,'-m','pytest','-q','--junitxml=results/all-datasets-tests.xml'],cwd=ROOT,check=True)
    run('report_all_datasets.py')
    run('verify_all_datasets.py')
    verification=read_json(out/'verification.json')
    path=ROOT/'README.md';text=path.read_text(encoding='utf-8')
    old="The five-dataset execution is being completed in `results/all-datasets-v1`; its [manifest](results/all-datasets-v1/manifest.json) is the authoritative live status."
    new="All applicable non-human E1–E5 evaluations are completed in `results/all-datasets-v1`, including all 27 WDC pairwise configurations across three seeds and 45 scaling trials. WDC ERP queries are inapplicable because the input contains offers rather than transactions."
    if old not in text:raise ValueError('Expected README live-status paragraph')
    text=text.replace(old,new)
    before="The sections below preserve the original two-dataset study and its instructions. Original code/results, the earlier improvements and the manuscript remain available separately."
    after=("Read the [five-dataset results](results/all-datasets-v1/REPORT.md), [current plan audit](notes/plan_coverage_all_datasets.md), and [independent verification](results/all-datasets-v1/verification.json). "
        +f"All {verification['tests']} tests and {verification['checks_passed']:,} audit checks passed. Verification replays all 12 graph-dataset seed evaluations and all 81 WDC seed/configuration cells, recomputes metrics and re-executes queries. "
        +"The private review bundle is `output/kg-erp-all-datasets-review.zip`. E6 is excluded; expanded manuscript integration, author/release sign-off and submission remain unfinished.\n\n"
        +"Full native source imports are distinct from matching scope: all 2,913 eligible Olist repeat-customer groups, a stable DataCo sample of 1,500 eligible groups, and all 1,378 eligible BPI vendor groups. The latter two use controlled source views, without annotated naturally dirty identity labels. Source-adapter and provenance corrections are recorded as exploratory amendments.\n\n"
        +"The sections below preserve the original two-dataset study and its instructions. Original code/results, the earlier improvements and the manuscript remain available separately.")
    if before not in text:raise ValueError('Expected historical scope paragraph')
    path.write_text(text.replace(before,after),encoding='utf-8')
    write_json(out/'finalization_manifest.json',{
        'status':'completed_and_verified','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'tests':verification['tests'],'checks_passed':verification['checks_passed'],
        'driver_sha256':digest(__file__),'report_sha256':digest(out/'REPORT.md'),
        'verification_sha256':digest(out/'verification.json'),
        'source_inputs_and_experimental_artifacts':'Immutable hashes in manifest.json; generated report/figure hashes in report_manifest.json.',
        'human_study':'excluded_by_user','publication':'Expanded manuscript integration, author/release review and submission are unfinished.'})
    run('package_all_datasets.py')
    print('Five-dataset non-human study verified and packaged',flush=True)


if __name__=='__main__':main()
