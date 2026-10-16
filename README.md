# Traceable Entity Reconciliation

**Context, Query Correctness, and Scale for Enterprise Knowledge Graphs**

Research software and reproducibility materials for the paper by **Muzakkiruddin Ahmed Mohammed and John R Talburt**, University of Arkansas at Little Rock.

[Read the paper](paper/revision-2026/main.pdf) · [Technical supplement](paper/revision-2026/supplement.pdf) · [Results](docs/RESULTS.md) · [Installation](docs/INSTALLATION.md) · [Datasets](docs/DATASETS.md)

KG-ERP preserves typed source observations and lineage, reconciles cross-source records using attributes and gated transaction context, and evaluates how identity decisions affect query answers. Independent SQL controls distinguish source fidelity, equivalent contextual evidence, and identity-dependent query correctness.

## Features

- Four enterprise graph tasks: Wide World Importers, Olist, DataCo, and BPI Challenge 2019.
- Attribute-only evaluation of all 27 official WDC Products configurations across three seeds.
- Common candidate pools, lexical features, frozen MiniLM cosine references, calibrated LR/RF comparators, gated context, and ablations.
- Source-preserving canonical views with type and one-to-one constraints, evidence records, and false-merge audits.
- Eight adapted queries per graph task, independent SQL/graph execution, and attribute/context identity-map comparisons.
- Robustness, calibration, selective-risk analysis, and 45 projected-workload scaling trials.

This is an exploratory research prototype. Task construction, retrieval losses, and measured limits are documented in the paper. No human study was performed.

## Installation

Python **3.12** is the recorded study runtime. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.fullstudy.lock
python -m pip install --no-deps -e .
python -m pytest -q
python scripts/verify_saved_results.py
```

The test suite and saved-score checks require no dataset downloads or model fitting. The saved-score verifier checks available fitted-model hashes when those files are installed; the smaller Git checkout intentionally omits fitted models.

For ordinary development, `python -m pip install -e ".[dev,fullstudy]"` resolves dependencies without reproducing the recorded lock. See [installation and configuration](docs/INSTALLATION.md).

## Main results

Three-seed mean F1 on the **same ambiguous candidate stratum**:

| Method | WWI | Olist | DataCo | BPI 2019 |
| --- | ---: | ---: | ---: | ---: |
| Attributes | 0.7688 | 0.6919 | 0.6962 | 0.1242 |
| Gated context | 0.7888 | 0.7842 | 0.6750 | 0.5734 |
| Difference | +0.0200 | +0.0923 | -0.0212 | +0.4492 |

Paired identity-query answer F1 **after identical one-to-one constraints**, using arithmetic means over the same three seeds:

| Identity map | WWI | Olist | DataCo | BPI 2019 |
| --- | ---: | ---: | ---: | ---: |
| Attributes | 0.8021 | 0.6942 | 0.7897 | 0.0630 |
| Gated context | 0.8318 | 0.7426 | 0.7468 | 0.5283 |

Context reduces average performance on DataCo. BPI uses coarse history-derived baseline descriptors rather than vendor names. WWI's transaction-link query slightly declines despite its identity-query improvement. Flat SQL context yields equal features and scores, so the matching comparison concerns evidence rather than graph storage.

WDC's fully unseen mean F1 is **0.5369** for attributes and **0.5584** with a frozen embedding cosine. Eighteen used WDC configurations have offer intersections and are flagged; all nine fully unseen cells are offer-disjoint. All 45 scaling trials complete, but larger replicated workloads lose substantial true pairs during bounded retrieval.

Conditional pair F1, full-positive-universe pair F1, canonical-link precision/recall, and query-answer F1 use different denominators. Full results and uncertainty are in [docs/RESULTS.md](docs/RESULTS.md), [the expanded study report](results/all-datasets-v1/REPORT.md), and [the revision controls](results/manuscript-revision-v1/).

## Research materials and datasets

The Git repository contains source, configurations, tests, saved predictions/results, provenance records, manuscript sources, and license notices. Fitted models, encoder weights, and permitted compressed data are distributed in the companion **`kg-erp-research-materials-v1.1.0.zip`** asset rather than Git history. The main repository needs no Git LFS.

After the companion archive is uploaded under the `v1.1.0` release:

```bash
python scripts/fetch_research_materials.py --repository OWNER/REPOSITORY
python scripts/materialize_release_data.py
```

For a local downloaded archive:

```bash
python scripts/fetch_research_materials.py --archive /path/to/kg-erp-research-materials-v1.1.0.zip
python scripts/materialize_release_data.py
```

The fetch helper validates SHA-256 values and restores only declared assets. It does not replace software, results, or existing differing files. The materializer checks decoded hashes. Original sensitive DataCo fields and Olist review prose are excluded from the distribution; historical DataCo name-based replay requires separately acquired original data. WDC raw offer text is acquired through official URLs because a rehosting license was not established. Dataset and pretrained-weight terms remain separate from the software license.

Read [dataset provenance and licenses](docs/DATASETS.md) and [reproduction boundaries](docs/REPRODUCIBILITY.md) before a full replay.

## Reproduce experiments

`configs/all_datasets.yaml` records seeds 17/29/43, group selection, learner catalogs, calibration and policy roles, context support, blocking, stress levels, bootstrap replicates, and scale targets. Use a new run ID:

```bash
python scripts/acquire_all_datasets.py
python -m kgerp.expanded --run-id expanded-reproduction
```

To reproduce the new paired query and split controls from the original frozen run, after restoring the required source inputs:

```bash
python paper/revision-2026/analysis/run_controls.py --project . --output results/new-paper-controls
```

This reexecutes queries and audits retained artifacts without refitting models or thresholds. Original raw inputs are required for an exact historical replay; privacy derivatives are not silently substituted for the original matching names.

## Build the paper

The current manuscript is `paper/revision-2026/main.tex`. With a TeX distribution and the packages listed in [the manuscript instructions](paper/revision-2026/README.md):

```bash
python paper/revision-2026/build.py
python paper/revision-2026/analysis/verify_manuscript.py
```

The supplied paper is 10 pages and the revision supplement is seven pages. `availability.tex` contains the repository-link placeholder. Historical publications remain available for comparison with earlier result runs.

## Verification

Recorded evidence includes **36 passing project tests**, **748 additional execution checks**, and **1,436 manuscript checks**. The new comparison contains **192 method/query cells**, 24 identity maps, and a complete independent audit of all 27 WDC configurations. The current publication package is tested again during assembly; exact outcomes are recorded in [VERIFICATION.md](VERIFICATION.md).

The expansion has not undergone independent fresh-environment retraining. Hosted CI results are only available after a repository is pushed; local and historical results are reported separately.

## Repository layout

```text
src/kgerp/                  Source ingestion, matching, queries, and evaluations
configs/                    Recorded protocols
tests/                      Control tests
scripts/                    Acquisition, preparation, verification, and material restoration
results/                    Saved experiments, predictions, and revision controls
data/manifests/             Source provenance and schemas
data/research_materials.json Companion asset index and hashes
paper/revision-2026/        Current paper, supplement, editable sources, and analyses
docs/                       Installation, methods, results, datasets, and reproduction
verification/               Test, metric, privacy, and publication verification
licenses/                   Upstream license notices
```

## Citation and license

Use [CITATION.cff](CITATION.cff) for the software artifact and the paper's actual bibliographic record when available. No acceptance or archival DOI is asserted.

Original software is licensed under [MIT](LICENSE). Datasets, model assets, manuscript rights, and upstream resources retain their own conditions; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Olist-derived materials retain applicable noncommercial and ShareAlike terms.
