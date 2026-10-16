# Experiments and controls

E1: native source fidelity and lineage, independent SQL reference sets and idempotence. WDC checks offer/split/schema integrity.

E2: four graph tasks x three seeds x 13 methods. Report ambiguous candidate-conditioned F1 and full-positive-universe end-to-end F1 separately. WDC covers 27 official configurations x three seeds x five methods; nine hardness/size settings x three seeds yield 27 fitted bundles reused over unseen variants. Train/test and validation/test offer overlap are both audited; all official pairs remain unchanged and disjoint subsets are separate diagnostics.

E3: eight adapted source-UID queries under unreconciled/predicted/oracle maps, plus eight native WWI queries. SQL/graph answers must match under the same identities; this equality alone does not certify identity correctness. Q2/Q4 are source-local. Monetary values retain source-unit semantics; BPI first-event anonymized cumulative case worth is not invoice payment, and overlapping tag totals are not global spend.

E4: frozen clean pairs/models/policies under attribute noise, optional missingness, anchor removal, relation removal, typed misleading rewiring and provenance missingness. Generic 12 conditions produce 1,872 method rows. WDC has six attribute-only conditions and 2,430 rows. Coverage/risk is preconstraint pair reliability; final accepted false merges are audited after one-to-one constraints.

E5: 1k/10k/100k node/offer targets x three trials x five datasets = 45. Graph targets include transactions/tags and may overshoot; matching record counts are separate. Later growth uses replicas, with every cross-replica true positive included. Capped output growth does not prove subquadratic computation. Prepared query timing has one warmup/five measured executions with common per-copy oracle identities and separate setup costs. Supplemental scorer-only timing excludes features; WDC supplemental volumes count tiled pair rows, not source offers.

Bootstrap: 1,000 paired canonical-group resamples per seed. Same-group pair weight is sampled group multiplicity; distinct-group pair weight is the product of endpoint multiplicities. Three seeds reuse populations; no pooled confirmatory confidence statement is made. Olist's pretraining seller-state adaptation followed a test-context diagnostic; source/provenance/query amendments are preserved. Expanded frozen replay differs from fresh-environment retraining of the original primary study. No human study was performed.

Run instructions and frozen configs are in the README and configs/. Source-safe dataset adapters and inference/evaluator separation remain in src/kgerp/expanded and the test suite.
