# Results and limitations

The complete available run history is under results/: initial-core-v1/v2 pilots, completed initial-core-v3, improvements-v1/v2, p9-replication-v1 and completed all-datasets-v1. Pilot and earlier results are historical, not interchangeable with the latest study. Each available run's manifest, scores, policies, metrics, fitted models, figures and amendments are retained where present.

Latest ambiguous conditional F1 (three-seed means): attributes/graph are WWI 0.7688/0.7888, Olist 0.6919/0.7842, DataCo 0.6962/0.6750, BPI 0.1242/0.5734. BPI's comparator is coarse modal history descriptors. Olist exact matching is stronger than learned attributes in this stratum. Flat SQL and graph scores are identical within 1e-12.

End-to-end graph pair F1 before constraints is 0.8261/0.7248/0.6486/0.4895. Candidate recall is 1.0000/0.8210/0.8744/0.7548. Clean postconstraint accepted links/false merges across seed trials are 107/1, 1092/37, 636/62 and 372/54. These trials share source populations.

WDC fully unseen mean F1 is 0.5369 attributes, 0.3919 cosine-only and 0.5584 augmented attributes. All 27 cells are included and 18 have development/test offer overlaps. Correlated configurations are not treated as independent population samples.

At the 100k target, main median seconds/recall are WWI 24.46/0.705, Olist 109.13/0.397, DataCo 69.57/0.594, BPI 47.62/0.172 and WDC 373.05/0.336. Completion under output caps comes with positive-recall losses; these are replicated projected workloads, not production systems.

Historical expanded verification records 36 control tests and 3,136 independent source/metric/model/query checks. This release runs the available tests and saved-score checks again, recording those genuine outcomes separately in VERIFICATION.md. It does not claim a new full expanded retraining or a raw-data replay against changed privacy derivatives.

The detailed paper, IEEE draft and supplement contain per-seed intervals, all WDC cells, query denominators, calibration, robustness, timings and failures. Generated CSVs and claim/hash provenance are retained. Negative findings and exploratory amendments remain visible.


## Revised-paper paired query controls

`results/manuscript-revision-v1` contains 192 new attribute/context method/query cells, 24 canonical maps, six BPI policy diagnoses, and an independently reproduced 27-cell WDC split audit. These are frozen-artifact analyses without new fitting.

| Mean Q6 answer F1 | WWI | Olist | DataCo | BPI 2019 |
| --- | ---: | ---: | ---: | ---: |
| Attributes | 0.8021 | 0.6942 | 0.7897 | 0.0630 |
| Gated context | 0.8318 | 0.7426 | 0.7468 | 0.5283 |

All six identity-sensitive template means improve on Olist and BPI and decline on DataCo. WWI improves on five templates, while Q5 declines from 1.0000 to 0.9980. SQL and graph answers agree in every new cell. BPI compares detailed histories against coarse modal descriptors, and these three-seed means have no pooled significance interpretation. Full per-seed counts, precision, recall, and paired differences are included.
