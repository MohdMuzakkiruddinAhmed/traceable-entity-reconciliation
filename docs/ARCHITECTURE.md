# Architecture and evidence boundaries

The pipeline imports native rows, creates typed property graphs, projects permitted source-local matching evidence, blocks pairs, computes attributes/context, applies frozen learned policies and adds constrained canonical views. It retains original observations rather than replacing them with canonical records.

Native imports use SQL reference foreign-key joins, lineage completeness, dangling-edge checks and idempotent edge import. Matching graphs use NetworkX; expanded native source imports use disk SQLite. A separate SQL context extractor operates on the same permitted projection and checks feature/score equality with graph traversal. Native source fidelity and flat-context equality are different controls.

Generic projections are Master -> Transaction -> ContextCategory. Context includes category/activity distributions, degree, shared tags, support, date and compatible quantity/value moments. Generic support requires two projected tags; explicit missing-value markers can count. WWI additionally uses independently inferred exact-name/category-compatible typed master anchors, excluding candidate endpoints. Completeness weights measure available provenance, not estimated correctness.

The gated score uses context when supported and otherwise returns attribute probability. A separate frozen threshold can still produce different decisions. Greedy descending-score acceptance enforces same type and one-to-one endpoints; canonical entities/links are added only after scoring. Rejected, abstained and conflicting candidates remain in audit records.

Truth, original master keys and canonical identifiers are evaluator-only. They do not enter attribute text, context summaries or model selection as features. Character TF-IDF is trained on training records only. Validation canonical groups are separated into model-selection, calibration and policy roles; cross-role pairs are omitted. Frozen models and policies are recorded before test metrics.

NoContext shares the attribute method; B4 and B5 use identical evidence/model/policy. NoGate changes fallback and policy, not the trained weighted learner. NoSupport retrains without support inputs but retains the original final gate. UnitWeights changes inputs to a frozen weighted estimator. NoTransactions removes the whole WWI transaction block but only numeric/date/unit features in generic adapters. OneHop concerns normalized projection depth. WDC has no transaction graph or ERP query workload.
