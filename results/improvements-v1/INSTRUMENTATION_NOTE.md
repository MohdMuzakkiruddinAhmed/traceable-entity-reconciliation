# Instrumentation correction

This exploratory run is retained. Its `e5/blocking_scale.csv` field named `attribute_feature_seconds` measured the full attribute-plus-context feature construction, including anchor inference and the independent SQL control. It must not be interpreted as attribute-only time. The report's matching/recall/query tables do not use this field.

`improvements-v2` renames it to `full_feature_seconds`, adds an explicit FlatWeighted scorer row under the identical learned policy, and snapshots the complete source used for the extension. Features, selection, calibration, threshold rules and candidate budgets are unchanged. Source corresponding to v1's recorded hashes is retained in `code_snapshot/`.
