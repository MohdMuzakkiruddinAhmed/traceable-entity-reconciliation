# Publication package verification

Verified locally on **15 October 2026** against the prepared repository.

| Check | Genuine result |
| --- | --- |
| Self-contained project tests | **36 passed**, zero failures, errors, or skips |
| Saved-score audit before optional model restoration | **2,270 checks passed** |
| Companion material restoration | **103 files restored and SHA-256 verified** |
| Saved-score audit with restored fitted models | **2,309 checks passed** |
| Current paper-result/PDF verification | **1,434 checks passed**; 10-page paper and seven-page supplement |
| Historical revision execution controls | **748 checks passed**, retained from the measured revision |
| Historical manuscript verification with build logs | **1,436 checks passed**, retained separately |
| Additional query analysis | 192 method/query cells, 24 canonical maps, all 27 WDC split cells |

The source-only checkout has no model files, so the saved-score verifier omits 39 model-checksum checks there. After restoring the companion models, all 39 checks run. The current paper check count omits two build-log checks when local TeX logs are absent; this differs from the historical 1,436-check run without discarding its evidence.

One test originally depended on external WWI files. The published test now constructs a small fixture and checks the actual WWI adapter's purchase-outer versus stock-unit conversion, unchanged native quantities, signed movements, and absent financial quantities. It exercises the adapter without external datasets. The original scientific source modules and saved experimental outputs are unchanged.

The companion ZIP passes CRC and per-member SHA-256 checks. Its asset index records exact file paths, sizes, and hashes. Privacy derivatives and source licenses are preserved from the previously verified distribution. Credentials, runtime environments, temporary files, original sensitive DataCo inputs, and raw WDC offer text are excluded.

Publication files retain their existing declarations and provenance. The supplied paper PDF matches the reviewed revision byte for byte.

These checks do not constitute new model training, full fresh-environment experimental replication, or a human study. Hosted GitHub CI has not run until this repository is published and its workflow completes.

Machine-readable reports are under `verification/`. Detailed source/model/query evidence for the original study remains under `results/all-datasets-v1/`; the additional paired analyses remain under `results/manuscript-revision-v1/`.
