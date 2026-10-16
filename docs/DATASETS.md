# Datasets, scope and redistribution

All available scientific measurements are preserved. The published data tree distinguishes lossless source assets, privacy derivatives and separately acquired offer text. `data/distribution/manifest.json` lists every compressed file, original SHA-256, published/decoded SHA-256, transformations and destination. No original measurement is recomputed or silently relabelled using sanitized data.

| Dataset | Primary source | License / release treatment |
| --- | --- | --- |
| Microsoft Wide World Importers Standard | https://github.com/microsoft/sql-server-samples | Microsoft MIT notice retained; fictional backup and six staged tables bundled losslessly |
| Olist version 2 | https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce | CC BY-NC-SA 4.0; all nine tables/rows retained, review title/message text blanked; same license retained for adaptations |
| DataCo version 5 | https://data.mendeley.com/datasets/8gx2fvg2k6/5 | CC BY 4.0; creators Fabian Constante, Fernando Silva, Antonio Pereira; original mirror provenance retained; all 180,519 rows/53 columns supplied with email/password/name/street/zip/coordinate values blanked |
| BPI Challenge 2019 | https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1 | CC BY 4.0; creator Boudewijn van Dongen; publisher anonymized XES losslessly compressed and MD5 provenance retained |
| WDC Products | https://webdatacommons.org/largescaleproductcorpus/wdc-products/index.html | Official repository code is BSD-3-Clause; raw offer-data rehosting license not established from checked sources. Offer text is fetched separately. All 27 official evaluation cells, opaque pair IDs/labels/scores and fitted evaluation parameters are included. |
| all-MiniLM-L6-v2 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Apache 2.0 model card; pinned revision/quantized weights/tokenizer/config retained |

DataCo original email/password values and directly identifying name/address fields are not in the release. Its privacy source views also blank names. This derivative is useful for schema/history inspection, but does not reproduce the original learned matching scores. To rerun that specific task, reacquire original inputs in a local private workspace and follow the recorded hashes/protocol. Olist review prose is unused by the matching adapter; sanitization preserves counts and matching histories but changes raw ZIP bytes.

Full native graph size differs from matching scope: WWI 249,786 nodes/274,856 edges versus 240 masters; Olist 1,569,937/1,775,243 versus 2,913 repeat customers; DataCo 267,103/607,478 versus a 1,500-customer sample; BPI 1,926,659/7,138,894 versus 1,378 vendors. WDC used pairwise files have 10,422 offers/1,600 groups, distinct from publisher whole-corpus counts.

Source SQLite databases and their WAL/SHM files are regenerable imports and are excluded. DataCo's unredacted database is additionally sensitive. Embedding cache hashes/vectors are regenerable and excluded. Evaluator maps use source/pseudonymous identifiers for non-leaking evaluation and are kept separate from inference.

The CC licenses require source attribution and disclosure of modifications. Olist adaptations retain noncommercial and ShareAlike conditions. The project MIT license does not supersede those conditions or imply a license for WDC offer content. See licenses/ and THIRD_PARTY_NOTICES.md.
