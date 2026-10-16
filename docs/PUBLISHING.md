# Publish this repository

Suggested repository name: `traceable-entity-reconciliation`. Visibility: **public**.

Create an empty repository under the intended GitHub account, without generating a second README or license. From this extracted directory:

```bash
git init -b main
git add .
git commit -m "Publish traceable entity reconciliation research artifact"
git remote add origin https://github.com/YOUR-ACCOUNT/traceable-entity-reconciliation.git
git push -u origin main
```

No Git LFS is needed. Create a release tagged `v1.1.0` at the initial commit and attach `kg-erp-research-materials-v1.1.0.zip` and its SHA-256 file. The source-only repository ZIP and paper-source ZIP may be attached as convenience downloads. Keep the material archive outside Git history.

Once the address is live, replace the placeholder in `paper/revision-2026/availability.tex` and use the real URL in the paper. CI is provided for Linux and Windows; verify its actual status in the Actions tab after pushing.

The source package excludes credentials, runtime environments, caches, original sensitive DataCo inputs, and raw WDC offer content. Preserve LICENSE, THIRD_PARTY_NOTICES.md, dataset provenance, and publication declarations when publishing.
