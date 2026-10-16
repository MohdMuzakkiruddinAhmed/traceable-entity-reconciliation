# Five-dataset manuscript sources

Editable detailed and IEEE LaTeX sources share references.bib and generated facts/tables. The ReportLab supplement generator is scripts/build_fullstudy_paper.py. Figures are supplied as vector PDF and high-resolution PNG. Generated tables and the claim registry preserve original measurement hashes and aggregation rules.

Install requirements.paper.lock in a separate environment and make pdflatex/bibtex with IEEEtran available. From the project root run `python scripts/build_fullstudy_paper.py`. It builds the three PDFs using included generated artifacts and does not refit experiments. Full regeneration/source checks need the original research inputs, including raw WDC and pre-sanitization DataCo/Olist assets not rehosted here.

Manuscripts are dated anonymous drafts and retain applicable disclosure/provenance information. The software release's MIT license and third-party notices govern this package; historical manuscript text reflects its drafting date. Review human author details and declarations before any actual paper submission. Bibliography source verification is in publication/bibliography_provenance.json; numerical/render checks are in verification/.
