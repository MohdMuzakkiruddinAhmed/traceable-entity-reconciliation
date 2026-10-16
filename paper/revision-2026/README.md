# Current paper and technical supplement

`main.pdf` is the verified 10-page manuscript; `supplement.pdf` is the seven-page revision supplement. `main.tex`, `supplement.tex`, `references.bib`, and `figures/` provide editable publication sources. The publication text and its required declarations are preserved from the supplied paper.

The shared repository-link placeholder is in `availability.tex`. Replace its command with `\newcommand{\repositorylink}{\url{https://github.com/OWNER/REPOSITORY}}` when the public address is known.

Upload this directory to Overleaf and select `main.tex` or `supplement.tex`, using pdfLaTeX. No Python or shell escape is needed with the included vector figure PDFs. Locally, install TeX Live or MiKTeX with IEEEtran, newtx, amsmath, cite, graphicx, booktabs, tabularx, microtype, hyperref, flushend, geometry, longtable, standalone, TikZ, and PGFPlots, then run `python build.py`. Use `--figures` to regenerate vector figures.

The build helper fails on compilation errors, undefined references/citations, and overfull boxes. It does not execute research experiments. `analysis/verify_manuscript.py` independently recomputes the included result summaries and checks the PDF page counts and text; install `analysis/requirements.txt` first. `analysis/run_controls.py --project /path/to/kg-erp --output /path/to/new-empty-results` reexecutes paired queries and WDC audits in the complete research workspace.

`data/` retains measured outputs, exact denominators, original archive/member hashes, and verification evidence. The original 18-page study supplement is preserved under `supporting/` for earlier E1--E5 diagnostics. It retains its historical title and labeling.
