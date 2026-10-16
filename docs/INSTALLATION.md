# Installation and configuration

Python 3.12 is the tested runtime. Install from the repository root because scripts and default configurations resolve against that root. For the recorded environment, install requirements.fullstudy.lock then `pip install --no-deps -e .`. Development tests need pytest. Expanded embedding work needs ONNX Runtime and tokenizers, declared by the fullstudy extra and lock. Manuscript generation has a separate requirements.paper.lock.

`python scripts/materialize_release_data.py` restores compressed assets and verifies decoded hashes. It does not automatically promote DataCo privacy derivatives into original matching input paths. This protects the distinction between published data and the historical source. Do not use a changed privacy copy to claim that the original name-based models reproduced.

The main package uses src layout. Entrypoints are `kgerp` / `python -m kgerp.cli` for core commands, `python -m kgerp.research_plus` for improvements, and `python -m kgerp.expanded` for the expanded study. Run each with --help for arguments. Recorded defaults live in configs/.

Core profiling and matching from staged WWI do not require Docker. To extract the complete sample again, materialize the backup, install Docker with Linux containers, run scripts/restore_wwi.py, then `python -m kgerp.cli extract`. The helper generates a local strong SQL credential in ignored .secrets/sql.env, uses a named isolated container/volume and publishes no host port. Existing research resources are retained. Nothing secret is distributed.

Original DataCo acquisition is a documented mirror fallback after publisher HTTP 403. BPI uses a publisher-linked XES and publisher MD5. WDC raw files are acquired from official download URLs; review their source terms separately. Original acquired files and databases remain ignored by Git.

For manuscripts, install a LaTeX distribution with IEEEtran and bibtex and ensure pdflatex/bibtex are on PATH; Poppler is needed for rendering checks. `python paper/revision-2026/build.py` builds from bundled generated tables/figures without refitting experiments. Dependencies are downloaded through pip, not bundled virtual environments.
