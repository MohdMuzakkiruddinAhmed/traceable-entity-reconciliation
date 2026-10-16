#!/usr/bin/env python3
"""Build the IEEE manuscript; optionally redraw figures from fixed reported values.

No training, benchmarking, experiment replay, source acquisition, or network access
is performed. Run from any directory: python path/to/build.py [--figures].
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIGURES = ("architecture", "decision_flow", "context_effects", "retrieval_scale")

def executable(*names: str) -> str:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(
        f"Missing required executable: {' or '.join(names)}. "
        "Install TeX Live or MiKTeX with IEEEtran, newtx, TikZ, PGFPlots, "
        "standalone, and the packages listed in main.tex."
    )

def run(command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as out:
        result = subprocess.run(command, cwd=ROOT, stdout=out,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-35:])
        raise RuntimeError(f"Build step failed: {' '.join(command)}\n{tail}\nFull log: {log}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true",
                        help="Redraw the four vector figures from the included fixed data/TikZ sources.")
    args = parser.parse_args()
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    try:
        latex = executable("pdflatex")
        bibtex = executable("bibtex", "bibtex.original", "bibtex8")
        options = ["-interaction=nonstopmode", "-halt-on-error", "-file-line-error"]
        if args.figures:
            for name in FIGURES:
                run([latex, *options, "-output-directory=build", f"figures/{name}.tex"],
                    build / f"{name}-build.txt")
                shutil.copy2(build / f"{name}.pdf", ROOT / "figures" / f"{name}.pdf")
        for name in FIGURES:
            if not (ROOT / "figures" / f"{name}.pdf").is_file():
                raise RuntimeError(f"Missing figure {name}.pdf; rerun with --figures.")
        run([latex, *options, "-output-directory=build", "main.tex"], build / "main-pass1.txt")
        run([bibtex, "build/main"], build / "bibliography-build.txt")
        # Keep the root bibliography current when TeX resolves that copy first.
        shutil.copy2(build / "main.bbl", ROOT / "main.bbl")
        run([latex, *options, "-output-directory=build", "main.tex"], build / "main-pass2.txt")
        run([latex, *options, "-output-directory=build", "main.tex"], build / "main-pass3.txt")
        log = (build / "main.log").read_text(errors="replace")
        errors = ("There were undefined references", "There were undefined citations", "Overfull ")
        remaining = [item for item in errors if item in log]
        if remaining:
            raise RuntimeError("Inspect unresolved typesetting warnings: " + ", ".join(remaining))
        shutil.copy2(build / "main.pdf", ROOT / "main.pdf")
        shutil.copy2(build / "main.bbl", ROOT / "main.bbl")
        if (ROOT / "supplement.tex").is_file():
            run([latex, *options, "-output-directory=build", "supplement.tex"], build / "supplement-pass1.txt")
            run([latex, *options, "-output-directory=build", "supplement.tex"], build / "supplement-pass2.txt")
            supplement_log = (build / "supplement.log").read_text(errors="replace")
            if any(item in supplement_log for item in errors):
                raise RuntimeError("Inspect unresolved supplement typesetting warnings")
            shutil.copy2(build / "supplement.pdf", ROOT / "supplement.pdf")
            print(f"Built {ROOT / 'supplement.pdf'}")
        print(f"Built {ROOT / 'main.pdf'}")
        print("Only manuscript assets were rebuilt. No research experiments were run.")
        return 0
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
