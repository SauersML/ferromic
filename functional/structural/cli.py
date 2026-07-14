"""Command-line entry point for the structure-vs-SNV decomposition.

Reproducible-from-cache (no API, no large inputs):

    # Recompute the master table, QC summary, and de-biased summary from the committed
    # per-window decomposition JSONs (functional/results/structural/*.json)
    python -m functional.structural.cli summarize

Upstream build/score stages (need ALPHAGENOME_API_KEY + network + the 2bit reference and
1000G panel resolved through functional.paths):

    python -m functional.structural.cli score-consensus --loci-file <loci.json> --out <ag_decomp.json> --arraydir <dir>
    python -m functional.structural.cli score-perhap    --out <perhap_debiased.json>   # de-biased headline
    python -m functional.structural.cli bg-stats        # background genetics (panel API)
    python -m functional.structural.cli anchor          # breakpoint-in-gene anchor (needs GENCODE)
    python -m functional.structural.cli recurrence      # recurrence contrasts (from results JSONs)
    python -m functional.structural.cli integrate       # per-locus verdicts (from results JSONs)

``score-consensus`` / ``score-perhap`` build the four counterfactual sequences per breakpoint
window (see the module README) and score SPLICE_SITE_USAGE with AlphaGenome.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_SCRIPT_STAGES = {  # module-level scripts, invoked as subprocesses so import stays side-effect-free
    "bg-stats": "functional.structural.bg_stats",
    "anchor": "functional.structural.anchor",
    "recurrence": "functional.structural.recurrence",
    "integrate": "functional.structural.integrate",
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]

    if cmd == "summarize":
        from . import summarize
        return summarize.main(rest)
    if cmd == "score-consensus":
        from . import score_consensus
        return score_consensus.main(rest)
    if cmd == "score-perhap":
        from . import score_perhap
        return score_perhap.main(rest)
    if cmd in _SCRIPT_STAGES:
        # optional --results-dir <dir> is passed to the script via STRUCTURAL_RESULTS_DIR
        ap = argparse.ArgumentParser(prog=f"structural {cmd}")
        ap.add_argument("--results-dir", default=None)
        a, extra = ap.parse_known_args(rest)
        env = dict(os.environ)
        if a.results_dir:
            env["STRUCTURAL_RESULTS_DIR"] = os.path.abspath(a.results_dir)
        return subprocess.call([sys.executable, "-m", _SCRIPT_STAGES[cmd], *extra], env=env)

    print(f"unknown command: {cmd}\n{__doc__}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
