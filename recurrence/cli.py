"""Unified stage entrypoint for the recurrence pipeline.

    python -m recurrence.cli simulate [--reps N --procs P --out PATH]   # msprime training set
    python -m recurrence.cli fit      [--sims PATH --outdir DIR]        # fit + validate on sims
    python -m recurrence.cli score    [--model ... --outdir DIR]        # apply to real inversions

``fit`` and ``score`` default to the committed inputs/results and run without msprime;
``simulate`` regenerates the training set and needs the ``recurrence`` extra.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(prog="recurrence", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("simulate", add_help=False)
    sub.add_parser("fit", add_help=False)
    sub.add_parser("score", add_help=False)
    args, rest = ap.parse_known_args(argv)

    if args.stage == "simulate":
        from . import simulate
        simulate.main(rest)
    elif args.stage == "fit":
        from . import fit
        fit.main(rest)
    elif args.stage == "score":
        from . import apply as apply_stage
        apply_stage.main(rest)


if __name__ == "__main__":
    main()
