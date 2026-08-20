"""Score one locus's envelope p-values under EVERY published human demography.

Why this exists: the envelope test's null was constant-size, and it turned out
that assumption -- not the conditioning window, not the Monte Carlo error, not
multiple testing -- dominates the answer. A hand-built bottleneck moved
17q21.31's p_balance from 0.009 to 0.26, while recent-expansion histories pushed
it below 1e-4. A single hand-picked history is therefore not reportable, and
neither is "constant size". The defensible object is the RANGE over published
human histories, which is what this script computes.

Demography enters exactly, via the coalescent time change (topology is
demography-invariant, only times move: SauersML/Descent,
Descent/Coalescent/VariableSize.lean, and Griffiths & Tavare 1994 as cited in
that repo's Coalescent/Program.lean). Specs come from
stats/human_demographies.py, which converts each stdpopsim HomSap model into a
piecewise-constant relative size via the inverse instantaneous coalescence rate
for a chosen sampling configuration.

Reading the output: what matters is not any single p but whether the locus stays
extreme across histories. A locus that is significant under every published
human demography is robust to our ignorance of the true one; a locus whose
p-value straddles 0.05 across the catalogue is telling you the signal is a
statement about the assumed history, not about the locus.

Output: one row per demography, sorted by p_balance, plus the min/median/max
summary that belongs in the text instead of a single number.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inversion_envelope_1kg as kg  # noqa: E402

RNG_SEED = 909


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--demographies", required=True,
                    help="TSV from stats/human_demographies.py")
    ap.add_argument("--inv-id", default="17:45585159-46292045")
    ap.add_argument("--n-cand", type=int, default=100_000)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    loci = kg.locus_list(a.workdir)
    target = None
    for key, files in loci.items():
        if f"{key[0]}:{key[1]}-{key[2]}" == a.inv_id:
            target = (key, files)
            break
    if target is None:
        sys.exit(f"locus {a.inv_id} not found in phy_outputs")
    key, files = target

    dem = list(csv.DictReader(open(a.demographies), delimiter="\t"))
    # constant size is the incumbent, so it belongs in the table as a row
    specs = [("CONSTANT", "reference", "const")]
    specs += [(d["model"], d["config"], d["spec"]) for d in dem]
    print(f"locus {a.inv_id}: {len(specs)} histories "
          f"(1 constant + {len(dem)} published), n_cand={a.n_cand}")

    tasks = [(key, files, a.workdir, RNG_SEED + i, 0.0, a.n_cand, spec)
             for i, (_m, _c, spec) in enumerate(specs)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for (model, cfg, spec), r in zip(specs, ex.map(kg.analyze_locus, tasks)):
            if r.get("status") != "OK":
                print(f"  {model}/{cfg}: {r.get('status')}", flush=True)
                continue
            rows.append({"model": model, "config": cfg,
                         "n": r["n"], "k_inv": r["k_inv"],
                         "B": r["B"], "A": r["A"],
                         "p_balance": r["p_balance"],
                         "p_balance_mcse": r["p_balance_mcse"],
                         "p_balance_lo": r["p_balance_lo"],
                         "p_balance_hi": r["p_balance_hi"],
                         "n_tail_balance": r["n_tail_balance"],
                         "p_sweep": r["p_sweep"],
                         "null_ess": r["null_ess"],
                         "null_candidates": r["null_candidates"],
                         "spec_epochs": spec.count(";") + 1 if spec != "const" else 0,
                         "spec": spec})
            print(f"  {model:34s} {cfg:28s} p_bal={r['p_balance']:.5f} "
                  f"+-{r['p_balance_mcse']:.5f} tail={r['n_tail_balance']} "
                  f"ess={r['null_ess']:.0f}", flush=True)

    if not rows:
        sys.exit("no histories scored")
    out = pd.DataFrame(rows).sort_values("p_balance")
    out.to_csv(a.out, sep="\t", index=False)

    p = out["p_balance"].to_numpy()
    tail_empty = int((out["n_tail_balance"] == 0).sum())
    print(f"\n===== {a.inv_id}: p_balance across {len(out)} human histories =====")
    print(f"B = {out['B'].iloc[0]:.3f} (an observed statistic; identical in "
          f"every row -- only the null moves)")
    print(f"min    {p.min():.5f}   ({out.iloc[0]['model']}/"
          f"{out.iloc[0]['config']})")
    print(f"median {np.median(p):.5f}")
    print(f"max    {p.max():.5f}   ({out.iloc[-1]['model']}/"
          f"{out.iloc[-1]['config']})")
    for thr in (0.05, 0.01):
        print(f"histories with p_balance < {thr}: {(p < thr).sum()} of {len(p)}")
    print(f"histories where the tail was empty (p is a resolution floor, not an "
          f"estimate): {tail_empty}")
    cols = ["model", "config", "p_balance", "p_balance_mcse", "p_balance_lo",
            "p_balance_hi", "n_tail_balance", "null_ess", "spec_epochs"]
    print("\nfull table (sorted):")
    print(out[cols].to_string(index=False))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
