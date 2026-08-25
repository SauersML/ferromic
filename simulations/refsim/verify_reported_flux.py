#!/usr/bin/env python3
"""Verify the complete reported gene-flux grid and its sampling semantics."""

from __future__ import annotations

import argparse
import random

import make_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", required=True)
    args = parser.parse_args()
    rows = make_report.load(args.rows)
    if len(rows) != 11_520:
        raise SystemExit(f"expected 11,520 successful loci; observed {len(rows):,}")

    if {row["scenario"] for row in rows} != {"single", "recurrent"}:
        raise SystemExit("the production grid must contain only single and recurrent arms")

    fluxes = [0.0, 1e-8, 1e-7, 1e-6]
    for scenario in ("single", "recurrent"):
        for flux in fluxes:
            selected = [
                row for row in rows
                if row["scenario"] == scenario and row["m_flux"] == flux
            ]
            if len(selected) != 1_440:
                raise SystemExit(
                    f"{scenario}, flux={flux}: expected 1,440 loci; "
                    f"observed {len(selected):,}"
                )

    single = [row for row in rows if row["scenario"] == "single"]
    if any(row["frac_admix_i"] != 1.0 or row["frac_admix_d"] != 0.0
           for row in single):
        raise SystemExit("single-event rows do not use the archived two-population model")

    recurrent = [row for row in rows if row["scenario"] == "recurrent"]
    for row in recurrent:
        rng = random.Random(row["seed"])
        expected_i = rng.randint(0, 10) / 10
        expected_d = rng.randint(0, 10) / 10
        if (row["frac_admix_i"], row["frac_admix_d"]) != (expected_i, expected_d):
            raise SystemExit(
                f"seed {row['seed']}: recurrent sampling differs from the public "
                "generator's two independent randint(0,10)/10 draws"
            )

    messages = []
    for scenario, label in (("single", "false-positive"),
                            ("recurrent", "power")):
        trend = make_report.armitage_trend(rows, scenario)
        counts = [sum(row["call"] for row in rows
                      if row["scenario"] == scenario and row["m_flux"] == flux)
                  for flux in fluxes]
        messages.append(f"{label} calls={counts}, trend p={trend['p']:.12g}")
    print("Verified complete production gene-flux grid; " + "; ".join(messages))


if __name__ == "__main__":
    main()
