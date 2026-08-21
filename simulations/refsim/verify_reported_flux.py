#!/usr/bin/env python3
"""Fail unless a complete flux sweep reproduces the reviewer-response numbers."""

from __future__ import annotations

import argparse
import math

import make_report


EXPECTED_COUNTS = {
    "single_repo": {0.0: 37, 1e-8: 49, 1e-7: 45, 1e-6: 66},
    "recurrent": {0.0: 1099, 1e-8: 1133, 1e-7: 1139, 1e-6: 1121},
}
EXPECTED_TREND_P = {"single_repo": 0.00712342373249036, "recurrent": 0.305858351751984}


def close(observed: float, expected: float, tolerance: float = 5e-10) -> None:
    if not math.isfinite(observed) or abs(observed - expected) > tolerance:
        raise SystemExit(f"observed {observed:.15g}; expected {expected:.15g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", required=True)
    args = parser.parse_args()
    rows = make_report.load(args.rows)
    if len(rows) != 11_520:
        raise SystemExit(f"expected 11,520 successful loci; observed {len(rows):,}")

    for scenario, expected_by_flux in EXPECTED_COUNTS.items():
        for flux, expected_calls in expected_by_flux.items():
            selected = [
                row for row in rows
                if row["scenario"] == scenario and row["m_flux"] == flux
            ]
            if len(selected) != 1_440:
                raise SystemExit(
                    f"{scenario}, flux={flux}: expected 1,440 loci; "
                    f"observed {len(selected):,}"
                )
            calls = sum(row["call"] for row in selected)
            if calls != expected_calls:
                raise SystemExit(
                    f"{scenario}, flux={flux}: expected {expected_calls} "
                    f"recurrent calls; observed {calls}"
                )

        trend = make_report.armitage_trend(rows, scenario)
        close(trend["p"], EXPECTED_TREND_P[scenario])

    highest = [
        row for row in rows
        if row["scenario"] == "single_repo" and row["m_flux"] == 1e-6
    ]
    calls = sum(row["call"] for row in highest)
    lo, hi = make_report.wilson_ci(calls, len(highest))
    close(calls / len(highest), 0.0458333333333333)
    close(lo, 0.0361872905573096)
    close(hi, 0.0578961601480864)
    print(
        "Reproduced the reported gene-flux result: FPR 4.6% "
        f"(95% Wilson CI {lo:.3f}-{hi:.3f}), trend p={EXPECTED_TREND_P['single_repo']:.4f}; "
        f"power trend p={EXPECTED_TREND_P['recurrent']:.4f}."
    )


if __name__ == "__main__":
    main()
