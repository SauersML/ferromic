#!/usr/bin/env python3
"""Audit the port against the pinned public source and continuous-flux contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

import refsim
import run_grid


HERE = Path(__file__).resolve().parent
LEDGER = HERE / "source_model_manifest.json"
HISTORICAL = {"P0_D", "P1_I", "P2_I", "P3_D", "Pa_I", "Pa_D"}


def public_source_checks(root: Path, ledger: dict) -> None:
    source = (root / ledger["generator"]).read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", source)
    required = (
        "N_a=6000",
        "frac_admixI=random.randint(0,10)/10",
        "frac_admixD=random.randint(0,10)/10",
        'ancestral=["P1_I","P2_I"]',
        'ancestral=["P0_D","P3_D"]',
    )
    missing = [token for token in required if token not in compact]
    if missing:
        raise SystemExit(f"public recurrent source changed; missing {missing}")

    manifest_path = root / ledger["manifest"]
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    observed = {
        (int(row["sampleHaploSize"]), float(row["inv_freq"]),
         float(row["rho"]), float(row["mig_const"]))
        for row in rows
    }
    expected = {
        (ledger["sample_haplotypes"], frequency, rho,
         ledger["within_orientation_migration"])
        for frequency in ledger["inversion_frequencies"]
        for rho in ledger["recombination_rates"]
    }
    if len(rows) != 18 or observed != expected:
        raise SystemExit("public model_manifest.young.tbl differs from the 18-row grid")
    if (root / "inputFiles" / "model_manifest.young.test.tbl").exists():
        raise SystemExit("unexpected test manifest in the pinned public repository")


def archived_single_checks(ledger: dict) -> None:
    path = HERE / ledger["vendored_path"]
    payload = path.read_bytes()
    observed_hash = hashlib.sha256(payload).hexdigest()
    if observed_hash != ledger["sha256"]:
        raise SystemExit(
            "vendored single-event source changed: "
            f"{observed_hash} != {ledger['sha256']}"
        )
    compact = re.sub(r"\s+", "", payload.decode("utf-8"))
    required = (
        "N_a=6000",
        "initial_size=N_a)",
        "initial_size=N_a/100)",
        "mig_mat=[[0,0],[0,0]]",
        "MassMigration(time=Tsp_p0_p1,source=1,destination=0,proportion=1.0)",
    )
    missing = [token for token in required if token not in compact]
    if missing:
        raise SystemExit(f"archived single-event source changed; missing {missing}")


def structural_checks() -> None:
    depth = refsim.TIME_DEPTHS["young"]
    no_flux = refsim.demography(
        depth["t01_23"], depth["t0_1"], depth["t2_3"], 1e-8, 0.3, 0.7, 0.0
    )
    sizes = {population.name: population.initial_size
             for population in no_flux.populations}
    expected_sizes = {
        "P_I": 600, "P_D": 6000, "P0_D": 60, "P1_I": 600,
        "P2_I": 600, "P3_D": 6000, "Pa_I": 600, "Pa_D": 6000,
        "P00": 6000,
    }
    if sizes != expected_sizes:
        raise SystemExit(f"recurrent effective sizes differ: {sizes}")

    single = refsim.demography_single(depth["t_inv"], 0.0)
    single_sizes = {population.name: population.initial_size
                    for population in single.populations}
    if single_sizes != {"P_I": 60, "P_D": 6000, "P00": 6000}:
        raise SystemExit(f"single-event effective sizes differ: {single_sizes}")


def continuous_flux_checks() -> None:
    depth = refsim.TIME_DEPTHS["young"]
    rate = 1e-7
    demography = refsim.demography(
        depth["t01_23"], depth["t0_1"], depth["t2_3"], 1e-8, 0.3, 0.7, rate
    )
    names = [population.name for population in demography.populations]
    orientation = {name: name.rsplit("_", 1)[-1]
                   for name in HISTORICAL}
    checked_active_sets = set()
    for epoch in demography.debug().epochs:
        active = {population.name for population in epoch.active_populations}
        historical = sorted(active & HISTORICAL)
        opposite = [(a, b) for a, b in combinations(historical, 2)
                    if orientation[a] != orientation[b]]
        if not opposite:
            continue
        checked_active_sets.add(tuple(historical))
        for a, b in opposite:
            observed = epoch.migration_matrix[names.index(a), names.index(b)]
            if observed != rate:
                raise SystemExit(
                    f"gene flux is absent for {a}<->{b} in epoch "
                    f"{epoch.start_time:g}-{epoch.end_time:g}: {observed}"
                )
    if len(checked_active_sets) != 3:
        raise SystemExit(
            "expected three historical population configurations; saw "
            f"{len(checked_active_sets)}"
        )

    single = refsim.demography_single(depth["t_inv"], rate)
    for epoch in single.debug().epochs:
        active = {population.name for population in epoch.active_populations}
        if {"P_I", "P_D"} <= active:
            names = [population.name for population in single.populations]
            if epoch.migration_matrix[names.index("P_I"), names.index("P_D")] != rate:
                raise SystemExit("single-event gene flux is not continuous")


def grid_checks(ledger: dict) -> None:
    rows = run_grid.build_grid("gene_flux")
    if len(rows) != 11_520:
        raise SystemExit(f"production grid has {len(rows)} rows, not 11,520")
    if {row["scenario"] for row in rows} != {"single", "recurrent"}:
        raise SystemExit("production scenarios changed")
    if sorted({row["m_flux"] for row in rows}) != ledger["gene_flux_rates"]:
        raise SystemExit("production flux ladder changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    public_source_checks(args.upstream_root, ledger["public_recurrent"])
    archived_single_checks(ledger["archived_single_event"])
    structural_checks()
    continuous_flux_checks()
    grid_checks(ledger["response_grid"])
    print("Verified public recurrent source, archived single-event port, "
          "unconstrained sampling grid, and continuous all-interval gene flux.")


if __name__ == "__main__":
    main()
