#!/usr/bin/env python3
"""Rebuild the profile-likelihood confidence intervals in the committed PheWAS tables.

Background
----------
The constrained logistic refit that ``phewas/models.py`` used to trace profile
likelihoods ignored its offset in the Newton step (fixed in c364c189). Every
interval labelled ``profile`` in the exported All of Us tables therefore stopped
short of the true profile bound on the side away from OR = 1, and in the harder
cases the search failed altogether, leaving a one-sided interval or none at all.
The likelihood-ratio test never used that refit, so ``Beta`` and
``P_LRT_Overall`` are exact.

Those two quantities pin the profile deviance at the estimate (zero) and at
OR = 1 (the LRT statistic). Under a quadratic profile log-likelihood, which holds
at the case counts in these tables (bounds within 0.002 log-OR of the true
profile bounds, identical coverage and significance calls in simulation), the
95% interval is

    SE = |ln OR| / z_p,   z_p = Phi^-1(1 - p_LRT / 2)
    CI = exp(ln OR -/+ 1.959964 * SE)

This is the construction the pipeline already uses when an interval is missing
(``_compute_overall_or_ci`` in ``phewas/run.py``).

Which rows are rewritten
------------------------
A row is rebuilt when its p-value comes from the likelihood-ratio test
(``lrt_mle``), its estimate is finite, and its interval is one of

* ``profile`` and two-sided (the collapsed intervals),
* ``profile`` but flagged invalid or one-sided (the failed searches),
* absent (no method recorded although the model converged).

Rows whose interval came from another method (``wald_mle``,
``profile_penalized``, score bootstraps) are left untouched; they never went
through the defective refit.

The tables are edited as text so every untouched cell stays byte-identical, and
the operation is idempotent: rebuilt rows carry ``CI_Method = lrt_quadratic`` and
are skipped on later runs. ``--check`` fails when an eligible row is still
pending, when a rebuilt interval no longer matches its Beta and p-value, or when
the combined within-ancestry table disagrees with its six sources.

Usage
-----
    python stats/rebuild_phewas_profile_cis.py            # rewrite the tables in place
    python stats/rebuild_phewas_profile_cis.py --check    # verify, exit 1 on any pending change
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

Z_CRIT = NormalDist().inv_cdf(0.975)
REBUILT_METHOD = "lrt_quadratic"
SOURCE_METHOD = "profile"
LRT_SOURCE = "lrt_mle"
ANCESTRIES = ("EUR", "AFR", "AMR", "EAS", "SAS", "MID")
POPULATION_FILES = {
    anc: Path("data") / "phewas_within_ancestry" / f"phewas_{anc.lower()}_within_ancestry_pcs.tsv"
    for anc in ANCESTRIES
}
MAIN_TABLE = Path("data") / "phewas_results.tsv"
TAG_TABLE = Path("data") / "all_pop_phewas_tag.tsv"
COMBINED_TABLE = Path("data") / "within_ancestry_pc_phewas_results.tsv"


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
def rebuilt_bounds(beta: float, p_value: float) -> Tuple[float, float]:
    """95% interval on the OR scale from the log-odds estimate and its LRT p-value."""
    if not (math.isfinite(beta) and math.isfinite(p_value)):
        raise ValueError("beta and p must be finite")
    if not (0.0 < p_value < 1.0):
        raise ValueError("p must lie strictly inside (0, 1)")
    if beta == 0.0:
        raise ValueError("beta is exactly zero; the LRT-implied standard error is undefined")
    z = NormalDist().inv_cdf(1.0 - p_value / 2.0)
    if not (math.isfinite(z) and z > 0.0):
        raise ValueError("LRT quantile is not positive")
    se = abs(beta) / z
    return math.exp(beta - Z_CRIT * se), math.exp(beta + Z_CRIT * se)


def _fmt_num(x: float) -> str:
    """Mirror phewas.models._fmt_num so OR_CI95 strings keep the pipeline format."""
    if not math.isfinite(x):
        if math.isnan(x):
            return "NA"
        return "+inf" if x > 0 else "-inf"
    ax = abs(float(x))
    if ax != 0 and (ax < 1e-3 or ax > 1e3):
        return f"{x:.3e}"
    return f"{x:.3f}"


def fmt_ci(lo: float, hi: float) -> str:
    return f"{_fmt_num(lo)},{_fmt_num(hi)}"


def _to_float(text: str) -> float:
    text = text.strip()
    if text == "" or text.lower() in {"na", "nan", "none", "null"}:
        return math.nan
    return float(text)


def _is_true(text: str) -> bool:
    return text.strip().lower() in {"true", "1", "yes"}


# ---------------------------------------------------------------------------
# Text-preserving TSV handling
# ---------------------------------------------------------------------------
@dataclass
class Table:
    path: Path
    header: List[str]
    rows: List[List[str]]
    newline: str
    trailing_newline: bool

    @classmethod
    def read(cls, path: Path) -> "Table":
        raw = path.read_bytes().decode("utf-8")
        first_line = raw.split("\n", 1)[0]
        newline = "\r\n" if first_line.endswith("\r") else "\n"
        trailing = raw.endswith(newline)
        body = raw[: -len(newline)] if trailing else raw
        lines = body.split(newline) if body else []
        if not lines:
            raise ValueError(f"{path} is empty")
        header = lines[0].split("\t")
        rows = [line.split("\t") for line in lines[1:]]
        width = len(header)
        for i, row in enumerate(rows, start=2):
            if len(row) != width:
                raise ValueError(f"{path}: line {i} has {len(row)} fields, header has {width}")
        return cls(path, header, rows, newline, trailing)

    def serialise(self) -> bytes:
        lines = ["\t".join(self.header)] + ["\t".join(r) for r in self.rows]
        text = self.newline.join(lines)
        if self.trailing_newline:
            text += self.newline
        return text.encode("utf-8")

    def col(self, name: str) -> int:
        try:
            return self.header.index(name)
        except ValueError as exc:
            raise KeyError(f"{self.path}: missing column {name!r}") from exc

    def has(self, name: str) -> bool:
        return name in self.header


# ---------------------------------------------------------------------------
# Blocks: one set of CI columns driven by one (beta, p) pair
# ---------------------------------------------------------------------------
@dataclass
class Block:
    label: str
    method_col: str
    lo_col: str
    hi_col: str
    ci95_col: Optional[str]
    valid_col: Optional[str]
    sided_col: Optional[str]
    label_col: Optional[str]
    beta_col: Optional[str]
    or_col: Optional[str]
    p_col: str
    source_cols: Sequence[str]
    mirror_cols: Dict[str, str] = field(default_factory=dict)  # source -> DISPLAY copy


def overall_block(table: Table, p_col: str, source_cols: Sequence[str]) -> Block:
    mirrors = {}
    for src, dst in (
        ("CI_Method", "CI_Method_DISPLAY"),
        ("CI_LO_OR", "CI_LO_OR_DISPLAY"),
        ("CI_HI_OR", "CI_HI_OR_DISPLAY"),
        ("OR_CI95", "OR_CI95_DISPLAY"),
        ("CI_Valid", "CI_Valid_DISPLAY"),
        ("CI_Label", "CI_Label_DISPLAY"),
    ):
        if table.has(src) and table.has(dst):
            mirrors[src] = dst
    return Block(
        label="overall",
        method_col="CI_Method",
        lo_col="CI_LO_OR",
        hi_col="CI_HI_OR",
        ci95_col="OR_CI95" if table.has("OR_CI95") else None,
        valid_col="CI_Valid" if table.has("CI_Valid") else None,
        sided_col="CI_Sided" if table.has("CI_Sided") else None,
        label_col="CI_Label" if table.has("CI_Label") else None,
        beta_col="Beta" if table.has("Beta") else None,
        or_col="OR" if table.has("OR") else None,
        p_col=p_col,
        source_cols=[c for c in source_cols if table.has(c)],
        mirror_cols=mirrors,
    )


def ancestry_block(table: Table, anc: str) -> Block:
    return Block(
        label=anc,
        method_col=f"{anc}_CI_Method",
        lo_col=f"{anc}_CI_LO_OR",
        hi_col=f"{anc}_CI_HI_OR",
        ci95_col=f"{anc}_CI95" if table.has(f"{anc}_CI95") else None,
        valid_col=f"{anc}_CI_Valid" if table.has(f"{anc}_CI_Valid") else None,
        sided_col=f"{anc}_CI_Sided" if table.has(f"{anc}_CI_Sided") else None,
        label_col=f"{anc}_CI_Label" if table.has(f"{anc}_CI_Label") else None,
        beta_col=None,
        or_col=f"{anc}_OR",
        p_col=f"{anc}_P",
        source_cols=[f"{anc}_P_Source"] if table.has(f"{anc}_P_Source") else [],
    )


@dataclass
class BlockReport:
    table: str
    block: str
    rebuilt: int = 0
    already: int = 0
    drift: int = 0
    categories: Dict[str, int] = field(default_factory=dict)  # what the rebuilt rows replaced
    skipped: Dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def count(self, category: str) -> None:
        self.categories[category] = self.categories.get(category, 0) + 1


def _row_inputs(table: Table, block: Block, row: List[str]) -> Tuple[float, float]:
    if block.beta_col is not None:
        beta = _to_float(row[table.col(block.beta_col)])
    else:
        or_val = _to_float(row[table.col(block.or_col)])
        beta = math.log(or_val) if math.isfinite(or_val) and or_val > 0 else math.nan
    p_value = _to_float(row[table.col(block.p_col)])
    return beta, p_value


def _lrt_backed(table: Table, block: Block, row: List[str]) -> bool:
    if not block.source_cols:
        return False
    return any(row[table.col(c)].strip() == LRT_SOURCE for c in block.source_cols)


def process_block(table: Table, block: Block, *, write: bool, tol: float = 1e-9) -> BlockReport:
    report = BlockReport(str(table.path), block.label)
    m = table.col(block.method_col)
    lo_ix = table.col(block.lo_col)
    hi_ix = table.col(block.hi_col)
    ci95_ix = table.col(block.ci95_col) if block.ci95_col else None
    valid_ix = table.col(block.valid_col) if block.valid_col else None
    sided_ix = table.col(block.sided_col) if block.sided_col else None
    label_ix = table.col(block.label_col) if block.label_col else None
    mirror_ix = {table.col(src): table.col(dst) for src, dst in block.mirror_cols.items()}

    notes_ix = table.col("Model_Notes") if (block.label == "overall" and table.has("Model_Notes")) else None

    for row in table.rows:
        method = row[m].strip()
        if method == REBUILT_METHOD:
            beta, p_value = _row_inputs(table, block, row)
            try:
                lo, hi = rebuilt_bounds(beta, p_value)
            except ValueError:
                report.drift += 1
                continue
            stored_lo, stored_hi = _to_float(row[lo_ix]), _to_float(row[hi_ix])
            if abs(stored_lo - lo) > tol * max(1.0, lo) or abs(stored_hi - hi) > tol * max(1.0, hi):
                report.drift += 1
            elif notes_ix is not None and _stale_note(row[notes_ix]):
                # The pipeline's diagnostic note still names the old method.
                if write:
                    row[notes_ix] = _update_note(row[notes_ix])
                report.rebuilt += 1
                report.count("diagnostic note updated")
            else:
                report.already += 1
            continue
        if method not in (SOURCE_METHOD, ""):
            continue  # wald_mle, profile_penalized, score bootstraps: never used the refit
        if not _lrt_backed(table, block, row):
            if method == SOURCE_METHOD:
                report.skip("profile interval without an LRT p-value (left as is)")
            continue
        beta, p_value = _row_inputs(table, block, row)
        if method == "" and not math.isfinite(beta):
            continue  # no converged model in this cell
        try:
            lo, hi = rebuilt_bounds(beta, p_value)
        except ValueError as exc:
            report.skip(f"cannot rebuild: {exc}")
            continue
        valid = _is_true(row[valid_ix]) if valid_ix is not None else True
        sided = row[sided_ix].strip() if sided_ix is not None else "two"
        if method == "":
            category = "no interval recorded"
        elif not valid:
            category = "profile search failed (flagged invalid)"
        elif sided == "one":
            category = "one-sided profile interval"
        else:
            category = "two-sided profile interval"
        if write:
            row[m] = REBUILT_METHOD
            row[lo_ix] = repr(lo)
            row[hi_ix] = repr(hi)
            if ci95_ix is not None:
                row[ci95_ix] = fmt_ci(lo, hi)
            if valid_ix is not None:
                row[valid_ix] = "True"
            if sided_ix is not None:
                row[sided_ix] = "two"
            if label_ix is not None:
                row[label_ix] = ""
            if notes_ix is not None:
                row[notes_ix] = _update_note(row[notes_ix])
            for src_ix, dst_ix in mirror_ix.items():
                row[dst_ix] = row[src_ix]
        report.rebuilt += 1
        report.count(category)
    return report


STALE_NOTE_TOKEN = "ci=" + SOURCE_METHOD
NEW_NOTE_TOKEN = "ci=" + REBUILT_METHOD


def _stale_note(note: str) -> bool:
    return STALE_NOTE_TOKEN in note.split(";")


def _update_note(note: str) -> str:
    """Rename the pipeline's ci=profile diagnostic token on a rebuilt row."""
    return ";".join(NEW_NOTE_TOKEN if tok == STALE_NOTE_TOKEN else tok for tok in note.split(";"))


# ---------------------------------------------------------------------------
# Combined within-ancestry table (derived from the six per-ancestry tables)
# ---------------------------------------------------------------------------
COMBINED_SYNC_COLUMNS = ("CI_Valid", "CI_Sided", "CI_LO_OR", "CI_HI_OR")


def sync_combined(root: Path, *, write: bool) -> Tuple[BlockReport, List[str]]:
    """Copy the (possibly rebuilt) interval columns from the six per-ancestry tables
    into the combined table, matching rows on (population, Phenotype, Inversion).
    Returns the report and a list of inconsistencies (rows whose values match
    neither the source table as committed nor its rebuilt values)."""
    combined = Table.read(root / COMBINED_TABLE)
    report = BlockReport(str(COMBINED_TABLE), "combined")
    problems: List[str] = []
    key_ix = (combined.col("population"), combined.col("Phenotype"), combined.col("Inversion"))
    sync_ix = [combined.col(c) for c in COMBINED_SYNC_COLUMNS]

    sources: Dict[str, Dict[Tuple[str, str], Tuple[List[str], str]]] = {}
    for anc, rel in POPULATION_FILES.items():
        src = Table.read(root / rel)
        p_ix, i_ix, m_ix = src.col("Phenotype"), src.col("Inversion"), src.col("CI_Method")
        s_ix = [src.col(c) for c in COMBINED_SYNC_COLUMNS]
        sources[anc] = {(r[p_ix], r[i_ix]): ([r[i] for i in s_ix], r[m_ix]) for r in src.rows}

    for row in combined.rows:
        anc = row[key_ix[0]].strip().upper()
        key = (row[key_ix[1]], row[key_ix[2]])
        src = sources.get(anc, {}).get(key)
        if src is None:
            problems.append(f"{anc} {key[0]} {key[1]}: no matching per-ancestry row")
            continue
        src_values, src_method = src
        same = all(_same_cell(row[i], v) for i, v in zip(sync_ix, src_values))
        if same:
            if src_method == REBUILT_METHOD:
                report.already += 1
            continue
        if src_method != REBUILT_METHOD:
            problems.append(f"{anc} {key[0]} {key[1]}: interval columns differ from the per-ancestry table")
            continue
        if write:
            for i, v in zip(sync_ix, src_values):
                row[i] = v
        report.rebuilt += 1

    if write and report.rebuilt:
        (root / COMBINED_TABLE).write_bytes(combined.serialise())
    return report, problems


def _same_cell(a: str, b: str) -> bool:
    if a.strip() == b.strip():
        return True
    try:
        fa, fb = _to_float(a), _to_float(b)
    except ValueError:
        return False
    if math.isnan(fa) and math.isnan(fb):
        return True
    if math.isinf(fa) or math.isinf(fb):
        return fa == fb
    return math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) <= 1e-12 * max(1.0, abs(fa))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def table_plan(root: Path) -> List[Tuple[Path, List[str], List[str], List[str]]]:
    """(path, p columns, source columns, ancestry blocks) for every affected table."""
    plan: List[Tuple[Path, List[str], List[str], List[str]]] = []
    plan.append((MAIN_TABLE, ["P_LRT_Overall"], ["P_Source_x", "P_Source"], list(ANCESTRIES)))
    plan.append((TAG_TABLE, ["P_LRT_Overall"], ["P_Source_x", "P_Source"], []))
    for rel in POPULATION_FILES.values():
        plan.append((rel, ["P_LRT_Overall"], ["P_Source_lrt", "P_Source"], []))
    return [(root / rel, p, s, a) for rel, p, s, a in plan]


def run(
    root: Path,
    *,
    write: bool,
    verbose: bool = True,
    only: Optional[set] = None,
) -> Tuple[List[BlockReport], List[str]]:
    """Rebuild (write=True) or audit (write=False) every table in the plan.

    ``only`` restricts the run to a set of repository-relative paths (used by the
    tests); the combined within-ancestry table is synchronised only when it is in
    scope."""
    reports: List[BlockReport] = []
    problems: List[str] = []
    for path, p_cols, source_cols, ancestries in table_plan(root):
        rel = path.relative_to(root)
        if only is not None and rel not in only:
            continue
        if not path.exists():
            problems.append(f"missing table: {path}")
            continue
        try:
            table = Table.read(path)
        except (ValueError, UnicodeDecodeError) as exc:
            problems.append(f"{path}: {exc}; refusing to edit")
            continue
        before = table.serialise()
        if before != path.read_bytes():
            problems.append(f"{path}: cannot round-trip the file byte-for-byte; refusing to edit")
            continue
        p_col = next((c for c in p_cols if table.has(c)), None)
        if p_col is None:
            problems.append(f"{path}: none of the p-value columns {p_cols} present")
            continue
        blocks = [overall_block(table, p_col, source_cols)]
        blocks += [ancestry_block(table, anc) for anc in ancestries if table.has(f"{anc}_CI_Method")]
        changed = False
        for block in blocks:
            rep = process_block(table, block, write=write)
            reports.append(rep)
            changed = changed or (write and rep.rebuilt > 0)
        if changed:
            path.write_bytes(table.serialise())
    if (only is None or COMBINED_TABLE in only) and (root / COMBINED_TABLE).exists():
        rep, probs = sync_combined(root, write=write)
        reports.append(rep)
        problems.extend(probs)
    if verbose:
        for rep in reports:
            cats = "; ".join(f"{v} {k}" for k, v in rep.categories.items())
            skipped = "; ".join(f"{v} {k}" for k, v in rep.skipped.items()) or "none"
            print(
                f"{rep.table} [{rep.block}]: rebuilt={rep.rebuilt}"
                + (f" ({cats})" if cats else "")
                + f" already={rep.already} drift={rep.drift} skipped={skipped}"
            )
        for prob in problems:
            print(f"PROBLEM: {prob}")
    return reports, problems


def pending(reports: Sequence[BlockReport], problems: Sequence[str]) -> List[str]:
    issues = list(problems)
    for rep in reports:
        if rep.rebuilt:
            issues.append(f"{rep.table} [{rep.block}]: {rep.rebuilt} interval(s) still need rebuilding")
        if rep.drift:
            issues.append(f"{rep.table} [{rep.block}]: {rep.drift} rebuilt interval(s) no longer match Beta and p")
    return issues


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root (default: this checkout)")
    parser.add_argument("--check", action="store_true", help="verify only; exit 1 if any rebuild is pending")
    args = parser.parse_args(argv)

    reports, problems = run(args.root, write=not args.check)
    if args.check:
        issues = pending(reports, problems)
        if issues:
            print("CHECK FAILED:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        print("CHECK OK: no LRT-backed profile intervals remain and all rebuilt intervals match their inputs.")
        return 0
    if problems:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
