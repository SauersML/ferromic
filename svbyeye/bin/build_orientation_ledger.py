#!/usr/bin/env python3
"""Call the chimpanzee orientation of each inversion from its alignment PAF.

Biological model
================
A polymorphic human inversion has two arrangements. The chimpanzee carries one
of them at the orthologous locus unless its own lineage rearranged the region.
Under parsimony, the arrangement shared with the chimpanzee is ancestral. So
the question for each locus is: at the orthologous chimpanzee locus, is the
sequence inside the inversion in the same order as its flanks (GRCh38 shares
the ancestral arrangement, ``direct``) or reversed (GRCh38 carries the derived
arrangement, ``inverted``)?

Orthology is positional, not best-hit. The two flanks are the synteny anchors:
they must align to one chimpanzee chromosome, on the same strand, in the
expected order. The interior must then align to the chimpanzee sequence that
lies between those anchors. Alignments of the interior or of a flank to any
other place are paralogous copies (the segmental duplications that mediate
these inversions) and carry no information about the arrangement.

Rules and their reasons
=======================
1. Divergence tiers (3%, then 5%, 8%, 12%). Orthologous human and chimpanzee
   sequence is about 1.2% divergent; duplications older than the split are
   more divergent. The tightest tier that leaves any alignment is used, as in
   the renderer.
2. Primary and secondary alignments are treated alike. Windows of nested and
   adjacent inversions overlap, so one chimpanzee segment is primary in one
   window and secondary in the next; the flag says nothing about orthology
   here. Orthology comes from position instead (rules 5, 6 and 8).
3. The span floor scales with the inversion: 3% of its length, at least 1 kb
   and at most 2 kb. A 1 kb inversion covered by one long colinear alignment
   is direct; a reversed alignment of at least 1 kb between two colinear
   alignments is inverted; alignment debris shorter than 3% of a large locus
   is ignored, and so is anything under 1 kb, which is seed-level noise for
   the asm20 preset.
4. The interior votes with its central 60%. The breakpoints of SD-mediated
   inversions sit inside inverted repeats, and the human inversion interval
   includes those repeats; the repeat copy that continues the flank aligns
   colinearly into the ends of the interval regardless of the arrangement.
   The central sequence is the single-copy sequence whose order the inversion
   changed.
5. Each flank votes with the alignments nearest its breakpoint, within the
   distance the plot displays (150 kb or 60% of the inversion length,
   whichever is larger). A rearrangement further out is a different event.
6. Flank anchors are chosen as a consistent pair. On each side the candidate
   runs are the colinear runs (alignments grouped by chimpanzee offset and
   strand) that carry at least 10 kb of the flank, plus any alignment spanning
   flank and interior, weighted by its part in the flank. The left and right
   candidates must share a strand and sit in the expected order on the
   chimpanzee contig; among such pairs the one with the most aligned sequence
   is taken. This resolves the two cases that a one-sided rule cannot: a
   colinear alignment spanning a small inversion is paired with itself, while
   an inverted block whose chimpanzee breakpoint overshoots the human call is
   rejected as a flank because it cannot pair with the true anchor on the
   other side.
7. Each chosen anchor must hold at least 60% against the largest run of the
   opposite strand that stays outside the interior in the same flank. Inside
   the inverted repeats that mediate recurrent inversions, both strands align
   in comparable amounts and neither is an anchor; the locus is then left
   uncalled, which is the honest reading of such a plot. The interior must be
   at least 30% covered by its dominant strand with an 80% majority, so a
   deletion or an unaligned duplication leaves a gap without producing a
   call from a sliver.
8. Only interior alignments placed between the two anchors on the chimpanzee
   contig vote. Interior sequence aligned anywhere else on the contig is a
   paralogous copy. With no consistent anchor pair the locus is left
   uncalled: a transposition, a paralogous placement, or a neighbouring
   rearrangement in one flank all end here.

Everything else is ``na``: unresolved, not ancestral.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

DIVERGENCE_TIERS = (0.03, 0.05, 0.08, 0.12)
MIN_FLANK_BP = 10_000
MIN_INTERIOR_FRACTION = 0.3
MIN_MAJORITY = 0.8
MIN_ANCHOR_SHARE = 0.6
MAX_CANDIDATES = 4
MIN_ALIGNMENT_BP = 1_000
NEAR_FLANK_BP = 150_000
NEAR_FLANK_FRACTION = 0.6
CORE_TRIM = 0.2
DIAGONAL_TOLERANCE = 20_000
ANCHOR_SLACK = 0.1


@dataclass
class Alignment:
    contig: str
    q_start: int
    q_end: int
    strand: str
    t_start: int
    t_end: int
    primary: bool
    divergence: float

    @property
    def t_span(self) -> int:
        return self.t_end - self.t_start

    @property
    def q_span(self) -> int:
        return self.q_end - self.q_start

    def overlap(self, start: int, end: int) -> int:
        return max(0, min(self.t_end, end) - max(self.t_start, start))

    def mapped(self, start: int, end: int) -> tuple[float, float] | None:
        """Chimpanzee interval of the part of this alignment inside [start, end)."""
        lo = max(self.t_start, start)
        hi = min(self.t_end, end)
        if hi <= lo:
            return None
        f1 = (lo - self.t_start) / self.t_span
        f2 = (hi - self.t_start) / self.t_span
        if self.strand == "+":
            return (self.q_start + f1 * self.q_span, self.q_start + f2 * self.q_span)
        return (self.q_end - f2 * self.q_span, self.q_end - f1 * self.q_span)


def read_paf(path: Path, offset: int) -> list[Alignment]:
    records = []
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                continue
            primary = True
            divergence = None
            for tag in fields[12:]:
                if tag.startswith("tp:A:"):
                    primary = tag[5:] == "P"
                elif tag.startswith("de:f:"):
                    divergence = float(tag[5:])
            if divergence is None:
                divergence = 1.0 - int(fields[9]) / max(int(fields[10]), 1)
            records.append(
                Alignment(
                    contig=fields[0],
                    q_start=int(fields[2]),
                    q_end=int(fields[3]),
                    strand=fields[4],
                    t_start=int(fields[7]) + offset,
                    t_end=int(fields[8]) + offset,
                    primary=primary,
                    divergence=divergence,
                )
            )
    return records


def strand_support(records: list[Alignment], start: int, end: int) -> dict[str, int]:
    total = {"+": 0, "-": 0}
    for record in records:
        total[record.strand] += record.overlap(start, end)
    return total


def majority(total: dict[str, int]) -> tuple[str, float, int]:
    amount = total["+"] + total["-"]
    if amount == 0:
        return "na", 0.0, 0
    strand = "+" if total["+"] >= total["-"] else "-"
    return strand, total[strand] / amount, total[strand]


def choose_contig(
    records: list[Alignment], region_start: int, inv_start: int, inv_end: int, region_end: int
) -> str:
    """The contig reaching both flanks with the most aligned sequence, as in the renderer."""
    best = None
    for contig in sorted({record.contig for record in records}):
        frame = [record for record in records if record.contig == contig]
        left = sum(r.overlap(region_start, inv_start) for r in frame)
        right = sum(r.overlap(inv_end, region_end) for r in frame)
        interior = sum(r.overlap(inv_start, inv_end) for r in frame)
        spans = min(left, right) > 0
        key = (spans, left + interior + right if spans else 0, left + right, left + interior + right)
        if best is None or key > best[0]:
            best = (key, contig)
    return best[1]


def mapped_interval(records: list[Alignment], start: int, end: int, strand: str):
    """Union and weighted midpoint of the chimpanzee intervals mapped from [start, end)."""
    lo = hi = None
    weight = 0
    total = 0.0
    for record in records:
        if record.strand != strand:
            continue
        mapped = record.mapped(start, end)
        if mapped is None:
            continue
        lo = mapped[0] if lo is None else min(lo, mapped[0])
        hi = mapped[1] if hi is None else max(hi, mapped[1])
        amount = record.overlap(start, end)
        weight += amount
        total += amount * (mapped[0] + mapped[1]) / 2
    if weight == 0:
        return None, None
    return (lo, hi), total / weight


@dataclass
class Anchor:
    strand: str
    support: int
    fraction: float
    q_mid: float


def diagonal(record: Alignment) -> float:
    """Chimpanzee offset of a colinear run; alignments on one run share it."""
    if record.strand == "+":
        return record.q_start - record.t_start
    return record.q_end + record.t_start


def candidate_runs(
    records: list[Alignment], start: int, end: int, core: tuple[int, int]
) -> list[Anchor]:
    """Candidate anchors inside [start, end): qualifying colinear runs and spanning alignments."""
    inside = [r for r in records if r.overlap(start, end) > 0]
    pure = [r for r in inside if r.overlap(core[0], core[1]) == 0]
    spanning = [r for r in inside if r.overlap(core[0], core[1]) > 0]

    def build_runs(candidates: list[Alignment]) -> list[list[Alignment]]:
        ordered = sorted(candidates, key=lambda r: (r.strand, diagonal(r)))
        runs: list[list[Alignment]] = []
        for record in ordered:
            if (
                runs
                and runs[-1][0].strand == record.strand
                and abs(diagonal(record) - diagonal(runs[-1][-1])) <= DIAGONAL_TOLERANCE
            ):
                runs[-1].append(record)
            else:
                runs.append([record])
        return runs

    def weight(run: list[Alignment]) -> int:
        return sum(r.overlap(start, end) for r in run)

    pure_runs = build_runs(pure)
    runs = [run for run in pure_runs if weight(run) >= MIN_FLANK_BP]
    runs += [run for run in build_runs(spanning) if weight(run) >= MIN_FLANK_BP]
    runs.sort(key=weight, reverse=True)
    anchors = []
    for run in runs[:MAX_CANDIDATES]:
        support = weight(run)
        opposite = max(
            (weight(other) for other in pure_runs if other[0].strand != run[0].strand), default=0
        )
        _, q_mid = mapped_interval(run, start, end, run[0].strand)
        anchors.append(Anchor(run[0].strand, support, support / (support + opposite), q_mid))
    return anchors


def anchor_pair(left: list[Anchor], right: list[Anchor]) -> tuple[Anchor, Anchor] | None:
    """The consistent (same strand, expected order) pair with the most support."""
    best = None
    for a in left:
        for b in right:
            if a.strand != b.strand:
                continue
            ordered = a.q_mid < b.q_mid if a.strand == "+" else a.q_mid > b.q_mid
            if not ordered:
                continue
            score = a.support + b.support
            if best is None or score > best[0]:
                best = (score, a, b)
    return None if best is None else (best[1], best[2])


def call_locus(records: list[Alignment], window: dict[str, str]) -> dict[str, object]:
    inv_start = int(window["inv_start"])
    inv_end = int(window["inv_end"])
    region_start = int(window["window_start"])
    region_end = int(window["window_end"])
    inv_length = inv_end - inv_start
    result: dict[str, object] = {"auto_call": "na", "auto_reason": ""}

    if not records:
        result["auto_reason"] = "no alignment"
        return result

    # Rule 1: divergence tier.
    divergence_limit = next(
        (tier for tier in DIVERGENCE_TIERS if any(r.divergence <= tier for r in records)),
        None,
    )
    if divergence_limit is None:
        result["auto_reason"] = "no alignment below 12% divergence"
        return result
    records = [r for r in records if r.divergence <= divergence_limit]

    # Rule 3: span floor scaled to the inversion.
    min_span = min(2000, max(0.03 * inv_length, MIN_ALIGNMENT_BP))
    records = [r for r in records if r.t_span >= min_span and r.q_span >= min_span]
    if not records:
        result["auto_reason"] = "no alignment above span floor"
        return result

    contig = choose_contig(records, region_start, inv_start, inv_end, region_end)
    frame = [r for r in records if r.contig == contig]
    result["chimp_contig"] = contig
    result["divergence_limit"] = divergence_limit
    result["min_span_bp"] = round(min_span)

    # Rules 5 and 6: one colinear run per flank, nearest the breakpoint.
    near = int(max(NEAR_FLANK_BP, NEAR_FLANK_FRACTION * inv_length))
    left_start = max(region_start, inv_start - near)
    right_end = min(region_end, inv_end + near)
    core_start = inv_start + int(CORE_TRIM * inv_length)
    core_end = inv_end - int(CORE_TRIM * inv_length)
    left_candidates = candidate_runs(frame, left_start, inv_start, (core_start, core_end))
    right_candidates = candidate_runs(frame, inv_end, right_end, (core_start, core_end))
    pair = anchor_pair(left_candidates, right_candidates)
    if pair is None:
        # Report the strongest candidate on each side so the reason is visible.
        for side, candidates in (("left", left_candidates), ("right", right_candidates)):
            top = candidates[0] if candidates else None
            result[f"{side}_vote"] = "na" if top is None else top.strand
            result[f"{side}_bp"] = 0 if top is None else top.support
            result[f"{side}_fraction"] = 0.0 if top is None else round(top.fraction, 3)
        if not left_candidates or not right_candidates:
            result["auto_reason"] = "flank without alignment"
        elif all(a.strand != b.strand for a in left_candidates for b in right_candidates):
            result["auto_reason"] = "flanks disagree"
        else:
            result["auto_reason"] = "flanks out of order on the chimpanzee contig"
        return result
    left_anchor, right_anchor = pair
    for side, anchor in (("left", left_anchor), ("right", right_anchor)):
        result[f"{side}_vote"] = anchor.strand
        result[f"{side}_bp"] = anchor.support
        result[f"{side}_fraction"] = round(anchor.fraction, 3)
    if left_anchor.fraction < MIN_ANCHOR_SHARE or right_anchor.fraction < MIN_ANCHOR_SHARE:
        result["auto_reason"] = "flank anchor not dominant over the opposite strand"
        return result
    strand = left_anchor.strand
    gap = (
        (left_anchor.q_mid, right_anchor.q_mid)
        if strand == "+"
        else (right_anchor.q_mid, left_anchor.q_mid)
    )

    # Rules 4 and 8: the central interior sequence placed between the anchors.
    slack = ANCHOR_SLACK * (gap[1] - gap[0])
    between = []
    for record in frame:
        mapped = record.mapped(core_start, core_end)
        if mapped is None:
            continue
        if mapped[0] >= gap[0] - slack and mapped[1] <= gap[1] + slack:
            between.append(record)
    interior, interior_frac, interior_bp = majority(strand_support(between, core_start, core_end))
    result["interior_vote"] = interior
    result["interior_bp"] = interior_bp
    result["interior_fraction"] = round(interior_frac, 3)
    if interior == "na":
        result["auto_reason"] = "interior without alignment between the flanks"
        return result
    if interior_bp < MIN_INTERIOR_FRACTION * (core_end - core_start):
        result["auto_reason"] = "interior support below minimum"
        return result
    if interior_frac < MIN_MAJORITY:
        result["auto_reason"] = "interior strand not a clear majority"
        return result

    if interior == strand:
        result["auto_call"] = "direct"
        result["auto_reason"] = "interior colinear with both flanks"
    else:
        result["auto_call"] = "inverted"
        result["auto_reason"] = "interior reversed relative to both flanks"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--previous-ledger", type=Path, required=True)
    parser.add_argument(
        "--paf-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory holding <inv_id>.paf files; repeatable",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.windows.open(newline="") as handle:
        windows = list(csv.DictReader(handle, delimiter="\t"))
    previous = {
        record["inv_id"]: record["classification"]
        for record in json.loads(args.previous_ledger.read_text())["responses"]
    }

    fields = [
        "inv_id",
        "chrom",
        "inv_start",
        "inv_end",
        "source",
        "chimp_contig",
        "divergence_limit",
        "min_span_bp",
        "left_vote",
        "left_bp",
        "left_fraction",
        "interior_vote",
        "interior_bp",
        "interior_fraction",
        "right_vote",
        "right_bp",
        "right_fraction",
        "auto_call",
        "auto_reason",
        "previous_call",
        "changed",
    ]
    counts: dict[str, int] = {}
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for window in windows:
            inv_id = window["inv_id"]
            paf = next(
                (d / f"{inv_id}.paf" for d in args.paf_dir if (d / f"{inv_id}.paf").is_file()),
                None,
            )
            records = [] if paf is None else read_paf(paf, int(window["window_start"]))
            result = call_locus(records, window)
            call = str(result["auto_call"])
            counts[call] = counts.get(call, 0) + 1
            row = {field: "" for field in fields}
            row.update(
                {
                    "inv_id": inv_id,
                    "chrom": window["chrom"],
                    "inv_start": window["inv_start"],
                    "inv_end": window["inv_end"],
                    "source": window["source"],
                    "previous_call": previous.get(inv_id, ""),
                    "changed": str(previous.get(inv_id, "") != call).lower(),
                }
            )
            row.update({k: v for k, v in result.items() if k in row})
            writer.writerow(row)
    print(f"Wrote {len(windows)} rows to {args.out}: {counts}")


if __name__ == "__main__":
    main()
