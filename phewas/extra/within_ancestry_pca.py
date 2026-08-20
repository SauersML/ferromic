"""Fit ancestry-specific ("fine-scale") principal components, one group at a time.

The PheWAS adjusts for the principal components published alongside the callset, which
are produced by projecting participants onto a cross-ancestry reference. Those axes are
built to separate continental groups, so inside a single genetic ancestry group they
carry very little variance and cannot control fine-scale (within-continent) structure.
This script fits components separately within each group so that structure is visible.

It is the producer for the ``--pc-source within-ancestry`` consumer in the PheWAS: it
writes ``within_ancestry_pcs_{pop}.tsv`` (``person_id`` plus ``WPC1..WPCk``) together
with a JSON sidecar recording exactly what was fit, which the analysis reads back and
which doubles as the provenance record for the methods.

Typical use inside the workbench::

    python -m phewas.extra.within_ancestry_pca sites \\
        --bim /mounted/arrays.bim --out local/sites/include_sites.tsv

    python -m phewas.extra.within_ancestry_pca fit \\
        --genotypes local/arrays --sites local/sites/include_sites.tsv \\
        --ancestry ancestry_preds.tsv --cohort cohort_person_ids.txt \\
        --group eur --out-dir within_ancestry_pcs

The genotypes must come from a single platform. Fitting some participants from the array
and others from sequence data invites the leading component to become a platform axis,
which would then be adjusted for as though it were ancestry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# Regions whose linkage disequilibrium extends far enough to bend principal components
# away from ancestry and toward local haplotype structure. These are the conventional
# exclusions; the tested inversions are added separately because they are the exposure.
#
# Coordinates are GRCh38 and are deliberately generous. Over-excluding costs a few
# thousand markers out of several hundred thousand, which is free; under-excluding lets
# a single haplotype block dominate a component, which silently corrupts the adjustment.
# Verify the build of any genotype source before use -- applying these to GRCh37 data
# would shift every interval by up to a few megabases.
LONG_RANGE_LD_REGIONS: tuple[tuple[str, int, int, str], ...] = (
    ("chr1", 46_500_000, 50_000_000, "chr1p13 long-range LD"),
    ("chr2", 85_000_000, 101_000_000, "chr2p11-q11 long-range LD"),
    ("chr3", 47_000_000, 50_000_000, "chr3p21 long-range LD"),
    ("chr5", 43_500_000, 52_000_000, "chr5q31 long-range LD"),
    ("chr6", 24_000_000, 36_000_000, "MHC"),
    ("chr8", 6_500_000, 13_500_000, "8p23.1 inversion / long-range LD"),
    ("chr11", 44_500_000, 57_500_000, "chr11 centromeric long-range LD"),
    ("chr12", 33_000_000, 41_000_000, "chr12 centromeric long-range LD"),
    ("chr17", 42_500_000, 48_000_000, "17q21.31 inversion / long-range LD"),
)

# Padding applied either side of each tested inversion. The exposure is imputed from
# variants in and around these intervals, so a component that loaded on them would
# partly be the exposure itself.
INVERSION_FLANK_BP = 1_000_000

# Production settings passed explicitly to gnomon's biobank fit path. LD uses current
# main's native safety policy: --ld alone selects an evenly spaced 100,000-marker budget
# and a 500 kbp physical window.
DEFAULT_COMPONENTS = 16
DEFAULT_THREADS = 4
DEFAULT_MAF = 0.01
DEFAULT_GENO = 0.05
DEFAULT_MIND = 0.05
GNOMON_LD_DEFAULT_MARKERS = 100_000
GNOMON_LD_DEFAULT_WINDOW_BP = 500_000

_INVERSION_ID = re.compile(r"^(chr[0-9XYM]+)-(\d+)-INV-(\d+)$")


@dataclass(frozen=True)
class Region:
    chrom: str
    start: int
    end: int
    reason: str


def _normalise_chrom(value: str) -> str:
    text = str(value).strip()
    return text if text.startswith("chr") else f"chr{text}"


def inversion_regions(inversion_ids: Iterable[str], flank: int = INVERSION_FLANK_BP) -> list[Region]:
    """Turn ``chr17-45585160-INV-706887`` style ids into padded exclusion intervals."""
    regions: list[Region] = []
    for inversion_id in sorted(inversion_ids):
        match = _INVERSION_ID.match(str(inversion_id).strip())
        if match is None:
            raise ValueError(f"Unrecognised inversion id: {inversion_id!r}")
        chrom, start, length = match.group(1), int(match.group(2)), int(match.group(3))
        regions.append(
            Region(
                chrom=_normalise_chrom(chrom),
                start=max(0, start - flank),
                end=start + length + flank,
                reason=f"tested inversion {inversion_id} +/- {flank:,} bp",
            )
        )
    return regions


def excluded_regions(inversion_ids: Optional[Iterable[str]] = None) -> list[Region]:
    if inversion_ids is None:
        from phewas import run as _run

        inversion_ids = _run.TARGET_INVERSIONS
    regions = [Region(c, s, e, why) for c, s, e, why in LONG_RANGE_LD_REGIONS]
    regions.extend(inversion_regions(inversion_ids))
    return regions


def build_site_list(
    bim_path: str,
    out_path: str,
    inversion_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Write the variant include-list consumed by ``gnomon fit --list``.

    Autosomes only, minus every long-range LD region and every tested inversion. The
    output is the two-column ``chromosome position`` format gnomon expects.
    """
    bim = pd.read_csv(
        bim_path,
        sep=r"\s+",
        header=None,
        names=["chrom", "variant_id", "cm", "pos", "a1", "a2"],
        dtype={"chrom": str, "variant_id": str, "pos": np.int64},
    )
    bim["chrom"] = bim["chrom"].map(_normalise_chrom)

    autosomes = {f"chr{i}" for i in range(1, 23)}
    keep = bim["chrom"].isin(autosomes)
    n_non_autosomal = int((~keep).sum())

    regions = excluded_regions(inversion_ids)
    dropped_by_region: dict[str, int] = {}
    for region in regions:
        hit = (
            keep
            & (bim["chrom"] == region.chrom)
            & (bim["pos"] >= region.start)
            & (bim["pos"] <= region.end)
        )
        dropped_by_region[region.reason] = int(hit.sum())
        keep &= ~hit

    sites = bim.loc[keep, ["chrom", "pos"]].sort_values(["chrom", "pos"])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    sites.to_csv(out_path, sep="\t", header=False, index=False)

    print(f"[sites] {len(bim):,} variants in {bim_path}")
    print(f"[sites] dropped {n_non_autosomal:,} non-autosomal")
    for reason, count in dropped_by_region.items():
        print(f"[sites] dropped {count:>8,}  {reason}")
    print(f"[sites] retained {len(sites):,} -> {out_path}")
    return sites


def _digest(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def _full_digest(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


# The dosage table's id column is "SampleID", so passing that file directly as --cohort
# works; without this its header would be read back as a participant.
_ID_HEADER_TOKENS = {
    "person_id", "research_id", "sample_id", "sampleid", "participant_id",
    "iid", "fid", "id", "s",
}


def _read_ids(path: str) -> list[str]:
    """Read participant ids from a one-per-line file or the first column of a table.

    The relatedness list published with the callset is headerless, but cohort files
    exported by hand usually are not, so a leading column name is dropped rather than
    silently treated as a participant.
    """
    frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, comment="#")
    values = [str(v).strip() for v in frame.iloc[:, 0].dropna().tolist()]
    if values and values[0].lower() in _ID_HEADER_TOKENS:
        values = values[1:]
    return values


def _read_indexed_table(path: str) -> pd.DataFrame:
    """Read a TSV whose first column is a participant id, keeping ids as strings."""
    frame = pd.read_csv(path, sep="\t", dtype=str)
    id_col = frame.columns[0]
    values = frame.drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce")
    values.index = pd.Index(frame[id_col].astype(str), name="person_id")
    return values


def build_keep_list(
    ancestry_path: str,
    group: str,
    out_path: str,
    cohort_path: Optional[str] = None,
    related_path: Optional[str] = None,
) -> list[str]:
    """Write the participant list for one ancestry group.

    The intersection matters: components must be fit on the same people the association
    models are fit on, or the two are not comparable.
    """
    ancestry = pd.read_csv(ancestry_path, sep="\t", dtype=str)
    id_col = next(
        (c for c in ("person_id", "research_id", "IID", "s") if c in ancestry.columns), None
    )
    label_col = next(
        (c for c in ("ANCESTRY", "ancestry_pred", "ancestry") if c in ancestry.columns), None
    )
    if id_col is None or label_col is None:
        raise ValueError(
            f"Ancestry table needs an id and a label column; found {list(ancestry.columns)[:8]}."
        )

    normalized = ancestry[label_col].astype(str).str.strip().str.lower()
    selected = set(ancestry.loc[normalized == group.strip().lower(), id_col].astype(str))
    print(f"[keep] {len(selected):,} participants labelled '{group}'")

    if cohort_path:
        cohort = set(_read_ids(cohort_path))
        selected &= cohort
        print(f"[keep] {len(selected):,} after intersecting the analysis cohort")
    if related_path:
        related = set(_read_ids(related_path))
        selected -= related
        print(f"[keep] {len(selected):,} after removing relatedness-flagged participants")

    ids = sorted(selected)
    if not ids:
        raise ValueError(f"No participants remain for group '{group}'.")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write("\n".join(ids) + "\n")
    print(f"[keep] wrote {len(ids):,} ids -> {out_path}")
    return ids


def _run_gnomon_fit(
    gnomon: str,
    genotypes: str,
    output_prefix: str,
    keep_path: str,
    sites_path: str,
    components: int,
    threads: int,
    maf: float,
    geno: float,
    mind: float,
    workdir: str,
) -> dict[str, object]:
    version = subprocess.run(
        [gnomon, "version"],
        cwd=workdir,
        text=True,
        capture_output=True,
        check=False,
    )
    if version.returncode != 0:
        raise SystemExit(
            f"Could not obtain gnomon build provenance with '{gnomon} version' "
            f"(exit {version.returncode}): {version.stderr.strip()}"
        )

    command = [
        gnomon,
        "fit",
        genotypes,
        "--out",
        output_prefix,
        "--keep",
        keep_path,
        "--list",
        sites_path,
        "--components",
        str(components),
        "--threads",
        str(threads),
        "--mind",
        str(mind),
        "--geno",
        str(geno),
        "--maf",
        str(maf),
        "--markers",
        str(GNOMON_LD_DEFAULT_MARKERS),
        "--ld",
        "--bp_window",
        str(GNOMON_LD_DEFAULT_WINDOW_BP),
    ]
    print("[fit] " + shlex.join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=workdir,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"gnomon fit failed with exit code {completed.returncode}.")
    return {
        "version": version.stdout.strip(),
        "executable": os.path.abspath(gnomon),
        "executable_sha256": _full_digest(gnomon),
        "command": command,
        "command_shell": shlex.join(command),
    }


# gnomon writes the fitted scores as a binary matrix with an embedded row-ID section,
# described by a JSON sidecar. Format constants mirror map/io.rs.
_MATRIX_MAGIC = b"GNPRJ001"
_MATRIX_VERSION = 3
_MATRIX_HEADER_LEN = 32
_MATRIX_ELEMENT_KIND_F64_LE = 1
_ROW_IDS_MAGIC = b"GNPSID01"
_ROW_IDS_VERSION = 1
_ROW_IDS_HEADER_LEN = 32

_FIT_ARTIFACTS = (
    "hwe.json",
    "hwe.project.bin",
    "hwe_scores.bin",
    "hwe_scores.metadata.json",
    "samples.tsv",
    "hwe_summary.tsv",
)


def _artifact_paths(output_prefix: str) -> dict[str, str]:
    """Return the artifact paths produced by current main's ``--out PREFIX``."""
    prefix = os.path.abspath(output_prefix)
    return {name: f"{prefix}.{name}" for name in _FIT_ARTIFACTS}


def _plink_prefix(genotype_path: str) -> str:
    """Resolve a PLINK1 prefix and require its BED/BIM/FAM files."""
    path = os.path.abspath(os.path.normpath(genotype_path))
    for suffix in (".bed", ".bim", ".fam"):
        if path.lower().endswith(suffix):
            path = path[: -len(suffix)]
            break
    missing = [
        path + suffix
        for suffix in (".bed", ".bim", ".fam")
        if not os.path.isfile(path + suffix)
    ]
    if missing:
        raise SystemExit(
            "The within-ancestry fit requires one indexed PLINK1 BED/BIM/FAM trio; "
            f"missing: {', '.join(missing)}"
        )
    return path


def _read_hwe_scores(bin_path: str, metadata_path: str) -> tuple[np.ndarray, list[str]]:
    """Decode gnomon's score matrix and the participant ids embedded alongside it."""
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    for key, expected in (
        ("version", _MATRIX_VERSION),
        ("kind", "scores"),
        ("layout", "column_major"),
        ("dtype", "f64_le"),
        ("element_kind", _MATRIX_ELEMENT_KIND_F64_LE),
        ("endianness", "little"),
        ("row_ids_embedded", True),
        ("row_id_field", "IID"),
    ):
        if metadata.get(key) != expected:
            raise SystemExit(
                f"Unexpected {key} in {metadata_path}: {metadata.get(key)!r} (expected {expected!r}). "
                "The gnomon output format has changed; update this reader before trusting the result."
            )
    rows = int(metadata["rows"])
    cols = int(metadata["cols"])
    if rows < 1 or cols < 1:
        raise SystemExit(f"{metadata_path} reports a non-positive matrix shape.")

    with open(bin_path, "rb") as handle:
        raw = handle.read()
    if len(raw) < _MATRIX_HEADER_LEN:
        raise SystemExit(f"{bin_path} is shorter than the gnomon matrix header.")
    if raw[:8] != _MATRIX_MAGIC:
        raise SystemExit(f"{bin_path} is not a gnomon score matrix.")
    if int.from_bytes(raw[8:12], "little") != _MATRIX_VERSION:
        raise SystemExit(f"{bin_path} has an unsupported matrix version.")
    if (
        int.from_bytes(raw[12:20], "little") != rows
        or int.from_bytes(raw[20:28], "little") != cols
    ):
        raise SystemExit(f"{bin_path} header shape disagrees with {metadata_path}.")
    if int.from_bytes(raw[28:32], "little") != _MATRIX_ELEMENT_KIND_F64_LE:
        raise SystemExit(f"{bin_path} does not contain little-endian f64 scores.")

    matrix_end = _MATRIX_HEADER_LEN + rows * cols * 8
    if len(raw) < matrix_end + _ROW_IDS_HEADER_LEN:
        raise SystemExit(f"{bin_path} is truncated before its row-ID section.")
    scores = (
        np.frombuffer(raw[_MATRIX_HEADER_LEN:matrix_end], dtype="<f8", count=rows * cols)
        .reshape((cols, rows))
        .T
    )

    section = raw[matrix_end:]
    if section[:8] != _ROW_IDS_MAGIC:
        raise SystemExit(f"{bin_path} is missing its row-ID section.")
    if int.from_bytes(section[8:12], "little") != _ROW_IDS_VERSION:
        raise SystemExit(f"{bin_path} has an unsupported row-ID version.")
    row_count = int.from_bytes(section[16:24], "little")
    string_bytes = int.from_bytes(section[24:32], "little")
    if row_count != rows:
        raise SystemExit(f"{bin_path} row-ID count disagrees with the score matrix.")

    offsets_end = _ROW_IDS_HEADER_LEN + (row_count + 1) * 8
    expected_section_bytes = offsets_end + string_bytes
    if len(section) != expected_section_bytes:
        raise SystemExit(
            f"{bin_path} row-ID section length disagrees with its embedded header."
        )
    offsets = [
        int.from_bytes(section[_ROW_IDS_HEADER_LEN + i * 8 : _ROW_IDS_HEADER_LEN + (i + 1) * 8], "little")
        for i in range(row_count + 1)
    ]
    if (
        offsets[0] != 0
        or offsets[-1] != string_bytes
        or any(left > right for left, right in zip(offsets, offsets[1:]))
    ):
        raise SystemExit(f"{bin_path} row-ID offsets are inconsistent.")
    table = section[offsets_end:]
    row_ids = [table[offsets[i] : offsets[i + 1]].decode("utf-8") for i in range(row_count)]
    return scores, row_ids


def _fit_artifacts(output_prefix: str) -> dict[str, str]:
    """Require the complete artifact set written by current main."""
    produced = _artifact_paths(output_prefix)
    absent = [name for name, path in produced.items() if not os.path.isfile(path)]
    if absent:
        raise SystemExit(
            f"gnomon fit did not produce required current-main artifacts: {', '.join(absent)}."
        )
    return produced


def _fit_summary(path: str) -> dict[str, str]:
    """Read current main's metric/value summary and require a certified solve."""
    frame = pd.read_csv(path, sep="\t", dtype=str)
    if list(frame.columns) != ["metric", "value"]:
        raise SystemExit(
            f"Unexpected gnomon fit-summary columns in {path}: {list(frame.columns)!r}."
        )
    if frame["metric"].duplicated().any():
        duplicate = frame.loc[frame["metric"].duplicated(), "metric"].iloc[0]
        raise SystemExit(f"Gnomon fit summary contains duplicate metric {duplicate!r}.")
    summary = dict(zip(frame["metric"], frame["value"]))
    if summary.get("converged", "").lower() != "true":
        raise SystemExit(
            "Gnomon returned artifacts without a certified converged eigensolve; "
            "refusing to write PheWAS covariates."
        )
    return summary


def _scores_frame(collected: dict[str, str], components: int) -> pd.DataFrame:
    scores, row_ids = _read_hwe_scores(
        collected["hwe_scores.bin"], collected["hwe_scores.metadata.json"]
    )
    if len(row_ids) != len(set(row_ids)):
        raise SystemExit("Gnomon's fitted score matrix contains duplicate participant IDs.")
    if scores.shape[1] < components:
        raise SystemExit(
            f"Expected {components} components but the fitted model returned {scores.shape[1]}. "
            "The requested count exceeded the rank the data support."
        )
    frame = pd.DataFrame(
        scores[:, :components],
        columns=[f"WPC{i}" for i in range(1, components + 1)],
    )
    frame.insert(0, "person_id", [str(v) for v in row_ids])
    return frame


def _quality_report(scores: pd.DataFrame, global_pcs: Optional[pd.DataFrame], dosages: Optional[pd.DataFrame]) -> dict:
    """Checks that decide whether these components are safe to adjust for."""
    report: dict[str, object] = {}
    values = scores.drop(columns=["person_id"])
    report["variance_per_component"] = {c: float(np.var(values[c])) for c in values.columns}

    if global_pcs is not None:
        indexed = scores.set_index("person_id")
        shared = indexed.index.intersection(global_pcs.index)
        if len(shared) > 10:
            first_global = [c for c in global_pcs.columns if re.fullmatch(r"PC\d+", str(c))][:5]
            report["max_abs_corr_with_global_pcs"] = {
                c: float(
                    np.nanmax(
                        np.abs(
                            [
                                np.corrcoef(indexed.loc[shared, c], global_pcs.loc[shared, g])[0, 1]
                                for g in first_global
                            ]
                        )
                    )
                )
                for c in values.columns
            }

    if dosages is not None:
        indexed = scores.set_index("person_id")
        shared = indexed.index.intersection(dosages.index)
        if len(shared) > 10:
            worst = {}
            for inversion in dosages.columns:
                dose = pd.to_numeric(dosages.loc[shared, inversion], errors="coerce")
                if not np.isfinite(dose).all() or float(np.std(dose)) == 0.0:
                    continue
                worst[str(inversion)] = float(
                    np.nanmax(
                        np.abs(
                            [np.corrcoef(indexed.loc[shared, c], dose)[0, 1] for c in values.columns]
                        )
                    )
                )
            # These should be near zero: the inversion loci are excluded from the sites
            # used to fit. A large value means the exclusion did not take effect, and the
            # components would partly absorb the exposure.
            report["max_abs_corr_with_inversion_dosage"] = worst
    return report


def fit_group(args: argparse.Namespace) -> None:
    group = args.group.strip().lower()
    if re.fullmatch(r"[a-z][a-z0-9_-]*", group) is None:
        raise SystemExit(
            "--group must start with a letter and contain only letters, digits, '_' or '-'."
        )
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    output_prefix = os.path.join(out_dir, f"gnomon_{group}")

    keep_path = args.keep or os.path.join(out_dir, f"keep_{group}.txt")
    if args.keep is None:
        ids = build_keep_list(
            ancestry_path=args.ancestry,
            group=group,
            out_path=keep_path,
            cohort_path=args.cohort,
            related_path=args.related,
        )
    else:
        ids = _read_ids(keep_path)
        print(f"[keep] using supplied list of {len(ids):,} ids")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"The keep list for group '{group}' contains duplicate participant IDs.")

    components = args.components
    if components < 1:
        raise SystemExit("--components must be at least 1.")
    if len(ids) <= components:
        raise SystemExit(
            f"Group '{group}' has {len(ids):,} participants, which cannot support "
            f"{components} ancestry-specific components."
        )
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1.")

    gnomon = args.gnomon
    if os.path.sep in gnomon:
        gnomon = os.path.abspath(gnomon)
    else:
        gnomon = shutil.which(gnomon) or gnomon
    if not os.path.isfile(gnomon) or not os.access(gnomon, os.X_OK):
        raise SystemExit(f"gnomon executable is not available: {gnomon}")

    genotypes = _plink_prefix(args.genotypes) + ".bed"
    print(
        f"[fit] group={group} n={len(ids):,} components={components} "
        f"threads={args.threads}; explicit LD policy: "
        f"{GNOMON_LD_DEFAULT_MARKERS:,} markers / {GNOMON_LD_DEFAULT_WINDOW_BP:,} bp"
    )

    provenance = _run_gnomon_fit(
        gnomon=gnomon,
        genotypes=genotypes,
        output_prefix=output_prefix,
        keep_path=os.path.abspath(keep_path),
        sites_path=os.path.abspath(args.sites),
        components=components,
        threads=args.threads,
        maf=args.maf,
        geno=args.geno,
        mind=args.mind,
        workdir=out_dir,
    )

    collected = _fit_artifacts(output_prefix)
    summary = _fit_summary(collected["hwe_summary.tsv"])
    scores = _scores_frame(collected, components)
    missing = sorted(set(ids) - set(scores["person_id"]))
    if missing:
        raise SystemExit(
            f"Gnomon sample QC removed {len(missing)} participants from the exact PheWAS "
            f"cohort (e.g. {', '.join(missing[:5])}). Refusing to change the association "
            "cohort; resolve genotype missingness upstream or use one matched cohort in both arms."
        )
    unexpected = sorted(set(scores["person_id"]) - set(ids))
    if unexpected:
        raise SystemExit(
            f"Gnomon returned {len(unexpected)} participants absent from --keep "
            f"(e.g. {', '.join(unexpected[:5])})."
        )

    global_pcs = _read_indexed_table(args.global_pcs) if args.global_pcs else None
    dosages = _read_indexed_table(args.dosages) if args.dosages else None

    report = _quality_report(scores, global_pcs, dosages)

    out_path = os.path.join(out_dir, f"within_ancestry_pcs_{group}.tsv")
    scores.to_csv(out_path, sep="\t", index=False)

    sidecar = {
        "population": group,
        "n_samples_requested": len(ids),
        "n_samples_retained": len(scores),
        "components": components,
        "genotypes": genotypes,
        "gnomon_output_prefix": output_prefix,
        "gnomon_artifacts": collected,
        "sites_list": os.path.abspath(args.sites),
        "sites_digest": _digest(args.sites),
        "keep_list": os.path.abspath(keep_path),
        "keep_digest": _digest(keep_path),
        "threads": args.threads,
        "maf": args.maf,
        "geno": args.geno,
        "mind": args.mind,
        "ld": {
            "enabled": True,
            "markers": GNOMON_LD_DEFAULT_MARKERS,
            "bp_window": GNOMON_LD_DEFAULT_WINDOW_BP,
            "source": "explicit gnomon fit arguments",
        },
        "strict_convergence": True,
        "gnomon": provenance,
        "hwe_summary": summary,
        "excluded_regions": [asdict(r) for r in excluded_regions()],
        "quality": report,
    }

    with open(os.path.join(out_dir, f"within_ancestry_pcs_{group}.json"), "w") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)

    print(f"[fit] wrote {out_path}")
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sites = sub.add_parser("sites", help="Build the variant include-list for fitting.")
    sites.add_argument("--bim", required=True, help="PLINK .bim describing the genotypes.")
    sites.add_argument("--out", required=True, help="Destination for the include-list.")

    fit = sub.add_parser("fit", help="Fit components for one ancestry group.")
    fit.add_argument("--genotypes", required=True, help="PLINK1 BED/BIM/FAM prefix.")
    fit.add_argument("--sites", required=True, help="Include-list from the 'sites' step.")
    fit.add_argument("--group", required=True, help="Ancestry label, e.g. eur.")
    fit.add_argument("--out-dir", default="within_ancestry_pcs")
    fit.add_argument("--ancestry", help="Table mapping participants to ancestry labels.")
    fit.add_argument("--cohort", help="Participant ids analysed by the PheWAS.")
    fit.add_argument("--related", help="Relatedness-flagged participants to exclude.")
    fit.add_argument("--keep", help="Pre-built keep list; skips building one.")
    fit.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    fit.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    fit.add_argument("--maf", type=float, default=DEFAULT_MAF)
    fit.add_argument("--geno", type=float, default=DEFAULT_GENO)
    fit.add_argument("--mind", type=float, default=DEFAULT_MIND)
    fit.add_argument("--gnomon", default=shutil.which("gnomon") or "gnomon")
    fit.add_argument("--global-pcs", help="Global PC table, for the concordance check.")
    fit.add_argument("--dosages", help="Inversion dosages, for the contamination check.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "sites":
        build_site_list(args.bim, args.out)
    elif args.command == "fit":
        if args.keep is None and not args.ancestry:
            raise SystemExit("fit needs either --keep or --ancestry.")
        fit_group(args)


if __name__ == "__main__":
    main()
