from __future__ import annotations
import os
import re
import csv
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
HAVE_TQDM = True


# ===========================
# Filename parsing (STRICT)
# ===========================
CDS_RE = re.compile(
    r'^group(?P<phy_group>[01])_'
    r'(?P<gene_name>[^_]+)_'
    r'(?P<gene_id>[^_]+)_'
    r'(?P<transcript_id>[^_]+)_'
    r'chr(?P<chr>[^_]+)_'
    r'cds_start(?P<cds_start>\d+)_cds_end(?P<cds_end>\d+)_'
    r'inv_start(?P<inv_start>\d+)_inv_end(?P<inv_end>\d+)\.phy$'
)

def parse_cds_filename(fn: str) -> Optional[Dict[str, str]]:
    m = CDS_RE.match(fn)
    if not m:
        return None
    g = m.groupdict()
    # Gene-token sanity flags (diagnostic only)
    gene_name = g["gene_name"]
    gene_id = g["gene_id"]
    transcript_id = g["transcript_id"]
    gene_name_anom = gene_name.upper().startswith(("ENSG","ENST","ENSP"))
    bad_gene_id = not gene_id.upper().startswith("ENSG")
    bad_tx_id   = not transcript_id.upper().startswith("ENST")
    return {
        "filename": fn,
        "phy_group": int(g["phy_group"]),
        "gene_name": gene_name,
        "gene_id": gene_id,
        "transcript_id": transcript_id,
        "chr": normalize_chr(g["chr"]),
        "cds_start": int(g["cds_start"]),
        "cds_end": int(g["cds_end"]),
        "inv_start": int(g["inv_start"]),
        "inv_end": int(g["inv_end"]),
        "_gene_name_anom": gene_name_anom,
        "_bad_gene_id": bad_gene_id,
        "_bad_tx_id": bad_tx_id,
    }

# ===========================
# PHYLIP parsing (STRICT)
# ===========================
class PhylipParseError(Exception):
    pass

# Handles non-standard single-line form: "<name_ends_with _L/_R><sequence>"
NONSTD_LINE_RE = re.compile(r'^(?P<name>.*?_[LR])(?P<seq>[ACGTRYKMSWBDHVN\-\.\?]+)$', re.IGNORECASE)

def read_nonstandard_phylip(path: str, n: int, m: int) -> List[Tuple[str,str]]:
    out: List[Tuple[str,str]] = []
    with open(path, "r") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        raise PhylipParseError("Empty file")
    idx += 1  # past header
    while len(out) < n and idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if line == "":
            continue
        mobj = NONSTD_LINE_RE.match(line)
        if not mobj:
            raise PhylipParseError(f"Non-standard PHYLIP: bad sequence line: '{line[:60]}...'")
        name = mobj.group("name")
        seq = mobj.group("seq").upper()
        if len(seq) != m:
            raise PhylipParseError(f"Sequence length mismatch: expected {m}, got {len(seq)} for {name}")
        out.append((name, seq))
    if len(out) != n:
        raise PhylipParseError(f"Expected {n} sequences; got {len(out)}")
    return out

def read_standard_phylip(path: str, n: int, m: int) -> List[Tuple[str,str]]:
    with open(path, "r") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        raise PhylipParseError("Empty file")
    idx += 1  # skip header
    names: List[str] = []
    seqs: List[str] = [""] * n
    # First block
    for i in range(n):
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        if idx >= len(lines):
            raise PhylipParseError(f"Truncated first block at sequence {i+1}/{n}")
        line = lines[idx]
        idx += 1
        if len(line) < 10:
            raise PhylipParseError("Standard PHYLIP requires >=10 chars for name field")
        name = line[:10].strip()
        seq_chunk = "".join(line[10:].split()).upper()
        if name == "":
            raise PhylipParseError("Empty name in standard PHYLIP")
        names.append(name)
        seqs[i] += seq_chunk
    # Subsequent blocks
    while any(len(s) < m for s in seqs):
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        for i in range(n):
            if idx >= len(lines):
                raise PhylipParseError("Unexpected EOF in interleaved blocks")
            line = lines[idx].strip()
            idx += 1
            if line == "":
                raise PhylipParseError("Empty line inside interleaved block")
            seq_chunk = "".join(line.split()).upper()
            seqs[i] += seq_chunk
    for i in range(n):
        if len(seqs[i]) != m:
            raise PhylipParseError(f"Length mismatch for '{names[i]}': expected {m}, got {len(seqs[i])}")
    return list(zip(names, seqs))

def read_phylip_sequences_strict(path: str) -> List[Tuple[str,str]]:
    if not os.path.exists(path):
        raise PhylipParseError(f"File not found: {path}")
    with open(path, "r") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        raise PhylipParseError("Empty file (no header)")
    header = lines[idx].strip()
    idx += 1
    parts = header.split()
    if len(parts) != 2:
        raise PhylipParseError(f"Malformed header: '{header}'")
    try:
        n = int(parts[0]); m = int(parts[1])
    except Exception:
        raise PhylipParseError(f"Non-integer header values: '{header}'")
    if n < 1 or m < 1:
        raise PhylipParseError(f"Non-positive n or m in header: '{header}'")
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        raise PhylipParseError("Header present but no sequences")
    first_line = lines[idx].strip()
    if NONSTD_LINE_RE.match(first_line):
        return read_nonstandard_phylip(path, n, m)
    else:
        return read_standard_phylip(path, n, m)

# ===========================
# Utilities
# ===========================
def normalize_chr(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    s = s.strip()
    if s.upper() in {"X", "Y", "MT", "M"}:
        return "MT" if s.upper() in {"MT", "M"} else s.upper()
    return s

def consensus_label(c: int) -> str:
    return "Recurrent" if c == 1 else "Single-event"

# ===========================
# Load inv_info.tsv (exact triplets w/ consensus in {0,1})
# ===========================
def load_inv_info_exact(path: str):
    if not os.path.exists(path):
        sys.exit(f"ERROR: inv_info.tsv not found at {path}")

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames or []

        def find_col(substr: str):
            for h in header:
                if substr == h or substr in h:
                    return h
            return None

        col_chr  = find_col("Chromosome")
        col_start= find_col("Start")
        col_end  = find_col("End")
        col_cons = find_col("0_single_1_recur_consensus")

        if not all([col_chr, col_start, col_end, col_cons]):
            sys.exit("ERROR: inv_info.tsv missing required columns. Found: " + ", ".join(header))

        triplet_to_cons: Dict[Tuple[str,int,int], int] = {}
        rows_used = 0

        for row in reader:
            chr_raw   = row.get(col_chr, "")
            start_raw = row.get(col_start, "")
            end_raw   = row.get(col_end, "")
            cons_raw  = row.get(col_cons, "")

            chr_norm = normalize_chr(chr_raw)
            try:
                s = int(str(start_raw).strip())
                e = int(str(end_raw).strip())
            except Exception:
                continue

            try:
                cons = int(str(cons_raw).strip())
            except Exception:
                continue

            if cons not in (0,1):
                continue

            triplet_to_cons[(chr_norm, s, e)] = cons
            rows_used += 1

    if rows_used == 0:
        sys.exit("ERROR: No (chr,start,end) rows with consensus ∈ {0,1} in inv_info.tsv.")
    return triplet_to_cons

# ===========================
# Vectorized fixed-diff caller
# ===========================
_ASCII = {b: ord(b) for b in ["A","C","G","T"]}
A = ord("A"); C = ord("C"); G = ord("G"); T = ord("T")

def encode_ascii_matrix(seqs: List[str]) -> np.ndarray:
    n = len(seqs); m = len(seqs[0])
    X = np.empty((n, m), dtype=np.uint8)
    for i, s in enumerate(seqs):
        X[i, :] = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    return X

def fixed_differences_between_groups(seqs_dir: List[str], seqs_inv: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      mask_diff : bool[m], sites that are fixed-different (A/C/G/T only, both groups fixed, alleles differ)
      base_dir  : uint8[m], ASCII code for the fixed base in DIRECT group (only valid where mask_diff True)
      base_inv  : uint8[m], ASCII code for the fixed base in INVERTED group (only valid where mask_diff True)
    """
    if not seqs_dir or not seqs_inv:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.uint8)

    m = len(seqs_dir[0])
    if any(len(s) != m for s in seqs_dir) or any(len(s) != m for s in seqs_inv):
        raise ValueError("Sequences in one or both groups have unequal lengths.")

    X0 = encode_ascii_matrix(seqs_dir)  # shape (n0, m)
    X1 = encode_ascii_matrix(seqs_inv)  # shape (n1, m)
    n0, _ = X0.shape
    n1, _ = X1.shape

    # Count A,C,G,T per column in each group
    def counts_acgt(X: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        return (np.count_nonzero(X == A, axis=0),
                np.count_nonzero(X == C, axis=0),
                np.count_nonzero(X == G, axis=0),
                np.count_nonzero(X == T, axis=0))

    a0,c0,g0,t0 = counts_acgt(X0)
    a1,c1,g1,t1 = counts_acgt(X1)

    # "Fixed to a single unambiguous base" masks per group
    fixed0 = (a0 == n0) | (c0 == n0) | (g0 == n0) | (t0 == n0)
    fixed1 = (a1 == n1) | (c1 == n1) | (g1 == n1) | (t1 == n1)

    # Determine which base each group is fixed to (where fixed)
    base_idx0 = np.full(m, -1, dtype=np.int8)
    base_idx1 = np.full(m, -1, dtype=np.int8)

    # Order A=0,C=1,G=2,T=3
    for idx, cnt in enumerate([a0, c0, g0, t0]):
        base_idx0[(cnt == n0)] = idx
    for idx, cnt in enumerate([a1, c1, g1, t1]):
        base_idx1[(cnt == n1)] = idx

    # Sites with both groups fixed and to different bases
    both_fixed = fixed0 & fixed1
    mask_diff = both_fixed & (base_idx0 != base_idx1)

    # Map base index back to ASCII
    idx_to_ascii = np.array([A, C, G, T], dtype=np.uint8)
    base_dir = np.where(mask_diff, idx_to_ascii[np.clip(base_idx0, 0, 3)], 0)
    base_inv = np.where(mask_diff, idx_to_ascii[np.clip(base_idx1, 0, 3)], 0)

    return mask_diff, base_dir, base_inv

# ===========================
# Main
# ===========================
def main():
    inv_info_path = "inv_info.tsv"
    triplet_to_cons = load_inv_info_exact(inv_info_path)

    # Discover .phy CDS files
    all_phy = sorted([f for f in os.listdir(".") if f.endswith(".phy") and os.path.isfile(f)])
    parsed: List[Dict] = []
    it = tqdm(all_phy, desc="Parsing filenames") if HAVE_TQDM else all_phy
    for fn in it:
        p = parse_cds_filename(fn)
        if p:
            parsed.append(p)

    if not parsed:
        print("No CDS .phy files matched the strict filename pattern; nothing to do.")
        return 0

    # Keep only CDS with EXACT (chr, inv_start, inv_end) present in inv_info.tsv & consensus in {0,1}
    kept: List[Dict] = []
    for rec in parsed:
        key = (rec["chr"], rec["inv_start"], rec["inv_end"])
        if key in triplet_to_cons:
            rec["_consensus"] = triplet_to_cons[key]
            kept.append(rec)

    if not kept:
        print("No CDS .phy files matched an exact (chr,start,end) present in inv_info.tsv with consensus ∈ {0,1}.")
        return 0

    # Index by (gene_name, inv_id, group) and enforce single file per side
    by_key_group = defaultdict(dict)  # (gene_name, inv_id) -> {0: rec0, 1: rec1}
    dup_violations = []
    for rec in kept:
        inv_id = f"{rec['chr']}:{rec['inv_start']}-{rec['inv_end']}"
        key = (rec["gene_name"], inv_id)
        grp = rec["phy_group"]
        if grp in by_key_group[key]:
            dup_violations.append((rec["gene_name"], inv_id, grp, by_key_group[key][grp]["filename"], rec["filename"]))
        else:
            by_key_group[key][grp] = rec

    if dup_violations:
        print("ASSERTION FAILED: Multiple CDS files for the same (gene, inversion, group). Please deduplicate upstream.")
        for (g, inv, grp, f_prev, f_new) in dup_violations[:10]:
            print(f"  gene={g} inv={inv} group={'Inverted' if grp==1 else 'Direct'} :: {f_prev}  AND  {f_new}")
        sys.exit(1)

    # Prepare outputs
    per_site_rows = []   # gene_inversion_fixed_differences.tsv
    summary_rows  = []   # fixed_diff_summary.tsv

    # Iterate gene × inversion keys that have BOTH groups
    items = sorted(by_key_group.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    it2 = tqdm(items, desc="Gene×Inversion", unit="pair") if HAVE_TQDM else items
    n_pairs = 0
    n_with_diffs = 0

    for (gene_name, inv_id), groups in it2:
        if not (0 in groups and 1 in groups):
            # print a terse skip
            continue

        rec0 = groups[0]
        rec1 = groups[1]
        cons = rec0["_consensus"]  # exact same inv_id so consensus identical

        # Read PHYLIP sequences
        try:
            pairs0 = read_phylip_sequences_strict(rec0["filename"])
            pairs1 = read_phylip_sequences_strict(rec1["filename"])
        except Exception as e:
            print(f"SKIP {gene_name} @ {inv_id}: PHYLIP parse error -> {e}")
            continue

        names0 = [n for n,_ in pairs0]; seqs0 = [s for _,s in pairs0]
        names1 = [n for n,_ in pairs1]; seqs1 = [s for _,s in pairs1]

        if len(seqs0) == 0 or len(seqs1) == 0:
            print(f"SKIP {gene_name} @ {inv_id}: one of the groups has zero sequences.")
            continue

        # Must have equal length between groups
        L0 = len(seqs0[0]); L1 = len(seqs1[0])
        if any(len(s) != L0 for s in seqs0) or any(len(s) != L1 for s in seqs1):
            print(f"SKIP {gene_name} @ {inv_id}: unequal sequence lengths within a group.")
            continue
        if L0 != L1:
            print(f"SKIP {gene_name} @ {inv_id}: different alignment lengths between groups (dir={L0}, inv={L1}).")
            continue

        # Compute fixed differences (vectorized)
        try:
            mask_diff, base_dir, base_inv = fixed_differences_between_groups(seqs0, seqs1)
        except Exception as e:
            print(f"SKIP {gene_name} @ {inv_id}: diff compute error -> {e}")
            continue

        n_pairs += 1
        pos_idx = np.nonzero(mask_diff)[0]
        n_fixed = int(pos_idx.size)
        k_dir = len(seqs0)
        k_inv = len(seqs1)

        # Print report for this gene × inversion
        if n_fixed > 0:
            n_with_diffs += 1
            print(f"\nGENE: {gene_name}  @ {inv_id}  [{consensus_label(cons)}]")
            print(f"  haplotypes: Direct={k_dir}, Inverted={k_inv}")
            print(f"  fixed differences (n={n_fixed}):")
            for j in pos_idx:
                # CDS positions reported 1-based
                cds_pos = int(j + 1)
                d_b = chr(int(base_dir[j]))
                i_b = chr(int(base_inv[j]))
                print(f"    cds_pos={cds_pos:<6}  Direct={d_b}  Inverted={i_b}")

        # Collect rows
        for j in pos_idx:
            per_site_rows.append({
                "gene_name": gene_name,
                "transcript_id": rec0["transcript_id"],  # same inv_id scope; any of the two records carries tx id
                "inv_id": inv_id,
                "consensus": cons,
                "consensus_label": consensus_label(cons),
                "n_direct": k_dir,
                "n_inverted": k_inv,
                "cds_pos_1based": int(j + 1),
                "direct_allele": chr(int(base_dir[j])),
                "inverted_allele": chr(int(base_inv[j]))
            })

        summary_rows.append({
            "gene_name": gene_name,
            "transcript_id": rec0["transcript_id"],
            "inv_id": inv_id,
            "consensus": cons,
            "consensus_label": consensus_label(cons),
            "n_direct": k_dir,
            "n_inverted": k_inv,
            "n_fixed_differences": n_fixed
        })

    # Write outputs
    per_site_out = "gene_inversion_fixed_differences.tsv"
    summary_out  = "fixed_diff_summary.tsv"

    with open(per_site_out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["gene_name","transcript_id","inv_id",
                    "consensus","consensus_label","n_direct","n_inverted",
                    "cds_pos_1based","direct_allele","inverted_allele"])
        for row in per_site_rows:
            w.writerow([
                row["gene_name"], row.get("transcript_id",""),
                row["inv_id"],
                row["consensus"], row["consensus_label"],
                row["n_direct"], row["n_inverted"],
                row["cds_pos_1based"], row["direct_allele"], row["inverted_allele"]
            ])

    with open(summary_out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["gene_name","transcript_id","inv_id",
                    "consensus","consensus_label","n_direct","n_inverted",
                    "n_fixed_differences"])
        for row in summary_rows:
            w.writerow([
                row["gene_name"], row.get("transcript_id",""),
                row["inv_id"],
                row["consensus"], row["consensus_label"],
                row["n_direct"], row["n_inverted"],
                row["n_fixed_differences"]
            ])

    print("\n=======================================")
    print("Summary:")
    print(f"  gene × inversion pairs evaluated: {n_pairs}")
    print(f"  with ≥1 fixed difference:         {n_with_diffs}")
    print(f"Wrote: {per_site_out}")
    print(f"Wrote: {summary_out}")
    print("Done.")
    return 0

# ===========================
# Entrypoint
# ===========================
if __name__ == "__main__":
    sys.exit(main())
