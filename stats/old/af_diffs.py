import os
import re
import multiprocessing as mp

# Global settings
OUTPUT_TSV = "inversion_allele_frequency_differences.tsv"

# Only match inversion-style PHYLIP filenames, not gene-style or other junk.
INVERSION_PHY_PATTERN = re.compile(
    r'^inversion_group([01])_([0-9]+)_start([0-9]+)_end([0-9]+)\.phy$'
)

# How many debug examples per pair to print when differences are found
DEBUG_DIFF_EXAMPLES_PER_PAIR = 20


def read_phylip_sequences(path):
    """
    Robust PHYLIP reader with verbose diagnostics.

    Supports:
    - Header: "<n_seq> <n_sites>"
    - Classic PHYLIP: 10-char name field + sequence (no separating whitespace)
    - Relaxed PHYLIP: name and sequence separated by whitespace
    - Interleaved or sequential layouts
    - Interleaved blocks that may or may not repeat names

    Returns:
        list[str]: sequences (A/C/G/T plus possibly other chars), all same length.
                   Length is min(n_sites, shortest parsed sequence).
                   Returns [] on any fatal inconsistency.
    """
    print(f"[READ] Reading PHYLIP file: {path}")

    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    # Skip leading empty lines
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx >= len(lines):
        print(f"[READ][WARN] {path}: file is empty or only whitespace")
        return []

    header_parts = lines[idx].split()
    if len(header_parts) < 2:
        print(f"[READ][WARN] {path}: malformed header line: '{lines[idx]}'")
        return []
    try:
        n_seq = int(header_parts[0])
        n_sites = int(header_parts[1])
    except ValueError:
        print(f"[READ][WARN] {path}: cannot parse n_seq / n_sites from header '{lines[idx]}'")
        return []

    print(f"[READ] {path}: header n_seq={n_seq}, n_sites={n_sites}")
    idx += 1

    # Skip blank lines after header
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    names = []
    seqs = []

    # Parse the first block: up to n_seq non-empty lines
    while idx < len(lines) and len(names) < n_seq:
        raw = lines[idx]
        idx += 1
        if not raw.strip():
            continue

        parts = raw.split()
        if len(parts) >= 2:
            # Relaxed: name whitespace sequence
            name = parts[0]
            seq = "".join(parts[1:])
        else:
            # Classic PHYLIP: first 10 chars = name, rest = sequence (may contain spaces)
            line = raw
            if len(line) <= 10:
                name = line.strip()
                seq = ""
            else:
                name = line[:10].strip()
                seq = line[10:].replace(" ", "")

        names.append(name)
        seqs.append(seq)

    print(f"[READ] {path}: first block parsed {len(names)} names")
    if len(names) != n_seq:
        print(f"[READ][WARN] {path}: expected {n_seq} sequences in first block, got {len(names)}")

    # Interleaved continuation blocks: keep appending until all sequences reach >= n_sites
    # We are deliberately forgiving about blank lines and formats.
    while any(len(s) < n_sites for s in seqs) and idx < len(lines):
        # Skip blank lines between blocks
        while idx < len(lines) and not lines[idx].strip():
            idx += 1

        if idx >= len(lines):
            break

        for i in range(n_seq):
            if idx >= len(lines):
                break
            raw = lines[idx]
            idx += 1

            if not raw.strip():
                # Allow sparse blocks; continue scanning
                continue

            parts = raw.split()
            if len(parts) >= 2:
                # Could be "name seq" or just split seq; treat everything after first token as sequence.
                chunk = "".join(parts[1:])
            else:
                # Could be classic-style continued, or bare sequence.
                line = raw
                if len(line) > 10:
                    # Assume first 10 columns are name (or padding), rest is seq.
                    chunk = line[10:].replace(" ", "")
                else:
                    chunk = line.strip().replace(" ", "")

            if chunk:
                seqs[i] += chunk

    if not seqs:
        print(f"[READ][WARN] {path}: no sequences parsed at all")
        return []

    # Determine effective usable length
    min_len = min(len(s) for s in seqs)
    if min_len == 0:
        print(f"[READ][WARN] {path}: at least one sequence length is 0; min_len=0")
    if min_len < n_sites:
        print(f"[READ][WARN] {path}: shortest seq len={min_len} < n_sites={n_sites}; trimming to {min_len}")

    eff_len = min(min_len, n_sites)

    if eff_len <= 0:
        print(f"[READ][WARN] {path}: effective alignment length {eff_len} <= 0; returning empty")
        return []

    seqs = [s[:eff_len] for s in seqs]

    # Diagnostics: show short summary of first few sequences
    print(f"[READ] {path}: final n_seq={len(seqs)}, eff_len={eff_len}")
    show = min(3, len(seqs))
    for i in range(show):
        prefix = seqs[i][:50]
        seq_name = names[i] if i < len(names) else 'NA'
        seq_len = len(seqs[i])
        print(f"[READ] {path}: seq[{i}] name={seq_name!r} prefix={prefix!r} ... len={seq_len}")

    return seqs


def collect_inversion_pairs():
    """
    Find valid inversion .phy files and pair group0 with group1
    for identical (chrom, start, end).

    Returns:
        list of (chrom, start, end, group0_path, group1_path)
    """
    loci = {}
    total_phy = 0

    print("[COLLECT] Scanning current directory for inversion PHYLIP files")

    for fname in sorted(os.listdir(".")):
        if fname.endswith(".phy"):
            total_phy += 1

        m = INVERSION_PHY_PATTERN.match(fname)
        if m:
            group = m.group(1)     # "0" or "1"
            chrom = m.group(2)
            start = int(m.group(3))
            end = int(m.group(4))
            key = (chrom, start, end)
            slot = loci.setdefault(key, {})
            if group in slot:
                print(f"[COLLECT][WARN] duplicate group{group} for {key}: {slot[group]} vs {fname}")
            slot[group] = fname
        else:
            # Only complain about inversion-like names that fail the strict pattern
            if fname.startswith("inversion_") and fname.endswith(".phy"):
                print(f"[COLLECT] Ignoring non-matching inversion-like .phy: {fname}")

    print(f"[COLLECT] Total .phy files seen: {total_phy}")
    print(f"[COLLECT] Distinct inversion loci keys (any group): {len(loci)}")

    pairs = []
    for (chrom, start, end), groups in sorted(loci.items()):
        g0 = groups.get("0")
        g1 = groups.get("1")
        if g0 and g1:
            pairs.append((chrom, start, end, g0, g1))
        else:
            print(f"[COLLECT][NOTE] Missing pair for {chrom}:{start}-{end} "
                  f"(group0={bool(g0)}, group1={bool(g1)})")

    print(f"[COLLECT] Paired loci with both group0 and group1: {len(pairs)}")
    for p in pairs[:20]:
        chrom, start, end, g0, g1 = p
        print(f"[COLLECT] Pair example: chr{chrom}:{start}-{end} | group0={g0} | group1={g1}")

    if not pairs:
        print("[COLLECT][WARN] No valid group0/group1 inversion pairs found. "
              "Check filename pattern: inversion_group[0|1]_CHR_startXXX_endYYY.phy")

    return pairs


def site_counts(seqs, position):
    """
    Count A/C/T/G at a given alignment column index across a list of sequences.

    Non-ACGT characters are ignored.
    """
    a = c = t = g = 0
    for s in seqs:
        if position >= len(s):
            continue
        base = s[position].upper()
        if base == "A":
            a += 1
        elif base == "C":
            c += 1
        elif base == "T":
            t += 1
        elif base == "G":
            g += 1
    return a, c, t, g


def allele_freqs_differ(dir_counts, inv_counts):
    """
    Compare allele frequencies (not raw counts) between DIRECT (group0) and
    INVERTED (group1) groups at one site.

    Returns True if any of the four bases has a different frequency.
    Uses exact integer cross-multiplication: no floating noise.
    """
    a_dir, c_dir, t_dir, g_dir = dir_counts
    a_inv, c_inv, t_inv, g_inv = inv_counts

    total_dir = a_dir + c_dir + t_dir + g_dir
    total_inv = a_inv + c_inv + t_inv + g_inv

    # Both groups have no informative alleles: no difference
    if total_dir == 0 and total_inv == 0:
        return False

    # One group has data and the other does not: frequencies differ
    if total_dir == 0 or total_inv == 0:
        return True

    # Cross-multiply to compare each base's frequency
    if a_dir * total_inv != a_inv * total_dir:
        return True
    if c_dir * total_inv != c_inv * total_dir:
        return True
    if t_dir * total_inv != t_inv * total_dir:
        return True
    if g_dir * total_inv != g_inv * total_dir:
        return True

    return False


def process_pair(task):
    """
    Process a single inversion locus: one group0 file and one group1 file.

    Input:
        (chrom, start, end, group0_path, group1_path)

    Output:
        list of rows:
        (Chromosome, Start, End,
         A_inv, C_inv, T_inv, G_inv,
         A_dir, C_dir, T_dir, G_dir)
    """
    chrom, start, end, path_dir, path_inv = task

    prefix = f"[PAIR chr{chrom}:{start}-{end}]"
    print(f"{prefix} Starting. group0={path_dir}, group1={path_inv}")

    seqs_dir = read_phylip_sequences(path_dir)
    seqs_inv = read_phylip_sequences(path_inv)

    print(f"{prefix} Parsed group0 seqs={len(seqs_dir)}, group1 seqs={len(seqs_inv)}")

    if not seqs_dir or not seqs_inv:
        print(f"{prefix} SKIP: missing sequences on one side.")
        return []

    max_len_dir = min(len(s) for s in seqs_dir)
    max_len_inv = min(len(s) for s in seqs_inv)
    n_cols = max_len_dir if max_len_dir < max_len_inv else max_len_inv

    print(f"{prefix} min_len_dir={max_len_dir}, min_len_inv={max_len_inv}, using n_cols={n_cols}")

    if n_cols <= 0:
        print(f"{prefix} SKIP: n_cols <= 0.")
        return []

    rows = []
    diff_sites = 0

    for pos in range(n_cols):
        dir_counts = site_counts(seqs_dir, pos)
        inv_counts = site_counts(seqs_inv, pos)

        if allele_freqs_differ(dir_counts, inv_counts):
            a_inv, c_inv, t_inv, g_inv = inv_counts
            a_dir, c_dir, t_dir, g_dir = dir_counts

            # If both groups have zero informative counts, ignore
            if (a_inv + c_inv + t_inv + g_inv) == 0 and (a_dir + c_dir + t_dir + g_dir) == 0:
                continue

            diff_sites += 1

            # Debug a subset of differing sites for this pair
            if diff_sites <= DEBUG_DIFF_EXAMPLES_PER_PAIR:
                print(
                    f"{prefix} DIFF@pos{pos}: "
                    f"inv(A,C,T,G)={inv_counts}, dir(A,C,T,G)={dir_counts}"
                )

            rows.append((
                chrom,
                start,
                end,
                a_inv,
                c_inv,
                t_inv,
                g_inv,
                a_dir,
                c_dir,
                t_dir,
                g_dir,
            ))

    print(f"{prefix} Finished. Differing sites={diff_sites}")
    return rows


def main():
    print("[MAIN] Starting inversion allele frequency comparison")
    pairs = collect_inversion_pairs()

    with open(OUTPUT_TSV, "w") as out:
        header_cols = [
            "Chromosome",
            "Start",
            "End",
            "A_inv_count",
            "C_inv_count",
            "T_inv_count",
            "G_inv_count",
            "A_dir_count",
            "C_dir_count",
            "T_dir_count",
            "G_dir_count",
        ]
        out.write("\t".join(header_cols) + "\n")

        if not pairs:
            print("[MAIN][WARN] No pairs to process. Output TSV will contain only header.")
            return

        cpu_count = mp.cpu_count()
        print(f"[MAIN] Using {cpu_count} CPU cores via multiprocessing")

        # Simple heuristic chunksize so imap_unordered is not too chatty
        chunksize = 1
        if len(pairs) > cpu_count * 4:
            chunksize = max(1, len(pairs) // (cpu_count * 4))

        total_rows = 0

        with mp.Pool(processes=cpu_count) as pool:
            for rows in pool.imap_unordered(process_pair, pairs, chunksize=chunksize):
                for r in rows:
                    (
                        chrom,
                        start,
                        end,
                        a_inv,
                        c_inv,
                        t_inv,
                        g_inv,
                        a_dir,
                        c_dir,
                        t_dir,
                        g_dir,
                    ) = r
                    out.write(
                        f"{chrom}\t{start}\t{end}\t"
                        f"{a_inv}\t{c_inv}\t{t_inv}\t{g_inv}\t"
                        f"{a_dir}\t{c_dir}\t{t_dir}\t{g_dir}\n"
                    )
                    total_rows += 1

        print(f"[MAIN] Done. Wrote {total_rows} rows with differing allele frequencies "
              f"to {OUTPUT_TSV}")
        if total_rows == 0:
            print("[MAIN] Note: no sites with differing allele frequencies were found; "
                  "TSV contains only header.")


if __name__ == "__main__":
    main()
