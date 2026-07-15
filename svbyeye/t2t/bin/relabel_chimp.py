#!/usr/bin/env python3
"""Rewrite chimpanzee FASTA headers to clean, chromosome-named query IDs.

Why: minimap2 uses the first whitespace token of each FASTA header as the query
name. The T2T chimp (NCBI) headers are GenBank accessions (e.g. 'CM054434.1 ...
chromosome 1'), which are opaque as plot labels and do not match the pipeline's
chimp detector (`grepl("panTro|chimp", ...)`). We map every sequence to a stable
id '<prefix>_<chrom>' (e.g. chimpT2T_chr1). The prefix carries 'chimp' so the
detector fires; the chromosome name makes plot rows readable.

Usage:
  relabel_chimp.py --in in.fa.gz --out out.fa.gz --prefix chimpT2T \
      [--report assembly_report.txt]   # NCBI assembly_report for accn->chrom map
  relabel_chimp.py --in panTro6.fa.gz --out out.fa.gz --prefix panTro6
      # UCSC names are already chrN; we just prefix them.
"""
import argparse, gzip, sys, re

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", dest="out", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--report", default=None,
                help="NCBI assembly_report.txt (GenBank-Accn -> chrom name)")
ap.add_argument("--map-out", default=None, help="write old->new name TSV")
args = ap.parse_args()

# Build accession -> chromosome-name map from an NCBI assembly report if given.
accn2chrom = {}
if args.report:
    with open(args.report) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            # cols: Sequence-Name Sequence-Role Assigned-Molecule
            #       Assigned-Molecule-Location/Type GenBank-Accn Relationship
            #       RefSeq-Accn Assembly-Unit Sequence-Length UCSC-style-name
            if len(c) < 10:
                continue
            seqname, role, mol = c[0], c[1], c[2]
            genbank = c[4]; ucsc = c[9]
            if role == "assembled-molecule":
                # Assigned-Molecule is like '1','2A','X','Y','MT'
                chrom = f"chr{mol}"
            elif ucsc and ucsc != "na":
                chrom = ucsc
            else:
                chrom = seqname
            accn2chrom[genbank] = chrom

def opener(path, mode):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)

seen = {}
def clean_name(raw_first):
    # raw_first = first whitespace token of the header (no '>')
    if raw_first in accn2chrom:
        base = accn2chrom[raw_first]
    else:
        base = raw_first
    # strip any existing 'chr' duplication and non-safe chars
    base = re.sub(r"[^A-Za-z0-9_.]", "_", base)
    name = f"{args.prefix}_{base}"
    if name in seen:
        seen[name] += 1
        name = f"{name}.{seen[name]}"
    else:
        seen[name] = 0
    return name

nmap = []
n = 0
with opener(args.inp, "rt") as fin, opener(args.out, "wt") as fout:
    for line in fin:
        if line.startswith(">"):
            raw_first = line[1:].split()[0]
            new = clean_name(raw_first)
            nmap.append((raw_first, new))
            fout.write(f">{new}\n")
            n += 1
        else:
            fout.write(line)

if args.map_out:
    with open(args.map_out, "w") as mh:
        mh.write("old_name\tnew_name\n")
        for o, nw in nmap:
            mh.write(f"{o}\t{nw}\n")

sys.stderr.write(f"relabelled {n} sequences with prefix {args.prefix}\n")
for o, nw in nmap[:30]:
    sys.stderr.write(f"  {o} -> {nw}\n")
