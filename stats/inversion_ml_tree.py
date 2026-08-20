"""Maximum-likelihood tree for one inversion locus, and the monophyly check.

Why this exists: the envelope test's null represents the arrangement as ONE
branch of the genealogy. That is a claim about the data, and until now nothing
verified it. If the inverted haplotypes are not reciprocally monophyletic, the
one-branch model does not describe the locus and no p-value from it means
anything -- exactly the objection that removed the recurrent loci, applied to
the one locus we kept.

So this builds the actual tree rather than a summary statistic:

  * writes the two arrangement alignments as one PHYLIP file, tips labelled by
    arrangement, non-ACGT collapsed to N
  * runs IQ-TREE with model selection and ultrafast bootstrap
  * reports whether the inverted tips form a clade, with bootstrap support
  * reports the root-to-split depth in substitutions/site, which is the direct
    ML read of the quantity d_cross estimates by counting differences

What it does NOT do is replace the coalescent test. A tree tells you what the
genealogy IS; it cannot tell you how surprising that genealogy is for an allele
at this frequency, which is the whole question. Treat this as the observation
side: a better-measured split depth, plus a check on the premise.

Output: <outdir>/<locus>.{phy,iqtree,treefile,...} + a printed summary.
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cds_selection_intron_control import INV_RE, read_phy  # noqa: E402

VALID = set("ACGT")


def locus_files(workdir, inv_id):
    chrom, se = inv_id.split(":")
    s, e = (int(x) for x in se.split("-"))
    found = {}
    for fn in os.listdir(os.path.join(workdir, "phy_outputs")):
        m = INV_RE.match(fn)
        if m and (m["chrom"], int(m["s"]), int(m["e"])) == (chrom, s, e):
            found[m["grp"]] = os.path.join(workdir, "phy_outputs", fn)
    if set(found) != {"0", "1"}:
        sys.exit(f"{inv_id}: need both arrangement alignments, got "
                 f"{sorted(found)}")
    return found


def write_phylip(seq_dir, seq_inv, path):
    """One alignment, tips tagged DIR_/INV_ so monophyly is readable off names.

    Columns that are not ACGT in every sequence become N: IQ-TREE handles
    ambiguity, but leaving raw placeholder characters in would let the model
    treat them as real states."""
    names, seqs, labels = [], [], []
    for tag, d in (("DIR", seq_dir), ("INV", seq_inv)):
        for nm in sorted(d):
            # phy names are SUPERPOP_POP_SAMPLE_...; keep it short and unique
            short = re.sub(r"[^A-Za-z0-9]", "_", nm)[:40]
            names.append(f"{tag}_{short}")
            seqs.append(d[nm].upper())
            labels.append(tag)
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        sys.exit("alignments differ in length")
    cols_ok = [all(s[j] in VALID for s in seqs) for j in range(L)]
    kept = [j for j, ok in enumerate(cols_ok) if ok]
    clean = ["".join(s[j] for j in kept) for s in seqs]
    with open(path, "w") as fh:
        fh.write(f"{len(clean)} {len(kept)}\n")
        for nm, s in zip(names, clean):
            fh.write(f"{nm:<45s} {s}\n")
    return names, labels, L, len(kept)


def parse_support(treefile, labels, names):
    """Are the INV tips a clade? Newick-agnostic: check every bipartition.

    Splits the tree string into clades by matching parentheses, and for each
    clade compares its tip set with the INV tip set. Returns (is_clade,
    support) where support is the label on that node if present."""
    tree = open(treefile).read().strip()
    inv = {n for n, l in zip(names, labels) if l == "INV"}
    best = (False, None)
    stack = []
    for i, ch in enumerate(tree):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            start = stack.pop()
            clade = tree[start + 1:i]
            tips = set(re.findall(r"[A-Za-z0-9_]+(?=:)", clade))
            tips = {t for t in tips if t in set(names)}
            if not tips:
                continue
            # a clade OR its complement being exactly the INV set both mean the
            # arrangements are reciprocally monophyletic on an unrooted tree
            if tips == inv or tips == set(names) - inv:
                m = re.match(r"\)([0-9.]+)", tree[i:])
                sup = float(m.group(1)) if m else None
                best = (True, sup)
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--inv-id", default="17:45585159-46292045")
    ap.add_argument("--iqtree", required=True, help="path to iqtree2 binary")
    ap.add_argument("--outdir", default="results/mltree")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--boot", type=int, default=1000,
                    help="ultrafast bootstrap replicates")
    a = ap.parse_args()
    os.chdir(a.workdir)
    os.makedirs(a.outdir, exist_ok=True)

    files = locus_files(a.workdir, a.inv_id)
    seq_dir, seq_inv = read_phy(files["0"]), read_phy(files["1"])
    tag = a.inv_id.replace(":", "_").replace("-", "_")
    phy = os.path.join(a.outdir, f"{tag}.phy")
    names, labels, L_raw, L_kept = write_phylip(seq_dir, seq_inv, phy)
    n_inv = labels.count("INV")
    print(f"{a.inv_id}: {len(names)} haplotypes ({n_inv} inverted, "
          f"{len(names) - n_inv} direct); {L_kept} of {L_raw} sites fully "
          f"called")

    cmd = [a.iqtree, "-s", phy, "-m", "MFP", "-B", str(a.boot),
           "-T", str(a.threads), "--prefix", os.path.join(a.outdir, tag),
           "-redo"]
    print("running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:])
        sys.exit(f"iqtree failed: {r.stderr[-2000:]}")

    treefile = os.path.join(a.outdir, f"{tag}.treefile")
    is_clade, support = parse_support(treefile, labels, names)
    print(f"\n=============== {a.inv_id} ===============")
    print(f"inverted haplotypes form a single clade: {is_clade}")
    if is_clade:
        print(f"  bootstrap support for that split: "
              f"{support if support is not None else 'not labelled'}")
        print("  => the one-branch premise of the envelope null HOLDS here")
    else:
        print("  => the arrangements are NOT reciprocally monophyletic; the "
              "one-branch null does not describe this locus")
    model = ""
    iqfile = os.path.join(a.outdir, f"{tag}.iqtree")
    if os.path.exists(iqfile):
        for line in open(iqfile):
            if line.startswith("Best-fit model"):
                model = line.strip()
                break
    if model:
        print(f"  {model}")
    print(f"\ntree: {treefile}")
    print(f"report: {iqfile}")


if __name__ == "__main__":
    main()
