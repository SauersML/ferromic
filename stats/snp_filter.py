import os, sys, subprocess

BASE = "subset"  # expects subset.bed/bim/fam in cwd
SMFILT = f"{BASE}_smfilt"
VFILT = f"{SMFILT}_vfilt_snp"
OUT_TXT = "major_alleles_99pct_biallelic_noindels.txt"

def exists_all(prefix, exts):
    return all(os.path.exists(f"{prefix}.{e}") for e in exts)

def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def ensure_sample_filter():
    if exists_all(SMFILT, ["bed","bim","fam"]):
        print(f"[skip] found {SMFILT}.bed/bim/fam")
        return
    run(["plink", "--bfile", BASE, "--mind", "0.01", "--make-bed", "--out", SMFILT])

def ensure_variant_filter():
    if exists_all(VFILT, ["bed","bim","fam"]):
        print(f"[skip] found {VFILT}.bed/bim/fam")
        return
    run([
        "plink", "--bfile", SMFILT,
        "--geno", "0.01",                 # present in >=99% of remaining samples
        "--snps-only", "just-acgt",       # exclude any indel (either allele)
        "--biallelic-only", "strict",     # no multi-allelics
        "--make-bed", "--out", VFILT
    ])

def ensure_freq():
    frq = f"{VFILT}.frq"
    if os.path.exists(frq):
        print(f"[skip] found {frq}")
        return
    run(["plink", "--bfile", VFILT, "--freq", "--out", VFILT])

def load_positions(bimp):
    pos = {}
    with open(bimp) as f:
        for line in f:
            p = line.split()
            if len(p) < 6: continue
            chrom, snp, cm, bp = p[0], p[1], p[2], p[3]
            pos[snp] = f"{chrom}:{bp}"
    return pos

def write_major_alleles():
    pos_by_id = load_positions(f"{VFILT}.bim")
    valid = {"A","C","G","T"}
    kept = 0
    total = 0
    with open(f"{VFILT}.frq") as frq, open(OUT_TXT, "w") as out:
        hdr = frq.readline()  # CHR SNP A1 A2 MAF NCHROBS
        for line in frq:
            total += 1
            p = line.split()
            if len(p) < 6: continue
            _, snp, a1, a2, maf_s, _ = p[:6]
            # enforce SNP-only alleles and polymorphic (biallelic with MAF>0)
            if a1 not in valid or a2 not in valid:
                continue
            try:
                maf = float(maf_s)
            except ValueError:
                continue
            if maf <= 0.0:
                continue
            pos = pos_by_id.get(snp)
            if not pos:
                continue
            # Major allele is A2 in PLINK .frq
            out.write(f"{pos} {a2}\n")
            kept += 1
    print(f"Wrote {kept} variants to {OUT_TXT} (from {total} .frq rows)")

def main():
    try:
        ensure_sample_filter()
        ensure_variant_filter()
        ensure_freq()
        write_major_alleles()
        print("Done.")
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"FATAL: command failed with exit code {e.returncode}\n")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
