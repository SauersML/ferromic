"""Model-free structural anchor: a breakpoint that falls inside a protein-coding gene
truncates/rearranges that gene by the rearrangement itself — SNVs cannot create a
breakpoint junction, so the consequence is structural by construction. Cross-reference the
measured QTL gene (exp #8) to flag where the *measured* signal is breakpoint-mediated.

Uses GENCODE protein-coding gene spans (data/gene_spans built from full gene models here)
and the exp #8 QTL top genes. No model, no scoring."""
import gzip, json, os, re, sys

try:
    from .. import paths as _paths
    from . import _data
except ImportError:
    import paths as _paths  # type: ignore
    import _data  # type: ignore

_RESULTS = os.environ.get("STRUCTURAL_RESULTS_DIR", _data.RESULTS_DIR)
GENCODE = _paths.resolve("gencode_gtf")
targets = {t["locus"]: t for t in _data.load_data("targets_master.json")}
analysis = _data.load_data("analysis_loci.json")
chroms = {targets[l]["chrom"] for l in analysis}

# full protein-coding gene models (gene + exon spans) on needed chroms
genes = {c: [] for c in chroms}
name_re = re.compile(r'gene_name "([^"]+)"'); type_re = re.compile(r'gene_type "([^"]+)"')
with gzip.open(GENCODE, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.split("\t")
        if p[2] != "gene" or p[0] not in genes:
            continue
        gt = type_re.search(p[8])
        if not gt or gt.group(1) != "protein_coding":
            continue
        nm = name_re.search(p[8])
        genes[p[0]].append((nm.group(1) if nm else "NA", int(p[3]) - 1, int(p[4])))

def genes_at(chrom, pos, pad=0):
    return [nm for (nm, s, e) in genes[chrom] if s - pad <= pos <= e + pad]

rows = []
for loc in analysis:
    t = targets[loc]; chrom = t["chrom"]; b1 = t["start38"]; b2 = t["end38"]
    gL = genes_at(chrom, b1); gR = genes_at(chrom, b2)
    bp_mediated = sorted(set(gL) | set(gR))
    qtl_gene = t.get("geuv_top_gene") or t.get("gtex_top_gene") or ""
    qtl_is_bp = any(qtl_gene and (qtl_gene == g or qtl_gene in g) for g in bp_mediated)
    rows.append({
        "locus": loc, "chrom": chrom, "bpL": b1, "bpR": b2,
        "genes_at_bpL": gL, "genes_at_bpR": gR,
        "breakpoint_in_gene": bool(bp_mediated),
        "bp_mediated_genes": bp_mediated,
        "measured_qtl_gene": qtl_gene,
        "measured_qtl_gene_is_breakpoint_mediated": bool(qtl_is_bp),
        "recur_consensus": t["recur_consensus"], "measured_any": t["measured_any"],
    })

n_bp = sum(r["breakpoint_in_gene"] for r in rows)
n_qtl_bp = sum(r["measured_qtl_gene_is_breakpoint_mediated"] for r in rows)
json.dump(rows, open(os.path.join(_RESULTS, "structural_anchor.json"), "w"), indent=1)
print(f"loci: {len(rows)}; breakpoint-in-protein-coding-gene: {n_bp}; "
      f"measured-QTL-gene is breakpoint-mediated (structural by construction): {n_qtl_bp}")
for r in rows:
    if r["breakpoint_in_gene"]:
        print(f"  {r['locus']:<28} bpL={r['genes_at_bpL']} bpR={r['genes_at_bpR']} "
              f"qtl={r['measured_qtl_gene']} qtl_bp={r['measured_qtl_gene_is_breakpoint_mediated']}")
print("wrote results/structural_anchor.json")
