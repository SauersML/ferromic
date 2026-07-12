"""Input-path resolution and provenance recording for the ``functional/`` analyses.

Every analysis reads its large inputs (reference genomes, AlphaMissense, Geuvadis
genotypes/expression, GTEx caches, AlphaGenome per-event scores) from paths that are
*not* committed to this repository. This module centralises where those live so no
analysis script hard-codes an absolute path.

Resolution order for an input ``key``:

1. an explicit path passed on the command line / to the function,
2. the environment variable ``FUNCTIONAL_<KEY>`` (e.g. ``FUNCTIONAL_ALPHAMISSENSE``),
3. ``<FUNCTIONAL_DATA_ROOT>/<default relative path>`` if ``FUNCTIONAL_DATA_ROOT`` is set,
4. otherwise raise, with the documented public source printed.

See ``functional/README.md`` for how to obtain each input.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Input:
    key: str
    relpath: str
    source: str  # human-readable provenance / where to fetch it


# Documented external inputs. `relpath` is relative to FUNCTIONAL_DATA_ROOT.
INPUTS = {
    "reference_fasta": Input(
        "reference_fasta", "reference/GRCh38.primary_assembly.genome.fa",
        "GENCODE GRCh38 primary assembly genome FASTA (release 47).",
    ),
    "gencode_gtf": Input(
        "gencode_gtf", "reference/gencode.v47.annotation.gtf.gz",
        "GENCODE v47 comprehensive gene annotation, GRCh38.",
    ),
    "alphamissense": Input(
        "alphamissense", "reference/AlphaMissense_hg38.tsv.gz",
        "AlphaMissense hg38 predictions (Cheng et al., Science 2023; Zenodo 10.5281/zenodo.8208688).",
    ),
    "clinvar_vcf": Input(
        "clinvar_vcf", "reference/clinvar_GRCh38.vcf.gz",
        "ClinVar GRCh38 VCF (NCBI ClinVar FTP).",
    ),
    "geuvadis_pgen": Input(
        "geuvadis_pgen", "geuvadis/geuvadis.pgen",
        "Geuvadis (E-GEUV-1) LCL genotypes in PLINK2 pgen (hg19); pvar/psam alongside.",
    ),
    "geuvadis_gene_rpkm": Input(
        "geuvadis_gene_rpkm", "geuvadis/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz",
        "Geuvadis gene RPKM matrix (library-depth-normalised, not PEER-corrected), hg19/GENCODE v12.",
    ),
    "geuvadis_junction": Input(
        "geuvadis_junction", "geuvadis_splicing/GD462.JunctionQuantCount.45N.50FN.samplename.resk10.txt.gz",
        "Geuvadis split-read junction counts (LeafCutter phenotype).",
    ),
    "geuvadis_exon": Input(
        "geuvadis_exon", "geuvadis_splicing/GD462.ExonQuantCount.45N.50FN.samplename.resk10.txt.gz",
        "Geuvadis exon PSI matrix.",
    ),
    "geuvadis_transcript": Input(
        "geuvadis_transcript", "geuvadis_splicing/GD462.TrQuantRPKM.50FN.samplename.resk10.txt.gz",
        "Geuvadis transcript-usage RPKM matrix.",
    ),
    "gtex_eqtls": Input(
        "gtex_eqtls", "gtex/gtex_eqtls.tsv",
        "GTEx v10 significant cis-eQTLs at inversion tag SNPs (GTEx portal API).",
    ),
    "alphagenome_scores": Input(
        "alphagenome_scores", "agscore",
        "Per-inversion AlphaGenome signed per-tissue RNA LFC + splice disruption (.npz), one per event.",
    ),
}


def data_root() -> str | None:
    return os.environ.get("FUNCTIONAL_DATA_ROOT")


def resolve(key: str, explicit: str | None = None) -> str:
    """Resolve an input to an absolute path. Raises FileNotFoundError with the
    documented public source if it cannot be found."""
    spec = INPUTS[key]
    if explicit:
        p = explicit
    elif os.environ.get(f"FUNCTIONAL_{key.upper()}"):
        p = os.environ[f"FUNCTIONAL_{key.upper()}"]
    elif data_root():
        p = os.path.join(data_root(), spec.relpath)
    else:
        raise FileNotFoundError(
            f"Input '{key}' not configured. Set FUNCTIONAL_{key.upper()}=<path>, "
            f"or FUNCTIONAL_DATA_ROOT so it resolves to '<root>/{spec.relpath}'. "
            f"Source: {spec.source}"
        )
    if not os.path.exists(p):
        raise FileNotFoundError(f"Input '{key}' resolved to '{p}' which does not exist. Source: {spec.source}")
    return p


def write_provenance(out_path: str, resolved: dict[str, str], extra: dict | None = None) -> None:
    """Record the resolved absolute input paths + a timestamp next to an output table,
    so every committed result is traceable to the exact inputs that produced it."""
    rec = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "inputs": {k: {"path": v, "source": INPUTS[k].source if k in INPUTS else "n/a"} for k, v in resolved.items()},
    }
    if extra:
        rec.update(extra)
    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=2)
