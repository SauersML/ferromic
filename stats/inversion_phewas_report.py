from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Hard-coded regions of interest (chrom:start-end)
# ---------------------------------------------------------------------------
REGION_STRINGS: Sequence[str] = (
    "chr1:13104252-13122521",
    "chr1:248128773-248428856",
    "chr1:25324280-25369060",
    "chr1:81650508-81707447",
    "chr10:15742803-15760427",
    "chr10:46983451-47468232",
    "chr10:79542901-80217413",
    "chr11:18249084-18265819",
    "chr11:50154999-50324102",
    "chr12:131312081-131644839",
    "chr12:46896694-46915975",
    "chr12:75713180-75731794",
    "chr13:53902573-54093373",
    "chr13:64598737-64616218",
    "chr14:85872200-85889269",
    "chr15:30618103-32153204",
    "chr15:84373375-84416696",
    "chr16:15384482-16276146",
    "chr16:16721273-18073542",
    "chr16:28471892-28637651",
    "chr16:75206214-75222748",
    "chr17:45585159-46292045",
    "chr2:130138212-131200602",
    "chr2:130138212-131530534",
    "chr2:45065588-45081777",
    "chr2:87995841-88180536",
    "chr2:91832040-92012663",
    "chr2:95800191-96024403",
    "chr20:47841016-47888238",
    "chr21:13987773-14056796",
    "chr21:19877734-20070170",
    "chr21:43821742-43835116",
    "chr3:195749463-195980207",
    "chr5:112179911-112236600",
    "chr5:124425560-124546657",
    "chr5:177946129-178003952",
    "chr6:141866310-141898728",
    "chr6:167209001-167357782",
    "chr6:76109081-76158474",
    "chr7:33990735-34010143",
    "chr7:54234014-54308393",
    "chr7:57835188-58119653",
    "chr7:5989046-6735643",
    "chr7:60911891-61578023",
    "chr7:65219157-65531823",
    "chr7:70961198-70973901",
    "chr7:73113989-74799029",
    "chr7:74869950-75058098",
    "chr7:97445822-97459601",
    "chr8:2343351-2378385",
    "chr8:7301024-12598379",
    "chr9:87942696-88110274",
    "chrX:103989434-104049428",
    "chrX:149599490-149655967",
    "chrX:149681035-149722249",
    "chrX:153149748-153250226",
    "chrX:154347246-154384867",
    "chrX:154591327-154613096",
    "chrX:155386727-155453982",
    "chrX:52077120-52176974",
    "chrX:72997772-73077479",
    "chr1:149843518-149850293",
    "chr11:71571819-71576724",
    "chr13:48199250-48206639",
    "chr14:60606118-60611659",
    "chr15:23345459-28389868",
    "chr16:35946683-35952382",
    "chr2:87987171-111255403",
    "chr20:24397726-24406144",
    "chr3:75454477-75455715",
    "chr4:33098066-33104924",
    "chr5:179643529-179651253",
    "chr6:123106049-123109821",
    "chr9:85797070-85801040",
    "chr1:108310642-108383736",
    "chr1:144376209-144600799",
    "chr10:46144703-46204851",
    "chr10:55007431-55013434",
    "chr10:65465445-65470067",
    "chr11:89920623-89923848",
    "chr13:52310302-52320954",
    "chr15:23295576-23318191",
    "chr16:14954790-15100859",
    "chr17:3082008-3221835",
    "chr17:16823490-18384190",
    "chr17:45573556-45585158",
    "chr20:29257746-29419750",
    "chr5:71009586-71093997",
    "chr7:62290674-62363143",
    "chr7:62408486-62456444",
    "chr9:12004038-12074552",
    "chrX:112504686-112516959",
    "chrX:152729753-152738707",
)

# ---------------------------------------------------------------------------
# External service configuration
# ---------------------------------------------------------------------------
DEFAULT_HEADERS = {
    "User-Agent": "ferromic-inversion-phewas/0.1 (+https://github.com/SauersML/ferromic)",
    "Accept": "application/json",
}
ENSEMBL_BASE = "https://rest.ensembl.org"
MYGENE_BASE = "https://mygene.info/v3"
MYVARIANT_BASE = "https://myvariant.info/v1"
GWAS_BASE = "https://www.ebi.ac.uk/gwas/rest/api"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BALANCED_TABLE_PATH = (REPO_ROOT / "data" / "balanced_recurrence_results.tsv").resolve()
OUTPUT_JSON_PATH = (SCRIPT_DIR / "inversion_phewas_report.json").resolve()
LD_POPULATION = "1000GENOMES:phase_3:EUR"
GWAS_PAGE_SIZE = 200
ASSOCIATION_PAGE_SIZE = 200
MAX_PROXY_SNPS = 1
MAX_GWAS_SNPS = 200
ASSOCIATION_WORKERS = 8
LD_WORKERS = 8

NONSYNONYMOUS_TERMS: Tuple[str, ...] = (
    "missense_variant",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "start_gained",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "frameshift_variant",
    "transcript_ablation",
    "transcript_amplification",
    "inframe_insertion",
    "inframe_deletion",
)

# ---------------------------------------------------------------------------
# Dataclasses and parsing helpers
# ---------------------------------------------------------------------------
@dataclass
class Region:
    label: str
    chrom: str
    chrom_no_prefix: str
    start: int
    end: int

    @property
    def ensembl_region(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}"

    @property
    def gwas_chrom(self) -> str:
        return self.chrom_no_prefix

    @classmethod
    def from_string(cls, text: str) -> "Region":
        chrom_part, coord_part = text.split(":", 1)
        start_text, end_text = coord_part.split("-", 1)
        start = int(start_text.replace(",", ""))
        end = int(end_text.replace(",", ""))
        chrom_norm = normalize_chrom(chrom_part)
        chrom_no_prefix = chrom_norm[3:]
        return cls(
            label=f"{chrom_norm}:{start}-{end}",
            chrom=chrom_norm,
            chrom_no_prefix=chrom_no_prefix,
            start=start,
            end=end,
        )


def normalize_chrom(value: str) -> str:
    text = value.strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    text = text.upper()
    if text == "M":
        text = "MT"
    return f"chr{text}"


@dataclass
class GeneRecord:
    ensembl_id: str
    symbol: str
    start: int
    end: int
    strand: int
    description: Optional[str]
    summary: Optional[str]
    has_nonsynonymous: bool
    nonsynonymous_variant_count: int


@dataclass
class GWASSignal:
    rsid: str
    pvalue: Optional[float]
    traits: List[str]
    study_accession: Optional[str]
    author_reported_genes: List[str]
    mapped_genes: List[str]
    r2_with_inversion: Optional[float]
    r2_population: Optional[str]
    inversion_proxy_rsid: Optional[str]


@dataclass
class RegionReport:
    region: Region
    inversion_id: Optional[str]
    genes: List[GeneRecord]
    gwas_signals: List[GWASSignal]
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "region": self.region.label,
            "chrom": self.region.chrom,
            "start": self.region.start,
            "end": self.region.end,
            "inversion_id": self.inversion_id,
            "genes": [asdict(g) for g in self.genes],
            "gwas_signals": [asdict(s) for s in self.gwas_signals],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Simple in-memory caches to avoid redundant API calls
# ---------------------------------------------------------------------------
_gene_summary_cache: Dict[str, Optional[str]] = {}
_gene_nonsyn_cache: Dict[str, Tuple[bool, int]] = {}
_rsid_cache: Dict[Tuple[str, int], Optional[str]] = {}
_ld_cache: Dict[Tuple[str, str, str], Optional[float]] = {}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def fetch_json(url: str, label: str, *, retries: int = 3, sleep: float = 0.0, timeout: float = 30.0) -> Optional[dict]:
    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            if sleep:
                time.sleep(sleep)
            return data
        except urllib.error.HTTPError as err:  # type: ignore[union-attr]
            last_error = err
            if err.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(sleep * (2 ** attempt))
                continue
            print(f"[WARN] {label}: HTTP {err.code} {err.reason} ({url})", file=sys.stderr)
            return None
        except Exception as exc:  # pragma: no cover - network errors
            last_error = exc
            if attempt < retries:
                time.sleep(sleep * (2 ** attempt))
                continue
            print(f"[WARN] {label}: {exc} ({url})", file=sys.stderr)
            return None
    if last_error:
        print(f"[WARN] {label}: giving up after {retries} attempts ({url})", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Gene annotation helpers
# ---------------------------------------------------------------------------
def get_gene_summary(ensembl_id: str) -> Optional[str]:
    if ensembl_id in _gene_summary_cache:
        return _gene_summary_cache[ensembl_id]
    encoded = urllib.parse.quote(ensembl_id, safe="")
    url = f"{MYGENE_BASE}/gene/{encoded}?fields=summary"
    data = fetch_json(url, f"mygene:{ensembl_id}", retries=2)
    summary = None
    if data:
        summary = data.get("summary")
    _gene_summary_cache[ensembl_id] = summary
    return summary


def check_gene_nonsynonymous(symbol: str) -> Tuple[bool, int]:
    key = symbol.upper()
    if key in _gene_nonsyn_cache:
        return _gene_nonsyn_cache[key]
    clauses = [f"snpeff.ann.effect:{term}" for term in NONSYNONYMOUS_TERMS]
    query = f"dbnsfp.genename:{symbol} AND (" + " OR ".join(clauses) + ")"
    params = urllib.parse.urlencode({"q": query, "size": 0})
    url = f"{MYVARIANT_BASE}/query?{params}"
    data = fetch_json(url, f"myvariant:{symbol}", retries=2)
    total = int(data.get("total", 0)) if data else 0
    result = (total > 0, total)
    _gene_nonsyn_cache[key] = result
    return result


def gather_genes(region: Region) -> List[GeneRecord]:
    url = f"{ENSEMBL_BASE}/overlap/region/human/{region.ensembl_region}?feature=gene;content-type=application/json"
    data = fetch_json(url, f"ensembl-genes:{region.label}")
    genes: List[GeneRecord] = []
    if not data:
        return genes
    for entry in data:
        gene_id = entry.get("gene_id") or entry.get("id")
        if not gene_id:
            continue
        symbol = entry.get("external_name") or gene_id
        try:
            start = int(entry.get("start", 0))
            end = int(entry.get("end", 0))
            strand = int(entry.get("strand", 0))
        except Exception:
            start, end, strand = 0, 0, 0
        summary = get_gene_summary(gene_id)
        has_nonsyn, nonsyn_count = check_gene_nonsynonymous(symbol)
        genes.append(
            GeneRecord(
                ensembl_id=gene_id,
                symbol=symbol,
                start=start,
                end=end,
                strand=strand,
                description=entry.get("description"),
                summary=summary,
                has_nonsynonymous=has_nonsyn,
                nonsynonymous_variant_count=nonsyn_count,
            )
        )
    genes.sort(key=lambda g: g.start)
    return genes


# ---------------------------------------------------------------------------
# Inversion ID and proxy SNP helpers
# ---------------------------------------------------------------------------
def load_inversion_lookup(path: Path) -> Dict[Tuple[str, int, int], str]:
    mapping: Dict[Tuple[str, int, int], str] = {}
    if not path.exists():
        return mapping
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            chrom_raw = row.get("Chromosome")
            start_raw = row.get("Start")
            end_raw = row.get("End")
            inv_id = (row.get("Inversion_ID") or "").strip()
            if not (chrom_raw and start_raw and end_raw and inv_id):
                continue
            chrom = normalize_chrom(chrom_raw)
            try:
                start = int(str(start_raw).replace(",", ""))
                end = int(str(end_raw).replace(",", ""))
            except ValueError:
                continue
            mapping[(chrom.lower(), start, end)] = inv_id
    return mapping


def load_inversion_proxy_snps(inv_id: str) -> Tuple[List[Tuple[str, int, Optional[str]]], int]:
    path = Path("data") / f"{inv_id}.snps.json"
    if not path.exists():
        return [], 0
    try:
        with path.open() as fh:
            entries = json.load(fh)
    except Exception as exc:  # pragma: no cover - malformed input
        print(f"[WARN] Failed to read {path}: {exc}", file=sys.stderr)
        return [], 0
    result: List[Tuple[str, int, Optional[str]]] = []
    total_available = len(entries)
    for item in entries[:MAX_PROXY_SNPS]:
        snp_id = str(item.get("id", ""))
        pos = item.get("pos")
        eff = item.get("effect_allele")
        if pos is None:
            continue
        chrom_part = snp_id.split(":", 1)[0] if snp_id else ""
        chrom = chrom_part or ""
        result.append((chrom, int(pos), eff if isinstance(eff, str) else None))
    return result, total_available


def lookup_rsid(chrom: str, pos: int) -> Optional[str]:
    key = (chrom, pos)
    if key in _rsid_cache:
        return _rsid_cache[key]
    if chrom.lower().startswith("chr"):
        query_chrom = chrom[3:]
    else:
        query_chrom = chrom
    region = f"{query_chrom}:{pos}-{pos}"
    url = f"{ENSEMBL_BASE}/overlap/region/human/{region}?feature=variation;content-type=application/json"
    data = fetch_json(url, f"rsid:{region}")
    rsid = None
    if data:
        for item in data:
            rsid = item.get("id")
            if rsid:
                break
    _rsid_cache[key] = rsid
    return rsid


def compute_ld(rsid1: str, rsid2: str, population: str) -> Optional[float]:
    key = tuple(sorted((rsid1, rsid2))) + (population,)
    if key in _ld_cache:
        return _ld_cache[key]
    url = (
        f"{ENSEMBL_BASE}/ld/human/pairwise/{rsid1}/{rsid2}?population="
        f"{urllib.parse.quote(population, safe='')};content-type=application/json"
    )
    data = fetch_json(url, f"ld:{rsid1}-{rsid2}", retries=2)
    r2 = None
    if data:
        try:
            if isinstance(data, list) and data:
                r2 = float(data[0].get("r2"))
        except (TypeError, ValueError):
            r2 = None
    _ld_cache[key] = r2
    return r2


def best_ld_for_rsid(
    rsid: str, proxy_rsids: Sequence[str], population: str
) -> Tuple[Optional[float], Optional[str]]:
    best_r2: Optional[float] = None
    best_proxy: Optional[str] = None
    for proxy in proxy_rsids:
        ld_val = compute_ld(proxy, rsid, population)
        if ld_val is None:
            continue
        if best_r2 is None or ld_val > best_r2:
            best_r2 = ld_val
            best_proxy = proxy
    return best_r2, best_proxy


def compute_best_ld_for_rsids(
    rsids: Sequence[str], proxy_rsids: Sequence[str], population: str
) -> Dict[str, Tuple[Optional[float], Optional[str]]]:
    results: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
    if not rsids or not proxy_rsids:
        return results
    with ThreadPoolExecutor(max_workers=LD_WORKERS) as pool:
        future_map = {
            pool.submit(best_ld_for_rsid, rsid, proxy_rsids, population): rsid
            for rsid in rsids
        }
        for future in as_completed(future_map):
            rsid = future_map[future]
            try:
                results[rsid] = future.result()
            except Exception as exc:  # pragma: no cover - network errors
                print(f"[WARN] Failed to compute LD for {rsid}: {exc}", file=sys.stderr)
                results[rsid] = (None, None)
    return results


# ---------------------------------------------------------------------------
# GWAS helpers
# ---------------------------------------------------------------------------
def fetch_region_gwas_snps(region: Region) -> Tuple[List[dict], int]:
    results: List[dict] = []
    total_elements = 0
    base = f"{GWAS_BASE}/singleNucleotidePolymorphisms/search/findByChromBpLocationRange"
    params = {
        "chrom": region.gwas_chrom,
        "bpStart": region.start,
        "bpEnd": region.end,
        "size": GWAS_PAGE_SIZE,
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"
    while url:
        data = fetch_json(url, f"gwas-snps:{region.label}")
        if not data:
            break
        snps = data.get("_embedded", {}).get("singleNucleotidePolymorphisms", [])
        results.extend(snps)
        page = data.get("page", {})
        try:
            total_elements = max(total_elements, int(page.get("totalElements", 0)))
        except (TypeError, ValueError):
            total_elements = max(total_elements, len(results))
        if len(results) >= MAX_GWAS_SNPS:
            results = results[:MAX_GWAS_SNPS]
            break
        next_link = data.get("_links", {}).get("next", {}).get("href")
        url = next_link
    if not total_elements:
        total_elements = len(results)
    return results, total_elements


def fetch_associations_for_snp(rsid: str) -> List[dict]:
    results: List[dict] = []
    url = f"{GWAS_BASE}/singleNucleotidePolymorphisms/{rsid}/associations?size={ASSOCIATION_PAGE_SIZE}"
    while url:
        data = fetch_json(url, f"gwas-assoc:{rsid}")
        if not data:
            break
        results.extend(data.get("_embedded", {}).get("associations", []))
        url = data.get("_links", {}).get("next", {}).get("href")
    return results


def fetch_associations_for_rsids(rsids: Sequence[str]) -> Dict[str, List[dict]]:
    results: Dict[str, List[dict]] = {}
    if not rsids:
        return results
    with ThreadPoolExecutor(max_workers=ASSOCIATION_WORKERS) as pool:
        future_map = {pool.submit(fetch_associations_for_snp, rsid): rsid for rsid in rsids}
        for future in as_completed(future_map):
            rsid = future_map[future]
            try:
                results[rsid] = future.result()
            except Exception as exc:  # pragma: no cover - network errors
                print(f"[WARN] Failed to fetch associations for {rsid}: {exc}", file=sys.stderr)
                results[rsid] = []
    return results


# ---------------------------------------------------------------------------
# Main aggregation routine per region
# ---------------------------------------------------------------------------
def build_region_report(
    region: Region,
    inversion_lookup: Dict[Tuple[str, int, int], str],
    *,
    ld_population: str,
) -> RegionReport:
    key = (region.chrom.lower(), region.start, region.end)
    inversion_id = inversion_lookup.get(key)
    notes: List[str] = []

    genes = gather_genes(region)
    if not genes:
        notes.append("No genes returned by Ensembl for this interval.")

    proxy_rsids: List[str] = []
    if inversion_id:
        proxies, total_proxy_count = load_inversion_proxy_snps(inversion_id)
        for chrom, pos, _allele in proxies:
            if not chrom:
                chrom = region.gwas_chrom
            rsid = lookup_rsid(chrom, pos)
            if rsid:
                proxy_rsids.append(rsid)
        proxy_rsids = sorted(set(proxy_rsids))
        if not proxy_rsids:
            notes.append("No local proxy SNPs with RSIDs found for LD computation.")
        elif total_proxy_count > len(proxies):
            notes.append(
                f"Using first {len(proxies)} of {total_proxy_count} proxy SNPs for LD calculations."
            )
    else:
        notes.append("Inversion ID not found in balanced_recurrence_results.tsv.")

    gwas_signals: List[GWASSignal] = []
    snp_records, total_snp_count = fetch_region_gwas_snps(region)
    if total_snp_count > len(snp_records):
        notes.append(
            f"Retrieved first {len(snp_records)} of {total_snp_count} GWAS SNPs for this region."
        )
    rsids = [record.get("rsId") for record in snp_records if record.get("rsId")]
    associations_map = fetch_associations_for_rsids(rsids)
    ld_results = compute_best_ld_for_rsids(rsids, proxy_rsids, ld_population)
    for snp_entry in snp_records:
        rsid = snp_entry.get("rsId")
        if not rsid:
            continue
        associations = associations_map.get(rsid, [])
        for assoc in associations:
            traits: List[str] = []
            author_genes: List[str] = []
            mapped_genes: List[str] = []
            for locus in assoc.get("loci", []) or []:
                for gene in locus.get("authorReportedGenes", []) or []:
                    name = gene.get("geneName")
                    if name:
                        author_genes.append(name)
                for m in locus.get("mappedGenes", []) or []:
                    name = m.get("geneName")
                    if name:
                        mapped_genes.append(name)
            pvalue = assoc.get("pvalue")
            if pvalue is None:
                mant = assoc.get("pvalueMantissa")
                exp = assoc.get("pvalueExponent")
                if mant is not None and exp is not None:
                    try:
                        pvalue = float(mant) * (10 ** int(exp))
                    except Exception:
                        pvalue = None
            study_accession = None
            best_r2, best_proxy = ld_results.get(rsid, (None, None))
            gwas_signals.append(
                GWASSignal(
                    rsid=rsid,
                    pvalue=pvalue,
                    traits=traits,
                    study_accession=study_accession,
                    author_reported_genes=sorted(set(author_genes)),
                    mapped_genes=sorted(set(mapped_genes)),
                    r2_with_inversion=best_r2,
                    r2_population=ld_population if best_r2 is not None else None,
                    inversion_proxy_rsid=best_proxy,
                )
            )
    if not gwas_signals:
        notes.append("No GWAS catalog signals retrieved for this interval.")

    return RegionReport(
        region=region,
        inversion_id=inversion_id,
        genes=genes,
        gwas_signals=gwas_signals,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
def main() -> int:
    inversion_lookup = load_inversion_lookup(BALANCED_TABLE_PATH)
    reports: List[RegionReport] = []
    for region_text in REGION_STRINGS:
        region = Region.from_string(region_text)
        report = build_region_report(
            region,
            inversion_lookup,
            ld_population=LD_POPULATION,
        )
        reports.append(report)
        print(
            f"Processed {region.label}: genes={len(report.genes)} GWAS_hits={len(report.gwas_signals)}",
            file=sys.stderr,
        )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "region_count": len(reports),
        "regions": [r.to_dict() for r in reports],
    }

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON_PATH.open("w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Report written to {OUTPUT_JSON_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
