#!/usr/bin/env python3
import os, sys, time, json, math, argparse
import requests
from typing import Dict, Any, List, Tuple, Optional

# ---- config defaults (adjust if their API field names differ) ----
DEFAULT_BASE = "https://compbio.ccia.org.au/splicevardb-api"  # per paper
VAR_ENDPOINT = "/variants"  # guessed from typical REST; adjust if docs differ
PAGE_SIZE = 1000            # tune to API limit
SLEEP = 0.2                 # politeness between pages

def get_session(token: Optional[str] = None) -> requests.Session:
    s = requests.Session()
    if token:
        s.headers.update({"Authorization": f"Bearer {token}"})
    s.headers.update({"Accept": "application/json"})
    return s

def fetch_page(s: requests.Session, base: str, offset: int, limit: int, build: str) -> List[Dict[str, Any]]:
    # Adjust query params to match actual API (e.g., page=, size=, cursor=)
    url = f"{base}{VAR_ENDPOINT}"
    params = {"build": build, "offset": offset, "limit": limit}
    r = s.get(url, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    # Some APIs return {"results":[...]} others return list directly
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    # if pagination is cursor-based, adapt here (return list + cursor)
    raise RuntimeError(f"Unrecognized payload schema at offset={offset}")

def to_safe(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ",".join(map(str, v))
    return str(v)

def write_tsv_header(out):
    cols = ["chrom","pos","ref","alt","gene","build","classification",
            "assay","splicing_outcome","pmids","source_id"]
    out.write("\t".join(cols) + "\n")
    return cols

def record_to_row(rec: Dict[str, Any]) -> Dict[str,str]:
    # Map keys based on likely field names; adjust if API differs
    return {
        "chrom": to_safe(rec.get("chromosome") or rec.get("chrom")),
        "pos":   to_safe(rec.get("position") or rec.get("pos")),  # 1-based
        "ref":   to_safe(rec.get("ref")),
        "alt":   to_safe(rec.get("alt")),
        "gene":  to_safe(rec.get("gene_symbol") or rec.get("gene")),
        "build": to_safe(rec.get("genome_build") or rec.get("build") or "GRCh38"),
        "classification": to_safe(rec.get("classification")),  # splice-altering / not / low-frequency
        "assay": to_safe(rec.get("assays") or rec.get("assay")),
        "splicing_outcome": to_safe(rec.get("splicing_outcome") or rec.get("outcome")),
        "pmids": to_safe(rec.get("pmids") or rec.get("pmid")),
        "source_id": to_safe(rec.get("id") or rec.get("_id")),
    }

def ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Bulk download SpliceVarDB variants.")
    ap.add_argument("--base", default=DEFAULT_BASE, help="API base URL")
    ap.add_argument("--token", default=os.getenv("SPLICEVARDB_TOKEN"))
    ap.add_argument("--build", default="GRCh38", choices=["GRCh38","GRCh37","hg38","hg19"])
    ap.add_argument("--page-size", type=int, default=PAGE_SIZE)
    ap.add_argument("--max-pages", type=int, default=200000)
    ap.add_argument("--out-jsonl", default="data/external/splicevardb/variants.jsonl")
    ap.add_argument("--out-tsv",   default="data/interim/splicevardb_union.tsv")
    ap.add_argument("--out-vcf",   default="data/interim/splicevardb.vcf", help="Optional VCF path (set '' to skip)")
    args = ap.parse_args()

    ensure_dir(args.out_jsonl)
    ensure_dir(args.out_tsv)
    if args.out_vcf:
        ensure_dir(args.out_vcf)

    s = get_session(args.token)
    total = 0

    # Stream JSONL
    with open(args.out_jsonl, "w") as jout:
        for page_i in range(args.max_pages):
            offset = page_i * args.page_size
            items = fetch_page(s, args.base, offset, args.page_size, args.build)
            if not items:
                break
            for rec in items:
                jout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += len(items)
            if len(items) < args.page_size:
                break
            time.sleep(SLEEP)

    print(f"Downloaded {total} records -> {args.out_jsonl}")

    # Build TSV
    with open(args.out_jsonl, "r") as jin, open(args.out_tsv, "w") as tout:
        cols = write_tsv_header(tout)
        seen = set()
        for line in jin:
            rec = json.loads(line)
            row = record_to_row(rec)
            key = (row["chrom"], row["pos"], row["ref"], row["alt"])
            if key in seen:
                continue
            seen.add(key)
            tout.write("\t".join(row[c] for c in cols) + "\n")

    print(f"Wrote TSV -> {args.out_tsv}")

    # Optional: write VCF
    if args.out_vcf:
        with open(args.out_tsv) as f, open(args.out_vcf, "w") as v:
            v.write("##fileformat=VCFv4.2\n")
            v.write("##source=SpliceVarDB\n")
            v.write("##INFO=<ID=GENE,Number=1,Type=String,Description=\"Gene symbol\">\n")
            v.write("##INFO=<ID=CLASS,Number=1,Type=String,Description=\"SpliceVarDB classification\">\n")
            v.write("##INFO=<ID=ASSAY,Number=1,Type=String,Description=\"Validation assays\">\n")
            v.write("##INFO=<ID=OUTCOME,Number=1,Type=String,Description=\"Splicing outcome\">\n")
            v.write("##INFO=<ID=PMID,Number=.,Type=String,Description=\"PMIDs\">\n")
            v.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            next(f)  # skip header
            for line in f:
                chrom,pos,ref,alt,gene,build,klass,assay,outcome,pmids,srcid = line.rstrip("\n").split("\t")
                info = f"GENE={gene};CLASS={klass};ASSAY={assay};OUTCOME={outcome};PMID={pmids};SRCID={srcid}"
                v.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t{info}\n")
        print(f"Wrote VCF -> {args.out_vcf}")

if __name__ == "__main__":
    main()
