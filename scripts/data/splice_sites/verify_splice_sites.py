#!/usr/bin/env python3
"""
Verify and summarize the splice site annotation file.

**Documentation**: docs/data/splice_sites/splice_site_annotations.md
**Data Source**: data/ensembl/splice_sites.tsv
**Output**: Markdown report to stdout; optional JSON summary

This script validates the schema, coordinates, and statistics documented in
splice_site_annotations.md. Use it to verify documentation accuracy after
data updates or to generate reproducible summaries.

Checks performed:
- Header presence and expected columns (name and order)
- Allowed values for site_type and strand
- Basic coordinate sanity: start < end, position within [start, end]
- Summary stats: total rows, unique genes, unique transcripts, avg transcripts/gene
- Donor/acceptor counts and percentages
- Chromosome distribution (top-N)
- Memory-safe reading with pandas (dtype hints)

Usage examples:
  # Generate verification report
  python scripts/data/splice_sites/verify_splice_sites.py \\
      --tsv data/ensembl/splice_sites.tsv \\
      --top-n 10

  # Save JSON summary for documentation
  python scripts/data/splice_sites/verify_splice_sites.py \\
      --tsv data/ensembl/splice_sites.tsv \\
      --top-n 10 \\
      --json-out docs/data/splice_sites/verification_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd

EXPECTED_COLUMNS = [
    "chrom",
    "start",
    "end",
    "position",
    "strand",
    "site_type",
    "gene_id",
    "transcript_id",
]

ALLOWED_STRAND = {"+", "-"}
ALLOWED_SITE_TYPES = {"donor", "acceptor"}


def load_table(path: str) -> pd.DataFrame:
    dtype_map = {
        "chrom": str,
        "start": "int64",
        "end": "int64",
        "position": "int64",
        "strand": str,
        "site_type": str,
        "gene_id": str,
        "transcript_id": str,
    }
    df = pd.read_csv(path, sep="\t", dtype=dtype_map)
    return df


def validate_schema(df: pd.DataFrame) -> Dict[str, object]:
    issues: List[str] = []
    columns_match = list(df.columns) == EXPECTED_COLUMNS
    if not columns_match:
        issues.append(
            f"Column mismatch. Found={list(df.columns)} Expected={EXPECTED_COLUMNS}"
        )
    return {
        "columns_match": columns_match,
        "issues": issues,
    }


def validate_values(df: pd.DataFrame) -> Dict[str, object]:
    issues: List[str] = []

    # strand
    strands = set(df["strand"].unique())
    bad_strands = sorted(list(strands - ALLOWED_STRAND))
    if bad_strands:
        issues.append(f"Unexpected strand values: {bad_strands}")

    # site_type
    site_types = set(df["site_type"].unique())
    bad_site_types = sorted(list(site_types - ALLOWED_SITE_TYPES))
    if bad_site_types:
        issues.append(f"Unexpected site_type values: {bad_site_types}")

    return {
        "unexpected_strands": bad_strands,
        "unexpected_site_types": bad_site_types,
        "issues": issues,
    }


def validate_coordinates(df: pd.DataFrame, sample_n: int = 100000) -> Dict[str, object]:
    issues: List[str] = []

    # Start < end
    bad_start_end = int((df["start"] >= df["end"]).sum())
    if bad_start_end > 0:
        issues.append(f"Rows with start >= end: {bad_start_end}")

    # Position within [start, end]
    bad_position = int(((df["position"] < df["start"]) | (df["position"] > df["end"])) .sum())
    if bad_position > 0:
        issues.append(f"Rows with position outside [start, end]: {bad_position}")

    # Optional quick sample to look for extreme span sizes
    sample = df.sample(n=min(sample_n, len(df)), random_state=42)
    span = sample["end"] - sample["start"]
    large_spans = int((span > 1000).sum())  # heuristic sanity check
    if large_spans > 0:
        issues.append(f"Sample found {large_spans} spans > 1000 bp (heuristic)")

    return {
        "bad_start_end": bad_start_end,
        "bad_position_range": bad_position,
        "issues": issues,
    }


def summarize(df: pd.DataFrame, top_n: int = 10) -> Dict[str, object]:
    total = int(len(df))
    unique_genes = int(df["gene_id"].nunique())
    unique_tx = int(df["transcript_id"].nunique())

    # avg transcripts per gene (unique transcripts per gene average)
    transcripts_per_gene = (
        df[["gene_id", "transcript_id"]].drop_duplicates().groupby("gene_id")[
            "transcript_id"
        ].nunique()
    )
    avg_tx_per_gene = float(transcripts_per_gene.mean()) if len(transcripts_per_gene) else 0.0

    # site_type distribution
    st_counts = df["site_type"].value_counts().to_dict()
    donor_count = int(st_counts.get("donor", 0))
    acceptor_count = int(st_counts.get("acceptor", 0))

    donor_pct = (donor_count / total * 100.0) if total else 0.0
    acceptor_pct = (acceptor_count / total * 100.0) if total else 0.0

    # chromosome distribution
    chrom_counts = df["chrom"].value_counts()
    top_chrom = chrom_counts.head(top_n).to_dict()
    top_chrom_pct = {k: (v / total * 100.0) for k, v in top_chrom.items()}

    # strand distribution
    strand_counts = df["strand"].value_counts().to_dict()

    return {
        "total_splice_sites": total,
        "unique_genes": unique_genes,
        "unique_transcripts": unique_tx,
        "avg_transcripts_per_gene": avg_tx_per_gene,
        "donor_count": donor_count,
        "acceptor_count": acceptor_count,
        "donor_pct": donor_pct,
        "acceptor_pct": acceptor_pct,
        "top_chromosomes": top_chrom,
        "top_chromosomes_pct": top_chrom_pct,
        "strand_counts": strand_counts,
    }


def render_markdown(summary: Dict[str, object], schema: Dict[str, object], values: Dict[str, object], coords: Dict[str, object], top_n: int) -> str:
    lines: List[str] = []
    lines.append("## Verification Report: splice_sites.tsv")
    lines.append("")

    # Schema
    lines.append("### Schema Validation")
    lines.append(f"- **columns_match**: {schema['columns_match']}")
    if schema["issues"]:
        for it in schema["issues"]:
            lines.append(f"- **issue**: {it}")
    lines.append("")

    # Value checks
    lines.append("### Value Validation")
    lines.append(f"- **unexpected_strands**: {values['unexpected_strands']}")
    lines.append(f"- **unexpected_site_types**: {values['unexpected_site_types']}")
    if values["issues"]:
        for it in values["issues"]:
            lines.append(f"- **issue**: {it}")
    lines.append("")

    # Coordinate checks
    lines.append("### Coordinate Validation")
    lines.append(f"- **bad_start_end**: {coords['bad_start_end']}")
    lines.append(f"- **bad_position_range**: {coords['bad_position_range']}")
    if coords["issues"]:
        for it in coords["issues"]:
            lines.append(f"- **issue**: {it}")
    lines.append("")

    # Summary
    lines.append("### Summary Statistics")
    lines.append(f"- **total_splice_sites**: {summary['total_splice_sites']}")
    lines.append(f"- **unique_genes**: {summary['unique_genes']}")
    lines.append(f"- **unique_transcripts**: {summary['unique_transcripts']}")
    lines.append(
        f"- **avg_transcripts_per_gene**: {summary['avg_transcripts_per_gene']:.2f}"
    )
    lines.append(
        f"- **donor/acceptor**: {summary['donor_count']} / {summary['acceptor_count']}"
    )
    lines.append(
        f"- **donor_pct / acceptor_pct**: {summary['donor_pct']:.1f}% / {summary['acceptor_pct']:.1f}%"
    )
    lines.append("- **strand_counts**: " + json.dumps(summary["strand_counts"]))
    lines.append("")

    # Top chromosomes
    lines.append(f"### Top {top_n} Chromosomes")
    lines.append("| Chromosome | Count | Percentage |")
    lines.append("|------------|-------|------------|")
    for chrom, cnt in summary["top_chromosomes"].items():
        pct = summary["top_chromosomes_pct"][chrom]
        lines.append(f"| {chrom} | {cnt} | {pct:.1f}% |")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Verify splice site annotations TSV")
    p.add_argument("--tsv", default="data/ensembl/splice_sites.tsv", help="Path to splice_sites.tsv")
    p.add_argument("--top-n", type=int, default=10, help="Top-N chromosomes to display")
    p.add_argument("--json-out", default=None, help="Optional path to write JSON summary")
    args = p.parse_args()

    try:
        df = load_table(args.tsv)
    except Exception as e:
        print(f"ERROR: failed to load {args.tsv}: {e}", file=sys.stderr)
        return 2

    schema = validate_schema(df)
    values = validate_values(df)
    coords = validate_coordinates(df)
    summary = summarize(df, top_n=args.top_n)

    # Emit JSON if requested
    if args.json_out:
        out = {
            "schema": schema,
            "values": values,
            "coordinates": coords,
            "summary": summary,
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)

    # Print Markdown summary
    md = render_markdown(summary, schema, values, coords, top_n=args.top_n)
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
