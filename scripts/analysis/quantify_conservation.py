#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import math
import sys
from collections import Counter
from typing import Dict, Iterable, List, Tuple, Optional

ALPHABET = ["A", "C", "G", "T"]
COMP = {"A":"T","C":"G","G":"C","T":"A","N":"N"}

def rc(seq: str) -> str:
    # Convert to string if needed (pyfaidx may return Sequence object)
    s = str(seq).upper()
    return "".join(COMP.get(b, "N") for b in reversed(s))

def safe_int(x: str) -> int:
    try:
        return int(x)
    except Exception:
        raise ValueError(f"Expected integer, got: {x!r}")

def read_sites_tsv(tsv_path: Path) -> Iterable[Dict[str,str]]:
    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row

def open_fasta(fasta_path: Path):
    try:
        from pyfaidx import Fasta  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pyfaidx is required to extract sequences from FASTA.\n"
            "Install with: mamba/conda install -c conda-forge pyfaidx  (or pip install pyfaidx)"
        ) from e
    return Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)

def extract_window(
    fasta,
    chrom: str,
    position_1based: int,
    strand: str,
    site_type: str,
    donor_exon_bases: int = 3,
    donor_intron_bases: int = 6,
    acceptor_intron_bases: int = 20,
    acceptor_exon_bases: int = 3,
) -> str:
    chrom = str(chrom)
    s = strand
    pos = int(position_1based)
    
    # Handle chromosome naming variations (chr1 vs 1)
    # Try original name first, then fallback to alternative
    try:
        chrom_seq = fasta[chrom]
    except KeyError:
        if chrom.startswith('chr'):
            try:
                chrom_seq = fasta[chrom[3:]]  # Try without 'chr'
                chrom = chrom[3:]
            except KeyError:
                raise KeyError(f"Chromosome '{chrom}' not found in FASTA (tried with and without 'chr' prefix)")
        else:
            try:
                chrom_seq = fasta[f'chr{chrom}']  # Try with 'chr'
                chrom = f'chr{chrom}'
            except KeyError:
                raise KeyError(f"Chromosome '{chrom}' not found in FASTA (tried with and without 'chr' prefix)")

    if site_type == "donor":
        if s == "+":
            start0 = pos - donor_exon_bases - 1
            end0 = pos + donor_intron_bases - 1  # Fixed: was missing -1
            seq = fasta[chrom][start0:end0]
        else:
            # Negative strand: match analyze_consensus_motifs.py logic
            start0 = pos - donor_intron_bases
            end0 = pos + donor_exon_bases
            seq = rc(fasta[chrom][start0:end0])
        return str(seq)

    elif site_type == "acceptor":
        if s == "+":
            start0 = (pos - 2) - acceptor_intron_bases
            end0 = (pos - 1) + acceptor_exon_bases + 1
            seq = fasta[chrom][start0:end0]
        else:
            start0 = pos - acceptor_exon_bases - 1
            end0 = pos + acceptor_intron_bases + 1  # Fixed: was missing +1
            seq = rc(fasta[chrom][start0:end0])
        return str(seq)

    else:
        raise ValueError(f"Unknown site_type: {site_type!r}")

def pfm(seqs: List[str], alphabet: List[str] = ALPHABET) -> List[Dict[str,int]]:
    if not seqs:
        return []
    L = len(seqs[0])
    for s in seqs:
        if len(s) != L:
            raise ValueError("All sequences must be the same length")
    mat: List[Dict[str,int]] = [dict.fromkeys(alphabet, 0) for _ in range(L)]
    for s in seqs:
        s = s.upper()
        for i, b in enumerate(s):
            if b not in alphabet:
                continue
            mat[i][b] += 1
    return mat

def ppm(pfm_mat: List[Dict[str,int]]) -> List[Dict[str,float]]:
    ppm_mat: List[Dict[str,float]] = []
    for col in pfm_mat:
        n = sum(col.values())
        if n == 0:
            ppm_mat.append({b: 0.0 for b in ALPHABET})
        else:
            ppm_mat.append({b: col.get(b,0)/n for b in ALPHABET})
    return ppm_mat

def log_odds(ppm_mat: List[Dict[str,float]], bg: Optional[Dict[str,float]] = None) -> List[Dict[str,float]]:
    if bg is None:
        bg = {b: 0.25 for b in ALPHABET}
    lo: List[Dict[str,float]] = []
    eps = 1e-9
    for col in ppm_mat:
        lo.append({b: math.log2((col.get(b,0.0)+eps) / (bg.get(b,eps))) for b in ALPHABET})
    return lo

def information_content(ppm_mat: List[Dict[str,float]], bg: Optional[Dict[str,float]] = None) -> List[float]:
    if bg is None:
        bg = {b: 0.25 for b in ALPHABET}
    ic = []
    eps = 1e-12
    for col in ppm_mat:
        s = 0.0
        for b in ALPHABET:
            p = max(col.get(b,0.0), eps)
            q = max(bg.get(b,eps), eps)
            s += p * math.log2(p/q)
        ic.append(s)
    return ic

def write_matrix_csv(out_path: Path, mat: List[Dict[str, float or int]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["pos"] + ALPHABET
        writer.writerow(header)
        for i, col in enumerate(mat):
            writer.writerow([i] + [col.get(b, 0) for b in ALPHABET])

def write_ic_csv(out_path: Path, ic: List[float]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pos", "IC_bits"])
        for i, v in enumerate(ic):
            writer.writerow([i, f"{v:.6f}"])

def summarize_core_dinucleotides(
    seqs: List[str], 
    site_type: str,
    acceptor_intron_bases: int = 20,
    acceptor_exon_bases: int = 3
) -> Dict[str,float]:
    total = len(seqs)
    counts = Counter()
    if site_type == "donor":
        for s in seqs:
            if len(s) < 6: 
                continue
            dinuc = s[3:5]  # Fixed: was s[4:6], should be s[3:5] for 0-based positions 3-4
            counts[dinuc] += 1
        canon_gt = counts.get("GT", 0) / total * 100 if total else 0.0
        noncanon_gc = counts.get("GC", 0) / total * 100 if total else 0.0
        return {"GT_%": canon_gt, "GC_%": noncanon_gc}
    else:  # acceptor
        if not seqs:
            return {"AG_%": 0.0}
        # AG is at positions [acceptor_intron_bases, acceptor_intron_bases+1]
        iA = acceptor_intron_bases  # Fixed: was L - 1 - 3 - 2
        iG = iA + 1
        for s in seqs:
            if len(s) <= iG: 
                continue
            dinuc = s[iA:iG+1]  # Extract AG dinucleotide
            counts[dinuc] += 1
        ag = counts.get("AG", 0) / total * 100 if total else 0.0
        return {"AG_%": ag}

def main(argv=None):
    p = argparse.ArgumentParser(description="Quantify conservation at splice sites (PFM/PPM/IC).")
    p.add_argument("--sites", required=True, type=Path, help="splice_sites_enhanced.tsv (schema as shown)")
    p.add_argument("--fasta", type=Path, help="Reference FASTA (required if no 'seq' column)")
    p.add_argument("--site-type", choices=["donor","acceptor","both"], default="both")
    p.add_argument("--donor-exon", type=int, default=3, help="Donor: # exonic bases upstream of boundary")
    p.add_argument("--donor-intron", type=int, default=6, help="Donor: # intronic bases downstream of boundary")
    p.add_argument("--acceptor-intron", type=int, default=20, help="Acceptor: # intronic bases upstream of AG")
    p.add_argument("--acceptor-exon", type=int, default=3, help="Acceptor: # exonic bases downstream of boundary")
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows; otherwise limit for quick runs")
    p.add_argument("--outdir", type=Path, default=Path("consensus_out"), help="Directory for CSV outputs")
    p.add_argument("--bg", choices=["uniform","empirical"], default="uniform", help="Background for IC/log-odds")
    args = p.parse_args(argv)

    rows = list(read_sites_tsv(args.sites))
    if args.max_rows and len(rows) > args.max_rows:
        rows = rows[:args.max_rows]
    
    # Standardize schema: handle both 'site_type' and 'splice_type' columns
    # (GRCh37/Ensembl uses 'splice_type', GRCh38/MANE may use 'site_type')
    if rows:
        first_row = rows[0]
        if 'splice_type' in first_row and 'site_type' not in first_row:
            # Rename splice_type to site_type for consistency
            for row in rows:
                if 'splice_type' in row:
                    row['site_type'] = row['splice_type']

    have_seq_col = rows and ("seq" in rows[0] and rows[0]["seq"])

    fasta = None
    if not have_seq_col:
        if not args.fasta:
            raise SystemExit("No 'seq' column detected. Please provide --fasta to extract sequences.")
        fasta = open_fasta(args.fasta)

    def collect(site_type: str) -> List[str]:
        seqs: List[str] = []
        for r in rows:
            if r.get("site_type") != site_type:
                continue
            if have_seq_col and r.get("seq"):
                s = r["seq"].upper()
            else:
                chrom = r["chrom"]
                pos = safe_int(r["position"])
                strand = r["strand"]
                s = extract_window(
                    fasta, chrom, pos, strand, site_type,
                    donor_exon_bases=args.donor_exon,
                    donor_intron_bases=args.donor_intron,
                    acceptor_intron_bases=args.acceptor_intron,
                    acceptor_exon_bases=args.acceptor_exon
                )
            seqs.append(s)
        return seqs

    site_types = ["donor","acceptor"] if args.site_type == "both" else [args.site_type]
    from collections import Counter
    all_bg_counts = Counter()
    per_type_counts: Dict[str,int] = {}

    seqs_by_type: Dict[str,List[str]] = {}
    for st in site_types:
        seqs = collect(st)
        seqs_by_type[st] = seqs
        per_type_counts[st] = len(seqs)
        for s in seqs:
            all_bg_counts.update(list(s))

    if args.bg == "empirical":
        total_bases = sum(all_bg_counts[b] for b in ALPHABET)
        bg = {b: (all_bg_counts[b]/total_bases if total_bases else 0.25) for b in ALPHABET}
    else:
        bg = {b: 0.25 for b in ALPHABET}

    args.outdir.mkdir(parents=True, exist_ok=True)

    for st in site_types:
        seqs = seqs_by_type[st]
        if not seqs:
            print(f"[{st}] No sequences found.", file=sys.stderr)
            continue
        L = len(seqs[0])
        if any(len(s)!=L for s in seqs):
            raise SystemExit(f"[{st}] Inconsistent sequence lengths in window.")

        mat_pfm = pfm(seqs)
        mat_ppm = ppm(mat_pfm)
        mat_lo = log_odds(mat_ppm, bg=bg)
        ic = information_content(mat_ppm, bg=bg)

        write_matrix_csv(args.outdir / f"{st}_pfm.csv", mat_pfm)
        write_matrix_csv(args.outdir / f"{st}_ppm.csv", mat_ppm)
        write_matrix_csv(args.outdir / f"{st}_logodds.csv", mat_lo)
        write_ic_csv(args.outdir / f"{st}_ic.csv", ic)

        dinuc = summarize_core_dinucleotides(
            seqs, st,
            acceptor_intron_bases=args.acceptor_intron,
            acceptor_exon_bases=args.acceptor_exon
        )

        print(f"=== {st.upper()} ===")
        print(f"n = {len(seqs)}; window length = {L} bases")
        print(f"Background = {args.bg} ({bg})")
        if st == "donor":
            print(f"Canonical GT%: {dinuc['GT_%']:.2f}; Non-canonical GC%: {dinuc['GC_%']:.2f}")
            print("Boundary convention: 'position' = first base of intron (+1; the 'G' in GT)")
            print(f"Window: [-{args.donor_exon} exon | +{args.donor_intron} intron]")
        else:
            print(f"Canonical AG%: {dinuc['AG_%']:.2f}")
            print("Boundary convention: 'position' = first base of exon (after AG)")
            print(f"Window: [-{args.acceptor_intron} intron ... AG | +{args.acceptor_exon} exon]")
        print(f"Wrote: {args.outdir}/{st}_pfm.csv, {st}_ppm.csv, {st}_logodds.csv, {st}_ic.csv")
        print()

if __name__ == "__main__":
    sys.exit(main())
