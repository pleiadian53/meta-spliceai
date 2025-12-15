
#!/usr/bin/env python3

"""
featurize_positions.py
----------------------
Builds *per-position* features for MetaSpliceAI's meta-learner, including:
- Base per-nucleotide scores (donor, acceptor, neither) from SpliceAI/OpenSpliceAI
- Variant-aware features (delta scores within a window, distance to variant)
- Genomic features (k-mers, GC, region class)

This script is a template; wire it to your actual score files and reference.
"""
import os, argparse, math
import pandas as pd
import numpy as np

def rolling_kmers(seq: str, k: int):
    seq = seq.upper()
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

def encode_kmer(kmer: str):
    # simple hash-based encoding; replace with a proper tokenizer or one-hot as needed
    return hash(kmer) % (10**6)

def annotate_region_class(dist_to_junc: int, canonical_window=2, near_window=50):
    if abs(dist_to_junc) <= canonical_window:
        return "canonical"
    if abs(dist_to_junc) <= near_window:
        return "near"
    return "deep_intronic"

def build_features_for_chrom(chrom_df, variants_df, window=400):
    """
    chrom_df: per-position base scores with columns:
        CHROM, POS, spliceai_donor, spliceai_acceptor, spliceai_neither, seq (optional)
    variants_df: variants on this chrom with columns:
        CHROM, POS, REF, ALT, donor_gain, donor_loss, acceptor_gain, acceptor_loss
    """
    # Index variants by position for quick lookup
    var_pos = variants_df.groupby("POS").agg({
        "donor_gain":"max", "donor_loss":"max",
        "acceptor_gain":"max", "acceptor_loss":"max"
    })

    # For each position, aggregate variant deltas within +/- window
    positions = chrom_df["POS"].values
    d_aggr = np.zeros((len(positions), 4), dtype=float)  # ALG, ALL, DLG, DLL (max in window)

    var_positions = var_pos.index.values
    for i, p in enumerate(positions):
        # find nearby variants by simple window scan (can be optimized with interval trees)
        lo, hi = p - window, p + window
        mask = (var_positions >= lo) & (var_positions <= hi)
        if not mask.any():
            continue
        sub = var_pos.iloc[np.where(mask)[0]]
        # max aggregation across window
        d_aggr[i,0] = sub["acceptor_gain"].max()
        d_aggr[i,1] = sub["acceptor_loss"].max()
        d_aggr[i,2] = sub["donor_gain"].max()
        d_aggr[i,3] = sub["donor_loss"].max()

    out = chrom_df.copy()
    out["ALG_max_w"] = d_aggr[:,0]
    out["ALL_max_w"] = d_aggr[:,1]
    out["DLG_max_w"] = d_aggr[:,2]
    out["DLL_max_w"] = d_aggr[:,3]

    # Example: region class using nearest junction distance if available
    if "dist_to_nearest_junction" in out.columns:
        out["region_class"] = out["dist_to_nearest_junction"].apply(annotate_region_class)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-scores", required=True, help="Per-position base scores (parquet/csv)")
    ap.add_argument("--variant-deltas", required=True, help="Per-variant deltas (parquet/csv)")
    ap.add_argument("--out", required=True, help="Output per-position feature table (parquet)")
    ap.add_argument("--window", type=int, default=400, help="Window for aggregating deltas around a position")
    args = ap.parse_args()

    def load_any(p):
        if p.endswith(".parquet"):
            return pd.read_parquet(p)
        sep = "," if p.endswith(".csv") else "\t"
        return pd.read_csv(p, sep=sep)

    base = load_any(args.base_scores)
    deltas = load_any(args.variant_deltas)

    # Ensure consistent naming
    req_b = {"CHROM","POS","spliceai_donor","spliceai_acceptor","spliceai_neither"}
    req_v = {"CHROM","POS","donor_gain","donor_loss","acceptor_gain","acceptor_loss"}
    missing_b = req_b - set(base.columns)
    missing_v = req_v - set(deltas.columns)
    if missing_b:
        raise SystemExit(f"Missing columns in base-scores: {missing_b}")
    if missing_v:
        raise SystemExit(f"Missing columns in variant-deltas: {missing_v}")

    out_df_list = []
    for chrom, chrom_df in base.groupby("CHROM"):
        v_chr = deltas[deltas["CHROM"] == chrom]
        out_df_list.append(build_features_for_chrom(chrom_df.sort_values("POS"), v_chr, window=args.window))

    out = pd.concat(out_df_list, ignore_index=True)
    # Save
    if args.out.endswith(".parquet"):
        out.to_parquet(args.out, index=False)
    else:
        out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {out.shape}")

if __name__ == "__main__":
    main()
