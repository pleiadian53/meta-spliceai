#!/usr/bin/env python3
"""
Utility script to test the transcript-level top-k accuracy implementation.

This script loads a sample of training data, loads annotation files,
and calculates both gene-level and transcript-level top-k accuracy for comparison.
"""
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from meta_spliceai.splice_engine.meta_models.evaluation.top_k_metrics import (
    calculate_cv_fold_top_k,
    report_top_k_accuracy
)
from meta_spliceai.splice_engine.meta_models.evaluation.transcript_mapping import (
    calculate_transcript_level_top_k,
    report_transcript_top_k
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test transcript-level top-k accuracy")
    parser.add_argument("--data-path", required=True,
                      help="Path to parquet file with training data")
    parser.add_argument("--splice-sites-path", 
                      default="data/ensembl/splice_sites.tsv",
                      help="Path to splice site annotations file")
    parser.add_argument("--transcript-features-path", 
                      default="data/ensembl/spliceai_analysis/transcript_features.tsv",
                      help="Path to transcript features file")
    parser.add_argument("--gene-features-path", 
                      default="data/ensembl/spliceai_analysis/gene_features.tsv",
                      help="Path to gene features file")
    parser.add_argument("--gene-col", default="gene_id",
                      help="Column name for gene IDs in dataset")
    parser.add_argument("--position-col", default="position",
                      help="Column name for genomic positions in dataset")
    parser.add_argument("--chrom-col", default="chrom",
                      help="Column name for chromosome in dataset")
    parser.add_argument("--donor-score-col", default="donor_score",
                      help="Column with raw donor probability from the base model")
    parser.add_argument("--acceptor-score-col", default="acceptor_score",
                      help="Column with raw acceptor probability from the base model")
    parser.add_argument("--sample", type=int, default=10000,
                      help="Number of rows to sample from dataset")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for sampling and reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Verify input files
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    print(f"Loading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    
    # Sample the dataset if requested
    if args.sample and args.sample < len(df):
        print(f"Sampling {args.sample} rows from dataset")
        df = df.sample(args.sample, random_state=args.seed)
    
    # Verify required columns
    required_cols = [args.gene_col, args.position_col, args.chrom_col, 
                    args.donor_score_col, args.acceptor_score_col, "splice_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Print dataset statistics
    print(f"\nDataset summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique genes: {df[args.gene_col].nunique()}")
    print(f"  Unique chromosomes: {df[args.chrom_col].nunique()}")
    print(f"  Splice type distribution:")
    print(df["splice_type"].value_counts())
    
    # Map splice_type to label (0=donor, 1=acceptor, 2=neither)
    # Handle case where 'neither' is currently encoded as '0' in the training data
    label_map = {"donor": 0, "acceptor": 1, "neither": 2, "0": 2}
    df["label"] = df["splice_type"].map(label_map)
    
    # Verify all values were mapped correctly
    unmapped = df[df["label"].isna()]
    if len(unmapped) > 0:
        print(f"WARNING: {len(unmapped)} rows have unmapped splice_type values:")
        print(unmapped["splice_type"].value_counts())
        # Fill unmapped values as 'neither' class
        df["label"] = df["label"].fillna(2)
    
    # Create probability columns from model scores
    df["prob_donor"] = df[args.donor_score_col]
    df["prob_acceptor"] = df[args.acceptor_score_col]
    
    # Calculate gene-level top-k accuracy
    print("\n==== Gene-level Top-K Accuracy ====")
    # Extract arrays from DataFrame
    y = df["label"].values
    probs = np.zeros((len(df), 3))
    probs[:, 0] = df["prob_donor"].values  # Donor probabilities in first column
    probs[:, 1] = df["prob_acceptor"].values  # Acceptor probabilities in second column
    # For neither probability, calculate 1 - (donor_prob + acceptor_prob) to ensure sum to 1
    probs[:, 2] = 1.0 - (probs[:, 0] + probs[:, 1])
    
    gene_top_k_metrics = calculate_cv_fold_top_k(
        X=np.zeros((len(df), 1)),  # Dummy X array, not used in calculation
        y=y,
        probs=probs,
        gene_ids=df[args.gene_col].values,
        donor_label=0,
        acceptor_label=1,
        neither_label=2
    )
    print(report_top_k_accuracy(gene_top_k_metrics))
    
    # Calculate transcript-level top-k accuracy if annotation files exist
    print("\n==== Transcript-Level Top-K Accuracy ====")
    if (os.path.exists(args.splice_sites_path) and 
        os.path.exists(args.transcript_features_path)):
        try:
            transcript_top_k_metrics = calculate_transcript_level_top_k(
                df=df,
                splice_sites_path=args.splice_sites_path,
                transcript_features_path=args.transcript_features_path,
                gene_features_path=args.gene_features_path if os.path.exists(args.gene_features_path) else None,
                position_col=args.position_col,
                chrom_col=args.chrom_col,
                label_col="label",
                prob_donor_col="prob_donor",
                prob_acceptor_col="prob_acceptor",
                site_type_col="splice_type",
                donor_label=0,
                acceptor_label=1,
                neither_label=2,
                use_cache=True
            )
            print(report_transcript_top_k(transcript_top_k_metrics))
            
            # Compare gene-level and transcript-level metrics
            print("\n==== Comparison ====")
            print(f"Gene-level top-k:      {gene_top_k_metrics['combined_top_k']:.4f}")
            print(f"Transcript-level top-k: {transcript_top_k_metrics.get('transcript_combined_top_k', float('nan')):.4f}")
            
        except Exception as e:
            print(f"Error calculating transcript-level metrics: {e}")
    else:
        print(f"Skipping transcript-level metrics: annotation files not found")
        if not os.path.exists(args.splice_sites_path):
            print(f"  Missing: {args.splice_sites_path}")
        if not os.path.exists(args.transcript_features_path):
            print(f"  Missing: {args.transcript_features_path}")


if __name__ == "__main__":
    main()
