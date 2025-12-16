#!/usr/bin/env python3
"""
Demonstration of Complete Coverage Inference Solution

This script demonstrates the complete solution architecture for addressing
the gap position problem in inference mode. It shows:

1. How sparse training data (883 positions) needs to be expanded to complete coverage (35,716 positions)
2. How uncertainty is detected using ONLY base model scores (no ground truth)
3. How meta-model is applied selectively to uncertain positions
4. How the final output schema ensures all required columns are present

This addresses the core issue: TN downsampling during training creates gaps,
but inference mode requires predictions for ALL positions in target genes.
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
import json
import time

def load_sparse_training_data(gene_id: str = "ENSG00000236172") -> pl.DataFrame:
    """Load the sparse training data that exists for the gene."""
    print("📂 Loading sparse training data (existing from training artifacts)...")
    
    # This is what we actually get from existing training artifacts
    sparse_file = f"results/test_complete_coverage_demo/gap_predictions/{gene_id}/analysis_sequences_filtered.tsv"
    
    if not Path(sparse_file).exists():
        print(f"⚠️  Sparse data file not found: {sparse_file}")
        print("Run the previous test first to generate training artifacts")
        return pl.DataFrame()
    
    df = pl.read_csv(sparse_file, separator='\t')
    print(f"   Loaded {df.height} sparse positions from training artifacts")
    print(f"   Position range: {df['position'].min()} to {df['position'].max()}")
    
    return df

def generate_complete_base_predictions(sparse_df: pl.DataFrame, 
                                     gene_length: int = 35716) -> pl.DataFrame:
    """
    Simulate generating complete base model predictions for ALL positions.
    
    In the real implementation, this would:
    1. Run SpliceAI base model on gap positions (missing 34,833 positions)
    2. Combine with existing sparse predictions
    3. Ensure continuous position coverage 1 to gene_length
    """
    print(f"🎯 Generating complete base model predictions for {gene_length} positions...")
    
    # Get existing positions
    if sparse_df.height > 0:
        existing_positions = set(sparse_df['position'].to_list())
    else:
        existing_positions = set()
    
    # Generate complete position range
    all_positions = set(range(1, gene_length + 1))
    gap_positions = all_positions - existing_positions
    
    print(f"   Existing positions: {len(existing_positions)}")
    print(f"   Gap positions to fill: {len(gap_positions)}")
    print(f"   Total coverage needed: {len(all_positions)}")
    
    # For demonstration, simulate base model predictions on gap positions
    # In real implementation, this would call the actual SpliceAI base model
    gap_data = []
    
    print("   Simulating base model predictions on gap positions...")
    
    for pos in sorted(gap_positions):
        # Simulate realistic base model scores
        # Most positions are "neither" with low scores for donor/acceptor
        
        # Generate realistic score distributions
        donor_score = np.random.beta(1, 10)  # Most scores are low
        acceptor_score = np.random.beta(1, 10)
        neither_score = np.random.beta(5, 2)  # Neither scores tend to be higher
        
        # Normalize to sum to reasonable total
        total = donor_score + acceptor_score + neither_score
        donor_score = donor_score / total * 0.8
        acceptor_score = acceptor_score / total * 0.8
        neither_score = neither_score / total * 0.8
        
        # Occasionally create some higher scoring positions (splice sites)
        if np.random.random() < 0.01:  # 1% chance of being a splice site
            if np.random.random() < 0.5:  # Donor
                donor_score = np.random.beta(5, 2)
                acceptor_score = np.random.beta(1, 5)
                neither_score = np.random.beta(1, 5)
            else:  # Acceptor
                donor_score = np.random.beta(1, 5)
                acceptor_score = np.random.beta(5, 2)
                neither_score = np.random.beta(1, 5)
        
        gap_data.append({
            'gene_id': sparse_df['gene_id'].first() if sparse_df.height > 0 else 'ENSG00000236172',
            'position': pos,
            'donor_score': float(donor_score),
            'acceptor_score': float(acceptor_score),
            'neither_score': float(neither_score),
            'sequence': 'N' * 200,  # Placeholder
            'chrom': '2',
            'strand': '-'
        })
    
    # Create gap predictions DataFrame
    gap_df = pl.DataFrame(gap_data)
    
    # Combine existing sparse data with gap predictions
    if sparse_df.height > 0:
        # Align columns
        essential_cols = ['gene_id', 'position', 'donor_score', 'acceptor_score', 'neither_score']
        
        # Check which columns exist in sparse data
        available_sparse_cols = [col for col in essential_cols if col in sparse_df.columns]
        available_gap_cols = [col for col in essential_cols if col in gap_df.columns]
        
        # Use common columns
        common_cols = list(set(available_sparse_cols) & set(available_gap_cols))
        
        if common_cols:
            combined_df = pl.concat([
                sparse_df.select(common_cols),
                gap_df.select(common_cols)
            ])
        else:
            # If no common columns, just use gap predictions
            combined_df = gap_df
    else:
        combined_df = gap_df
    
    # Sort by position for proper sequence
    complete_df = combined_df.sort('position')
    
    print(f"✅ Complete base predictions: {complete_df.height} total positions")
    
    return complete_df

def analyze_uncertainty_inference_mode(df: pl.DataFrame) -> pl.DataFrame:
    """
    Analyze uncertainty using ONLY base model scores (no ground truth).
    
    Key principle: In inference mode, we cannot access ground truth labels,
    so uncertainty must be inferred from base model score patterns.
    """
    print("🔍 Analyzing uncertainty using ONLY base model scores...")
    
    # Calculate uncertainty metrics from base scores only
    uncertainty_df = df.with_columns([
        # Max score (confidence in best prediction)
        pl.max_horizontal(['donor_score', 'acceptor_score', 'neither_score']).alias('max_score'),
        
        # Score entropy (uncertainty measure)
        pl.struct(['donor_score', 'acceptor_score', 'neither_score'])
        .map_elements(lambda x: calculate_entropy([x['donor_score'], x['acceptor_score'], x['neither_score']]))
        .alias('score_entropy'),
        
        # Score spread (difference between max and second max)
        pl.struct(['donor_score', 'acceptor_score', 'neither_score'])
        .map_elements(lambda x: calculate_score_spread([x['donor_score'], x['acceptor_score'], x['neither_score']]))
        .alias('score_spread'),
        
        # Predicted splice type
        pl.struct(['donor_score', 'acceptor_score', 'neither_score'])
        .map_elements(lambda x: get_predicted_type([x['donor_score'], x['acceptor_score'], x['neither_score']]))
        .alias('predicted_splice_type')
    ])
    
    # Define uncertainty thresholds
    confidence_threshold = 0.5
    entropy_threshold = 0.8
    spread_threshold = 0.15
    
    # Identify uncertain positions
    final_df = uncertainty_df.with_columns([
        # High uncertainty: low max score OR high entropy OR low spread
        (
            (pl.col('max_score') < confidence_threshold) |
            (pl.col('score_entropy') > entropy_threshold) |
            (pl.col('score_spread') < spread_threshold)
        ).alias('is_uncertain'),
        
        # Confidence categories
        pl.when(pl.col('max_score') >= 0.8)
        .then(pl.lit('high'))
        .when(pl.col('max_score') >= 0.5)
        .then(pl.lit('medium'))
        .otherwise(pl.lit('low'))
        .alias('confidence_level')
    ])
    
    # Log uncertainty statistics
    total_positions = final_df.height
    uncertain_positions = final_df.filter(pl.col('is_uncertain')).height
    uncertainty_rate = uncertain_positions / total_positions if total_positions > 0 else 0
    
    print(f"   Uncertainty analysis results:")
    print(f"     Total positions: {total_positions}")
    print(f"     Uncertain positions: {uncertain_positions} ({uncertainty_rate:.1%})")
    
    return final_df

def apply_meta_model_selectively(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply meta-model only to uncertain positions and create final output schema.
    
    This creates the complete output schema with all required columns:
    - gene_id, position, donor_score, acceptor_score, neither_score (base)
    - donor_meta, acceptor_meta, neither_meta (meta-adjusted or same as base)
    - splice_type (predicted type), is_adjusted (0/1 flag)
    """
    print("🧠 Applying meta-model selectively to uncertain positions...")
    
    # Initialize meta scores as copies of base scores
    result_df = df.with_columns([
        pl.col('donor_score').alias('donor_meta'),
        pl.col('acceptor_score').alias('acceptor_meta'),
        pl.col('neither_score').alias('neither_meta'),
        pl.lit('base_model').alias('prediction_source'),
        pl.lit(0).alias('is_adjusted')
    ])
    
    # Get uncertain positions for meta-model application
    uncertain_positions = df.filter(pl.col('is_uncertain') == True)
    uncertain_count = uncertain_positions.height
    
    print(f"   Applying meta-model to {uncertain_count} uncertain positions...")
    
    if uncertain_count > 0:
        # In real implementation, this would:
        # 1. Extract features for uncertain positions using optimized feature enrichment
        # 2. Load and apply trained meta-model
        # 3. Update meta scores with recalibrated values
        
        # For demonstration, apply systematic adjustments
        uncertain_indices = uncertain_positions['position'].to_list()
        uncertain_filter = pl.col('position').is_in(uncertain_indices)
        
        # Apply meta-model adjustments (placeholder logic)
        adjusted_df = result_df.with_columns([
            # Adjust scores for uncertain positions
            pl.when(uncertain_filter)
            .then(pl.col('donor_score') * 1.1)  # Meta-model boost
            .otherwise(pl.col('donor_meta'))
            .alias('donor_meta'),
            
            pl.when(uncertain_filter)
            .then(pl.col('acceptor_score') * 1.1)
            .otherwise(pl.col('acceptor_meta'))
            .alias('acceptor_meta'),
            
            pl.when(uncertain_filter)
            .then(pl.col('neither_score') * 0.9)  # Meta-model reduction
            .otherwise(pl.col('neither_meta'))
            .alias('neither_meta'),
            
            # Update metadata
            pl.when(uncertain_filter)
            .then(pl.lit('meta_model'))
            .otherwise(pl.col('prediction_source'))
            .alias('prediction_source'),
            
            pl.when(uncertain_filter)
            .then(pl.lit(1))
            .otherwise(pl.col('is_adjusted'))
            .alias('is_adjusted')
        ])
        
        result_df = adjusted_df
    
    # Add final splice type prediction based on meta scores
    final_df = result_df.with_columns([
        pl.struct(['donor_meta', 'acceptor_meta', 'neither_meta'])
        .map_elements(lambda x: get_predicted_type([x['donor_meta'], x['acceptor_meta'], x['neither_meta']]))
        .alias('splice_type')
    ])
    
    # Create final output schema with all required columns
    final_output = final_df.select([
        'gene_id', 'position',
        'donor_score', 'acceptor_score', 'neither_score',  # Base model scores
        'donor_meta', 'acceptor_meta', 'neither_meta',     # Meta model scores
        'splice_type', 'is_adjusted', 'prediction_source', # Predictions and flags
        'confidence_level', 'max_score', 'score_entropy',  # Uncertainty metrics
        'is_uncertain'  # Include uncertainty flag
    ])
    
    meta_applied = final_output.filter(pl.col('is_adjusted') == 1).height
    print(f"   Meta-model applied to {meta_applied} positions")
    
    return final_output

def validate_continuous_coverage(df: pl.DataFrame, expected_length: int) -> dict:
    """Validate that position coverage is continuous with no gaps."""
    print("✅ Validating continuous position coverage...")
    
    positions = sorted(df['position'].to_list())
    
    # Check for gaps
    gaps = []
    for i in range(len(positions) - 1):
        if positions[i+1] - positions[i] > 1:
            gaps.append((positions[i], positions[i+1]))
    
    # Check completeness
    min_pos, max_pos = min(positions), max(positions)
    actual_coverage = len(positions)
    
    validation = {
        'is_continuous': len(gaps) == 0,
        'positions_covered': actual_coverage,
        'expected_length': expected_length,
        'coverage_complete': actual_coverage == expected_length,
        'position_range': (min_pos, max_pos),
        'gaps_found': len(gaps)
    }
    
    if validation['is_continuous'] and validation['coverage_complete']:
        print(f"   ✅ Perfect continuous coverage: {actual_coverage} positions")
    else:
        print(f"   ⚠️  Coverage issues: {len(gaps)} gaps, {actual_coverage}/{expected_length} positions")
    
    return validation

# Helper functions
def calculate_entropy(scores):
    """Calculate normalized entropy of score distribution."""
    scores = np.array(scores)
    total = np.sum(scores)
    if total <= 0:
        return 0.0
    probs = scores / total
    probs = np.maximum(probs, 1e-10)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(3))  # Normalize by max entropy

def calculate_score_spread(scores):
    """Calculate spread between highest and second highest scores."""
    sorted_scores = sorted(scores, reverse=True)
    return float(sorted_scores[0] - sorted_scores[1])

def get_predicted_type(scores):
    """Get predicted splice type based on max score."""
    score_names = ['donor', 'acceptor', 'neither']
    max_idx = np.argmax(scores)
    return score_names[max_idx]

def main():
    """Demonstrate the complete coverage inference solution."""
    parser = argparse.ArgumentParser(description="Demonstrate complete coverage inference")
    parser.add_argument("--gene", default="ENSG00000236172", help="Gene ID to test")
    parser.add_argument("--output-dir", default="results/complete_coverage_demo", help="Output directory")
    
    args = parser.parse_args()
    
    print("🧬 Complete Coverage Inference Architecture Demonstration")
    print("=" * 70)
    print(f"Target gene: {args.gene}")
    print(f"Expected length: 35,716 bp (from gene_features.tsv)")
    print(f"Problem: Training artifacts have only 883 sparse positions")
    print(f"Solution: Generate complete coverage with meta-model selectivity")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Load sparse training data (what we currently have)
    sparse_df = load_sparse_training_data(args.gene)
    
    # Step 2: Generate complete base model predictions (fill gaps)
    complete_df = generate_complete_base_predictions(sparse_df, gene_length=35716)
    
    if complete_df.height == 0:
        print("❌ Failed to generate complete predictions")
        return
    
    # Step 3: Analyze uncertainty using only base model scores
    uncertainty_df = analyze_uncertainty_inference_mode(complete_df)
    
    # Step 4: Apply meta-model selectively to uncertain positions
    final_df = apply_meta_model_selectively(uncertainty_df)
    
    # Step 5: Validate continuous coverage
    validation = validate_continuous_coverage(final_df, expected_length=35716)
    
    # Step 6: Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "complete_coverage_predictions.parquet"
    final_df.write_parquet(output_file)
    
    # Generate summary
    runtime = time.time() - start_time
    uncertain_count = final_df.filter(pl.col('is_uncertain')).height
    meta_applied = final_df.filter(pl.col('is_adjusted') == 1).height
    
    summary = {
        'success': True,
        'total_positions': final_df.height,
        'uncertain_positions': uncertain_count,
        'meta_model_applied': meta_applied,
        'meta_application_rate': meta_applied / final_df.height,
        'continuous_coverage': validation['is_continuous'],
        'coverage_complete': validation['coverage_complete'],
        'runtime_seconds': runtime,
        'output_file': str(output_file)
    }
    
    summary_file = output_dir / "demo_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Display results
    print("=" * 70)
    print("🎉 COMPLETE COVERAGE DEMONSTRATION RESULTS")
    print("=" * 70)
    print(f"✅ Total positions: {final_df.height:,}")
    print(f"🔍 Uncertain positions: {uncertain_count:,} ({uncertain_count/final_df.height*100:.1f}%)")
    print(f"🧠 Meta-model applied: {meta_applied:,} ({meta_applied/final_df.height*100:.1f}%)")
    print(f"📐 Continuous coverage: {'✅ Yes' if validation['is_continuous'] else '❌ No'}")
    print(f"📊 Coverage complete: {'✅ Yes' if validation['coverage_complete'] else '❌ No'}")
    print(f"⏱️  Runtime: {runtime:.1f} seconds")
    print(f"📁 Results: {output_file}")
    print("=" * 70)
    
    # Validate output schema
    required_cols = ['gene_id', 'position', 'donor_score', 'acceptor_score', 'neither_score',
                    'donor_meta', 'acceptor_meta', 'neither_meta', 'splice_type', 'is_adjusted']
    missing_cols = [col for col in required_cols if col not in final_df.columns]
    
    if not missing_cols:
        print("✅ All required output columns present")
    else:
        print(f"❌ Missing required columns: {missing_cols}")
    
    print("\n📋 OUTPUT SCHEMA VALIDATION:")
    print(f"   Columns: {len(final_df.columns)}")
    print(f"   Required columns: {required_cols}")
    print(f"   All present: {'✅ Yes' if not missing_cols else '❌ No'}")
    
    # Sample of final data
    print("\n📊 SAMPLE OUTPUT DATA:")
    sample_df = final_df.head(10)
    print(sample_df.select(['gene_id', 'position', 'splice_type', 'is_adjusted', 'confidence_level']).to_pandas())

if __name__ == "__main__":
    main()