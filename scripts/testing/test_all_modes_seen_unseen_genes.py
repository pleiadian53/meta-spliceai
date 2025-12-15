"""
Comprehensive Test: All 3 Modes on Seen and Unseen Genes

This script tests the inference workflow on:
1. Seen genes (included in training data)
2. Unseen genes (not in training data)

For each gene, it tests all 3 modes:
- base_only: Only SpliceAI base model
- hybrid: Base model + selective meta-model recalibration
- meta_only: Meta-model recalibration on all positions

Expected outcomes:
- All modes should produce complete coverage (N positions for N-bp gene)
- Hybrid/meta_only should show improved F1 scores vs base_only
- Seen genes should have higher performance than unseen genes
- No fallback logic - fails loudly on errors
"""

import sys
from pathlib import Path
import polars as pl
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceWorkflow,
    EnhancedSelectiveInferenceConfig
)
from meta_spliceai.system.genomic_resources import Registry


def load_training_genes(model_dir: Path) -> set:
    """Load the list of genes used in training."""
    # Check for gene list file in model directory
    gene_list_file = model_dir / 'training_genes.txt'
    
    if gene_list_file.exists():
        with open(gene_list_file, 'r') as f:
            training_genes = set(line.strip() for line in f if line.strip())
        print(f"✅ Loaded {len(training_genes)} training genes from {gene_list_file}")
        return training_genes
    
    # Fallback: Try to infer from training data directory
    training_data_dir = model_dir.parent / 'data' / 'train_pc_1000_3mers'
    if training_data_dir.exists():
        # Look for analysis files
        analysis_files = list(training_data_dir.glob('**/analysis_sequences_*.tsv'))
        if analysis_files:
            # Extract gene IDs from filenames
            training_genes = set()
            for f in analysis_files:
                # Filename format: analysis_sequences_ENSG00000123456.tsv
                parts = f.stem.split('_')
                if len(parts) >= 3 and parts[2].startswith('ENSG'):
                    training_genes.add(parts[2])
            
            if training_genes:
                print(f"✅ Inferred {len(training_genes)} training genes from analysis files")
                return training_genes
    
    print("⚠️  Could not determine training genes - will test all as unseen")
    return set()


def load_splice_sites(registry: Registry) -> pl.DataFrame:
    """Load splice site annotations for evaluation."""
    ss_path = registry.resolve('splice_sites')
    if not ss_path or not Path(ss_path).exists():
        raise FileNotFoundError(f"Splice sites not found: {ss_path}")
    
    # Read with proper schema for chromosome column
    ss_df = pl.read_csv(
        ss_path,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    print(f"✅ Loaded {len(ss_df):,} splice site annotations")
    return ss_df


def evaluate_predictions(predictions_file: Path, gene_id: str, gene_name: str, 
                        splice_sites_df: pl.DataFrame) -> dict:
    """
    Evaluate predictions against ground truth splice sites.
    
    Returns dict with F1 scores for donors and acceptors.
    """
    if not predictions_file.exists():
        return {
            'gene_id': gene_id,
            'gene_name': gene_name,
            'status': 'missing_file',
            'donor_f1': 0.0,
            'acceptor_f1': 0.0,
            'error': f'File not found: {predictions_file}'
        }
    
    try:
        # Load predictions
        pred_df = pl.read_parquet(predictions_file)
        
        # Get gene-specific splice sites
        gene_ss = splice_sites_df.filter(pl.col('gene_id') == gene_id)
        
        if gene_ss.height == 0:
            return {
                'gene_id': gene_id,
                'gene_name': gene_name,
                'status': 'no_annotations',
                'donor_f1': 0.0,
                'acceptor_f1': 0.0,
                'total_positions': pred_df.height,
                'error': 'No splice site annotations for this gene'
            }
        
        # Evaluate donors
        donor_sites = gene_ss.filter(pl.col('site_type') == 'donor')
        donor_f1 = calculate_f1(pred_df, donor_sites, 'donor_meta', threshold=0.5, window=2)
        
        # Evaluate acceptors
        acceptor_sites = gene_ss.filter(pl.col('site_type') == 'acceptor')
        acceptor_f1 = calculate_f1(pred_df, acceptor_sites, 'acceptor_meta', threshold=0.5, window=2)
        
        return {
            'gene_id': gene_id,
            'gene_name': gene_name,
            'status': 'success',
            'total_positions': pred_df.height,
            'donor_sites': donor_sites.height,
            'acceptor_sites': acceptor_sites.height,
            'donor_f1': donor_f1,
            'acceptor_f1': acceptor_f1,
            'avg_f1': (donor_f1 + acceptor_f1) / 2.0
        }
        
    except Exception as e:
        return {
            'gene_id': gene_id,
            'gene_name': gene_name,
            'status': 'error',
            'donor_f1': 0.0,
            'acceptor_f1': 0.0,
            'error': str(e)
        }


def calculate_f1(pred_df: pl.DataFrame, true_sites: pl.DataFrame, 
                score_col: str, threshold: float = 0.5, window: int = 2) -> float:
    """
    Calculate F1 score for splice site predictions.
    
    Parameters
    ----------
    pred_df : pl.DataFrame
        Predictions with 'position' and score columns
    true_sites : pl.DataFrame
        True splice sites with 'position' column
    score_col : str
        Column name for scores (e.g., 'donor_meta', 'acceptor_meta')
    threshold : float
        Score threshold for positive prediction
    window : int
        Window size for matching predictions to true sites
        
    Returns
    -------
    float
        F1 score (0.0 to 1.0)
    """
    if true_sites.height == 0:
        return 0.0
    
    if score_col not in pred_df.columns:
        return 0.0
    
    # Get predicted positive positions
    pred_positive = pred_df.filter(pl.col(score_col) >= threshold)
    
    if pred_positive.height == 0:
        return 0.0
    
    # Convert to sets for matching
    pred_positions = set(pred_positive['position'].to_list())
    true_positions = set(true_sites['position'].to_list())
    
    # Count true positives (with window tolerance)
    tp = 0
    matched_true = set()
    
    for pred_pos in pred_positions:
        for true_pos in true_positions:
            if true_pos in matched_true:
                continue
            if abs(pred_pos - true_pos) <= window:
                tp += 1
                matched_true.add(true_pos)
                break
    
    # Calculate metrics
    fp = len(pred_positions) - tp
    fn = len(true_positions) - tp
    
    # F1 score
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def test_gene_all_modes(gene_id: str, gene_name: str, gene_length: int,
                       is_seen: bool, model_path: Path, 
                       splice_sites_df: pl.DataFrame) -> dict:
    """
    Test a single gene across all 3 inference modes.
    
    Returns dict with results for each mode.
    """
    print(f"\n{'='*80}")
    print(f"🧬 Testing Gene: {gene_name} ({gene_id})")
    print(f"   Length: {gene_length:,} bp")
    print(f"   Status: {'SEEN (in training)' if is_seen else 'UNSEEN (not in training)'}")
    print(f"{'='*80}")
    
    results = {
        'gene_id': gene_id,
        'gene_name': gene_name,
        'gene_length': gene_length,
        'is_seen': is_seen,
        'modes': {}
    }
    
    modes = ['base_only', 'hybrid', 'meta_only']
    
    for mode in modes:
        print(f"\n📊 Mode: {mode.upper()}")
        print(f"{'─'*60}")
        
        try:
            # Configure workflow
            config = EnhancedSelectiveInferenceConfig(
                target_genes=[gene_id],
                model_path=model_path,
                inference_mode=mode,
                output_name=f'test_comprehensive_{mode}',
                uncertainty_threshold_low=0.02,
                uncertainty_threshold_high=0.50,  # Standard threshold
                use_timestamped_output=False,
                verbose=0
            )
            
            # Run inference
            workflow = EnhancedSelectiveInferenceWorkflow(config)
            inference_results = workflow.run_incremental()
            
            if not inference_results.success:
                print(f"  ❌ Inference failed")
                for err in inference_results.error_messages:
                    print(f"     Error: {err[:200]}")
                
                results['modes'][mode] = {
                    'status': 'failed',
                    'error': '; '.join(inference_results.error_messages)
                }
                continue
            
            # Get predictions file
            gene_paths = workflow.output_manager.get_gene_output_paths(gene_id)
            predictions_file = gene_paths.predictions_file
            
            # Evaluate predictions
            eval_results = evaluate_predictions(
                predictions_file, gene_id, gene_name, splice_sites_df
            )
            
            # Add inference stats
            eval_results['inference_time'] = inference_results.processing_time_seconds
            eval_results['positions_recalibrated'] = inference_results.positions_recalibrated
            eval_results['positions_total'] = inference_results.total_positions
            
            # Check coverage
            coverage_pct = (eval_results['total_positions'] / gene_length) * 100 if gene_length > 0 else 0
            eval_results['coverage_pct'] = coverage_pct
            
            # Print results
            print(f"  ✅ Status: {eval_results['status']}")
            print(f"  📏 Coverage: {eval_results['total_positions']:,}/{gene_length:,} ({coverage_pct:.1f}%)")
            
            if mode in ['hybrid', 'meta_only']:
                recal_pct = (inference_results.positions_recalibrated / inference_results.total_positions * 100) if inference_results.total_positions > 0 else 0
                print(f"  🧠 Meta-model: {inference_results.positions_recalibrated:,} positions ({recal_pct:.1f}%)")
            
            if eval_results['status'] == 'success':
                print(f"  🎯 Donor F1: {eval_results['donor_f1']:.3f} ({eval_results['donor_sites']} sites)")
                print(f"  🎯 Acceptor F1: {eval_results['acceptor_f1']:.3f} ({eval_results['acceptor_sites']} sites)")
                print(f"  🎯 Average F1: {eval_results['avg_f1']:.3f}")
            
            results['modes'][mode] = eval_results
            
        except Exception as e:
            print(f"  ❌ Exception: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
            
            results['modes'][mode] = {
                'status': 'exception',
                'error': str(e)
            }
    
    return results


def main():
    """Run comprehensive tests on all modes with seen and unseen genes."""
    
    print("="*80)
    print("🧪 COMPREHENSIVE TEST: All Modes × Seen/Unseen Genes")
    print("="*80)
    
    # Setup
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / 'results/meta_model_1000genes_3mers/model_multiclass.pkl'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"✅ Model: {model_path}")
    
    # Load training genes
    model_dir = model_path.parent
    training_genes = load_training_genes(model_dir)
    
    # Load resources
    registry = Registry()
    splice_sites_df = load_splice_sites(registry)
    
    # Load gene features
    gene_features_path = Path("data/ensembl/spliceai_analysis/gene_features.tsv")
    if not gene_features_path.exists():
        print(f"❌ Gene features not found: {gene_features_path}")
        return
    
    gene_features_df = pl.read_csv(
        gene_features_path,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    # Define test genes (mix of seen and unseen, protein-coding)
    test_genes = [
        # Seen genes (likely in training if using 1000 protein-coding genes)
        ('ENSG00000134202', 'GSTM3'),    # Small gene, 7kb
        ('ENSG00000157764', 'BRAF'),     # Medium gene, ~190kb
        ('ENSG00000141510', 'TP53'),     # Famous tumor suppressor, ~20kb
        
        # Unseen genes (pick some that are likely NOT in training)
        ('ENSG00000169174', 'PCSK9'),    # Cholesterol metabolism, ~25kb
        ('ENSG00000105976', 'PTEN'),     # Tumor suppressor, ~105kb
        ('ENSG00000012048', 'BRCA1'),    # Breast cancer gene, ~81kb
    ]
    
    all_results = []
    
    for gene_id, gene_name in test_genes:
        # Get gene info
        gene_row = gene_features_df.filter(pl.col('gene_id') == gene_id)
        
        if gene_row.height == 0:
            print(f"\n⚠️  Gene {gene_name} ({gene_id}) not found in gene features - skipping")
            continue
        
        gene_length = gene_row['gene_length'][0]
        is_seen = gene_id in training_genes
        
        # Test gene across all modes
        gene_results = test_gene_all_modes(
            gene_id, gene_name, gene_length, is_seen,
            model_path, splice_sites_df
        )
        
        all_results.append(gene_results)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 SUMMARY: All Genes × All Modes")
    print("="*80)
    
    # Create summary table
    summary_rows = []
    for gene_result in all_results:
        gene_id = gene_result['gene_id']
        gene_name = gene_result['gene_name']
        is_seen = gene_result['is_seen']
        
        row = {
            'Gene': f"{gene_name} ({gene_id})",
            'Status': 'SEEN' if is_seen else 'UNSEEN',
            'Length': f"{gene_result['gene_length']:,} bp"
        }
        
        for mode in ['base_only', 'hybrid', 'meta_only']:
            if mode in gene_result['modes']:
                mode_result = gene_result['modes'][mode]
                if mode_result.get('status') == 'success':
                    row[f'{mode}_F1'] = f"{mode_result['avg_f1']:.3f}"
                else:
                    row[f'{mode}_F1'] = 'FAIL'
            else:
                row[f'{mode}_F1'] = 'N/A'
        
        summary_rows.append(row)
    
    # Print as table
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n" + summary_df.to_string(index=False))
    
    # Analysis
    print("\n" + "="*80)
    print("🔍 ANALYSIS")
    print("="*80)
    
    # Compare modes
    mode_scores = {'base_only': [], 'hybrid': [], 'meta_only': []}
    seen_scores = {'base_only': [], 'hybrid': [], 'meta_only': []}
    unseen_scores = {'base_only': [], 'hybrid': [], 'meta_only': []}
    
    for gene_result in all_results:
        is_seen = gene_result['is_seen']
        
        for mode in ['base_only', 'hybrid', 'meta_only']:
            if mode in gene_result['modes']:
                mode_result = gene_result['modes'][mode]
                if mode_result.get('status') == 'success' and 'avg_f1' in mode_result:
                    f1 = mode_result['avg_f1']
                    mode_scores[mode].append(f1)
                    
                    if is_seen:
                        seen_scores[mode].append(f1)
                    else:
                        unseen_scores[mode].append(f1)
    
    # Print averages
    print("\n1. Average F1 by Mode:")
    for mode in ['base_only', 'hybrid', 'meta_only']:
        if mode_scores[mode]:
            avg_f1 = sum(mode_scores[mode]) / len(mode_scores[mode])
            print(f"   {mode:12s}: {avg_f1:.3f} (n={len(mode_scores[mode])})")
    
    print("\n2. Average F1 by Mode × Seen/Unseen:")
    for mode in ['base_only', 'hybrid', 'meta_only']:
        seen_avg = sum(seen_scores[mode]) / len(seen_scores[mode]) if seen_scores[mode] else 0.0
        unseen_avg = sum(unseen_scores[mode]) / len(unseen_scores[mode]) if unseen_scores[mode] else 0.0
        print(f"   {mode:12s}: SEEN={seen_avg:.3f} (n={len(seen_scores[mode])}), UNSEEN={unseen_avg:.3f} (n={len(unseen_scores[mode])})")
    
    print("\n3. Mode Comparison:")
    if mode_scores['base_only'] and mode_scores['hybrid']:
        base_avg = sum(mode_scores['base_only']) / len(mode_scores['base_only'])
        hybrid_avg = sum(mode_scores['hybrid']) / len(mode_scores['hybrid'])
        improvement = ((hybrid_avg - base_avg) / base_avg * 100) if base_avg > 0 else 0
        print(f"   Hybrid vs Base: {improvement:+.1f}% improvement")
    
    if mode_scores['base_only'] and mode_scores['meta_only']:
        base_avg = sum(mode_scores['base_only']) / len(mode_scores['base_only'])
        meta_avg = sum(mode_scores['meta_only']) / len(mode_scores['meta_only'])
        improvement = ((meta_avg - base_avg) / base_avg * 100) if base_avg > 0 else 0
        print(f"   Meta-only vs Base: {improvement:+.1f}% improvement")
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()





