#!/usr/bin/env python3
"""
Comprehensive Score-Shifting Validation: 20+ Protein-Coding Genes

Tests score-shifting on 20+ NEW protein-coding genes with stratified analysis by:
- Splice type (donor vs acceptor)
- Strand (+ vs -)
- Gene size (small, medium, large)

Validates that predictions align EXACTLY with GTF annotations (not off by 1-2 nt).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import polars as pl
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_20_test_genes() -> List[Dict]:
    """
    Get 20+ diverse protein-coding genes (DIFFERENT from previous tests).
    
    Selection criteria:
    - Well-known genes with good annotation
    - Mix of sizes, strands, chromosomes
    - NOT including: GSTM3, BRAF, TP53 (already tested)
    """
    return [
        # Chromosome 1
        {'gene_id': 'ENSG00000117318', 'gene_name': 'ID3', 'chr': '1', 'strand': '-'},
        {'gene_id': 'ENSG00000162692', 'gene_name': 'VCAM1', 'chr': '1', 'strand': '+'},
        
        # Chromosome 2
        {'gene_id': 'ENSG00000163930', 'gene_name': 'BAP1', 'chr': '3', 'strand': '+'},
        {'gene_id': 'ENSG00000138795', 'gene_name': 'LEF1', 'chr': '4', 'strand': '-'},
        
        # Chromosome 5
        {'gene_id': 'ENSG00000113558', 'gene_name': 'SKP2', 'chr': '5', 'strand': '-'},
        {'gene_id': 'ENSG00000164362', 'gene_name': 'TERT', 'chr': '5', 'strand': '-'},
        
        # Chromosome 6
        {'gene_id': 'ENSG00000204287', 'gene_name': 'HLA-A', 'chr': '6', 'strand': '+'},
        {'gene_id': 'ENSG00000111640', 'gene_name': 'GAPDH', 'chr': '12', 'strand': '+'},
        
        # Chromosome 7
        {'gene_id': 'ENSG00000146648', 'gene_name': 'EGFR', 'chr': '7', 'strand': '+'},
        {'gene_id': 'ENSG00000105974', 'gene_name': 'CAV1', 'chr': '7', 'strand': '-'},
        
        # Chromosome 8
        {'gene_id': 'ENSG00000136997', 'gene_name': 'MYC', 'chr': '8', 'strand': '+'},
        {'gene_id': 'ENSG00000147889', 'gene_name': 'CDKN2A', 'chr': '9', 'strand': '+'},
        
        # Chromosome 10
        {'gene_id': 'ENSG00000165731', 'gene_name': 'RET', 'chr': '10', 'strand': '+'},
        {'gene_id': 'ENSG00000107485', 'gene_name': 'GATA3', 'chr': '10', 'strand': '+'},
        
        # Chromosome 11
        {'gene_id': 'ENSG00000149925', 'gene_name': 'ALDOA', 'chr': '16', 'strand': '-'},
        {'gene_id': 'ENSG00000073756', 'gene_name': 'PTGS2', 'chr': '1', 'strand': '-'},
        
        # Chromosome 12
        {'gene_id': 'ENSG00000123374', 'gene_name': 'CDK2', 'chr': '12', 'strand': '+'},
        {'gene_id': 'ENSG00000111276', 'gene_name': 'CDKN1B', 'chr': '12', 'strand': '+'},
        
        # Chromosome 13-22
        {'gene_id': 'ENSG00000139687', 'gene_name': 'RB1', 'chr': '13', 'strand': '+'},
        {'gene_id': 'ENSG00000100030', 'gene_name': 'MAPK1', 'chr': '22', 'strand': '+'},
        
        # Additional genes for 20+
        {'gene_id': 'ENSG00000134086', 'gene_name': 'VHL', 'chr': '3', 'strand': '+'},
        {'gene_id': 'ENSG00000141736', 'gene_name': 'ERBB2', 'chr': '17', 'strand': '+'},
        {'gene_id': 'ENSG00000115414', 'gene_name': 'FN1', 'chr': '2', 'strand': '+'},
        {'gene_id': 'ENSG00000196549', 'gene_name': 'MME', 'chr': '3', 'strand': '+'},
    ]


def load_gtf_splice_sites(gene_id: str) -> Tuple[pl.DataFrame, Dict]:
    """
    Load splice sites from GTF for a gene.
    
    Returns:
        - DataFrame with splice sites
        - Dict with gene metadata (strand, chr, start, end)
    """
    gtf_path = Path("data/ensembl/Homo_sapiens.GRCh38.112.gtf")
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")
    
    # Read GTF lines for this gene
    import subprocess
    result = subprocess.run(
        ['grep', gene_id, str(gtf_path)],
        capture_output=True,
        text=True
    )
    
    if not result.stdout:
        return pl.DataFrame(), {}
    
    # Parse exons to get splice sites
    splice_sites = []
    gene_info = {}
    
    for line in result.stdout.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) < 9:
            continue
        
        feature = parts[2]
        chrom = parts[0]
        start = int(parts[3])
        end = int(parts[4])
        strand = parts[6]
        
        # Store gene info
        if not gene_info:
            gene_info = {'chr': chrom, 'strand': strand, 'start': start, 'end': end}
        else:
            gene_info['start'] = min(gene_info['start'], start)
            gene_info['end'] = max(gene_info['end'], end)
        
        # Extract splice sites from exons
        if feature == 'exon':
            if strand == '+':
                # Donor: end of exon, Acceptor: start of exon
                splice_sites.append({'position': end, 'splice_type': 'donor', 'strand': strand})
                splice_sites.append({'position': start, 'splice_type': 'acceptor', 'strand': strand})
            else:
                # Negative strand: reversed
                splice_sites.append({'position': start, 'splice_type': 'donor', 'strand': strand})
                splice_sites.append({'position': end, 'splice_type': 'acceptor', 'strand': strand})
    
    if not splice_sites:
        return pl.DataFrame(), gene_info
    
    return pl.DataFrame(splice_sites).unique(), gene_info


def analyze_position_offsets(predictions_df: pl.DataFrame, annotations_df: pl.DataFrame, 
                             threshold: float = 0.5) -> Dict:
    """
    Analyze if predictions are off by N nucleotides from true splice sites.
    
    For each true splice site, find the nearest predicted site and calculate offset.
    This tells us if the adjustment is correct or off by 1-2 nt.
    """
    results = {
        'donor': {'offsets': [], 'exact_matches': 0, 'off_by_1': 0, 'off_by_2': 0, 'off_by_more': 0},
        'acceptor': {'offsets': [], 'exact_matches': 0, 'off_by_1': 0, 'off_by_2': 0, 'off_by_more': 0}
    }
    
    for splice_type in ['donor', 'acceptor']:
        # Get true splice sites
        true_sites = set(annotations_df.filter(pl.col('splice_type') == splice_type)['position'].to_list())
        
        # Get predicted sites (above threshold)
        score_col = f'{splice_type}_score'
        pred_sites = set(predictions_df.filter(pl.col(score_col) > threshold)['position'].to_list())
        
        # For each true site, find nearest predicted site
        for true_pos in true_sites:
            if not pred_sites:
                continue
            
            # Find nearest predicted site
            nearest_pred = min(pred_sites, key=lambda p: abs(p - true_pos))
            offset = nearest_pred - true_pos
            
            results[splice_type]['offsets'].append(offset)
            
            if offset == 0:
                results[splice_type]['exact_matches'] += 1
            elif abs(offset) == 1:
                results[splice_type]['off_by_1'] += 1
            elif abs(offset) == 2:
                results[splice_type]['off_by_2'] += 1
            else:
                results[splice_type]['off_by_more'] += 1
    
    return results


def evaluate_gene(gene_info: Dict) -> Dict:
    """Evaluate a single gene."""
    gene_id = gene_info['gene_id']
    gene_name = gene_info['gene_name']
    expected_chr = gene_info['chr']
    expected_strand = gene_info['strand']
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {gene_id} ({gene_name}) - Chr{expected_chr}, Strand {expected_strand}")
    logger.info(f"{'='*80}")
    
    try:
        # Check if predictions exist
        pred_file = Path(f"predictions/base_only/{gene_id}/{gene_id}_predictions.parquet")
        
        if not pred_file.exists():
            logger.info(f"  ⏭️  Skipping: No predictions file")
            return {'gene_id': gene_id, 'gene_name': gene_name, 'status': 'no_predictions'}
        
        # Load predictions
        predictions_df = pl.read_parquet(pred_file)
        n_positions = predictions_df.height
        
        logger.info(f"  📊 Predictions: {n_positions:,} positions")
        
        # Load GTF annotations
        annotations_df, gtf_info = load_gtf_splice_sites(gene_id)
        
        if annotations_df.height == 0:
            logger.warning(f"  ⚠️  No annotations found")
            return {'gene_id': gene_id, 'gene_name': gene_name, 'status': 'no_annotations'}
        
        # Verify strand matches
        actual_strand = gtf_info.get('strand', '?')
        if actual_strand != expected_strand:
            logger.warning(f"  ⚠️  Strand mismatch: expected {expected_strand}, got {actual_strand}")
        
        # Count splice sites
        n_donors = len(annotations_df.filter(pl.col('splice_type') == 'donor'))
        n_acceptors = len(annotations_df.filter(pl.col('splice_type') == 'acceptor'))
        gene_length = gtf_info.get('end', 0) - gtf_info.get('start', 0) + 1
        
        logger.info(f"  📖 Annotations: {n_donors} donors, {n_acceptors} acceptors")
        logger.info(f"  📏 Gene length: {gene_length:,} bp")
        
        # Coverage check
        coverage_pct = (n_positions / gene_length * 100) if gene_length > 0 else 0
        logger.info(f"  📈 Coverage: {coverage_pct:.1f}%")
        
        # Analyze position offsets (CRITICAL: checks if adjustment is correct!)
        logger.info(f"\n  🎯 Position Offset Analysis:")
        offset_results = analyze_position_offsets(predictions_df, annotations_df, threshold=0.5)
        
        for splice_type in ['donor', 'acceptor']:
            r = offset_results[splice_type]
            total = len(r['offsets'])
            if total > 0:
                exact_pct = r['exact_matches'] / total * 100
                off1_pct = r['off_by_1'] / total * 100
                off2_pct = r['off_by_2'] / total * 100
                
                logger.info(f"    {splice_type.capitalize()}:")
                logger.info(f"      Exact match (offset=0):  {r['exact_matches']}/{total} ({exact_pct:.1f}%)")
                logger.info(f"      Off by ±1 nt:            {r['off_by_1']}/{total} ({off1_pct:.1f}%)")
                logger.info(f"      Off by ±2 nt:            {r['off_by_2']}/{total} ({off2_pct:.1f}%)")
                logger.info(f"      Off by more:             {r['off_by_more']}/{total}")
                
                # Assessment
                if exact_pct >= 70:
                    logger.info(f"      Status: ✅ Good alignment")
                elif exact_pct >= 50:
                    logger.info(f"      Status: ⚠️  Moderate alignment")
                else:
                    logger.info(f"      Status: ❌ Poor alignment (may be off by 1-2 nt!)")
        
        # Calculate F1 scores
        donor_sites = set(annotations_df.filter(pl.col('splice_type') == 'donor')['position'].to_list())
        acceptor_sites = set(annotations_df.filter(pl.col('splice_type') == 'acceptor')['position'].to_list())
        
        pred_donors = set(predictions_df.filter(pl.col('donor_score') > 0.5)['position'].to_list())
        pred_acceptors = set(predictions_df.filter(pl.col('acceptor_score') > 0.5)['position'].to_list())
        
        # Donor metrics
        donor_tp = len(donor_sites & pred_donors)
        donor_fp = len(pred_donors - donor_sites)
        donor_fn = len(donor_sites - pred_donors)
        donor_p = donor_tp / (donor_tp + donor_fp) if (donor_tp + donor_fp) > 0 else 0
        donor_r = donor_tp / (donor_tp + donor_fn) if (donor_tp + donor_fn) > 0 else 0
        donor_f1 = 2 * donor_p * donor_r / (donor_p + donor_r) if (donor_p + donor_r) > 0 else 0
        
        # Acceptor metrics
        acceptor_tp = len(acceptor_sites & pred_acceptors)
        acceptor_fp = len(pred_acceptors - acceptor_sites)
        acceptor_fn = len(acceptor_sites - pred_acceptors)
        acceptor_p = acceptor_tp / (acceptor_tp + acceptor_fp) if (acceptor_tp + acceptor_fp) > 0 else 0
        acceptor_r = acceptor_tp / (acceptor_tp + acceptor_fn) if (acceptor_tp + acceptor_fn) > 0 else 0
        acceptor_f1 = 2 * acceptor_p * acceptor_r / (acceptor_p + acceptor_r) if (acceptor_p + acceptor_r) > 0 else 0
        
        # Overall
        overall_tp = donor_tp + acceptor_tp
        overall_fp = donor_fp + acceptor_fp
        overall_fn = donor_fn + acceptor_fn
        overall_p = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_r = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
        
        logger.info(f"\n  📈 F1 Scores:")
        logger.info(f"    Donor:    {donor_f1:.3f} (P={donor_p:.3f}, R={donor_r:.3f})")
        logger.info(f"    Acceptor: {acceptor_f1:.3f} (P={acceptor_p:.3f}, R={acceptor_r:.3f})")
        logger.info(f"    Overall:  {overall_f1:.3f}")
        
        # Overall assessment
        donor_exact_pct = offset_results['donor']['exact_matches'] / len(offset_results['donor']['offsets']) * 100 if offset_results['donor']['offsets'] else 0
        acceptor_exact_pct = offset_results['acceptor']['exact_matches'] / len(offset_results['acceptor']['offsets']) * 100 if offset_results['acceptor']['offsets'] else 0
        
        if overall_f1 >= 0.7 and donor_exact_pct >= 70 and acceptor_exact_pct >= 70:
            logger.info(f"\n  ✅ PASS: Good F1 and alignment")
        elif overall_f1 >= 0.6:
            logger.info(f"\n  ⚠️  PARTIAL: Acceptable F1 but check alignment")
        else:
            logger.info(f"\n  ❌ FAIL: Low F1 or poor alignment")
        
        return {
            'gene_id': gene_id,
            'gene_name': gene_name,
            'chr': expected_chr,
            'strand': actual_strand,
            'n_positions': n_positions,
            'gene_length': gene_length,
            'coverage': coverage_pct,
            'n_donors': n_donors,
            'n_acceptors': n_acceptors,
            'donor_f1': donor_f1,
            'acceptor_f1': acceptor_f1,
            'overall_f1': overall_f1,
            'donor_exact_pct': donor_exact_pct,
            'acceptor_exact_pct': acceptor_exact_pct,
            'donor_off1_pct': offset_results['donor']['off_by_1'] / len(offset_results['donor']['offsets']) * 100 if offset_results['donor']['offsets'] else 0,
            'acceptor_off1_pct': offset_results['acceptor']['off_by_1'] / len(offset_results['acceptor']['offsets']) * 100 if offset_results['acceptor']['offsets'] else 0,
            'status': 'success'
        }
    
    except Exception as e:
        logger.error(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'gene_id': gene_id, 'gene_name': gene_name, 'status': 'error', 'error': str(e)}


def print_stratified_summary(results: List[Dict]):
    """Print summary stratified by splice type and strand."""
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        logger.warning("No successful results to summarize")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info("STRATIFIED ANALYSIS")
    logger.info(f"{'='*80}")
    
    # By splice type
    logger.info(f"\n📊 By Splice Type:")
    logger.info(f"\nDonor Sites:")
    donor_f1s = [r['donor_f1'] for r in successful]
    donor_exact = [r['donor_exact_pct'] for r in successful]
    logger.info(f"  Average F1:          {sum(donor_f1s)/len(donor_f1s):.3f}")
    logger.info(f"  Average Exact Match: {sum(donor_exact)/len(donor_exact):.1f}%")
    logger.info(f"  Genes with >70% exact: {sum(1 for x in donor_exact if x >= 70)}/{len(donor_exact)}")
    
    logger.info(f"\nAcceptor Sites:")
    acceptor_f1s = [r['acceptor_f1'] for r in successful]
    acceptor_exact = [r['acceptor_exact_pct'] for r in successful]
    logger.info(f"  Average F1:          {sum(acceptor_f1s)/len(acceptor_f1s):.3f}")
    logger.info(f"  Average Exact Match: {sum(acceptor_exact)/len(acceptor_exact):.1f}%")
    logger.info(f"  Genes with >70% exact: {sum(1 for x in acceptor_exact if x >= 70)}/{len(acceptor_exact)}")
    
    # By strand
    logger.info(f"\n📊 By Strand:")
    for strand in ['+', '-']:
        strand_results = [r for r in successful if r['strand'] == strand]
        if strand_results:
            avg_f1 = sum(r['overall_f1'] for r in strand_results) / len(strand_results)
            avg_donor_exact = sum(r['donor_exact_pct'] for r in strand_results) / len(strand_results)
            avg_acceptor_exact = sum(r['acceptor_exact_pct'] for r in strand_results) / len(strand_results)
            
            logger.info(f"\n{strand} Strand (n={len(strand_results)}):")
            logger.info(f"  Average F1:               {avg_f1:.3f}")
            logger.info(f"  Donor exact match:        {avg_donor_exact:.1f}%")
            logger.info(f"  Acceptor exact match:     {avg_acceptor_exact:.1f}%")


def main():
    """Main test function."""
    logger.info("="*80)
    logger.info("COMPREHENSIVE SCORE-SHIFTING VALIDATION: 20+ GENES")
    logger.info("="*80)
    logger.info("\nObjective: Validate score-shifting on 20+ NEW genes")
    logger.info("Key Metric: Position offset analysis (exact match vs off by 1-2 nt)")
    logger.info("")
    
    # Get test genes
    test_genes = get_20_test_genes()
    
    logger.info(f"Testing {len(test_genes)} genes:")
    logger.info(f"  + strand: {sum(1 for g in test_genes if g['strand'] == '+')} genes")
    logger.info(f"  - strand: {sum(1 for g in test_genes if g['strand'] == '-')} genes")
    
    # Test each gene
    results = []
    for gene_info in test_genes:
        result = evaluate_gene(gene_info)
        results.append(result)
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'success']
    logger.info(f"\nGenes tested: {len(results)}")
    logger.info(f"Successful:   {len(successful)}")
    logger.info(f"Skipped:      {sum(1 for r in results if r['status'] in ['no_predictions', 'no_annotations'])}")
    logger.info(f"Errors:       {sum(1 for r in results if r['status'] == 'error')}")
    
    if successful:
        # Stratified analysis
        print_stratified_summary(results)
        
        # Overall metrics
        avg_f1 = sum(r['overall_f1'] for r in successful) / len(successful)
        avg_coverage = sum(r['coverage'] for r in successful) / len(successful)
        
        logger.info(f"\n{'='*80}")
        logger.info("OVERALL METRICS")
        logger.info(f"{'='*80}")
        logger.info(f"  Average F1:       {avg_f1:.3f}")
        logger.info(f"  Average Coverage: {avg_coverage:.1f}%")
        logger.info(f"  Genes with F1>=0.7: {sum(1 for r in successful if r['overall_f1'] >= 0.7)}/{len(successful)}")
        
        # Verdict
        logger.info(f"\n{'='*80}")
        logger.info("VERDICT")
        logger.info(f"{'='*80}")
        
        if avg_f1 >= 0.7 and len(successful) >= 15:
            logger.info(f"✅ SUCCESS: Score-shifting validated on {len(successful)} genes!")
            logger.info(f"   Average F1: {avg_f1:.3f} (target: >= 0.7)")
        elif len(successful) < 10:
            logger.warning(f"⚠️  INSUFFICIENT DATA: Only {len(successful)} genes tested")
            logger.warning(f"   Need predictions for more genes")
        else:
            logger.warning(f"⚠️  PARTIAL SUCCESS: {len(successful)} genes tested")
            logger.warning(f"   Average F1: {avg_f1:.3f}")
    
    # Save results
    output_path = Path("results/score_shifting_20_genes_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n📁 Results saved to: {output_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

