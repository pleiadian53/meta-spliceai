#!/usr/bin/env python3
"""
Generalized Inference Analyzer for Splice Site Prediction

A modular, scalable analysis tool that can handle arbitrary numbers of genes
(from tens to thousands) in the splice site prediction inference workflow.

COMPLETE WORKFLOW EXAMPLE:
=========================

This analyzer is designed to work with results from the main inference workflow.
Here's the complete process from gene discovery to analysis:

Step 0: Identify Test Genes (Optional)
--------------------------------------
If you don't have a specific gene set, use the gene finder to identify suitable
test genes for different scenarios:

# Find default gene sets (6 training, 8 unseen with artifacts, 8 unseen without artifacts)
bash meta_spliceai/splice_engine/meta_models/workflows/inference/find_test_genes.sh

# Find larger gene sets for comprehensive testing
bash meta_spliceai/splice_engine/meta_models/workflows/inference/find_test_genes.sh \
    --scenario1-count 20 \
    --scenario2a-count 50 \
    --scenario2b-count 100 \
    --output large_test_genes.json \
    --verbose

# Focus on unseen genes for generalization testing
bash meta_spliceai/splice_engine/meta_models/workflows/inference/find_test_genes.sh \
    --scenario1-count 5 \
    --scenario2a-count 10 \
    --scenario2b-count 50 \
    --output unseen_focus_genes.json

# Use the generated gene lists in subsequent steps
export TEST_GENES_FILE="test_genes.json"  # or your custom output file

# Extract gene lists for use in inference commands
python3 -c "
import json
with open('$TEST_GENES_FILE') as f:
    data = json.load(f)
    
# Print gene lists for easy copying
for scenario in ['scenario1', 'scenario2a', 'scenario2b']:
    if data[scenario]['genes']:
        genes = [g['gene_id'] for g in data[scenario]['genes']]
        print(f'{scenario}_genes = {genes}')
"

Step 1: Run Main Inference Workflow
-----------------------------------
First, generate inference results using the main workflow for different modes:

# Base-only mode (SpliceAI only)
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4 \
    --training-dataset train_pc_1000_3mers \
    --genes-file test_genes.txt \
    --output-dir results/test_scenario_base \
    --inference-mode base_only \
    --enable-chunked-processing \
    --chunk-size 5000 \
    --verbose

# Hybrid mode (SpliceAI + Meta-model for uncertain positions)
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4 \
    --training-dataset train_pc_1000_3mers \
    --genes-file test_genes.txt \
    --output-dir results/test_scenario_hybrid \
    --inference-mode hybrid \
    --enable-chunked-processing \
    --chunk-size 5000 \
    --verbose

# Meta-only mode (Meta-model for all positions)
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4 \
    --training-dataset train_pc_1000_3mers \
    --genes-file test_genes.txt \
    --output-dir results/test_scenario_meta \
    --inference-mode meta_only \
    --enable-chunked-processing \
    --chunk-size 5000 \
    --verbose

Step 2: Analyze Results with Inference Analyzer
-----------------------------------------------
Once inference is complete, analyze and compare the results:

# Option A: Specify individual directories
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.inference_analyzer \
    --base-dir results/test_scenario_base \
    --hybrid-dir results/test_scenario_hybrid \
    --meta-dir results/test_scenario_meta \
    --output-dir analysis_results \
    --batch-size 50 \
    --verbose

# Option B: Use common results directory (recommended for organized workflows)
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.inference_analyzer \
    --results-dir results \
    --base-suffix test_scenario_base \
    --hybrid-suffix test_scenario_hybrid \
    --meta-suffix test_scenario_meta \
    --output-dir analysis_results \
    --batch-size 50 \
    --verbose

Step 3: Review Analysis Outputs
-------------------------------
The analyzer generates comprehensive reports:

analysis_results/
├── summary_report.json          # High-level metrics across all modes
├── detailed_report.json         # Per-gene detailed metrics
├── per_gene_metrics.csv         # Tabular data for further analysis
└── analysis_log.txt             # Processing log

Key Metrics Generated:
- F1 Score, Average Precision (AP), ROC-AUC for each mode
- Per-gene performance breakdown
- Optimal thresholds for each gene/mode combination
- Statistical summaries and distributions
- Gene-level metadata (length, chromosome, etc.)

USAGE PATTERNS:
==============

1. Small-scale Analysis (1-20 genes):
   - Use default batch size (100)
   - Include --verbose for detailed progress
   - Analyze specific genes with --gene-id option

2. Medium-scale Analysis (20-200 genes):
   - Set --batch-size 25-50 for memory efficiency
   - Use --results-dir for cleaner command lines
   - Monitor progress with standard verbosity

3. Large-scale Analysis (200+ genes):
   - Set --batch-size 10-25 to manage memory
   - Consider running on high-memory systems
   - Use --output-dir on fast storage for I/O performance

PERFORMANCE CONSIDERATIONS:
==========================

Memory Usage:
- ~10-50MB per gene depending on gene size and prediction density
- Batch processing keeps peak memory manageable
- Large genes (>500kb) may require smaller batch sizes

Processing Speed:
- ~0.1-2 seconds per gene for metric calculation
- I/O bound for large numbers of small genes
- CPU bound for small numbers of large genes

Storage Requirements:
- Input: Parquet files from inference workflow (~1-10MB per gene)
- Output: JSON/CSV reports (~100KB-1MB total for most analyses)

INTEGRATION WITH OTHER TOOLS:
============================

This analyzer integrates with:
- batch_comparator.py: For statistical comparison between modes
- MLflow: Automatic logging of analysis results
- Jupyter notebooks: CSV outputs work directly with pandas
- Custom analysis scripts: JSON outputs provide programmatic access

For advanced comparative analysis, pipe results to batch_comparator.py:
python batch_comparator.py --results-file analysis_results/detailed_report.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from datetime import datetime
import warnings
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import argparse
import sys

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeneMetrics:
    """Metrics for a single gene."""
    gene_id: str
    total_positions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    ap_score: float
    auc_score: float
    optimal_threshold: float
    mode: str
    processing_time: Optional[float] = None
    gene_length: Optional[int] = None
    chromosome: Optional[str] = None

@dataclass
class ModeComparison:
    """Comparison metrics between inference modes."""
    mode: str
    total_genes: int
    total_positions: int
    overall_precision: float
    overall_recall: float
    overall_f1: float
    overall_ap: float
    overall_auc: float
    mean_gene_f1: float
    std_gene_f1: float
    genes_with_perfect_recall: int
    genes_with_zero_fps: int

class InferenceAnalyzer:
    """
    Main analysis framework for inference results.
    
    Handles results from 1 to 10,000+ genes with streaming/partial results support.
    """
    
    def __init__(self, output_dirs: Dict[str, str], batch_size: int = 100):
        """
        Initialize the analyzer.
        
        Args:
            output_dirs: Dict mapping modes to output directories
            batch_size: Process genes in batches for memory efficiency
        """
        self.output_dirs = output_dirs
        self.batch_size = batch_size
        self.results_cache = {}
        self.metrics_cache = {}
        
        # Validate directories
        for mode, dir_path in output_dirs.items():
            if not Path(dir_path).exists():
                logger.warning(f"Directory not found for {mode}: {dir_path}")
    
    def load_gene_predictions(self, mode: str, gene_id: str) -> Optional[pd.DataFrame]:
        """Load predictions for a specific gene and mode."""
        mode_dir = Path(self.output_dirs[mode])
        
        # Try different possible file locations
        possible_paths = [
            mode_dir / "genes" / gene_id / f"{gene_id}_predictions.parquet",
            mode_dir / "selective_inference" / "predictions" / f"{gene_id}_predictions.parquet",
            mode_dir / f"{gene_id}_predictions.parquet"
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    df['source_gene'] = gene_id
                    df['source_mode'] = mode
                    return df
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
        
        return None
    
    def get_available_genes(self, mode: str) -> List[str]:
        """Get list of available genes for a mode."""
        mode_dir = Path(self.output_dirs[mode])
        genes = []
        
        # Check genes subdirectory
        genes_dir = mode_dir / "genes"
        if genes_dir.exists():
            genes = [d.name for d in genes_dir.iterdir() if d.is_dir()]
        
        # Check selective_inference directory
        selective_dir = mode_dir / "selective_inference" / "predictions"
        if selective_dir.exists():
            for file in selective_dir.glob("*_predictions.parquet"):
                gene_id = file.stem.replace("_predictions", "")
                # Only add if it looks like a gene ID (starts with ENSG)
                if gene_id.startswith("ENSG") and gene_id not in genes:
                    genes.append(gene_id)
        
        return genes
    
    def calculate_gene_metrics(self, df: pd.DataFrame, mode: str, 
                             gene_id: str) -> Optional[GeneMetrics]:
        """Calculate metrics for a single gene."""
        if df.empty:
            return None
        
        # Get true labels
        y_true = (df['splice_type'].isin(['donor', 'acceptor'])).astype(int)
        
        # Get predictions based on mode
        if mode == 'meta_only' and 'donor_meta' in df.columns:
            y_scores = np.maximum(
                df['donor_meta'].fillna(0).values,
                df['acceptor_meta'].fillna(0).values
            )
        else:
            y_scores = np.maximum(
                df['donor_score'].fillna(0).values,
                df['acceptor_score'].fillna(0).values
            )
        
        # Find optimal threshold for F1
        best_f1 = 0
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_scores >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Calculate metrics at optimal threshold
        y_pred = (y_scores >= best_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases where confusion matrix is not 2x2
            if len(cm) == 1:
                if y_true.sum() == 0:  # All negatives
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                else:  # All positives
                    tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate AUC and AP
        try:
            ap = average_precision_score(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
        except:
            ap = 0
            auc = 0
        
        # Extract gene metadata
        gene_length = None
        chromosome = None
        if 'gene_length' in df.columns:
            gene_length = df['gene_length'].iloc[0] if len(df) > 0 else None
        if 'chrom' in df.columns:
            chromosome = df['chrom'].iloc[0] if len(df) > 0 else None
        
        return GeneMetrics(
            gene_id=gene_id,
            total_positions=len(df),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=precision,
            recall=recall,
            f1_score=best_f1,
            ap_score=ap,
            auc_score=auc,
            optimal_threshold=best_threshold,
            mode=mode,
            gene_length=gene_length,
            chromosome=chromosome
        )
    
    def analyze_batch(self, gene_ids: List[str], mode: str) -> List[GeneMetrics]:
        """Process a batch of genes for a specific mode."""
        metrics = []
        
        for gene_id in gene_ids:
            df = self.load_gene_predictions(mode, gene_id)
            if df is not None:
                gene_metrics = self.calculate_gene_metrics(df, mode, gene_id)
                if gene_metrics:
                    metrics.append(gene_metrics)
                    logger.debug(f"Processed {gene_id} for {mode}")
                else:
                    logger.warning(f"No metrics calculated for {gene_id} in {mode}")
            else:
                logger.warning(f"No predictions found for {gene_id} in {mode}")
        
        return metrics
    
    def analyze_all_modes(self) -> Dict[str, List[GeneMetrics]]:
        """Analyze all modes and return comprehensive results."""
        all_results = {}
        
        for mode in self.output_dirs.keys():
            logger.info(f"Analyzing {mode} mode...")
            available_genes = self.get_available_genes(mode)
            
            if not available_genes:
                logger.warning(f"No genes found for {mode}")
                continue
            
            # Process in batches
            all_metrics = []
            for i in range(0, len(available_genes), self.batch_size):
                batch = available_genes[i:i + self.batch_size]
                batch_metrics = self.analyze_batch(batch, mode)
                all_metrics.extend(batch_metrics)
                logger.info(f"Processed batch {i//self.batch_size + 1}/{(len(available_genes) + self.batch_size - 1)//self.batch_size}")
            
            all_results[mode] = all_metrics
            logger.info(f"Completed {mode}: {len(all_metrics)} genes")
        
        return all_results
    
    def generate_summary_report(self, results: Dict[str, List[GeneMetrics]]) -> Dict:
        """Create high-level summary for all genes."""
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'modes_analyzed': list(results.keys()),
            'total_genes': {mode: len(metrics) for mode, metrics in results.items()},
            'mode_comparisons': {}
        }
        
        for mode, metrics in results.items():
            if not metrics:
                continue
            
            # Aggregate metrics
            total_positions = sum(m.total_positions for m in metrics)
            total_tp = sum(m.true_positives for m in metrics)
            total_fp = sum(m.false_positives for m in metrics)
            total_fn = sum(m.false_negatives for m in metrics)
            total_tn = sum(m.true_negatives for m in metrics)
            
            # Overall metrics
            overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
            
            # Gene-level statistics
            f1_scores = [m.f1_score for m in metrics]
            ap_scores = [m.ap_score for m in metrics]
            auc_scores = [m.auc_score for m in metrics]
            
            genes_with_perfect_recall = sum(1 for m in metrics if m.recall == 1.0)
            genes_with_zero_fps = sum(1 for m in metrics if m.false_positives == 0)
            
            summary['mode_comparisons'][mode] = {
                'total_genes': len(metrics),
                'total_positions': total_positions,
                'overall_precision': overall_precision,
                'overall_recall': overall_recall,
                'overall_f1': overall_f1,
                'overall_ap': np.mean(ap_scores),
                'overall_auc': np.mean(auc_scores),
                'mean_gene_f1': np.mean(f1_scores),
                'std_gene_f1': np.std(f1_scores),
                'genes_with_perfect_recall': genes_with_perfect_recall,
                'genes_with_zero_fps': genes_with_zero_fps,
                'f1_score_range': [min(f1_scores), max(f1_scores)],
                'ap_score_range': [min(ap_scores), max(ap_scores)],
                'auc_score_range': [min(auc_scores), max(auc_scores)]
            }
        
        return summary
    
    def generate_detailed_report(self, results: Dict[str, List[GeneMetrics]], 
                               gene_id: Optional[str] = None) -> Dict:
        """Create detailed report for specific gene(s) or all genes."""
        if gene_id:
            # Filter for specific gene
            filtered_results = {}
            for mode, metrics in results.items():
                filtered_results[mode] = [m for m in metrics if m.gene_id == gene_id]
        else:
            filtered_results = results
        
        detailed_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'gene_specific': gene_id is not None,
            'target_gene': gene_id,
            'per_gene_metrics': {}
        }
        
        # Collect all unique genes
        all_genes = set()
        for metrics in filtered_results.values():
            all_genes.update(m.gene_id for m in metrics)
        
        for gene in sorted(all_genes):
            gene_metrics = {}
            for mode, metrics in filtered_results.items():
                mode_metrics = [m for m in metrics if m.gene_id == gene]
                if mode_metrics:
                    gene_metrics[mode] = asdict(mode_metrics[0])
            
            if gene_metrics:
                detailed_report['per_gene_metrics'][gene] = gene_metrics
        
        return detailed_report
    
    def save_results(self, results: Dict[str, List[GeneMetrics]], 
                    output_dir: str, summary: Dict, detailed: Dict):
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary report
        with open(output_path / "summary_report.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            json.dump(convert_numpy(summary), f, indent=2)
        
        # Save detailed report
        with open(output_path / "detailed_report.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            json.dump(convert_numpy(detailed), f, indent=2)
        
        # Save per-gene metrics as CSV
        all_metrics = []
        for mode, metrics in results.items():
            for metric in metrics:
                metric_dict = asdict(metric)
                all_metrics.append(metric_dict)
        
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(output_path / "per_gene_metrics.csv", index=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def run_analysis(self, output_dir: str) -> Dict:
        """Run complete analysis pipeline."""
        logger.info("Starting inference analysis...")
        
        # Analyze all modes
        results = self.analyze_all_modes()
        
        # Generate reports
        summary = self.generate_summary_report(results)
        detailed = self.generate_detailed_report(results)
        
        # Save results
        self.save_results(results, output_dir, summary, detailed)
        
        logger.info("Analysis completed successfully!")
        return results

def main():
    """Command-line interface for the inference analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze inference results from main_inference_workflow.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

# Option A: Specify individual directories
python inference_analyzer.py \\
    --base-dir results/test_scenario_base \\
    --hybrid-dir results/test_scenario_hybrid \\
    --meta-dir results/test_scenario_meta \\
    --output-dir analysis_results

# Option B: Use common results directory (recommended)
python inference_analyzer.py \\
    --results-dir results \\
    --base-suffix test_scenario_base \\
    --hybrid-suffix test_scenario_hybrid \\
    --meta-suffix test_scenario_meta \\
    --output-dir analysis_results

# Analyze specific gene only
python inference_analyzer.py \\
    --results-dir results \\
    --base-suffix test_scenario_base \\
    --hybrid-suffix test_scenario_hybrid \\
    --meta-suffix test_scenario_meta \\
    --gene-id ENSG00000123456 \\
    --output-dir gene_specific_analysis
        """
    )
    
    # Directory specification options
    group1 = parser.add_argument_group('Individual Directory Specification')
    group1.add_argument("--base-dir", help="Base model results directory")
    group1.add_argument("--hybrid-dir", help="Hybrid model results directory")
    group1.add_argument("--meta-dir", help="Meta model results directory")
    
    group2 = parser.add_argument_group('Common Directory Specification (Recommended)')
    group2.add_argument("--results-dir", help="Common results directory containing all mode subdirectories")
    group2.add_argument("--base-suffix", default="base", 
                       help="Suffix for base model directory (default: base)")
    group2.add_argument("--hybrid-suffix", default="hybrid",
                       help="Suffix for hybrid model directory (default: hybrid)")
    group2.add_argument("--meta-suffix", default="meta",
                       help="Suffix for meta model directory (default: meta)")
    
    # Analysis options
    parser.add_argument("--output-dir", default="analysis_results", 
                       help="Output directory for analysis results")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing genes")
    parser.add_argument("--gene-id", help="Analyze specific gene only")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Build output directories dict
    output_dirs = {}
    
    # Check if using common results directory approach
    if args.results_dir:
        results_path = Path(args.results_dir)
        if not results_path.exists():
            print(f"Error: Results directory not found: {args.results_dir}")
            sys.exit(1)
        
        # Build paths using suffixes
        potential_dirs = {
            'base_only': results_path / args.base_suffix,
            'hybrid': results_path / args.hybrid_suffix,
            'meta_only': results_path / args.meta_suffix
        }
        
        # Add directories that exist
        for mode, dir_path in potential_dirs.items():
            if dir_path.exists():
                output_dirs[mode] = str(dir_path)
            else:
                print(f"Warning: {mode} directory not found: {dir_path}")
    
    # Individual directory specification (takes precedence)
    if args.base_dir:
        output_dirs['base_only'] = args.base_dir
    if args.hybrid_dir:
        output_dirs['hybrid'] = args.hybrid_dir
    if args.meta_dir:
        output_dirs['meta_only'] = args.meta_dir
    
    if not output_dirs:
        print("Error: Must specify at least one results directory")
        print("Use either --results-dir with suffixes, or individual --base-dir/--hybrid-dir/--meta-dir")
        sys.exit(1)
    
    print(f"Found {len(output_dirs)} mode directories:")
    for mode, path in output_dirs.items():
        print(f"  {mode}: {path}")
    
    # Run analysis
    analyzer = InferenceAnalyzer(output_dirs, batch_size=args.batch_size)
    results = analyzer.run_analysis(args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    for mode, metrics in results.items():
        if metrics:
            print(f"\n{mode.upper()}:")
            print(f"  Genes processed: {len(metrics)}")
            print(f"  Mean F1: {np.mean([m.f1_score for m in metrics]):.3f}")
            print(f"  Mean AP: {np.mean([m.ap_score for m in metrics]):.3f}")
            print(f"  Mean AUC: {np.mean([m.auc_score for m in metrics]):.3f}")

if __name__ == "__main__":
    main()
