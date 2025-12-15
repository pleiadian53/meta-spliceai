"""Enhanced Output Messaging Utilities
=====================================
Provides color-coded output functions for better visibility during long-running processes.
"""

from typing import Dict, List, Optional, Set
from pathlib import Path


class OutputEnhancer:
    """Enhanced output messaging with color coding and progress tracking."""
    
    # ANSI color codes
    STYLES = {
        'bold': '\033[1m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'yellow': '\033[93m',
        'magenta': '\033[95m',
        'red': '\033[91m',
        'white': '\033[97m'
    }
    END_STYLE = '\033[0m'
    
    def __init__(self, verbose: int = 1):
        """Initialize the output enhancer.
        
        Parameters
        ----------
        verbose
            Verbosity level. If 0, color coding is disabled.
        """
        self.verbose = verbose
        self._disable_colors = verbose == 0
    
    def _apply_style(self, text: str, style: str) -> str:
        """Apply color style to text if colors are enabled."""
        if self._disable_colors:
            return text
        return f"{self.STYLES.get(style, '')}{text}{self.END_STYLE}"
    
    def print_batch_header(self, batch_ix: int, total_batches: int, gene_count: int) -> None:
        """Print a color-coded batch header for easy progress tracking.
        
        Parameters
        ----------
        batch_ix
            Current batch index (1-based)
        total_batches
            Total number of batches
        gene_count
            Number of genes in this batch
        """
        prefix = f"batch_{batch_ix:05d}"
        progress = f"[{batch_ix}/{total_batches}]"
        
        # Determine batch type and color
        if batch_ix == total_batches and gene_count < 50:
            color = 'yellow'
            batch_type = "FINAL"
        elif batch_ix <= 3:
            color = 'green'
            batch_type = "MAIN"
        else:
            color = 'blue'
            batch_type = "MAIN"
        
        header = f"[{prefix}] {progress} {batch_type} BATCH - Processing {gene_count} genes"
        colored_header = self._apply_style(header, color)
        
        print("\n" + "="*80)
        print(colored_header)
        print("="*80)
    
    def print_gene_selection_summary(self, all_gene_ids: List[str], n_genes: int, 
                                   subset_policy: str, normalized_additional_genes: Set[str]) -> None:
        """Print a color-coded gene selection summary.
        
        Parameters
        ----------
        all_gene_ids
            List of all selected gene IDs
        n_genes
            Number of genes requested via policy
        subset_policy
            Gene selection policy used
        normalized_additional_genes
            Set of additional genes (normalized to Ensembl IDs)
        """
        header = self._apply_style("🧬 GENE SELECTION COMPLETED", "green bold")
        print("\n" + "="*80)
        print(header)
        print("="*80)
        
        summary_label = self._apply_style("📊 Gene Selection Summary:", "cyan")
        print(summary_label)
        
        total_genes = len(all_gene_ids)
        total_display = self._apply_style(str(total_genes), "bold")
        print(f"   🎯 Total Genes Selected: {total_display}")
        
        additional_count = len(normalized_additional_genes) if normalized_additional_genes else 0
        
        if subset_policy == "all":
            print(f"   🌐 All {total_genes} available genes selected")
            if additional_count > 0:
                print(f"   ⚠️  Gene file ignored (--subset-policy all selects all genes)")
        elif additional_count > 0:
            random_count = total_genes - additional_count
            print(f"   📁 {additional_count} genes from user file")
            print(f"   🎲 {random_count} additional genes via '{subset_policy}' policy")
        else:
            print(f"   🎲 All {total_genes} genes selected via '{subset_policy}' policy")
        
        print("="*80)
    
    def print_completion_summary(self, master_dir: Path, total_rows: str, 
                               all_gene_ids: List[str], total_batches: int) -> None:
        """Print a color-coded completion summary.
        
        Parameters
        ----------
        master_dir
            Path to the master dataset directory
        total_rows
            Total number of rows in the dataset
        all_gene_ids
            List of all gene IDs in the dataset
        total_batches
            Total number of batches processed
        """
        completion_msg = self._apply_style("🎉 INCREMENTAL BUILDER COMPLETED SUCCESSFULLY! 🎉", "green bold")
        print("\n" + "="*80)
        print(completion_msg)
        print("="*80)
        
        summary_label = self._apply_style("📊 Master Dataset Summary:", "cyan")
        print(summary_label)
        print(f"   📁 Location: {master_dir}")
        print(f"   📈 Total Rows: {total_rows}")
        print(f"   🧬 Total Genes: {len(all_gene_ids)}")
        print(f"   📦 Total Batches: {total_batches}")
        print("="*80)
    
    def print_workflow_start(self, missing_genes: Set[str]) -> None:
        """Print workflow start message.
        
        Parameters
        ----------
        missing_genes
            Set of genes that need workflow processing
        """
        if not missing_genes:
            return
            
        header = self._apply_style("🔄 SPLICE PREDICTION WORKFLOW STARTING", "yellow bold")
        print("\n" + "="*80)
        print(header)
        print("="*80)
        
        summary_label = self._apply_style("📋 Workflow Summary:", "cyan")
        print(summary_label)
        print(f"   🎯 Processing {len(missing_genes)} missing genes")
        print(f"   🔧 Generating required artifacts")
        print("="*80)
    
    def print_validation_summary(self, expected_genes: int, actual_genes: int, 
                               coverage_pct: float) -> None:
        """Print artifact coverage validation summary.
        
        Parameters
        ----------
        expected_genes
            Number of genes expected
        actual_genes
            Number of genes found in artifacts
        coverage_pct
            Coverage percentage
        """
        if coverage_pct < 100:
            status = self._apply_style("⚠️  PARTIAL COVERAGE", "yellow bold")
        else:
            status = self._apply_style("✅ FULL COVERAGE", "green bold")
        
        print("\n" + "="*80)
        print(status)
        print("="*80)
        
        summary_label = self._apply_style("📊 Artifact Coverage Summary:", "cyan")
        print(summary_label)
        print(f"   🎯 Expected Genes: {expected_genes}")
        print(f"   📁 Found Genes: {actual_genes}")
        print(f"   📈 Coverage: {coverage_pct:.1f}%")
        print("="*80)


def create_output_enhancer(verbose: int = 1) -> OutputEnhancer:
    """Create an output enhancer instance.
    
    Parameters
    ----------
    verbose
        Verbosity level
        
    Returns
    -------
    OutputEnhancer
        Configured output enhancer instance
    """
    return OutputEnhancer(verbose=verbose) 