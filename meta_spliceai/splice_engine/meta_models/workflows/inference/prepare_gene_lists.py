#!/usr/bin/env python3
"""
Streamlined Gene List Preparation for Inference Workflow

One-stop utility that finds suitable genes and creates ready-to-use gene list files
for the inference workflow. No intermediate steps required.

SIMPLIFIED WORKFLOW:
1. Run this script with desired gene types and counts
2. Use the generated .txt files directly with main_inference_workflow.py
3. Copy-paste the generated inference commands

GENE TYPES:
- 'training' or 'training_genes': Genes used in meta-model training
- 'unseen' or 'unseen_genes': Genes NOT used in meta-model training  
- 'mixed': Combination of training and unseen genes

GENE TYPE FILTERING:
- Use --gene-types to filter by gene types (e.g., protein_coding, lncRNA)
- Consistent with strategic_gene_selector.py and incremental_builder.py
- Maintains gene type consistency across the entire workflow
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional
import sys
import os

# Import data resource manager for systematic path resolution
from .data_resource_manager import create_inference_data_manager

class StreamlinedGenePreparator:
    """Streamlined gene preparation with integrated discovery."""
    
    def __init__(self, training_dataset: str = "train_pc_1000_3mers", verbose: bool = False):
        self.training_dataset = training_dataset
        self.verbose = verbose
        self.project_root = self._find_project_root()
        
        # Initialize data resource manager for systematic path resolution
        self.data_manager = create_inference_data_manager(
            project_root=self.project_root,
            auto_detect=True
        )
        
        self.log(f"📁 Project root: {self.project_root}")
        self.log(f"🎯 Training dataset: {training_dataset}")
        self.log(f"🔧 Data manager initialized with {self.data_manager.genome_build} / Ensembl {self.data_manager.ensembl_release}")
        

        
    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current_dir = Path(__file__).resolve()
        
        # Look for characteristic project files
        key_files = [
            "data/ensembl/spliceai_analysis/gene_features.tsv",
            "meta_spliceai/__init__.py",
        ]
        
        # Start from current directory and go up
        for parent in [current_dir] + list(current_dir.parents):
            for key_file in key_files:
                if (parent / key_file).exists():
                    if self.verbose:
                        print(f"✅ Found project root: {parent}")
                    return parent
        
        # Fallback: assume current working directory
        cwd = Path.cwd()
        if self.verbose:
            print(f"⚠️  Using current working directory as project root: {cwd}")
        return cwd
    
    def log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load_training_genes(self) -> Set[str]:
        """Load genes from training dataset."""
        try:
            manifest_path = self.project_root / self.training_dataset / "master" / "gene_manifest.csv"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
            
            train_manifest = pd.read_csv(manifest_path)
            training_genes = set(train_manifest['gene_id'].tolist())
            
            self.log(f"✅ Loaded {len(training_genes)} training genes")
            return training_genes
            
        except Exception as e:
            print(f"❌ Error loading training genes: {e}")
            sys.exit(1)
    
    def load_all_genes(self) -> pd.DataFrame:
        """Load all available genes with metadata."""
        try:
            # Use data resource manager for systematic path resolution
            gene_features_path = self.data_manager.get_gene_features_path()
            if not gene_features_path or not gene_features_path.exists():
                raise FileNotFoundError(f"Gene features not found via data manager (expected: {gene_features_path})")
            
            gene_features = pd.read_csv(gene_features_path, sep='\t')
            
            self.log(f"✅ Loaded {len(gene_features)} total genes with metadata")
            return gene_features
            
        except Exception as e:
            print(f"❌ Error loading gene features: {e}")
            sys.exit(1)
    
    def select_training_genes(self, training_genes: Set[str], count: int,
                            gene_types: Optional[List[str]] = None,
                            min_length: int = 10000, max_length: int = 500000) -> List[str]:
        """Select diverse training genes with optional gene type filtering."""
        gene_features = self.load_all_genes()
        
        # Filter to training genes with reasonable length
        training_gene_features = gene_features[
            (gene_features['gene_id'].isin(training_genes)) &
            (gene_features['gene_length'] >= min_length) &
            (gene_features['gene_length'] <= max_length)
        ]
        
        # Apply gene type filter if specified
        if gene_types:
            training_gene_features = training_gene_features[
                training_gene_features['gene_type'].isin(gene_types)
            ]
            self.log(f"🧬 Filtered to gene types: {gene_types}")
        
        if training_gene_features.empty:
            # Fallback to any training genes (with gene type filter if specified)
            training_gene_features = gene_features[gene_features['gene_id'].isin(training_genes)]
            if gene_types:
                training_gene_features = training_gene_features[
                    training_gene_features['gene_type'].isin(gene_types)
                ]
                if training_gene_features.empty:
                    self.log(f"⚠️ No training genes found with gene types {gene_types}")
                    return []
        
        # Select diverse genes across chromosomes
        selected_genes = []
        chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
                      '11', '12', '13', '14', '15', '16', '17', '18', '19', 
                      '20', '21', '22', 'X', 'Y']
        
        genes_per_chrom = max(1, count // len(chromosomes))
        
        for chrom in chromosomes:
            if len(selected_genes) >= count:
                break
                
            chrom_genes = training_gene_features[training_gene_features['chrom'] == chrom]
            if not chrom_genes.empty:
                # Sort by gene length for consistent selection
                chrom_genes = chrom_genes.sort_values('gene_length')
                
                # Take genes from this chromosome
                n_to_take = min(genes_per_chrom, len(chrom_genes), count - len(selected_genes))
                selected = chrom_genes.head(n_to_take)['gene_id'].tolist()
                selected_genes.extend(selected)
        
        # Fill remaining slots if needed
        if len(selected_genes) < count:
            remaining = training_gene_features[~training_gene_features['gene_id'].isin(selected_genes)]
            if not remaining.empty:
                additional_needed = count - len(selected_genes)
                additional = remaining.head(additional_needed)['gene_id'].tolist()
                selected_genes.extend(additional)
        
        self.log(f"✅ Selected {len(selected_genes)} training genes (requested: {count})")
        return selected_genes[:count]
    
    def select_unseen_genes(self, training_genes: Set[str], count: int,
                          gene_types: Optional[List[str]] = None,
                          min_length: int = 5000, max_length: int = 200000) -> List[str]:
        """Select diverse unseen genes (not in training) with optional gene type filtering."""
        gene_features = self.load_all_genes()
        
        # Filter to unseen genes with reasonable length AND that have splice sites
        # Use data resource manager for systematic splice sites path resolution
        try:
            splice_sites_path = self.data_manager.get_splice_sites_path()
            if splice_sites_path and splice_sites_path.exists():
                # Load genes that have splice sites
                splice_sites_df = pd.read_csv(splice_sites_path, sep='\t')
                genes_with_splice_sites = set(splice_sites_df['gene_id'].unique())
                self.log(f"✅ Found {len(genes_with_splice_sites)} genes with splice sites from {splice_sites_path}")
            else:
                genes_with_splice_sites = set()
                self.log(f"⚠️  Splice sites file not found via data manager (expected path: {splice_sites_path})")
        except Exception as e:
            genes_with_splice_sites = set()
            self.log(f"⚠️  Error loading splice sites via data manager: {e}")
        
        # Filter to unseen genes with splice sites and reasonable length
        unseen_gene_features = gene_features[
            (~gene_features['gene_id'].isin(training_genes)) &
            (gene_features['gene_length'] >= min_length) &
            (gene_features['gene_length'] <= max_length) &
            (gene_features['gene_id'].isin(genes_with_splice_sites))  # Only genes with splice sites
        ]
        
        # Apply gene type filter if specified
        if gene_types:
            unseen_gene_features = unseen_gene_features[
                unseen_gene_features['gene_type'].isin(gene_types)
            ]
            self.log(f"🧬 Filtered unseen genes to types: {gene_types}")
        
        if unseen_gene_features.empty:
            # Fallback to any unseen genes with splice sites (with gene type filter if specified)
            unseen_gene_features = gene_features[
                (~gene_features['gene_id'].isin(training_genes)) &
                (gene_features['gene_id'].isin(genes_with_splice_sites))
            ]
            if gene_types:
                unseen_gene_features = unseen_gene_features[
                    unseen_gene_features['gene_type'].isin(gene_types)
                ]
                if unseen_gene_features.empty:
                    self.log(f"⚠️ No unseen genes found with gene types {gene_types}")
        
        if unseen_gene_features.empty:
            if gene_types:
                print(f"❌ Warning: No unseen genes found with gene types {gene_types} and splice sites")
            else:
                print("❌ Warning: No unseen genes with splice sites found")
            return []
        
        # Select diverse genes across chromosomes AND gene lengths
        selected_genes = []
        chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
                      '11', '12', '13', '14', '15', '16', '17', '18', '19', 
                      '20', '21', '22', 'X', 'Y']
        
        # Create length bins for diversity: small, medium, large genes
        length_bins = [
            (min_length, 10000, "small"),
            (10000, 50000, "medium"), 
            (50000, max_length, "large")
        ]
        
        genes_per_chrom = max(1, count // len(chromosomes))
        
        for chrom in chromosomes:
            if len(selected_genes) >= count:
                break
                
            chrom_genes = unseen_gene_features[unseen_gene_features['chrom'] == chrom]
            if not chrom_genes.empty:
                # Select genes from different length bins for diversity
                chrom_selected = []
                for min_len, max_len, bin_name in length_bins:
                    bin_genes = chrom_genes[
                        (chrom_genes['gene_length'] >= min_len) & 
                        (chrom_genes['gene_length'] < max_len)
                    ]
                    if not bin_genes.empty:
                        # Take one gene from this length bin
                        selected_gene = bin_genes.sample(n=1)['gene_id'].iloc[0]
                        chrom_selected.append(selected_gene)
                        if len(chrom_selected) >= genes_per_chrom:
                            break
                
                # If we need more genes from this chromosome, take any remaining
                if len(chrom_selected) < genes_per_chrom:
                    remaining_chrom_genes = chrom_genes[~chrom_genes['gene_id'].isin(chrom_selected)]
                    if not remaining_chrom_genes.empty:
                        additional_needed = min(genes_per_chrom - len(chrom_selected), len(remaining_chrom_genes))
                        additional = remaining_chrom_genes.sample(n=additional_needed)['gene_id'].tolist()
                        chrom_selected.extend(additional)
                
                selected_genes.extend(chrom_selected)
        
        # Fill remaining slots if needed with any available genes
        if len(selected_genes) < count:
            remaining = unseen_gene_features[~unseen_gene_features['gene_id'].isin(selected_genes)]
            if not remaining.empty:
                additional_needed = count - len(selected_genes)
                additional = remaining.sample(n=min(additional_needed, len(remaining)))['gene_id'].tolist()
                selected_genes.extend(additional)
        
        # Show length distribution of selected genes
        if self.verbose and selected_genes:
            selected_features = gene_features[gene_features['gene_id'].isin(selected_genes)]
            length_stats = selected_features['gene_length'].describe()
            self.log(f"✅ Selected {len(selected_genes)} unseen genes with splice sites")
            self.log(f"   Length distribution: min={int(length_stats['min'])}, "
                    f"median={int(length_stats['50%'])}, max={int(length_stats['max'])} bp")
        
        return selected_genes[:count]
    
    def prepare_gene_lists(self, gene_requests: Dict[str, int], output_dir: str = ".",
                          prefix: str = "", create_combined: bool = False,
                          gene_types: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Prepare gene lists based on requests.
        
        Args:
            gene_requests: Dict mapping gene types to counts
                          e.g., {'training': 10, 'unseen': 20}
            output_dir: Directory to save gene files
            prefix: Prefix for output filenames
            create_combined: Whether to create combined file
            gene_types: Optional list of gene types to filter by
                       e.g., ['protein_coding'] or ['protein_coding', 'lncRNA']
            
        Returns:
            Dict mapping gene types to output file paths
        """
        print("🧬 PREPARING GENE LISTS FOR INFERENCE WORKFLOW")
        print("="*60)
        if gene_types:
            print(f"🧬 Gene type filter: {gene_types}")
        else:
            print("🧬 Gene type filter: None (all gene types)")
        
        # Load data
        training_genes = self.load_training_genes()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare gene lists
        output_files = {}
        all_genes = []
        
        for gene_type, count in gene_requests.items():
            if count <= 0:
                continue
                
            print(f"\n🎯 Selecting {count} {gene_type} genes...")
            
            # Select genes based on type
            if gene_type in ['training', 'training_genes']:
                selected_genes = self.select_training_genes(training_genes, count, gene_types=gene_types)
                file_suffix = "training_genes"
            elif gene_type in ['unseen', 'unseen_genes']:
                selected_genes = self.select_unseen_genes(training_genes, count, gene_types=gene_types)
                file_suffix = "unseen_genes"
            elif gene_type in ['mixed']:
                # Half training, half unseen
                train_count = count // 2
                unseen_count = count - train_count
                train_genes = self.select_training_genes(training_genes, train_count, gene_types=gene_types)
                unseen_genes = self.select_unseen_genes(training_genes, unseen_count, gene_types=gene_types)
                selected_genes = train_genes + unseen_genes
                file_suffix = "mixed_genes"
            else:
                print(f"⚠️ Unknown gene type: {gene_type}")
                continue
            
            if not selected_genes:
                print(f"❌ No genes selected for {gene_type}")
                continue
            
            # Create filename
            if prefix:
                filename = f"{prefix}_{file_suffix}.txt"
            else:
                filename = f"{file_suffix}.txt"
            
            output_file = output_path / filename
            
            # Write gene list
            with open(output_file, 'w') as f:
                for gene in selected_genes:
                    f.write(f"{gene}\n")
            
            output_files[gene_type] = str(output_file)
            all_genes.extend(selected_genes)
            
            print(f"✅ {gene_type}: {len(selected_genes)} genes → {output_file}")
        
        # Create combined file if requested
        if create_combined and all_genes:
            combined_filename = f"{prefix}_all_genes.txt" if prefix else "all_genes.txt"
            combined_file = output_path / combined_filename
            
            with open(combined_file, 'w') as f:
                for gene in all_genes:
                    f.write(f"{gene}\n")
            
            output_files['combined'] = str(combined_file)
            print(f"✅ Combined: {len(all_genes)} genes → {combined_file}")
        
        return output_files
    
    def generate_inference_commands(self, output_files: Dict[str, str], 
                                  study_name: str, model_path: str, 
                                  training_dataset: str) -> None:
        """Generate ready-to-use inference commands for each gene set type."""
        
        print("\n" + "="*80)
        print("🧪 READY-TO-USE INFERENCE COMMANDS")
        print("="*80)
        
        # Generate commands for each gene file separately to avoid conflicts
        for gene_type, gene_file in output_files.items():
            if gene_type == 'combined':
                continue  # Skip combined file to avoid duplication
                
            print(f"\n" + "="*60)
            print(f"📋 COMMANDS FOR {gene_type.upper()} GENES")
            print("="*60)
            print(f"Gene file: {gene_file}")
            print(f"Study name: {study_name}")
            
            # Create gene-type-specific identifiers to prevent conflicts
            gene_suffix = gene_type.replace('_genes', '').replace('_', '')  # unseen_genes -> unseen
            
            # Generate commands for each mode
            modes = [
                ("base_only", "Base-only (SpliceAI predictions only)"),
                ("hybrid", "Hybrid (SpliceAI + Meta-model for uncertain positions)"),
                ("meta_only", "Meta-only (Meta-model recalibration for all positions)")
            ]
            
            print(f"\n📋 COPY-PASTE COMMANDS:")
            print("-" * 40)
            
            for mode, description in modes:
                print(f"\n# {description} - {gene_type} genes")
                print(f"python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \\")
                print(f"    --model {model_path} \\")
                print(f"    --training-dataset {training_dataset} \\")
                print(f"    --genes-file {gene_file} \\")
                print(f"    --output-dir results/{study_name}_{gene_suffix}_{mode} \\")
                print(f"    --inference-mode {mode} \\")
                print(f"    --enable-chunked-processing \\")
                print(f"    --chunk-size 5000 \\")
                
                # Add mode-specific options
                if mode == "hybrid":
                    print(f"    --uncertainty-low 0.02 \\")
                    print(f"    --uncertainty-high 0.80 \\")
                
                print(f"    --verbose \\")
                print(f"    --mlflow-enable \\")
                print(f"    --mlflow-experiment \"{study_name}_{gene_suffix}\" \\")
                print(f"    2>&1 | tee logs/{gene_suffix}_{mode}_inference.log")
                print(f"")
            
            print(f"# Note: Complete coverage and chunked processing are auto-enabled for all modes")
        
        # If there's a combined file, show commands for it too
        if 'combined' in output_files:
            print(f"\n" + "="*60)
            print(f"📋 COMMANDS FOR COMBINED GENE SET")
            print("="*60)
            print(f"Gene file: {output_files['combined']}")
            
            for mode, description in modes:
                print(f"\n# {description} - combined gene set")
                print(f"python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \\")
                print(f"    --model {model_path} \\")
                print(f"    --training-dataset {training_dataset} \\")
                print(f"    --genes-file {output_files['combined']} \\")
                print(f"    --output-dir results/{study_name}_combined_{mode} \\")
                print(f"    --inference-mode {mode} \\")
                print(f"    --enable-chunked-processing \\")
                print(f"    --chunk-size 5000 \\")
                
                if mode == "hybrid":
                    print(f"    --uncertainty-low 0.02 \\")
                    print(f"    --uncertainty-high 0.80 \\")
                
                print(f"    --verbose \\")
                print(f"    --mlflow-enable \\")
                print(f"    --mlflow-experiment \"{study_name}_combined\" \\")
                print(f"    2>&1 | tee logs/combined_{mode}_inference.log")
                print(f"")
        
        # Analysis commands for each gene set
        print(f"\n" + "="*80)
        print("📊 ANALYSIS COMMANDS (Run after all three modes complete)")
        print("="*80)
        
        # Generate analysis commands for each gene set
        for gene_type in output_files.keys():
            if gene_type == 'combined':
                continue  # Handle combined separately
                
            gene_suffix = gene_type.replace('_genes', '').replace('_', '')
            
            print(f"\n" + "="*60)
            print(f"📊 ANALYSIS FOR {gene_type.upper()} GENES")
            print("="*60)
            
            print(f"\n# Analyze and compare {gene_type} results")
            print(f"python -m meta_spliceai.splice_engine.meta_models.workflows.inference.inference_analyzer \\")
            print(f"    --results-dir results \\")
            print(f"    --base-suffix {study_name}_{gene_suffix}_base_only \\")
            print(f"    --hybrid-suffix {study_name}_{gene_suffix}_hybrid \\")
            print(f"    --meta-suffix {study_name}_{gene_suffix}_meta_only \\")
            print(f"    --output-dir results/{study_name}_{gene_suffix}_analysis_results \\")
            print(f"    --batch-size 25 \\")
            print(f"    --verbose")
            
            print(f"\n# Statistical comparison for {gene_type} genes")
            print(f"python -m meta_spliceai.splice_engine.meta_models.workflows.inference.batch_comparator \\")
            print(f"    --analysis-results results/{study_name}_{gene_suffix}_analysis_results/detailed_report.json \\")
            print(f"    --output-dir results/{study_name}_{gene_suffix}_statistical_comparison \\")
            print(f"    --reference-mode base_only \\")
            print(f"    --create-plots")
        
        # Combined analysis if applicable
        if 'combined' in output_files:
            print(f"\n" + "="*60)
            print(f"📊 ANALYSIS FOR COMBINED GENE SET")
            print("="*60)
            
            print(f"\n# Analyze and compare combined results")
            print(f"python -m meta_spliceai.splice_engine.meta_models.workflows.inference.inference_analyzer \\")
            print(f"    --results-dir results \\")
            print(f"    --base-suffix {study_name}_combined_base_only \\")
            print(f"    --hybrid-suffix {study_name}_combined_hybrid \\")
            print(f"    --meta-suffix {study_name}_combined_meta_only \\")
            print(f"    --output-dir results/{study_name}_combined_analysis_results \\")
            print(f"    --batch-size 25 \\")
            print(f"    --verbose")
            
            print(f"\n# Statistical comparison for combined gene set")
            print(f"python -m meta_spliceai.splice_engine.meta_models.workflows.inference.batch_comparator \\")
            print(f"    --analysis-results results/{study_name}_combined_analysis_results/detailed_report.json \\")
            print(f"    --output-dir results/{study_name}_combined_statistical_comparison \\")
            print(f"    --reference-mode base_only \\")
            print(f"    --create-plots")
        
        print(f"\n💡 TIPS:")
        print(f"• Create log directory first: mkdir -p logs")
        print(f"• Each gene set generates separate results to avoid conflicts")
        print(f"• Compare results between gene sets to analyze generalization")

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Streamlined gene list preparation for inference workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

# Quick start: 10 unseen genes for generalization testing
python prepare_gene_lists.py --unseen 10 --study-name "generalization_test"

# Training genes study: 20 genes from training set
python prepare_gene_lists.py --training 20 --study-name "training_validation"

# Mixed study: 15 training + 25 unseen genes
python prepare_gene_lists.py \\
    --training 15 \\
    --unseen 25 \\
    --study-name "comprehensive_study" \\
    --create-combined

# Large-scale unseen genes study
python prepare_gene_lists.py \\
    --unseen 100 \\
    --study-name "large_generalization" \\
    --prefix "large_study" \\
    --output-dir gene_lists \\
    --verbose

# Custom gene size filtering
python prepare_gene_lists.py \\
    --unseen 50 \\
    --min-length 20000 \\
    --max-length 100000 \\
    --study-name "medium_genes_study"

# Gene type filtering (protein-coding only)
python prepare_gene_lists.py \\
    --training 15 \\
    --unseen 25 \\
    --gene-types protein_coding \\
    --study-name "protein_coding_study"

# Multiple gene types
python prepare_gene_lists.py \\
    --unseen 30 \\
    --gene-types protein_coding lncRNA \\
    --study-name "pc_lnc_study"

GENE TYPES:
- training/training_genes: Genes used in meta-model training (good performance expected)
- unseen/unseen_genes: Genes NOT in training (tests generalization)
- mixed: Combination of both types

GENE TYPE FILTERING:
- Use --gene-types to filter by gene types (e.g., protein_coding, lncRNA)
- Consistent with strategic_gene_selector.py and incremental_builder.py
- If not specified, all gene types are included

OUTPUT:
- Individual .txt files ready for --genes-file parameter
- Ready-to-use inference commands with proper parameters
- Analysis commands for post-inference comparison
        """
    )
    
    # Gene selection arguments
    parser.add_argument("--training", "--training-genes", type=int, default=0,
                       help="Number of training genes to select")
    parser.add_argument("--unseen", "--unseen-genes", type=int, default=0,
                       help="Number of unseen genes to select")
    parser.add_argument("--mixed", type=int, default=0,
                       help="Number of mixed genes (half training, half unseen)")
    
    # Selection criteria
    parser.add_argument("--min-length", type=int, default=5000,
                       help="Minimum gene length (default: 5000)")
    parser.add_argument("--max-length", type=int, default=500000,
                       help="Maximum gene length (default: 500000)")
    parser.add_argument("--gene-types", nargs='*',
                       help="Gene types to include (e.g., protein_coding lncRNA). If not specified, all gene types are included.")
    
    # Output options
    parser.add_argument("--output-dir", default=".",
                       help="Output directory for gene files (default: current directory)")
    parser.add_argument("--prefix", default="",
                       help="Prefix for output filenames")
    parser.add_argument("--create-combined", action="store_true",
                       help="Create combined file with all selected genes")
    
    # Study configuration
    parser.add_argument("--study-name", default="comparison_study",
                       help="Study name for generated commands (default: comparison_study)")
    parser.add_argument("--model-path", default="results/gene_cv_pc_1000_3mers_run_4",
                       help="Model path for generated commands")
    parser.add_argument("--training-dataset", default="train_pc_1000_3mers",
                       help="Training dataset path")
    
    # Processing options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate arguments
    total_genes = args.training + args.unseen + args.mixed
    if total_genes == 0:
        print("❌ Error: Must specify at least one gene type with count > 0")
        print("Use --training, --unseen, or --mixed with positive counts")
        sys.exit(1)
    
    if args.min_length >= args.max_length:
        print("❌ Error: min-length must be less than max-length")
        sys.exit(1)
    
    # Build gene requests
    gene_requests = {}
    if args.training > 0:
        gene_requests['training'] = args.training
    if args.unseen > 0:
        gene_requests['unseen'] = args.unseen
    if args.mixed > 0:
        gene_requests['mixed'] = args.mixed
    
    print(f"Gene selection requests: {gene_requests}")
    print(f"Gene length range: {args.min_length:,} - {args.max_length:,} bp")
    print(f"Study name: {args.study_name}")
    
    # Run gene preparation
    preparator = StreamlinedGenePreparator(
        training_dataset=args.training_dataset,
        verbose=args.verbose
    )
    
    output_files = preparator.prepare_gene_lists(
        gene_requests=gene_requests,
        output_dir=args.output_dir,
        prefix=args.prefix,
        create_combined=args.create_combined,
        gene_types=args.gene_types
    )
    
    if not output_files:
        print("❌ No gene files created")
        sys.exit(1)
    
    print(f"\n✅ Successfully created {len(output_files)} gene list files")
    
    # Generate usage examples
    preparator.generate_inference_commands(
        output_files=output_files,
        study_name=args.study_name,
        model_path=args.model_path,
        training_dataset=args.training_dataset
    )
    
    print(f"\n📁 Gene list files saved to: {args.output_dir}")
    print(f"🚀 Ready to run inference workflow!")

if __name__ == "__main__":
    main()
