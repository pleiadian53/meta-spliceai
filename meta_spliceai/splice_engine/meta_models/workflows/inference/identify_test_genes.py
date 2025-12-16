#!/usr/bin/env python3
"""
🧬 Test Gene Identification Utility

Systematically identifies example genes for testing all three major inference scenarios:
- Scenario 1: Genes in training data with unseen positions (due to TN downsampling)
- Scenario 2A: Unseen genes (not in training) with existing artifacts
- Scenario 2B: Unseen genes (not in training) that are completely unprocessed

Usage:
    python identify_test_genes.py [--training-dataset DATASET] [--output OUTPUT] [--verbose]
"""

import argparse
import json
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys
import os

class TestGeneIdentifier:
    """Identifies genes for testing different inference scenarios."""
    
    def __init__(self, training_dataset: str = "train_pc_1000_3mers", verbose: bool = False):
        self.training_dataset = training_dataset
        self.verbose = verbose
        self.results = {}
        self.project_root = self._find_project_root()
        
    def _find_project_root(self) -> Path:
        """Find the project root directory by looking for key files."""
        current_dir = Path(__file__).resolve()
        
        # Look for characteristic project files
        key_files = [
            "data/ensembl/spliceai_analysis/gene_features.tsv",
            "meta_spliceai/__init__.py",
            "setup.py",
            "pyproject.toml"
        ]
        
        # Start from current directory and go up
        for parent in [current_dir] + list(current_dir.parents):
            for key_file in key_files:
                if (parent / key_file).exists():
                    self.log(f"✅ Found project root: {parent}")
                    return parent
        
        # Fallback: assume current working directory
        cwd = Path.cwd()
        self.log(f"⚠️  Using current working directory as project root: {cwd}")
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
            
            self.log(f"✅ Loaded {len(training_genes)} training genes from {manifest_path}")
            return training_genes
            
        except Exception as e:
            print(f"❌ Error loading training genes: {e}")
            sys.exit(1)
    
    def load_all_genes(self) -> Set[str]:
        """Load all available genes from gene features."""
        try:
            gene_features_path = self.project_root / "data/ensembl/spliceai_analysis/gene_features.tsv"
            if not gene_features_path.exists():
                raise FileNotFoundError(f"Gene features not found: {gene_features_path}")
            
            gene_features = pd.read_csv(gene_features_path, sep='\t')
            all_genes = set(gene_features['gene_id'].tolist())
            
            self.log(f"✅ Loaded {len(all_genes)} total genes from {gene_features_path}")
            return all_genes
            
        except Exception as e:
            print(f"❌ Error loading gene features: {e}")
            sys.exit(1)
    
    def find_artifact_directories(self) -> List[Path]:
        """Find directories containing analysis_sequences artifacts."""
        artifact_dirs = []
        
        # Check common artifact locations
        possible_dirs = [
            self.project_root / "data/ensembl/spliceai_eval/meta_models",
            self.project_root / "data/ensembl/spliceai_analysis/meta_models",
            self.project_root / "results" / "meta_models",
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists():
                # Look for analysis_sequences files
                analysis_files = list(dir_path.glob("analysis_sequences_*.tsv"))
                if analysis_files:
                    artifact_dirs.append(dir_path)
                    self.log(f"✅ Found artifact directory: {dir_path} ({len(analysis_files)} files)")
        
        return artifact_dirs
    
    def load_genes_with_artifacts(self, artifact_dirs: List[Path]) -> Set[str]:
        """Load genes that have existing artifacts."""
        genes_with_artifacts = set()
        
        for artifact_dir in artifact_dirs:
            analysis_files = list(artifact_dir.glob("analysis_sequences_*.tsv"))
            
            for file_path in analysis_files:
                try:
                    # Read just the gene_id column to avoid loading large files
                    if file_path.stat().st_size > 100 * 1024 * 1024:  # > 100MB
                        # For large files, sample first 1000 rows
                        df = pd.read_csv(file_path, sep='\t', usecols=['gene_id'], nrows=1000)
                    else:
                        df = pd.read_csv(file_path, sep='\t', usecols=['gene_id'])
                    
                    file_genes = set(df['gene_id'].unique())
                    genes_with_artifacts.update(file_genes)
                    
                    self.log(f"   📄 {file_path.name}: {len(file_genes)} unique genes")
                    
                except Exception as e:
                    self.log(f"   ⚠️  Error reading {file_path.name}: {e}")
                    continue
        
        self.log(f"✅ Found {len(genes_with_artifacts)} genes with existing artifacts")
        return genes_with_artifacts
    
    def identify_scenario1_genes(self, training_genes: Set[str], count: int = 6) -> List[str]:
        """Identify Scenario 1 genes: In training data with unseen positions."""
        # For Scenario 1, we want genes that are in training
        # We'll select a diverse set from different chromosomes
        
        try:
            gene_features_path = self.project_root / "data/ensembl/spliceai_analysis/gene_features.tsv"
            gene_features = pd.read_csv(gene_features_path, sep='\t')
            
            # Filter to training genes only
            training_gene_features = gene_features[gene_features['gene_id'].isin(training_genes)]
            
            # Filter for genes with reasonable length (avoid tiny or huge genes)
            medium_genes = training_gene_features[
                (training_gene_features['gene_length'] >= 10000) & 
                (training_gene_features['gene_length'] <= 500000)
            ]
            
            if medium_genes.empty:
                # Fallback to any training genes if no medium-sized ones
                medium_genes = training_gene_features
            
            # Sample genes from different chromosomes for diversity
            scenario1_genes = []
            chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
                          '11', '12', '13', '14', '15', '16', '17', '18', '19', 
                          '20', '21', '22', 'X', 'Y']
            
            genes_per_chrom = max(1, count // len(chromosomes))
            
            for chrom in chromosomes:
                if len(scenario1_genes) >= count:
                    break
                    
                chrom_genes = medium_genes[medium_genes['chrom'] == chrom]
                if not chrom_genes.empty:
                    # Sort by gene length for consistent selection
                    chrom_genes = chrom_genes.sort_values('gene_length')
                    
                    # Take up to genes_per_chrom genes from this chromosome
                    n_to_take = min(genes_per_chrom, len(chrom_genes), count - len(scenario1_genes))
                    selected_genes = chrom_genes.head(n_to_take)['gene_id'].tolist()
                    scenario1_genes.extend(selected_genes)
            
            # If we still need more genes, add from any remaining training genes
            if len(scenario1_genes) < count:
                remaining_genes = medium_genes[~medium_genes['gene_id'].isin(scenario1_genes)]
                if not remaining_genes.empty:
                    additional_needed = count - len(scenario1_genes)
                    additional_genes = remaining_genes.head(additional_needed)['gene_id'].tolist()
                    scenario1_genes.extend(additional_genes)
            
            self.log(f"✅ Identified {len(scenario1_genes)} Scenario 1 genes (requested: {count})")
            return scenario1_genes[:count]  # Ensure we don't exceed requested count
            
        except Exception as e:
            print(f"❌ Error identifying Scenario 1 genes: {e}")
            return list(training_genes)[:count]  # Fallback
    
    def identify_scenario2a_genes(self, training_genes: Set[str], 
                                 genes_with_artifacts: Set[str], count: int = 8) -> List[str]:
        """Identify Scenario 2A genes: Unseen genes with existing artifacts."""
        # Genes that have artifacts but are NOT in training
        scenario2a_candidates = genes_with_artifacts - training_genes
        
        if not scenario2a_candidates:
            self.log("⚠️  No Scenario 2A genes found (no artifacts for unseen genes)")
            return []
        
        try:
            # Get gene features for better selection
            gene_features_path = self.project_root / "data/ensembl/spliceai_analysis/gene_features.tsv"
            gene_features = pd.read_csv(gene_features_path, sep='\t')
            
            # Filter to scenario 2A candidates
            candidate_features = gene_features[gene_features['gene_id'].isin(scenario2a_candidates)]
            
            if candidate_features.empty:
                # Fallback to simple list selection
                scenario2a_genes = list(scenario2a_candidates)[:count]
            else:
                # Select diverse genes from different chromosomes
                scenario2a_genes = []
                chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
                              '11', '12', '13', '14', '15', '16', '17', '18', '19', 
                              '20', '21', '22', 'X', 'Y']
                
                genes_per_chrom = max(1, count // len(chromosomes))
                
                for chrom in chromosomes:
                    if len(scenario2a_genes) >= count:
                        break
                        
                    chrom_genes = candidate_features[candidate_features['chrom'] == chrom]
                    if not chrom_genes.empty:
                        # Sort by gene length for consistent selection
                        chrom_genes = chrom_genes.sort_values('gene_length')
                        
                        # Take up to genes_per_chrom genes from this chromosome
                        n_to_take = min(genes_per_chrom, len(chrom_genes), count - len(scenario2a_genes))
                        selected_genes = chrom_genes.head(n_to_take)['gene_id'].tolist()
                        scenario2a_genes.extend(selected_genes)
                
                # If we still need more genes, add from any remaining candidates
                if len(scenario2a_genes) < count:
                    remaining_genes = candidate_features[~candidate_features['gene_id'].isin(scenario2a_genes)]
                    if not remaining_genes.empty:
                        additional_needed = count - len(scenario2a_genes)
                        additional_genes = remaining_genes.head(additional_needed)['gene_id'].tolist()
                        scenario2a_genes.extend(additional_genes)
                    else:
                        # Final fallback
                        remaining_candidates = scenario2a_candidates - set(scenario2a_genes)
                        scenario2a_genes.extend(list(remaining_candidates)[:count - len(scenario2a_genes)])
        
        except Exception as e:
            self.log(f"⚠️  Error in advanced Scenario 2A selection: {e}")
            # Fallback to simple selection
            scenario2a_genes = list(scenario2a_candidates)[:count]
        
        self.log(f"✅ Identified {len(scenario2a_genes)} Scenario 2A genes (requested: {count})")
        return scenario2a_genes[:count]
    
    def identify_scenario2b_genes(self, training_genes: Set[str], 
                                 genes_with_artifacts: Set[str],
                                 all_genes: Set[str], count: int = 8) -> List[str]:
        """Identify Scenario 2B genes: Unseen genes with NO existing artifacts."""
        # Genes that are NOT in training AND do NOT have artifacts
        scenario2b_candidates = all_genes - training_genes - genes_with_artifacts
        
        if not scenario2b_candidates:
            self.log("⚠️  No Scenario 2B genes found")
            return []
        
        try:
            # Select diverse genes from different chromosomes
            gene_features_path = self.project_root / "data/ensembl/spliceai_analysis/gene_features.tsv"
            gene_features = pd.read_csv(gene_features_path, sep='\t')
            
            # Filter to scenario 2B candidates
            candidate_features = gene_features[gene_features['gene_id'].isin(scenario2b_candidates)]
            
            if candidate_features.empty:
                # Fallback to simple selection
                scenario2b_genes = list(scenario2b_candidates)[:count]
            else:
                # Filter for genes with reasonable length (avoid tiny or huge genes)
                medium_genes = candidate_features[
                    (candidate_features['gene_length'] >= 5000) & 
                    (candidate_features['gene_length'] <= 200000)
                ]
                
                if medium_genes.empty:
                    # If no medium genes, use all candidates
                    medium_genes = candidate_features
                
                # Sample genes from different chromosomes for diversity
                scenario2b_genes = []
                chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
                              '11', '12', '13', '14', '15', '16', '17', '18', '19', 
                              '20', '21', '22', 'X', 'Y']
                
                genes_per_chrom = max(1, count // len(chromosomes))
                
                for chrom in chromosomes:
                    if len(scenario2b_genes) >= count:
                        break
                        
                    chrom_genes = medium_genes[medium_genes['chrom'] == chrom]
                    if not chrom_genes.empty:
                        # Sort by gene length for consistent selection
                        chrom_genes = chrom_genes.sort_values('gene_length')
                        
                        # Take up to genes_per_chrom genes from this chromosome
                        n_to_take = min(genes_per_chrom, len(chrom_genes), count - len(scenario2b_genes))
                        selected_genes = chrom_genes.head(n_to_take)['gene_id'].tolist()
                        scenario2b_genes.extend(selected_genes)
                
                # If we still need more genes, add from any remaining candidates
                if len(scenario2b_genes) < count:
                    remaining_genes = medium_genes[~medium_genes['gene_id'].isin(scenario2b_genes)]
                    if not remaining_genes.empty:
                        additional_needed = count - len(scenario2b_genes)
                        additional_genes = remaining_genes.head(additional_needed)['gene_id'].tolist()
                        scenario2b_genes.extend(additional_genes)
                    else:
                        # Final fallback
                        remaining_candidates = scenario2b_candidates - set(scenario2b_genes)
                        scenario2b_genes.extend(list(remaining_candidates)[:count - len(scenario2b_genes)])
            
            self.log(f"✅ Identified {len(scenario2b_genes)} Scenario 2B genes (requested: {count})")
            return scenario2b_genes[:count]
            
        except Exception as e:
            print(f"❌ Error identifying Scenario 2B genes: {e}")
            return list(scenario2b_candidates)[:count]  # Fallback
    
    def get_gene_details(self, genes: List[str]) -> List[Dict]:
        """Get detailed information about genes."""
        try:
            gene_features_path = self.project_root / "data/ensembl/spliceai_analysis/gene_features.tsv"
            gene_features = pd.read_csv(gene_features_path, sep='\t')
            
            details = []
            for gene_id in genes:
                gene_info = gene_features[gene_features['gene_id'] == gene_id]
                if not gene_info.empty:
                    info = gene_info.iloc[0]
                    details.append({
                        'gene_id': gene_id,
                        'gene_name': info.get('gene_name', 'Unknown'),
                        'chrom': info.get('chrom', 'Unknown'),
                        'gene_length': int(info.get('gene_length', 0)),
                        'gene_type': info.get('gene_type', 'Unknown')
                    })
                else:
                    details.append({
                        'gene_id': gene_id,
                        'gene_name': 'Unknown',
                        'chrom': 'Unknown',
                        'gene_length': 0,
                        'gene_type': 'Unknown'
                    })
            
            return details
            
        except Exception as e:
            self.log(f"⚠️  Error getting gene details: {e}")
            return [{'gene_id': gene_id} for gene_id in genes]
    
    def run_identification(self, scenario1_count: int = 6, scenario2a_count: int = 8, 
                          scenario2b_count: int = 8) -> Dict:
        """Run complete gene identification for all scenarios."""
        print("🧬 IDENTIFYING TEST GENES FOR INFERENCE SCENARIOS")
        print("=" * 60)
        
        # Load data
        training_genes = self.load_training_genes()
        all_genes = self.load_all_genes()
        artifact_dirs = self.find_artifact_directories()
        genes_with_artifacts = self.load_genes_with_artifacts(artifact_dirs)
        
        print(f"\n📊 GENE STATISTICS:")
        print(f"  Total genes in reference: {len(all_genes):,}")
        print(f"  Genes in training data: {len(training_genes):,}")
        print(f"  Genes with artifacts: {len(genes_with_artifacts):,}")
        print(f"  Unseen genes (not in training): {len(all_genes - training_genes):,}")
        print(f"  Unseen genes with artifacts: {len(genes_with_artifacts - training_genes):,}")
        print(f"  Unseen genes without artifacts: {len(all_genes - training_genes - genes_with_artifacts):,}")
        
        # Identify genes for each scenario
        print(f"\n🎯 IDENTIFYING SCENARIO GENES:")
        print(f"  Scenario 1 (training genes): {scenario1_count} requested")
        print(f"  Scenario 2A (unseen + artifacts): {scenario2a_count} requested")
        print(f"  Scenario 2B (unseen, no artifacts): {scenario2b_count} requested")
        
        scenario1_genes = self.identify_scenario1_genes(training_genes, scenario1_count)
        scenario2a_genes = self.identify_scenario2a_genes(training_genes, genes_with_artifacts, scenario2a_count)
        scenario2b_genes = self.identify_scenario2b_genes(training_genes, genes_with_artifacts, all_genes, scenario2b_count)
        
        # Get detailed information
        scenario1_details = self.get_gene_details(scenario1_genes)
        scenario2a_details = self.get_gene_details(scenario2a_genes)
        scenario2b_details = self.get_gene_details(scenario2b_genes)
        
        # Compile results
        results = {
            'metadata': {
                'training_dataset': self.training_dataset,
                'total_genes': len(all_genes),
                'training_genes': len(training_genes),
                'genes_with_artifacts': len(genes_with_artifacts),
                'artifact_directories': [str(d) for d in artifact_dirs],
                'requested_counts': {
                    'scenario1': scenario1_count,
                    'scenario2a': scenario2a_count,
                    'scenario2b': scenario2b_count
                }
            },
            'scenario1': {
                'description': 'Genes in training data with unseen positions (due to TN downsampling)',
                'count': len(scenario1_genes),
                'requested': scenario1_count,
                'genes': scenario1_details
            },
            'scenario2a': {
                'description': 'Unseen genes (not in training) with existing artifacts',
                'count': len(scenario2a_genes),
                'requested': scenario2a_count,
                'genes': scenario2a_details
            },
            'scenario2b': {
                'description': 'Unseen genes (not in training) that are completely unprocessed',
                'count': len(scenario2b_genes),
                'requested': scenario2b_count,
                'genes': scenario2b_details
            }
        }
        
        return results
    
    def print_results(self, results: Dict) -> None:
        """Print formatted results."""
        print("\n" + "=" * 60)
        print("📋 TEST GENE IDENTIFICATION RESULTS")
        print("=" * 60)
        
        metadata = results['metadata']
        print(f"\n📊 Dataset Statistics:")
        print(f"  Training Dataset: {metadata['training_dataset']}")
        print(f"  Total Genes: {metadata['total_genes']:,}")
        print(f"  Training Genes: {metadata['training_genes']:,}")
        print(f"  Genes with Artifacts: {metadata['genes_with_artifacts']:,}")
        print(f"  Artifact Directories: {len(metadata['artifact_directories'])}")
        
        # Print each scenario
        for scenario_key in ['scenario1', 'scenario2a', 'scenario2b']:
            scenario = results[scenario_key]
            print(f"\n🎯 {scenario_key.upper()}: {scenario['description']}")
            print(f"   Count: {scenario['count']} genes")
            
            if scenario['genes']:
                print(f"   Example genes:")
                for i, gene in enumerate(scenario['genes'][:5]):  # Show first 5
                    gene_name = gene.get('gene_name', 'Unknown')
                    chrom = gene.get('chrom', '?')
                    length = gene.get('gene_length', 0)
                    print(f"     {i+1}. {gene['gene_id']} ({gene_name}) - Chr{chrom}, {length:,}bp")
                
                if len(scenario['genes']) > 5:
                    print(f"     ... and {len(scenario['genes']) - 5} more")
            else:
                print(f"     ⚠️  No genes found for this scenario")
    
    def save_results(self, results: Dict, output_file: str) -> None:
        """Save results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Identify test genes for inference scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

# Basic usage with default counts (6, 8, 8 genes)
python identify_test_genes.py --output test_genes.json

# Request specific numbers of genes for each scenario
python identify_test_genes.py \\
    --scenario1-count 10 \\
    --scenario2a-count 15 \\
    --scenario2b-count 20 \\
    --output large_gene_set.json

# Generate a large test set for comprehensive analysis
python identify_test_genes.py \\
    --scenario1-count 50 \\
    --scenario2a-count 100 \\
    --scenario2b-count 100 \\
    --output comprehensive_test_genes.json \\
    --verbose

# Focus on scenario 2B (unseen genes) for generalization testing
python identify_test_genes.py \\
    --scenario1-count 5 \\
    --scenario2a-count 5 \\
    --scenario2b-count 50 \\
    --output scenario2b_focus.json
        """
    )
    
    parser.add_argument("--training-dataset", default="train_pc_1000_3mers",
                       help="Training dataset directory (default: train_pc_1000_3mers)")
    parser.add_argument("--output", default="test_genes_identification.json",
                       help="Output JSON file (default: test_genes_identification.json)")
    
    # Gene count arguments
    parser.add_argument("--scenario1-count", type=int, default=6,
                       help="Number of Scenario 1 genes (training genes) to identify (default: 6)")
    parser.add_argument("--scenario2a-count", type=int, default=8,
                       help="Number of Scenario 2A genes (unseen with artifacts) to identify (default: 8)")
    parser.add_argument("--scenario2b-count", type=int, default=8,
                       help="Number of Scenario 2B genes (unseen without artifacts) to identify (default: 8)")
    
    # Selection strategy arguments
    parser.add_argument("--min-gene-length", type=int, default=5000,
                       help="Minimum gene length for selection (default: 5000)")
    parser.add_argument("--max-gene-length", type=int, default=500000,
                       help="Maximum gene length for selection (default: 500000)")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.scenario1_count < 0 or args.scenario2a_count < 0 or args.scenario2b_count < 0:
        print("❌ Error: Gene counts must be non-negative")
        sys.exit(1)
    
    if args.scenario1_count + args.scenario2a_count + args.scenario2b_count == 0:
        print("❌ Error: At least one scenario must have count > 0")
        sys.exit(1)
    
    # Run identification
    identifier = TestGeneIdentifier(
        training_dataset=args.training_dataset,
        verbose=args.verbose
    )
    
    results = identifier.run_identification(
        scenario1_count=args.scenario1_count,
        scenario2a_count=args.scenario2a_count,
        scenario2b_count=args.scenario2b_count
    )
    identifier.print_results(results)
    identifier.save_results(results, args.output)
    
    # Print usage examples
    print("\n" + "=" * 60)
    print("🧪 USAGE EXAMPLES")
    print("=" * 60)
    
    for scenario_key in ['scenario1', 'scenario2a', 'scenario2b']:
        scenario = results[scenario_key]
        if scenario['genes']:
            genes = [g['gene_id'] for g in scenario['genes'][:3]]
            genes_str = ','.join(genes)
            
            print(f"\n# Test {scenario_key.upper()}")
            print(f"python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \\")
            print(f"    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\")
            print(f"    --training-dataset {args.training_dataset} \\")
            print(f"    --genes {genes_str} \\")
            print(f"    --output-dir results/test_{scenario_key} \\")
            print(f"    --inference-mode hybrid \\")
            print(f"    --verbose")

if __name__ == "__main__":
    main()