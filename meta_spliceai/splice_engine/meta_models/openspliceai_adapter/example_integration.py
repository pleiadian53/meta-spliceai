"""
Example script demonstrating OpenSpliceAI integration with MetaSpliceAI workflows.

This script shows how to use the OpenSpliceAI adapter to enhance your existing
meta-learning pipeline with OpenSpliceAI's robust data preprocessing capabilities.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.openspliceai_adapter import (
    OpenSpliceAIPreprocessor,
    OpenSpliceAIAdapterConfig
)
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)

def example_basic_usage():
    """Basic usage example with default MetaSpliceAI data paths."""
    print("=== Basic OpenSpliceAI Integration Example ===")
    
    # Initialize preprocessor with default MetaSpliceAI paths
    preprocessor = OpenSpliceAIPreprocessor(
        gtf_file="data/ensembl/Homo_sapiens.GRCh38.112.gtf",
        genome_fasta="data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
        output_dir="data/openspliceai_processed",
        verbose=2
    )
    
    # Create OpenSpliceAI-format datasets
    print("\n1. Creating OpenSpliceAI datasets...")
    openspliceai_datasets = preprocessor.create_openspliceai_datasets()
    
    # Create MetaSpliceAI-compatible data
    print("\n2. Creating MetaSpliceAI-compatible data...")
    splicesurveyor_data = preprocessor.create_splicesurveyor_compatible_data()
    
    # Get quality metrics
    print("\n3. Quality metrics:")
    metrics = preprocessor.get_quality_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    return openspliceai_datasets, splicesurveyor_data

def example_targeted_gene_analysis():
    """Example focusing on specific genes (e.g., ALS-related genes)."""
    print("\n=== Targeted Gene Analysis Example ===")
    
    # Focus on ALS-related genes
    als_genes = ["STMN2", "UNC13A", "TARDBP", "FUS", "SOD1", "C9orf72"]
    
    # Create configuration for targeted analysis
    config = OpenSpliceAIAdapterConfig(
        gtf_file="data/ensembl/Homo_sapiens.GRCh38.112.gtf",
        genome_fasta="data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
        output_dir="data/openspliceai_als_analysis",
        flanking_size=2000,  # Larger context for disease analysis
        biotype="protein-coding",
        target_genes=als_genes
    )
    
    preprocessor = OpenSpliceAIPreprocessor(config=config, verbose=2)
    
    # Create training datasets optimized for disease analysis
    print(f"\n1. Processing {len(als_genes)} ALS-related genes...")
    datasets = preprocessor.create_training_datasets(
        flanking_size=2000,
        output_format="parquet"  # Use Parquet for better integration
    )
    
    print(f"\n2. Created datasets:")
    for name, path in datasets.items():
        if os.path.exists(path):
            print(f"   {name}: {path} ({os.path.getsize(path)/1024:.1f} KB)")
    
    return datasets

def example_integration_with_existing_workflow():
    """Example showing integration with existing MetaSpliceAI workflow."""
    print("\n=== Integration with Existing Workflow Example ===")
    
    # Create enhanced configuration
    config = OpenSpliceAIAdapterConfig(
        flanking_size=400,
        biotype="protein-coding",
        parse_type="all_isoforms",  # Include all isoforms for comprehensive analysis
        canonical_only=False,
        output_dir="data/enhanced_workflow"
    )
    
    preprocessor = OpenSpliceAIPreprocessor(config=config, verbose=1)
    
    # Method 1: Use OpenSpliceAI preprocessing then integrate
    print("\n1. Using OpenSpliceAI preprocessing...")
    enhanced_data = preprocessor.create_splicesurveyor_compatible_data(
        use_openspliceai_preprocessing=True
    )
    
    # Method 2: Enhance existing MetaSpliceAI workflow
    print("\n2. Enhancing existing workflow...")
    try:
        # This would integrate with your existing workflow
        workflow_results = run_enhanced_splice_prediction_workflow(
            target_genes=["STMN2", "UNC13A"],  # Test with specific genes
            verbosity=1,
            test_mode=True  # Use test mode for faster execution
        )
        
        # Combine with OpenSpliceAI enhancements
        enhanced_results = preprocessor.integrate_with_splicesurveyor_workflow(
            workflow_config=config,
            enhance_with_openspliceai=True
        )
        
        print("   Successfully integrated workflows!")
        return enhanced_results
        
    except Exception as e:
        print(f"   Workflow integration failed: {e}")
        print("   This is expected if the full MetaSpliceAI data isn't available")
        return enhanced_data

def example_disease_case_study_preparation():
    """Example preparing data for disease case studies."""
    print("\n=== Disease Case Study Data Preparation ===")
    
    # Configuration for disease studies
    disease_config = OpenSpliceAIAdapterConfig(
        flanking_size=10000,  # Maximum context for cryptic splice sites
        biotype="all",  # Include non-coding genes for comprehensive analysis
        parse_type="all_isoforms",
        canonical_only=False,
        remove_paralogs=True,  # Important for disease studies
        output_dir="data/disease_case_studies"
    )
    
    preprocessor = OpenSpliceAIPreprocessor(config=disease_config, verbose=2)
    
    # Create datasets suitable for different disease contexts
    disease_contexts = {
        "cancer": {
            "genes": ["BRCA1", "BRCA2", "TP53", "MET"],
            "flanking_size": 2000
        },
        "neurodegeneration": {
            "genes": ["STMN2", "UNC13A", "MAPT", "PSEN1", "PSEN2"],
            "flanking_size": 10000  # Larger context for cryptic exons
        },
        "cystic_fibrosis": {
            "genes": ["CFTR"],
            "flanking_size": 10000  # For deep intronic mutations
        }
    }
    
    results = {}
    for context, params in disease_contexts.items():
        print(f"\n1. Preparing {context} case study data...")
        
        context_datasets = preprocessor.create_training_datasets(
            flanking_size=params["flanking_size"],
            target_genes=params["genes"],
            output_format="hdf5"
        )
        
        results[context] = context_datasets
        print(f"   Created {len(context_datasets)} datasets for {context}")
    
    return results

def main():
    """Run all examples."""
    print("OpenSpliceAI Integration Examples")
    print("=" * 50)
    
    try:
        # Basic usage
        basic_results = example_basic_usage()
        
        # Targeted analysis
        targeted_results = example_targeted_gene_analysis()
        
        # Workflow integration
        workflow_results = example_integration_with_existing_workflow()
        
        # Disease case study preparation
        disease_results = example_disease_case_study_preparation()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nNext steps:")
        print("1. Examine the created datasets in the output directories")
        print("2. Use the datasets with your meta-learning pipeline")
        print("3. Compare results with and without OpenSpliceAI preprocessing")
        
    except Exception as e:
        print(f"\nExample execution failed: {e}")
        print("This is expected if the required data files are not available.")
        print("Please ensure the following files exist:")
        print("- data/ensembl/Homo_sapiens.GRCh38.112.gtf")
        print("- data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa")

if __name__ == "__main__":
    main()
