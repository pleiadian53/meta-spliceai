#!/usr/bin/env python3
"""Demo: Meta-Model Inference Data Management
===========================================

This script demonstrates proper data organization, artifact storage, and gene
manifest management for meta-model inference workflows.

EXAMPLE USAGE:
    # RECOMMENDED: Run as module from project root (paths relative to project root):
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.demo_data_management \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes ENSG00000104435,ENSG00000006420 \
        --artifacts-dir ./inference_artifacts

    # ALTERNATIVE: Run script directly from project root:
    cd /home/bchiu/work/splice-surveyor  # Change to your project root
    python meta_spliceai/splice_engine/meta_models/workflows/inference/demo_data_management.py \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes ENSG00000104435,ENSG00000006420

    # COMPREHENSIVE WORKFLOW with caching verification:
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.demo_data_management \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes-file test_genes.txt \
        --artifacts-dir ./inference_cache \
        --verify-reusability \
        --verbose

FEATURES:
- Structured artifact directory creation and organization
- Gene manifest creation and tracking
- Inference artifact preservation and verification  
- Cache reusability testing
- Storage efficiency analysis
- Data integrity validation

REQUIREMENTS:
- Pre-trained meta-model (.pkl file)
- Training dataset directory
- Target genes for processing
- Write access to artifacts directory
"""

import argparse
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import tempfile

import polars as pl
import pandas as pd
import numpy as np

# Add project root to Python path
import os
project_root = os.path.join(os.path.expanduser("~"), "work/splice-surveyor")
sys.path.insert(0, project_root)

from meta_spliceai.splice_engine.meta_models.workflows.selective_meta_inference import (
    run_selective_meta_inference,
    SelectiveInferenceConfig,
    setup_inference_directories
)


class DataManagementDemo:
    """Demonstrate data management capabilities for inference workflows."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.artifacts_created = []
        self.manifest_entries = []
    
    def verify_directory_structure(self, base_dir: Path) -> Dict[str, Any]:
        """Verify that inference directory structure is properly created."""
        
        if self.verbose:
            print(f"   🗂️  Verifying directory structure in: {base_dir}")
        
        # Expected subdirectories
        expected_dirs = {
            'artifacts': base_dir / 'artifacts',
            'features': base_dir / 'features', 
            'predictions': base_dir / 'predictions',
            'manifests': base_dir / 'manifests'
        }
        
        structure_info = {
            'base_directory': str(base_dir),
            'exists': base_dir.exists(),
            'subdirectories': {},
            'readme_exists': (base_dir / 'README.md').exists(),
            'created_timestamp': None
        }
        
        for name, path in expected_dirs.items():
            structure_info['subdirectories'][name] = {
                'path': str(path),
                'exists': path.exists(),
                'writeable': path.exists() and os.access(path, os.W_OK),
                'file_count': len(list(path.glob('*'))) if path.exists() else 0
            }
        
        # Check for timestamp file
        timestamp_file = base_dir / '.created_timestamp'
        if timestamp_file.exists():
            with open(timestamp_file, 'r') as f:
                structure_info['created_timestamp'] = f.read().strip()
        
        if self.verbose:
            print(f"      ✅ Base directory exists: {structure_info['exists']}")
            for name, info in structure_info['subdirectories'].items():
                status = "✅" if info['exists'] else "❌"
                print(f"      {status} {name}: {info['path']} ({info['file_count']} files)")
        
        return structure_info
    
    def create_inference_directories(self, base_dir: Path) -> Dict[str, Path]:
        """Create inference directory structure and verify."""
        
        if self.verbose:
            print(f"\n📁 STEP 1: Creating inference directory structure")
        
        # Use the utility function to create directories
        directories = setup_inference_directories(base_dir)
        
        # Add timestamp
        timestamp_file = base_dir / '.created_timestamp'
        with open(timestamp_file, 'w') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Verify structure
        structure_info = self.verify_directory_structure(base_dir)
        
        if self.verbose:
            print(f"   ✅ Directory structure created and verified")
        
        return directories
    
    def verify_gene_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        """Verify gene manifest structure and content."""
        
        if not manifest_path.exists():
            return {
                'exists': False,
                'error': f"Manifest file not found: {manifest_path}"
            }
        
        try:
            # Read manifest
            manifest_df = pl.read_csv(manifest_path)
            
            # Expected columns
            expected_cols = {'gene_id', 'status', 'timestamp', 'positions_processed', 'output_file'}
            actual_cols = set(manifest_df.columns)
            missing_cols = expected_cols - actual_cols
            extra_cols = actual_cols - expected_cols
            
            # Get basic stats
            total_genes = len(manifest_df)
            completed_genes = len(manifest_df.filter(pl.col('status') == 'completed'))
            failed_genes = len(manifest_df.filter(pl.col('status') == 'failed'))
            
            manifest_info = {
                'exists': True,
                'path': str(manifest_path),
                'total_entries': total_genes,
                'completed_genes': completed_genes,
                'failed_genes': failed_genes,
                'completion_rate': completed_genes / total_genes if total_genes > 0 else 0,
                'expected_columns': list(expected_cols),
                'actual_columns': list(actual_cols),
                'missing_columns': list(missing_cols),
                'extra_columns': list(extra_cols),
                'schema_valid': len(missing_cols) == 0
            }
            
            # Get recent entries
            if total_genes > 0:
                recent_entries = manifest_df.sort('timestamp', descending=True).head(5)
                manifest_info['recent_entries'] = recent_entries.to_dicts()
            
            return manifest_info
            
        except Exception as e:
            return {
                'exists': True,
                'error': f"Failed to read manifest: {e}"
            }
    
    def run_inference_with_tracking(
        self,
        model_path: Path,
        training_dataset_path: Path,
        target_genes: List[str],
        artifacts_dir: Path,
        max_positions_per_gene: int = 5000
    ) -> Dict[str, Any]:
        """Run inference while tracking artifacts and manifest updates."""
        
        if self.verbose:
            print(f"\n🔬 STEP 2: Running inference with artifact tracking")
            print(f"   🧬 Processing {len(target_genes)} genes")
        
        # Record initial state
        initial_files = set()
        if artifacts_dir.exists():
            for path in artifacts_dir.rglob('*'):
                if path.is_file():
                    initial_files.add(str(path.relative_to(artifacts_dir)))
        
        try:
            # Configure inference
            selective_config = SelectiveInferenceConfig(
                model_path=str(model_path),
                target_genes=target_genes,
                training_dataset_path=str(training_dataset_path) if training_dataset_path else None,
                uncertainty_threshold_low=0.02,
                uncertainty_threshold_high=0.80,
                max_positions_per_gene=max_positions_per_gene,
                inference_base_dir=artifacts_dir,
                verbose=max(0, 1 if self.verbose else 0),
                cleanup_intermediates=False  # Keep artifacts for analysis
            )
            
            # Run inference
            start_time = time.time()
            workflow_results = run_selective_meta_inference(selective_config)
            end_time = time.time()
            
            if not workflow_results.get('success', False):
                raise RuntimeError("Inference workflow failed")
            
            # Record final state
            final_files = set()
            new_files = []
            total_size = 0
            
            if artifacts_dir.exists():
                for path in artifacts_dir.rglob('*'):
                    if path.is_file():
                        rel_path = str(path.relative_to(artifacts_dir))
                        final_files.add(rel_path)
                        
                        if rel_path not in initial_files:
                            new_files.append({
                                'path': rel_path,
                                'size_mb': path.stat().st_size / 1024 / 1024,
                                'extension': path.suffix
                            })
                        
                        total_size += path.stat().st_size
            
            # Analyze artifacts
            artifacts_analysis = {
                'success': True,
                'runtime_seconds': end_time - start_time,
                'initial_files_count': len(initial_files),
                'final_files_count': len(final_files),
                'new_files_count': len(new_files),
                'new_files': new_files,
                'total_size_mb': total_size / 1024 / 1024,
                'genes_processed': len(target_genes),
                'predictions_generated': len(workflow_results.get('predictions', [])),
                'workflow_results': workflow_results
            }
            
            # Categorize new files by type
            file_types = {}
            for file_info in new_files:
                ext = file_info['extension'] or 'no_extension'
                if ext not in file_types:
                    file_types[ext] = {'count': 0, 'total_size_mb': 0}
                file_types[ext]['count'] += 1
                file_types[ext]['total_size_mb'] += file_info['size_mb']
            
            artifacts_analysis['file_types'] = file_types
            
            if self.verbose:
                print(f"   ✅ Inference completed in {artifacts_analysis['runtime_seconds']:.1f}s")
                print(f"   📁 New artifacts created: {artifacts_analysis['new_files_count']}")
                print(f"   💾 Total storage used: {artifacts_analysis['total_size_mb']:.1f} MB")
                
                if file_types:
                    print(f"   📊 File types created:")
                    for ext, info in file_types.items():
                        print(f"      {ext}: {info['count']} files ({info['total_size_mb']:.1f} MB)")
            
            return artifacts_analysis
            
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Inference failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'runtime_seconds': 0
            }
    
    def verify_artifact_organization(self, artifacts_dir: Path) -> Dict[str, Any]:
        """Verify proper organization of inference artifacts."""
        
        if self.verbose:
            print(f"\n🗂️  STEP 3: Verifying artifact organization")
        
        organization_info = {
            'directory_structure': self.verify_directory_structure(artifacts_dir),
            'artifact_counts': {},
            'storage_analysis': {},
            'manifest_analysis': {}
        }
        
        # Count artifacts by type and location
        artifact_patterns = {
            'analysis_sequences': 'analysis_sequences_*.tsv',
            'splice_positions': 'splice_positions_enhanced_*.tsv',
            'feature_matrices': 'features/**/*.parquet',
            'predictions': 'predictions/**/*.tsv',
            'manifests': 'manifests/**/*.csv'
        }
        
        for artifact_type, pattern in artifact_patterns.items():
            files = list(artifacts_dir.glob(pattern))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            organization_info['artifact_counts'][artifact_type] = {
                'count': len(files),
                'total_size_mb': total_size / 1024 / 1024,
                'files': [str(f.relative_to(artifacts_dir)) for f in files[:5]]  # First 5 files
            }
        
        # Analyze gene manifest if it exists
        manifest_file = artifacts_dir / 'manifests' / 'gene_manifest.csv'
        organization_info['manifest_analysis'] = self.verify_gene_manifest(manifest_file)
        
        if self.verbose:
            print(f"   📊 Artifact organization summary:")
            for artifact_type, info in organization_info['artifact_counts'].items():
                print(f"      {artifact_type}: {info['count']} files ({info['total_size_mb']:.1f} MB)")
        
        return organization_info
    
    def test_cache_reusability(
        self,
        artifacts_dir: Path,
        target_genes: List[str]
    ) -> Dict[str, Any]:
        """Test if artifacts can be reused efficiently."""
        
        if self.verbose:
            print(f"\n🔄 STEP 4: Testing artifact reusability")
        
        # Check for gene manifest
        manifest_file = artifacts_dir / 'manifests' / 'gene_manifest.csv'
        
        if not manifest_file.exists():
            return {
                'reusable': False,
                'reason': 'No gene manifest found'
            }
        
        try:
            manifest_df = pl.read_csv(manifest_file)
            
            # Check which genes are already processed
            processed_genes = set(manifest_df.filter(
                pl.col('status') == 'completed'
            )['gene_id'].to_list())
            
            target_set = set(target_genes)
            already_processed = target_set.intersection(processed_genes)
            need_processing = target_set - processed_genes
            
            reusability_info = {
                'reusable': True,
                'manifest_genes': len(processed_genes),
                'target_genes': len(target_genes),
                'already_processed': len(already_processed),
                'need_processing': len(need_processing),
                'cache_hit_rate': len(already_processed) / len(target_genes),
                'processed_gene_list': list(already_processed),
                'remaining_gene_list': list(need_processing)
            }
            
            if self.verbose:
                print(f"   📈 Cache analysis:")
                print(f"      Total genes in manifest: {reusability_info['manifest_genes']}")
                print(f"      Target genes: {reusability_info['target_genes']}")
                print(f"      Already processed: {reusability_info['already_processed']}")
                print(f"      Cache hit rate: {reusability_info['cache_hit_rate']:.1%}")
            
            return reusability_info
            
        except Exception as e:
            return {
                'reusable': False,
                'reason': f'Error reading manifest: {e}'
            }
    
    def run_demo(
        self,
        model_path: Path,
        training_dataset_path: Path,
        target_genes: List[str],
        artifacts_dir: Path,
        verify_reusability: bool = False,
        max_positions_per_gene: int = 5000
    ) -> Dict[str, Any]:
        """Run comprehensive data management demonstration."""
        
        print("🗂️  Meta-Model Inference Data Management Demo")
        print("=" * 55)
        print(f"   📁 Model: {model_path}")
        print(f"   📁 Training data: {training_dataset_path}")
        print(f"   🧬 Target genes: {len(target_genes)}")
        print(f"   📁 Artifacts directory: {artifacts_dir}")
        
        demo_results = {
            'config': {
                'model_path': str(model_path),
                'training_dataset_path': str(training_dataset_path),
                'target_genes': target_genes,
                'artifacts_dir': str(artifacts_dir),
                'max_positions_per_gene': max_positions_per_gene
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # Step 1: Create directory structure
            directories = self.create_inference_directories(artifacts_dir)
            demo_results['directory_setup'] = {
                'success': True,
                'directories': {k: str(v) for k, v in directories.items()}
            }
            
            # Step 2: Run inference with tracking
            inference_results = self.run_inference_with_tracking(
                model_path=model_path,
                training_dataset_path=training_dataset_path,
                target_genes=target_genes,
                artifacts_dir=artifacts_dir,
                max_positions_per_gene=max_positions_per_gene
            )
            demo_results['inference_tracking'] = inference_results
            
            # Step 3: Verify organization
            organization_results = self.verify_artifact_organization(artifacts_dir)
            demo_results['organization_verification'] = organization_results
            
            # Step 4: Test reusability (if requested)
            if verify_reusability:
                reusability_results = self.test_cache_reusability(artifacts_dir, target_genes)
                demo_results['reusability_analysis'] = reusability_results
            
            # Summary
            demo_results['success'] = True
            demo_results['summary'] = {
                'directories_created': len(directories),
                'artifacts_generated': inference_results.get('new_files_count', 0),
                'storage_used_mb': inference_results.get('total_size_mb', 0),
                'runtime_seconds': inference_results.get('runtime_seconds', 0),
                'manifest_valid': organization_results['manifest_analysis'].get('schema_valid', False)
            }
            
            # Save demo results
            results_file = artifacts_dir / 'data_management_demo_results.json'
            with open(results_file, 'w') as f:
                json.dump(demo_results, f, indent=2, default=str)
            
            if self.verbose:
                print(f"\n✅ Data Management Demo Completed Successfully!")
                print(f"   📁 Directories created: {demo_results['summary']['directories_created']}")
                print(f"   📄 Artifacts generated: {demo_results['summary']['artifacts_generated']}")
                print(f"   💾 Storage used: {demo_results['summary']['storage_used_mb']:.1f} MB")
                print(f"   📊 Results saved: {results_file}")
            
            return demo_results
            
        except Exception as e:
            demo_results['success'] = False
            demo_results['error'] = str(e)
            
            if self.verbose:
                print(f"\n❌ Demo failed: {e}")
            
            return demo_results


def parse_genes_input(genes_arg: str, genes_file: Optional[str]) -> List[str]:
    """Parse gene input from command line or file."""
    if genes_file:
        genes_path = Path(genes_file)
        if not genes_path.exists():
            raise FileNotFoundError(f"Gene file not found: {genes_file}")
        
        with open(genes_path, 'r') as f:
            genes = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not genes:
            raise ValueError(f"No genes found in file: {genes_file}")
        
        return genes
    
    elif genes_arg:
        genes = [g.strip() for g in genes_arg.split(',') if g.strip()]
        if not genes:
            raise ValueError("No valid genes provided")
        return genes
    
    else:
        raise ValueError("Must provide either --genes or --genes-file")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Meta-Model Inference Data Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE USAGE:
    # RECOMMENDED: Run as module from project root:
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.demo_data_management \\
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
        --training-dataset train_pc_1000_3mers \\
        --genes ENSG00000104435,ENSG00000006420

    # ALTERNATIVE: Run script from project root:
    cd /home/bchiu/work/splice-surveyor  # Change to your project root
    python meta_spliceai/splice_engine/meta_models/workflows/inference/demo_data_management.py \\
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
        --training-dataset train_pc_1000_3mers \\
        --genes-file test_genes.txt \\
        --verify-reusability \\
        --verbose

NOTE: Paths like "results/..." are relative to project root.
      Use absolute paths if running from different directories.
        """
    )
    
    # Required arguments
    parser.add_argument('--model', '-m', required=True, type=Path,
                       help='Path to pre-trained meta-model (.pkl file)')
    parser.add_argument('--training-dataset', '-t', required=True, type=Path,
                       help='Path to training dataset directory')
    
    # Gene specification (mutually exclusive)
    gene_group = parser.add_mutually_exclusive_group(required=True)
    gene_group.add_argument('--genes', '-g', type=str,
                           help='Comma-separated list of gene IDs')
    gene_group.add_argument('--genes-file', '-gf', type=str,
                           help='Path to file containing gene IDs (one per line)')
    
    # Optional arguments
    parser.add_argument('--artifacts-dir', '-a', type=Path,
                       default=Path.cwd() / "inference_artifacts",
                       help='Directory for storing inference artifacts')
    parser.add_argument('--max-positions', type=int, default=5000,
                       help='Maximum positions per gene (default: 5000)')
    parser.add_argument('--verify-reusability', action='store_true',
                       help='Test artifact cache reusability')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress all output except errors')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.model.exists():
        print(f"❌ Model file not found: {args.model}")
        return 1
    
    if not args.training_dataset.exists():
        print(f"❌ Training dataset not found: {args.training_dataset}")
        return 1
    
    # Parse genes
    try:
        target_genes = parse_genes_input(args.genes, args.genes_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Gene input error: {e}")
        return 1
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    try:
        # Create demo instance and run
        demo = DataManagementDemo(verbose=verbose)
        
        results = demo.run_demo(
            model_path=args.model,
            training_dataset_path=args.training_dataset,
            target_genes=target_genes,
            artifacts_dir=args.artifacts_dir,
            verify_reusability=args.verify_reusability,
            max_positions_per_gene=args.max_positions
        )
        
        if results['success']:
            if not args.quiet:
                print(f"\n✅ Demo completed successfully!")
                summary = results['summary']
                print(f"📁 Artifacts: {summary['artifacts_generated']}")
                print(f"💾 Storage: {summary['storage_used_mb']:.1f} MB") 
                print(f"⏱️  Runtime: {summary['runtime_seconds']:.1f}s")
            return 0
        else:
            print(f"❌ Demo failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())