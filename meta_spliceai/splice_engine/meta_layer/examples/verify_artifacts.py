#!/usr/bin/env python3
"""
verify_artifacts.py - Verify base layer artifacts are available for meta-layer training.

This script checks that base layer artifacts exist and summarizes their contents.

Usage:
    mamba activate metaspliceai
    python meta_spliceai/splice_engine/meta_layer/examples/verify_artifacts.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[5]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_layer.core import MetaLayerConfig, ArtifactLoader


def main():
    print("=" * 60)
    print("Meta-Layer Artifact Verification")
    print("=" * 60)
    print("\nThis script verifies that the meta_layer package correctly")
    print("integrates with the genomic_resources system for path resolution.")
    
    # Test both base models
    for base_model in ['openspliceai', 'spliceai']:
        print(f"\n{'='*60}")
        print(f"Base Model: {base_model.upper()}")
        print("=" * 60)
        
        try:
            config = MetaLayerConfig(base_model=base_model)
            print(f"\nConfiguration (from genomic_resources):")
            print(f"  Artifacts dir: {config.artifacts_dir}")
            print(f"  Annotation source: {config.annotation_source}")
            print(f"  Genome build: {config.genome_build}")
            print(f"  Coordinate column: {config.coordinate_column}")
            
            # Check if directory exists
            if not config.artifacts_dir.exists():
                print(f"\n❌ Artifacts directory not found!")
                print(f"   Run the base layer first:")
                print(f"   run_base_model --base-model {base_model} --mode production")
                continue
            
            # Load artifacts
            loader = ArtifactLoader(config)
            
            # Get statistics
            stats = loader.get_statistics()
            print(f"\nArtifact Statistics:")
            print(f"  Number of files: {stats['num_files']}")
            print(f"  Total positions: {stats['total_positions']:,}")
            print(f"  Total size: {stats['total_size_mb']:.1f} MB")
            print(f"  Chromosomes: {', '.join(stats['chromosomes'][:5])}...")
            
            # Get feature columns
            feature_cols = loader.get_feature_columns()
            print(f"\nFeature Columns:")
            print(f"  Sequence: {feature_cols['sequence']}")
            print(f"  Base scores: {feature_cols['base_scores']}")
            print(f"  Context scores: {feature_cols['context_scores']}")
            print(f"  Derived features: {len(feature_cols['derived_features'])} columns")
            print(f"  Labels: {feature_cols['labels']}")
            
            # Load sample data
            print(f"\nSample Data (first 5 rows from chr21):")
            try:
                df = loader.load_analysis_sequences(
                    chromosomes=['21'],
                    columns=['gene_id', 'splice_type', 'donor_score', 'acceptor_score'],
                    verbose=False
                )
                print(df.head(5))
            except Exception as e:
                print(f"  Could not load chr21: {e}")
            
            print(f"\n✅ {base_model} artifacts verified successfully!")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

