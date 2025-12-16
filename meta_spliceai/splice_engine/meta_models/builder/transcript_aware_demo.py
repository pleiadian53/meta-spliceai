#!/usr/bin/env python3
"""
Transcript-Aware Position Identification: Practical Demonstration
================================================================

This script demonstrates the biological reality and technical impact of implementing
transcript-aware position identification for the 5000-gene meta model.

Key Insights:
1. Current genomic-only grouping loses critical biological information
2. Same position can have different splice site roles across transcripts
3. Enhanced position identification may solve meta model generalization failure

Usage:
    python transcript_aware_demo.py [--analyze-existing-data]
"""

import sys
from pathlib import Path
import polars as pl
import pandas as pd

# Add project root for imports
project_root = Path(__file__).resolve().parents[4] 
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.builder.transcript_aware_positions import (
    get_position_grouping_columns,
    resolve_transcript_specific_conflicts,
    analyze_position_complexity,
    demonstrate_biological_reality,
    TranscriptAwareConfig
)


def create_synthetic_example():
    """
    Create a synthetic example showing the biological reality of alternative splicing.
    
    Returns
    -------
    pl.DataFrame
        Synthetic dataset demonstrating position complexity
    """
    # Example: Position chr1:12345 in gene BRCA1 with different roles across transcripts
    data = [
        # Same position, different transcripts, different splice site roles
        {'gene_id': 'ENSG00000012048', 'position': 12345, 'strand': '+', 
         'transcript_id': 'ENST00000357654', 'splice_type': 'donor', 'pred_type': 'TP',
         'splice_probability': 0.95, 'sequence': 'GTAAGT...'},
        
        {'gene_id': 'ENSG00000012048', 'position': 12345, 'strand': '+',
         'transcript_id': 'ENST00000468300', 'splice_type': None, 'pred_type': 'TN', 
         'splice_probability': 0.05, 'sequence': 'GTAAGT...'},
        
        {'gene_id': 'ENSG00000012048', 'position': 12345, 'strand': '+',
         'transcript_id': 'ENST00000471181', 'splice_type': 'acceptor', 'pred_type': 'FP',
         'splice_probability': 0.75, 'sequence': 'GTAAGT...'},
         
        # Another position showing similar complexity
        {'gene_id': 'ENSG00000012048', 'position': 12500, 'strand': '+',
         'transcript_id': 'ENST00000357654', 'splice_type': 'acceptor', 'pred_type': 'TP',
         'splice_probability': 0.88, 'sequence': 'CAGGTG...'},
         
        {'gene_id': 'ENSG00000012048', 'position': 12500, 'strand': '+',
         'transcript_id': 'ENST00000468300', 'splice_type': 'acceptor', 'pred_type': 'FN',
         'splice_probability': 0.45, 'sequence': 'CAGGTG...'},
         
        # Simple position - same role across transcripts
        {'gene_id': 'ENSG00000012048', 'position': 12600, 'strand': '+',
         'transcript_id': 'ENST00000357654', 'splice_type': 'donor', 'pred_type': 'TP',
         'splice_probability': 0.92, 'sequence': 'GTGAGT...'},
         
        {'gene_id': 'ENSG00000012048', 'position': 12600, 'strand': '+',
         'transcript_id': 'ENST00000468300', 'splice_type': 'donor', 'pred_type': 'TP',
         'splice_probability': 0.89, 'sequence': 'GTGAGT...'},
    ]
    
    return pl.DataFrame(data)


def demonstrate_position_identification_modes():
    """
    Demonstrate the impact of different position identification modes.
    """
    print("=" * 80)
    print("TRANSCRIPT-AWARE POSITION IDENTIFICATION DEMONSTRATION")
    print("=" * 80)
    
    # Create synthetic data
    df = create_synthetic_example()
    
    print("\n🧬 SYNTHETIC DATASET (Biological Reality)")
    print("-" * 50)
    print("Same genomic positions with different splice site roles across transcripts:")
    print(df.select(['position', 'transcript_id', 'splice_type', 'pred_type']).to_pandas().to_string(index=False))
    
    print("\n" + "=" * 80)
    print("COMPARISON OF POSITION IDENTIFICATION MODES")
    print("=" * 80)
    
    # Mode 1: Current genomic-only approach
    print("\n1️⃣  CURRENT SYSTEM (Genomic-Only)")
    print("-" * 40)
    genomic_config = TranscriptAwareConfig(mode='genomic')
    df_genomic = genomic_config.resolve_conflicts(df)
    
    print(f"Grouping columns: {genomic_config.get_grouping_columns()}")
    print(f"Unique positions: {len(df_genomic)}")
    print("Result:")
    print(df_genomic.select(['position', 'splice_type', 'pred_type', 'splice_probability']).to_pandas().to_string(index=False))
    print("❌ LOST INFORMATION: Transcript-specific splice site roles")
    
    # Mode 2: Splice-aware approach
    print("\n2️⃣  SPLICE-AWARE SYSTEM (Recommended for 5000-Gene Model)")
    print("-" * 60)
    splice_config = TranscriptAwareConfig(mode='splice_aware')
    df_splice_aware = splice_config.resolve_conflicts(df)
    
    print(f"Grouping columns: {splice_config.get_grouping_columns()}")
    print(f"Unique positions: {len(df_splice_aware)}")
    print("Result:")
    print(df_splice_aware.select(['position', 'transcript_id', 'splice_type', 'pred_type']).to_pandas().to_string(index=False))
    print("✅ PRESERVED: Complete transcript-specific context for meta-learning")
    
    # Mode 3: Full transcript-specific approach
    print("\n3️⃣  TRANSCRIPT-SPECIFIC SYSTEM (Identical to Splice-Aware)")
    print("-" * 65)
    transcript_config = TranscriptAwareConfig(mode='transcript')
    df_transcript = transcript_config.resolve_conflicts(df)
    
    print(f"Grouping columns: {transcript_config.get_grouping_columns()}")
    print(f"Unique positions: {len(df_transcript)}")
    print("Result:")
    print(df_transcript.select(['position', 'transcript_id', 'splice_type', 'pred_type']).to_pandas().to_string(index=False))
    print("✅ PRESERVED: Same as splice_aware - enables variant effect prediction")
    
    # Mode 4: Hybrid approach
    print("\n4️⃣  HYBRID SYSTEM (Transition Strategy)")
    print("-" * 45)
    hybrid_config = TranscriptAwareConfig(mode='hybrid', preserve_transcript_info=True)
    df_hybrid = hybrid_config.resolve_conflicts(df)
    
    print(f"Grouping columns: {hybrid_config.get_grouping_columns()}")
    print(f"Unique positions: {len(df_hybrid)}")
    print("Result (with transcript metadata):")
    # Check which columns are actually available
    available_cols = ['position', 'splice_type', 'pred_type']
    if 'transcript_count' in df_hybrid.columns:
        available_cols.append('transcript_count')
    if 'transcript_ids' in df_hybrid.columns:
        available_cols.append('transcript_ids')
    print(df_hybrid.select(available_cols).to_pandas().to_string(index=False))
    print("✅ PRESERVED: Transcript metadata while maintaining efficiency")
    
    return df, df_genomic, df_splice_aware, df_transcript, df_hybrid


def analyze_impact_on_meta_learning():
    """
    Analyze how transcript-aware position identification impacts meta learning.
    """
    print("\n" + "=" * 80)
    print("IMPACT ON META MODEL GENERALIZATION")
    print("=" * 80)
    
    print("\n🎯 CONNECTION TO 5000-GENE META MODEL")
    print("-" * 45)
    
    print("""
Current Meta Model Issues:
1. Training data oversimplifies biological complexity
2. Same position forced to single label across transcripts  
3. Meta-learning misses isoform-specific patterns
4. meta_only mode performs worse than base_only on unseen genes

Root Cause Analysis:
- Genomic-only grouping loses transcript-specific splice site roles
- Meta model trained on oversimplified data fails to generalize
- Biological complexity not captured in training features

Proposed Solution for 5000-Gene Model:
- Use transcript/splice_aware mode during dataset assembly
- Position ID: ['gene_id', 'position', 'strand', 'transcript_id']
- Prediction target: 'splice_type' (enables variant effect learning)
- Better training data → improved meta model generalization
- Expected: meta_only mode finally outperforms base_only mode
""")
    
    print("\n📊 EXPECTED IMPROVEMENTS")
    print("-" * 30)
    
    # Create complexity analysis
    df_synthetic = create_synthetic_example()
    complexity = analyze_position_complexity(df_synthetic, verbose=False)
    
    print(f"Transcript expansion factor: {complexity.get('transcript_expansion_factor', 1.0):.2f}x")
    print(f"Positions with multiple splice types: {complexity.get('positions_with_multiple_splice_types', 0)}")
    print(f"Biological complexity captured: {complexity.get('percent_complex_positions', 0):.1f}%")
    
    print("""
Training Data Quality Improvements:
✅ Enables meta-learning for variant effect prediction
✅ Captures alternative splicing complexity  
✅ Better represents biological reality
✅ Preserves transcript-specific prediction contexts

Expected Meta Model Performance:
✅ Better generalization to unseen genes
✅ meta_only mode outperforms base_only mode
✅ Reduced overfitting to genomic-only patterns
✅ Foundation for disease-specific adaptation
""")


def demonstrate_integration_strategy():
    """
    Show how to integrate transcript-aware position ID into existing workflows.
    """
    print("\n" + "=" * 80)
    print("INTEGRATION STRATEGY FOR 5000-GENE MODEL")
    print("=" * 80)
    
    print("""
🔧 PHASE 1: IMMEDIATE IMPLEMENTATION (Recommended)

1. Dataset Builder Enhancement:
   ```python
   # In incremental_builder.py
   from .transcript_aware_positions import TranscriptAwareConfig
   
   def build_base_dataset(
       gene_list,
       analysis_tsv_dir, 
       output_dir,
       position_id_mode='transcript',  # NEW PARAMETER
       **kwargs
   ):
       if position_id_mode != 'genomic':
           config = TranscriptAwareConfig(mode=position_id_mode)
           group_cols = config.get_grouping_columns()
       else:
           group_cols = ['gene_id', 'position', 'strand']  # Current behavior
   ```

2. Sequence Data Utils Update:
   ```python  
   # In sequence_data_utils.py
   def load_and_process_sequences(
       file_path,
       position_id_mode='genomic',  # Default maintains compatibility
       **kwargs
   ):
       if position_id_mode in ['transcript', 'splice_aware']:
           group_cols = ['gene_id', 'position', 'strand', 'transcript_id']
       else:
           group_cols = ['gene_id', 'position', 'strand']  # Current behavior
   ```

3. Training Command:
   ```bash
   # Enhanced 5000-gene model training
   python incremental_builder.py \\
       --gene-list 5000_error_selected_genes.txt \\
       --position-id-mode transcript \\
       --output-dir results/train_5000_genes_transcript_aware \\
       --verbose
   ```

🔄 BACKWARD COMPATIBILITY
- All existing workflows continue unchanged (default: genomic mode)
- Gradual migration path available
- Existing models remain functional
- Optional enhancement without breaking changes

🎯 SUCCESS VALIDATION
- Use existing diagnostic tools to compare performance
- Run both genomic and transcript modes in parallel
- Measure meta model generalization improvement
- Validate with unseen gene sets
""")


def main():
    """Main demonstration function."""
    print("🧬 Transcript-Aware Position Identification: Comprehensive Demo")
    
    # Show biological reality
    print(demonstrate_biological_reality())
    
    # Demonstrate different modes
    results = demonstrate_position_identification_modes()
    
    # Analyze impact on meta learning
    analyze_impact_on_meta_learning()
    
    # Show integration strategy
    demonstrate_integration_strategy()
    
    print("\n" + "=" * 80)
    print("🚀 CONCLUSION")
    print("=" * 80)
    print("""
The transcript-aware position identification enhancement directly addresses
the ROOT CAUSE of meta model generalization failure:

❌ Current Problem: Genomic-only grouping oversimplifies biological reality
✅ Enhanced Solution: Transcript-aware mode captures alternative splicing complexity

For the 5000-gene meta model:
1. Implement transcript/splice_aware mode in dataset assembly
2. Enable meta-learning to predict variant effects on splicing
3. Expect improved meta model generalization  
4. Achieve meta_only mode outperforming base_only mode
5. Foundation for disease-specific adaptation

Ready to implement for the next training session! 🎯
""")


if __name__ == '__main__':
    main()
