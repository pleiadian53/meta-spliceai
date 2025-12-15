"""
OpenSpliceAI Integration Bridge for Case Studies

This module provides seamless integration between the OpenSpliceAI adapter
and the case studies validation workflows, enabling comprehensive variant
analysis with alternative splicing pattern detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ..data_sources.base import SpliceMutation, IngestionResult
from ..formats.variant_standardizer import VariantStandardizer
from .alternative_splicing_pipeline import AlternativeSplicingPipeline, AlternativeSpliceSite


class OpenSpliceAIIntegrationBridge:
    """
    Bridge between OpenSpliceAI adapter and case studies workflows.
    
    This class provides:
    1. Direct integration with AlignedSpliceExtractor for splice site extraction
    2. Conversion between OpenSpliceAI and MetaSpliceAI coordinate systems
    3. Fallback mechanisms for missing splice site annotations
    4. Schema adaptation for multi-model compatibility
    """
    
    def __init__(self, work_dir: Path, verbosity: int = 1):
        """
        Initialize the integration bridge.
        
        Parameters
        ----------
        work_dir : Path
            Working directory for outputs
        verbosity : int
            Verbosity level (0=silent, 1=normal, 2=verbose)
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.verbosity = verbosity
        
        # Initialize components
        self.variant_standardizer = VariantStandardizer()
        self.alternative_pipeline = AlternativeSplicingPipeline(work_dir)
        
        # Try to import OpenSpliceAI adapter
        self.openspliceai_available = False
        try:
            from meta_spliceai.splice_engine.meta_models.openspliceai_adapter import (
                AlignedSpliceExtractor,
                OpenSpliceAIAdapterConfig
            )
            self.AlignedSpliceExtractor = AlignedSpliceExtractor
            self.OpenSpliceAIAdapterConfig = OpenSpliceAIAdapterConfig
            self.openspliceai_available = True
            if self.verbosity >= 1:
                print("[Integration] ✅ OpenSpliceAI adapter available")
        except ImportError as e:
            if self.verbosity >= 1:
                print(f"[Integration] ⚠️  OpenSpliceAI adapter not available: {e}")
    
    def extract_splice_sites_with_fallback(
        self,
        gene_data_dir: Path,
        output_dir: Path,
        use_openspliceai: bool = True
    ) -> pd.DataFrame:
        """
        Extract splice sites with OpenSpliceAI fallback mechanism.
        
        Implements the fallback strategy from memory:
        1. Try to use existing splice_sites.tsv if available
        2. Fall back to OpenSpliceAI extraction if missing
        3. Convert to MetaSpliceAI format
        
        Parameters
        ----------
        gene_data_dir : Path
            Directory containing gene data
        output_dir : Path
            Output directory for splice sites
        use_openspliceai : bool
            Whether to use OpenSpliceAI as fallback
            
        Returns
        -------
        pd.DataFrame
            Splice sites in MetaSpliceAI format
        """
        splice_sites_file = output_dir / "splice_sites.tsv"
        
        # Check if native splice_sites.tsv exists
        if splice_sites_file.exists():
            if self.verbosity >= 1:
                print(f"[Integration] Using existing splice sites: {splice_sites_file}")
            return pd.read_csv(splice_sites_file, sep='\t')
        
        # Use OpenSpliceAI fallback if available
        if use_openspliceai and self.openspliceai_available:
            if self.verbosity >= 1:
                print("[Integration] Splice sites not found, using OpenSpliceAI fallback")
            
            return self._extract_with_openspliceai(gene_data_dir, output_dir)
        
        # No fallback available
        if self.verbosity >= 1:
            print("[Integration] ⚠️  No splice sites available and OpenSpliceAI not configured")
        
        return pd.DataFrame()
    
    def _extract_with_openspliceai(
        self,
        gene_data_dir: Path,
        output_dir: Path
    ) -> pd.DataFrame:
        """
        Extract splice sites using OpenSpliceAI AlignedSpliceExtractor.
        
        This implements the 100% equivalence extraction validated in previous work.
        """
        try:
            # Configure extractor for MetaSpliceAI compatibility
            config = {
                'coordinate_system': 'metaspliceai',  # Critical for alignment
                'include_canonical': True,
                'include_cryptic': True,
                'flanking_size': 2000,
                'verbosity': self.verbosity
            }
            
            # Initialize extractor
            extractor = self.AlignedSpliceExtractor(**config)
            
            # Find parquet files in gene directory
            parquet_files = list(gene_data_dir.glob("*.parquet"))
            
            if not parquet_files:
                if self.verbosity >= 1:
                    print(f"[Integration] No parquet files found in {gene_data_dir}")
                return pd.DataFrame()
            
            # Extract splice sites from all files
            all_sites = []
            for pq_file in parquet_files:
                sites_df = extractor.extract_splice_sites(pq_file)
                all_sites.append(sites_df)
            
            # Combine all sites
            combined_sites = pd.concat(all_sites, ignore_index=True)
            
            # Save in TSV format for MetaSpliceAI compatibility
            output_file = output_dir / "splice_sites.tsv"
            combined_sites.to_csv(output_file, sep='\t', index=False)
            
            if self.verbosity >= 1:
                print(f"[Integration] Extracted {len(combined_sites)} splice sites via OpenSpliceAI")
                print(f"[Integration] Saved to: {output_file}")
            
            return combined_sites
            
        except Exception as e:
            if self.verbosity >= 1:
                print(f"[Integration] Failed to extract with OpenSpliceAI: {e}")
            return pd.DataFrame()
    
    def process_clinvar_with_openspliceai(
        self,
        clinvar_mutations: List[SpliceMutation],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Process ClinVar mutations through OpenSpliceAI for alternative splicing analysis.
        
        Parameters
        ----------
        clinvar_mutations : List[SpliceMutation]
            List of ClinVar splice mutations
        output_dir : Path
            Output directory
            
        Returns
        -------
        Dict[str, Any]
            Analysis results including alternative splice sites
        """
        if self.verbosity >= 1:
            print(f"[Integration] Processing {len(clinvar_mutations)} ClinVar mutations")
        
        # Convert mutations to VCF format
        vcf_path = output_dir / "clinvar_variants.vcf"
        self._mutations_to_vcf(clinvar_mutations, vcf_path)
        
        # Process through alternative splicing pipeline
        if vcf_path.exists():
            # Mock gene annotations for now
            gene_annotations = pd.DataFrame()
            
            # Run pipeline
            alternative_sites_df = self.alternative_pipeline.process_vcf_to_alternative_sites(
                vcf_path, gene_annotations
            )
            
            # Analyze patterns
            patterns = self._analyze_splicing_patterns(alternative_sites_df)
            
            results = {
                'total_mutations': len(clinvar_mutations),
                'alternative_sites': len(alternative_sites_df),
                'patterns': patterns,
                'output_files': {
                    'vcf': str(vcf_path),
                    'alternative_sites': str(output_dir / 'alternative_splice_sites.tsv')
                }
            }
            
            if self.verbosity >= 1:
                print(f"[Integration] Identified {len(alternative_sites_df)} alternative sites")
                print(f"[Integration] Pattern analysis: {patterns}")
            
            return results
        
        return {'error': 'Failed to create VCF file'}
    
    def _mutations_to_vcf(self, mutations: List[SpliceMutation], output_path: Path):
        """Convert SpliceMutation objects to VCF format."""
        with open(output_path, 'w') as f:
            # Write VCF header
            f.write("##fileformat=VCFv4.2\n")
            f.write("##source=MetaSpliceAI_CaseStudies\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            
            # Write variants
            for mut in mutations:
                chrom = f"chr{mut.chrom}" if not mut.chrom.startswith('chr') else mut.chrom
                pos = mut.position
                ref = mut.ref_allele if mut.ref_allele else "N"
                alt = mut.alt_allele if mut.alt_allele else "N"
                info = f"GENE={mut.gene_symbol};EVENT={mut.splice_event_type.value}"
                
                f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\n")
    
    def _analyze_splicing_patterns(self, sites_df: pd.DataFrame) -> Dict[str, int]:
        """Analyze alternative splicing patterns from sites."""
        patterns = {
            'canonical_disruption': 0,
            'cryptic_activation': 0,
            'exon_skipping': 0,
            'intron_retention': 0
        }
        
        if sites_df.empty:
            return patterns
        
        # Count pattern types
        if 'splice_category' in sites_df.columns:
            for category in sites_df['splice_category'].unique():
                if 'canonical' in category.lower() and 'disrupt' in category.lower():
                    patterns['canonical_disruption'] += 1
                elif 'cryptic' in category.lower():
                    patterns['cryptic_activation'] += 1
        
        # Detect exon skipping patterns (simplified)
        if 'delta_score' in sites_df.columns:
            # Large negative delta scores at canonical sites suggest exon skipping
            canonical_loss = sites_df[
                (sites_df.get('splice_category', '').str.contains('canonical', na=False)) &
                (sites_df['delta_score'] < -0.5)
            ]
            patterns['exon_skipping'] = len(canonical_loss)
        
        return patterns
    
    def create_schema_adapter(self, source_format: str, target_format: str = 'metaspliceai'):
        """
        Create schema adapter for format conversion.
        
        Implements the schema adapter pattern from memory for multi-model support.
        
        Parameters
        ----------
        source_format : str
            Source format ('openspliceai', 'spliceai', 'pangolin', etc.)
        target_format : str
            Target format (default: 'metaspliceai')
            
        Returns
        -------
        SchemaAdapter
            Configured schema adapter
        """
        # Define schema mappings
        schema_mappings = {
            'openspliceai': {
                'chromosome': 'chrom',
                'position': 'start',
                'splice_type': 'site_type',
                'score': 'splice_score'
            },
            'spliceai': {
                'chrom': 'chromosome',
                'pos': 'position',
                'strand': 'strand',
                'type': 'site_type'
            }
        }
        
        class SchemaAdapter:
            """Simple schema adapter for format conversion."""
            
            def __init__(self, mapping: Dict[str, str]):
                self.mapping = mapping
            
            def convert(self, df: pd.DataFrame) -> pd.DataFrame:
                """Convert DataFrame schema."""
                result = df.copy()
                
                for source_col, target_col in self.mapping.items():
                    if source_col in result.columns:
                        result = result.rename(columns={source_col: target_col})
                
                return result
        
        mapping = schema_mappings.get(source_format, {})
        return SchemaAdapter(mapping)
    
    def run_integrated_validation(
        self,
        ingestion_result: IngestionResult,
        meta_model_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run integrated validation workflow with OpenSpliceAI enhancement.
        
        Parameters
        ----------
        ingestion_result : IngestionResult
            Result from database ingestion
        meta_model_path : Optional[Path]
            Path to meta-model
            
        Returns
        -------
        Dict[str, Any]
            Validation results with OpenSpliceAI enhancements
        """
        results = {
            'database': ingestion_result.database,
            'mutations_analyzed': len(ingestion_result.mutations),
            'openspliceai_enhanced': False
        }
        
        # Process mutations through OpenSpliceAI if available
        if self.openspliceai_available and ingestion_result.mutations:
            clinvar_mutations = [
                m for m in ingestion_result.mutations 
                if m.clinical_significance and 'pathogenic' in m.clinical_significance.lower()
            ]
            
            if clinvar_mutations:
                openspliceai_results = self.process_clinvar_with_openspliceai(
                    clinvar_mutations[:100],  # Limit for performance
                    self.work_dir / 'openspliceai_analysis'
                )
                
                results['openspliceai_enhanced'] = True
                results['alternative_sites_found'] = openspliceai_results.get('alternative_sites', 0)
                results['splicing_patterns'] = openspliceai_results.get('patterns', {})
        
        return results


def demonstrate_integration():
    """Demonstrate the OpenSpliceAI integration capabilities."""
    from pathlib import Path
    
    work_dir = Path("./case_studies/openspliceai_integration_demo")
    bridge = OpenSpliceAIIntegrationBridge(work_dir, verbosity=2)
    
    print("\n" + "="*60)
    print("OpenSpliceAI Integration Bridge Demonstration")
    print("="*60 + "\n")
    
    # Test schema adapter
    print("📋 Testing Schema Adapter Pattern...")
    adapter = bridge.create_schema_adapter('openspliceai', 'metaspliceai')
    
    test_df = pd.DataFrame({
        'chromosome': ['1', '2'],
        'position': [1000, 2000],
        'splice_type': ['donor', 'acceptor'],
        'score': [0.9, 0.8]
    })
    
    converted_df = adapter.convert(test_df)
    print(f"   Original columns: {list(test_df.columns)}")
    print(f"   Converted columns: {list(converted_df.columns)}")
    
    # Test fallback mechanism
    print("\n🔄 Testing Fallback Mechanism...")
    gene_dir = Path("./test_data/genes")
    output_dir = work_dir / "fallback_test"
    
    sites_df = bridge.extract_splice_sites_with_fallback(
        gene_dir, output_dir, use_openspliceai=True
    )
    
    if not sites_df.empty:
        print(f"   ✅ Extracted {len(sites_df)} splice sites")
    else:
        print("   ℹ️  No splice sites extracted (expected if no test data)")
    
    print("\n✅ Integration bridge demonstration complete!")


if __name__ == "__main__":
    demonstrate_integration()
