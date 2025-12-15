"""
Disease-Specific Alternative Splicing Analysis

This module provides specialized workflows for analyzing alternative splice sites
in the context of specific diseases, focusing on the adaptive capacity of the
meta-learning layer to predict disease-induced alternative splicing patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import json
import logging

from ..data_types import AlternativeSpliceSite
from ..data_sources.base import SpliceMutation
from .regulatory_features import RegulatoryFeatureExtractor, RegulatoryContext
from .alternative_splicing_pipeline import AlternativeSplicingPipeline


@dataclass
class DiseaseAlternativeSplicingResult:
    """Results from disease-specific alternative splicing analysis."""
    disease_name: str
    total_mutations: int
    alternative_sites_detected: int
    cryptic_sites_activated: int
    canonical_sites_disrupted: int
    novel_splice_patterns: int
    meta_model_improvement: float
    tissue_specific_patterns: Dict[str, int]
    validation_evidence: Dict[str, int]
    clinical_significance_breakdown: Dict[str, int]


@dataclass
class AlternativeSplicingPattern:
    """Represents a disease-specific alternative splicing pattern."""
    pattern_id: str
    disease: str
    gene_symbol: str
    pattern_type: str  # 'exon_skipping', 'cryptic_activation', 'intron_retention', etc.
    affected_sites: List[AlternativeSpliceSite]
    frequency: float
    clinical_impact: str
    therapeutic_target: bool = False


class DiseaseSpecificAlternativeSplicingAnalyzer:
    """
    Analyzer for disease-specific alternative splicing patterns.
    
    This class focuses on identifying and characterizing alternative splice sites
    that are specifically induced by disease mutations, demonstrating the
    adaptive capacity of the meta-learning layer.
    """
    
    def __init__(self, 
                 work_dir: Path,
                 regulatory_data_dir: Optional[Path] = None,
                 verbosity: int = 1):
        """
        Initialize disease-specific alternative splicing analyzer.
        
        Parameters
        ----------
        work_dir : Path
            Working directory for analysis outputs
        regulatory_data_dir : Optional[Path]
            Directory containing regulatory data
        verbosity : int
            Verbosity level
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.verbosity = verbosity
        self.logger = self._setup_logging()
        
        # Initialize components
        self.alternative_pipeline = AlternativeSplicingPipeline(work_dir)
        
        if regulatory_data_dir:
            self.regulatory_extractor = RegulatoryFeatureExtractor(
                regulatory_data_dir, verbosity
            )
        else:
            self.regulatory_extractor = None
        
        # Disease-specific configurations
        self.disease_configs = self._load_disease_configurations()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the analyzer."""
        logger = logging.getLogger("DiseaseSpecificAlternativeSplicingAnalyzer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbosity >= 1 else logging.WARNING)
        return logger
    
    def _load_disease_configurations(self) -> Dict[str, Dict]:
        """Load disease-specific analysis configurations."""
        return {
            'cystic_fibrosis': {
                'primary_genes': ['CFTR'],
                'splice_patterns': ['cryptic_pseudoexon', 'exon_skipping'],
                'tissues': ['lung', 'pancreas', 'intestine'],
                'therapeutic_targets': True
            },
            'breast_cancer': {
                'primary_genes': ['BRCA1', 'BRCA2'],
                'splice_patterns': ['exon_skipping', 'canonical_disruption'],
                'tissues': ['breast', 'ovary'],
                'therapeutic_targets': True
            },
            'lung_cancer': {
                'primary_genes': ['MET', 'EGFR', 'KRAS'],
                'splice_patterns': ['exon_skipping', 'cryptic_activation'],
                'tissues': ['lung'],
                'therapeutic_targets': True
            },
            'als_ftd': {
                'primary_genes': ['UNC13A', 'STMN2', 'MAPT'],
                'splice_patterns': ['cryptic_exon', 'intron_retention'],
                'tissues': ['brain', 'spinal_cord'],
                'therapeutic_targets': True
            },
            'spinal_muscular_atrophy': {
                'primary_genes': ['SMN1', 'SMN2'],
                'splice_patterns': ['exon_skipping', 'splice_switching'],
                'tissues': ['muscle', 'spinal_cord'],
                'therapeutic_targets': True
            }
        }
    
    def analyze_disease_alternative_splicing(self, 
                                           disease_name: str,
                                           mutations: List[SpliceMutation],
                                           meta_model_path: Optional[Path] = None) -> DiseaseAlternativeSplicingResult:
        """
        Analyze alternative splicing patterns for a specific disease.
        
        Parameters
        ----------
        disease_name : str
            Name of the disease
        mutations : List[SpliceMutation]
            Disease-associated splice mutations
        meta_model_path : Optional[Path]
            Path to trained meta-model
            
        Returns
        -------
        DiseaseAlternativeSplicingResult
            Comprehensive analysis results
        """
        self.logger.info(f"Analyzing alternative splicing for {disease_name} with {len(mutations)} mutations")
        
        # Create disease-specific output directory
        disease_dir = self.work_dir / disease_name.replace(' ', '_').lower()
        disease_dir.mkdir(exist_ok=True)
        
        # Convert mutations to VCF format
        vcf_path = disease_dir / f"{disease_name}_mutations.vcf"
        self._mutations_to_vcf(mutations, vcf_path)
        
        # Process through alternative splicing pipeline
        alternative_sites_df = self.alternative_pipeline.process_vcf_to_alternative_sites(
            vcf_path, gene_annotations=pd.DataFrame()
        )
        
        # Convert DataFrame to AlternativeSpliceSite objects
        alternative_sites = self._dataframe_to_sites(alternative_sites_df)
        
        # Enhance with regulatory features if available
        if self.regulatory_extractor:
            alternative_sites = self.regulatory_extractor.enhance_alternative_sites_with_regulatory_features(
                alternative_sites
            )
        
        # Analyze disease-specific patterns
        patterns = self._identify_disease_patterns(disease_name, alternative_sites)
        
        # Classify splice site types
        cryptic_sites = [s for s in alternative_sites if 'cryptic' in s.splice_category.lower()]
        canonical_disrupted = [s for s in alternative_sites if 'canonical' in s.splice_category.lower() and 'disrupt' in s.splice_category.lower()]
        
        # Analyze tissue specificity
        tissue_patterns = self._analyze_tissue_specificity(disease_name, alternative_sites)
        
        # Analyze validation evidence
        validation_breakdown = self._analyze_validation_evidence(alternative_sites)
        
        # Analyze clinical significance
        clinical_breakdown = self._analyze_clinical_significance(alternative_sites)
        
        # Calculate meta-model improvement (mock for now)
        meta_improvement = self._calculate_meta_model_improvement(alternative_sites, meta_model_path)
        
        # Create result
        result = DiseaseAlternativeSplicingResult(
            disease_name=disease_name,
            total_mutations=len(mutations),
            alternative_sites_detected=len(alternative_sites),
            cryptic_sites_activated=len(cryptic_sites),
            canonical_sites_disrupted=len(canonical_disrupted),
            novel_splice_patterns=len(patterns),
            meta_model_improvement=meta_improvement,
            tissue_specific_patterns=tissue_patterns,
            validation_evidence=validation_breakdown,
            clinical_significance_breakdown=clinical_breakdown
        )
        
        # Save detailed results
        self._save_disease_analysis_results(disease_dir, result, alternative_sites, patterns)
        
        self.logger.info(f"Disease analysis complete: {len(alternative_sites)} alternative sites detected")
        return result
    
    def _mutations_to_vcf(self, mutations: List[SpliceMutation], output_path: Path):
        """Convert SpliceMutation objects to VCF format."""
        with open(output_path, 'w') as f:
            # Write VCF header
            f.write("##fileformat=VCFv4.2\n")
            f.write("##source=MetaSpliceAI_DiseaseAnalysis\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            
            # Write variants
            for mut in mutations:
                chrom = f"chr{mut.chrom}" if not mut.chrom.startswith('chr') else mut.chrom
                pos = mut.position
                ref = mut.ref_allele if mut.ref_allele else "N"
                alt = mut.alt_allele if mut.alt_allele else "N"
                
                info_parts = [f"GENE={mut.gene_symbol}"]
                if mut.splice_event_type:
                    info_parts.append(f"EVENT={mut.splice_event_type.value}")
                if mut.clinical_significance:
                    info_parts.append(f"CLIN_SIG={mut.clinical_significance}")
                
                info = ";".join(info_parts)
                f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\n")
    
    def _dataframe_to_sites(self, df: pd.DataFrame) -> List[AlternativeSpliceSite]:
        """Convert DataFrame to AlternativeSpliceSite objects."""
        sites = []
        
        for _, row in df.iterrows():
            site = AlternativeSpliceSite(
                chrom=row.get('chromosome', ''),
                position=int(row.get('position', 0)),
                strand=row.get('strand', '+'),
                site_type=row.get('site_type', 'unknown'),
                splice_category=row.get('splice_category', 'unknown'),
                delta_score=float(row.get('delta_score', 0)),
                ref_score=float(row.get('ref_score', 0)),
                alt_score=float(row.get('alt_score', 0)),
                variant_id=row.get('variant_id', ''),
                gene_symbol=row.get('gene_symbol', ''),
                clinical_significance=row.get('clinical_significance'),
                validation_evidence=row.get('validation_evidence')
            )
            sites.append(site)
        
        return sites
    
    def _identify_disease_patterns(self, 
                                 disease_name: str, 
                                 sites: List[AlternativeSpliceSite]) -> List[AlternativeSplicingPattern]:
        """Identify disease-specific alternative splicing patterns."""
        patterns = []
        
        # Get disease configuration
        config = self.disease_configs.get(disease_name.lower().replace(' ', '_'), {})
        primary_genes = config.get('primary_genes', [])
        expected_patterns = config.get('splice_patterns', [])
        
        # Group sites by gene and pattern type
        gene_sites = {}
        for site in sites:
            if site.gene_symbol not in gene_sites:
                gene_sites[site.gene_symbol] = []
            gene_sites[site.gene_symbol].append(site)
        
        # Identify patterns for each gene
        for gene_symbol, gene_sites_list in gene_sites.items():
            # Check for exon skipping patterns
            if self._has_exon_skipping_pattern(gene_sites_list):
                pattern = AlternativeSplicingPattern(
                    pattern_id=f"{disease_name}_{gene_symbol}_exon_skipping",
                    disease=disease_name,
                    gene_symbol=gene_symbol,
                    pattern_type='exon_skipping',
                    affected_sites=gene_sites_list,
                    frequency=self._calculate_pattern_frequency(gene_sites_list),
                    clinical_impact='high' if gene_symbol in primary_genes else 'medium',
                    therapeutic_target=gene_symbol in primary_genes and config.get('therapeutic_targets', False)
                )
                patterns.append(pattern)
            
            # Check for cryptic activation patterns
            cryptic_sites = [s for s in gene_sites_list if 'cryptic' in s.splice_category.lower()]
            if cryptic_sites:
                pattern = AlternativeSplicingPattern(
                    pattern_id=f"{disease_name}_{gene_symbol}_cryptic_activation",
                    disease=disease_name,
                    gene_symbol=gene_symbol,
                    pattern_type='cryptic_activation',
                    affected_sites=cryptic_sites,
                    frequency=self._calculate_pattern_frequency(cryptic_sites),
                    clinical_impact='high' if gene_symbol in primary_genes else 'medium',
                    therapeutic_target=gene_symbol in primary_genes and config.get('therapeutic_targets', False)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _has_exon_skipping_pattern(self, sites: List[AlternativeSpliceSite]) -> bool:
        """Check if sites show exon skipping pattern."""
        # Look for paired donor loss and acceptor loss
        donor_losses = [s for s in sites if s.site_type == 'donor' and s.delta_score < -0.3]
        acceptor_losses = [s for s in sites if s.site_type == 'acceptor' and s.delta_score < -0.3]
        
        return len(donor_losses) > 0 and len(acceptor_losses) > 0
    
    def _calculate_pattern_frequency(self, sites: List[AlternativeSpliceSite]) -> float:
        """Calculate frequency score for a splicing pattern."""
        if not sites:
            return 0.0
        
        # Use mean absolute delta score as frequency proxy
        return np.mean([abs(s.delta_score) for s in sites])
    
    def _analyze_tissue_specificity(self, 
                                  disease_name: str, 
                                  sites: List[AlternativeSpliceSite]) -> Dict[str, int]:
        """Analyze tissue-specific patterns for the disease."""
        config = self.disease_configs.get(disease_name.lower().replace(' ', '_'), {})
        relevant_tissues = config.get('tissues', ['brain', 'heart', 'liver', 'muscle', 'blood'])
        
        # Mock tissue-specific analysis
        tissue_patterns = {}
        for tissue in relevant_tissues:
            # Count sites with tissue-specific evidence
            tissue_specific_sites = [
                s for s in sites 
                if s.validation_evidence and tissue in s.validation_evidence.lower()
            ]
            tissue_patterns[tissue] = len(tissue_specific_sites)
        
        return tissue_patterns
    
    def _analyze_validation_evidence(self, sites: List[AlternativeSpliceSite]) -> Dict[str, int]:
        """Analyze validation evidence breakdown."""
        evidence_counts = {}
        
        for site in sites:
            if site.validation_evidence:
                evidence_types = site.validation_evidence.split('|')
                for evidence in evidence_types:
                    evidence = evidence.strip()
                    evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1
        
        return evidence_counts
    
    def _analyze_clinical_significance(self, sites: List[AlternativeSpliceSite]) -> Dict[str, int]:
        """Analyze clinical significance breakdown."""
        significance_counts = {}
        
        for site in sites:
            if site.clinical_significance:
                significance = site.clinical_significance.lower()
                significance_counts[significance] = significance_counts.get(significance, 0) + 1
            else:
                significance_counts['unknown'] = significance_counts.get('unknown', 0) + 1
        
        return significance_counts
    
    def _calculate_meta_model_improvement(self, 
                                        sites: List[AlternativeSpliceSite],
                                        meta_model_path: Optional[Path]) -> float:
        """Calculate meta-model improvement over base model."""
        # Mock implementation - would use actual meta-model predictions
        if not sites:
            return 0.0
        
        # Simulate improvement based on site characteristics
        high_confidence_sites = [
            s for s in sites 
            if abs(s.delta_score) > 0.5 and s.validation_evidence
        ]
        
        improvement = len(high_confidence_sites) / len(sites) * 0.15  # Mock 15% max improvement
        return improvement
    
    def _save_disease_analysis_results(self, 
                                     output_dir: Path,
                                     result: DiseaseAlternativeSplicingResult,
                                     sites: List[AlternativeSpliceSite],
                                     patterns: List[AlternativeSplicingPattern]):
        """Save comprehensive disease analysis results."""
        # Save summary results
        summary_file = output_dir / "disease_analysis_summary.json"
        summary_data = {
            'disease_name': result.disease_name,
            'total_mutations': result.total_mutations,
            'alternative_sites_detected': result.alternative_sites_detected,
            'cryptic_sites_activated': result.cryptic_sites_activated,
            'canonical_sites_disrupted': result.canonical_sites_disrupted,
            'novel_splice_patterns': result.novel_splice_patterns,
            'meta_model_improvement': result.meta_model_improvement,
            'tissue_specific_patterns': result.tissue_specific_patterns,
            'validation_evidence': result.validation_evidence,
            'clinical_significance_breakdown': result.clinical_significance_breakdown
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Save detailed sites
        sites_file = output_dir / "alternative_splice_sites_detailed.tsv"
        sites_data = []
        for site in sites:
            sites_data.append({
                'chromosome': site.chrom,
                'position': site.position,
                'strand': site.strand,
                'site_type': site.site_type,
                'splice_category': site.splice_category,
                'delta_score': site.delta_score,
                'ref_score': site.ref_score,
                'alt_score': site.alt_score,
                'variant_id': site.variant_id,
                'gene_symbol': site.gene_symbol,
                'clinical_significance': site.clinical_significance,
                'validation_evidence': site.validation_evidence
            })
        
        sites_df = pd.DataFrame(sites_data)
        sites_df.to_csv(sites_file, sep='\t', index=False)
        
        # Save patterns
        patterns_file = output_dir / "splice_patterns.json"
        patterns_data = []
        for pattern in patterns:
            patterns_data.append({
                'pattern_id': pattern.pattern_id,
                'disease': pattern.disease,
                'gene_symbol': pattern.gene_symbol,
                'pattern_type': pattern.pattern_type,
                'affected_sites_count': len(pattern.affected_sites),
                'frequency': pattern.frequency,
                'clinical_impact': pattern.clinical_impact,
                'therapeutic_target': pattern.therapeutic_target
            })
        
        with open(patterns_file, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        self.logger.info(f"Saved disease analysis results to {output_dir}")
    
    def compare_diseases(self, 
                        disease_results: List[DiseaseAlternativeSplicingResult]) -> pd.DataFrame:
        """Compare alternative splicing patterns across diseases."""
        comparison_data = []
        
        for result in disease_results:
            comparison_data.append({
                'disease': result.disease_name,
                'total_mutations': result.total_mutations,
                'alternative_sites': result.alternative_sites_detected,
                'cryptic_sites': result.cryptic_sites_activated,
                'canonical_disrupted': result.canonical_sites_disrupted,
                'novel_patterns': result.novel_splice_patterns,
                'meta_improvement': result.meta_model_improvement,
                'sites_per_mutation': result.alternative_sites_detected / max(result.total_mutations, 1),
                'cryptic_rate': result.cryptic_sites_activated / max(result.alternative_sites_detected, 1)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_file = self.work_dir / "disease_comparison.tsv"
        comparison_df.to_csv(comparison_file, sep='\t', index=False)
        
        self.logger.info(f"Disease comparison saved to {comparison_file}")
        return comparison_df


def demonstrate_disease_specific_analysis():
    """Demonstrate disease-specific alternative splicing analysis."""
    from pathlib import Path
    from ..data_sources.base import SpliceMutation, SpliceEventType
    
    print("\n" + "="*60)
    print("Disease-Specific Alternative Splicing Analysis Demonstration")
    print("="*60 + "\n")
    
    # Initialize analyzer
    work_dir = Path("./case_studies/disease_specific_analysis")
    analyzer = DiseaseSpecificAlternativeSplicingAnalyzer(work_dir, verbosity=2)
    
    # Create mock mutations for different diseases
    cf_mutations = [
        SpliceMutation(
            mutation_id="CFTR_c.3718-2477C>T",
            gene_symbol="CFTR",
            chrom="7",
            position=117559593,
            ref_allele="G",
            alt_allele="T",
            splice_event_type=SpliceEventType.CRYPTIC_EXON_INCLUSION,
            clinical_significance="pathogenic",
            validation_method="minigene_assay"
        )
    ]
    
    brca_mutations = [
        SpliceMutation(
            mutation_id="BRCA1_c.5266dupC",
            gene_symbol="BRCA1",
            chrom="17",
            position=43094077,
            ref_allele="G",
            alt_allele="A",
            splice_event_type=SpliceEventType.EXON_SKIPPING,
            clinical_significance="pathogenic",
            validation_method="rna_seq"
        )
    ]
    
    # Analyze each disease
    diseases_to_analyze = [
        ("cystic_fibrosis", cf_mutations),
        ("breast_cancer", brca_mutations)
    ]
    
    results = []
    for disease_name, mutations in diseases_to_analyze:
        print(f"\n🦠 Analyzing {disease_name}...")
        
        result = analyzer.analyze_disease_alternative_splicing(
            disease_name, mutations
        )
        
        results.append(result)
        
        print(f"   ✅ {result.alternative_sites_detected} alternative sites detected")
        print(f"   🧬 {result.cryptic_sites_activated} cryptic sites activated")
        print(f"   📉 {result.canonical_sites_disrupted} canonical sites disrupted")
        print(f"   📈 {result.meta_model_improvement:.3f} meta-model improvement")
    
    # Compare diseases
    print(f"\n📊 Comparing {len(results)} diseases...")
    comparison_df = analyzer.compare_diseases(results)
    
    print(f"   Disease comparison created with {len(comparison_df)} rows")
    print(f"   Columns: {list(comparison_df.columns)}")
    
    # Show top diseases by alternative sites
    if not comparison_df.empty:
        top_diseases = comparison_df.nlargest(3, 'alternative_sites')
        print(f"\n🏆 Top diseases by alternative sites:")
        for _, row in top_diseases.iterrows():
            print(f"   {row['disease']}: {row['alternative_sites']} sites")
    
    print("\n✅ Disease-specific alternative splicing analysis demonstration complete!")


if __name__ == "__main__":
    demonstrate_disease_specific_analysis()


