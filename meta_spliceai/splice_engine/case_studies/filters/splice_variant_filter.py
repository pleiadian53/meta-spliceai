#!/usr/bin/env python3
"""
Comprehensive Splice Variant Filter Module

This module provides advanced filtering capabilities for splice-affecting variants
from multiple data sources (ClinVar, SpliceVarDB, etc.) with support for:

- Comprehensive Sequence Ontology (SO) term coverage
- Clinical significance and review status filtering
- Population frequency filtering (when available)
- Data source-specific filtering logic
- Configurable filtering criteria

Usage:
    from splice_variant_filter import SpliceVariantFilter, FilterConfig
    
    config = FilterConfig(
        pathogenicity_threshold='likely_pathogenic',
        include_uncertain=False,
        max_population_frequency=0.01
    )
    
    filter = SpliceVariantFilter(config)
    filtered_variants = filter.filter_variants(variants_df, source='clinvar')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Set, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources for splice variant filtering."""
    CLINVAR = "clinvar"
    SPLICEVARDB = "splicevardb"
    MUTSPLICEDB = "mutsplicedb"
    DBASS = "dbass"
    CUSTOM = "custom"


class PathogenicityLevel(Enum):
    """Pathogenicity levels in order of severity."""
    BENIGN = 1
    LIKELY_BENIGN = 2
    UNCERTAIN = 3
    LIKELY_PATHOGENIC = 4
    PATHOGENIC = 5


@dataclass
class FilterConfig:
    """Configuration for splice variant filtering."""
    
    # Pathogenicity filtering
    pathogenicity_threshold: str = "likely_pathogenic"  # Minimum pathogenicity level
    include_uncertain: bool = False  # Include variants of uncertain significance
    require_review_status: bool = True  # Require evidence-based review status
    min_review_confidence: str = "criteria_provided"  # Minimum review confidence
    
    # Splice impact filtering
    splice_terms_strict: bool = True  # Use strict SO term matching vs keyword matching
    include_deep_intronic: bool = True  # Include deep intronic variants (potential cryptic sites)
    intronic_distance_threshold: int = 50  # Max distance from exon boundary for intronic variants
    
    # Population frequency filtering
    enable_frequency_filter: bool = False  # Enable population frequency filtering
    max_population_frequency: float = 0.01  # Maximum allele frequency (1%)
    frequency_sources: List[str] = field(default_factory=lambda: ["gnomAD", "1000G", "ESP"])
    
    # Quality filtering
    min_quality_score: Optional[float] = None  # Minimum variant quality score
    exclude_failed_filters: bool = True  # Exclude variants that failed quality filters
    
    # Data source specific settings
    clinvar_specific: Dict[str, Any] = field(default_factory=dict)
    splicevardb_specific: Dict[str, Any] = field(default_factory=dict)


class SpliceVariantFilter:
    """Comprehensive filter for splice-affecting variants from multiple data sources."""
    
    # Comprehensive Sequence Ontology terms for splice-affecting variants
    SPLICE_SO_TERMS = {
        # Core splice site variants
        'SO:0001575': 'splice_donor_variant',
        'SO:0001574': 'splice_acceptor_variant',
        'SO:0001630': 'splice_region_variant',
        'SO:0001629': 'splice_polypyrimidine_tract_variant',
        
        # Intronic variants (potential cryptic sites)
        'SO:0001627': 'intron_variant',
        'SO:0002019': 'start_lost',  # Can affect splicing in 5' UTR
        'SO:0001624': '5_prime_UTR_variant',
        'SO:0001623': '3_prime_UTR_variant',
        
        # Exonic variants that can affect splicing
        'SO:0001792': 'non_coding_transcript_exon_variant',
        'SO:0001619': 'non_coding_transcript_variant',
        'SO:0001580': 'coding_sequence_variant',
        
        # Extended splice-related terms
        'SO:0001968': 'splice_donor_5th_base_variant',
        'SO:0001969': 'splice_donor_region_variant',
        'SO:0001970': 'splice_acceptor_region_variant',
    }
    
    # Keywords for fallback splice detection (when SO terms not available)
    SPLICE_KEYWORDS = {
        'splice', 'donor', 'acceptor', 'intronic', 'intron', 'exon_boundary',
        'splice_site', 'canonical_splice', 'cryptic_splice', 'branch_point'
    }
    
    # ClinVar clinical significance mapping
    CLINVAR_PATHOGENICITY_MAP = {
        'pathogenic': PathogenicityLevel.PATHOGENIC,
        'likely_pathogenic': PathogenicityLevel.LIKELY_PATHOGENIC,
        'pathogenic/likely_pathogenic': PathogenicityLevel.PATHOGENIC,
        'uncertain_significance': PathogenicityLevel.UNCERTAIN,
        'likely_benign': PathogenicityLevel.LIKELY_BENIGN,
        'benign': PathogenicityLevel.BENIGN,
        'benign/likely_benign': PathogenicityLevel.BENIGN,
    }
    
    # ClinVar review status confidence levels
    CLINVAR_REVIEW_CONFIDENCE = {
        'practice_guideline': 5,
        'reviewed_by_expert_panel': 4,
        'criteria_provided_multiple_submitters_no_conflicts': 3,
        'criteria_provided_conflicting_interpretations': 2,
        'criteria_provided_single_submitter': 2,
        'no_assertion_criteria_provided': 1,
        'no_assertion_provided': 0,
    }
    
    def __init__(self, config: FilterConfig):
        """
        Initialize splice variant filter.
        
        Parameters
        ----------
        config : FilterConfig
            Filter configuration settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Precompile regex patterns for efficiency
        self._compile_patterns()
        
        # Initialize data source specific handlers
        self._init_data_source_handlers()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # HGVS intronic position pattern (e.g., c.123+45A>G, c.456-12del)
        self.intronic_pattern = re.compile(r'[+-]\d+', re.IGNORECASE)
        
        # Population frequency patterns
        self.frequency_patterns = {
            'gnomAD': re.compile(r'AF_gnomAD?[_=]([0-9.e-]+)', re.IGNORECASE),
            '1000G': re.compile(r'AF_1000G?[_=]([0-9.e-]+)', re.IGNORECASE),
            'ESP': re.compile(r'AF_ESP[_=]([0-9.e-]+)', re.IGNORECASE),
        }
    
    def _init_data_source_handlers(self):
        """Initialize data source specific filtering handlers."""
        self.source_handlers = {
            DataSource.CLINVAR: self._filter_clinvar_specific,
            DataSource.SPLICEVARDB: self._filter_splicevardb_specific,
            DataSource.MUTSPLICEDB: self._filter_mutsplicedb_specific,
            DataSource.DBASS: self._filter_dbass_specific,
            DataSource.CUSTOM: self._filter_custom_specific,
        }
    
    def filter_variants(
        self,
        variants_df: pd.DataFrame,
        source: Union[str, DataSource],
        return_stats: bool = True
    ) -> Union[pd.DataFrame, tuple]:
        """
        Filter variants based on splice impact and pathogenicity criteria.
        
        Parameters
        ----------
        variants_df : pd.DataFrame
            Input variants dataframe
        source : str or DataSource
            Data source type for source-specific filtering
        return_stats : bool
            Whether to return filtering statistics
        
        Returns
        -------
        pd.DataFrame or tuple
            Filtered variants dataframe, optionally with filtering statistics
        """
        if isinstance(source, str):
            source = DataSource(source.lower())
        
        self.logger.info(f"Filtering {len(variants_df)} variants from {source.value}")
        
        # Initialize filtering statistics
        stats = {
            'input_count': len(variants_df),
            'splice_affecting': 0,
            'pathogenic': 0,
            'high_confidence': 0,
            'frequency_filtered': 0,
            'quality_filtered': 0,
            'final_count': 0,
            'filter_stages': {}
        }
        
        # Stage 1: Identify splice-affecting variants
        splice_mask = self._identify_splice_affecting(variants_df, source)
        variants_splice = variants_df[splice_mask].copy()
        stats['splice_affecting'] = len(variants_splice)
        stats['filter_stages']['splice_affecting'] = len(variants_splice)
        
        if variants_splice.empty:
            self.logger.warning("No splice-affecting variants found")
            return (variants_splice, stats) if return_stats else variants_splice
        
        # Stage 2: Filter by pathogenicity
        pathogenic_mask = self._filter_by_pathogenicity(variants_splice, source)
        variants_pathogenic = variants_splice[pathogenic_mask].copy()
        stats['pathogenic'] = len(variants_pathogenic)
        stats['filter_stages']['pathogenic'] = len(variants_pathogenic)
        
        # Stage 3: Filter by review status/confidence
        if self.config.require_review_status:
            confidence_mask = self._filter_by_confidence(variants_pathogenic, source)
            variants_confident = variants_pathogenic[confidence_mask].copy()
            stats['high_confidence'] = len(variants_confident)
            stats['filter_stages']['high_confidence'] = len(variants_confident)
        else:
            variants_confident = variants_pathogenic
        
        # Stage 4: Population frequency filtering
        if self.config.enable_frequency_filter:
            frequency_mask = self._filter_by_frequency(variants_confident, source)
            variants_freq_filtered = variants_confident[frequency_mask].copy()
            stats['frequency_filtered'] = len(variants_freq_filtered)
            stats['filter_stages']['frequency_filtered'] = len(variants_freq_filtered)
        else:
            variants_freq_filtered = variants_confident
        
        # Stage 5: Quality filtering
        quality_mask = self._filter_by_quality(variants_freq_filtered, source)
        variants_final = variants_freq_filtered[quality_mask].copy()
        stats['quality_filtered'] = len(variants_final)
        stats['final_count'] = len(variants_final)
        stats['filter_stages']['final'] = len(variants_final)
        
        # Stage 6: Data source specific filtering
        variants_final = self.source_handlers[source](variants_final)
        stats['final_count'] = len(variants_final)
        
        self.logger.info(f"Filtering complete: {stats['input_count']} → {stats['final_count']} variants")
        
        return (variants_final, stats) if return_stats else variants_final
    
    def _identify_splice_affecting(self, df: pd.DataFrame, source: DataSource) -> pd.Series:
        """Identify splice-affecting variants using SO terms and keywords."""
        splice_mask = pd.Series(False, index=df.index)
        
        # Method 1: SO term matching (preferred)
        if self.config.splice_terms_strict:
            so_columns = ['molecular_consequence', 'mc', 'consequence', 'annotation']
            for col in so_columns:
                if col in df.columns:
                    so_terms_pattern = '|'.join(self.SPLICE_SO_TERMS.values())
                    splice_mask |= df[col].str.contains(so_terms_pattern, case=False, na=False)
                    break
        
        # Method 2: Keyword matching (fallback)
        if not splice_mask.any():
            keyword_columns = ['molecular_consequence', 'mc', 'consequence', 'annotation', 'effect']
            for col in keyword_columns:
                if col in df.columns:
                    keywords_pattern = '|'.join(self.SPLICE_KEYWORDS)
                    splice_mask |= df[col].str.contains(keywords_pattern, case=False, na=False)
                    break
        
        # Method 3: HGVS-based detection for intronic variants
        if self.config.include_deep_intronic:
            hgvs_columns = ['hgvs_c', 'hgvs_cdna', 'cdna_change', 'hgvs']
            for col in hgvs_columns:
                if col in df.columns:
                    # Look for intronic positions (e.g., c.123+45A>G)
                    intronic_mask = df[col].str.contains(self.intronic_pattern, na=False)
                    
                    # Filter by distance threshold if specified
                    if self.config.intronic_distance_threshold:
                        def check_distance(hgvs_str):
                            if pd.isna(hgvs_str):
                                return False
                            matches = self.intronic_pattern.findall(str(hgvs_str))
                            if matches:
                                distances = [abs(int(match.replace('+', '').replace('-', ''))) 
                                           for match in matches]
                                return any(d <= self.config.intronic_distance_threshold for d in distances)
                            return False
                        
                        distance_mask = df[col].apply(check_distance)
                        intronic_mask &= distance_mask
                    
                    splice_mask |= intronic_mask
                    break
        
        return splice_mask
    
    def _filter_by_pathogenicity(self, df: pd.DataFrame, source: DataSource) -> pd.Series:
        """Filter variants by pathogenicity level."""
        pathogenic_mask = pd.Series(False, index=df.index)
        
        # Get pathogenicity threshold
        threshold_str = self.config.pathogenicity_threshold.lower().replace(' ', '_')
        threshold_level = PathogenicityLevel.LIKELY_PATHOGENIC  # Default
        
        for key, level in self.CLINVAR_PATHOGENICITY_MAP.items():
            if threshold_str in key:
                threshold_level = level
                break
        
        # Check clinical significance columns
        clin_sig_columns = ['clinical_significance', 'clnsig', 'pathogenicity', 'significance']
        for col in clin_sig_columns:
            if col in df.columns:
                for idx, value in df[col].items():
                    if pd.isna(value):
                        continue
                    
                    value_clean = str(value).lower().replace(' ', '_')
                    variant_level = PathogenicityLevel.UNCERTAIN  # Default
                    
                    # Map to pathogenicity level
                    for key, level in self.CLINVAR_PATHOGENICITY_MAP.items():
                        if key in value_clean:
                            variant_level = level
                            break
                    
                    # Apply threshold
                    meets_threshold = variant_level.value >= threshold_level.value
                    
                    # Handle uncertain significance
                    if variant_level == PathogenicityLevel.UNCERTAIN:
                        meets_threshold = self.config.include_uncertain
                    
                    pathogenic_mask.loc[idx] = meets_threshold
                
                break
        
        return pathogenic_mask
    
    def _filter_by_confidence(self, df: pd.DataFrame, source: DataSource) -> pd.Series:
        """Filter variants by review status and confidence level."""
        confidence_mask = pd.Series(True, index=df.index)  # Default: pass all
        
        if source == DataSource.CLINVAR:
            review_columns = ['review_status', 'clnrevstat', 'review_confidence']
            min_confidence = self.config.min_review_confidence.lower().replace(' ', '_')
            
            for col in review_columns:
                if col in df.columns:
                    for idx, value in df[col].items():
                        if pd.isna(value):
                            confidence_mask.loc[idx] = False
                            continue
                        
                        value_clean = str(value).lower().replace(' ', '_').replace(',', '_')
                        
                        # Get confidence score
                        confidence_score = 0
                        for status, score in self.CLINVAR_REVIEW_CONFIDENCE.items():
                            if status in value_clean:
                                confidence_score = max(confidence_score, score)
                        
                        # Get minimum required score
                        min_score = self.CLINVAR_REVIEW_CONFIDENCE.get(min_confidence, 2)
                        
                        confidence_mask.loc[idx] = confidence_score >= min_score
                    
                    break
        
        return confidence_mask
    
    def _filter_by_frequency(self, df: pd.DataFrame, source: DataSource) -> pd.Series:
        """Filter variants by population frequency."""
        frequency_mask = pd.Series(True, index=df.index)  # Default: pass all
        
        # Look for frequency information in various columns
        freq_columns = ['af', 'allele_frequency', 'population_frequency', 'gnomad_af', 'info']
        
        for col in freq_columns:
            if col in df.columns:
                for idx, value in df[col].items():
                    if pd.isna(value):
                        continue
                    
                    max_freq = 0.0
                    value_str = str(value)
                    
                    # Try direct numeric conversion
                    try:
                        freq = float(value_str)
                        max_freq = max(max_freq, freq)
                    except ValueError:
                        # Try regex extraction for INFO fields
                        for source_name, pattern in self.frequency_patterns.items():
                            if source_name.lower() in self.config.frequency_sources:
                                matches = pattern.findall(value_str)
                                if matches:
                                    try:
                                        freq = float(matches[0])
                                        max_freq = max(max_freq, freq)
                                    except ValueError:
                                        continue
                    
                    # Apply frequency threshold
                    frequency_mask.loc[idx] = max_freq <= self.config.max_population_frequency
                
                break
        
        return frequency_mask
    
    def _filter_by_quality(self, df: pd.DataFrame, source: DataSource) -> pd.Series:
        """Filter variants by quality metrics."""
        quality_mask = pd.Series(True, index=df.index)  # Default: pass all
        
        # Quality score filtering
        if self.config.min_quality_score is not None:
            qual_columns = ['qual', 'quality', 'quality_score']
            for col in qual_columns:
                if col in df.columns:
                    quality_mask &= (df[col] >= self.config.min_quality_score) | df[col].isna()
                    break
        
        # Filter status filtering
        if self.config.exclude_failed_filters:
            filter_columns = ['filter', 'filter_status', 'filters']
            for col in filter_columns:
                if col in df.columns:
                    # Pass variants with PASS, missing, or empty filter status
                    pass_mask = (df[col] == 'PASS') | (df[col] == '.') | df[col].isna() | (df[col] == '')
                    quality_mask &= pass_mask
                    break
        
        return quality_mask
    
    # Data source specific filtering methods
    def _filter_clinvar_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ClinVar-specific filtering logic."""
        # Additional ClinVar-specific filters can be added here
        return df
    
    def _filter_splicevardb_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply SpliceVarDB-specific filtering logic."""
        # SpliceVarDB variants are already experimentally validated
        return df
    
    def _filter_mutsplicedb_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply MutSpliceDB-specific filtering logic."""
        return df
    
    def _filter_dbass_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply DBASS-specific filtering logic."""
        return df
    
    def _filter_custom_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply custom filtering logic."""
        return df
    
    def get_filter_summary(self, stats: Dict[str, Any]) -> str:
        """Generate a human-readable filter summary."""
        summary = f"""
Splice Variant Filtering Summary
================================
Input variants: {stats['input_count']:,}
Splice-affecting: {stats['splice_affecting']:,}
Pathogenic: {stats['pathogenic']:,}
High confidence: {stats['high_confidence']:,}
Frequency filtered: {stats['frequency_filtered']:,}
Quality filtered: {stats['quality_filtered']:,}
Final variants: {stats['final_count']:,}

Retention rate: {stats['final_count']/stats['input_count']*100:.1f}%
        """
        return summary.strip()


# Factory functions for common filter configurations
def create_clinvar_filter(
    pathogenicity_threshold: str = "likely_pathogenic",
    include_uncertain: bool = False,
    require_review_status: bool = True,
    enable_frequency_filter: bool = False
) -> SpliceVariantFilter:
    """Create a ClinVar-optimized splice variant filter."""
    config = FilterConfig(
        pathogenicity_threshold=pathogenicity_threshold,
        include_uncertain=include_uncertain,
        require_review_status=require_review_status,
        min_review_confidence="criteria_provided",
        enable_frequency_filter=enable_frequency_filter,
        max_population_frequency=0.01,
        splice_terms_strict=True,
        include_deep_intronic=True,
        intronic_distance_threshold=50
    )
    return SpliceVariantFilter(config)


def create_research_filter(
    include_uncertain: bool = True,
    max_population_frequency: float = 0.05,
    require_review_status: bool = False
) -> SpliceVariantFilter:
    """Create a research-oriented filter with more permissive criteria."""
    config = FilterConfig(
        pathogenicity_threshold="uncertain",
        include_uncertain=include_uncertain,
        require_review_status=require_review_status,
        enable_frequency_filter=True,
        max_population_frequency=max_population_frequency,
        splice_terms_strict=False,
        include_deep_intronic=True,
        intronic_distance_threshold=200
    )
    return SpliceVariantFilter(config)


def create_clinical_filter(
    pathogenicity_threshold: str = "pathogenic",
    require_review_status: bool = True,
    max_population_frequency: float = 0.001
) -> SpliceVariantFilter:
    """Create a clinical-grade filter with strict criteria."""
    config = FilterConfig(
        pathogenicity_threshold=pathogenicity_threshold,
        include_uncertain=False,
        require_review_status=require_review_status,
        min_review_confidence="criteria_provided_multiple_submitters_no_conflicts",
        enable_frequency_filter=True,
        max_population_frequency=max_population_frequency,
        splice_terms_strict=True,
        include_deep_intronic=False,
        exclude_failed_filters=True
    )
    return SpliceVariantFilter(config)
