#!/usr/bin/env python3
"""
Universal VCF Parser for Splice Analysis

A production-ready, general-purpose VCF parser designed for splice variant analysis
across multiple data sources and annotation systems. This module provides:

- Universal VCF parsing with configurable field extraction
- Comprehensive splice variant detection using SO terms and keywords  
- Support for multiple annotation systems (VEP, SnpEff, ClinVar, etc.)
- Configurable filtering and output formats
- Integration with MetaSpliceAI workflows

Key Features:
- Handles any VCF file format (not just ClinVar)
- Configurable INFO field extraction
- Advanced splice impact classification
- Multiple output formats (TSV, Parquet, JSON)
- Production-ready error handling and logging

Usage:
    from universal_vcf_parser import UniversalVCFParser, VCFParsingConfig
    
    config = VCFParsingConfig(
        info_fields=['CLNSIG', 'MC', 'CSQ'],  # Fields to extract
        splice_detection_mode='comprehensive',
        output_format='tsv'
    )
    
    parser = UniversalVCFParser(config)
    variants_df = parser.parse_vcf('variants.vcf.gz')
"""

import pandas as pd
import numpy as np
import pysam
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AnnotationSystem(Enum):
    """Supported annotation systems."""
    CLINVAR = "clinvar"
    VEP = "vep"
    SNPEFF = "snpeff"
    ANNOVAR = "annovar"
    CUSTOM = "custom"


class SpliceDetectionMode(Enum):
    """Splice detection strategies."""
    STRICT = "strict"          # Only core splice site SO terms
    COMPREHENSIVE = "comprehensive"  # Extended SO terms + keywords
    PERMISSIVE = "permissive"  # All potential splice-affecting variants
    CUSTOM = "custom"          # User-defined criteria


@dataclass
class VCFParsingConfig:
    """Configuration for universal VCF parsing."""
    
    # Input/output settings
    input_vcf: Optional[Path] = None
    output_dir: Optional[Path] = None
    output_format: str = "tsv"  # tsv, parquet, json
    
    # Field extraction configuration
    info_fields: List[str] = field(default_factory=lambda: ['CLNSIG', 'MC', 'CSQ', 'ANN'])
    format_fields: List[str] = field(default_factory=lambda: ['GT', 'DP', 'GQ'])
    extract_all_info: bool = False  # Extract all INFO fields
    
    # Annotation system configuration
    annotation_system: AnnotationSystem = AnnotationSystem.CLINVAR
    vep_csq_format: Optional[str] = None  # VEP CSQ field format
    snpeff_ann_format: Optional[str] = None  # SnpEff ANN field format
    
    # Splice detection configuration
    splice_detection_mode: SpliceDetectionMode = SpliceDetectionMode.COMPREHENSIVE
    custom_splice_terms: List[str] = field(default_factory=list)
    custom_splice_keywords: List[str] = field(default_factory=list)
    
    # Filtering configuration
    apply_quality_filter: bool = True
    min_quality_score: Optional[float] = None
    exclude_failed_filters: bool = True
    
    # Clinical significance filtering
    pathogenicity_filter: Optional[List[str]] = None  # e.g., ['Pathogenic', 'Likely_pathogenic']
    include_uncertain: bool = True
    
    # Variant type filtering
    variant_types: Optional[List[str]] = None  # e.g., ['SNV', 'deletion', 'insertion']
    
    # Performance settings
    chunk_size: int = 10000  # Process variants in chunks
    max_variants: Optional[int] = None  # Limit for testing
    
    # Output customization
    include_sequences: bool = False  # Extract reference/alt sequences
    sequence_context: int = 50  # Context size for sequence extraction
    
    # Advanced options
    normalize_coordinates: bool = True  # Standardize coordinate representation
    validate_reference: bool = True  # Validate REF alleles against FASTA
    reference_fasta: Optional[Path] = None


class UniversalVCFParser:
    """
    Universal VCF parser for splice variant analysis.
    
    Designed to handle VCF files from any source with configurable field extraction
    and comprehensive splice variant detection.
    """
    
    # Comprehensive Sequence Ontology terms for splice detection
    CORE_SPLICE_SO_TERMS = {
        'SO:0001575': 'splice_donor_variant',
        'SO:0001574': 'splice_acceptor_variant', 
        'SO:0001630': 'splice_region_variant',
        'SO:0001629': 'splice_polypyrimidine_tract_variant',
    }
    
    EXTENDED_SPLICE_SO_TERMS = {
        # Intronic variants (cryptic sites)
        'SO:0001627': 'intron_variant',
        'SO:0002019': 'start_lost',
        'SO:0001624': '5_prime_UTR_variant',
        'SO:0001623': '3_prime_UTR_variant',
        
        # Exonic variants (ESE/ESS effects)
        'SO:0001583': 'missense_variant',
        'SO:0001819': 'synonymous_variant',
        'SO:0001587': 'stop_gained',
        'SO:0001578': 'stop_lost',
        
        # Extended splice-specific terms
        'SO:0001968': 'splice_donor_5th_base_variant',
        'SO:0001969': 'splice_donor_region_variant',
        'SO:0001970': 'splice_acceptor_region_variant',
        
        # Structural variants
        'SO:0001822': 'inframe_insertion',
        'SO:0001821': 'inframe_deletion',
    }
    
    # Keywords for fallback splice detection
    SPLICE_KEYWORDS = {
        'splice', 'donor', 'acceptor', 'intron', 'exon', 'splice_site',
        'canonical_splice', 'cryptic_splice', 'branch_point', 'polypyrimidine',
        'splicing', 'splice_region', 'exon_boundary'
    }
    
    # Enhanced splice mechanism categories
    SPLICE_MECHANISM_CATEGORIES = {
        'direct_splice_site': 'Direct disruption of canonical splice sites (±1-2bp)',
        'exonic_boundary_effect': 'Exonic variants in splice region (±3bp boundary)',
        'exonic_ese_ess_effect': 'Exonic splicing enhancer/silencer disruption',
        'exonic_extended_region': 'Extended exonic splice regions (±4-8bp)',
        'intronic_cryptic_site': 'Intronic variants creating/disrupting cryptic sites',
        'utr_regulatory_effect': 'UTR variants affecting splicing regulation',
        'structural_splice_effect': 'Large structural variants affecting splice sites',
        'polypyrimidine_tract_effect': 'Variants affecting polypyrimidine tract',
        'keyword_based_detection': 'Detected by splice-related keywords only',
        'indirect_splice_effect': 'Other potential splice effects'
    }
    
    def __init__(self, config: VCFParsingConfig):
        """Initialize parser with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize reference FASTA if provided
        self.fasta = None
        if config.reference_fasta and config.reference_fasta.exists():
            try:
                from pyfaidx import Fasta
                self.fasta = Fasta(str(config.reference_fasta))
                self.logger.info(f"Loaded reference FASTA: {config.reference_fasta}")
            except ImportError:
                self.logger.warning("pyfaidx not available - sequence extraction disabled")
        
        # Initialize annotation system handlers
        self._init_annotation_handlers()
        
        # Initialize splice detection terms based on mode
        self._init_splice_detection_terms()
    
    def _init_annotation_handlers(self):
        """Initialize annotation system-specific handlers."""
        self.annotation_handlers = {
            AnnotationSystem.CLINVAR: self._parse_clinvar_annotations,
            AnnotationSystem.VEP: self._parse_vep_annotations,
            AnnotationSystem.SNPEFF: self._parse_snpeff_annotations,
            AnnotationSystem.ANNOVAR: self._parse_annovar_annotations,
            AnnotationSystem.CUSTOM: self._parse_custom_annotations,
        }
    
    def _init_splice_detection_terms(self):
        """Initialize splice detection terms based on configuration."""
        if self.config.splice_detection_mode == SpliceDetectionMode.STRICT:
            self.splice_so_terms = self.CORE_SPLICE_SO_TERMS
            self.splice_keywords = set()
        elif self.config.splice_detection_mode == SpliceDetectionMode.COMPREHENSIVE:
            self.splice_so_terms = {**self.CORE_SPLICE_SO_TERMS, **self.EXTENDED_SPLICE_SO_TERMS}
            self.splice_keywords = self.SPLICE_KEYWORDS
        elif self.config.splice_detection_mode == SpliceDetectionMode.PERMISSIVE:
            self.splice_so_terms = {**self.CORE_SPLICE_SO_TERMS, **self.EXTENDED_SPLICE_SO_TERMS}
            self.splice_keywords = self.SPLICE_KEYWORDS | {'variant', 'mutation'}  # Very broad
        else:  # CUSTOM
            self.splice_so_terms = {term: term for term in self.config.custom_splice_terms}
            self.splice_keywords = set(self.config.custom_splice_keywords)
    
    def parse_vcf(self, vcf_path: Union[str, Path]) -> pd.DataFrame:
        """
        Parse VCF file and extract variant information.
        
        Parameters
        ----------
        vcf_path : Union[str, Path]
            Path to VCF file
            
        Returns
        -------
        pd.DataFrame
            Parsed variants with configurable fields
        """
        vcf_path = Path(vcf_path)
        if not vcf_path.exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")
        
        self.logger.info(f"Parsing VCF file: {vcf_path}")
        
        variants = []
        variant_count = 0
        
        try:
            with pysam.VariantFile(str(vcf_path)) as vcf:
                # Extract header information
                header_info = self._extract_header_info(vcf.header)
                
                for record in vcf:
                    if self.config.max_variants and variant_count >= self.config.max_variants:
                        break
                    
                    # Skip multiallelic variants if not split
                    if record.alts and len(record.alts) > 1:
                        self.logger.debug(f"Skipping multiallelic variant at {record.chrom}:{record.pos}")
                        continue
                    
                    # Parse variant record
                    variant_data = self._parse_variant_record(record, header_info)
                    if variant_data:
                        variants.append(variant_data)
                        variant_count += 1
                    
                    if variant_count % self.config.chunk_size == 0:
                        self.logger.info(f"Processed {variant_count} variants...")
        
        except Exception as e:
            self.logger.error(f"Error parsing VCF file: {str(e)}")
            raise
        
        self.logger.info(f"Successfully parsed {len(variants)} variants")
        
        # Convert to DataFrame
        df = pd.DataFrame(variants)
        
        if df.empty:
            self.logger.warning("No variants were parsed from VCF file")
            return df
        
        # Apply post-processing
        df = self._post_process_variants(df)
        
        # Apply filters if configured
        if self._should_apply_filters():
            df = self._apply_filters(df)
        
        # Save output if configured
        if self.config.output_dir:
            self._save_output(df)
        
        return df
    
    def _extract_header_info(self, header) -> Dict:
        """Extract relevant information from VCF header."""
        info = {
            'contigs': list(header.contigs.keys()),
            'info_fields': list(header.info.keys()),
            'format_fields': list(header.formats.keys()) if hasattr(header, 'formats') else [],
            'samples': list(header.samples) if hasattr(header, 'samples') else [],
        }
        
        # Extract annotation system information
        if 'CSQ' in header.info:
            # VEP annotations
            csq_description = header.info['CSQ'].description
            info['vep_csq_format'] = self._parse_vep_csq_format(csq_description)
        
        if 'ANN' in header.info:
            # SnpEff annotations  
            ann_description = header.info['ANN'].description
            info['snpeff_ann_format'] = self._parse_snpeff_ann_format(ann_description)
        
        return info
    
    def _parse_variant_record(self, record, header_info: Dict) -> Optional[Dict]:
        """Parse a single VCF variant record."""
        try:
            # Basic VCF fields
            variant_data = {
                'chrom': record.chrom,
                'pos': record.pos,
                'id': record.id or '.',
                'ref': record.ref,
                'alt': record.alts[0] if record.alts else '.',
                'qual': record.qual if record.qual is not None else '.',
                'filter': ';'.join(record.filter) if record.filter else '.',
            }
            
            # Extract INFO fields
            info_data = self._extract_info_fields(record, header_info)
            variant_data.update(info_data)
            
            # Extract FORMAT fields if samples present
            if record.samples and self.config.format_fields:
                format_data = self._extract_format_fields(record)
                variant_data.update(format_data)
            
            # Classify variant type
            variant_data['variant_type'] = self._classify_variant_type(
                variant_data['ref'], variant_data['alt']
            )
            
            # Detect splice impact
            splice_info = self._detect_splice_impact(variant_data, header_info)
            variant_data.update(splice_info)
            
            # Extract sequences if requested
            if self.config.include_sequences and self.fasta:
                sequence_data = self._extract_sequences(variant_data)
                variant_data.update(sequence_data)
            
            return variant_data
            
        except Exception as e:
            self.logger.warning(f"Error parsing variant {record.chrom}:{record.pos}: {e}")
            return None
    
    def _extract_info_fields(self, record, header_info: Dict) -> Dict:
        """Extract INFO fields based on configuration."""
        info_data = {}
        
        if self.config.extract_all_info:
            # Extract all available INFO fields
            for key in header_info['info_fields']:
                value = record.info.get(key)
                info_data[f"INFO_{key}"] = self._format_info_value(value)
        else:
            # Extract specified INFO fields
            for field in self.config.info_fields:
                if field in record.info:
                    value = record.info.get(field)
                    info_data[field] = self._format_info_value(value)
        
        # Parse annotation system-specific fields
        annotation_handler = self.annotation_handlers.get(self.config.annotation_system)
        if annotation_handler:
            annotation_data = annotation_handler(record, header_info)
            info_data.update(annotation_data)
        
        return info_data
    
    def _format_info_value(self, value) -> str:
        """Format INFO field value for consistent output."""
        if value is None:
            return '.'
        elif isinstance(value, (list, tuple)):
            return ','.join(str(v) for v in value)
        else:
            return str(value)
    
    def _extract_format_fields(self, record) -> Dict:
        """Extract FORMAT fields from sample data."""
        format_data = {}
        
        if record.samples:
            # For now, extract from first sample (can be extended for multi-sample)
            sample = next(iter(record.samples.values()))
            
            for field in self.config.format_fields:
                if field in sample:
                    value = sample[field]
                    format_data[f"FORMAT_{field}"] = self._format_info_value(value)
        
        return format_data
    
    def _classify_variant_type(self, ref: str, alt: str) -> str:
        """Classify variant type based on REF and ALT alleles."""
        if alt == '.':
            return 'unknown'
        elif len(ref) == 1 and len(alt) == 1:
            return 'SNV'
        elif len(ref) > len(alt):
            return 'deletion'
        elif len(ref) < len(alt):
            return 'insertion'
        else:
            return 'complex'
    
    def _detect_splice_impact(self, variant_data: Dict, header_info: Dict) -> Dict:
        """Detect splice impact based on annotations."""
        splice_info = {
            'affects_splicing': False,
            'splice_confidence': 'none',  # none, low, medium, high
            'splice_mechanism': [],  # List of potential mechanisms
            'splice_terms': [],  # SO terms found
        }
        
        # Method 1: SO term detection
        so_terms_found = []
        for field_name, field_value in variant_data.items():
            if field_value and field_value != '.':
                field_str = str(field_value).lower()
                
                # Check for SO terms
                for so_term, description in self.splice_so_terms.items():
                    if so_term.lower() in field_str or description.lower() in field_str:
                        so_terms_found.append(so_term)
        
        # Method 2: Keyword detection
        keywords_found = []
        for field_name, field_value in variant_data.items():
            if field_value and field_value != '.':
                field_str = str(field_value).lower()
                
                for keyword in self.splice_keywords:
                    if keyword in field_str:
                        keywords_found.append(keyword)
        
        # Determine splice impact
        if so_terms_found:
            splice_info['affects_splicing'] = True
            splice_info['splice_terms'] = so_terms_found
            
            # Enhanced mechanism and confidence classification
            mechanism, confidence = self._classify_splice_mechanism_and_confidence(so_terms_found, variant_data)
            splice_info['splice_confidence'] = confidence
            splice_info['splice_mechanism'].append(mechanism)
        
        elif keywords_found and self.config.splice_detection_mode != SpliceDetectionMode.STRICT:
            splice_info['affects_splicing'] = True
            splice_info['splice_confidence'] = 'low'
            splice_info['splice_mechanism'] = ['keyword_based_detection']
        
        return splice_info
    
    def _classify_splice_mechanism_and_confidence(self, so_terms_found: List[str], variant_data: Dict) -> Tuple[str, str]:
        """
        Enhanced classification of splice mechanism and confidence.
        
        Parameters
        ----------
        so_terms_found : List[str]
            List of SO terms found in variant annotations
        variant_data : Dict
            Variant data dictionary
            
        Returns
        -------
        Tuple[str, str]
            (mechanism, confidence_level)
        """
        # Convert SO terms to descriptions for easier matching
        term_descriptions = []
        for term in so_terms_found:
            if term in self.CORE_SPLICE_SO_TERMS:
                term_descriptions.append(self.CORE_SPLICE_SO_TERMS[term])
            elif term in self.EXTENDED_SPLICE_SO_TERMS:
                term_descriptions.append(self.EXTENDED_SPLICE_SO_TERMS[term])
            else:
                term_descriptions.append(term)
        
        # HIGH confidence mechanisms
        if any(desc in ['splice_donor_variant', 'splice_acceptor_variant'] for desc in term_descriptions):
            return 'direct_splice_site', 'high'
        
        if 'splice_region_variant' in term_descriptions:
            return 'exonic_boundary_effect', 'high'
        
        if 'splice_polypyrimidine_tract_variant' in term_descriptions:
            return 'polypyrimidine_tract_effect', 'high'
        
        # MEDIUM confidence mechanisms (intronic)
        if any('intron' in desc.lower() for desc in term_descriptions):
            return 'intronic_cryptic_site', 'medium'
        
        # LOW confidence mechanisms (exonic effects)
        if any(desc in ['missense_variant', 'synonymous_variant', 'stop_gained', 'stop_lost'] for desc in term_descriptions):
            return 'exonic_ese_ess_effect', 'low'
        
        if any(desc in ['splice_donor_region_variant', 'splice_acceptor_region_variant', 'splice_donor_5th_base_variant'] for desc in term_descriptions):
            return 'exonic_extended_region', 'low'
        
        # UTR effects
        if any(desc in ['5_prime_UTR_variant', '3_prime_UTR_variant'] for desc in term_descriptions):
            return 'utr_regulatory_effect', 'low'
        
        # Structural variants
        if any(desc in ['inframe_insertion', 'inframe_deletion'] for desc in term_descriptions):
            return 'structural_splice_effect', 'low'
        
        # Fallback for other terms
        return 'indirect_splice_effect', 'low'
    
    def get_mechanism_description(self, mechanism: str) -> str:
        """
        Get human-readable description for splice mechanism.
        
        Parameters
        ----------
        mechanism : str
            Mechanism category
            
        Returns
        -------
        str
            Human-readable description
        """
        return self.SPLICE_MECHANISM_CATEGORIES.get(mechanism, f"Unknown mechanism: {mechanism}")
    
    def _extract_sequences(self, variant_data: Dict) -> Dict:
        """Extract reference and alternative sequences."""
        sequence_data = {}
        
        if not self.fasta:
            return sequence_data
        
        try:
            chrom = variant_data['chrom']
            pos = int(variant_data['pos'])
            ref = variant_data['ref']
            alt = variant_data['alt']
            
            # Normalize chromosome name
            chrom_variants = [chrom, f"chr{chrom}", chrom.replace("chr", "")]
            actual_chrom = None
            
            for variant in chrom_variants:
                if variant in self.fasta.keys():
                    actual_chrom = variant
                    break
            
            if not actual_chrom:
                return sequence_data
            
            # Extract context sequence
            context_start = max(0, pos - self.config.sequence_context - 1)
            context_end = pos + len(ref) + self.config.sequence_context - 1
            
            # Get reference sequence
            ref_context = str(self.fasta[actual_chrom][context_start:context_end]).upper()
            
            # Construct alternative sequence
            var_offset = pos - context_start - 1
            alt_context = (
                ref_context[:var_offset] +
                alt +
                ref_context[var_offset + len(ref):]
            )
            
            sequence_data.update({
                'ref_sequence': ref_context,
                'alt_sequence': alt_context,
                'sequence_start': context_start + 1,  # Convert to 1-based
                'sequence_end': context_end,
                'variant_position_in_sequence': var_offset + 1,
            })
            
        except Exception as e:
            self.logger.warning(f"Error extracting sequences for {variant_data.get('chrom')}:{variant_data.get('pos')}: {e}")
        
        return sequence_data
    
    def _parse_clinvar_annotations(self, record, header_info: Dict) -> Dict:
        """Parse ClinVar-specific annotations."""
        data = {}
        
        # Clinical significance
        if 'CLNSIG' in record.info:
            clnsig = record.info['CLNSIG']
            data['clinical_significance'] = self._format_info_value(clnsig)
            
            # Determine pathogenicity
            if clnsig:
                clnsig_str = str(clnsig).lower().replace(' ', '_')
                data['is_pathogenic'] = any(term in clnsig_str 
                                          for term in ['pathogenic', 'likely_pathogenic'])
            else:
                data['is_pathogenic'] = False
        
        # Molecular consequence
        if 'MC' in record.info:
            data['molecular_consequence'] = self._format_info_value(record.info['MC'])
        
        # Disease information
        if 'CLNDN' in record.info:
            data['disease'] = self._format_info_value(record.info['CLNDN'])
        
        # Review status
        if 'CLNREVSTAT' in record.info:
            data['review_status'] = self._format_info_value(record.info['CLNREVSTAT'])
        
        return data
    
    def _parse_vep_annotations(self, record, header_info: Dict) -> Dict:
        """Parse VEP CSQ annotations."""
        data = {}
        
        if 'CSQ' in record.info and 'vep_csq_format' in header_info:
            csq_data = record.info['CSQ']
            csq_format = header_info['vep_csq_format']
            
            if csq_data and csq_format:
                # Parse first CSQ entry (can be extended for multiple transcripts)
                csq_values = str(csq_data).split(',')[0].split('|')
                
                if len(csq_values) >= len(csq_format):
                    for i, field_name in enumerate(csq_format):
                        if i < len(csq_values):
                            data[f"VEP_{field_name}"] = csq_values[i] or '.'
        
        return data
    
    def _parse_snpeff_annotations(self, record, header_info: Dict) -> Dict:
        """Parse SnpEff ANN annotations."""
        data = {}
        
        if 'ANN' in record.info:
            ann_data = record.info['ANN']
            
            if ann_data:
                # Parse first ANN entry
                ann_values = str(ann_data).split(',')[0].split('|')
                
                # SnpEff standard format
                if len(ann_values) >= 4:
                    data['SNPEFF_effect'] = ann_values[1] or '.'
                    data['SNPEFF_impact'] = ann_values[2] or '.'
                    data['SNPEFF_gene'] = ann_values[3] or '.'
                    data['SNPEFF_transcript'] = ann_values[6] if len(ann_values) > 6 else '.'
        
        return data
    
    def _parse_annovar_annotations(self, record, header_info: Dict) -> Dict:
        """Parse ANNOVAR annotations."""
        data = {}
        
        # ANNOVAR typically uses multiple INFO fields
        annovar_fields = ['Func.refGene', 'Gene.refGene', 'ExonicFunc.refGene']
        for field in annovar_fields:
            if field in record.info:
                data[f"ANNOVAR_{field.replace('.', '_')}"] = self._format_info_value(record.info[field])
        
        return data
    
    def _parse_custom_annotations(self, record, header_info: Dict) -> Dict:
        """Parse custom annotations based on configuration."""
        data = {}
        
        # Extract any fields specified in config
        for field in self.config.info_fields:
            if field in record.info:
                data[field] = self._format_info_value(record.info[field])
        
        return data
    
    def _parse_vep_csq_format(self, description: str) -> List[str]:
        """Parse VEP CSQ field format from header description."""
        # Extract field names from VEP CSQ description
        # Format: "Consequence annotations from Ensembl VEP. Format: Allele|Consequence|..."
        if 'Format:' in description:
            format_part = description.split('Format:')[1].strip()
            return format_part.split('|')
        return []
    
    def _parse_snpeff_ann_format(self, description: str) -> List[str]:
        """Parse SnpEff ANN field format from header description."""
        # SnpEff has a standard format
        return [
            'Allele', 'Annotation', 'Annotation_Impact', 'Gene_Name', 'Gene_ID',
            'Feature_Type', 'Feature_ID', 'Transcript_BioType', 'Rank', 'HGVS.c',
            'HGVS.p', 'cDNA.pos/cDNA.length', 'CDS.pos/CDS.length', 'AA.pos/AA.length',
            'Distance', 'ERRORS/WARNINGS/INFO'
        ]
    
    def _post_process_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing to parsed variants."""
        # Convert data types
        if 'pos' in df.columns:
            df['pos'] = pd.to_numeric(df['pos'], errors='coerce')
        
        if 'qual' in df.columns:
            df['qual'] = pd.to_numeric(df['qual'], errors='coerce')
        
        # Standardize boolean columns
        bool_columns = ['affects_splicing', 'is_pathogenic']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        return df
    
    def _should_apply_filters(self) -> bool:
        """Determine if any filters should be applied."""
        return (
            self.config.apply_quality_filter or
            self.config.pathogenicity_filter or
            self.config.variant_types or
            self.config.min_quality_score is not None
        )
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply configured filters to variants."""
        original_count = len(df)
        
        # Quality filtering
        if self.config.apply_quality_filter and self.config.min_quality_score is not None:
            if 'qual' in df.columns:
                df = df[df['qual'] >= self.config.min_quality_score]
                self.logger.info(f"Quality filter: {len(df)}/{original_count} variants passed")
        
        # Filter failed variants
        if self.config.exclude_failed_filters and 'filter' in df.columns:
            df = df[df['filter'].isin(['.', 'PASS'])]
            self.logger.info(f"Filter status: {len(df)}/{original_count} variants passed")
        
        # Pathogenicity filtering
        if self.config.pathogenicity_filter and 'clinical_significance' in df.columns:
            pathogenic_mask = df['clinical_significance'].str.contains(
                '|'.join(self.config.pathogenicity_filter), case=False, na=False
            )
            df = df[pathogenic_mask]
            self.logger.info(f"Pathogenicity filter: {len(df)}/{original_count} variants passed")
        
        # Variant type filtering
        if self.config.variant_types and 'variant_type' in df.columns:
            df = df[df['variant_type'].isin(self.config.variant_types)]
            self.logger.info(f"Variant type filter: {len(df)}/{original_count} variants passed")
        
        return df
    
    def _save_output(self, df: pd.DataFrame):
        """Save parsed variants to file."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == 'tsv':
            output_file = self.config.output_dir / "parsed_variants.tsv"
            df.to_csv(output_file, sep='\t', index=False)
        elif self.config.output_format == 'parquet':
            output_file = self.config.output_dir / "parsed_variants.parquet"
            df.to_parquet(output_file, index=False)
        elif self.config.output_format == 'json':
            output_file = self.config.output_dir / "parsed_variants.json"
            df.to_json(output_file, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")
        
        self.logger.info(f"Saved {len(df)} variants to: {output_file}")
        
        # Save summary statistics
        stats_file = self.config.output_dir / "parsing_stats.json"
        stats = self._generate_parsing_stats(df)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Saved parsing statistics to: {stats_file}")
    
    def _generate_parsing_stats(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive parsing statistics."""
        stats = {
            'total_variants': len(df),
            'variant_types': df['variant_type'].value_counts().to_dict() if 'variant_type' in df.columns else {},
            'chromosomes': df['chrom'].value_counts().to_dict() if 'chrom' in df.columns else {},
        }
        
        # Splice-specific statistics
        if 'affects_splicing' in df.columns:
            stats['splice_affecting'] = int(df['affects_splicing'].sum())
            stats['splice_percentage'] = float(df['affects_splicing'].mean() * 100)
        
        if 'is_pathogenic' in df.columns:
            stats['pathogenic'] = int(df['is_pathogenic'].sum())
            stats['pathogenic_percentage'] = float(df['is_pathogenic'].mean() * 100)
        
        if 'affects_splicing' in df.columns and 'is_pathogenic' in df.columns:
            splice_pathogenic = df['affects_splicing'] & df['is_pathogenic']
            stats['splice_pathogenic'] = int(splice_pathogenic.sum())
        
        # Quality statistics
        if 'qual' in df.columns:
            qual_series = pd.to_numeric(df['qual'], errors='coerce')
            stats['quality_stats'] = {
                'mean': float(qual_series.mean()),
                'median': float(qual_series.median()),
                'min': float(qual_series.min()),
                'max': float(qual_series.max()),
            }
        
        return stats


# Factory functions for common configurations
def create_clinvar_parser(
    splice_detection: str = "comprehensive",
    include_uncertain: bool = True,
    include_sequences: bool = False
) -> UniversalVCFParser:
    """Create a ClinVar-optimized VCF parser."""
    config = VCFParsingConfig(
        info_fields=['CLNSIG', 'MC', 'CLNDN', 'CLNREVSTAT', 'TYPE'],
        annotation_system=AnnotationSystem.CLINVAR,
        splice_detection_mode=SpliceDetectionMode(splice_detection),
        include_uncertain=include_uncertain,
        include_sequences=include_sequences,
        apply_quality_filter=True,
        exclude_failed_filters=True,
    )
    return UniversalVCFParser(config)


def create_vep_parser(
    splice_detection: str = "comprehensive",
    pathogenicity_filter: Optional[List[str]] = None,
    include_sequences: bool = False
) -> UniversalVCFParser:
    """Create a VEP-optimized VCF parser."""
    config = VCFParsingConfig(
        info_fields=['CSQ'],
        annotation_system=AnnotationSystem.VEP,
        splice_detection_mode=SpliceDetectionMode(splice_detection),
        pathogenicity_filter=pathogenicity_filter,
        include_sequences=include_sequences,
        apply_quality_filter=True,
    )
    return UniversalVCFParser(config)


def create_research_parser(
    info_fields: List[str] = None,
    splice_detection: str = "permissive",
    max_variants: Optional[int] = None
) -> UniversalVCFParser:
    """Create a research-optimized VCF parser with flexible configuration."""
    config = VCFParsingConfig(
        info_fields=info_fields or ['CLNSIG', 'MC', 'CSQ', 'ANN'],
        extract_all_info=True if not info_fields else False,
        annotation_system=AnnotationSystem.CUSTOM,
        splice_detection_mode=SpliceDetectionMode(splice_detection),
        include_uncertain=True,
        include_sequences=True,
        max_variants=max_variants,
        apply_quality_filter=False,  # Keep all variants for research
    )
    return UniversalVCFParser(config)


def parse_vcf_for_splice_analysis(
    vcf_path: Union[str, Path],
    output_dir: Union[str, Path],
    annotation_system: str = "clinvar",
    splice_detection: str = "comprehensive",
    reference_fasta: Optional[Union[str, Path]] = None,
    max_variants: Optional[int] = None
) -> pd.DataFrame:
    """
    Convenience function to parse VCF for splice analysis.
    
    Parameters
    ----------
    vcf_path : Union[str, Path]
        Path to VCF file
    output_dir : Union[str, Path] 
        Output directory for results
    annotation_system : str
        Annotation system ('clinvar', 'vep', 'snpeff', 'custom')
    splice_detection : str
        Splice detection mode ('strict', 'comprehensive', 'permissive')
    reference_fasta : Optional[Union[str, Path]]
        Reference FASTA for sequence extraction
    max_variants : Optional[int]
        Maximum variants to process (for testing)
        
    Returns
    -------
    pd.DataFrame
        Parsed variants ready for splice analysis
    """
    config = VCFParsingConfig(
        input_vcf=Path(vcf_path),
        output_dir=Path(output_dir),
        annotation_system=AnnotationSystem(annotation_system.lower()),
        splice_detection_mode=SpliceDetectionMode(splice_detection),
        reference_fasta=Path(reference_fasta) if reference_fasta else None,
        include_sequences=True if reference_fasta else False,
        max_variants=max_variants,
    )
    
    parser = UniversalVCFParser(config)
    return parser.parse_vcf(vcf_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Universal VCF Parser for Splice Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse ClinVar VCF with comprehensive splice detection
  python universal_vcf_parser.py \\
      --vcf clinvar.vcf.gz \\
      --output-dir results/ \\
      --annotation-system clinvar \\
      --splice-detection comprehensive
  
  # Parse VEP-annotated VCF with sequence extraction
  python universal_vcf_parser.py \\
      --vcf variants_vep.vcf.gz \\
      --output-dir results/ \\
      --annotation-system vep \\
      --reference-fasta GRCh38.fa \\
      --include-sequences
  
  # Research mode with all fields
  python universal_vcf_parser.py \\
      --vcf research_variants.vcf.gz \\
      --output-dir results/ \\
      --splice-detection permissive \\
      --extract-all-info \\
      --max-variants 1000
        """
    )
    
    parser.add_argument('--vcf', required=True, help='Input VCF file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--annotation-system', default='clinvar', 
                       choices=['clinvar', 'vep', 'snpeff', 'annovar', 'custom'],
                       help='Annotation system type')
    parser.add_argument('--splice-detection', default='comprehensive',
                       choices=['strict', 'comprehensive', 'permissive', 'custom'],
                       help='Splice detection mode')
    parser.add_argument('--reference-fasta', help='Reference FASTA file for sequence extraction')
    parser.add_argument('--include-sequences', action='store_true',
                       help='Extract reference and alternative sequences')
    parser.add_argument('--extract-all-info', action='store_true',
                       help='Extract all INFO fields')
    parser.add_argument('--max-variants', type=int, help='Maximum variants to process')
    parser.add_argument('--output-format', default='tsv', choices=['tsv', 'parquet', 'json'],
                       help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = VCFParsingConfig(
        input_vcf=Path(args.vcf),
        output_dir=Path(args.output_dir),
        annotation_system=AnnotationSystem(args.annotation_system),
        splice_detection_mode=SpliceDetectionMode(args.splice_detection),
        reference_fasta=Path(args.reference_fasta) if args.reference_fasta else None,
        include_sequences=args.include_sequences,
        extract_all_info=args.extract_all_info,
        max_variants=args.max_variants,
        output_format=args.output_format,
    )
    
    # Parse VCF
    parser = UniversalVCFParser(config)
    df = parser.parse_vcf(args.vcf)
    
    print(f"\n✅ Successfully parsed {len(df)} variants")
    if 'affects_splicing' in df.columns:
        splice_count = df['affects_splicing'].sum()
        print(f"🧬 Found {splice_count} splice-affecting variants ({splice_count/len(df)*100:.1f}%)")
    
    if 'is_pathogenic' in df.columns:
        pathogenic_count = df['is_pathogenic'].sum()
        print(f"⚠️  Found {pathogenic_count} pathogenic variants ({pathogenic_count/len(df)*100:.1f}%)")

