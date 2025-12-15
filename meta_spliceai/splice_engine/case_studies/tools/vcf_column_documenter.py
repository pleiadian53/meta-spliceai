#!/usr/bin/env python3
"""
VCF Column Documentation Tool

A comprehensive tool for analyzing and documenting VCF column values, their meanings,
and possible values in both structured (JSON) and human-readable (Markdown) formats.

This tool is specifically designed for ClinVar and other VCF files to provide:
- Column definitions and meanings
- Value enumeration and frequency analysis
- Human-readable documentation generation
- Structured metadata for programmatic use

Usage:
    python vcf_column_documenter.py --vcf clinvar.vcf.gz --output-dir docs/
    python vcf_column_documenter.py --vcf clinvar.vcf.gz --output-dir docs/ --max-variants 10000
"""

import argparse
import json
import logging
import pandas as pd
import pysam
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import CaseStudyResourceManager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)


def resolve_structured_path(path_str: str) -> Path:
    """
    Resolve a path using the MetaSpliceAI structured convention.
    
    Supports both absolute paths and relative paths using the data/ structure:
    - data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz
    - data/ensembl/clinvar/vcf/docs/
    
    Parameters
    ----------
    path_str : str
        Path string to resolve
        
    Returns
    -------
    Path
        Resolved absolute path
    """
    path = Path(path_str)
    
    # If already absolute, return as-is
    if path.is_absolute():
        return path
    
    # If relative and starts with 'data/', resolve against project root
    if str(path).startswith('data/'):
        project_root = Path(__file__).parent.parent.parent.parent.parent
        return project_root / path
    
    # For other relative paths, resolve against current working directory
    return Path.cwd() / path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent.parent


@dataclass
class ColumnDocumentation:
    """Documentation for a single column."""
    
    name: str
    description: str
    data_type: str
    possible_values: List[str] = field(default_factory=list)
    value_counts: Dict[str, int] = field(default_factory=dict)
    null_count: int = 0
    unique_count: int = 0
    example_values: List[str] = field(default_factory=list)
    source: str = "VCF"  # VCF, INFO, FORMAT, etc.
    is_required: bool = False
    notes: str = ""


@dataclass
class VCFDocumentationConfig:
    """Configuration for VCF documentation generation."""
    
    input_vcf: Path
    output_dir: Path
    max_variants: Optional[int] = None
    sample_size: int = 10000  # For value enumeration
    include_rare_values: bool = True
    min_frequency: float = 0.001  # 0.1% minimum frequency
    output_formats: List[str] = field(default_factory=lambda: ['json', 'markdown'])
    verbose: bool = True
    
    # Enhanced configuration options
    max_categorical_values: int = 100  # Max values to show in main report
    max_numerical_examples: int = 10   # Max examples for numerical columns
    generate_lookup_tables: bool = True  # Generate separate lookup files
    use_smart_sampling: bool = False    # Default: process full dataset for accurate statistics


class VCFColumnDocumenter:
    """Comprehensive VCF column documentation tool."""
    
    # ClinVar-specific column definitions
    CLINVAR_COLUMN_DEFINITIONS = {
        # Standard VCF columns
        'CHROM': {
            'description': 'Chromosome or contig name',
            'data_type': 'string',
            'source': 'VCF',
            'is_required': True,
            'notes': 'Chromosome identifier (1-22, X, Y, MT)'
        },
        'POS': {
            'description': 'Reference position (1-based)',
            'data_type': 'integer',
            'source': 'VCF',
            'is_required': True,
            'notes': 'Genomic coordinate of the variant'
        },
        'ID': {
            'description': 'Variant identifier',
            'data_type': 'string',
            'source': 'VCF',
            'is_required': False,
            'notes': 'ClinVar variation ID or rsID'
        },
        'REF': {
            'description': 'Reference allele',
            'data_type': 'string',
            'source': 'VCF',
            'is_required': True,
            'notes': 'Reference sequence at this position'
        },
        'ALT': {
            'description': 'Alternative allele(s)',
            'data_type': 'string',
            'source': 'VCF',
            'is_required': True,
            'notes': 'Alternative sequence(s) separated by commas'
        },
        'QUAL': {
            'description': 'Quality score',
            'data_type': 'float',
            'source': 'VCF',
            'is_required': False,
            'notes': 'Phred-scaled quality score'
        },
        'FILTER': {
            'description': 'Filter status',
            'data_type': 'string',
            'source': 'VCF',
            'is_required': False,
            'notes': 'PASS or filter names that failed'
        },
        
        # ClinVar-specific INFO fields
        'CLNSIG': {
            'description': 'Clinical significance',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Pathogenic, Likely_pathogenic, Uncertain_significance, etc.'
        },
        'CLNREVSTAT': {
            'description': 'Review status',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Review status of the clinical significance'
        },
        'MC': {
            'description': 'Molecular consequence',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Sequence Ontology terms describing variant effect'
        },
        'CLNDN': {
            'description': 'Disease name',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Disease or phenotype associated with the variant'
        },
        'CLNACC': {
            'description': 'ClinVar accession',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'ClinVar accession number'
        },
        'CLNDISDB': {
            'description': 'Disease database',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Database identifiers for the disease'
        },
        'CLNHGVS': {
            'description': 'HGVS nomenclature',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Human Genome Variation Society nomenclature'
        },
        'CLNSRC': {
            'description': 'Clinical significance source',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Source of the clinical significance assertion'
        },
        'CLNSRCID': {
            'description': 'Source identifier',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Identifier in the source database'
        },
        'CLNVC': {
            'description': 'Variant type',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Type of variant (SNV, deletion, insertion, etc.)'
        },
        'CLNVCSO': {
            'description': 'Sequence Ontology ID',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Sequence Ontology identifier for the variant type'
        },
        'CLNVI': {
            'description': 'Variant identifier',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Variant identifier in external databases'
        },
        'GENEINFO': {
            'description': 'Gene information',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Gene symbol and identifier'
        },
        'ORIGIN': {
            'description': 'Allele origin',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Origin of the variant (germline, somatic, etc.)'
        },
        'SSR': {
            'description': 'Somatic status',
            'data_type': 'integer',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Somatic status (0=unknown, 1=germline, 2=somatic, 3=both)'
        },
        'TYPE': {
            'description': 'Variant type (bcftools)',
            'data_type': 'string',
            'source': 'INFO',
            'is_required': False,
            'notes': 'Variant type as determined by bcftools'
        }
    }
    
    # Known value mappings for ClinVar fields
    CLINVAR_VALUE_MEANINGS = {
        'CLNSIG': {
            'Pathogenic': 'Variant is known to cause disease',
            'Likely_pathogenic': 'Variant is likely to cause disease',
            'Uncertain_significance': 'Clinical significance is uncertain',
            'Likely_benign': 'Variant is likely not disease-causing',
            'Benign': 'Variant is known not to cause disease',
            'Conflicting_interpretations_of_pathogenicity': 'Different sources disagree on significance',
            'not_provided': 'Clinical significance not provided',
            'other': 'Other clinical significance'
        },
        'CLNREVSTAT': {
            'practice_guideline': 'Reviewed by practice guideline',
            'expert_panel': 'Reviewed by expert panel',
            'multiple_submitters': 'Multiple submitters, no conflicts',
            'single_submitter': 'Single submitter',
            'no_assertion_criteria_provided': 'No assertion criteria provided',
            'no_assertion_provided': 'No assertion provided',
            'no_interpretation_for_the_single_variant': 'No interpretation for single variant',
            'conflicting_interpretations': 'Conflicting interpretations',
            'criteria_provided': 'Criteria provided',
            'no_assertion': 'No assertion'
        },
        'ORIGIN': {
            'germline': 'Variant present in germline',
            'somatic': 'Variant present in somatic cells',
            'de_novo': 'Variant arose de novo',
            'inherited': 'Variant inherited from parent',
            'maternal': 'Variant inherited from mother',
            'paternal': 'Variant inherited from father',
            'biparental': 'Variant inherited from both parents',
            'uniparental': 'Variant inherited from one parent',
            'not_provided': 'Origin not provided',
            'unknown': 'Origin unknown'
        },
        'SSR': {
            '0': 'Unknown somatic status',
            '1': 'Germline variant',
            '2': 'Somatic variant',
            '3': 'Both germline and somatic'
        }
    }
    
    def __init__(self, config: VCFDocumentationConfig):
        """Initialize the VCF column documenter."""
        self.config = config
        self.logger = self._setup_logging()
        self.documentation = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        return logger
    
    def analyze_vcf_columns(self) -> Dict[str, ColumnDocumentation]:
        """Analyze VCF columns and generate comprehensive documentation."""
        self.logger.info(f"🔍 Analyzing VCF columns: {self.config.input_vcf}")
        
        # Read VCF header to get column information
        header_info = self._read_vcf_header()
        
        # Sample variants for value analysis
        sample_data = self._sample_variants()
        
        # Generate documentation for each column
        documentation = {}
        
        # Standard VCF columns
        for col in ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER']:
            if col in sample_data.columns:
                documentation[col] = self._document_column(col, sample_data[col], header_info)
        
        # INFO fields
        for info_field in header_info.get('info_fields', []):
            if info_field in sample_data.columns:
                documentation[info_field] = self._document_column(info_field, sample_data[info_field], header_info)
        
        # FORMAT fields (if any)
        for format_field in header_info.get('format_fields', []):
            col_name = f"FORMAT_{format_field}"
            if col_name in sample_data.columns:
                documentation[col_name] = self._document_column(col_name, sample_data[col_name], header_info)
        
        self.documentation = documentation
        self.logger.info(f"✅ Documented {len(documentation)} columns")
        return documentation
    
    def _read_vcf_header(self) -> Dict[str, Any]:
        """Read VCF header information."""
        header_info = {
            'info_fields': [],
            'format_fields': [],
            'samples': []
        }
        
        try:
            with pysam.VariantFile(str(self.config.input_vcf)) as vcf:
                # Get INFO fields
                for field in vcf.header.info:
                    header_info['info_fields'].append(field)
                
                # Get FORMAT fields
                for field in vcf.header.formats:
                    header_info['format_fields'].append(field)
                
                # Get sample names
                header_info['samples'] = list(vcf.header.samples)
                
        except Exception as e:
            self.logger.error(f"Error reading VCF header: {e}")
            raise
        
        return header_info
    
    def _sample_variants(self) -> pd.DataFrame:
        """Sample variants from VCF for analysis using smart sampling strategy."""
        # Default to full dataset analysis unless explicitly using smart sampling
        if self.config.use_smart_sampling and not self.config.max_variants:
            return self._smart_sample_variants()
        else:
            return self._simple_sample_variants()
    
    def _smart_sample_variants(self) -> pd.DataFrame:
        """Smart sampling to ensure representative coverage of all chromosomes and value types."""
        self.logger.info(f"📊 Smart sampling variants for comprehensive analysis")
        
        variants = []
        variant_count = 0
        chromosome_counts = {}
        max_per_chromosome = 5000  # Ensure we get variants from all chromosomes
        
        try:
            with pysam.VariantFile(str(self.config.input_vcf)) as vcf:
                for record in vcf:
                    chrom = record.chrom
                    
                    # Smart sampling: limit per chromosome to ensure diversity
                    if chromosome_counts.get(chrom, 0) >= max_per_chromosome:
                        continue
                    
                    # Convert record to dictionary
                    variant_data = self._record_to_dict(record)
                    variants.append(variant_data)
                    variant_count += 1
                    chromosome_counts[chrom] = chromosome_counts.get(chrom, 0) + 1
                    
                    if variant_count % 10000 == 0:
                        self.logger.info(f"  Processed {variant_count:,} variants from {len(chromosome_counts)} chromosomes...")
                    
                    # Stop if we have good coverage (reasonable sample size with chromosome diversity)
                    if variant_count >= 100000 and len(chromosome_counts) >= 20:
                        self.logger.info(f"  Good coverage achieved: {len(chromosome_counts)} chromosomes")
                        break
        
        except Exception as e:
            self.logger.error(f"Error in smart sampling: {e}")
            raise
        
        self.logger.info(f"✅ Smart sampled {len(variants):,} variants from {len(chromosome_counts)} chromosomes")
        self.logger.info(f"  Chromosome distribution: {dict(list(chromosome_counts.items())[:10])}...")
        return pd.DataFrame(variants)
    
    def _simple_sample_variants(self) -> pd.DataFrame:
        """Simple sequential sampling (original method)."""
        self.logger.info(f"📊 Sequential sampling variants (max: {self.config.max_variants or 'all'})")
        
        variants = []
        variant_count = 0
        
        try:
            with pysam.VariantFile(str(self.config.input_vcf)) as vcf:
                for record in vcf:
                    if self.config.max_variants and variant_count >= self.config.max_variants:
                        break
                    
                    variant_data = self._record_to_dict(record)
                    variants.append(variant_data)
                    variant_count += 1
                    
                    if variant_count % 10000 == 0:
                        self.logger.info(f"  Processed {variant_count:,} variants...")
        
        except Exception as e:
            self.logger.error(f"Error sampling variants: {e}")
            raise
        
        self.logger.info(f"✅ Sampled {len(variants):,} variants")
        return pd.DataFrame(variants)
    
    def _record_to_dict(self, record) -> Dict:
        """Convert VCF record to dictionary."""
        variant_data = {
            'CHROM': record.chrom,
            'POS': record.pos,
            'ID': record.id if record.id else '.',
            'REF': record.ref,
            'ALT': ','.join(str(alt) for alt in record.alts) if record.alts else '.',
            'QUAL': record.qual if record.qual else '.',
            'FILTER': ','.join(record.filter.keys()) if record.filter else 'PASS'
        }
        
        # Add INFO fields
        for key, value in record.info.items():
            if isinstance(value, (list, tuple)):
                variant_data[key] = ','.join(str(v) for v in value)
            else:
                variant_data[key] = str(value) if value is not None else '.'
        
        # Add FORMAT fields (from first sample if available)
        if record.samples:
            sample = next(iter(record.samples.values()))
            for key, value in sample.items():
                if isinstance(value, (list, tuple)):
                    variant_data[f"FORMAT_{key}"] = ','.join(str(v) for v in value)
                else:
                    variant_data[f"FORMAT_{key}"] = str(value) if value is not None else '.'
        
        return variant_data
    
    def _document_column(self, col_name: str, series: pd.Series, header_info: Dict) -> ColumnDocumentation:
        """Generate documentation for a single column with intelligent value handling."""
        self.logger.debug(f"Documenting column: {col_name}")
        
        # Get base definition
        definition = self.CLINVAR_COLUMN_DEFINITIONS.get(col_name, {})
        
        # Analyze values
        value_counts = series.value_counts().to_dict()
        null_count = series.isnull().sum() + (series == '.').sum()
        unique_count = series.nunique()
        
        # Determine data type
        data_type = self._infer_data_type(series)
        
        # Smart value enumeration based on column type
        possible_values, truncated_values = self._get_smart_value_enumeration(col_name, series, value_counts, data_type)
        
        # Get example values (smart examples for different types)
        example_values = self._get_smart_examples(col_name, series, data_type)
        
        # Get value meanings if available
        value_meanings = self.CLINVAR_VALUE_MEANINGS.get(col_name, {})
        
        doc = ColumnDocumentation(
            name=col_name,
            description=definition.get('description', f'Column {col_name}'),
            data_type=data_type,
            possible_values=possible_values,
            value_counts=value_counts if not truncated_values else dict(list(value_counts.items())[:self.config.max_categorical_values]),
            null_count=int(null_count),
            unique_count=int(unique_count),
            example_values=example_values,
            source=definition.get('source', 'VCF'),
            is_required=definition.get('is_required', False),
            notes=definition.get('notes', '')
        )
        
        # Add truncation note if values were truncated
        if truncated_values:
            doc.notes += f" [Note: {unique_count:,} unique values total, showing top {len(possible_values)}]"
        
        return doc
    
    def _get_smart_value_enumeration(self, col_name: str, series: pd.Series, value_counts: Dict, data_type: str) -> Tuple[List[str], bool]:
        """Smart value enumeration based on column characteristics."""
        truncated = False
        
        # Numerical columns (especially positions): show examples only
        if col_name in ['POS'] or (data_type in ['integer', 'float', 'numeric'] and series.nunique() > 1000):
            # For numerical columns with many values, show range and examples
            sorted_values = sorted([v for v in value_counts.keys() if str(v) != '.'])
            if len(sorted_values) > self.config.max_numerical_examples:
                # Show min, max, and some examples from different ranges
                examples = []
                examples.append(str(sorted_values[0]))  # Min
                examples.append(str(sorted_values[len(sorted_values)//4]))  # 25th percentile
                examples.append(str(sorted_values[len(sorted_values)//2]))  # Median
                examples.append(str(sorted_values[3*len(sorted_values)//4]))  # 75th percentile
                examples.append(str(sorted_values[-1]))  # Max
                # Add a few random examples
                import random
                random_samples = random.sample(sorted_values[1:-1], min(5, len(sorted_values)-2))
                examples.extend([str(v) for v in random_samples])
                return examples[:self.config.max_numerical_examples], True
            else:
                return [str(v) for v in sorted_values], False
        
        # Categorical columns: limit enumeration
        elif series.nunique() > self.config.max_categorical_values:
            # Show most frequent values
            top_values = list(value_counts.keys())[:self.config.max_categorical_values]
            return [str(v) for v in top_values], True
        
        # Small categorical columns: show all
        else:
            return [str(v) for v in value_counts.keys()], False
    
    def _get_smart_examples(self, col_name: str, series: pd.Series, data_type: str) -> List[str]:
        """Get smart examples based on column type."""
        clean_series = series.dropna()
        clean_series = clean_series[clean_series != '.']
        
        if len(clean_series) == 0:
            return []
        
        # For numerical columns, show range
        if data_type in ['integer', 'float', 'numeric']:
            try:
                numeric_values = pd.to_numeric(clean_series, errors='coerce').dropna()
                if len(numeric_values) > 0:
                    examples = [
                        f"Min: {numeric_values.min()}",
                        f"Max: {numeric_values.max()}",
                        f"Median: {numeric_values.median()}"
                    ]
                    return examples
            except:
                pass
        
        # For categorical columns, show diverse examples
        return clean_series.head(5).tolist()
    
    def _infer_data_type(self, series: pd.Series) -> str:
        """Infer data type from series values."""
        # Remove null values and '.' for type inference
        clean_series = series.dropna()
        clean_series = clean_series[clean_series != '.']
        
        if len(clean_series) == 0:
            return 'string'
        
        # Try to convert to numeric
        try:
            pd.to_numeric(clean_series, errors='raise')
            return 'numeric'
        except (ValueError, TypeError):
            pass
        
        # Check if all values are integers
        try:
            clean_series.astype(int)
            return 'integer'
        except (ValueError, TypeError):
            pass
        
        # Check if all values are floats
        try:
            clean_series.astype(float)
            return 'float'
        except (ValueError, TypeError):
            pass
        
        return 'string'
    
    def generate_structured_output(self) -> Dict[str, Any]:
        """Generate structured JSON output."""
        structured_output = {
            'metadata': {
                'vcf_file': str(self.config.input_vcf),
                'analysis_date': pd.Timestamp.now().isoformat(),
                'total_columns': len(self.documentation),
                'sample_size': self.config.sample_size,
                'max_variants': self.config.max_variants
            },
            'columns': {}
        }
        
        for col_name, col_doc in self.documentation.items():
            structured_output['columns'][col_name] = {
                'name': col_doc.name,
                'description': col_doc.description,
                'data_type': col_doc.data_type,
                'source': col_doc.source,
                'is_required': col_doc.is_required,
                'notes': col_doc.notes,
                'statistics': {
                    'total_values': len(col_doc.value_counts),
                    'null_count': col_doc.null_count,
                    'unique_count': col_doc.unique_count,
                    'example_values': col_doc.example_values
                },
                'possible_values': col_doc.possible_values,
                'value_counts': col_doc.value_counts
            }
        
        return structured_output
    
    def generate_human_readable_output(self) -> str:
        """Generate human-readable Markdown output."""
        md_content = []
        
        # Header
        md_content.append("# VCF Column Documentation")
        md_content.append("")
        md_content.append(f"**VCF File**: `{self.config.input_vcf}`")
        md_content.append(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append(f"**Total Columns**: {len(self.documentation)}")
        md_content.append(f"**Sample Size**: {self.config.sample_size:,} variants")
        md_content.append("")
        
        # Summary table
        md_content.append("## Column Summary")
        md_content.append("")
        md_content.append("| Column | Type | Source | Required | Description |")
        md_content.append("|--------|------|--------|----------|-------------|")
        
        for col_name, col_doc in self.documentation.items():
            required = "✅" if col_doc.is_required else "❌"
            md_content.append(f"| `{col_name}` | {col_doc.data_type} | {col_doc.source} | {required} | {col_doc.description} |")
        
        md_content.append("")
        
        # Detailed documentation for each column
        for col_name, col_doc in self.documentation.items():
            md_content.append(f"## {col_name}")
            md_content.append("")
            md_content.append(f"**Description**: {col_doc.description}")
            md_content.append(f"**Data Type**: {col_doc.data_type}")
            md_content.append(f"**Source**: {col_doc.source}")
            md_content.append(f"**Required**: {'Yes' if col_doc.is_required else 'No'}")
            
            if col_doc.notes:
                md_content.append(f"**Notes**: {col_doc.notes}")
            
            md_content.append("")
            
            # Statistics
            md_content.append("### Statistics")
            md_content.append(f"- **Total Values**: {len(col_doc.value_counts):,}")
            md_content.append(f"- **Unique Values**: {col_doc.unique_count:,}")
            md_content.append(f"- **Null Values**: {col_doc.null_count:,}")
            md_content.append("")
            
            # Value counts
            if col_doc.value_counts:
                md_content.append("### Value Distribution")
                md_content.append("")
                md_content.append("| Value | Count | Percentage |")
                md_content.append("|-------|-------|------------|")
                
                total_count = sum(col_doc.value_counts.values())
                for value, count in sorted(col_doc.value_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_count) * 100
                    md_content.append(f"| `{value}` | {count:,} | {percentage:.2f}% |")
                
                md_content.append("")
            
            # Value meanings (if available)
            value_meanings = self.CLINVAR_VALUE_MEANINGS.get(col_name, {})
            if value_meanings:
                md_content.append("### Value Meanings")
                md_content.append("")
                for value, meaning in value_meanings.items():
                    if value in col_doc.value_counts:
                        md_content.append(f"- **`{value}`**: {meaning}")
                md_content.append("")
            
            md_content.append("---")
            md_content.append("")
        
        return "\n".join(md_content)
    
    def save_documentation(self):
        """Save documentation in requested formats."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate outputs
        structured_output = self.generate_structured_output()
        human_readable_output = self.generate_human_readable_output()
        
        # Save JSON output
        if 'json' in self.config.output_formats:
            json_file = self.config.output_dir / "vcf_column_documentation.json"
            with open(json_file, 'w') as f:
                json.dump(structured_output, f, indent=2)
            self.logger.info(f"✅ Saved structured documentation: {json_file}")
        
        # Save Markdown output
        if 'markdown' in self.config.output_formats:
            md_file = self.config.output_dir / "vcf_column_documentation.md"
            with open(md_file, 'w') as f:
                f.write(human_readable_output)
            self.logger.info(f"✅ Saved human-readable documentation: {md_file}")
        
        # Generate lookup tables for columns with many values
        if self.config.generate_lookup_tables:
            self._generate_lookup_tables()
        
        # Save summary CSV
        csv_file = self.config.output_dir / "vcf_column_summary.csv"
        summary_data = []
        for col_name, col_doc in self.documentation.items():
            summary_data.append({
                'column': col_name,
                'description': col_doc.description,
                'data_type': col_doc.data_type,
                'source': col_doc.source,
                'is_required': col_doc.is_required,
                'unique_count': col_doc.unique_count,
                'null_count': col_doc.null_count,
                'total_values': len(col_doc.value_counts)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_file, index=False)
        self.logger.info(f"✅ Saved summary CSV: {csv_file}")
    
    def _generate_lookup_tables(self):
        """Generate separate lookup tables for columns with many distinct values."""
        lookup_dir = self.config.output_dir / "lookup_tables"
        lookup_dir.mkdir(exist_ok=True)
        
        lookup_count = 0
        for col_name, col_doc in self.documentation.items():
            # Generate lookup table for columns with many values
            if col_doc.unique_count > self.config.max_categorical_values:
                lookup_file = lookup_dir / f"{col_name}_lookup.csv"
                
                # Create lookup table
                lookup_data = []
                for value, count in col_doc.value_counts.items():
                    percentage = (count / sum(col_doc.value_counts.values())) * 100
                    lookup_entry = {
                        'value': value,
                        'count': count,
                        'percentage': f"{percentage:.3f}%"
                    }
                    
                    # Add meaning if available
                    value_meanings = self.CLINVAR_VALUE_MEANINGS.get(col_name, {})
                    if str(value) in value_meanings:
                        lookup_entry['meaning'] = value_meanings[str(value)]
                    
                    lookup_data.append(lookup_entry)
                
                # Sort by count (descending)
                lookup_data.sort(key=lambda x: x['count'], reverse=True)
                
                # Save lookup table
                lookup_df = pd.DataFrame(lookup_data)
                lookup_df.to_csv(lookup_file, index=False)
                lookup_count += 1
                self.logger.info(f"  Generated lookup table: {lookup_file}")
        
        if lookup_count > 0:
            self.logger.info(f"✅ Generated {lookup_count} lookup tables in: {lookup_dir}")
            
            # Create lookup index
            index_file = lookup_dir / "README.md"
            with open(index_file, 'w') as f:
                f.write("# VCF Column Lookup Tables\n\n")
                f.write("This directory contains detailed lookup tables for columns with many distinct values.\n\n")
                f.write("## Available Lookup Tables\n\n")
                
                for col_name, col_doc in self.documentation.items():
                    if col_doc.unique_count > self.config.max_categorical_values:
                        f.write(f"- **{col_name}**: `{col_name}_lookup.csv` ({col_doc.unique_count:,} unique values)\n")
                        f.write(f"  - {col_doc.description}\n")
                        f.write(f"  - Data type: {col_doc.data_type}\n")
                        f.write(f"  - Source: {col_doc.source}\n\n")
            
            self.logger.info(f"✅ Generated lookup index: {index_file}")


def main():
    """Command-line interface for VCF column documentation."""
    parser = argparse.ArgumentParser(
        description="VCF Column Documentation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset analysis (recommended for accurate statistics)
  python vcf_column_documenter.py --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz --output-dir results/documentation/
  
  # With absolute paths
  python vcf_column_documenter.py --vcf /path/to/clinvar.vcf.gz --output-dir /path/to/docs/
  
  # Quick analysis with smart sampling (faster but less accurate)
  python vcf_column_documenter.py --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz --output-dir results/documentation/ --enable-smart-sampling
  
  # Limited variants for testing
  python vcf_column_documenter.py --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz --output-dir results/documentation/ --max-variants 50000
  
  # JSON only output
  python vcf_column_documenter.py --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz --output-dir results/documentation/ --formats json
        """
    )
    
    parser.add_argument('--vcf', required=True, help='Input VCF file (supports structured paths like data/ensembl/clinvar/vcf/clinvar.vcf.gz)')
    parser.add_argument('--output-dir', required=True, help='Output directory (supports structured paths like data/ensembl/clinvar/vcf/docs/)')
    parser.add_argument('--max-variants', type=int, help='Maximum variants to analyze (if not specified, uses smart sampling)')
    parser.add_argument('--sample-size', type=int, default=10000, help='Sample size for analysis (deprecated, use smart sampling)')
    parser.add_argument('--formats', nargs='+', default=['json', 'markdown'], 
                       choices=['json', 'markdown'], help='Output formats')
    parser.add_argument('--max-categorical-values', type=int, default=100, 
                       help='Maximum categorical values to show in main report (default: 100)')
    parser.add_argument('--max-numerical-examples', type=int, default=10,
                       help='Maximum examples for numerical columns (default: 10)')
    parser.add_argument('--no-lookup-tables', action='store_true',
                       help='Disable generation of separate lookup tables')
    parser.add_argument('--enable-smart-sampling', action='store_true',
                       help='Enable smart sampling (default: process full dataset)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Resolve paths using structured convention
    input_vcf = resolve_structured_path(args.vcf)
    output_dir = resolve_structured_path(args.output_dir)
    
    # Validate input VCF exists
    if not input_vcf.exists():
        print(f"❌ Error: VCF file not found: {input_vcf}")
        print(f"   Resolved from: {args.vcf}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Show path resolution if verbose
    if args.verbose:
        print(f"🔍 Path Resolution:")
        print(f"   Input VCF: {args.vcf} → {input_vcf}")
        print(f"   Output Dir: {args.output_dir} → {output_dir}")
        print()
    
    # Create configuration
    config = VCFDocumentationConfig(
        input_vcf=input_vcf,
        output_dir=output_dir,
        max_variants=args.max_variants,
        sample_size=args.sample_size,
        output_formats=args.formats,
        verbose=args.verbose,
        max_categorical_values=args.max_categorical_values,
        max_numerical_examples=args.max_numerical_examples,
        generate_lookup_tables=not args.no_lookup_tables,
        use_smart_sampling=args.enable_smart_sampling
    )
    
    # Create documenter and run analysis
    documenter = VCFColumnDocumenter(config)
    
    try:
        # Analyze columns
        documentation = documenter.analyze_vcf_columns()
        
        # Save documentation
        documenter.save_documentation()
        
        print(f"\n✅ Successfully documented {len(documentation)} columns")
        print(f"📁 Output saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

