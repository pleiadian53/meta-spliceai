#!/usr/bin/env python3
"""
Complete ClinVar Pipeline: Raw VCF → WT/ALT Ready Data

This script automates the complete pipeline from raw ClinVar VCF download
to data ready for WT/ALT sequence construction and delta score calculations.

Pipeline Steps:
1. Data Preparation: Filter to main chromosomes and validate
2. VCF Normalization: Proper bcftools normalization with multiallelic splitting
3. Universal Parsing: Comprehensive splice variant detection and parsing
4. Sequence Construction: WT/ALT sequence preparation for delta score analysis

Usage:
    # Basic usage
    python complete_clinvar_pipeline.py --input clinvar_20250831.vcf.gz --output results/

    # With custom reference
    python complete_clinvar_pipeline.py \
        --input clinvar_20250831.vcf.gz \
        --output results/ \
        --reference-fasta data/reference/GRCh38.fa

    # Research mode (include all variants)
    python complete_clinvar_pipeline.py \
        --input clinvar_20250831.vcf.gz \
        --output results/ \
        --research-mode \
        --max-variants 50000

Author: MetaSpliceAI Team
Date: 2025-09-12
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import pysam

# Import MetaSpliceAI modules
try:
    from .vcf_preprocessing import (
        VCFPreprocessor, 
        VCFPreprocessingConfig,
        preprocess_clinvar_vcf
    )
except ImportError as e:
    print(f"Warning: Could not import vcf_preprocessing: {e}")
    VCFPreprocessor = None
    VCFPreprocessingConfig = None

try:
    from .universal_vcf_parser import (
        UniversalVCFParser,
        VCFParsingConfig,
        AnnotationSystem,
        SpliceDetectionMode,
        create_clinvar_parser
    )
except ImportError as e:
    print(f"Warning: Could not import universal_vcf_parser: {e}")
    UniversalVCFParser = None

try:
    from ..meta_models.workflows.inference.data_resource_manager import InferenceDataResourceManager
except ImportError:
    # Fallback implementation
    class InferenceDataResourceManager:
        def get_fasta_file(self):
            raise FileNotFoundError("Reference FASTA not found via resource manager")


@dataclass
class CompletePipelineConfig:
    """Configuration for complete ClinVar pipeline."""
    
    # Input/output paths
    input_vcf: Path
    output_dir: Path
    reference_fasta: Optional[Path] = None
    
    # Data preparation options
    filter_chromosomes: bool = True
    main_chromosomes: List[str] = None
    validate_input: bool = True
    
    # Normalization options
    split_multiallelics: bool = True
    left_align: bool = True
    trim_alleles: bool = True
    
    # Parsing options
    splice_detection_mode: str = "comprehensive"
    include_sequences: bool = True
    sequence_context: int = 50
    
    # Filtering options
    research_mode: bool = False  # Include all variants vs. splice-focused
    pathogenic_only: bool = False
    min_quality: Optional[float] = None
    
    # Performance options
    threads: int = 4
    memory_gb: int = 8
    max_variants: Optional[int] = None
    chunk_size: int = 10000
    
    # Output options
    output_formats: List[str] = None  # ['tsv', 'parquet', 'json']
    create_summary: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.main_chromosomes is None:
            self.main_chromosomes = [
                '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT'
            ]
        
        if self.output_formats is None:
            self.output_formats = ['tsv', 'parquet']


class CompleteClinVarPipeline:
    """Complete pipeline from raw ClinVar VCF to WT/ALT ready data."""
    
    def __init__(self, config: CompletePipelineConfig):
        """
        Initialize complete pipeline.
        
        Parameters
        ----------
        config : CompletePipelineConfig
            Pipeline configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Check required modules
        self._check_dependencies()
        
        self.resource_manager = InferenceDataResourceManager()
        
        # Pipeline state
        self.results = {}
        self.intermediate_files = {}
        self.stats = {}
        
        # Resolve reference FASTA if not provided
        if not self.config.reference_fasta:
            self.config.reference_fasta = self._get_reference_fasta()
    
    def _check_dependencies(self):
        """Check that required modules are available."""
        if VCFPreprocessor is None:
            raise ImportError("VCFPreprocessor not available. Please check vcf_preprocessing.py imports.")
        
        if UniversalVCFParser is None:
            raise ImportError("UniversalVCFParser not available. Please check universal_vcf_parser.py imports.")
    
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
    
    def _get_reference_fasta(self) -> Path:
        """Get reference FASTA path from resource manager."""
        try:
            return self.resource_manager.get_fasta_file()
        except Exception as e:
            self.logger.warning(f"Could not get reference FASTA from resource manager: {e}")
            
            # Fallback to common paths
            common_paths = [
                Path("/data/reference/GRCh38.fa"),
                Path("/data/reference/hg38.fa"),
                Path("data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"),
                Path("data/reference/GRCh38.fa")
            ]
            
            for path in common_paths:
                if path.exists():
                    self.logger.info(f"Using reference FASTA: {path}")
                    return path
            
            raise FileNotFoundError(
                "Reference FASTA not found. Please specify --reference-fasta or ensure "
                "reference is available in standard locations."
            )
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Returns
        -------
        Dict[str, Any]
            Pipeline results and statistics
        """
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting Complete ClinVar Pipeline")
            self.logger.info(f"Input VCF: {self.config.input_vcf}")
            self.logger.info(f"Output Directory: {self.config.output_dir}")
            self.logger.info(f"Reference FASTA: {self.config.reference_fasta}")
            
            # Create output directory
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 0: Data Preparation
            prepared_vcf = self.step0_data_preparation()
            
            # Step 1: VCF Normalization
            normalized_vcf = self.step1_vcf_normalization(prepared_vcf)
            
            # Step 2: Universal VCF Parsing
            parsed_variants = self.step2_universal_parsing(normalized_vcf)
            
            # Step 3: WT/ALT Sequence Construction
            wt_alt_ready_data = self.step3_sequence_construction(parsed_variants)
            
            # Step 4: Output Generation
            output_files = self.step4_output_generation(wt_alt_ready_data)
            
            # Generate summary
            if self.config.create_summary:
                summary = self._generate_pipeline_summary(start_time)
                self.results['summary'] = summary
            
            self.logger.info(f"✅ Pipeline completed successfully in {time.time() - start_time:.1f}s")
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def step0_data_preparation(self) -> Path:
        """
        Step 0: Data Preparation - Filter to main chromosomes and validate.
        
        Returns
        -------
        Path
            Path to prepared VCF file
        """
        self.logger.info("=== Step 0: Data Preparation ===")
        
        if not self.config.input_vcf.exists():
            raise FileNotFoundError(f"Input VCF not found: {self.config.input_vcf}")
        
        # Check if input is already filtered
        if self._is_already_filtered():
            self.logger.info("✓ Input VCF appears to be already filtered to main chromosomes")
            prepared_vcf = self.config.input_vcf
        else:
            prepared_vcf = self._filter_to_main_chromosomes()
        
        # Validate prepared VCF
        if self.config.validate_input:
            self._validate_prepared_vcf(prepared_vcf)
        
        self.intermediate_files['prepared_vcf'] = prepared_vcf
        self.logger.info(f"✓ Data preparation completed: {prepared_vcf}")
        return prepared_vcf
    
    def _is_already_filtered(self) -> bool:
        """Check if VCF is already filtered to main chromosomes."""
        try:
            # Quick check of first few records
            with pysam.VariantFile(str(self.config.input_vcf)) as vcf:
                chromosomes = set()
                for i, record in enumerate(vcf):
                    if i > 100:  # Check first 100 records
                        break
                    chromosomes.add(record.chrom)
                
                # Check if we find any non-main chromosomes
                non_main_chroms = chromosomes - set(self.config.main_chromosomes) - set([f"chr{c}" for c in self.config.main_chromosomes])
                
                return len(non_main_chroms) == 0
                
        except Exception as e:
            self.logger.warning(f"Could not check chromosome filtering: {e}")
            return False
    
    def _filter_to_main_chromosomes(self) -> Path:
        """Filter VCF to main chromosomes only."""
        output_vcf = self.config.output_dir / f"{self.config.input_vcf.stem}_main_chroms.vcf.gz"
        
        # Build chromosome list for bcftools (handle chr prefix)
        chrom_list = []
        for chrom in self.config.main_chromosomes:
            chrom_list.extend([chrom, f"chr{chrom}"])
        
        chrom_str = ','.join(chrom_list)
        
        self.logger.info(f"Filtering to main chromosomes: {', '.join(self.config.main_chromosomes)}")
        
        cmd = [
            'bcftools', 'view',
            '-r', chrom_str,
            '-Oz',  # Compressed output
            '-o', str(output_vcf),
            str(self.config.input_vcf)
        ]
        
        self.logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Chromosome filtering failed: {result.stderr}")
        
        # Index the filtered VCF
        subprocess.run(['bcftools', 'index', str(output_vcf)], check=True)
        
        self.logger.info(f"✓ Filtered VCF created: {output_vcf}")
        return output_vcf
    
    def _validate_prepared_vcf(self, vcf_path: Path):
        """Validate prepared VCF file."""
        try:
            with pysam.VariantFile(str(vcf_path)) as vcf:
                # Count variants
                variant_count = sum(1 for _ in vcf)
                
                # Reset and check chromosomes
                vcf.close()
                with pysam.VariantFile(str(vcf_path)) as vcf:
                    chromosomes = set()
                    for record in vcf:
                        chromosomes.add(record.chrom)
                        if len(chromosomes) > 50:  # Reasonable limit
                            break
                
                self.stats['prepared_variants'] = variant_count
                self.stats['prepared_chromosomes'] = sorted(list(chromosomes))
                
                self.logger.info(f"✓ Validation passed: {variant_count:,} variants across {len(chromosomes)} chromosomes")
                
        except Exception as e:
            self.logger.error(f"VCF validation failed: {e}")
            raise
    
    def step1_vcf_normalization(self, input_vcf: Path) -> Path:
        """
        Step 1: VCF Normalization using VCFPreprocessor.
        
        Parameters
        ----------
        input_vcf : Path
            Input VCF file to normalize
            
        Returns
        -------
        Path
            Path to normalized VCF file
        """
        self.logger.info("=== Step 1: VCF Normalization ===")
        
        output_vcf = self.config.output_dir / f"{input_vcf.stem}_normalized.vcf.gz"
        
        # Check if normalized VCF already exists and is valid
        if output_vcf.exists() and output_vcf.with_suffix('.vcf.gz.tbi').exists():
            self.logger.info(f"✓ Normalized VCF already exists: {output_vcf}")
            
            # Quick validation to ensure file integrity
            try:
                # Check if file is readable and has variants
                result = subprocess.run(['bcftools', 'view', '-H', str(output_vcf)], 
                                     capture_output=True, text=True, check=True)
                variant_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                
                if variant_count > 0:
                    self.logger.info(f"✓ Existing normalized VCF is valid ({variant_count} variants)")
                    self.intermediate_files['normalized_vcf'] = output_vcf
                    return output_vcf
                else:
                    self.logger.warning("Existing normalized VCF appears empty, re-normalizing...")
            except subprocess.CalledProcessError:
                self.logger.warning("Existing normalized VCF appears corrupted, re-normalizing...")
        
        # Create preprocessing configuration
        preprocessing_config = VCFPreprocessingConfig(
            input_vcf=input_vcf,
            output_vcf=output_vcf,
            reference_fasta=self.config.reference_fasta,
            split_multiallelics=self.config.split_multiallelics,
            left_align=self.config.left_align,
            trim_alleles=self.config.trim_alleles,
            threads=self.config.threads,
            create_index=True,
            validate_output=True
        )
        
        # Run normalization
        preprocessor = VCFPreprocessor(preprocessing_config)
        normalized_vcf = preprocessor.normalize_vcf()
        
        # Get normalization statistics
        norm_stats = preprocessor.get_normalization_stats()
        self.stats['normalization'] = norm_stats
        
        self.intermediate_files['normalized_vcf'] = normalized_vcf
        self.logger.info(f"✓ VCF normalization completed: {normalized_vcf}")
        return normalized_vcf
    
    def step2_universal_parsing(self, normalized_vcf: Path) -> pd.DataFrame:
        """
        Step 2: Universal VCF Parsing with comprehensive splice detection.
        
        Parameters
        ----------
        normalized_vcf : Path
            Normalized VCF file
            
        Returns
        -------
        pd.DataFrame
            Parsed variants with comprehensive annotations
        """
        self.logger.info("=== Step 2: Universal VCF Parsing ===")
        
        # Create parsing configuration
        parsing_config = VCFParsingConfig(
            input_vcf=normalized_vcf,
            output_dir=self.config.output_dir / "parsing",
            output_format="tsv",  # We'll handle multiple formats later
            annotation_system=AnnotationSystem.CLINVAR,
            splice_detection_mode=SpliceDetectionMode(self.config.splice_detection_mode),
            include_sequences=self.config.include_sequences,
            sequence_context=self.config.sequence_context,
            reference_fasta=self.config.reference_fasta if self.config.include_sequences else None,
            max_variants=self.config.max_variants,
            chunk_size=self.config.chunk_size,
            pathogenicity_filter=None if self.config.research_mode else ['Pathogenic', 'Likely_pathogenic'] if self.config.pathogenic_only else None,
            min_quality_score=self.config.min_quality
        )
        
        # Create and run parser
        parser = UniversalVCFParser(parsing_config)
        parsed_variants = parser.parse_vcf(normalized_vcf)
        
        # Store parsing statistics
        self.stats['parsing'] = {
            'total_variants': len(parsed_variants),
            'splice_affecting_variants': len(parsed_variants[parsed_variants['is_splice_affecting']]) if 'is_splice_affecting' in parsed_variants.columns else 0,
            'pathogenic_variants': len(parsed_variants[parsed_variants['is_pathogenic']]) if 'is_pathogenic' in parsed_variants.columns else 0,
            'columns': list(parsed_variants.columns)
        }
        
        self.logger.info(f"✓ Universal parsing completed: {len(parsed_variants):,} variants parsed")
        self.logger.info(f"  - Splice-affecting: {self.stats['parsing']['splice_affecting_variants']:,}")
        
        return parsed_variants
    
    def step3_sequence_construction(self, parsed_variants: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: WT/ALT Sequence Construction for delta score analysis.
        
        Parameters
        ----------
        parsed_variants : pd.DataFrame
            Parsed variants from Step 2
            
        Returns
        -------
        pd.DataFrame
            Variants with WT/ALT sequences ready for analysis
        """
        self.logger.info("=== Step 3: WT/ALT Sequence Construction ===")
        
        if not self.config.include_sequences:
            self.logger.info("⚠️  Sequence construction skipped (include_sequences=False)")
            return parsed_variants
        
        # Check if sequences are already included
        if 'ref_sequence' in parsed_variants.columns and 'alt_sequence' in parsed_variants.columns:
            self.logger.info("✓ WT/ALT sequences already present in parsed data")
            
            # Validate sequences
            valid_sequences = (
                parsed_variants['ref_sequence'].notna() & 
                parsed_variants['alt_sequence'].notna() &
                (parsed_variants['ref_sequence'] != '') &
                (parsed_variants['alt_sequence'] != '')
            )
            
            self.stats['sequence_construction'] = {
                'total_variants': len(parsed_variants),
                'variants_with_sequences': valid_sequences.sum(),
                'sequence_success_rate': valid_sequences.mean()
            }
            
            self.logger.info(f"✓ Sequence validation: {valid_sequences.sum():,}/{len(parsed_variants):,} variants have valid sequences")
            
        else:
            self.logger.warning("⚠️  No sequences found in parsed data. Ensure include_sequences=True and reference FASTA is provided.")
            self.stats['sequence_construction'] = {
                'total_variants': len(parsed_variants),
                'variants_with_sequences': 0,
                'sequence_success_rate': 0.0
            }
        
        return parsed_variants
    
    def step4_output_generation(self, final_data: pd.DataFrame) -> Dict[str, Path]:
        """
        Step 4: Generate output files in requested formats.
        
        Parameters
        ----------
        final_data : pd.DataFrame
            Final processed data
            
        Returns
        -------
        Dict[str, Path]
            Dictionary mapping format names to output file paths
        """
        self.logger.info("=== Step 4: Output Generation ===")
        
        output_files = {}
        
        for format_name in self.config.output_formats:
            if format_name == 'tsv':
                output_path = self.config.output_dir / "clinvar_wt_alt_ready.tsv"
                final_data.to_csv(output_path, sep='\t', index=False)
                
            elif format_name == 'parquet':
                output_path = self.config.output_dir / "clinvar_wt_alt_ready.parquet"
                final_data.to_parquet(output_path, index=False)
                
            elif format_name == 'json':
                output_path = self.config.output_dir / "clinvar_wt_alt_ready.json"
                final_data.to_json(output_path, orient='records', lines=True)
                
            else:
                self.logger.warning(f"Unknown output format: {format_name}")
                continue
            
            output_files[format_name] = output_path
            self.logger.info(f"✓ Generated {format_name.upper()}: {output_path}")
        
        self.results['output_files'] = output_files
        return output_files
    
    def _generate_pipeline_summary(self, start_time: float) -> Dict[str, Any]:
        """Generate comprehensive pipeline summary."""
        runtime = time.time() - start_time
        
        summary = {
            'pipeline_info': {
                'input_vcf': str(self.config.input_vcf),
                'output_directory': str(self.config.output_dir),
                'reference_fasta': str(self.config.reference_fasta),
                'runtime_seconds': runtime,
                'runtime_formatted': f"{runtime:.1f}s"
            },
            'configuration': {
                'research_mode': self.config.research_mode,
                'splice_detection_mode': self.config.splice_detection_mode,
                'include_sequences': self.config.include_sequences,
                'sequence_context': self.config.sequence_context,
                'max_variants': self.config.max_variants,
                'output_formats': self.config.output_formats
            },
            'statistics': self.stats,
            'intermediate_files': {k: str(v) for k, v in self.intermediate_files.items()},
            'output_files': {k: str(v) for k, v in self.results.get('output_files', {}).items()}
        }
        
        # Write summary to file
        summary_path = self.config.output_dir / "pipeline_summary.json"
        import json
        
        # Convert numpy/pandas types to JSON serializable types
        def convert_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'to_dict'):  # pandas Series/DataFrame
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        json_serializable_summary = convert_types(summary)
        
        with open(summary_path, 'w') as f:
            json.dump(json_serializable_summary, f, indent=2)
        
        self.logger.info(f"✓ Pipeline summary written to: {summary_path}")
        
        # Print summary to console
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print pipeline summary to console."""
        print("\n" + "="*60)
        print("🎉 COMPLETE CLINVAR PIPELINE SUMMARY")
        print("="*60)
        
        print(f"📁 Input:  {summary['pipeline_info']['input_vcf']}")
        print(f"📁 Output: {summary['pipeline_info']['output_directory']}")
        print(f"⏱️  Runtime: {summary['pipeline_info']['runtime_formatted']}")
        
        if 'parsing' in summary['statistics']:
            stats = summary['statistics']['parsing']
            print(f"\n📊 Results:")
            print(f"   • Total variants processed: {stats['total_variants']:,}")
            print(f"   • Splice-affecting variants: {stats['splice_affecting_variants']:,}")
            if stats['pathogenic_variants'] > 0:
                print(f"   • Pathogenic variants: {stats['pathogenic_variants']:,}")
        
        if 'sequence_construction' in summary['statistics']:
            seq_stats = summary['statistics']['sequence_construction']
            print(f"   • Variants with WT/ALT sequences: {seq_stats['variants_with_sequences']:,}")
            print(f"   • Sequence success rate: {seq_stats['sequence_success_rate']:.1%}")
        
        print(f"\n📄 Output files:")
        for format_name, file_path in summary['output_files'].items():
            print(f"   • {format_name.upper()}: {file_path}")
        
        print("\n✅ Pipeline completed successfully!")
        print("="*60)


def create_default_config(input_vcf: str, output_dir: str, **kwargs) -> CompletePipelineConfig:
    """Create default pipeline configuration."""
    return CompletePipelineConfig(
        input_vcf=Path(input_vcf),
        output_dir=Path(output_dir),
        **kwargs
    )


def main():
    """Command-line interface for complete ClinVar pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete ClinVar Pipeline: Raw VCF → WT/ALT Ready Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python complete_clinvar_pipeline.py --input clinvar_20250831.vcf.gz --output results/

  # With custom reference
  python complete_clinvar_pipeline.py \\
      --input clinvar_20250831.vcf.gz \\
      --output results/ \\
      --reference-fasta data/reference/GRCh38.fa

  # Research mode with all variants
  python complete_clinvar_pipeline.py \\
      --input clinvar_20250831.vcf.gz \\
      --output results/ \\
      --research-mode \\
      --max-variants 50000
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input ClinVar VCF file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory')
    
    # Optional arguments
    parser.add_argument('--reference-fasta', '-r',
                       help='Reference FASTA file (auto-detected if not provided)')
    
    # Processing options
    parser.add_argument('--research-mode', action='store_true',
                       help='Include all variants (not just splice-affecting)')
    parser.add_argument('--pathogenic-only', action='store_true',
                       help='Include only pathogenic/likely pathogenic variants')
    parser.add_argument('--splice-detection', choices=['strict', 'comprehensive', 'permissive'],
                       default='comprehensive', help='Splice detection mode')
    
    # Sequence options
    parser.add_argument('--no-sequences', action='store_true',
                       help='Skip WT/ALT sequence construction')
    parser.add_argument('--sequence-context', type=int, default=50,
                       help='Sequence context size (default: 50)')
    
    # Performance options
    parser.add_argument('--max-variants', type=int,
                       help='Maximum variants to process (for testing)')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of threads (default: 4)')
    parser.add_argument('--memory', type=int, default=8,
                       help='Memory limit in GB (default: 8)')
    
    # Output options
    parser.add_argument('--output-formats', nargs='+', 
                       choices=['tsv', 'parquet', 'json'],
                       default=['tsv', 'parquet'],
                       help='Output formats (default: tsv parquet)')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip pipeline summary generation')
    
    # Filtering options
    parser.add_argument('--no-chromosome-filter', action='store_true',
                       help='Skip filtering to main chromosomes')
    parser.add_argument('--min-quality', type=float,
                       help='Minimum variant quality score')
    
    # Misc options
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Create configuration
    config = CompletePipelineConfig(
        input_vcf=Path(args.input),
        output_dir=Path(args.output),
        reference_fasta=Path(args.reference_fasta) if args.reference_fasta else None,
        filter_chromosomes=not args.no_chromosome_filter,
        splice_detection_mode=args.splice_detection,
        include_sequences=not args.no_sequences,
        sequence_context=args.sequence_context,
        research_mode=args.research_mode,
        pathogenic_only=args.pathogenic_only,
        min_quality=args.min_quality,
        threads=args.threads,
        memory_gb=args.memory,
        max_variants=args.max_variants,
        output_formats=args.output_formats,
        create_summary=not args.no_summary,
        verbose=not args.quiet
    )
    
    try:
        # Run pipeline
        pipeline = CompleteClinVarPipeline(config)
        results = pipeline.run_complete_pipeline()
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
