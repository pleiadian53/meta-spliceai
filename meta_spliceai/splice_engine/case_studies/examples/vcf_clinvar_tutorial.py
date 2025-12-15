#!/usr/bin/env python3
"""
Tutorial: Parsing Real ClinVar VCF Data for Splice Site Analysis

This tutorial demonstrates how to:
1. Set up the ClinVar dataset directory structure
2. Download and parse real ClinVar VCF files
3. Extract pathogenic variants affecting splicing
4. Standardize variants and construct WT/ALT sequences
5. Prepare data for OpenSpliceAI delta score computation

Requirements:
- Download ClinVar VCF from: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/
  Example: clinvar_20250101.vcf.gz (use latest available)
"""

import os
import sys
import gzip
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from case_studies.formats.variant_standardizer import VariantStandardizer
from case_studies.analysis.splicing_pattern_analyzer import SplicingPatternAnalyzer, SpliceSite
from case_studies.filters.splice_variant_filter import create_clinvar_filter
from case_studies.workflows.vcf_preprocessing import preprocess_clinvar_vcf, VCFPreprocessingConfig, VCFPreprocessor
from case_studies.data_sources.resource_manager import create_case_study_resource_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_clinvar_directory() -> Path:
    """
    Set up the ClinVar dataset directory structure using case study resource manager.
    
    Returns
    -------
    Path
        Path to the ClinVar data directory
    """
    # Create case study resource manager
    resource_manager = create_case_study_resource_manager()
    
    # Get ClinVar data directory
    clinvar_dir = resource_manager.get_clinvar_dir()
    
    logger.info(f"ClinVar directory set up at: {clinvar_dir}")
    
    subdirs = ["vcf", "processed", "splice_variants", "logs"]
    for subdir in subdirs:
        (clinvar_dir / subdir).mkdir(exist_ok=True)
    
    logger.info(f"ClinVar directory structure created at: {clinvar_dir}")
    return clinvar_dir


def parse_clinvar_vcf(
    vcf_path: str,
    output_dir: Optional[str] = None,
    max_variants: Optional[int] = None
) -> pd.DataFrame:
    """
    Parse ClinVar VCF file and extract relevant variant information.
    
    Parameters
    ----------
    vcf_path : str
        Path to ClinVar VCF file (can be gzipped)
    output_dir : Optional[str]
        Directory to save processed data
    max_variants : Optional[int]
        Maximum number of variants to process (for testing)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with parsed variant information
    """
    variants = []
    
    # Determine if file is gzipped
    open_func = gzip.open if vcf_path.endswith('.gz') else open
    mode = 'rt' if vcf_path.endswith('.gz') else 'r'
    
    with open_func(vcf_path, mode) as f:
        for line_num, line in enumerate(f):
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 8:
                continue
            
            chrom = fields[0]
            pos = int(fields[1])
            var_id = fields[2]
            ref = fields[3]
            alt = fields[4]
            qual = fields[5]
            filter_status = fields[6]
            info = fields[7]
            
            # Parse INFO field
            info_dict = {}
            for item in info.split(';'):
                if '=' in item:
                    key, value = item.split('=', 1)
                    info_dict[key] = value
            
            # Extract clinical significance
            clnsig = info_dict.get('CLNSIG', '')
            clndn = info_dict.get('CLNDN', '')
            clnrevstat = info_dict.get('CLNREVSTAT', '')
            mc = info_dict.get('MC', '')  # Molecular consequence
            
            # Check if variant affects splicing
            affects_splicing = False
            if mc:
                splice_terms = ['splice', 'intron', 'exon']
                affects_splicing = any(term in mc.lower() for term in splice_terms)
            
            # Check for pathogenic/likely pathogenic variants
            is_pathogenic = False
            if clnsig:
                path_terms = ['pathogenic', 'likely_pathogenic']
                is_pathogenic = any(term in clnsig.lower().replace(' ', '_') for term in path_terms)
            
            variant_data = {
                'chrom': chrom,
                'pos': pos,
                'var_id': var_id,
                'ref': ref,
                'alt': alt,
                'qual': qual,
                'filter': filter_status,
                'clnsig': clnsig,
                'clndn': clndn,
                'clnrevstat': clnrevstat,
                'mc': mc,
                'affects_splicing': affects_splicing,
                'is_pathogenic': is_pathogenic
            }
            
            variants.append(variant_data)
            
            if max_variants and len(variants) >= max_variants:
                logger.info(f"Reached max_variants limit: {max_variants}")
                break
    
    df = pd.DataFrame(variants)
    logger.info(f"Parsed {len(df)} variants from VCF")
    
    # Save processed data if output directory specified
    if output_dir:
        output_path = Path(output_dir) / "clinvar_variants_all.tsv"
        df.to_csv(output_path, sep='\t', index=False)
        logger.info(f"Saved all variants to: {output_path}")
        
        # Save splice-affecting pathogenic variants
        splice_df = df[(df['affects_splicing']) & (df['is_pathogenic'])]
        if not splice_df.empty:
            splice_path = Path(output_dir) / "clinvar_splice_pathogenic.tsv"
            splice_df.to_csv(splice_path, sep='\t', index=False)
            logger.info(f"Saved {len(splice_df)} splice-affecting pathogenic variants to: {splice_path}")
    
    return df


def filter_splice_variants(df: pd.DataFrame, use_advanced_filter: bool = True) -> pd.DataFrame:
    """
    Filter variants for those affecting splicing using advanced filtering logic.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed ClinVar variants
    use_advanced_filter : bool
        Whether to use the advanced splice variant filter
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with splice-affecting variants
    """
    if use_advanced_filter:
        # Use the comprehensive splice variant filter
        splice_filter = create_clinvar_filter(
            pathogenicity_threshold="likely_pathogenic",
            include_uncertain=False,
            require_review_status=False,  # Many ClinVar entries lack detailed review status
            enable_frequency_filter=False  # VCF may not have frequency data
        )
        
        # Map column names for the filter
        df_mapped = df.copy()
        column_mapping = {
            'clnsig': 'clinical_significance',
            'mc': 'molecular_consequence',
            'clnrevstat': 'review_status'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df_mapped.columns and new_col not in df_mapped.columns:
                df_mapped[new_col] = df_mapped[old_col]
        
        try:
            splice_df, filter_stats = splice_filter.filter_variants(
                df_mapped, 
                source='clinvar', 
                return_stats=True
            )
            
            logger.info("Advanced filtering results:")
            logger.info(f"  Input variants: {filter_stats['input_count']}")
            logger.info(f"  Splice-affecting: {filter_stats['splice_affecting']}")
            logger.info(f"  Pathogenic: {filter_stats['pathogenic']}")
            logger.info(f"  Final filtered: {filter_stats['final_count']}")
            
        except Exception as e:
            logger.warning(f"Advanced filtering failed: {e}")
            logger.info("Falling back to simple filtering...")
            use_advanced_filter = False
    
    if not use_advanced_filter:
        # Fallback to simple filtering logic
        splice_df = df[(df['affects_splicing']) & (df['is_pathogenic'])].copy()
        logger.info(f"Simple filtering: {len(splice_df)} pathogenic splice-affecting variants")
    
    # Add variant type classification
    if not splice_df.empty:
        splice_df = splice_df.copy()  # Avoid SettingWithCopyWarning
        splice_df['var_type'] = splice_df.apply(
            lambda row: classify_variant_type(row['ref'], row['alt']), 
            axis=1
        )
    
    # Summary statistics
    if not splice_df.empty and 'var_type' in splice_df.columns:
        logger.info("\nVariant type distribution:")
        for var_type, count in splice_df['var_type'].value_counts().items():
            logger.info(f"  {var_type}: {count}")
    else:
        logger.info("No pathogenic splice-affecting variants found in this sample")
    
    return splice_df


def classify_variant_type(ref: str, alt: str) -> str:
    """
    Classify variant type based on ref and alt alleles.
    
    Parameters
    ----------
    ref : str
        Reference allele
    alt : str
        Alternative allele
    
    Returns
    -------
    str
        Variant type classification
    """
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    elif len(ref) > len(alt):
        return "Deletion"
    elif len(ref) < len(alt):
        return "Insertion"
    else:
        return "Complex"


def process_splice_variants_with_standardizer(
    splice_df: pd.DataFrame,
    standardizer: Optional[VariantStandardizer] = None
) -> List[Dict]:
    """
    Process splice variants using the VariantStandardizer.
    
    Parameters
    ----------
    splice_df : pd.DataFrame
        DataFrame with splice-affecting variants
    standardizer : Optional[VariantStandardizer]
        Variant standardizer instance
    
    Returns
    -------
    List[Dict]
        List of standardized variant records
    """
    if standardizer is None:
        standardizer = VariantStandardizer()
    
    standardized_variants = []
    
    for _, row in splice_df.iterrows():
        try:
            # Standardize variant
            std_var = standardizer.standardize_from_vcf(
                chrom=row['chrom'],
                pos=row['pos'],
                ref=row['ref'],
                alt=row['alt']
            )
            
            # Add clinical annotations
            std_var_dict = {
                'chrom': std_var.chrom,
                'start': std_var.start,
                'end': std_var.end,
                'ref': std_var.ref,
                'alt': std_var.alt,
                'var_type': std_var.variant_type,  # Use the correct attribute name
                'is_normalized': std_var.is_normalized,
                'coordinate_system': std_var.coordinate_system,
                'reference_genome': std_var.reference_genome,
                'clinvar_id': row['var_id'],
                'clinical_significance': row['clnsig'],
                'disease': row['clndn'],
                'molecular_consequence': row['mc']
            }
            
            standardized_variants.append(std_var_dict)
            
        except Exception as e:
            logger.warning(f"Error standardizing variant {row['var_id']}: {e}")
            continue
    
    logger.info(f"Successfully standardized {len(standardized_variants)} variants")
    return standardized_variants


def load_reference_sequence(fasta_path: str, chrom: str, start: int, end: int) -> str:
    """
    Load reference sequence from FASTA file (placeholder for actual implementation).
    
    Parameters
    ----------
    fasta_path : str
        Path to reference FASTA file
    chrom : str
        Chromosome
    start : int
        Start position (0-based)
    end : int
        End position (0-based)
    
    Returns
    -------
    str
        Reference sequence
    """
    # This is a placeholder - actual implementation would use pysam or similar
    # For now, return a mock sequence
    length = end - start
    mock_seq = "ACGT" * (length // 4 + 1)
    return mock_seq[:length]


def construct_wt_alt_sequences(
    standardized_variants: List[Dict],
    context_size: int = 200,
    fasta_path: Optional[str] = None
) -> List[Dict]:
    """
    Construct wildtype and alternative sequences for variants.
    
    Parameters
    ----------
    standardized_variants : List[Dict]
        List of standardized variant records
    context_size : int
        Number of bases to include on each side of variant
    fasta_path : Optional[str]
        Path to reference FASTA file
    
    Returns
    -------
    List[Dict]
        Variant records with WT and ALT sequences
    """
    sequences = []
    
    for var in standardized_variants:
        # Get genomic context
        seq_start = max(0, var['start'] - context_size)
        seq_end = var['end'] + context_size
        
        if fasta_path and os.path.exists(fasta_path):
            # Load actual reference sequence
            ref_seq = load_reference_sequence(
                fasta_path, var['chrom'], seq_start, seq_end
            )
        else:
            # Use mock sequence for demonstration
            ref_seq = "N" * (seq_end - seq_start)
            logger.debug("Using mock sequence (no FASTA provided)")
        
        # Construct WT sequence
        wt_seq = ref_seq
        
        # Construct ALT sequence
        var_offset = var['start'] - seq_start
        alt_seq = (
            ref_seq[:var_offset] +
            var['alt'] +
            ref_seq[var_offset + len(var['ref']):]
        )
        
        var['wt_sequence'] = wt_seq
        var['alt_sequence'] = alt_seq
        var['seq_start'] = seq_start
        var['seq_end'] = seq_end
        var['var_position_in_seq'] = var_offset
        
        sequences.append(var)
    
    logger.info(f"Constructed WT/ALT sequences for {len(sequences)} variants")
    return sequences


def save_results(
    variants_with_sequences: List[Dict],
    output_dir: str,
    format: str = 'tsv'
) -> None:
    """
    Save processed variants with sequences to file.
    
    Parameters
    ----------
    variants_with_sequences : List[Dict]
        Variant records with sequences
    output_dir : str
        Output directory
    format : str
        Output format ('tsv' or 'parquet')
    """
    df = pd.DataFrame(variants_with_sequences)
    
    if df.empty:
        logger.warning("No variants to save - dataframe is empty")
        return
    
    output_path = Path(output_dir) / f"clinvar_splice_variants_processed.{format}"
    
    if format == 'tsv':
        df.to_csv(output_path, sep='\t', index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved {len(df)} processed variants to: {output_path}")
    
    # Save summary statistics
    stats_path = Path(output_dir) / "processing_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Total variants processed: {len(df)}\n")
        if 'var_type' in df.columns:
            f.write(f"Variant types:\n")
            for var_type, count in df['var_type'].value_counts().items():
                f.write(f"  {var_type}: {count}\n")
        if 'chrom' in df.columns:
            f.write(f"\nChromosomes represented:\n")
            for chrom in sorted(df['chrom'].unique()):
                f.write(f"  {chrom}: {len(df[df['chrom'] == chrom])}\n")
    
    logger.info(f"Saved processing statistics to: {stats_path}")


def main():
    """Main function to run the ClinVar VCF processing tutorial."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ClinVar VCF Processing Tutorial for Splice Site Analysis'
    )
    parser.add_argument(
        '--vcf-file',
        type=str,
        help='Specific VCF file to process (optional - will auto-detect if not provided)'
    )
    parser.add_argument(
        '--max-variants',
        type=int,
        default=1000,
        help='Maximum number of variants to process (default: 1000, use 0 for all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Base output directory (default: data/ensembl/clinvar)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ClinVar VCF Processing Tutorial for Splice Site Analysis")
    print("=" * 80)
    
    # Step 1: Set up directory structure using genomic resource manager
    print("\n1. Setting up ClinVar directory structure...")
    clinvar_dir = setup_clinvar_directory()
    print(f"   ClinVar directory: {clinvar_dir}")
    
    # Step 2: Find or create VCF file
    print("\n2. Locating ClinVar VCF file...")
    if vcf_file:
        vcf_path = Path(vcf_file)
        if not vcf_path.exists():
            raise FileNotFoundError(f"Specified VCF file not found: {vcf_file}")
    else:
        # Look for existing VCF files
        vcf_files = list(clinvar_dir.glob("*.vcf*"))
        if vcf_files:
            vcf_path = vcf_files[0]
            print(f"   Found existing VCF: {vcf_path}")
        else:
            # Create sample VCF for demonstration
            vcf_path = clinvar_dir / "sample_clinvar.vcf"
            print(f"   Creating sample VCF: {vcf_path}")
            create_sample_clinvar_vcf(vcf_path)
    
    print(f"   Using VCF file: {vcf_path}")
    
    # Step 2.5: VCF Normalization (NEW)
    if normalize_vcf:
        print("\n2.5. Normalizing VCF with bcftools...")
        try:
            normalized_vcf = preprocess_clinvar_vcf(
                input_vcf=vcf_path,
                output_dir=clinvar_dir / "normalized"
            )
            print(f"   VCF normalized: {normalized_vcf}")
            vcf_path = normalized_vcf  # Use normalized VCF for downstream processing
            
            # Get normalization stats
            config = VCFPreprocessingConfig(
                input_vcf=vcf_path,
                output_vcf=normalized_vcf
            )
            preprocessor = VCFPreprocessor(config)
            stats = preprocessor.get_normalization_stats()
            
            if 'number of records' in stats:
                print(f"   Normalized VCF contains {stats['number of records']} variants")
            if 'multiallelic_sites_remaining' in stats:
                print(f"   Multiallelic sites remaining: {stats['multiallelic_sites_remaining']}")
                
        except Exception as e:
            print(f"   Warning: VCF normalization failed ({e})")
            print(f"   Continuing with original VCF: {vcf_path}")
    
    # Step 3: Parse VCF file
    print("\n3. Parsing VCF file...")
    raw_variants = parse_clinvar_vcf(vcf_path, max_variants=max_variants)
    print(f"   Parsed {len(raw_variants)} variants")
    
    # Step 4: Filter for splice-affecting variants
    print("\n4. Filtering for splice-affecting pathogenic variants...")
    
    # Use the comprehensive splice variant filter
    try:
        splice_filter = create_clinvar_filter()
        splice_variants = splice_filter.filter_variants(pd.DataFrame(raw_variants))
        splice_variants = splice_variants.to_dict('records')
        print(f"   Found {len(splice_variants)} splice-affecting variants using comprehensive filter")
    except Exception as e:
        print(f"   Warning: Comprehensive filter failed ({e}), using fallback")
        splice_variants = filter_splice_variants(pd.DataFrame(raw_variants))
        splice_variants = splice_variants.to_dict('records')
        print(f"   Found {len(splice_variants)} splice-affecting variants using fallback filter")
    
    if not splice_variants:
        print("   No splice-affecting variants found. Creating sample data for demonstration.")
        splice_variants = raw_variants[:5]  # Use first 5 variants as examples
    
    # Step 5: Standardize variants
    print("\n5. Standardizing variant representations...")
    standardizer = VariantStandardizer()
    standardized_variants = []
    
    for variant in splice_variants:
        try:
            std_variant = standardizer.standardize_from_vcf(
                variant['chrom'], variant['pos'], variant['ref'], variant['alt']
            )
            
            # Add original metadata
            variant_dict = {
                'chrom': std_variant.chrom,
                'start': std_variant.start,
                'end': std_variant.end,
                'ref': std_variant.ref,
                'alt': std_variant.alt,
                'var_type': std_variant.variant_type,
                'clinvar_id': variant.get('id', ''),
                'clinical_significance': variant.get('clinical_significance', ''),
                'disease': variant.get('disease', ''),
                'molecular_consequence': variant.get('molecular_consequence', ''),
                'review_status': variant.get('review_status', ''),
                'normalized': normalize_vcf  # Track if VCF was normalized
            }
            standardized_variants.append(variant_dict)
            
        except Exception as e:
            print(f"   Warning: Failed to standardize variant {variant.get('id', 'unknown')}: {e}")
    
    print(f"   Standardized {len(standardized_variants)} variants")
    
    # Step 6: Construct WT/ALT sequences
    print("\n6. Constructing WT/ALT sequences...")
    
    # Get reference FASTA using resource manager
    resource_manager = create_case_study_resource_manager()
    try:
        fasta_path = resource_manager.get_fasta_path(validate=True)
        print(f"   Using reference FASTA: {fasta_path}")
    except FileNotFoundError:
        print(f"   Reference FASTA not found via resource manager")
        print("   Using mock sequences for demonstration")
        fasta_path = None
    
    variants_with_sequences = construct_wt_alt_sequences(
        standardized_variants,
        context_size=200,
        fasta_path=fasta_path
    )
    
    # Step 7: Save results
    print("\n7. Saving processed results...")
    save_results(
        variants_with_sequences,
        output_dir=str(clinvar_dir / "splice_variants"),
        format='tsv'
    )
    
    # Step 8: Display sample results
    print("\n8. Sample processed variants:")
    print("-" * 80)
    
    if variants_with_sequences:
        for i, var in enumerate(variants_with_sequences[:3], 1):
            print(f"\nVariant {i}:")
            print(f"  ClinVar ID: {var.get('clinvar_id', 'N/A')}")
            print(f"  Location: {var['chrom']}:{var['start']}-{var['end']}")
            print(f"  Type: {var.get('var_type', 'N/A')}")
            print(f"  Ref/Alt: {var['ref']}/{var['alt']}")
            print(f"  Clinical Significance: {var.get('clinical_significance', 'N/A')}")
            print(f"  Disease: {var.get('disease', 'N/A')[:50]}...")  # Truncate long disease names
            print(f"  VCF Normalized: {var.get('normalized', False)}")
            
            if var.get('wt_sequence'):
                print(f"  WT sequence length: {len(var['wt_sequence'])}")
                print(f"  ALT sequence length: {len(var['alt_sequence'])}")
    else:
        print("\nNo variants were successfully processed.")
        print("This may be due to issues with variant standardization.")
        print("Check the log messages above for details.")
    
    print("\n" + "=" * 80)
    print("Tutorial completed successfully!")
    print(f"Results saved in: {clinvar_dir / 'splice_variants'}")
    print("\nNext steps:")
    print("1. Download actual ClinVar VCF from NCBI")
    print("2. Process full dataset without max_variants limit")
    print("3. Integrate with OpenSpliceAI for delta score computation")
    print("4. Analyze splicing patterns with SplicingPatternAnalyzer")
    if normalize_vcf:
        print("5. VCF normalization ensures clean REF/ALT for sequence construction")
    print("=" * 80)


def create_sample_clinvar_vcf(output_path: Path) -> None:
    """
    Create a sample ClinVar VCF file for demonstration.
    
    Parameters
    ----------
    output_path : Path
        Path to save the sample VCF file
    """
    vcf_content = """##fileformat=VCFv4.2
##source=ClinVar
##reference=GRCh38
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	150547	rs1234567	A	G	.	PASS	ALLELEID=12345;CLNSIG=Pathogenic;CLNDN=Hereditary_cancer_syndrome;CLNREVSTAT=criteria_provided,_multiple_submitters;MC=SO:0001574|splice_acceptor_variant
chr2	234567	rs2345678	C	T	.	PASS	ALLELEID=23456;CLNSIG=Likely_pathogenic;CLNDN=Cardiomyopathy;CLNREVSTAT=criteria_provided,_single_submitter;MC=SO:0001575|splice_donor_variant
chr3	345678	rs3456789	G	A	.	PASS	ALLELEID=34567;CLNSIG=Pathogenic;CLNDN=Retinitis_pigmentosa;CLNREVSTAT=reviewed_by_expert_panel;MC=SO:0001630|intron_variant
chr4	456789	rs4567890	T	C	.	PASS	ALLELEID=45678;CLNSIG=Pathogenic/Likely_pathogenic;CLNDN=Neurodevelopmental_disorder;CLNREVSTAT=criteria_provided,_multiple_submitters;MC=SO:0001627|splice_region_variant
chr5	567890	rs5678901	AT	A	.	PASS	ALLELEID=56789;CLNSIG=Pathogenic;CLNDN=Lynch_syndrome;CLNREVSTAT=practice_guideline;MC=SO:0001574|splice_acceptor_variant
chr6	678901	rs6789012	C	CG	.	PASS	ALLELEID=67890;CLNSIG=Likely_pathogenic;CLNDN=Muscular_dystrophy;CLNREVSTAT=criteria_provided,_single_submitter;MC=SO:0001575|splice_donor_variant
chr7	789012	rs7890123	AGC	A	.	PASS	ALLELEID=78901;CLNSIG=Pathogenic;CLNDN=Epilepsy;CLNREVSTAT=reviewed_by_expert_panel;MC=SO:0001627|splice_region_variant
chr8	890123	rs8901234	G	T	.	PASS	ALLELEID=89012;CLNSIG=Uncertain_significance;CLNDN=Deafness;CLNREVSTAT=criteria_provided,_conflicting_interpretations;MC=SO:0001630|intron_variant
chr9	901234	rs9012345	A	C	.	PASS	ALLELEID=90123;CLNSIG=Benign;CLNDN=Not_specified;CLNREVSTAT=criteria_provided,_single_submitter;MC=SO:0001819|synonymous_variant
chr10	1012345	rs10123456	T	G	.	PASS	ALLELEID=101234;CLNSIG=Pathogenic;CLNDN=Cystic_fibrosis;CLNREVSTAT=practice_guideline;MC=SO:0001574|splice_acceptor_variant
"""
    
    with open(output_path, 'w') as f:
        f.write(vcf_content)


if __name__ == "__main__":
    main()
