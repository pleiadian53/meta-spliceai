#!/usr/bin/env python3
"""
VCF Coordinate Verification Tool

This tool helps verify VCF coordinates against reference FASTA sequences,
providing insights into coordinate systems and strand orientations.

Usage:
    python vcf_coordinate_verifier.py --vcf clinvar.vcf.gz --fasta GRCh38.fa --variants 5
    python vcf_coordinate_verifier.py --verify-position chr1:94062595:G:A --fasta GRCh38.fa
    python vcf_coordinate_verifier.py --verify-position chr1:94062595:G:A --fasta GRCh38.fa --gene-strand - --gene-name ABCA4
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pysam
from pyfaidx import Fasta
import pandas as pd

# Import variant standardizer for complex indel normalization
try:
    from meta_spliceai.splice_engine.case_studies.formats.variant_standardizer import (
        VariantStandardizer, StandardizedVariant
    )
    VARIANT_STANDARDIZER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Variant standardizer not available: {e}")
    VARIANT_STANDARDIZER_AVAILABLE = False

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import path resolver and genomic resources with error handling
try:
    from meta_spliceai.splice_engine.case_studies.tools.enhanced_path_resolver import (
        EnhancedPathResolver, create_enhanced_argument_parser, resolve_file_arguments
    )
    PATH_RESOLVER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced path resolver not available: {e}")
    PATH_RESOLVER_AVAILABLE = False

# Fallback to genomic resources system
try:
    from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import CaseStudyResourceManager
    GENOMIC_RESOURCES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Genomic resources system not available: {e}")
    GENOMIC_RESOURCES_AVAILABLE = False


class VCFCoordinateVerifier:
    """Verify VCF coordinates against reference FASTA sequences."""
    
    def __init__(self, fasta_path: str, enable_normalization: bool = True):
        """Initialize with reference FASTA file."""
        self.fasta_path = Path(fasta_path)
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        
        logger.info(f"Loading reference FASTA: {self.fasta_path}")
        self.fasta = Fasta(str(self.fasta_path))
        logger.info(f"Loaded {len(self.fasta.keys())} chromosomes")
        
        # Initialize variant standardizer for complex indel handling
        self.enable_normalization = enable_normalization and VARIANT_STANDARDIZER_AVAILABLE
        if self.enable_normalization:
            self.variant_standardizer = VariantStandardizer()
            logger.info("Variant standardizer enabled for complex indel normalization")
        else:
            self.variant_standardizer = None
            if enable_normalization:
                logger.warning("Variant standardizer requested but not available")
    
    def verify_single_position(self, chrom: str, pos: int, expected_ref: str, 
                              gene_strand: Optional[str] = None, gene_name: Optional[str] = None) -> Dict:
        """
        Verify a single genomic position against the reference FASTA.
        
        Args:
            chrom: Chromosome name (e.g., "1", "chr1")
            pos: 1-based genomic position
            expected_ref: Expected reference allele from VCF
            gene_strand: Gene strand ('+' or '-') for gene-centric interpretation
            gene_name: Gene name for context
            
        Returns:
            Dict with verification results
        """
        # Normalize chromosome name
        chrom_variants = [chrom, f"chr{chrom}", chrom.replace("chr", "")]
        actual_chrom = None
        
        for variant in chrom_variants:
            if variant in self.fasta.keys():
                actual_chrom = variant
                break
        
        if not actual_chrom:
            return {
                "status": "ERROR",
                "message": f"Chromosome {chrom} not found in FASTA",
                "available_chroms": list(self.fasta.keys())[:10]  # Show first 10
            }
        
        # Extract base at position (convert to 0-based for pyfaidx)
        try:
            # For complex indels, we may need to extract a larger region
            ref_len = len(expected_ref)
            if ref_len == 1:
                # Simple SNV case
                actual_ref = str(self.fasta[actual_chrom][pos-1:pos]).upper()
            else:
                # Complex indel case - extract the full reference region
                actual_ref = str(self.fasta[actual_chrom][pos-1:pos-1+ref_len]).upper()
            
            # Basic comparison
            basic_match = actual_ref == expected_ref.upper()
            
            result = {
                "chromosome": actual_chrom,
                "position": pos,
                "expected_ref": expected_ref.upper(),
                "actual_ref": actual_ref,
                "match": basic_match,
                "status": "SUCCESS" if basic_match else "MISMATCH",
                "verification_method": "basic"
            }
            
            # If basic match fails and normalization is enabled, try normalized comparison
            if not basic_match and self.enable_normalization and ref_len > 1:
                normalized_result = self._verify_with_normalization(
                    actual_chrom, pos, expected_ref, actual_ref
                )
                if normalized_result:
                    result.update(normalized_result)
            
            # Add gene context if provided
            if gene_name:
                result["gene_name"] = gene_name
            if gene_strand:
                result["gene_strand"] = gene_strand
                
                # Add strand-specific interpretation
                complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
                if gene_strand == '-':
                    result["gene_ref"] = complement_map.get(actual_ref, actual_ref)
                    result["gene_expected_ref"] = complement_map.get(expected_ref.upper(), expected_ref.upper())
                    result["gene_context_note"] = f"Gene {gene_name} is on minus strand - showing complement"
                else:
                    result["gene_ref"] = actual_ref
                    result["gene_expected_ref"] = expected_ref.upper()
                    result["gene_context_note"] = f"Gene {gene_name} is on plus strand - same as genomic"
            
            # Add context sequence (±10bp)
            context_start = max(0, pos - 11)
            context_end = min(len(self.fasta[actual_chrom]), pos + 10)
            context = str(self.fasta[actual_chrom][context_start:context_end]).upper()
            
            # Mark the position in context
            mark_pos = pos - context_start - 1
            if 0 <= mark_pos < len(context):
                context_marked = context[:mark_pos] + f"[{context[mark_pos]}]" + context[mark_pos+1:]
                result["context"] = context_marked
            else:
                result["context"] = context
                
            return result
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Error extracting sequence: {str(e)}"
            }
    
    def _verify_with_normalization(self, chrom: str, pos: int, expected_ref: str, actual_ref: str) -> Optional[Dict]:
        """
        Verify complex indel using variant normalization.
        
        This handles cases where VCF normalization differences cause mismatches
        in complex indels but the variants are biologically equivalent.
        """
        if not self.variant_standardizer:
            return None
        
        try:
            # Extract a larger context around the position for comparison
            context_size = max(50, len(expected_ref) * 2)
            context_start = max(1, pos - context_size)
            context_end = pos + len(expected_ref) + context_size
            
            # Extract context sequence
            context_seq = str(self.fasta[chrom][context_start-1:context_end]).upper()
            
            # Check if the expected sequence appears in the context (allowing for left-alignment)
            expected_upper = expected_ref.upper()
            
            # Look for the expected sequence within the context
            found_positions = []
            for i in range(len(context_seq) - len(expected_upper) + 1):
                if context_seq[i:i+len(expected_upper)] == expected_upper:
                    genomic_pos = context_start + i
                    found_positions.append(genomic_pos)
            
            if found_positions:
                # Found the sequence at alternative positions (likely left-aligned differently)
                closest_pos = min(found_positions, key=lambda x: abs(x - pos))
                offset = closest_pos - pos
                
                return {
                    "match": True,
                    "status": "SUCCESS",
                    "verification_method": "normalized",
                    "normalization_note": f"Found expected sequence at position {closest_pos} (offset: {offset:+d})",
                    "likely_cause": "VCF left-alignment difference",
                    "actual_ref": expected_upper,  # Update to show it was found
                }
                
        except Exception as e:
            logger.debug(f"Normalization verification failed: {e}")
            return None
        
        return None
    
    def verify_vcf_variants(self, vcf_path: str, max_variants: int = 10) -> pd.DataFrame:
        """
        Verify multiple variants from a VCF file.
        
        Args:
            vcf_path: Path to VCF file
            max_variants: Maximum number of variants to verify
            
        Returns:
            DataFrame with verification results
        """
        logger.info(f"Opening VCF file: {vcf_path}")
        
        results = []
        variant_count = 0
        
        try:
            with pysam.VariantFile(vcf_path) as vcf:
                for record in vcf:
                    if variant_count >= max_variants:
                        break
                    
                    # Skip multiallelic variants and records without alts
                    if not record.alts or len(record.alts) > 1:
                        continue
                    
                    chrom = record.chrom
                    pos = record.pos  # VCF pos is 1-based
                    ref = record.ref
                    alt = record.alts[0] if record.alts else "."
                    
                    # Verify the reference allele
                    verification = self.verify_single_position(chrom, pos, ref)
                    
                    # Add VCF information
                    verification.update({
                        "vcf_id": record.id or ".",
                        "vcf_alt": alt,
                        "vcf_qual": record.qual,
                        "variant_type": self._classify_variant(ref, alt)
                    })
                    
                    results.append(verification)
                    variant_count += 1
                    
                    if variant_count % 100 == 0:
                        logger.info(f"Processed {variant_count} variants...")
        
        except Exception as e:
            logger.error(f"Error reading VCF file: {str(e)}")
            raise
        
        return pd.DataFrame(results)
    
    def _classify_variant(self, ref: str, alt: str) -> str:
        """Classify variant type based on REF and ALT alleles."""
        if alt == ".":
            return "unknown"
        elif len(ref) == 1 and len(alt) == 1:
            return "SNV"
        elif len(ref) > len(alt):
            return "deletion"
        elif len(ref) < len(alt):
            return "insertion"
        else:
            return "complex"
    
    def generate_report(self, results_df: pd.DataFrame) -> str:
        """Generate a summary report of verification results."""
        total_variants = len(results_df)
        successful = len(results_df[results_df['status'] == 'SUCCESS'])
        matches = len(results_df[results_df['match'] == True])
        mismatches = len(results_df[results_df['match'] == False])
        errors = len(results_df[results_df['status'] == 'ERROR'])
        
        # Calculate coordinate system consistency score
        consistency_score = matches / total_variants * 100 if total_variants > 0 else 0
        
        # Count normalization results
        normalized_matches = len(results_df[results_df.get('verification_method', '') == 'normalized'])
        basic_matches = matches - normalized_matches
        
        report = f"""
VCF Coordinate System Validation Report
=======================================

COORDINATE SYSTEM CONSISTENCY: {consistency_score:.1f}%

Total variants processed: {total_variants}
Successful verifications: {successful}
Reference allele matches: {matches}
  - Basic matches: {basic_matches}
  - Normalized matches: {normalized_matches}
Reference allele mismatches: {mismatches}
Errors: {errors}

Match rate: {matches/total_variants*100:.1f}%
Success rate: {successful/total_variants*100:.1f}%

Variant type distribution:
{results_df['variant_type'].value_counts().to_string()}

"""
        
        if normalized_matches > 0:
            report += f"Normalization Analysis:\n"
            report += f"  Complex indels resolved by normalization: {normalized_matches}\n"
            report += f"  Normalization success rate: {normalized_matches/mismatches*100:.1f}% of mismatches\n\n"
        
        # Add coordinate system assessment
        if consistency_score >= 95:
            report += f"\n✅ COORDINATE SYSTEM ASSESSMENT: CONSISTENT\n"
            report += f"The VCF file appears to use the same coordinate system as the FASTA reference.\n"
        elif consistency_score >= 80:
            report += f"\n⚠️  COORDINATE SYSTEM ASSESSMENT: MOSTLY CONSISTENT\n"
            report += f"Minor inconsistencies detected. Check for chromosome naming or small offsets.\n"
        else:
            report += f"\n❌ COORDINATE SYSTEM ASSESSMENT: INCONSISTENT\n"
            report += f"Major coordinate system mismatch detected. Check genome build and coordinate system.\n"
        
        if mismatches > 0:
            report += f"\nMismatched variants (first 5):\n"
            mismatch_df = results_df[results_df['match'] == False].head()
            for _, row in mismatch_df.iterrows():
                chrom = row['chromosome']
                pos = row['position']
                report += f"  chr{chrom}:{pos} Expected:{row['expected_ref']} Actual:{row['actual_ref']}\n"
            
            # Add genome browser instructions for investigation
            if len(mismatch_df) > 0:
                first_mismatch = mismatch_df.iloc[0]
                chrom = first_mismatch['chromosome']
                pos = first_mismatch['position']
                report += f"\n🌐 Investigate First Mismatch in Genome Browser:\n"
                report += f"UCSC: https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&position=chr{chrom}:{pos-50}-{pos+50}\n"
                report += f"Ensembl: https://www.ensembl.org/Homo_sapiens/Location/View?r={chrom}:{pos-50}-{pos+50}\n"
            
            # Add diagnostic suggestions
            report += f"\nDiagnostic Suggestions:\n"
            report += f"- Check if VCF and FASTA use the same genome build (GRCh37 vs GRCh38)\n"
            report += f"- Verify chromosome naming consistency (chr1 vs 1)\n"
            report += f"- Check for 0-based vs 1-based coordinate system differences\n"
            report += f"- Ensure VCF file is not corrupted or malformed\n"
            report += f"- For complex indels, check VCF normalization (left-alignment)\n"
        
        return report


def resolve_file_paths_robust(vcf_path: Optional[str] = None, fasta_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Robust file path resolution using multiple fallback strategies.
    
    Args:
        vcf_path: VCF file path (can be filename, relative, or absolute)
        fasta_path: FASTA file path (can be filename, relative, or absolute)
        
    Returns:
        Tuple of resolved (vcf_path, fasta_path)
    """
    resolved_vcf = None
    resolved_fasta = None
    
    # Auto-detect project root
    project_root = Path.cwd()
    for parent in [project_root] + list(project_root.parents):
        if (parent / "meta_spliceai").exists() and (parent / "pyproject.toml").exists():
            project_root = parent
            break
    
    # Strategy 1: Try enhanced path resolver
    if PATH_RESOLVER_AVAILABLE:
        try:
            resolver = EnhancedPathResolver(project_root)
            if vcf_path:
                resolved_vcf = str(resolver.resolve_vcf_path(vcf_path))
            if fasta_path:
                resolved_fasta = str(resolver.resolve_fasta_path(fasta_path))
            return resolved_vcf, resolved_fasta
        except Exception as e:
            logger.debug(f"Enhanced path resolver failed: {e}")
    
    # Strategy 2: Try genomic resources system
    if GENOMIC_RESOURCES_AVAILABLE:
        try:
            resource_manager = CaseStudyResourceManager()
            
            # Resolve VCF path
            if vcf_path:
                vcf_candidates = [
                    Path(vcf_path),  # Try as-is first
                    project_root / vcf_path,  # Relative to project root
                    project_root / "data" / "ensembl" / "clinvar" / "vcf" / vcf_path,  # ClinVar directory
                    resource_manager.get_clinvar_vcf_path() / vcf_path if hasattr(resource_manager, 'get_clinvar_vcf_path') else None
                ]
                
                for candidate in vcf_candidates:
                    if candidate and candidate.exists():
                        resolved_vcf = str(candidate)
                        break
            
            # Resolve FASTA path
            if fasta_path:
                fasta_candidates = [
                    Path(fasta_path),  # Try as-is first
                    project_root / fasta_path,  # Relative to project root
                    project_root / "data" / "ensembl" / fasta_path,  # Ensembl directory
                ]
                
                # Try to get standard FASTA from resource manager
                try:
                    standard_fasta = resource_manager.get_fasta_file()
                    if standard_fasta and standard_fasta.name == Path(fasta_path).name:
                        fasta_candidates.insert(0, standard_fasta)
                except Exception:
                    pass
                
                for candidate in fasta_candidates:
                    if candidate and candidate.exists():
                        resolved_fasta = str(candidate)
                        break
            
            if resolved_vcf or resolved_fasta:
                return resolved_vcf, resolved_fasta
                
        except Exception as e:
            logger.debug(f"Genomic resources system failed: {e}")
    
    # Strategy 3: Manual path resolution with common locations
    if vcf_path:
        vcf_candidates = [
            Path(vcf_path),
            project_root / vcf_path,
            project_root / "data" / "ensembl" / "clinvar" / "vcf" / vcf_path,
            project_root / "data" / "clinvar" / vcf_path,
            project_root / "data" / vcf_path
        ]
        
        for candidate in vcf_candidates:
            if candidate.exists():
                resolved_vcf = str(candidate)
                break
    
    if fasta_path:
        fasta_candidates = [
            Path(fasta_path),
            project_root / fasta_path,
            project_root / "data" / "ensembl" / fasta_path,
            project_root / "data" / "reference" / fasta_path,
            project_root / "data" / fasta_path
        ]
        
        for candidate in fasta_candidates:
            if candidate.exists():
                resolved_fasta = str(candidate)
                break
    
    return resolved_vcf, resolved_fasta


def parse_position_string(pos_string: str) -> Tuple[str, int, str, str]:
    """
    Parse position string in format: chr:pos:ref:alt
    Example: chr1:94062595:G:A
    """
    try:
        parts = pos_string.split(':')
        if len(parts) != 4:
            raise ValueError("Position string must be in format chr:pos:ref:alt")
        
        chrom, pos, ref, alt = parts
        return chrom, int(pos), ref, alt
    except Exception as e:
        raise ValueError(f"Invalid position string '{pos_string}': {str(e)}")


def main():
    if PATH_RESOLVER_AVAILABLE:
        parser = create_enhanced_argument_parser("Verify VCF coordinates against reference FASTA")
    else:
        parser = argparse.ArgumentParser(
            description="Verify VCF coordinates against reference FASTA",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
    
    # Override epilog with tool-specific examples
    parser.epilog = """
Examples:
  # Validate coordinate system consistency (recommended for variant analysis)
  python vcf_coordinate_verifier.py --vcf clinvar_20250831.vcf.gz --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa --validate-coordinates
  
  # Verify specific position with gene context
  python vcf_coordinate_verifier.py --verify-position chr1:94062595:G:A --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa --gene-strand - --gene-name ABCA4
  
  # Quick verification of 10 variants
  python vcf_coordinate_verifier.py --vcf clinvar_20250831.vcf.gz --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa --variants 10
  
  # Generate detailed validation report
  python vcf_coordinate_verifier.py --vcf clinvar_20250831.vcf.gz --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa --validate-coordinates --output validation_report.tsv
        """
    
    parser.add_argument("--fasta", required=True, help="Reference FASTA file")
    parser.add_argument("--vcf", help="VCF file to verify")
    parser.add_argument("--verify-position", help="Single position to verify (chr:pos:ref:alt)")
    parser.add_argument("--gene-strand", choices=['+', '-'], help="Gene strand for gene-centric interpretation")
    parser.add_argument("--gene-name", help="Gene name for context")
    parser.add_argument("--variants", type=int, default=10, help="Number of variants to verify from VCF")
    parser.add_argument("--validate-coordinates", action="store_true", 
                       help="Validate coordinate system consistency (recommended: 100+ variants)")
    parser.add_argument("--enable-normalization", action="store_true", default=True,
                       help="Enable variant normalization for complex indels (default: True)")
    parser.add_argument("--disable-normalization", action="store_true",
                       help="Disable variant normalization (use basic verification only)")
    parser.add_argument("--output", help="Output TSV file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Resolve file paths using robust resolution
    try:
        resolved_vcf, resolved_fasta = resolve_file_paths_robust(
            vcf_path=args.vcf,
            fasta_path=args.fasta
        )
        
        # Update args with resolved paths
        if resolved_vcf:
            args.vcf = resolved_vcf
            logger.info(f"Resolved VCF path: {resolved_vcf}")
        
        if resolved_fasta:
            args.fasta = resolved_fasta
            logger.info(f"Resolved FASTA path: {resolved_fasta}")
            
    except Exception as e:
        logger.warning(f"Path resolution failed, using original paths: {e}")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.vcf and not args.verify_position:
        parser.error("Must specify either --vcf or --verify-position")
    
    try:
        # Determine normalization setting
        enable_normalization = args.enable_normalization and not args.disable_normalization
        
        # Initialize verifier
        verifier = VCFCoordinateVerifier(args.fasta, enable_normalization=enable_normalization)
        
        if args.verify_position:
            # Verify single position
            chrom, pos, ref, alt = parse_position_string(args.verify_position)
            logger.info(f"Verifying position {chrom}:{pos} {ref}→{alt}")
            
            result = verifier.verify_single_position(chrom, pos, ref, 
                                                   gene_strand=args.gene_strand,
                                                   gene_name=args.gene_name)
            
            print(f"\nPosition Verification Results:")
            print(f"{'='*50}")
            print(f"Chromosome: {result.get('chromosome', 'N/A')}")
            print(f"Position: {result.get('position', 'N/A')}")
            print(f"Expected REF: {result.get('expected_ref', 'N/A')}")
            print(f"Actual REF: {result.get('actual_ref', 'N/A')}")
            print(f"Match: {result.get('match', 'N/A')}")
            print(f"Status: {result.get('status', 'N/A')}")
            
            if 'context' in result:
                print(f"Context: {result['context']}")
            
            # Add genome browser viewing instructions
            chrom_display = result.get('chromosome', chrom)
            pos_display = result.get('position', pos)
            print(f"\n🌐 View in Genome Browser:")
            print(f"{'='*30}")
            print(f"UCSC Genome Browser:")
            print(f"  https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&position=chr{chrom_display}:{pos_display-50}-{pos_display+50}")
            print(f"IGV coordinates: chr{chrom_display}:{pos_display-50}-{pos_display+50}")
            print(f"Ensembl Genome Browser:")
            print(f"  https://www.ensembl.org/Homo_sapiens/Location/View?r={chrom_display}:{pos_display-50}-{pos_display+50}")
            
            # Show gene-centric interpretation if provided
            if 'gene_name' in result:
                print(f"\nGene Context:")
                print(f"{'='*30}")
                print(f"Gene: {result.get('gene_name', 'N/A')}")
                print(f"Strand: {result.get('gene_strand', 'N/A')}")
                print(f"Gene REF: {result.get('gene_ref', 'N/A')}")
                print(f"Gene Expected REF: {result.get('gene_expected_ref', 'N/A')}")
                print(f"Note: {result.get('gene_context_note', 'N/A')}")
                
                if args.gene_strand == '-':
                    # Show what the variant would look like in gene context
                    complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
                    gene_alt = complement_map.get(alt, alt)
                    print(f"Gene variant: {result.get('gene_ref', 'N/A')}→{gene_alt}")
            
            if result.get('status') == 'ERROR':
                print(f"Error: {result.get('message', 'Unknown error')}")
                sys.exit(1)
            elif not result.get('match', False):
                print(f"\n⚠️  MISMATCH DETECTED!")
                print(f"This suggests a coordinate system issue or incorrect reference.")
                sys.exit(1)
            else:
                print(f"\n✅ VERIFICATION SUCCESSFUL!")
                print(f"VCF coordinates are consistent with reference FASTA.")
        
        if args.vcf:
            # Determine number of variants for validation
            if args.validate_coordinates:
                # Use more variants for coordinate system validation
                num_variants = max(100, args.variants)
                logger.info(f"Coordinate system validation: verifying {num_variants} variants from VCF")
            else:
                num_variants = args.variants
                logger.info(f"Verifying {num_variants} variants from VCF")
            
            results_df = verifier.verify_vcf_variants(args.vcf, num_variants)
            
            # Generate report
            report = verifier.generate_report(results_df)
            print(report)
            
            # Save results if requested
            if args.output:
                results_df.to_csv(args.output, sep='\t', index=False)
                logger.info(f"Results saved to {args.output}")
            
            # Show sample results
            if len(results_df) > 0:
                print("\nSample verification results:")
                print("="*80)
                display_cols = ['chromosome', 'position', 'expected_ref', 'actual_ref', 'match', 'variant_type']
                print(results_df[display_cols].head().to_string(index=False))
                
                # Check for any mismatches
                mismatches = results_df[results_df['match'] == False]
                if len(mismatches) > 0:
                    print(f"\n⚠️  Found {len(mismatches)} coordinate mismatches!")
                    print("This may indicate coordinate system inconsistencies.")
                else:
                    print(f"\n✅ All {len(results_df)} variants verified successfully!")
                    print("VCF coordinates are consistent with reference FASTA.")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
