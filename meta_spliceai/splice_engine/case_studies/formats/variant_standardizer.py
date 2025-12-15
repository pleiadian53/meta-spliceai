"""
Variant standardizer for genomic coordinates and representations.

Converts between different coordinate systems (0-based, 1-based),
normalizes variant representations, and handles coordinate liftover.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re
from pathlib import Path

from .hgvs_parser import HGVSParser, HGVSVariant


@dataclass
class StandardizedVariant:
    """Standardized variant representation."""
    
    # Genomic coordinates (1-based, inclusive)
    chrom: str
    start: int  # 1-based start position
    end: int    # 1-based end position (inclusive)
    ref: str    # Reference allele
    alt: str    # Alternate allele
    
    # Variant type
    variant_type: str  # "SNV", "insertion", "deletion", "indel", "complex"
    
    # Coordinate system metadata
    coordinate_system: str = "1-based"  # "0-based" or "1-based"
    reference_genome: str = "GRCh38"    # "GRCh37", "GRCh38", etc.
    
    # Original representation
    original_representation: str = ""
    
    # Normalization status
    is_normalized: bool = False
    normalization_method: str = ""


class VariantStandardizer:
    """Standardizes variant representations across different formats."""
    
    def __init__(self, reference_genome: str = "GRCh38"):
        """
        Initialize variant standardizer.
        
        Parameters
        ----------
        reference_genome : str
            Reference genome build (GRCh37, GRCh38, etc.)
        """
        self.reference_genome = reference_genome
        self.hgvs_parser = HGVSParser()
    
    def standardize_coordinates(self, chrom: str, position: int, 
                              ref: str, alt: str,
                              coordinate_system: str = "1-based") -> StandardizedVariant:
        """
        Standardize variant coordinates to consistent format.
        
        Parameters
        ----------
        chrom : str
            Chromosome identifier
        position : int
            Variant position
        ref : str
            Reference allele
        alt : str
            Alternate allele
        coordinate_system : str
            Input coordinate system ("0-based" or "1-based")
            
        Returns
        -------
        StandardizedVariant
            Standardized variant representation
        """
        # Normalize chromosome name
        normalized_chrom = self._normalize_chromosome(chrom)
        
        # Convert to 1-based coordinates if needed
        if coordinate_system == "0-based":
            start_pos = position + 1
        else:
            start_pos = position
        
        # Determine variant type and calculate end position
        variant_type, end_pos = self._classify_variant(start_pos, ref, alt)
        
        # Normalize alleles
        normalized_ref, normalized_alt = self._normalize_alleles(ref, alt)
        
        return StandardizedVariant(
            chrom=normalized_chrom,
            start=start_pos,
            end=end_pos,
            ref=normalized_ref,
            alt=normalized_alt,
            variant_type=variant_type,
            coordinate_system="1-based",
            reference_genome=self.reference_genome,
            original_representation=f"{chrom}:{position}:{ref}>{alt}",
            is_normalized=True,
            normalization_method="basic_standardization"
        )
    
    def _normalize_chromosome(self, chrom: str) -> str:
        """Normalize chromosome representation."""
        # Remove 'chr' prefix if present
        if chrom.startswith('chr'):
            chrom = chrom[3:]
        
        # Handle special cases
        if chrom.upper() == 'MT':
            return 'MT'
        elif chrom.upper() in ['X', 'Y']:
            return chrom.upper()
        
        # Ensure numeric chromosomes are strings
        try:
            chrom_num = int(chrom)
            if 1 <= chrom_num <= 22:
                return str(chrom_num)
        except ValueError:
            pass
        
        return chrom
    
    def _classify_variant(self, position: int, ref: str, alt: str) -> Tuple[str, int]:
        """
        Classify variant type and calculate end position.
        
        Returns
        -------
        Tuple[str, int]
            (variant_type, end_position)
        """
        ref_len = len(ref)
        alt_len = len(alt)
        
        if ref_len == 1 and alt_len == 1:
            # Single nucleotide variant
            return "SNV", position
        elif ref_len == 1 and alt_len > 1:
            # Insertion
            return "insertion", position
        elif ref_len > 1 and alt_len == 1:
            # Deletion
            return "deletion", position + ref_len - 1
        elif ref_len > 1 and alt_len > 1:
            # Complex indel
            return "indel", position + max(ref_len, alt_len) - 1
        else:
            # Fallback
            return "complex", position + ref_len - 1
    
    def _normalize_alleles(self, ref: str, alt: str) -> Tuple[str, str]:
        """Normalize allele representations."""
        # Convert to uppercase
        ref = ref.upper()
        alt = alt.upper()
        
        # Remove common prefix and suffix for complex variants
        if len(ref) > 1 or len(alt) > 1:
            # Find common prefix
            prefix_len = 0
            min_len = min(len(ref), len(alt))
            for i in range(min_len):
                if ref[i] == alt[i]:
                    prefix_len += 1
                else:
                    break
            
            # Find common suffix (only if we haven't consumed the entire string)
            suffix_len = 0
            if prefix_len < min_len:
                for i in range(1, min_len - prefix_len + 1):
                    if ref[-i] == alt[-i]:
                        suffix_len += 1
                    else:
                        break
            
            # Extract the differing part
            if prefix_len > 0 or suffix_len > 0:
                end_idx_ref = len(ref) - suffix_len if suffix_len > 0 else len(ref)
                end_idx_alt = len(alt) - suffix_len if suffix_len > 0 else len(alt)
                
                ref = ref[prefix_len:end_idx_ref]
                alt = alt[prefix_len:end_idx_alt]
                
                # Ensure we don't have empty alleles
                if not ref:
                    ref = "-"
                if not alt:
                    alt = "-"
        
        return ref, alt
    
    def standardize_from_hgvs(self, hgvs_string: str) -> Optional[StandardizedVariant]:
        """
        Standardize variant from HGVS notation.
        
        Parameters
        ----------
        hgvs_string : str
            HGVS variant description
            
        Returns
        -------
        StandardizedVariant or None
            Standardized variant if parsing successful
        """
        hgvs_variant = self.hgvs_parser.parse(hgvs_string)
        
        if not hgvs_variant.is_valid:
            return None
        
        # For now, we can only handle variants with genomic positions
        # In practice, you'd need transcript-to-genomic mapping
        if not hgvs_variant.start_position:
            return None
        
        # Determine variant type
        if hgvs_variant.variant_type:
            variant_type = hgvs_variant.variant_type.value
        else:
            variant_type = "unknown"
        
        # Calculate end position
        if hgvs_variant.end_position:
            end_pos = hgvs_variant.end_position
        else:
            end_pos = hgvs_variant.start_position
        
        # Handle alleles
        ref = hgvs_variant.reference_allele or ""
        alt = hgvs_variant.alternate_allele or ""
        
        return StandardizedVariant(
            chrom="",  # Would need transcript mapping to determine
            start=hgvs_variant.start_position,
            end=end_pos,
            ref=ref,
            alt=alt,
            variant_type=variant_type,
            coordinate_system="1-based",
            reference_genome=self.reference_genome,
            original_representation=hgvs_string,
            is_normalized=True,
            normalization_method="hgvs_parsing"
        )
    
    def standardize_from_vcf(self, chrom: str, pos: int, ref: str, alt: str) -> StandardizedVariant:
        """
        Standardize variant from VCF format.
        
        VCF uses 1-based coordinates for SNVs and the position before
        the event for indels.
        
        Parameters
        ----------
        chrom : str
            Chromosome from VCF
        pos : int
            Position from VCF (1-based)
        ref : str
            Reference allele from VCF
        alt : str
            Alternate allele from VCF
            
        Returns
        -------
        StandardizedVariant
            Standardized variant
        """
        # VCF coordinates are already 1-based, but may need adjustment for indels
        normalized_chrom = self._normalize_chromosome(chrom)
        
        # Classify the variant
        variant_type, end_pos = self._classify_variant(pos, ref, alt)
        
        # Normalize alleles
        normalized_ref, normalized_alt = self._normalize_alleles(ref, alt)
        
        return StandardizedVariant(
            chrom=normalized_chrom,
            start=pos,
            end=end_pos,
            ref=normalized_ref,
            alt=normalized_alt,
            variant_type=variant_type,
            coordinate_system="1-based",
            reference_genome=self.reference_genome,
            original_representation=f"{chrom}:{pos}:{ref}>{alt}",
            is_normalized=True,
            normalization_method="vcf_standardization"
        )
    
    def standardize_from_bed(self, chrom: str, start: int, end: int, 
                           ref: str, alt: str) -> StandardizedVariant:
        """
        Standardize variant from BED format.
        
        BED uses 0-based coordinates.
        
        Parameters
        ----------
        chrom : str
            Chromosome from BED
        start : int
            Start position from BED (0-based)
        end : int
            End position from BED (0-based, exclusive)
        ref : str
            Reference allele
        alt : str
            Alternate allele
            
        Returns
        -------
        StandardizedVariant
            Standardized variant
        """
        # Convert from 0-based to 1-based
        start_1based = start + 1
        end_1based = end  # BED end is exclusive, so this becomes inclusive
        
        normalized_chrom = self._normalize_chromosome(chrom)
        
        # Classify the variant
        variant_type, _ = self._classify_variant(start_1based, ref, alt)
        
        # Normalize alleles
        normalized_ref, normalized_alt = self._normalize_alleles(ref, alt)
        
        return StandardizedVariant(
            chrom=normalized_chrom,
            start=start_1based,
            end=end_1based,
            ref=normalized_ref,
            alt=normalized_alt,
            variant_type=variant_type,
            coordinate_system="1-based",
            reference_genome=self.reference_genome,
            original_representation=f"{chrom}:{start}-{end}:{ref}>{alt}",
            is_normalized=True,
            normalization_method="bed_standardization"
        )
    
    def to_vcf_format(self, variant: StandardizedVariant) -> Dict[str, Union[str, int]]:
        """
        Convert standardized variant to VCF format.
        
        Parameters
        ----------
        variant : StandardizedVariant
            Standardized variant
            
        Returns
        -------
        Dict[str, Union[str, int]]
            VCF-formatted variant fields
        """
        # VCF uses 1-based coordinates, which matches our standard
        vcf_chrom = f"chr{variant.chrom}" if not variant.chrom.startswith('chr') else variant.chrom
        
        return {
            "CHROM": vcf_chrom,
            "POS": variant.start,
            "REF": variant.ref,
            "ALT": variant.alt,
            "QUAL": ".",
            "FILTER": "PASS",
            "INFO": f"SVTYPE={variant.variant_type}"
        }
    
    def to_bed_format(self, variant: StandardizedVariant) -> Dict[str, Union[str, int]]:
        """
        Convert standardized variant to BED format.
        
        Parameters
        ----------
        variant : StandardizedVariant
            Standardized variant
            
        Returns
        -------
        Dict[str, Union[str, int]]
            BED-formatted variant fields
        """
        # Convert from 1-based to 0-based for BED
        bed_start = variant.start - 1
        bed_end = variant.end  # BED end is exclusive, so add 1 to inclusive end
        
        bed_chrom = f"chr{variant.chrom}" if not variant.chrom.startswith('chr') else variant.chrom
        
        return {
            "chrom": bed_chrom,
            "chromStart": bed_start,
            "chromEnd": bed_end,
            "name": f"{variant.ref}>{variant.alt}",
            "score": 0,
            "strand": "."
        }
    
    def batch_standardize(self, variants: List[Dict[str, Union[str, int]]], 
                         input_format: str = "auto") -> List[StandardizedVariant]:
        """
        Standardize multiple variants.
        
        Parameters
        ----------
        variants : List[Dict[str, Union[str, int]]]
            List of variant dictionaries
        input_format : str
            Input format ("vcf", "bed", "hgvs", "auto")
            
        Returns
        -------
        List[StandardizedVariant]
            List of standardized variants
        """
        standardized = []
        
        for variant in variants:
            try:
                if input_format == "vcf" or (input_format == "auto" and "POS" in variant):
                    std_variant = self.standardize_from_vcf(
                        variant["CHROM"], variant["POS"], 
                        variant["REF"], variant["ALT"]
                    )
                elif input_format == "bed" or (input_format == "auto" and "chromStart" in variant):
                    std_variant = self.standardize_from_bed(
                        variant["chrom"], variant["chromStart"], variant["chromEnd"],
                        variant.get("ref", ""), variant.get("alt", "")
                    )
                elif input_format == "hgvs" or (input_format == "auto" and "hgvs" in variant):
                    std_variant = self.standardize_from_hgvs(variant["hgvs"])
                else:
                    # Try coordinate-based standardization
                    std_variant = self.standardize_coordinates(
                        variant.get("chrom", ""), variant.get("position", 0),
                        variant.get("ref", ""), variant.get("alt", "")
                    )
                
                if std_variant:
                    standardized.append(std_variant)
                    
            except Exception as e:
                print(f"Failed to standardize variant {variant}: {e}")
                continue
        
        return standardized
    
    def validate_variant(self, variant: StandardizedVariant) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate a standardized variant.
        
        Parameters
        ----------
        variant : StandardizedVariant
            Variant to validate
            
        Returns
        -------
        Dict[str, Union[bool, List[str]]]
            Validation results
        """
        is_valid = True
        errors = []
        warnings = []
        
        # Check chromosome
        if not variant.chrom:
            is_valid = False
            errors.append("Missing chromosome")
        elif variant.chrom not in [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']:
            warnings.append(f"Unusual chromosome: {variant.chrom}")
        
        # Check coordinates
        if variant.start <= 0:
            is_valid = False
            errors.append("Invalid start position")
        
        if variant.end < variant.start:
            is_valid = False
            errors.append("End position before start position")
        
        # Check alleles
        if not variant.ref and not variant.alt:
            is_valid = False
            errors.append("Missing both reference and alternate alleles")
        
        # Check for valid nucleotides
        valid_nucleotides = set('ATCGN-')
        if variant.ref and not set(variant.ref.upper()).issubset(valid_nucleotides):
            warnings.append(f"Non-standard nucleotides in reference: {variant.ref}")
        
        if variant.alt and not set(variant.alt.upper()).issubset(valid_nucleotides):
            warnings.append(f"Non-standard nucleotides in alternate: {variant.alt}")
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings
        } 