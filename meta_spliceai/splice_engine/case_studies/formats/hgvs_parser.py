"""
HGVS (Human Genome Variation Society) notation parser.

Parses HGVS variant descriptions commonly found in splice mutation databases
and converts them to standardized genomic coordinates.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, List
from enum import Enum


class VariantType(Enum):
    """Types of genomic variants."""
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"
    DUPLICATION = "duplication"
    DELINS = "deletion_insertion"
    INVERSION = "inversion"
    COMPLEX = "complex"


class CoordinateType(Enum):
    """Coordinate reference types."""
    GENOMIC = "g"      # Genomic coordinates
    CODING = "c"       # Coding DNA coordinates
    PROTEIN = "p"      # Protein coordinates
    MITOCHONDRIAL = "m"  # Mitochondrial coordinates
    RNA = "r"          # RNA coordinates
    NON_CODING = "n"   # Non-coding RNA coordinates


@dataclass
class HGVSVariant:
    """Parsed HGVS variant representation."""
    
    # Original notation
    hgvs_string: str
    
    # Reference sequence
    reference_sequence: Optional[str] = None
    
    # Coordinate information
    coordinate_type: Optional[CoordinateType] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    
    # Variant details
    variant_type: Optional[VariantType] = None
    reference_allele: Optional[str] = None
    alternate_allele: Optional[str] = None
    
    # Intronic positions (for splice site analysis)
    intronic_offset: Optional[int] = None  # e.g., +1, -2 from exon boundary
    exon_number: Optional[int] = None
    
    # Parsing metadata
    is_valid: bool = False
    parsing_errors: List[str] = None
    
    def __post_init__(self):
        if self.parsing_errors is None:
            self.parsing_errors = []


class HGVSParser:
    """Parser for HGVS variant notation."""
    
    def __init__(self):
        """Initialize HGVS parser with common patterns."""
        
        # Common HGVS patterns
        self.patterns = {
            # Genomic substitutions: g.123A>T
            'genomic_substitution': re.compile(
                r'g\.(\d+)([ATCG])>([ATCG])', re.IGNORECASE
            ),
            
            # Coding substitutions: c.123A>T, c.123+1G>A (intronic)
            'coding_substitution': re.compile(
                r'c\.(\d+)([\+\-]\d+)?([ATCG])>([ATCG])', re.IGNORECASE
            ),
            
            # Deletions: g.123del, c.123_125del
            'deletion': re.compile(
                r'[gc]\.(\d+)(?:_(\d+))?del([ATCG]*)', re.IGNORECASE
            ),
            
            # Insertions: g.123_124insA, c.123+1_123+2insG
            'insertion': re.compile(
                r'[gc]\.(\d+)([\+\-]\d+)?_(\d+)([\+\-]\d+)?ins([ATCG]+)', re.IGNORECASE
            ),
            
            # Deletion-insertions: g.123_125delinsAT
            'delins': re.compile(
                r'[gc]\.(\d+)(?:_(\d+))?delins([ATCG]+)', re.IGNORECASE
            ),
            
            # Splice site specific patterns
            'splice_donor': re.compile(
                r'c\.(\d+)\+(\d+)([ATCG])>([ATCG])', re.IGNORECASE
            ),
            'splice_acceptor': re.compile(
                r'c\.(\d+)\-(\d+)([ATCG])>([ATCG])', re.IGNORECASE
            ),
            
            # Reference sequence patterns
            'reference_seq': re.compile(
                r'(NM_\d+\.\d+|NR_\d+\.\d+|NC_\d+\.\d+|NG_\d+\.\d+)'
            )
        }
    
    def parse(self, hgvs_string: str) -> HGVSVariant:
        """
        Parse HGVS variant string.
        
        Parameters
        ----------
        hgvs_string : str
            HGVS variant description
            
        Returns
        -------
        HGVSVariant
            Parsed variant object
        """
        variant = HGVSVariant(hgvs_string=hgvs_string.strip())
        
        try:
            # Extract reference sequence if present
            ref_match = self.patterns['reference_seq'].search(hgvs_string)
            if ref_match:
                variant.reference_sequence = ref_match.group(1)
                # Remove reference sequence from string for easier parsing
                hgvs_clean = re.sub(self.patterns['reference_seq'], '', hgvs_string)
                hgvs_clean = hgvs_clean.lstrip(':')
            else:
                hgvs_clean = hgvs_string
            
            # Determine coordinate type
            if 'g.' in hgvs_clean:
                variant.coordinate_type = CoordinateType.GENOMIC
            elif 'c.' in hgvs_clean:
                variant.coordinate_type = CoordinateType.CODING
            elif 'p.' in hgvs_clean:
                variant.coordinate_type = CoordinateType.PROTEIN
            elif 'r.' in hgvs_clean:
                variant.coordinate_type = CoordinateType.RNA
            elif 'n.' in hgvs_clean:
                variant.coordinate_type = CoordinateType.NON_CODING
            elif 'm.' in hgvs_clean:
                variant.coordinate_type = CoordinateType.MITOCHONDRIAL
            
            # Parse based on variant type
            if self._parse_substitution(hgvs_clean, variant):
                pass
            elif self._parse_deletion(hgvs_clean, variant):
                pass
            elif self._parse_insertion(hgvs_clean, variant):
                pass
            elif self._parse_delins(hgvs_clean, variant):
                pass
            else:
                variant.parsing_errors.append(f"Unknown variant pattern: {hgvs_clean}")
                return variant
            
            variant.is_valid = True
            
        except Exception as e:
            variant.parsing_errors.append(f"Parsing error: {str(e)}")
        
        return variant
    
    def _parse_substitution(self, hgvs_clean: str, variant: HGVSVariant) -> bool:
        """Parse substitution variants."""
        
        # Try splice site patterns first (more specific)
        donor_match = self.patterns['splice_donor'].match(hgvs_clean)
        if donor_match:
            variant.variant_type = VariantType.SUBSTITUTION
            variant.start_position = int(donor_match.group(1))
            variant.intronic_offset = int(donor_match.group(2))
            variant.reference_allele = donor_match.group(3).upper()
            variant.alternate_allele = donor_match.group(4).upper()
            return True
        
        acceptor_match = self.patterns['splice_acceptor'].match(hgvs_clean)
        if acceptor_match:
            variant.variant_type = VariantType.SUBSTITUTION
            variant.start_position = int(acceptor_match.group(1))
            variant.intronic_offset = -int(acceptor_match.group(2))  # Negative for acceptor
            variant.reference_allele = acceptor_match.group(3).upper()
            variant.alternate_allele = acceptor_match.group(4).upper()
            return True
        
        # Try general coding substitutions
        coding_match = self.patterns['coding_substitution'].match(hgvs_clean)
        if coding_match:
            variant.variant_type = VariantType.SUBSTITUTION
            variant.start_position = int(coding_match.group(1))
            if coding_match.group(2):  # Intronic offset present
                offset_str = coding_match.group(2)
                variant.intronic_offset = int(offset_str)
            variant.reference_allele = coding_match.group(3).upper()
            variant.alternate_allele = coding_match.group(4).upper()
            return True
        
        # Try genomic substitutions
        genomic_match = self.patterns['genomic_substitution'].match(hgvs_clean)
        if genomic_match:
            variant.variant_type = VariantType.SUBSTITUTION
            variant.start_position = int(genomic_match.group(1))
            variant.reference_allele = genomic_match.group(2).upper()
            variant.alternate_allele = genomic_match.group(3).upper()
            return True
        
        return False
    
    def _parse_deletion(self, hgvs_clean: str, variant: HGVSVariant) -> bool:
        """Parse deletion variants."""
        match = self.patterns['deletion'].match(hgvs_clean)
        if match:
            variant.variant_type = VariantType.DELETION
            variant.start_position = int(match.group(1))
            if match.group(2):  # Range deletion
                variant.end_position = int(match.group(2))
            else:  # Single position deletion
                variant.end_position = variant.start_position
            if match.group(3):  # Deleted sequence specified
                variant.reference_allele = match.group(3).upper()
            return True
        return False
    
    def _parse_insertion(self, hgvs_clean: str, variant: HGVSVariant) -> bool:
        """Parse insertion variants."""
        match = self.patterns['insertion'].match(hgvs_clean)
        if match:
            variant.variant_type = VariantType.INSERTION
            variant.start_position = int(match.group(1))
            variant.end_position = int(match.group(3))
            variant.alternate_allele = match.group(5).upper()
            # Handle intronic offsets if present
            if match.group(2):
                variant.intronic_offset = int(match.group(2))
            return True
        return False
    
    def _parse_delins(self, hgvs_clean: str, variant: HGVSVariant) -> bool:
        """Parse deletion-insertion variants."""
        match = self.patterns['delins'].match(hgvs_clean)
        if match:
            variant.variant_type = VariantType.DELINS
            variant.start_position = int(match.group(1))
            if match.group(2):
                variant.end_position = int(match.group(2))
            else:
                variant.end_position = variant.start_position
            variant.alternate_allele = match.group(3).upper()
            return True
        return False
    
    def is_splice_site_variant(self, variant: HGVSVariant) -> bool:
        """
        Determine if variant affects splice sites.
        
        Parameters
        ----------
        variant : HGVSVariant
            Parsed variant
            
        Returns
        -------
        bool
            True if variant likely affects splicing
        """
        if not variant.is_valid:
            return False
        
        # Intronic variants near exon boundaries
        if variant.intronic_offset is not None:
            # Classic splice sites: +1,+2 (donor) and -1,-2 (acceptor)
            if abs(variant.intronic_offset) <= 2:
                return True
            # Extended splice regions: up to +/-20
            if abs(variant.intronic_offset) <= 20:
                return True
        
        # Exonic variants near splice sites (last/first few bases of exons)
        # This would require exon boundary information to determine accurately
        
        return False
    
    def get_splice_site_type(self, variant: HGVSVariant) -> Optional[str]:
        """
        Determine splice site type affected by variant.
        
        Parameters
        ----------
        variant : HGVSVariant
            Parsed variant
            
        Returns
        -------
        str or None
            "donor", "acceptor", or None
        """
        if not self.is_splice_site_variant(variant):
            return None
        
        if variant.intronic_offset is None:
            return None
        
        if variant.intronic_offset > 0:
            return "donor"  # Positive offset = downstream of exon = donor site
        else:
            return "acceptor"  # Negative offset = upstream of exon = acceptor site
    
    def parse_batch(self, hgvs_strings: List[str]) -> List[HGVSVariant]:
        """
        Parse multiple HGVS strings.
        
        Parameters
        ----------
        hgvs_strings : List[str]
            List of HGVS variant descriptions
            
        Returns
        -------
        List[HGVSVariant]
            List of parsed variants
        """
        return [self.parse(hgvs) for hgvs in hgvs_strings]
    
    def get_parsing_statistics(self, variants: List[HGVSVariant]) -> Dict[str, int]:
        """
        Get parsing statistics for a batch of variants.
        
        Parameters
        ----------
        variants : List[HGVSVariant]
            List of parsed variants
            
        Returns
        -------
        Dict[str, int]
            Statistics including success rate, error counts, etc.
        """
        total = len(variants)
        valid = sum(1 for v in variants if v.is_valid)
        splice_variants = sum(1 for v in variants if self.is_splice_site_variant(v))
        
        variant_types = {}
        for v in variants:
            if v.variant_type:
                vtype = v.variant_type.value
                variant_types[vtype] = variant_types.get(vtype, 0) + 1
        
        return {
            "total_variants": total,
            "successfully_parsed": valid,
            "parsing_errors": total - valid,
            "success_rate": valid / total if total > 0 else 0.0,
            "splice_site_variants": splice_variants,
            "variant_type_counts": variant_types
        } 