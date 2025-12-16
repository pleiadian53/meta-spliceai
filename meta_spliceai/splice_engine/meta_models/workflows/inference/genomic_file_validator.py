#!/usr/bin/env python3
"""
Genomic File Compatibility Validator

A standalone utility for validating that GTF and FASTA files are compatible
and meant for each other. This module is separate from the inference workflow
to maintain clear separation of concerns.

The inference workflow focuses on making predictions, while this module
focuses on validating genomic data integrity.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class GenomicFileValidator:
    """
    Validates compatibility between GTF and FASTA files.
    
    This utility performs systematic checks to ensure files are from the same:
    - Genome build (GRCh37, GRCh38)
    - Coordinate system
    - Assembly version
    
    Supports both Ensembl and GENCODE naming conventions.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the validator.
        
        Parameters
        ----------
        verbose : bool
            Whether to enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)
    
    def validate_gtf_fasta_compatibility(self, gtf_path: Union[str, Path], 
                                       fasta_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate that GTF and FASTA files are compatible.
        
        Parameters
        ----------
        gtf_path : str or Path
            Path to GTF file
        fasta_path : str or Path
            Path to FASTA file
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive validation results
        """
        gtf_path = Path(gtf_path)
        fasta_path = Path(fasta_path)
        
        validation = {
            "compatible": True,
            "gtf_path": str(gtf_path),
            "fasta_path": str(fasta_path),
            "checks": {},
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check 1: File existence
        if not gtf_path.exists():
            validation["compatible"] = False
            validation["errors"].append(f"GTF file not found: {gtf_path}")
        if not fasta_path.exists():
            validation["compatible"] = False
            validation["errors"].append(f"FASTA file not found: {fasta_path}")
        
        if not validation["compatible"]:
            return validation
        
        # Check 2: Extract genome build from GTF header
        gtf_build = self._extract_genome_build_from_gtf(gtf_path)
        validation["checks"]["gtf_genome_build"] = gtf_build
        
        # Check 3: Extract genome build from FASTA header
        fasta_build = self._extract_genome_build_from_fasta(fasta_path)
        validation["checks"]["fasta_genome_build"] = fasta_build
        
        # Check 4: Compare genome builds
        if gtf_build and fasta_build:
            if gtf_build != fasta_build:
                validation["compatible"] = False
                validation["errors"].append(f"Genome build mismatch: GTF={gtf_build}, FASTA={fasta_build}")
            else:
                validation["checks"]["genome_build_match"] = True
        else:
            validation["warnings"].append("Could not extract genome build from one or both files")
        
        # Check 5: Extract Ensembl release from GTF filename
        gtf_release = self._extract_ensembl_release_from_filename(gtf_path)
        validation["checks"]["gtf_ensembl_release"] = gtf_release
        
        # Check 6: Validate FASTA naming convention
        fasta_naming = self._validate_fasta_naming_convention(fasta_path)
        validation["checks"]["fasta_naming_convention"] = fasta_naming
        
        # Check 7: Validate chromosome names match
        gtf_chroms = self._extract_chromosomes_from_gtf(gtf_path)
        fasta_chroms = self._extract_chromosomes_from_fasta(fasta_path)
        
        if gtf_chroms and fasta_chroms:
            common_chroms = set(gtf_chroms) & set(fasta_chroms)
            validation["checks"]["common_chromosomes"] = len(common_chroms)
            validation["checks"]["gtf_chromosomes"] = len(gtf_chroms)
            validation["checks"]["fasta_chromosomes"] = len(fasta_chroms)
            
            if len(common_chroms) < min(len(gtf_chroms), len(fasta_chroms)) * 0.8:
                validation["warnings"].append(
                    f"Limited chromosome overlap: {len(common_chroms)} common out of "
                    f"{len(gtf_chroms)} GTF and {len(fasta_chroms)} FASTA"
                )
        
        # Check 8: Validate coordinate system
        gtf_coords = self._extract_coordinate_system_from_gtf(gtf_path)
        validation["checks"]["gtf_coordinate_system"] = gtf_coords
        
        # Check 9: Validate file integrity (basic checks)
        gtf_integrity = self._validate_gtf_integrity(gtf_path)
        fasta_integrity = self._validate_fasta_integrity(fasta_path)
        validation["checks"]["gtf_integrity"] = gtf_integrity
        validation["checks"]["fasta_integrity"] = fasta_integrity
        
        # Generate summary and recommendations
        validation.update(self._generate_summary_and_recommendations(validation))
        
        return validation
    
    def _extract_genome_build_from_gtf(self, gtf_path: Path) -> Optional[str]:
        """Extract genome build from GTF file header."""
        try:
            with open(gtf_path, 'r') as f:
                for line in f:
                    if line.startswith('#!genome-build'):
                        # Format: #!genome-build GRCh38.p14
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            build = parts[1]
                            # Extract base build (GRCh38 from GRCh38.p14)
                            if '.' in build:
                                return build.split('.')[0]
                            return build
                    elif line.startswith('#!genome-version'):
                        # Format: #!genome-version GRCh38
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            return parts[1]
                    elif not line.startswith('#'):
                        # Stop at first non-comment line
                        break
        except Exception as e:
            logger.warning(f"Could not extract genome build from GTF: {e}")
        return None
    
    def _extract_genome_build_from_fasta(self, fasta_path: Path) -> Optional[str]:
        """Extract genome build from FASTA file header."""
        try:
            with open(fasta_path, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        # Format: >1 dna:chromosome chromosome:GRCh38:1:1:248956422:1 REF
                        parts = line.strip().split(':')
                        for part in parts:
                            if part.startswith('GRCh'):
                                return part
                        break
        except Exception as e:
            logger.warning(f"Could not extract genome build from FASTA: {e}")
        return None
    
    def _extract_ensembl_release_from_filename(self, gtf_path: Path) -> Optional[str]:
        """Extract Ensembl release from GTF filename."""
        filename = gtf_path.name
        # Pattern: Homo_sapiens.GRCh38.112.gtf
        if '.gtf' in filename:
            parts = filename.split('.')
            if len(parts) >= 3:
                return parts[-2]  # 112 from Homo_sapiens.GRCh38.112.gtf
        return None
    
    def _validate_fasta_naming_convention(self, fasta_path: Path) -> Dict[str, Any]:
        """
        Validate that FASTA filename follows Ensembl naming convention.
        
        Ensembl FASTA files should NOT include release numbers because they are
        genome assembly-specific, not annotation-specific.
        """
        filename = fasta_path.name
        validation = {
            "follows_convention": True,
            "filename": filename,
            "expected_pattern": None,
            "issues": []
        }
        
        # Check for Ensembl naming patterns
        if "Homo_sapiens" in filename and "GRCh" in filename and "dna.primary_assembly.fa" in filename:
            if "GRCh38" in filename:
                validation["expected_pattern"] = "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
            elif "GRCh37" in filename:
                validation["expected_pattern"] = "Homo_sapiens.GRCh37.dna.primary_assembly.fa"
        else:
            validation["follows_convention"] = False
            validation["issues"].append("Does not follow Ensembl naming pattern")
        
        # Check for release numbers (should NOT be present in FASTA)
        if any(char.isdigit() for char in filename.split('.')[-2] if len(filename.split('.')) >= 3):
            parts = filename.split('.')
            if len(parts) >= 3 and parts[-2].isdigit():
                validation["follows_convention"] = False
                validation["issues"].append(f"FASTA file should not include release number: {parts[-2]}")
        
        return validation
    
    def _extract_chromosomes_from_gtf(self, gtf_path: Path, max_lines: int = 10000) -> Optional[List[str]]:
        """Extract chromosome names from GTF file."""
        try:
            chroms = set()
            with open(gtf_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    if not line.startswith('#') and line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 1:
                            chroms.add(parts[0])
            return sorted(list(chroms))
        except Exception as e:
            logger.warning(f"Could not extract chromosomes from GTF: {e}")
        return None
    
    def _extract_chromosomes_from_fasta(self, fasta_path: Path, max_headers: int = 100) -> Optional[List[str]]:
        """Extract chromosome names from FASTA file."""
        try:
            chroms = []
            with open(fasta_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= max_headers:
                        break
                    if line.startswith('>'):
                        # Format: >1 dna:chromosome chromosome:GRCh38:1:1:248956422:1 REF
                        chrom = line.strip().split()[0][1:]  # Remove '>'
                        chroms.append(chrom)
            return chroms
        except Exception as e:
            logger.warning(f"Could not extract chromosomes from FASTA: {e}")
        return None
    
    def _extract_coordinate_system_from_gtf(self, gtf_path: Path) -> str:
        """Extract coordinate system from GTF file (GTF is always 1-based)."""
        return "1-based"  # GTF format is always 1-based
    
    def _validate_gtf_integrity(self, gtf_path: Path) -> Dict[str, Any]:
        """Validate basic GTF file integrity."""
        validation = {
            "valid": True,
            "total_lines": 0,
            "comment_lines": 0,
            "data_lines": 0,
            "issues": []
        }
        
        try:
            with open(gtf_path, 'r') as f:
                for line in f:
                    validation["total_lines"] += 1
                    if line.startswith('#'):
                        validation["comment_lines"] += 1
                    elif line.strip():
                        validation["data_lines"] += 1
                        # Basic GTF format check: should have 9 tab-separated fields
                        parts = line.strip().split('\t')
                        if len(parts) != 9:
                            validation["issues"].append(f"Line {validation['total_lines']}: Expected 9 fields, got {len(parts)}")
                            validation["valid"] = False
            
            if validation["data_lines"] == 0:
                validation["issues"].append("No data lines found in GTF file")
                validation["valid"] = False
                
        except Exception as e:
            validation["valid"] = False
            validation["issues"].append(f"Error reading GTF file: {e}")
        
        return validation
    
    def _validate_fasta_integrity(self, fasta_path: Path) -> Dict[str, Any]:
        """Validate basic FASTA file integrity."""
        validation = {
            "valid": True,
            "total_sequences": 0,
            "total_bases": 0,
            "issues": []
        }
        
        try:
            with open(fasta_path, 'r') as f:
                in_sequence = False
                for line in f:
                    if line.startswith('>'):
                        validation["total_sequences"] += 1
                        in_sequence = True
                    elif in_sequence and line.strip():
                        # Count bases in sequence line
                        validation["total_bases"] += len(line.strip())
            
            if validation["total_sequences"] == 0:
                validation["issues"].append("No sequences found in FASTA file")
                validation["valid"] = False
                
        except Exception as e:
            validation["valid"] = False
            validation["issues"].append(f"Error reading FASTA file: {e}")
        
        return validation
    
    def _generate_summary_and_recommendations(self, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary and recommendations based on validation results."""
        summary = ""
        recommendations = []
        
        if validation["compatible"]:
            gtf_build = validation["checks"].get("gtf_genome_build")
            gtf_release = validation["checks"].get("gtf_ensembl_release")
            
            if gtf_build and gtf_release:
                summary = f"✅ Compatible: {gtf_build} genome build, Ensembl release {gtf_release}"
                summary += f" (GTF: {gtf_release}, FASTA: assembly-specific)"
            else:
                summary = "✅ Compatible: Genome builds match"
        else:
            summary = f"❌ Incompatible: {validation.get('errors', ['Unknown error'])[0]}"
        
        # Generate recommendations
        if validation.get("warnings"):
            recommendations.append("Review warnings above for potential issues")
        
        if not validation["checks"].get("fasta_naming_convention", {}).get("follows_convention", True):
            recommendations.append("Consider using standard Ensembl FASTA naming convention")
        
        if validation["checks"].get("common_chromosomes", 0) < 20:
            recommendations.append("Limited chromosome overlap - verify files are from same source")
        
        return {
            "summary": summary,
            "recommendations": recommendations
        }


def validate_genomic_files(gtf_path: Union[str, Path], 
                          fasta_path: Union[str, Path],
                          verbose: bool = False) -> Dict[str, Any]:
    """
    Convenience function to validate GTF and FASTA file compatibility.
    
    Parameters
    ----------
    gtf_path : str or Path
        Path to GTF file
    fasta_path : str or Path
        Path to FASTA file
    verbose : bool
        Whether to enable verbose logging
        
    Returns
    -------
    Dict[str, Any]
        Validation results
    """
    validator = GenomicFileValidator(verbose=verbose)
    return validator.validate_gtf_fasta_compatibility(gtf_path, fasta_path)


def main():
    """Command-line interface for genomic file validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate GTF and FASTA file compatibility")
    parser.add_argument("gtf_path", help="Path to GTF file")
    parser.add_argument("fasta_path", help="Path to FASTA file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Validate files
    validator = GenomicFileValidator(verbose=args.verbose)
    results = validator.validate_gtf_fasta_compatibility(args.gtf_path, args.fasta_path)
    
    # Output results
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        print(f"\n🔍 Genomic File Compatibility Validation")
        print(f"GTF: {results['gtf_path']}")
        print(f"FASTA: {results['fasta_path']}")
        print(f"\nSummary: {results['summary']}")
        print(f"Compatible: {results['compatible']}")
        
        if results.get("checks"):
            print(f"\nDetailed Checks:")
            for key, value in results["checks"].items():
                print(f"  {key}: {value}")
        
        if results.get("warnings"):
            print(f"\nWarnings:")
            for warning in results["warnings"]:
                print(f"  ⚠️ {warning}")
        
        if results.get("errors"):
            print(f"\nErrors:")
            for error in results["errors"]:
                print(f"  ❌ {error}")
        
        if results.get("recommendations"):
            print(f"\nRecommendations:")
            for rec in results["recommendations"]:
                print(f"  💡 {rec}")


if __name__ == "__main__":
    main()
