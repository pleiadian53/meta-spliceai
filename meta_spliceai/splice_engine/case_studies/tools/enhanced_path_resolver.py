
#!/usr/bin/env python3
"""
Enhanced Path Resolver for Case Study Tools

This module provides smart path resolution for VCF files, FASTA files, and other
case study data sources, supporting:
1. Relative paths from project directory
2. Filename-only specification with automatic directory resolution
3. Integration with resource manager for standard data locations
"""

from pathlib import Path
from typing import Optional, Union
import os
import argparse
import sys

from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import CaseStudyResourceManager


class EnhancedPathResolver:
    """Smart path resolver for case study data files."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize path resolver.
        
        Parameters
        ----------
        project_root : Path, optional
            Project root directory (auto-detected if None)
        """
        # Auto-detect project root if not provided
        if project_root is None:
            # Find project root by looking for key files
            current_dir = Path.cwd()
            for parent in [current_dir] + list(current_dir.parents):
                if (parent / "meta_spliceai").exists() and (parent / "pyproject.toml").exists():
                    project_root = parent
                    break
            
            if project_root is None:
                project_root = current_dir
        
        self.project_root = Path(project_root)
        
        # Initialize resource manager for standard paths
        try:
            self.resource_manager = CaseStudyResourceManager()
        except Exception as e:
            print(f"⚠️  Resource manager initialization failed: {e}")
            self.resource_manager = None
    
    def resolve_vcf_path(self, vcf_input: str) -> Path:
        """
        Resolve VCF file path with smart detection.
        
        Supports:
        1. Absolute paths: /full/path/to/file.vcf.gz
        2. Relative paths: data/clinvar/file.vcf.gz
        3. Filename only: clinvar_20250831.vcf.gz (auto-resolves to ClinVar dir)
        
        Parameters
        ----------
        vcf_input : str
            VCF path specification
            
        Returns
        -------
        Path
            Resolved VCF file path
            
        Examples
        --------
        >>> resolver = EnhancedPathResolver()
        >>> # Filename only - auto-resolves to ClinVar directory
        >>> path = resolver.resolve_vcf_path("clinvar_20250831.vcf.gz")
        >>> # Returns: data/ensembl/case_studies/clinvar/clinvar_20250831.vcf.gz
        
        >>> # Relative path from project root
        >>> path = resolver.resolve_vcf_path("data/custom/my_variants.vcf.gz")
        >>> # Returns: /project/root/data/custom/my_variants.vcf.gz
        """
        
        vcf_path = Path(vcf_input)
        
        # Case 1: Absolute path
        if vcf_path.is_absolute():
            if vcf_path.exists():
                return vcf_path
            else:
                raise FileNotFoundError(f"VCF file not found: {vcf_path}")
        
        # Case 2: Relative path from project root
        if "/" in vcf_input:
            full_path = self.project_root / vcf_path
            if full_path.exists():
                return full_path
            else:
                raise FileNotFoundError(f"VCF file not found: {full_path}")
        
        # Case 3: Filename only - try smart resolution
        return self._resolve_filename_only_vcf(vcf_input)
    
    def _resolve_filename_only_vcf(self, filename: str) -> Path:
        """Resolve filename-only VCF specification to full path."""
        
        # Standard search locations in priority order
        search_locations = []
        
        # 1. Use resource manager for standard locations
        if self.resource_manager:
            # ClinVar directory
            clinvar_path = self.resource_manager.get_clinvar_dir() / filename
            search_locations.append(clinvar_path)
            
            # Other standard directories
            search_locations.extend([
                self.resource_manager.get_splicevardb_dir() / filename,
                self.resource_manager.get_custom_dir() / filename,
                self.resource_manager.get_processed_dir() / filename,
                self.resource_manager.get_normalized_vcf_dir() / filename
            ])
        
        # 2. Standard project relative paths (actual structure)
        search_locations.extend([
            self.project_root / "data" / "ensembl" / "clinvar" / "vcf" / filename,  # Actual ClinVar VCF location
            self.project_root / "data" / "ensembl" / "case_studies" / "clinvar" / filename,
            self.project_root / "data" / "ensembl" / "case_studies" / "custom" / filename,
            self.project_root / "data" / "clinvar" / filename,
            self.project_root / "data" / "vcf" / filename,
            self.project_root / filename  # Current directory fallback
        ])
        
        # Find first existing file
        for path in search_locations:
            if path.exists():
                print(f"✅ Resolved VCF: {filename} → {path}")
                return path
        
        # If not found, provide helpful error with available VCF files
        search_dirs = [str(p.parent) for p in search_locations[:5]]  # Show first 5
        available_vcfs = []
        
        # Look for available VCF files in the main search directories
        for search_path in search_locations[:3]:  # Check first 3 directories
            search_dir = search_path.parent
            if search_dir.exists():
                available_vcfs.extend([f.name for f in search_dir.glob("*.vcf.gz")])
                available_vcfs.extend([f.name for f in search_dir.glob("*.vcf")])
        
        error_msg = f"VCF file '{filename}' not found in standard locations.\n"
        error_msg += f"Searched directories: {search_dirs}\n"
        
        if available_vcfs:
            error_msg += f"Available VCF files: {sorted(set(available_vcfs))[:5]}\n"  # Show first 5
            
        error_msg += "Use full or relative path if file is in non-standard location."
        raise FileNotFoundError(error_msg)
    
    def resolve_fasta_path(self, fasta_input: str) -> Path:
        """
        Resolve FASTA file path with smart detection.
        
        Supports:
        1. Absolute paths: /full/path/to/genome.fa
        2. Relative paths: data/reference/GRCh38.fa
        3. Standard names: GRCh38.fa (auto-resolves to genomic resources)
        
        Parameters
        ----------
        fasta_input : str
            FASTA path specification
            
        Returns
        -------
        Path
            Resolved FASTA file path
        """
        
        fasta_path = Path(fasta_input)
        
        # Case 1: Absolute path
        if fasta_path.is_absolute():
            if fasta_path.exists():
                return fasta_path
            else:
                raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        
        # Case 2: Relative path from project root
        if "/" in fasta_input:
            full_path = self.project_root / fasta_path
            if full_path.exists():
                return full_path
            else:
                raise FileNotFoundError(f"FASTA file not found: {full_path}")
        
        # Case 3: Direct filename search only - no fallback to defaults
        
        # Case 4: Search standard locations (flexible for any FASTA filename)
        search_locations = [
            self.project_root / "data" / "ensembl" / fasta_input,  # Direct in ensembl directory
            self.project_root / "data" / "reference" / fasta_input,
            self.project_root / "data" / fasta_input,
            self.project_root / fasta_input
        ]
        
        for path in search_locations:
            if path.exists():
                print(f"✅ Resolved FASTA: {fasta_input} → {path}")
                return path
        
        # Provide helpful error message with available FASTA files
        available_fastas = []
        for search_dir in [self.project_root / "data" / "ensembl", self.project_root / "data" / "reference"]:
            if search_dir.exists():
                available_fastas.extend([f.name for f in search_dir.glob("*.fa")])
                available_fastas.extend([f.name for f in search_dir.glob("*.fasta")])
        
        error_msg = f"FASTA file '{fasta_input}' not found.\n"
        error_msg += f"Searched directories: {[str(p.parent) for p in search_locations[:3]]}\n"
        
        if available_fastas:
            error_msg += f"Available FASTA files: {sorted(set(available_fastas))[:5]}\n"  # Show first 5
        
        error_msg += "Use full or relative path if file is in non-standard location."
        raise FileNotFoundError(error_msg)
    
    def resolve_dataset_path(self, dataset_input: str) -> Path:
        """
        Resolve training dataset path with smart detection.
        
        Supports dataset directory resolution for meta-model paths.
        """
        
        dataset_path = Path(dataset_input)
        
        # Case 1: Absolute path
        if dataset_path.is_absolute():
            if dataset_path.exists():
                return dataset_path
            else:
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Case 2: Relative path from project root
        full_path = self.project_root / dataset_path
        if full_path.exists():
            return full_path
        
        # Case 3: Standard dataset names
        standard_locations = [
            self.project_root / dataset_input,  # Direct in project root
            self.project_root / "data" / dataset_input,
            self.project_root / "datasets" / dataset_input
        ]
        
        for path in standard_locations:
            if path.exists():
                print(f"✅ Resolved dataset: {dataset_input} → {path}")
                return path
        
        raise FileNotFoundError(f"Dataset '{dataset_input}' not found in standard locations")


def create_enhanced_argument_parser(description: str) -> argparse.ArgumentParser:
    """
    Create argument parser with enhanced path resolution support.
    
    This function creates a parser that supports smart path resolution
    for common case study file types.
    """
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Path Resolution Examples:
  --vcf clinvar_20250831.vcf.gz                    # Auto-resolves to ClinVar directory
  --vcf data/custom/my_variants.vcf.gz             # Relative to project root
  --vcf /full/path/to/variants.vcf.gz              # Absolute path
  
  --fasta GRCh38.fa                                # Auto-resolves to genomic resources
  --fasta data/reference/genome.fa                 # Relative to project root
  --fasta /full/path/to/genome.fa                  # Absolute path
        """
    )
    
    return parser


def resolve_file_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """
    Resolve file path arguments using enhanced path resolution.
    
    This function processes parsed arguments and resolves file paths
    using smart detection and resource manager integration.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Returns
    -------
    argparse.Namespace
        Arguments with resolved file paths
    """
    
    resolver = EnhancedPathResolver()
    
    # Resolve VCF path if provided
    if hasattr(args, 'vcf') and args.vcf:
        try:
            args.vcf = str(resolver.resolve_vcf_path(args.vcf))
            print(f"📁 VCF resolved: {args.vcf}")
        except FileNotFoundError as e:
            print(f"❌ VCF resolution failed: {e}")
            sys.exit(1)
    
    # Resolve FASTA path if provided
    if hasattr(args, 'fasta') and args.fasta:
        try:
            args.fasta = str(resolver.resolve_fasta_path(args.fasta))
            print(f"📁 FASTA resolved: {args.fasta}")
        except FileNotFoundError as e:
            print(f"❌ FASTA resolution failed: {e}")
            sys.exit(1)
    
    # Resolve dataset path if provided
    if hasattr(args, 'dataset') and args.dataset:
        try:
            args.dataset = str(resolver.resolve_dataset_path(args.dataset))
            print(f"📁 Dataset resolved: {args.dataset}")
        except FileNotFoundError as e:
            print(f"❌ Dataset resolution failed: {e}")
            sys.exit(1)
    
    # Resolve model path if provided
    if hasattr(args, 'model') and args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            full_model_path = resolver.project_root / model_path
            if full_model_path.exists():
                args.model = str(full_model_path)
                print(f"📁 Model resolved: {args.model}")
    
    return args


if __name__ == "__main__":
    # Test the enhanced path resolver
    print("🔍 Testing Enhanced Path Resolver")
    print("=" * 50)
    
    resolver = EnhancedPathResolver()
    print(f"Project root: {resolver.project_root}")
    
    # Test VCF resolution
    test_cases = [
        "clinvar_20250831.vcf.gz",  # Filename only
        "data/custom/test.vcf.gz",  # Relative path
    ]
    
    for test_case in test_cases:
        try:
            resolved = resolver.resolve_vcf_path(test_case)
            print(f"✅ {test_case} → {resolved}")
        except FileNotFoundError as e:
            print(f"❌ {test_case} → {e}")
    
    # Test FASTA resolution
    try:
        fasta_path = resolver.resolve_fasta_path("GRCh38.fa")
        print(f"✅ FASTA resolved: {fasta_path}")
    except Exception as e:
        print(f"❌ FASTA resolution: {e}")
    
    print("\n🎯 Enhanced path resolution ready for integration")
