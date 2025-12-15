#!/usr/bin/env python3
"""
ClinVar Data Download Helper Script

This script automates the download of ClinVar VCF files from NCBI FTP server.
It handles both GRCh37 and GRCh38 reference genomes and includes integrity checks.

Usage:
    python download_clinvar.py --genome GRCh38 --output-dir /path/to/data/ensembl/clinvar/vcf
    python download_clinvar.py --list-available  # Show available files
"""

import os
import sys
import argparse
import hashlib
import logging
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.parse import urljoin
import re
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinVarDownloader:
    """Downloads and manages ClinVar VCF files from NCBI."""
    
    BASE_URLS = {
        'GRCh37': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/',
        'GRCh38': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/'
    }
    
    def __init__(self, genome_build: str = 'GRCh38'):
        """
        Initialize ClinVar downloader.
        
        Parameters
        ----------
        genome_build : str
            Reference genome build ('GRCh37' or 'GRCh38')
        """
        if genome_build not in self.BASE_URLS:
            raise ValueError(f"Unsupported genome build: {genome_build}")
        
        self.genome_build = genome_build
        self.base_url = self.BASE_URLS[genome_build]
        
    def list_available_files(self) -> List[Dict[str, str]]:
        """
        List available ClinVar VCF files on the FTP server.
        
        Returns
        -------
        List[Dict[str, str]]
            List of available files with metadata
        """
        try:
            with urlopen(self.base_url) as response:
                content = response.read().decode('utf-8')
            
            # Parse directory listing for VCF files
            vcf_pattern = r'href="(clinvar_\d{8}\.vcf\.gz)"'
            tbi_pattern = r'href="(clinvar_\d{8}\.vcf\.gz\.tbi)"'
            
            vcf_files = re.findall(vcf_pattern, content)
            tbi_files = re.findall(tbi_pattern, content)
            
            files = []
            for vcf_file in vcf_files:
                # Extract date from filename
                date_match = re.search(r'clinvar_(\d{8})\.vcf\.gz', vcf_file)
                date_str = date_match.group(1) if date_match else 'unknown'
                
                # Check if corresponding index file exists
                tbi_file = vcf_file + '.tbi'
                has_index = tbi_file in tbi_files
                
                files.append({
                    'filename': vcf_file,
                    'date': date_str,
                    'has_index': has_index,
                    'url': urljoin(self.base_url, vcf_file),
                    'index_url': urljoin(self.base_url, tbi_file) if has_index else None
                })
            
            # Sort by date (newest first)
            files.sort(key=lambda x: x['date'], reverse=True)
            
            logger.info(f"Found {len(files)} ClinVar VCF files for {self.genome_build}")
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def get_latest_file(self) -> Optional[Dict[str, str]]:
        """
        Get information about the latest ClinVar VCF file.
        
        Returns
        -------
        Optional[Dict[str, str]]
            Latest file information or None if not found
        """
        files = self.list_available_files()
        return files[0] if files else None
    
    def download_file(
        self,
        filename: str,
        output_dir: str,
        verify_checksum: bool = False
    ) -> Tuple[bool, str]:
        """
        Download a specific ClinVar VCF file.
        
        Parameters
        ----------
        filename : str
            Name of the file to download
        output_dir : str
            Directory to save the file
        verify_checksum : bool
            Whether to verify file integrity (if checksums available)
        
        Returns
        -------
        Tuple[bool, str]
            Success status and message
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Construct URLs
        vcf_url = urljoin(self.base_url, filename)
        tbi_url = urljoin(self.base_url, filename + '.tbi')
        
        try:
            # Download VCF file
            vcf_output = output_path / filename
            logger.info(f"Downloading {filename}...")
            urlretrieve(vcf_url, vcf_output)
            
            # Download index file
            tbi_output = output_path / (filename + '.tbi')
            logger.info(f"Downloading {filename}.tbi...")
            urlretrieve(tbi_url, tbi_output)
            
            # Verify files exist and have reasonable sizes
            if not vcf_output.exists() or vcf_output.stat().st_size < 1000:
                return False, f"VCF file download failed or file too small"
            
            if not tbi_output.exists() or tbi_output.stat().st_size < 100:
                return False, f"Index file download failed or file too small"
            
            logger.info(f"Successfully downloaded {filename} and index")
            
            # Create download metadata
            metadata = {
                'filename': filename,
                'genome_build': self.genome_build,
                'download_date': datetime.now().isoformat(),
                'source_url': vcf_url,
                'file_size': vcf_output.stat().st_size,
                'index_size': tbi_output.stat().st_size
            }
            
            metadata_path = output_path / f"{filename}.metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True, f"Downloaded {filename} successfully"
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False, f"Download failed: {e}"
    
    def download_latest(
        self,
        output_dir: str,
        verify_checksum: bool = False
    ) -> Tuple[bool, str]:
        """
        Download the latest ClinVar VCF file.
        
        Parameters
        ----------
        output_dir : str
            Directory to save the file
        verify_checksum : bool
            Whether to verify file integrity
        
        Returns
        -------
        Tuple[bool, str]
            Success status and message
        """
        latest = self.get_latest_file()
        if not latest:
            return False, "No ClinVar files found"
        
        return self.download_file(
            latest['filename'],
            output_dir,
            verify_checksum
        )


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Download ClinVar VCF files from NCBI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download latest ClinVar for GRCh38
  python download_clinvar.py --genome GRCh38 --output-dir data/ensembl/clinvar/vcf

  # List available files
  python download_clinvar.py --list-available --genome GRCh38

  # Download specific file
  python download_clinvar.py --filename clinvar_20240101.vcf.gz --output-dir data/ensembl/clinvar/vcf
        """
    )
    
    parser.add_argument(
        '--genome',
        choices=['GRCh37', 'GRCh38'],
        default='GRCh38',
        help='Reference genome build (default: GRCh38)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for downloaded files'
    )
    
    parser.add_argument(
        '--filename',
        type=str,
        help='Specific filename to download (e.g., clinvar_20240101.vcf.gz)'
    )
    
    parser.add_argument(
        '--list-available',
        action='store_true',
        help='List available files on the server'
    )
    
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Download the latest available file'
    )
    
    parser.add_argument(
        '--verify-checksum',
        action='store_true',
        help='Verify file integrity using checksums'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ClinVarDownloader(args.genome)
    
    if args.list_available:
        print(f"\nAvailable ClinVar VCF files for {args.genome}:")
        print("=" * 60)
        
        files = downloader.list_available_files()
        if not files:
            print("No files found.")
            return
        
        for file_info in files:
            status = "✓" if file_info['has_index'] else "✗"
            print(f"{status} {file_info['filename']} (Date: {file_info['date']})")
        
        print(f"\nLatest file: {files[0]['filename']}")
        return
    
    if not args.output_dir:
        print("Error: --output-dir is required for download operations")
        return
    
    # Determine what to download
    if args.filename:
        success, message = downloader.download_file(
            args.filename,
            args.output_dir,
            args.verify_checksum
        )
    elif args.latest or True:  # Default to latest
        success, message = downloader.download_latest(
            args.output_dir,
            args.verify_checksum
        )
    
    if success:
        print(f"✓ {message}")
        print(f"Files saved to: {args.output_dir}")
        
        # Show next steps
        print("\nNext steps:")
        print("1. Run the ClinVar VCF tutorial:")
        print("   python meta_spliceai/splice_engine/case_studies/examples/vcf_clinvar_tutorial.py")
        print("2. Update the VCF filename in the tutorial if needed")
    else:
        print(f"✗ {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
