"""Download and index genomic resources from Ensembl.

Provides functionality to:
- Download GTF and FASTA files from Ensembl FTP
- Verify checksums (if available)
- Decompress gzipped files
- Index FASTA files using pyfaidx
"""

import os
import gzip
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve
from urllib.error import URLError

from .config import load_config, filename


def _fetch(url: str, dest: Path, verbose: bool = True) -> Path:
    """Download a file from URL to destination.
    
    Parameters
    ----------
    url : str
        URL to download from
    dest : Path
        Destination file path
    verbose : bool
        Print progress messages
        
    Returns
    -------
    Path
        Path to downloaded file
        
    Raises
    ------
    URLError
        If download fails
    """
    if verbose:
        print(f"[download] Fetching {url}")
        print(f"[download] Destination: {dest}")
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urlretrieve(url, dest)
        if verbose:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"[download] Downloaded {size_mb:.1f} MB")
        return dest
    except URLError as e:
        raise URLError(f"Failed to download {url}: {e}")


def _gunzip(gz_path: Path, out_path: Optional[Path] = None, 
            remove_gz: bool = True, verbose: bool = True) -> Path:
    """Decompress a gzipped file.
    
    Parameters
    ----------
    gz_path : Path
        Path to .gz file
    out_path : Path, optional
        Output path (defaults to gz_path without .gz extension)
    remove_gz : bool
        Remove .gz file after decompression
    verbose : bool
        Print progress messages
        
    Returns
    -------
    Path
        Path to decompressed file
    """
    if out_path is None:
        out_path = gz_path.with_suffix('')  # Remove .gz
    
    if verbose:
        print(f"[download] Decompressing {gz_path.name}")
    
    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    if remove_gz:
        gz_path.unlink()
        if verbose:
            print(f"[download] Removed {gz_path.name}")
    
    if verbose:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"[download] Decompressed to {out_path.name} ({size_mb:.1f} MB)")
    
    return out_path


def ensure_faidx(fasta_path: Path, force: bool = False, verbose: bool = True) -> Path:
    """Create FASTA index using pyfaidx.
    
    Parameters
    ----------
    fasta_path : Path
        Path to FASTA file
    force : bool
        Force re-indexing even if .fai exists
    verbose : bool
        Print progress messages
        
    Returns
    -------
    Path
        Path to .fai index file
    """
    fai_path = Path(str(fasta_path) + ".fai")
    
    if fai_path.exists() and not force:
        if verbose:
            print(f"[download] Index already exists: {fai_path.name}")
        return fai_path
    
    if verbose:
        print(f"[download] Indexing FASTA with pyfaidx...")
    
    try:
        from pyfaidx import Faidx
        Faidx(str(fasta_path))
        if verbose:
            print(f"[download] Created index: {fai_path.name}")
        return fai_path
    except ImportError:
        if verbose:
            print("[download] Warning: pyfaidx not available, skipping FASTA indexing")
            print("[download] Install with: pip install pyfaidx")
        return None
    except Exception as e:
        if verbose:
            print(f"[download] Warning: Failed to index FASTA: {e}")
        return None


def compute_checksum(file_path: Path, algorithm: str = "md5") -> str:
    """Compute checksum of a file.
    
    Parameters
    ----------
    file_path : Path
        Path to file
    algorithm : str
        Hash algorithm ('md5', 'sha256')
        
    Returns
    -------
    str
        Hexadecimal checksum
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def verify_checksum(file_path: Path, expected_checksum: str, 
                   algorithm: str = "md5", verbose: bool = True) -> bool:
    """Verify file checksum.
    
    Parameters
    ----------
    file_path : Path
        Path to file
    expected_checksum : str
        Expected checksum value
    algorithm : str
        Hash algorithm ('md5', 'sha256')
    verbose : bool
        Print progress messages
        
    Returns
    -------
    bool
        True if checksum matches, False otherwise
    """
    if verbose:
        print(f"[download] Verifying {algorithm} checksum...")
    
    actual = compute_checksum(file_path, algorithm)
    matches = actual.lower() == expected_checksum.lower()
    
    if verbose:
        if matches:
            print(f"[download] ✓ Checksum verified")
        else:
            print(f"[download] ✗ Checksum mismatch!")
            print(f"  Expected: {expected_checksum}")
            print(f"  Actual:   {actual}")
    
    return matches


def fetch_ensembl(
    *,
    build: str = "GRCh38",
    release: str = "112",
    fetch_gtf: bool = True,
    fetch_fasta: bool = True,
    index_fasta: bool = True,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Download and prepare Ensembl GTF and FASTA files.
    
    Parameters
    ----------
    build : str
        Genome build (e.g., 'GRCh38', 'GRCh37')
    release : str
        Ensembl release version (e.g., '112', '106')
    fetch_gtf : bool
        Download GTF file
    fetch_fasta : bool
        Download FASTA file
    index_fasta : bool
        Create FASTA index (.fai)
    force : bool
        Force re-download even if files exist
    dry_run : bool
        Print what would be done without actually doing it
    verbose : bool
        Print progress messages
        
    Returns
    -------
    Tuple[Path, Path, Path]
        Paths to (GTF, FASTA, FASTA index) or None if not fetched
        
    Examples
    --------
    >>> fetch_ensembl(build="GRCh38", release="112", dry_run=True)
    [download] DRY RUN - would download:
      GTF: https://ftp.ensembl.org/pub/release-112/gtf/homo_sapiens/Homo_sapiens.GRCh38.112.gtf.gz
      FASTA: https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
    """
    # Load config with overrides
    cfg = load_config()
    cfg.build = build
    cfg.release = release
    
    # Get build-specific configuration
    build_cfg = cfg.builds[build]
    base_url = build_cfg["ensembl_base"].format(release=release)
    
    # Prepare file paths
    gtf_filename = filename("gtf", cfg)
    fasta_filename = filename("fasta", cfg)
    
    gtf_url = f"{base_url}/{build_cfg['subpaths']['gtf']}/{gtf_filename}.gz"
    fasta_url = f"{base_url}/{build_cfg['subpaths']['fasta']}/{fasta_filename}.gz"
    
    # Destination directory - use build-specific subdirectory
    # This ensures GRCh37 and GRCh38 files don't conflict
    dest_dir = cfg.data_root / build
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    gtf_path = dest_dir / gtf_filename
    fasta_path = dest_dir / fasta_filename
    fai_path = Path(str(fasta_path) + ".fai") if index_fasta else None
    
    if dry_run:
        print("[download] DRY RUN - would download:")
        if fetch_gtf:
            print(f"  GTF: {gtf_url}")
            print(f"    → {gtf_path}")
        if fetch_fasta:
            print(f"  FASTA: {fasta_url}")
            print(f"    → {fasta_path}")
        if index_fasta:
            print(f"  Index: {fai_path}")
        return (gtf_path if fetch_gtf else None, 
                fasta_path if fetch_fasta else None,
                fai_path if index_fasta else None)
    
    # Download GTF
    gtf_result = None
    if fetch_gtf:
        if gtf_path.exists() and not force:
            if verbose:
                print(f"[download] GTF already exists: {gtf_path}")
            gtf_result = gtf_path
        else:
            try:
                gz_path = dest_dir / f"{gtf_filename}.gz"
                _fetch(gtf_url, gz_path, verbose=verbose)
                gtf_result = _gunzip(gz_path, gtf_path, remove_gz=True, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"[download] Failed to download GTF: {e}")
                gtf_result = None
    
    # Download FASTA
    fasta_result = None
    fai_result = None
    if fetch_fasta:
        if fasta_path.exists() and not force:
            if verbose:
                print(f"[download] FASTA already exists: {fasta_path}")
            fasta_result = fasta_path
        else:
            try:
                gz_path = dest_dir / f"{fasta_filename}.gz"
                _fetch(fasta_url, gz_path, verbose=verbose)
                fasta_result = _gunzip(gz_path, fasta_path, remove_gz=True, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"[download] Failed to download FASTA: {e}")
                fasta_result = None
        
        # Index FASTA
        if fasta_result and index_fasta:
            fai_result = ensure_faidx(fasta_result, force=force, verbose=verbose)
    
    if verbose:
        print("\n[download] Summary:")
        print(f"  GTF:   {gtf_result or 'not fetched'}")
        print(f"  FASTA: {fasta_result or 'not fetched'}")
        if fai_result:
            print(f"  Index: {fai_result}")
    
    return gtf_result, fasta_result, fai_result
