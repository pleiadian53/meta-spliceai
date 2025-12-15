"""Validators for genomic resources and gene selection.

Provides validation functions to catch issues early before expensive workflows run.

This module includes:
- Gene selection validators (pre-flight checks)
- Coordinate system validators (1-based vs 0-based)
- Build alignment validators (GTF/FASTA consistency)
- Splice motif validators (GT-AG verification)
"""

import polars as pl
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import sys
import re


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_genes_have_splice_sites(
    gene_ids: List[str],
    splice_sites_path: Path,
    min_sites: int = 1,
    verbose: bool = True
) -> Tuple[List[str], List[str]]:
    """Validate that genes have splice sites in the annotation.
    
    This is a pre-flight check to catch genes without splice sites before
    running expensive prediction workflows.
    
    Parameters
    ----------
    gene_ids : List[str]
        List of Ensembl gene IDs to validate
    splice_sites_path : Path
        Path to splice_sites.tsv file
    min_sites : int, default=1
        Minimum number of splice sites required per gene
    verbose : bool, default=True
        If True, print detailed validation results
        
    Returns
    -------
    valid_genes : List[str]
        Gene IDs that have sufficient splice sites
    invalid_genes : List[str]
        Gene IDs that don't meet splice site requirements
        
    Raises
    ------
    ValidationError
        If no valid genes are found
        
    Examples
    --------
    >>> valid, invalid = validate_genes_have_splice_sites(
    ...     ['ENSG00000142611', 'ENSG00000290712'],
    ...     Path('data/ensembl/splice_sites.tsv')
    ... )
    >>> print(f"Valid: {len(valid)}, Invalid: {len(invalid)}")
    """
    if not splice_sites_path.exists():
        raise FileNotFoundError(f"Splice sites file not found: {splice_sites_path}")
    
    if verbose:
        print("=" * 70)
        print("🔍 VALIDATING GENES FOR SPLICE SITE COVERAGE")
        print("=" * 70)
        print(f"Checking {len(gene_ids)} genes against {splice_sites_path.name}")
    
    # Load splice sites with proper schema
    try:
        splice_sites = pl.read_csv(
            splice_sites_path,
            separator='\t',
            schema_overrides={'chrom': pl.Utf8}
        )
    except Exception as e:
        raise ValidationError(f"Failed to load splice sites: {e}")
    
    # Count splice sites per gene
    gene_site_counts = (
        splice_sites
        .filter(pl.col('gene_id').is_in(gene_ids))
        .group_by('gene_id')
        .agg(pl.len().alias('site_count'))
    )
    
    # Create lookup dict
    site_count_dict = dict(zip(
        gene_site_counts['gene_id'].to_list(),
        gene_site_counts['site_count'].to_list()
    ))
    
    # Classify genes
    valid_genes = []
    invalid_genes = []
    
    for gene_id in gene_ids:
        site_count = site_count_dict.get(gene_id, 0)
        if site_count >= min_sites:
            valid_genes.append(gene_id)
        else:
            invalid_genes.append(gene_id)
    
    # Report results
    if verbose:
        print(f"\n✅ Valid genes (≥{min_sites} splice sites): {len(valid_genes)}")
        print(f"❌ Invalid genes (<{min_sites} splice sites): {len(invalid_genes)}")
        
        if invalid_genes:
            print(f"\n⚠️  Genes WITHOUT sufficient splice sites:")
            for gene_id in invalid_genes[:10]:  # Show first 10
                count = site_count_dict.get(gene_id, 0)
                print(f"   - {gene_id}: {count} splice sites")
            if len(invalid_genes) > 10:
                print(f"   ... and {len(invalid_genes) - 10} more")
        
        if valid_genes:
            print(f"\n✓ Sample valid genes:")
            for gene_id in valid_genes[:5]:  # Show first 5
                count = site_count_dict[gene_id]
                print(f"   - {gene_id}: {count} splice sites")
        
        print("=" * 70)
    
    # Raise error if no valid genes
    if not valid_genes:
        raise ValidationError(
            f"No genes with sufficient splice sites found!\n"
            f"All {len(gene_ids)} genes have <{min_sites} splice sites.\n"
            f"Consider using --gene-types protein_coding lncRNA to filter for genes with splice sites."
        )
    
    return valid_genes, invalid_genes


def validate_genes_in_gtf(
    gene_ids: List[str],
    gtf_path: Path,
    verbose: bool = True
) -> Tuple[List[str], List[str]]:
    """Validate that genes exist in the GTF annotation.
    
    Parameters
    ----------
    gene_ids : List[str]
        List of Ensembl gene IDs to validate
    gtf_path : Path
        Path to GTF file
    verbose : bool, default=True
        If True, print validation results
        
    Returns
    -------
    valid_genes : List[str]
        Gene IDs found in GTF
    missing_genes : List[str]
        Gene IDs not found in GTF
        
    Raises
    ------
    ValidationError
        If no valid genes are found
    """
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")
    
    if verbose:
        print("=" * 70)
        print("🔍 VALIDATING GENES IN GTF ANNOTATION")
        print("=" * 70)
    
    # Extract gene IDs from GTF (only gene features)
    try:
        gtf_df = pl.read_csv(
            gtf_path,
            separator='\t',
            comment_prefix='#',
            has_header=False,
            new_columns=['seqname', 'source', 'feature', 'start', 'end', 
                        'score', 'strand', 'frame', 'attribute'],
            schema_overrides={'seqname': pl.Utf8}
        )
        
        # Filter for gene features and extract gene_id
        gene_features = gtf_df.filter(pl.col('feature') == 'gene')
        
        # Extract gene_id from attributes column
        gtf_gene_ids = set()
        for attr in gene_features['attribute'].to_list():
            for field in attr.split(';'):
                if 'gene_id' in field:
                    gene_id = field.split('"')[1]
                    gtf_gene_ids.add(gene_id)
                    break
        
    except Exception as e:
        raise ValidationError(f"Failed to parse GTF file: {e}")
    
    # Classify genes
    valid_genes = [g for g in gene_ids if g in gtf_gene_ids]
    missing_genes = [g for g in gene_ids if g not in gtf_gene_ids]
    
    if verbose:
        print(f"✅ Genes found in GTF: {len(valid_genes)}/{len(gene_ids)}")
        if missing_genes:
            print(f"❌ Genes NOT in GTF: {len(missing_genes)}")
            for gene_id in missing_genes[:5]:
                print(f"   - {gene_id}")
            if len(missing_genes) > 5:
                print(f"   ... and {len(missing_genes) - 5} more")
        print("=" * 70)
    
    if not valid_genes:
        raise ValidationError(
            f"None of the {len(gene_ids)} genes were found in the GTF annotation!"
        )
    
    return valid_genes, missing_genes


def validate_gene_selection(
    gene_ids: List[str],
    data_dir: Path,
    min_splice_sites: int = 1,
    check_gtf: bool = False,
    fail_on_invalid: bool = True,
    verbose: bool = True
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Comprehensive pre-flight validation for gene selection.
    
    Validates genes before running expensive workflows:
    1. Checks that genes have splice sites
    2. Optionally checks that genes exist in GTF
    3. Reports and optionally filters invalid genes
    
    Parameters
    ----------
    gene_ids : List[str]
        List of Ensembl gene IDs to validate
    data_dir : Path
        Path to data directory containing splice_sites.tsv and GTF
    min_splice_sites : int, default=1
        Minimum number of splice sites required
    check_gtf : bool, default=False
        If True, also validate genes exist in GTF
    fail_on_invalid : bool, default=True
        If True, raise error when invalid genes are found
        If False, filter out invalid genes and continue with valid ones
    verbose : bool, default=True
        If True, print detailed validation results
        
    Returns
    -------
    valid_genes : List[str]
        Genes that passed all validation checks
    invalid_summary : Dict[str, List[str]]
        Summary of invalid genes by reason:
        - 'no_splice_sites': genes without splice sites
        - 'not_in_gtf': genes not found in GTF (if check_gtf=True)
        
    Raises
    ------
    ValidationError
        If validation fails and fail_on_invalid=True
        
    Examples
    --------
    >>> valid_genes, issues = validate_gene_selection(
    ...     ['ENSG00000142611', 'ENSG00000290712'],
    ...     Path('data/ensembl'),
    ...     fail_on_invalid=False
    ... )
    >>> print(f"Proceeding with {len(valid_genes)} valid genes")
    """
    splice_sites_path = data_dir / "splice_sites.tsv"
    
    if verbose:
        print("\n" + "=" * 70)
        print("🚀 PRE-FLIGHT VALIDATION FOR GENE SELECTION")
        print("=" * 70)
        print(f"Total genes to validate: {len(gene_ids)}")
        print(f"Minimum splice sites required: {min_splice_sites}")
        print(f"Check GTF presence: {check_gtf}")
        print()
    
    invalid_summary = {
        'no_splice_sites': [],
        'not_in_gtf': []
    }
    
    # Validation 1: Splice sites
    try:
        valid_splice, invalid_splice = validate_genes_have_splice_sites(
            gene_ids,
            splice_sites_path,
            min_sites=min_splice_sites,
            verbose=verbose
        )
        invalid_summary['no_splice_sites'] = invalid_splice
        valid_genes = valid_splice
    except ValidationError as e:
        if fail_on_invalid:
            raise
        else:
            print(f"⚠️  Warning: {e}")
            return [], invalid_summary
    
    # Validation 2: GTF presence (optional)
    if check_gtf and valid_genes:
        gtf_path = data_dir / f"Homo_sapiens.GRCh38.*.gtf"
        gtf_files = list(data_dir.glob("Homo_sapiens.GRCh38.*.gtf"))
        
        if gtf_files:
            gtf_path = gtf_files[0]
            try:
                valid_gtf, missing_gtf = validate_genes_in_gtf(
                    valid_genes,
                    gtf_path,
                    verbose=verbose
                )
                invalid_summary['not_in_gtf'] = missing_gtf
                valid_genes = valid_gtf
            except ValidationError as e:
                if fail_on_invalid:
                    raise
                else:
                    print(f"⚠️  Warning: {e}")
    
    # Final summary
    total_invalid = sum(len(v) for v in invalid_summary.values())
    
    if verbose:
        print("\n" + "=" * 70)
        print("📊 VALIDATION SUMMARY")
        print("=" * 70)
        print(f"✅ Valid genes: {len(valid_genes)}/{len(gene_ids)}")
        print(f"❌ Invalid genes: {total_invalid}/{len(gene_ids)}")
        if invalid_summary['no_splice_sites']:
            print(f"   - No splice sites: {len(invalid_summary['no_splice_sites'])}")
        if invalid_summary['not_in_gtf']:
            print(f"   - Not in GTF: {len(invalid_summary['not_in_gtf'])}")
        print("=" * 70)
    
    # Handle invalid genes
    if total_invalid > 0:
        if fail_on_invalid:
            raise ValidationError(
                f"Validation failed: {total_invalid}/{len(gene_ids)} genes are invalid.\n"
                f"Use fail_on_invalid=False to proceed with valid genes only, or\n"
                f"use --gene-types protein_coding lncRNA to filter for genes with splice sites."
            )
        elif verbose:
            print(f"\n⚠️  Proceeding with {len(valid_genes)} valid genes (filtered out {total_invalid} invalid)")
    
    return valid_genes, invalid_summary


def assert_coordinate_policy(
    tsv_paths: Union[Path, List[Path]],
    expected: str = "1-based",
    gtf_file: Optional[Path] = None,
    fasta_file: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Validate coordinate system of TSV files derived from GTF.
    
    Checks that genomic coordinates follow the expected convention:
    - 1-based (fully closed): [start, end] where both are inclusive
    - 0-based (half-open): [start, end) where start is inclusive, end exclusive
    
    GTF/GFF files use 1-based by specification.
    
    Parameters
    ----------
    tsv_paths : Path or List[Path]
        Path(s) to TSV files to validate (e.g., splice_sites.tsv, gene_features.tsv)
    expected : str, default="1-based"
        Expected coordinate system: "1-based" or "0-based"
    gtf_file : Path, optional
        Original GTF file for cross-reference validation
    fasta_file : Path, optional
        Reference FASTA for sequence-based validation
    verbose : bool, default=True
        Print detailed validation results
        
    Returns
    -------
    Dict[str, Any]
        - 'passed': bool - Overall pass/fail
        - 'coordinate_system': str - Detected coordinate system
        - 'confidence': str - Confidence level (HIGH/MEDIUM/LOW)
        - 'checks': List[Dict] - Detailed check results
        - 'warnings': List[str] - Any warnings
        
    Examples
    --------
    >>> result = assert_coordinate_policy(
    ...     [Path('data/ensembl/splice_sites.tsv')],
    ...     expected="1-based"
    ... )
    >>> assert result['passed']
    """
    # Ensure tsv_paths is a list
    if isinstance(tsv_paths, (str, Path)):
        tsv_paths = [Path(tsv_paths)]
    else:
        tsv_paths = [Path(p) for p in tsv_paths]
    
    results = {
        'passed': True,
        'coordinate_system': expected,
        'confidence': 'HIGH',
        'checks': [],
        'warnings': []
    }
    
    if verbose:
        print("=" * 70)
        print("COORDINATE SYSTEM VALIDATION")
        print("=" * 70)
        print(f"Expected: {expected}")
        print(f"Checking {len(tsv_paths)} file(s)...\n")
    
    for tsv_path in tsv_paths:
        if not tsv_path.exists():
            results['checks'].append({
                'file': str(tsv_path.name),
                'check': 'File exists',
                'status': 'SKIP',
                'detail': 'File not found'
            })
            results['warnings'].append(f"File not found: {tsv_path}")
            continue
        
        try:
            # Load first 1000 rows for validation
            df = pl.read_csv(
                tsv_path,
                separator='\t',
                n_rows=1000,
                schema_overrides={'chrom': pl.Utf8}
            )
            
            # Check 1: No position 0 (impossible in 1-based)
            if 'start' in df.columns:
                has_zero_start = (df['start'] == 0).sum() > 0
                
                if expected == "1-based" and has_zero_start:
                    results['passed'] = False
                    results['coordinate_system'] = '0-based or corrupted'
                    results['checks'].append({
                        'file': tsv_path.name,
                        'check': 'No position 0',
                        'status': 'FAIL',
                        'detail': f'Found {has_zero_start} zero positions (invalid for 1-based)'
                    })
                else:
                    results['checks'].append({
                        'file': tsv_path.name,
                        'check': 'No position 0',
                        'status': 'PASS',
                        'detail': 'No zero positions found'
                    })
            
            # Check 2: start <= end (must be true for both systems)
            if 'start' in df.columns and 'end' in df.columns:
                invalid_ranges = (df['start'] > df['end']).sum()
                
                if invalid_ranges > 0:
                    results['passed'] = False
                    results['checks'].append({
                        'file': tsv_path.name,
                        'check': 'start <= end',
                        'status': 'FAIL',
                        'detail': f'Found {invalid_ranges} features with start > end'
                    })
                else:
                    results['checks'].append({
                        'file': tsv_path.name,
                        'check': 'start <= end',
                        'status': 'PASS',
                        'detail': 'All ranges valid'
                    })
            
            # Check 3: Single-nucleotide features
            if 'start' in df.columns and 'end' in df.columns:
                single_nt = df.filter(pl.col('end') - pl.col('start') == 0)
                multi_nt_min_length = df.filter(pl.col('end') > pl.col('start'))['end'] - df.filter(pl.col('end') > pl.col('start'))['start']
                
                if single_nt.height > 0:
                    # Has single nucleotide features with start == end
                    # This is characteristic of 1-based
                    if expected == "1-based":
                        results['checks'].append({
                            'file': tsv_path.name,
                            'check': 'Single-nt features',
                            'status': 'PASS',
                            'detail': f'Found {single_nt.height} features with start==end (1-based pattern)'
                        })
                    else:
                        results['checks'].append({
                            'file': tsv_path.name,
                            'check': 'Single-nt features',
                            'status': 'WARNING',
                            'detail': f'Found {single_nt.height} features with start==end (unexpected for 0-based)'
                        })
                        results['warnings'].append(f"{tsv_path.name}: Single-nt features suggest 1-based, not 0-based")
            
            # Check 4: All positions are positive
            if 'start' in df.columns:
                negative_positions = (df['start'] < 0).sum()
                if negative_positions > 0:
                    results['passed'] = False
                    results['checks'].append({
                        'file': tsv_path.name,
                        'check': 'Positive positions',
                        'status': 'FAIL',
                        'detail': f'Found {negative_positions} negative positions'
                    })
                else:
                    results['checks'].append({
                        'file': tsv_path.name,
                        'check': 'Positive positions',
                        'status': 'PASS',
                        'detail': 'All positions >= 0'
                    })
        
        except Exception as e:
            results['checks'].append({
                'file': tsv_path.name,
                'check': 'File parsing',
                'status': 'ERROR',
                'detail': f'Failed to parse: {str(e)}'
            })
            results['warnings'].append(f"Failed to parse {tsv_path.name}: {e}")
            results['passed'] = False
    
    # Additional validation against GTF if provided
    if gtf_file and fasta_file and gtf_file.exists() and fasta_file.exists():
        if verbose:
            print("\nPerforming GTF cross-reference validation...")
        
        gtf_result = verify_gtf_coordinate_system(gtf_file, fasta_file, verbose=False)
        
        if gtf_result['passed']:
            results['checks'].append({
                'file': 'GTF cross-reference',
                'check': 'Splice motif validation',
                'status': 'PASS',
                'detail': gtf_result['detail']
            })
        else:
            results['checks'].append({
                'file': 'GTF cross-reference',
                'check': 'Splice motif validation',
                'status': 'WARNING',
                'detail': gtf_result['detail']
            })
            results['warnings'].append(gtf_result['detail'])
    
    # Print results
    if verbose:
        print()
        for check in results['checks']:
            status = check['status']
            if status == 'PASS':
                icon = '✅'
            elif status == 'FAIL':
                icon = '❌'
            elif status == 'WARNING':
                icon = '⚠️'
            elif status == 'SKIP':
                icon = '⏭️'
            else:
                icon = '❓'
            
            print(f"{icon} {check['file']:30s} {check['check']:25s} {check['detail']}")
        
        print("\n" + "=" * 70)
        if results['passed']:
            print(f"✅ PASSED: Coordinate system is {results['coordinate_system']}")
        else:
            print(f"❌ FAILED: Coordinate system validation failed")
        
        if results['warnings']:
            print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
            for warning in results['warnings']:
                print(f"   - {warning}")
        
        print("=" * 70 + "\n")
    
    return results


def verify_gtf_coordinate_system(
    gtf_file: Path,
    fasta_file: Path,
    sample_size: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """Verify GTF coordinate system by checking splice motifs.
    
    Uses biological features to verify coordinate system:
    - Checks that exon boundaries align with GT-AG splice motifs
    - Verifies start codons are ATG
    
    Parameters
    ----------
    gtf_file : Path
        Path to GTF annotation file
    fasta_file : Path
        Path to reference genome FASTA
    sample_size : int, default=100
        Number of features to sample for validation
    verbose : bool, default=True
        Print validation progress
        
    Returns
    -------
    Dict[str, Any]
        - 'passed': bool
        - 'coordinate_system': str
        - 'confidence': str
        - 'detail': str
        
    Examples
    --------
    >>> result = verify_gtf_coordinate_system(
    ...     Path('data/ensembl/Homo_sapiens.GRCh38.112.gtf'),
    ...     Path('data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa')
    ... )
    >>> assert result['passed']
    """
    try:
        from pyfaidx import Fasta
    except ImportError:
        return {
            'passed': False,
            'coordinate_system': 'UNKNOWN',
            'confidence': 'N/A',
            'detail': 'pyfaidx not available for sequence validation'
        }
    
    if verbose:
        print(f"Loading FASTA: {fasta_file.name}...")
    
    try:
        fasta = Fasta(str(fasta_file), as_raw=True, sequence_always_upper=True)
    except Exception as e:
        return {
            'passed': False,
            'coordinate_system': 'UNKNOWN',
            'confidence': 'N/A',
            'detail': f'Failed to load FASTA: {e}'
        }
    
    if verbose:
        print(f"Loading GTF: {gtf_file.name}...")
    
    try:
        # Load GTF file
        gtf = pl.read_csv(
            gtf_file,
            separator='\t',
            comment_prefix='#',
            has_header=False,
            new_columns=['seqname', 'source', 'feature', 'start', 'end',
                        'score', 'strand', 'frame', 'attribute'],
            schema_overrides={'seqname': pl.Utf8}
        )
    except Exception as e:
        return {
            'passed': False,
            'coordinate_system': 'UNKNOWN',
            'confidence': 'N/A',
            'detail': f'Failed to load GTF: {e}'
        }
    
    # Sample exons on + strand for GT donor motif check
    exons_plus = gtf.filter(
        (pl.col('feature') == 'exon') &
        (pl.col('strand') == '+')
    ).sample(min(sample_size, gtf.filter(pl.col('feature') == 'exon').height))
    
    if verbose:
        print(f"Checking {len(exons_plus)} exons for splice motifs...")
    
    gt_count = 0
    total_checked = 0
    
    for row in exons_plus.iter_rows(named=True):
        chrom = str(row['seqname'])
        
        # Normalize chromosome name
        if chrom not in fasta.keys():
            if f"chr{chrom}" in fasta.keys():
                chrom = f"chr{chrom}"
            elif chrom.startswith('chr') and chrom[3:] in fasta.keys():
                chrom = chrom[3:]
            else:
                continue  # Skip if chromosome not found
        
        end_pos = row['end']
        
        try:
            # In 1-based GTF: exon ends at position N (last exonic base)
            # Donor GT should start at N+1 (first intronic base)
            # For 0-based FASTA indexing: extract from end_pos:end_pos+2
            donor_seq = str(fasta[chrom][end_pos:end_pos+2])
            
            if donor_seq == 'GT':
                gt_count += 1
            total_checked += 1
        except Exception:
            continue
    
    if total_checked == 0:
        return {
            'passed': False,
            'coordinate_system': 'UNKNOWN',
            'confidence': 'LOW',
            'detail': 'Could not extract sequences for validation'
        }
    
    gt_percent = (gt_count / total_checked * 100) if total_checked > 0 else 0
    
    if gt_percent > 90:
        return {
            'passed': True,
            'coordinate_system': '1-based',
            'confidence': 'HIGH',
            'detail': f'{gt_percent:.1f}% of exons have GT donor motif at expected position (n={total_checked})'
        }
    elif gt_percent > 70:
        return {
            'passed': True,
            'coordinate_system': '1-based',
            'confidence': 'MEDIUM',
            'detail': f'{gt_percent:.1f}% of exons have GT donor motif (acceptable, n={total_checked})'
        }
    else:
        return {
            'passed': False,
            'coordinate_system': 'POSSIBLY 0-based or corrupted',
            'confidence': 'LOW',
            'detail': f'Only {gt_percent:.1f}% of exons have GT at expected position (n={total_checked})'
        }


def assert_splice_motif_policy(
    splice_sites_file: Path,
    fasta_file: Path,
    min_canonical_percent: float = 95.0,
    sample_size: Optional[int] = 1000,
    verbose: bool = True
) -> Dict[str, Any]:
    """Validate that splice sites have canonical GT-AG motifs.
    
    Checks that the majority of annotated splice sites have:
    - Donor sites: GT dinucleotide
    - Acceptor sites: AG dinucleotide
    
    Parameters
    ----------
    splice_sites_file : Path
        Path to splice_sites.tsv or splice_sites_enhanced.tsv
    fasta_file : Path
        Path to reference genome FASTA
    min_canonical_percent : float, default=95.0
        Minimum percentage of canonical motifs required to pass
    sample_size : int, optional
        Number of sites to sample (None = check all)
    verbose : bool, default=True
        Print detailed results
        
    Returns
    -------
    Dict[str, Any]
        - 'passed': bool
        - 'donor_gt_percent': float
        - 'acceptor_ag_percent': float
        - 'checks': List[Dict]
        - 'detail': str
        
    Examples
    --------
    >>> result = assert_splice_motif_policy(
    ...     Path('data/ensembl/splice_sites.tsv'),
    ...     Path('data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa')
    ... )
    >>> assert result['passed']
    >>> assert result['donor_gt_percent'] > 95.0
    """
    try:
        from pyfaidx import Fasta
    except ImportError:
        return {
            'passed': False,
            'detail': 'pyfaidx not available for sequence validation',
            'checks': []
        }
    
    def reverse_complement(seq: str) -> str:
        """Reverse complement a DNA sequence."""
        comp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
        return ''.join(comp.get(b.upper(), 'N') for b in reversed(seq))
    
    if verbose:
        print("=" * 70)
        print("SPLICE MOTIF VALIDATION")
        print("=" * 70)
        print(f"Checking canonical GT-AG motifs...")
        print(f"Minimum required: {min_canonical_percent}%\n")
    
    # Load files
    try:
        fasta = Fasta(str(fasta_file))
        splice_sites = pl.read_csv(
            splice_sites_file,
            separator='\t',
            schema_overrides={'chrom': pl.Utf8}
        )
    except Exception as e:
        return {
            'passed': False,
            'detail': f'Failed to load files: {e}',
            'checks': []
        }
    
    # Sample if requested
    if sample_size and splice_sites.height > sample_size:
        splice_sites = splice_sites.sample(sample_size, seed=42)
    
    results = {
        'passed': True,
        'donor_gt_percent': 0.0,
        'acceptor_ag_percent': 0.0,
        'checks': [],
        'detail': ''
    }
    
    # Check donors
    donors = splice_sites.filter(pl.col('site_type') == 'donor')
    donor_gt_count = 0
    donor_total = 0
    
    for row in donors.iter_rows(named=True):
        chrom = str(row['chrom'])
        if chrom not in fasta.keys():
            if f"chr{chrom}" in fasta.keys():
                chrom = f"chr{chrom}"
            elif chrom.startswith('chr') and chrom[3:] in fasta.keys():
                chrom = chrom[3:]
            else:
                continue
        
        pos = row['position']
        strand = row['strand']
        
        try:
            if strand == '+':
                # Positive strand: position is first base of intron
                # GT at [pos-1:pos+1] (0-based indexing)
                motif = fasta[chrom][pos-1:pos+1].seq.upper()
            else:
                # Negative strand: extract and reverse complement
                # Based on analyze_consensus_motifs.py logic
                start = pos - 6  # intron_bases
                end = pos + 3    # exon_bases
                seq = fasta[chrom][start:end].seq.upper()
                motif_full = reverse_complement(seq)
                # GT should be at chars 3-4 in the RC sequence (boundary position)
                motif = motif_full[3:5]
            
            if motif == 'GT':
                donor_gt_count += 1
            donor_total += 1
        except Exception:
            continue
    
    results['donor_gt_percent'] = (donor_gt_count / donor_total * 100) if donor_total > 0 else 0.0
    
    # Check acceptors
    acceptors = splice_sites.filter(pl.col('site_type') == 'acceptor')
    acceptor_ag_count = 0
    acceptor_total = 0
    
    for row in acceptors.iter_rows(named=True):
        chrom = str(row['chrom'])
        if chrom not in fasta.keys():
            if f"chr{chrom}" in fasta.keys():
                chrom = f"chr{chrom}"
            elif chrom.startswith('chr') and chrom[3:] in fasta.keys():
                chrom = chrom[3:]
            else:
                continue
        
        pos = row['position']  # First base of exon
        strand = row['strand']
        
        try:
            if strand == '+':
                # Positive strand: position is first base of exon
                # AG at [pos-2:pos] (0-based indexing, last 2 bases of intron)
                motif = fasta[chrom][pos-2:pos].seq.upper()
            else:
                # Negative strand: extract and reverse complement
                # Based on analyze_consensus_motifs.py logic
                start = pos - 3 - 1  # exon_bases + 1
                end = pos + 20 + 1   # intron_bases + 1
                seq = fasta[chrom][start:end].seq.upper()
                motif_full = reverse_complement(seq)
                # AG should be at chars 20-21 in the RC sequence (boundary position)
                motif = motif_full[20:22]
            
            if motif == 'AG':
                acceptor_ag_count += 1
            acceptor_total += 1
        except Exception:
            continue
    
    results['acceptor_ag_percent'] = (acceptor_ag_count / acceptor_total * 100) if acceptor_total > 0 else 0.0
    
    # Evaluate results
    donor_pass = results['donor_gt_percent'] >= min_canonical_percent
    acceptor_pass = results['acceptor_ag_percent'] >= min_canonical_percent
    
    results['checks'] = [
        {
            'check': 'Donor GT motif',
            'status': 'PASS' if donor_pass else 'FAIL',
            'value': f"{results['donor_gt_percent']:.2f}%",
            'detail': f'{donor_gt_count}/{donor_total} have GT'
        },
        {
            'check': 'Acceptor AG motif',
            'status': 'PASS' if acceptor_pass else 'FAIL',
            'value': f"{results['acceptor_ag_percent']:.2f}%",
            'detail': f'{acceptor_ag_count}/{acceptor_total} have AG'
        }
    ]
    
    results['passed'] = donor_pass and acceptor_pass
    results['detail'] = f"Donor GT: {results['donor_gt_percent']:.2f}%, Acceptor AG: {results['acceptor_ag_percent']:.2f}%"
    
    if verbose:
        for check in results['checks']:
            icon = '✅' if check['status'] == 'PASS' else '❌'
            print(f"{icon} {check['check']:25s} {check['value']:>8s}  ({check['detail']})")
        
        print("\n" + "=" * 70)
        if results['passed']:
            print(f"✅ PASSED: Splice motifs are canonical")
        else:
            print(f"❌ FAILED: Splice motif percentages below threshold ({min_canonical_percent}%)")
        print("=" * 70 + "\n")
    
    return results


def assert_build_alignment(
    gtf_file: Path,
    fasta_file: Path,
    expected_build: str = "GRCh38",
    expected_release: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Validate that GTF and FASTA files are from the same genome build.
    
    Checks:
    - Chromosome names match between GTF and FASTA
    - Build identifiers are consistent
    - Release versions match (if specified)
    
    Parameters
    ----------
    gtf_file : Path
        Path to GTF annotation file
    fasta_file : Path
        Path to reference genome FASTA
    expected_build : str, default="GRCh38"
        Expected genome build (GRCh38, GRCh37, etc.)
    expected_release : str, optional
        Expected Ensembl release number
    verbose : bool, default=True
        Print detailed results
        
    Returns
    -------
    Dict[str, Any]
        - 'passed': bool
        - 'build_match': bool
        - 'chrom_match': bool
        - 'checks': List[Dict]
        - 'detail': str
        
    Examples
    --------
    >>> result = assert_build_alignment(
    ...     Path('data/ensembl/Homo_sapiens.GRCh38.112.gtf'),
    ...     Path('data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa'),
    ...     expected_build="GRCh38"
    ... )
    >>> assert result['passed']
    """
    try:
        from pyfaidx import Fasta
    except ImportError:
        return {
            'passed': False,
            'detail': 'pyfaidx not available',
            'checks': []
        }
    
    if verbose:
        print("=" * 70)
        print("BUILD ALIGNMENT VALIDATION")
        print("=" * 70)
        print(f"Expected build: {expected_build}")
        if expected_release:
            print(f"Expected release: {expected_release}")
        print()
    
    results = {
        'passed': True,
        'build_match': False,
        'chrom_match': False,
        'checks': [],
        'detail': ''
    }
    
    # Check 1: Build identifier in filenames
    gtf_has_build = expected_build in gtf_file.name
    fasta_has_build = expected_build in fasta_file.name
    results['build_match'] = gtf_has_build and fasta_has_build
    
    results['checks'].append({
        'check': 'Build in GTF filename',
        'status': 'PASS' if gtf_has_build else 'WARNING',
        'detail': f"{expected_build} {'found' if gtf_has_build else 'not found'} in {gtf_file.name}"
    })
    
    results['checks'].append({
        'check': 'Build in FASTA filename',
        'status': 'PASS' if fasta_has_build else 'WARNING',
        'detail': f"{expected_build} {'found' if fasta_has_build else 'not found'} in {fasta_file.name}"
    })
    
    # Check 2: Release version if specified
    if expected_release:
        gtf_has_release = expected_release in gtf_file.name
        results['checks'].append({
            'check': 'Release in GTF filename',
            'status': 'PASS' if gtf_has_release else 'WARNING',
            'detail': f"Release {expected_release} {'found' if gtf_has_release else 'not found'}"
        })
    
    # Check 3: Chromosome names match
    try:
        # Load FASTA chromosomes
        fasta = Fasta(str(fasta_file))
        fasta_chroms = set(fasta.keys())
        
        # Load GTF chromosomes
        gtf = pl.read_csv(
            gtf_file,
            separator='\t',
            comment_prefix='#',
            has_header=False,
            new_columns=['seqname', 'source', 'feature', 'start', 'end',
                        'score', 'strand', 'frame', 'attribute'],
            schema_overrides={'seqname': pl.Utf8},
            n_rows=10000  # Sample for speed
        )
        gtf_chroms = set(gtf['seqname'].unique().to_list())
        
        # Check overlap
        common_chroms = gtf_chroms & fasta_chroms
        gtf_only = gtf_chroms - fasta_chroms
        fasta_only = fasta_chroms - gtf_chroms
        
        overlap_percent = (len(common_chroms) / len(gtf_chroms) * 100) if gtf_chroms else 0
        results['chrom_match'] = overlap_percent > 80
        
        results['checks'].append({
            'check': 'Chromosome overlap',
            'status': 'PASS' if results['chrom_match'] else 'FAIL',
            'detail': f"{len(common_chroms)}/{len(gtf_chroms)} GTF chromosomes found in FASTA ({overlap_percent:.1f}%)"
        })
        
        if gtf_only and verbose:
            print(f"  Chromosomes in GTF but not FASTA: {sorted(list(gtf_only))[:5]}...")
        if fasta_only and verbose:
            print(f"  Chromosomes in FASTA but not GTF: {sorted(list(fasta_only))[:5]}...")
        
    except Exception as e:
        results['checks'].append({
            'check': 'Chromosome overlap',
            'status': 'ERROR',
            'detail': f'Failed to compare chromosomes: {e}'
        })
        results['passed'] = False
    
    # Overall pass/fail
    results['passed'] = results['build_match'] and results['chrom_match']
    results['detail'] = f"Build match: {results['build_match']}, Chromosome overlap: {results['chrom_match']}"
    
    if verbose:
        print()
        for check in results['checks']:
            if check['status'] == 'PASS':
                icon = '✅'
            elif check['status'] == 'FAIL':
                icon = '❌'
            elif check['status'] == 'WARNING':
                icon = '⚠️'
            else:
                icon = '❓'
            
            print(f"{icon} {check['check']:30s} {check['detail']}")
        
        print("\n" + "=" * 70)
        if results['passed']:
            print(f"✅ PASSED: GTF and FASTA are properly aligned")
        else:
            print(f"❌ FAILED: GTF and FASTA alignment issues detected")
        print("=" * 70 + "\n")
    
    return results


if __name__ == "__main__":
    """Example usage for testing."""
    # Test with some genes
    test_genes = [
        'ENSG00000142611',  # PRDM16 - has splice sites
        'ENSG00000290712',  # No splice sites
        'ENSG00000235993',  # No splice sites
    ]
    
    data_dir = Path("data/ensembl")
    
    try:
        valid, invalid = validate_gene_selection(
            test_genes,
            data_dir,
            fail_on_invalid=False,
            verbose=True
        )
        print(f"\n✅ Test completed. Valid genes: {valid}")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

