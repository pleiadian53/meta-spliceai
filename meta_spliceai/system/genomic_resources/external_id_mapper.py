"""External Gene ID Mapping Utilities

This module provides utilities to fetch gene ID mappings from external sources:
- NCBI Gene database
- Ensembl BioMart
- HGNC (HUGO Gene Nomenclature Committee)

These mappings are used to cross-reference genes across different annotation sources.

Examples:
    >>> from meta_spliceai.system.genomic_resources import fetch_gene_mappings
    >>> 
    >>> # Fetch mappings for a list of gene symbols
    >>> mappings = fetch_gene_mappings(['BRCA1', 'TP53', 'EGFR'])
    >>> 
    >>> # mappings['BRCA1'] = {
    >>> #     'symbol': 'BRCA1',
    >>> #     'ensembl_id': 'ENSG00000012048',
    >>> #     'ncbi_gene_id': '672',
    >>> #     'hgnc_id': 'HGNC:1100'
    >>> # }
"""

from typing import Dict, List, Optional, Set
from pathlib import Path
import json
import time


def create_manual_mane_ensembl_mapping() -> Dict[str, str]:
    """Create manual mapping from MANE gene symbols to Ensembl IDs.
    
    This is a fallback for common genes when external APIs are unavailable.
    
    Returns
    -------
    Dict[str, str]
        Mapping from gene symbol to Ensembl ID
    """
    # Common genes used in testing and validation
    return {
        'BRCA1': 'ENSG00000012048',
        'BRCA2': 'ENSG00000139618',
        'TP53': 'ENSG00000141510',
        'EGFR': 'ENSG00000146648',
        'MYC': 'ENSG00000136997',
        'KRAS': 'ENSG00000133703',
        'GAPDH': 'ENSG00000111640',
        'ACTB': 'ENSG00000075624',
        'TTN': 'ENSG00000155657',
        'APOE': 'ENSG00000130203',
        'CFTR': 'ENSG00000001626',
        'HBB': 'ENSG00000244734',
        'INS': 'ENSG00000254647',
        'ALB': 'ENSG00000163631',
        'PTEN': 'ENSG00000171862',
        'RB1': 'ENSG00000139687',
        'APC': 'ENSG00000134982',
        'VHL': 'ENSG00000134086',
        'CDKN2A': 'ENSG00000147889',
        'ATM': 'ENSG00000149311',
    }


def load_ensembl_grch37_to_grch38_mapping(ensembl_grch37_file: Path) -> Dict[str, str]:
    """Load Ensembl ID to gene symbol mapping from GRCh37.
    
    Parameters
    ----------
    ensembl_grch37_file : Path
        Path to Ensembl GRCh37 gene_features.tsv
    
    Returns
    -------
    Dict[str, str]
        Mapping from Ensembl ID to gene symbol
    """
    import polars as pl
    
    df = pl.read_csv(
        str(ensembl_grch37_file),
        separator='\t',
        schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
    )
    
    # Create mapping: Ensembl ID → gene symbol
    mapping = {}
    for row in df.iter_rows(named=True):
        gene_id = row.get('gene_id', '')
        gene_name = row.get('gene_name', '')
        if gene_id.startswith('ENSG') and gene_name:
            mapping[gene_id] = gene_name
    
    return mapping


def create_mane_to_ensembl_mapping(
    mane_gene_features: Path,
    ensembl_grch37_gene_features: Path,
    output_file: Optional[Path] = None
) -> Dict[str, str]:
    """Create mapping from MANE gene IDs to Ensembl IDs.
    
    Strategy:
    1. Match by gene symbol (most reliable for well-annotated genes)
    2. Use manual mapping for common genes
    3. Save mapping to file for reuse
    
    Parameters
    ----------
    mane_gene_features : Path
        Path to MANE gene_features.tsv
    ensembl_grch37_gene_features : Path
        Path to Ensembl GRCh37 gene_features.tsv
    output_file : Optional[Path]
        Path to save mapping file
    
    Returns
    -------
    Dict[str, str]
        Mapping from MANE gene ID to Ensembl ID
    """
    import polars as pl
    
    # Load MANE genes
    mane_df = pl.read_csv(
        str(mane_gene_features),
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    # Load Ensembl genes
    ensembl_df = pl.read_csv(
        str(ensembl_grch37_gene_features),
        separator='\t',
        schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
    )
    
    # Create symbol → Ensembl ID mapping
    symbol_to_ensembl = {}
    for row in ensembl_df.iter_rows(named=True):
        gene_name = row.get('gene_name', '')
        gene_id = row.get('gene_id', '')
        if gene_name and gene_id.startswith('ENSG'):
            # Handle multiple Ensembl IDs per symbol (use first one)
            if gene_name not in symbol_to_ensembl:
                symbol_to_ensembl[gene_name] = gene_id
    
    # Create MANE gene ID → Ensembl ID mapping
    mane_to_ensembl = {}
    manual_mapping = create_manual_mane_ensembl_mapping()
    
    for row in mane_df.iter_rows(named=True):
        mane_gene_id = row.get('gene_id', '')
        gene_symbol = row.get('gene_name', '')
        
        if not mane_gene_id or not gene_symbol:
            continue
        
        # Try symbol-based mapping first
        if gene_symbol in symbol_to_ensembl:
            mane_to_ensembl[mane_gene_id] = symbol_to_ensembl[gene_symbol]
        # Fallback to manual mapping
        elif gene_symbol in manual_mapping:
            mane_to_ensembl[mane_gene_id] = manual_mapping[gene_symbol]
    
    # Save mapping if requested
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        mapping_data = {
            'source': 'mane_to_ensembl',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_mappings': len(mane_to_ensembl),
            'mappings': mane_to_ensembl
        }
        
        with open(output_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"✅ Saved MANE→Ensembl mapping to: {output_file}")
        print(f"   Total mappings: {len(mane_to_ensembl):,}")
    
    return mane_to_ensembl


def load_mane_to_ensembl_mapping(mapping_file: Path) -> Dict[str, str]:
    """Load MANE→Ensembl mapping from file.
    
    Parameters
    ----------
    mapping_file : Path
        Path to mapping JSON file
    
    Returns
    -------
    Dict[str, str]
        Mapping from MANE gene ID to Ensembl ID
    """
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    
    return data.get('mappings', {})


def fetch_ensembl_ids_for_symbols(
    gene_symbols: List[str],
    species: str = 'homo_sapiens'
) -> Dict[str, Optional[str]]:
    """Fetch Ensembl IDs for gene symbols using Ensembl REST API.
    
    This is a fallback method when local mappings are insufficient.
    
    Parameters
    ----------
    gene_symbols : List[str]
        List of gene symbols
    species : str, default='homo_sapiens'
        Species name
    
    Returns
    -------
    Dict[str, Optional[str]]
        Mapping from gene symbol to Ensembl ID (None if not found)
    
    Notes
    -----
    This function requires internet connection and may be rate-limited.
    Use sparingly and cache results.
    """
    try:
        import requests
    except ImportError:
        print("⚠️  requests library not available - cannot fetch from Ensembl API")
        return {symbol: None for symbol in gene_symbols}
    
    mapping = {}
    base_url = "https://rest.ensembl.org"
    
    for symbol in gene_symbols:
        try:
            # Query Ensembl REST API
            url = f"{base_url}/lookup/symbol/{species}/{symbol}"
            response = requests.get(url, headers={"Content-Type": "application/json"})
            
            if response.status_code == 200:
                data = response.json()
                mapping[symbol] = data.get('id')
            else:
                mapping[symbol] = None
            
            # Rate limiting
            time.sleep(0.1)
        
        except Exception as e:
            print(f"⚠️  Failed to fetch Ensembl ID for {symbol}: {e}")
            mapping[symbol] = None
    
    return mapping


# Convenience function
def get_or_create_mane_ensembl_mapping(
    data_dir: Path,
    force_recreate: bool = False
) -> Dict[str, str]:
    """Get or create MANE→Ensembl mapping.
    
    Parameters
    ----------
    data_dir : Path
        Data directory containing gene_features files
    force_recreate : bool, default=False
        Force recreation even if mapping file exists
    
    Returns
    -------
    Dict[str, str]
        Mapping from MANE gene ID to Ensembl ID
    """
    mapping_file = data_dir / 'mane' / 'GRCh38' / 'mane_to_ensembl_mapping.json'
    
    # Load existing mapping if available
    if mapping_file.exists() and not force_recreate:
        print(f"📂 Loading existing MANE→Ensembl mapping from: {mapping_file}")
        return load_mane_to_ensembl_mapping(mapping_file)
    
    # Create new mapping
    print("🔨 Creating MANE→Ensembl mapping...")
    mane_features = data_dir / 'mane' / 'GRCh38' / 'gene_features.tsv'
    ensembl_features = data_dir / 'ensembl' / 'GRCh37' / 'gene_features.tsv'
    
    if not mane_features.exists():
        raise FileNotFoundError(f"MANE gene features not found: {mane_features}")
    if not ensembl_features.exists():
        raise FileNotFoundError(f"Ensembl gene features not found: {ensembl_features}")
    
    mapping = create_mane_to_ensembl_mapping(
        mane_features,
        ensembl_features,
        output_file=mapping_file
    )
    
    return mapping




