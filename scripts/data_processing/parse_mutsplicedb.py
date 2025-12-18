#!/usr/bin/env python3
"""
Parse MutSpliceDB Export to Extract Aberrant Splice Site Coordinates.

This script processes the CSV/Excel export from MutSpliceDB and extracts
precise genomic coordinates for variant-induced aberrant splice sites.

Usage
-----
python parse_mutsplicedb.py \
    --input data/mutsplicedb/mutsplicedb_export.csv \
    --output data/mutsplicedb/splice_sites_induced.tsv \
    --gtf data/mane/GRCh38/MANE.GRCh38.v1.3.ensembl_genomic.gtf

Input
-----
MutSpliceDB export with columns:
- Gene Symbol
- HGVS Notation (e.g., NM_000546.5:c.375+5G>A)
- Splicing Effect (e.g., "Exon 4 skipping", "Intron retention")
- Sample Name
- Sample Source

Output
------
TSV file with aberrant splice site coordinates:
- chrom, position, strand, site_type, inducing_variant, effect_type, ...

See Also
--------
- docs/wishlist/ABERRANT_SPLICE_SITE_ANNOTATIONS.md
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExonInfo:
    """Exon information from GTF."""
    chrom: str
    start: int  # 1-based
    end: int    # 1-based, inclusive
    strand: str
    exon_number: int
    transcript_id: str
    gene_name: str


@dataclass
class InducedSpliceSite:
    """Variant-induced aberrant splice site."""
    chrom: str
    position: int
    strand: str
    site_type: str  # donor_cryptic, acceptor_cryptic, donor_canonical_lost, etc.
    inducing_variant: str
    effect_type: str  # exon_skipping, intron_retention, cryptic_activation
    gene: str
    transcript_id: str
    evidence_source: str
    evidence_samples: str
    confidence: str  # high, medium, low
    notes: str = ""


@dataclass
class ParsedEffect:
    """Parsed splicing effect from MutSpliceDB description."""
    effect_type: str
    affected_exons: List[int] = field(default_factory=list)
    affected_introns: List[int] = field(default_factory=list)
    cryptic_position: Optional[int] = None
    raw_description: str = ""


# =============================================================================
# GTF PARSING
# =============================================================================

def load_gtf_exons(gtf_path: str) -> Dict[str, List[ExonInfo]]:
    """
    Load exon information from GTF file.
    
    Returns
    -------
    Dict[str, List[ExonInfo]]
        Mapping from transcript_id to list of exons (sorted by exon number)
    """
    logger.info(f"Loading GTF from {gtf_path}...")
    
    exons_by_transcript = {}
    
    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            if fields[2] != 'exon':
                continue
            
            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            
            # Parse attributes
            attrs = {}
            for attr in fields[8].split(';'):
                attr = attr.strip()
                if not attr:
                    continue
                
                # Handle both formats: key "value" and key=value
                if ' ' in attr:
                    key, value = attr.split(' ', 1)
                    value = value.strip('"')
                elif '=' in attr:
                    key, value = attr.split('=', 1)
                    value = value.strip('"')
                else:
                    continue
                
                attrs[key] = value
            
            transcript_id = attrs.get('transcript_id', '')
            gene_name = attrs.get('gene_name', attrs.get('gene_id', ''))
            exon_number = int(attrs.get('exon_number', 0))
            
            if not transcript_id or exon_number == 0:
                continue
            
            exon = ExonInfo(
                chrom=chrom,
                start=start,
                end=end,
                strand=strand,
                exon_number=exon_number,
                transcript_id=transcript_id,
                gene_name=gene_name
            )
            
            if transcript_id not in exons_by_transcript:
                exons_by_transcript[transcript_id] = []
            
            exons_by_transcript[transcript_id].append(exon)
    
    # Sort exons by number
    for tx_id in exons_by_transcript:
        exons_by_transcript[tx_id].sort(key=lambda e: e.exon_number)
    
    logger.info(f"Loaded {len(exons_by_transcript)} transcripts with exons")
    return exons_by_transcript


def get_transcript_from_hgvs(hgvs: str) -> Optional[str]:
    """Extract transcript ID from HGVS notation."""
    # NM_000546.5:c.375+5G>A → NM_000546.5
    match = re.match(r'(N[MR]_\d+\.\d+)', hgvs)
    if match:
        return match.group(1)
    return None


# =============================================================================
# EFFECT PARSING
# =============================================================================

def parse_splicing_effect(effect_description: str) -> ParsedEffect:
    """
    Parse MutSpliceDB splicing effect description.
    
    Examples
    --------
    "Exon 4 skipping" → ParsedEffect(effect_type='exon_skipping', affected_exons=[4])
    "Exon 3-4 skipping" → ParsedEffect(effect_type='exon_skipping', affected_exons=[3, 4])
    "Intron 5 retention" → ParsedEffect(effect_type='intron_retention', affected_introns=[5])
    "Cryptic donor activation" → ParsedEffect(effect_type='cryptic_activation')
    """
    effect = ParsedEffect(raw_description=effect_description)
    desc_lower = effect_description.lower()
    
    # Exon skipping patterns
    exon_skip_patterns = [
        r'exon\s+(\d+)\s+(?:is\s+)?skip',
        r'skip(?:ping|s)?\s+(?:of\s+)?exon\s+(\d+)',
        r'exon\s+(\d+)\s*[-–]\s*(\d+)\s+skip',
        r'exons?\s+(\d+)\s+and\s+(\d+)\s+skip',
    ]
    
    for pattern in exon_skip_patterns:
        match = re.search(pattern, desc_lower)
        if match:
            effect.effect_type = 'exon_skipping'
            groups = match.groups()
            for g in groups:
                if g:
                    effect.affected_exons.append(int(g))
            return effect
    
    # Intron retention patterns
    intron_patterns = [
        r'intron\s+(\d+)\s+retention',
        r'retention\s+(?:of\s+)?intron\s+(\d+)',
        r'intron\s+(\d+)\s+(?:is\s+)?retained',
    ]
    
    for pattern in intron_patterns:
        match = re.search(pattern, desc_lower)
        if match:
            effect.effect_type = 'intron_retention'
            effect.affected_introns.append(int(match.group(1)))
            return effect
    
    # Cryptic site patterns
    if 'cryptic' in desc_lower:
        if 'donor' in desc_lower:
            effect.effect_type = 'cryptic_donor'
        elif 'acceptor' in desc_lower:
            effect.effect_type = 'cryptic_acceptor'
        else:
            effect.effect_type = 'cryptic_activation'
        
        # Try to extract position
        pos_match = re.search(r'at\s+(?:position\s+)?(\d+)', desc_lower)
        if pos_match:
            effect.cryptic_position = int(pos_match.group(1))
        
        return effect
    
    # Partial/alternative exon usage
    if 'partial' in desc_lower or 'alternative' in desc_lower:
        effect.effect_type = 'alternative_exon'
        
        exon_match = re.search(r'exon\s+(\d+)', desc_lower)
        if exon_match:
            effect.affected_exons.append(int(exon_match.group(1)))
        
        return effect
    
    # Unknown effect
    effect.effect_type = 'unknown'
    return effect


# =============================================================================
# COORDINATE DERIVATION
# =============================================================================

def derive_junction_from_exon_skipping(
    exons: List[ExonInfo],
    skipped_exon_numbers: List[int]
) -> List[InducedSpliceSite]:
    """
    Derive junction coordinates from exon skipping.
    
    When exon N is skipped, a novel junction forms:
    - Donor: end of exon N-1
    - Acceptor: start of exon N+1
    """
    if not exons or not skipped_exon_numbers:
        return []
    
    sites = []
    exon_dict = {e.exon_number: e for e in exons}
    
    for skip_num in sorted(skipped_exon_numbers):
        prev_exon = exon_dict.get(skip_num - 1)
        next_exon = exon_dict.get(skip_num + 1)
        
        if not prev_exon or not next_exon:
            continue
        
        # The novel junction connects:
        # - Donor at end of previous exon
        # - Acceptor at start of next exon
        
        # For + strand: donor is at end position, acceptor at start
        # For - strand: donor is at start position, acceptor at end
        
        if prev_exon.strand == '+':
            donor_pos = prev_exon.end
            acceptor_pos = next_exon.start
        else:
            donor_pos = prev_exon.start
            acceptor_pos = next_exon.end
        
        # Record the novel junction (these are "canonical" sites being used
        # in a novel combination, not cryptic sites)
        sites.append(InducedSpliceSite(
            chrom=prev_exon.chrom,
            position=donor_pos,
            strand=prev_exon.strand,
            site_type='donor_novel_junction',
            inducing_variant='',  # Will be filled by caller
            effect_type='exon_skipping',
            gene=prev_exon.gene_name,
            transcript_id=prev_exon.transcript_id,
            evidence_source='',
            evidence_samples='',
            confidence='high',
            notes=f'Novel junction from exon {skip_num} skipping'
        ))
        
        sites.append(InducedSpliceSite(
            chrom=next_exon.chrom,
            position=acceptor_pos,
            strand=next_exon.strand,
            site_type='acceptor_novel_junction',
            inducing_variant='',
            effect_type='exon_skipping',
            gene=next_exon.gene_name,
            transcript_id=next_exon.transcript_id,
            evidence_source='',
            evidence_samples='',
            confidence='high',
            notes=f'Novel junction from exon {skip_num} skipping'
        ))
    
    return sites


def derive_sites_from_intron_retention(
    exons: List[ExonInfo],
    retained_intron_numbers: List[int]
) -> List[InducedSpliceSite]:
    """
    Derive affected sites from intron retention.
    
    When intron N is retained (between exon N and N+1):
    - The canonical donor of exon N is NOT used
    - The canonical acceptor of exon N+1 is NOT used
    """
    if not exons or not retained_intron_numbers:
        return []
    
    sites = []
    exon_dict = {e.exon_number: e for e in exons}
    
    for intron_num in retained_intron_numbers:
        # Intron N is between exon N and exon N+1
        donor_exon = exon_dict.get(intron_num)
        acceptor_exon = exon_dict.get(intron_num + 1)
        
        if not donor_exon or not acceptor_exon:
            continue
        
        # Mark these sites as "not used" / disrupted
        if donor_exon.strand == '+':
            donor_pos = donor_exon.end
            acceptor_pos = acceptor_exon.start
        else:
            donor_pos = donor_exon.start
            acceptor_pos = acceptor_exon.end
        
        sites.append(InducedSpliceSite(
            chrom=donor_exon.chrom,
            position=donor_pos,
            strand=donor_exon.strand,
            site_type='donor_canonical_lost',
            inducing_variant='',
            effect_type='intron_retention',
            gene=donor_exon.gene_name,
            transcript_id=donor_exon.transcript_id,
            evidence_source='',
            evidence_samples='',
            confidence='high',
            notes=f'Intron {intron_num} retention - donor not used'
        ))
        
        sites.append(InducedSpliceSite(
            chrom=acceptor_exon.chrom,
            position=acceptor_pos,
            strand=acceptor_exon.strand,
            site_type='acceptor_canonical_lost',
            inducing_variant='',
            effect_type='intron_retention',
            gene=acceptor_exon.gene_name,
            transcript_id=acceptor_exon.transcript_id,
            evidence_source='',
            evidence_samples='',
            confidence='high',
            notes=f'Intron {intron_num} retention - acceptor not used'
        ))
    
    return sites


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_mutsplicedb_entry(
    row: pd.Series,
    exons_by_transcript: Dict[str, List[ExonInfo]],
    refseq_to_ensembl: Optional[Dict[str, str]] = None
) -> List[InducedSpliceSite]:
    """
    Process a single MutSpliceDB entry to extract induced splice sites.
    """
    sites = []
    
    # Extract fields (adjust column names based on actual export)
    gene = row.get('Gene Symbol', row.get('gene_symbol', ''))
    hgvs = row.get('HGVS Notation', row.get('hgvs', ''))
    effect_desc = row.get('Splicing Effect', row.get('splicing_effect', ''))
    sample = row.get('Sample Name', row.get('sample', ''))
    source = row.get('Sample Source', row.get('source', 'MutSpliceDB'))
    
    if not effect_desc:
        return sites
    
    # Parse effect description
    parsed_effect = parse_splicing_effect(effect_desc)
    
    if parsed_effect.effect_type == 'unknown':
        logger.debug(f"Could not parse effect: {effect_desc}")
        return sites
    
    # Get transcript from HGVS
    transcript_id = get_transcript_from_hgvs(hgvs) if hgvs else None
    
    # Try to find exons for this transcript
    exons = None
    if transcript_id:
        # Try direct lookup
        exons = exons_by_transcript.get(transcript_id)
        
        # Try without version
        if not exons:
            tx_base = transcript_id.split('.')[0]
            for tx_id in exons_by_transcript:
                if tx_id.startswith(tx_base):
                    exons = exons_by_transcript[tx_id]
                    break
        
        # Try RefSeq to Ensembl mapping
        if not exons and refseq_to_ensembl:
            ensembl_id = refseq_to_ensembl.get(transcript_id)
            if ensembl_id:
                exons = exons_by_transcript.get(ensembl_id)
    
    # If no transcript, try to find by gene name
    if not exons and gene:
        for tx_id, tx_exons in exons_by_transcript.items():
            if tx_exons and tx_exons[0].gene_name == gene:
                exons = tx_exons
                break
    
    if not exons:
        logger.debug(f"Could not find exons for {gene} / {transcript_id}")
        return sites
    
    # Derive sites based on effect type
    if parsed_effect.effect_type == 'exon_skipping' and parsed_effect.affected_exons:
        sites = derive_junction_from_exon_skipping(exons, parsed_effect.affected_exons)
    
    elif parsed_effect.effect_type == 'intron_retention' and parsed_effect.affected_introns:
        sites = derive_sites_from_intron_retention(exons, parsed_effect.affected_introns)
    
    elif parsed_effect.effect_type in ['cryptic_donor', 'cryptic_acceptor', 'cryptic_activation']:
        # For cryptic sites, we may not have exact positions without more analysis
        # Mark these for manual review or further processing
        logger.info(f"Cryptic site detected but position unknown: {gene} - {effect_desc}")
    
    # Fill in common fields
    for site in sites:
        site.inducing_variant = hgvs
        site.evidence_source = source
        site.evidence_samples = sample
    
    return sites


def main():
    parser = argparse.ArgumentParser(
        description='Parse MutSpliceDB export to extract aberrant splice site coordinates'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to MutSpliceDB CSV/Excel export'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output path for induced splice sites TSV'
    )
    parser.add_argument(
        '--gtf',
        required=True,
        help='Path to GTF file for exon coordinates'
    )
    parser.add_argument(
        '--refseq-mapping',
        help='Optional: RefSeq to Ensembl transcript ID mapping file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load GTF
    gtf_path = Path(args.gtf)
    if not gtf_path.exists():
        logger.error(f"GTF file not found: {gtf_path}")
        sys.exit(1)
    
    exons_by_transcript = load_gtf_exons(str(gtf_path))
    
    # Load RefSeq mapping if provided
    refseq_to_ensembl = None
    if args.refseq_mapping:
        mapping_path = Path(args.refseq_mapping)
        if mapping_path.exists():
            mapping_df = pd.read_csv(mapping_path, sep='\t')
            refseq_to_ensembl = dict(zip(
                mapping_df['refseq_id'],
                mapping_df['ensembl_id']
            ))
            logger.info(f"Loaded {len(refseq_to_ensembl)} RefSeq-Ensembl mappings")
    
    # Load MutSpliceDB export
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info(f"Loading MutSpliceDB export from {input_path}...")
    
    if input_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)
    
    logger.info(f"Loaded {len(df)} entries")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Process each entry
    all_sites = []
    success_count = 0
    
    for idx, row in df.iterrows():
        try:
            sites = process_mutsplicedb_entry(row, exons_by_transcript, refseq_to_ensembl)
            if sites:
                all_sites.extend(sites)
                success_count += 1
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
    
    logger.info(f"Successfully processed {success_count}/{len(df)} entries")
    logger.info(f"Extracted {len(all_sites)} induced splice sites")
    
    # Convert to DataFrame and save
    if all_sites:
        output_df = pd.DataFrame([
            {
                'chrom': s.chrom,
                'position': s.position,
                'strand': s.strand,
                'site_type': s.site_type,
                'inducing_variant': s.inducing_variant,
                'effect_type': s.effect_type,
                'gene': s.gene,
                'transcript_id': s.transcript_id,
                'evidence_source': s.evidence_source,
                'evidence_samples': s.evidence_samples,
                'confidence': s.confidence,
                'notes': s.notes
            }
            for s in all_sites
        ])
        
        # Remove duplicates
        output_df = output_df.drop_duplicates(
            subset=['chrom', 'position', 'strand', 'site_type', 'inducing_variant']
        )
        
        # Sort
        output_df = output_df.sort_values(['chrom', 'position'])
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, sep='\t', index=False)
        
        logger.info(f"Saved {len(output_df)} unique sites to {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total entries processed: {len(df)}")
        print(f"Entries with extractable sites: {success_count}")
        print(f"Total induced sites: {len(output_df)}")
        print(f"\nSite type distribution:")
        print(output_df['site_type'].value_counts().to_string())
        print(f"\nEffect type distribution:")
        print(output_df['effect_type'].value_counts().to_string())
    else:
        logger.warning("No sites extracted. Check input data and GTF matching.")


if __name__ == '__main__':
    main()

