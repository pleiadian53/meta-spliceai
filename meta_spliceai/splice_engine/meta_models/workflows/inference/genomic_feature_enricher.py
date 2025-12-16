"""Genomic Feature Enricher for Inference Output

Adds genomic metadata and derived features to inference predictions
using the systematic genomic_resources package.
"""

from pathlib import Path
from typing import Optional, Dict, List
import polars as pl

from meta_spliceai.system.genomic_resources import Registry


class GenomicFeatureEnricher:
    """Enriches inference predictions with genomic features.
    
    Adds columns from gene_features, transcript_features, exon_features
    to match training data format and enable proper validation.
    
    Parameters
    ----------
    registry : Registry, optional
        Genomic resources registry. If None, creates new instance.
    verbose : bool, default=True
        Print progress messages
        
    Examples
    --------
    >>> enricher = GenomicFeatureEnricher()
    >>> predictions = enricher.enrich(predictions_df)
    >>> 'absolute_position' in predictions.columns
    True
    """
    
    def __init__(self, registry: Optional[Registry] = None, verbose: bool = True):
        self.registry = registry or Registry()
        self.verbose = verbose
        self._gene_features = None
        self._transcript_features = None
        self._exon_features = None
        
    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[genomic-enricher] {message}")
    
    def _load_gene_features(self) -> pl.DataFrame:
        """Load gene features table.
        
        Prefers the full gene_features.tsv from spliceai_analysis/ over partial ones.
        """
        if self._gene_features is None:
            # Try spliceai_analysis first (full gene set)
            full_path = self.registry.top / "spliceai_analysis" / "gene_features.tsv"
            if full_path.exists():
                gene_features_path = str(full_path)
                self._log(f"Loading gene features from {full_path.name} (full gene set)")
            else:
                # Fallback to Registry resolution
                gene_features_path = self.registry.resolve("gene_features")
                if gene_features_path is None:
                    raise FileNotFoundError("gene_features.tsv not found")
                self._log(f"Loading gene features from {Path(gene_features_path).name}")
            
            self._gene_features = pl.read_csv(
                gene_features_path,
                separator='\t',
                schema_overrides={'chrom': pl.Utf8}
            )
            self._log(f"  Loaded {len(self._gene_features):,} genes")
        
        return self._gene_features
    
    def _load_transcript_features(self) -> Optional[pl.DataFrame]:
        """Load transcript features table (optional)."""
        if self._transcript_features is None:
            transcript_features_path = self.registry.resolve("transcript_features")
            if transcript_features_path is None:
                self._log("  transcript_features.tsv not found (optional)")
                return None
            
            self._log(f"Loading transcript features from {Path(transcript_features_path).name}")
            self._transcript_features = pl.read_csv(
                transcript_features_path,
                separator='\t',
                schema_overrides={'chrom': pl.Utf8}
            )
            self._log(f"  Loaded {len(self._transcript_features):,} transcripts")
        
        return self._transcript_features
    
    def _load_exon_features(self) -> Optional[pl.DataFrame]:
        """Load exon features table (optional)."""
        if self._exon_features is None:
            exon_features_path = self.registry.resolve("exon_features")
            if exon_features_path is None:
                self._log("  exon_features.tsv not found (optional)")
                return None
            
            self._log(f"Loading exon features from {Path(exon_features_path).name}")
            self._exon_features = pl.read_csv(
                exon_features_path,
                separator='\t',
                schema_overrides={'chrom': pl.Utf8}
            )
            self._log(f"  Loaded {len(self._exon_features):,} exons")
        
        return self._exon_features
    
    def _load_overlapping_genes(self) -> Optional[pl.DataFrame]:
        """Load overlapping genes table (for num_overlaps)."""
        try:
            overlapping_path = self.registry.resolve("overlapping_genes")
            if overlapping_path is None:
                self._log("  overlapping_genes.tsv not found (optional)")
                return None
            
            self._log(f"Loading overlapping genes from {Path(overlapping_path).name}")
            overlapping_df = pl.read_csv(
                overlapping_path,
                separator='\t',
                schema_overrides={'chrom': pl.Utf8}
            )
            self._log(f"  Loaded {len(overlapping_df):,} overlapping gene pairs")
            
            # Aggregate num_overlaps by gene_id (take max for each gene)
            gene_overlaps = overlapping_df.group_by('gene_id_1').agg([
                pl.col('num_overlaps').max().alias('num_overlaps')
            ]).rename({'gene_id_1': 'gene_id'})
            
            self._log(f"  Aggregated to {len(gene_overlaps):,} unique genes")
            return gene_overlaps
            
        except Exception as e:
            self._log(f"  ⚠️  Failed to load overlapping genes: {e}")
            return None
    
    def enrich(self, df: pl.DataFrame, 
               include_critical: bool = True,
               include_useful: bool = True,
               include_structure: bool = True,
               include_flags: bool = True) -> pl.DataFrame:
        """Enrich predictions with genomic features.
        
        Parameters
        ----------
        df : pl.DataFrame
            Predictions dataframe with at minimum: gene_id, position, strand
        include_critical : bool, default=True
            Add critical columns: absolute_position, gene_start, gene_end, gene_length
        include_useful : bool, default=True
            Add useful columns: gene_name, gene_type, distance_to_*, n_splice_sites
        include_structure : bool, default=True
            Add structure columns: num_exons, avg_exon_length, total_*_length, etc.
        include_flags : bool, default=True
            Add flag columns: has_gene_info, has_tx_info, missing_*_feats
            
        Returns
        -------
        pl.DataFrame
            Enriched predictions with additional genomic feature columns
        """
        if self.verbose:
            print("=" * 80)
            print("🧬 ENRICHING PREDICTIONS WITH GENOMIC FEATURES")
            print("=" * 80)
            print()
        
        # Validate required columns
        required = ['gene_id', 'position']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Load gene features
        gene_features = self._load_gene_features()
        
        # Load transcript features (for tx_start, tx_end, transcript_length)
        transcript_features = self._load_transcript_features()
        
        # Load overlapping genes (for num_overlaps)
        overlapping_genes = self._load_overlapping_genes()
        
        # Determine which columns to add
        columns_to_add = []
        
        # CRITICAL columns (always recommended)
        # Map to actual column names in gene_features (may be 'start'/'end' not 'gene_start'/'gene_end')
        if include_critical:
            critical_cols = [
                'start', 'end', 'gene_length', 'gene_name',
                'chrom', 'strand'  # May already exist, but ensure from gene_features
            ]
            columns_to_add.extend(critical_cols)
        
        # USEFUL columns
        if include_useful:
            useful_cols = ['gene_type']
            columns_to_add.extend(useful_cols)
        
        # STRUCTURE columns
        if include_structure:
            structure_cols = [
                'num_exons', 'avg_exon_length', 'median_exon_length',
                'total_exon_length', 'total_intron_length', 
                'num_overlaps'
            ]
            # Only add if they exist in gene_features
            available_structure = [col for col in structure_cols if col in gene_features.columns]
            columns_to_add.extend(available_structure)
        
        # TRANSCRIPT columns (from transcript_features)
        if transcript_features is not None:
            transcript_cols = ['transcript_length', 'start', 'end']  # start/end will be renamed to tx_start/tx_end
            available_transcript = [col for col in transcript_cols if col in transcript_features.columns]
            columns_to_add.extend(available_transcript)
        
        # Get unique columns (avoid duplicates)
        columns_to_add = list(set(['gene_id'] + columns_to_add))
        
        # Select columns that actually exist
        available_columns = [col for col in columns_to_add if col in gene_features.columns]
        
        self._log(f"Joining {len(available_columns)} genomic feature columns")
        
        # Check for existing columns that would conflict
        existing_cols = set(df.columns)
        gf_cols_to_add = [col for col in available_columns if col != 'gene_id']
        
        # Separate into conflicting and non-conflicting
        non_conflicting = [col for col in gf_cols_to_add if col not in existing_cols]
        conflicting = [col for col in gf_cols_to_add if col in existing_cols]
        
        if conflicting:
            self._log(f"  Skipping {len(conflicting)} columns that already exist: {conflicting[:3]}...")
        
        # Select only gene_id + non-conflicting columns for join
        join_columns = ['gene_id'] + non_conflicting
        
        # Join with gene features
        enriched = df.join(
            gene_features.select(join_columns),
            on='gene_id',
            how='left'
        )
        
        # Join with transcript features (if available and if structure features requested)
        # CRITICAL: Skip this join if include_structure=False to avoid row multiplication
        # (one gene can have multiple transcripts, causing N×M row explosion)
        if transcript_features is not None and include_structure:
            transcript_join_cols = ['gene_id', 'transcript_length', 'start', 'end']
            available_transcript_join = [col for col in transcript_join_cols if col in transcript_features.columns]
            
            if len(available_transcript_join) > 1:  # More than just gene_id
                self._log(f"Joining {len(available_transcript_join)-1} transcript feature columns")
                
                # Rename transcript start/end to tx_start/tx_end to avoid conflicts
                transcript_select = transcript_features.select(available_transcript_join)
                if 'start' in transcript_select.columns:
                    transcript_select = transcript_select.rename({'start': 'tx_start'})
                if 'end' in transcript_select.columns:
                    transcript_select = transcript_select.rename({'end': 'tx_end'})
                
                enriched = enriched.join(
                    transcript_select,
                    on='gene_id',
                    how='left'
                )
        elif transcript_features is not None and not include_structure:
            self._log("Skipping transcript feature join (include_structure=False) to avoid row multiplication")
        
        # Join with overlapping genes (if available)
        if overlapping_genes is not None:
            self._log("Joining overlapping genes data")
            enriched = enriched.join(
                overlapping_genes,
                on='gene_id',
                how='left'
            )
        
        # Rename columns to match training data format (start → gene_start, end → gene_end)
        if 'start' in enriched.columns and 'gene_start' not in enriched.columns:
            enriched = enriched.rename({'start': 'gene_start'})
        if 'end' in enriched.columns and 'gene_end' not in enriched.columns:
            enriched = enriched.rename({'end': 'gene_end'})
        
        # Calculate DERIVED columns
        if include_critical:
            self._log("Calculating derived coordinate columns...")
            
            # absolute_position (strand-aware genomic coordinate)
            if 'absolute_position' not in enriched.columns:
                # Check if we have the required columns
                if 'gene_start' not in enriched.columns or 'gene_end' not in enriched.columns:
                    self._log("  ⚠️  Missing gene_start/gene_end, skipping absolute_position")
                elif 'strand' not in enriched.columns:
                    self._log("  ⚠️  Missing strand column, skipping absolute_position")
                else:
                    enriched = enriched.with_columns([
                        pl.when(pl.col('strand') == '+')
                          .then(pl.col('gene_start') + pl.col('position'))
                          .when(pl.col('strand') == '-')
                          .then(pl.col('gene_end') - pl.col('position'))
                          .otherwise(None)
                          .alias('absolute_position')
                    ])
                    self._log("  ✅ Added absolute_position")
        
        if include_useful:
            self._log("Calculating position context columns...")
            
            # distance_to_start (always relative position)
            if 'distance_to_start' not in enriched.columns:
                enriched = enriched.with_columns([
                    pl.col('position').alias('distance_to_start')
                ])
                self._log("  ✅ Added distance_to_start")
            
            # distance_to_end
            if 'distance_to_end' not in enriched.columns and 'gene_length' in enriched.columns:
                enriched = enriched.with_columns([
                    (pl.col('gene_length') - pl.col('position')).alias('distance_to_end')
                ])
                self._log("  ✅ Added distance_to_end")
            
            # n_splice_sites (from splice_sites if available)
            if 'n_splice_sites' not in enriched.columns:
                try:
                    splice_sites_path = self.registry.resolve("splice_sites")
                    if splice_sites_path:
                        self._log("  Loading splice site counts...")
                        splice_sites = pl.read_csv(
                            splice_sites_path,
                            separator='\t',
                            schema_overrides={'chrom': pl.Utf8}
                        )
                        
                        # Count splice sites per gene
                        splice_counts = splice_sites.group_by('gene_id').agg([
                            pl.len().alias('n_splice_sites')
                        ])
                        
                        # Join with predictions
                        enriched = enriched.join(
                            splice_counts,
                            on='gene_id',
                            how='left'
                        )
                        
                        # Fill nulls with 0
                        enriched = enriched.with_columns([
                            pl.col('n_splice_sites').fill_null(0)
                        ])
                        
                        self._log("  ✅ Added n_splice_sites")
                except Exception as e:
                    self._log(f"  ⚠️  Could not add n_splice_sites: {e}")
        
        # Add FLAG columns
        if include_flags:
            self._log("Adding metadata flags...")
            
            # has_gene_info: whether gene was found in gene_features
            if 'has_gene_info' not in enriched.columns:
                enriched = enriched.with_columns([
                    pl.col('gene_name').is_not_null().alias('has_gene_info')
                ])
            
            # missing_transcript_feats: count of null transcript columns
            if 'missing_transcript_feats' not in enriched.columns and 'transcript_id' in enriched.columns:
                # Count how many transcript-related columns are null
                tx_cols = [col for col in enriched.columns if 'transcript' in col.lower() or 'tx_' in col]
                if tx_cols:
                    enriched = enriched.with_columns([
                        sum([pl.col(col).is_null().cast(pl.Int32) for col in tx_cols]).alias('missing_transcript_feats')
                    ])
        
        if self.verbose:
            added_cols = set(enriched.columns) - set(df.columns)
            print()
            print(f"✅ Enrichment complete!")
            print(f"   Added {len(added_cols)} new columns")
            print(f"   Total columns: {len(enriched.columns)}")
            print()
            print("Key columns added:")
            for col in sorted(added_cols):
                if col in ['absolute_position', 'gene_start', 'gene_end', 'gene_name', 
                          'gene_type', 'gene_length', 'n_splice_sites']:
                    print(f"   ⭐ {col}")
            print()
        
        return enriched
    
    def get_added_columns(self, 
                         include_critical: bool = True,
                         include_useful: bool = True,
                         include_structure: bool = True,
                         include_flags: bool = True) -> List[str]:
        """Get list of columns that would be added by enrichment.
        
        Useful for documentation and validation.
        
        Returns
        -------
        List[str]
            Names of columns that will be added
        """
        columns = []
        
        if include_critical:
            columns.extend([
                'gene_start', 'gene_end', 'gene_length', 'gene_name',
                'absolute_position'
            ])
        
        if include_useful:
            columns.extend([
                'gene_type', 'distance_to_start', 'distance_to_end', 
                'n_splice_sites'
            ])
        
        if include_structure:
            columns.extend([
                'num_exons', 'avg_exon_length', 'median_exon_length',
                'total_exon_length', 'total_intron_length', 'num_overlaps'
            ])
        
        if include_flags:
            columns.extend([
                'has_gene_info', 'missing_transcript_feats'
            ])
        
        return columns


def enrich_predictions_with_genomic_features(
    predictions: pl.DataFrame,
    registry: Optional[Registry] = None,
    include_all: bool = True,
    verbose: bool = True
) -> pl.DataFrame:
    """Convenience function to enrich predictions with genomic features.
    
    Parameters
    ----------
    predictions : pl.DataFrame
        Predictions to enrich
    registry : Registry, optional
        Genomic resources registry
    include_all : bool, default=True
        Include all available features (critical + useful + structure + flags)
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    pl.DataFrame
        Enriched predictions
        
    Examples
    --------
    >>> from meta_spliceai.splice_engine.meta_models.workflows.inference.genomic_feature_enricher import enrich_predictions_with_genomic_features
    >>> enriched = enrich_predictions_with_genomic_features(predictions_df)
    """
    enricher = GenomicFeatureEnricher(registry=registry, verbose=verbose)
    return enricher.enrich(
        predictions,
        include_critical=True,
        include_useful=include_all,
        include_structure=include_all,
        include_flags=include_all
    )

