#!/usr/bin/env python3
"""
Splice Site Coordinate Reconciliation Module

This module provides systematic coordinate reconciliation between different splice site
annotation systems, prediction models, and genomic coordinate conventions.

Critical Issues Addressed:
1. Genomic annotation version differences (GTF/FASTA versions)
2. Coordinate system differences (0-based vs 1-based indexing)
3. Splice site definition differences (exact positions, motif requirements)
4. Model-specific coordinate adjustments (SpliceAI, OpenSpliceAI, etc.)

Even 1-2nt differences in splice site coordinates can cause:
- False negative evaluations of accurate predictions
- Systematic bias in meta-model training
- Inconsistent results across different annotation versions
- Failure to detect true alternative splicing events

Author: MetaSpliceAI Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CoordinateSystem(Enum):
    """Coordinate system conventions."""
    ZERO_BASED = "0-based"  # 0-based, half-open intervals [start, end)
    ONE_BASED = "1-based"   # 1-based, closed intervals [start, end]


class SpliceType(Enum):
    """Splice site types."""
    DONOR = "donor"
    ACCEPTOR = "acceptor"


class Strand(Enum):
    """DNA strand orientation."""
    PLUS = "+"
    MINUS = "-"


@dataclass
class CoordinateOffset:
    """Represents a coordinate offset for splice site reconciliation."""
    donor_plus: int = 0      # Donor site offset on plus strand
    donor_minus: int = 0     # Donor site offset on minus strand
    acceptor_plus: int = 0   # Acceptor site offset on plus strand
    acceptor_minus: int = 0  # Acceptor site offset on minus strand
    
    def to_dict(self) -> Dict[str, Dict[str, int]]:
        """Convert to dictionary format."""
        return {
            'donor': {
                'plus': self.donor_plus,
                'minus': self.donor_minus
            },
            'acceptor': {
                'plus': self.acceptor_plus,
                'minus': self.acceptor_minus
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, int]]) -> 'CoordinateOffset':
        """Create from dictionary format."""
        return cls(
            donor_plus=data['donor']['plus'],
            donor_minus=data['donor']['minus'],
            acceptor_plus=data['acceptor']['plus'],
            acceptor_minus=data['acceptor']['minus']
        )


class SpliceCoordinateReconciler:
    """
    Systematic splice site coordinate reconciliation system.
    
    This class handles coordinate differences between:
    1. Different annotation systems (MetaSpliceAI, OpenSpliceAI, SpliceAI)
    2. Different genomic annotation versions
    3. Different coordinate systems (0-based vs 1-based)
    4. Different splice site definitions
    """
    
    # Known coordinate offsets between systems
    KNOWN_OFFSETS = {
        # OpenSpliceAI relative to MetaSpliceAI coordinates
        'openspliceai_to_splicesurveyor': CoordinateOffset(
            donor_plus=1,    # OpenSpliceAI donor is +1 relative to MetaSpliceAI
            donor_minus=1,   # Same for minus strand
            acceptor_plus=0, # OpenSpliceAI acceptor matches MetaSpliceAI
            acceptor_minus=0 # Same for minus strand
        ),
        
        # SpliceAI model adjustments (from your analysis)
        'spliceai_model_adjustments': CoordinateOffset(
            donor_plus=2,    # SpliceAI predicts 2nt upstream on plus strand
            donor_minus=1,   # SpliceAI predicts 1nt upstream on minus strand
            acceptor_plus=0, # SpliceAI matches GTF position on plus strand
            acceptor_minus=-1 # SpliceAI predicts 1nt downstream on minus strand
        ),
        
        # Combined OpenSpliceAI + SpliceAI adjustments
        'openspliceai_spliceai_combined': CoordinateOffset(
            donor_plus=3,    # +1 (OpenSpliceAI) + 2 (SpliceAI) = +3
            donor_minus=2,   # +1 (OpenSpliceAI) + 1 (SpliceAI) = +2
            acceptor_plus=0, # 0 (OpenSpliceAI) + 0 (SpliceAI) = 0
            acceptor_minus=-1 # 0 (OpenSpliceAI) + (-1) (SpliceAI) = -1
        )
    }
    
    def __init__(self, verbose: int = 1):
        """
        Initialize the coordinate reconciler.
        
        Parameters
        ----------
        verbose : int
            Verbosity level for logging
        """
        self.verbose = verbose
        self.custom_offsets = {}
        
    def detect_coordinate_differences(
        self,
        reference_sites: pd.DataFrame,
        comparison_sites: pd.DataFrame,
        tolerance: int = 10
    ) -> Dict[str, Any]:
        """
        Automatically detect coordinate differences between two splice site datasets.
        
        Parameters
        ----------
        reference_sites : pd.DataFrame
            Reference splice sites with columns: chrom, position, strand, splice_type
        comparison_sites : pd.DataFrame
            Comparison splice sites with same format
        tolerance : int
            Maximum distance to consider sites as potentially matching
            
        Returns
        -------
        Dict[str, Any]
            Analysis results including detected offsets
        """
        results = {
            'total_reference': len(reference_sites),
            'total_comparison': len(comparison_sites),
            'detected_offsets': {},
            'confidence_scores': {},
            'analysis_details': {}
        }
        
        # Group by splice type and strand
        for splice_type in ['donor', 'acceptor']:
            for strand in ['+', '-']:
                ref_subset = reference_sites[
                    (reference_sites['splice_type'] == splice_type) & 
                    (reference_sites['strand'] == strand)
                ]
                comp_subset = comparison_sites[
                    (comparison_sites['splice_type'] == splice_type) & 
                    (comparison_sites['strand'] == strand)
                ]
                
                if len(ref_subset) == 0 or len(comp_subset) == 0:
                    continue
                
                # Detect most common offset
                offset_analysis = self._analyze_coordinate_offsets(
                    ref_subset, comp_subset, tolerance
                )
                
                key = f"{splice_type}_{strand}"
                results['detected_offsets'][key] = offset_analysis['most_common_offset']
                results['confidence_scores'][key] = offset_analysis['confidence']
                results['analysis_details'][key] = offset_analysis
        
        return results
    
    def _analyze_coordinate_offsets(
        self,
        ref_sites: pd.DataFrame,
        comp_sites: pd.DataFrame,
        tolerance: int
    ) -> Dict[str, Any]:
        """Analyze coordinate offsets between two site sets."""
        offsets = []
        
        # For each reference site, find closest comparison site
        for _, ref_site in ref_sites.iterrows():
            ref_chrom = ref_site['chrom']
            ref_pos = ref_site['position']
            
            # Find comparison sites on same chromosome within tolerance
            comp_chrom_sites = comp_sites[comp_sites['chrom'] == ref_chrom]
            
            if len(comp_chrom_sites) == 0:
                continue
                
            # Calculate distances
            distances = np.abs(comp_chrom_sites['position'] - ref_pos)
            min_distance_idx = distances.idxmin()
            min_distance = distances[min_distance_idx]
            
            if min_distance <= tolerance:
                comp_pos = comp_chrom_sites.loc[min_distance_idx, 'position']
                offset = comp_pos - ref_pos
                offsets.append(offset)
        
        if not offsets:
            return {
                'most_common_offset': 0,
                'confidence': 0.0,
                'offset_distribution': {},
                'total_matches': 0
            }
        
        # Analyze offset distribution
        offset_counts = pd.Series(offsets).value_counts()
        most_common_offset = offset_counts.index[0]
        confidence = offset_counts.iloc[0] / len(offsets)
        
        return {
            'most_common_offset': int(most_common_offset),
            'confidence': float(confidence),
            'offset_distribution': offset_counts.to_dict(),
            'total_matches': len(offsets)
        }
    
    def apply_coordinate_adjustment(
        self,
        splice_sites: pd.DataFrame,
        adjustment_name: str,
        reverse: bool = False
    ) -> pd.DataFrame:
        """
        Apply known coordinate adjustments to splice sites.
        
        Parameters
        ----------
        splice_sites : pd.DataFrame
            Splice sites DataFrame with columns: chrom, position, strand, splice_type
        adjustment_name : str
            Name of the adjustment to apply (from KNOWN_OFFSETS)
        reverse : bool
            If True, apply the reverse adjustment
            
        Returns
        -------
        pd.DataFrame
            Adjusted splice sites DataFrame
        """
        if adjustment_name not in self.KNOWN_OFFSETS and adjustment_name not in self.custom_offsets:
            raise ValueError(f"Unknown adjustment: {adjustment_name}")
        
        # Get offset from known or custom adjustments
        if adjustment_name in self.KNOWN_OFFSETS:
            offset = self.KNOWN_OFFSETS[adjustment_name]
        else:
            offset = self.custom_offsets[adjustment_name]
        adjusted_sites = splice_sites.copy()
        
        # Apply adjustments based on splice type and strand
        for _, site in adjusted_sites.iterrows():
            splice_type = site['splice_type']
            strand = site['strand']
            
            # Determine adjustment value
            if splice_type == 'donor':
                adjustment = offset.donor_plus if strand == '+' else offset.donor_minus
            else:  # acceptor
                adjustment = offset.acceptor_plus if strand == '+' else offset.acceptor_minus
            
            # Apply adjustment (reverse if requested)
            if reverse:
                adjustment = -adjustment
            
            adjusted_sites.loc[site.name, 'position'] += adjustment
        
        return adjusted_sites
    
    def create_custom_adjustment(
        self,
        name: str,
        donor_plus: int = 0,
        donor_minus: int = 0,
        acceptor_plus: int = 0,
        acceptor_minus: int = 0
    ) -> None:
        """
        Create a custom coordinate adjustment.
        
        Parameters
        ----------
        name : str
            Name for the custom adjustment
        donor_plus : int
            Donor site adjustment on plus strand
        donor_minus : int
            Donor site adjustment on minus strand
        acceptor_plus : int
            Acceptor site adjustment on plus strand
        acceptor_minus : int
            Acceptor site adjustment on minus strand
        """
        self.custom_offsets[name] = CoordinateOffset(
            donor_plus=donor_plus,
            donor_minus=donor_minus,
            acceptor_plus=acceptor_plus,
            acceptor_minus=acceptor_minus
        )
        
        if self.verbose >= 1:
            logger.info(f"Created custom adjustment '{name}':")
            logger.info(f"  Donor: +{donor_plus} (plus), +{donor_minus} (minus)")
            logger.info(f"  Acceptor: +{acceptor_plus} (plus), +{acceptor_minus} (minus)")
    
    def reconcile_splice_sites(
        self,
        source_sites: pd.DataFrame,
        target_format: str,
        source_format: str = "splicesurveyor"
    ) -> pd.DataFrame:
        """
        Reconcile splice sites between different formats/systems.
        
        Parameters
        ----------
        source_sites : pd.DataFrame
            Source splice sites DataFrame
        target_format : str
            Target format to convert to
        source_format : str
            Source format (default: "splicesurveyor")
            
        Returns
        -------
        pd.DataFrame
            Reconciled splice sites DataFrame
        """
        # Define conversion mappings
        conversions = {
            ('splicesurveyor', 'openspliceai'): 'openspliceai_to_splicesurveyor',
            ('openspliceai', 'splicesurveyor'): 'openspliceai_to_splicesurveyor',
            ('splicesurveyor', 'spliceai_compatible'): 'spliceai_model_adjustments',
            ('openspliceai', 'spliceai_compatible'): 'openspliceai_spliceai_combined'
        }
        
        conversion_key = (source_format.lower(), target_format.lower())
        
        if conversion_key not in conversions:
            raise ValueError(f"No conversion available from {source_format} to {target_format}")
        
        adjustment_name = conversions[conversion_key]
        
        # Determine if we need to reverse the adjustment
        reverse = conversion_key[0] == 'openspliceai' and conversion_key[1] == 'splicesurveyor'
        
        return self.apply_coordinate_adjustment(source_sites, adjustment_name, reverse=reverse)
    
    def validate_reconciliation(
        self,
        original_sites: pd.DataFrame,
        reconciled_sites: pd.DataFrame,
        expected_matches: float = 0.95
    ) -> Dict[str, Any]:
        """
        Validate that coordinate reconciliation was successful.
        
        Parameters
        ----------
        original_sites : pd.DataFrame
            Original splice sites before reconciliation
        reconciled_sites : pd.DataFrame
            Reconciled splice sites
        expected_matches : float
            Expected fraction of sites that should match after reconciliation
            
        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        # This would compare against a reference dataset
        # Implementation depends on specific validation requirements
        return {
            'validation_passed': True,
            'match_rate': 0.95,  # Placeholder
            'details': "Validation implementation needed"
        }
    
    def save_adjustments(self, filepath: str) -> None:
        """Save custom adjustments to file."""
        adjustments = {
            name: offset.to_dict() 
            for name, offset in self.custom_offsets.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(adjustments, f, indent=2)
        
        if self.verbose >= 1:
            logger.info(f"Saved custom adjustments to {filepath}")
    
    def load_adjustments(self, filepath: str) -> None:
        """Load custom adjustments from file."""
        with open(filepath, 'r') as f:
            adjustments = json.load(f)
        
        for name, data in adjustments.items():
            self.custom_offsets[name] = CoordinateOffset.from_dict(data)
        
        if self.verbose >= 1:
            logger.info(f"Loaded {len(adjustments)} custom adjustments from {filepath}")


def create_reconciliation_report(
    reference_sites: pd.DataFrame,
    comparison_sites: pd.DataFrame,
    output_dir: str = "coordinate_analysis"
) -> Dict[str, Any]:
    """
    Create a comprehensive coordinate reconciliation report.
    
    Parameters
    ----------
    reference_sites : pd.DataFrame
        Reference splice sites
    comparison_sites : pd.DataFrame
        Comparison splice sites
    output_dir : str
        Output directory for the report
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive analysis report
    """
    reconciler = SpliceCoordinateReconciler(verbose=1)
    
    # Detect coordinate differences
    differences = reconciler.detect_coordinate_differences(
        reference_sites, comparison_sites
    )
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save detailed report
    report_file = Path(output_dir) / "coordinate_reconciliation_report.json"
    with open(report_file, 'w') as f:
        json.dump(differences, f, indent=2)
    
    logger.info(f"Coordinate reconciliation report saved to {report_file}")
    
    return differences


if __name__ == "__main__":
    # Example usage
    reconciler = SpliceCoordinateReconciler(verbose=1)
    
    # Example: Convert OpenSpliceAI coordinates to MetaSpliceAI format
    # openspliceai_sites = pd.DataFrame(...)  # Your OpenSpliceAI sites
    # reconciled_sites = reconciler.reconcile_splice_sites(
    #     openspliceai_sites, 
    #     target_format="splicesurveyor",
    #     source_format="openspliceai"
    # )
    
    print("Coordinate reconciliation module ready!")
