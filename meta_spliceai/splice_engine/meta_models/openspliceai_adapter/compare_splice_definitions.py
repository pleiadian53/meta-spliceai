#!/usr/bin/env python3
"""
Splice Site Definition Comparison: MetaSpliceAI vs OpenSpliceAI

This script analyzes and compares the exact splice site position definitions
used by your MetaSpliceAI workflow versus OpenSpliceAI, identifying the
specific coordinate differences that require adjustment.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

# Add the openspliceai_adapter to path
sys.path.insert(0, str(Path(__file__).parent))

class SpliceSiteDefinitionAnalyzer:
    """
    Analyzes splice site coordinate definitions and identifies differences
    between MetaSpliceAI and OpenSpliceAI approaches.
    """
    
    def __init__(self, verbose: int = 1):
        self.verbose = verbose
        
        # Known adjustments from your workflow
        self.known_spliceai_adjustments = {
            'donor': {
                'plus': 2,   # SpliceAI predicts donor sites 2nt upstream on + strand
                'minus': 1   # SpliceAI predicts donor sites 1nt upstream on - strand
            },
            'acceptor': {
                'plus': 0,   # SpliceAI predicts acceptor sites at exact position on + strand
                'minus': -1  # SpliceAI predicts acceptor sites 1nt downstream on - strand
            }
        }
    
    def analyze_splice_site_definitions(self) -> Dict[str, Any]:
        """
        Analyze the splice site coordinate definitions used by different systems.
        """
        
        print("="*80)
        print("SPLICE SITE COORDINATE DEFINITION ANALYSIS")
        print("="*80)
        
        analysis = {
            'coordinate_systems': {},
            'splice_site_definitions': {},
            'adjustment_requirements': {},
            'compatibility_assessment': {}
        }
        
        # 1. Coordinate System Analysis
        print("\n1. COORDINATE SYSTEM COMPARISON")
        print("-" * 40)
        
        coordinate_systems = {
            'GTF_Standard': {
                'description': 'GTF/GFF3 standard (your splice_sites.tsv)',
                'base': '1-based',
                'donor_definition': 'Last nucleotide of exon (GT dinucleotide start)',
                'acceptor_definition': 'First nucleotide of exon (AG dinucleotide end)',
                'strand_handling': 'Coordinates always in genomic orientation'
            },
            'SpliceAI_Original': {
                'description': 'Original SpliceAI model predictions',
                'base': '0-based (internal)',
                'donor_definition': 'Systematic offset: -2nt (+strand), -1nt (-strand)',
                'acceptor_definition': 'Exact position (+strand), +1nt (-strand)',
                'strand_handling': 'Predictions require adjustment'
            },
            'OpenSpliceAI': {
                'description': 'OpenSpliceAI implementation',
                'base': '0-based (Python)',
                'donor_definition': 'One base after end of current exon',
                'acceptor_definition': 'At the start of next exon',
                'strand_handling': 'Strand-aware coordinate calculation'
            }
        }
        
        analysis['coordinate_systems'] = coordinate_systems
        
        for system, details in coordinate_systems.items():
            print(f"\n{system}:")
            print(f"  Base: {details['base']}")
            print(f"  Donor: {details['donor_definition']}")
            print(f"  Acceptor: {details['acceptor_definition']}")
            print(f"  Strand: {details['strand_handling']}")
        
        # 2. Splice Site Biology Review
        print("\n\n2. SPLICE SITE BIOLOGY REFERENCE")
        print("-" * 40)
        
        biology_reference = {
            'canonical_motifs': {
                'donor': 'GT (5\' splice site)',
                'acceptor': 'AG (3\' splice site)'
            },
            'coordinate_definition': {
                'donor_exact': 'Position of G in GT dinucleotide',
                'acceptor_exact': 'Position of G in AG dinucleotide',
                'note': 'These are the biologically functional positions'
            },
            'strand_considerations': {
                'plus_strand': 'Sequence read 5\' → 3\' (left to right)',
                'minus_strand': 'Sequence read 3\' → 5\' (right to left, reverse complement)',
                'coordinate_impact': 'Minus strand requires coordinate transformation'
            }
        }
        
        analysis['splice_site_definitions'] = biology_reference
        
        print("Canonical Motifs:")
        print(f"  Donor (5' splice site): {biology_reference['canonical_motifs']['donor']}")
        print(f"  Acceptor (3' splice site): {biology_reference['canonical_motifs']['acceptor']}")
        
        print("\nBiological Coordinates:")
        print(f"  Donor: {biology_reference['coordinate_definition']['donor_exact']}")
        print(f"  Acceptor: {biology_reference['coordinate_definition']['acceptor_exact']}")
        
        # 3. OpenSpliceAI Implementation Analysis
        print("\n\n3. OPENSPLICEAI COORDINATE CALCULATION")
        print("-" * 40)
        
        openspliceai_logic = self.analyze_openspliceai_coordinates()
        analysis['openspliceai_implementation'] = openspliceai_logic
        
        print("OpenSpliceAI Logic (from create_datafile.py):")
        print("```python")
        print("# Donor site is one base after the end of the current exon")
        print("first_site = exons[i].end - gene.start  # Adjusted for python indexing")
        print("# Acceptor site is at the start of the next exon")
        print("second_site = exons[i + 1].start - gene.start")
        print("")
        print("if gene.strand == '+':") 
        print("    d_idx = first_site")
        print("    a_idx = second_site")
        print("elif gene.strand == '-':")
        print("    d_idx = len(labels) - second_site - 1")
        print("    a_idx = len(labels) - first_site - 1")
        print("```")
        
        # 4. Your Workflow Adjustments Analysis
        print("\n\n4. YOUR WORKFLOW ADJUSTMENT REQUIREMENTS")
        print("-" * 40)
        
        adjustment_analysis = self.analyze_adjustment_requirements()
        analysis['adjustment_requirements'] = adjustment_analysis
        
        print("Required Adjustments (from your splice_utils.py):")
        for splice_type in ['donor', 'acceptor']:
            print(f"\n{splice_type.upper()} sites:")
            for strand in ['plus', 'minus']:
                offset = self.known_spliceai_adjustments[splice_type][strand]
                direction = "downstream" if offset > 0 else "upstream" if offset < 0 else "exact"
                print(f"  {strand} strand: {offset:+d} nt ({direction})")
        
        # 5. Compatibility Assessment
        print("\n\n5. COMPATIBILITY ASSESSMENT")
        print("-" * 40)
        
        compatibility = self.assess_compatibility()
        analysis['compatibility_assessment'] = compatibility
        
        if compatibility['requires_adjustment']:
            print("⚠️  ADJUSTMENT REQUIRED")
            print(f"   Reason: {compatibility['reason']}")
            print(f"   Impact: {compatibility['impact']}")
        else:
            print("✅ FULLY COMPATIBLE")
            print(f"   Reason: {compatibility['reason']}")
        
        print(f"\nRecommendation: {compatibility['recommendation']}")
        
        return analysis
    
    def analyze_openspliceai_coordinates(self) -> Dict[str, Any]:
        """Analyze OpenSpliceAI coordinate calculation logic."""
        
        return {
            'donor_calculation': {
                'description': 'One base after end of current exon',
                'formula_plus': 'exons[i].end - gene.start',
                'formula_minus': 'len(labels) - (exons[i+1].start - gene.start) - 1',
                'biological_meaning': 'Position of G in GT dinucleotide'
            },
            'acceptor_calculation': {
                'description': 'At the start of next exon',
                'formula_plus': 'exons[i+1].start - gene.start',
                'formula_minus': 'len(labels) - (exons[i].end - gene.start) - 1',
                'biological_meaning': 'Position of G in AG dinucleotide'
            },
            'coordinate_system': {
                'base': '0-based (Python indexing)',
                'reference': 'Gene start position',
                'strand_handling': 'Reverse complement for minus strand'
            }
        }
    
    def analyze_adjustment_requirements(self) -> Dict[str, Any]:
        """Analyze why adjustments are needed in your workflow."""
        
        return {
            'source_of_discrepancy': 'SpliceAI model training vs GTF annotation differences',
            'systematic_offsets': self.known_spliceai_adjustments,
            'biological_explanation': {
                'donor_plus_2nt': 'SpliceAI predicts 2nt upstream of true GT position on + strand',
                'donor_minus_1nt': 'SpliceAI predicts 1nt upstream of true GT position on - strand',
                'acceptor_plus_0nt': 'SpliceAI predicts exact AG position on + strand',
                'acceptor_minus_neg1nt': 'SpliceAI predicts 1nt downstream of true AG position on - strand'
            },
            'adjustment_necessity': 'Required to align SpliceAI predictions with GTF coordinates'
        }
    
    def assess_compatibility(self) -> Dict[str, Any]:
        """Assess compatibility between OpenSpliceAI and your workflow."""
        
        return {
            'requires_adjustment': True,
            'reason': 'OpenSpliceAI uses different coordinate calculation than your GTF-derived annotations',
            'impact': 'Position offsets may differ between OpenSpliceAI and your current workflow',
            'recommendation': 'Use the format compatibility adapter to ensure consistent coordinate systems',
            'solution': 'The adapter handles coordinate transformation automatically'
        }
    
    def create_test_comparison(self, output_dir: str = "splice_definition_test") -> Dict[str, Any]:
        """Create a test to compare splice site definitions with actual data."""
        
        print("\n\n6. CREATING TEST COMPARISON")
        print("-" * 40)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a synthetic example to illustrate the differences
        test_case = {
            'gene_info': {
                'gene_id': 'ENSG00000TEST',
                'chromosome': '1',
                'strand': '+',
                'gene_start': 1000000,
                'gene_end': 1010000
            },
            'exon_structure': [
                {'start': 1000000, 'end': 1001000},  # Exon 1
                {'start': 1005000, 'end': 1006000},  # Exon 2
                {'start': 1008000, 'end': 1010000}   # Exon 3
            ]
        }
        
        # Calculate splice sites using different methods
        splice_sites = self.calculate_splice_sites_different_methods(test_case)
        
        # Save test results
        test_file = os.path.join(output_dir, "splice_definition_comparison.json")
        with open(test_file, 'w') as f:
            json.dump({
                'test_case': test_case,
                'splice_sites': splice_sites,
                'coordinate_systems': self.coordinate_systems_summary()
            }, f, indent=2)
        
        print(f"Test comparison saved: {test_file}")
        
        # Display results
        print("\nSPLICE SITE COORDINATE COMPARISON:")
        print("Exon Structure: [1000000-1001000] --- [1005000-1006000] --- [1008000-1010000]")
        print("                     Exon1              Exon2              Exon3")
        print("                          |donor1  acceptor2|donor2  acceptor3|")
        print("")
        
        for method, sites in splice_sites.items():
            print(f"{method}:")
            for site in sites:
                print(f"  {site['type']} at position {site['position']} (genomic: {site['genomic_pos']})")
        
        return {
            'test_file': test_file,
            'splice_sites': splice_sites,
            'test_case': test_case
        }
    
    def calculate_splice_sites_different_methods(self, test_case: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Calculate splice sites using different coordinate systems."""
        
        gene_start = test_case['gene_info']['gene_start']
        exons = test_case['exon_structure']
        
        methods = {}
        
        # Method 1: GTF Standard (your current approach)
        gtf_sites = []
        for i in range(len(exons) - 1):
            # Donor: last position of current exon
            donor_pos = exons[i]['end']
            gtf_sites.append({
                'type': 'donor',
                'position': donor_pos - gene_start,  # Relative to gene
                'genomic_pos': donor_pos,
                'exon_boundary': f"exon{i+1}_end"
            })
            
            # Acceptor: first position of next exon
            acceptor_pos = exons[i+1]['start']
            gtf_sites.append({
                'type': 'acceptor', 
                'position': acceptor_pos - gene_start,  # Relative to gene
                'genomic_pos': acceptor_pos,
                'exon_boundary': f"exon{i+2}_start"
            })
        
        methods['GTF_Standard'] = gtf_sites
        
        # Method 2: OpenSpliceAI approach
        openspliceai_sites = []
        for i in range(len(exons) - 1):
            # Donor: one base after end of current exon (0-based)
            first_site = exons[i]['end'] - gene_start
            openspliceai_sites.append({
                'type': 'donor',
                'position': first_site,
                'genomic_pos': exons[i]['end'],
                'exon_boundary': f"exon{i+1}_end+1"
            })
            
            # Acceptor: at start of next exon (0-based)
            second_site = exons[i+1]['start'] - gene_start
            openspliceai_sites.append({
                'type': 'acceptor',
                'position': second_site,
                'genomic_pos': exons[i+1]['start'],
                'exon_boundary': f"exon{i+2}_start"
            })
        
        methods['OpenSpliceAI'] = openspliceai_sites
        
        # Method 3: SpliceAI with adjustments
        spliceai_sites = []
        for site in gtf_sites:
            adjusted_site = site.copy()
            site_type = site['type']
            strand = 'plus'  # Test case uses + strand
            
            adjustment = self.known_spliceai_adjustments[site_type][strand]
            adjusted_site['position'] = site['position'] + adjustment
            adjusted_site['genomic_pos'] = site['genomic_pos'] + adjustment
            adjusted_site['adjustment'] = f"{adjustment:+d}nt"
            
            spliceai_sites.append(adjusted_site)
        
        methods['SpliceAI_Adjusted'] = spliceai_sites
        
        return methods
    
    def coordinate_systems_summary(self) -> Dict[str, Any]:
        """Provide a summary of coordinate system differences."""
        
        return {
            'key_differences': {
                'base_indexing': {
                    'GTF': '1-based',
                    'OpenSpliceAI': '0-based',
                    'impact': 'Off-by-one differences in position calculation'
                },
                'splice_site_definition': {
                    'GTF': 'Exact exon boundaries',
                    'OpenSpliceAI': 'Biologically functional positions',
                    'SpliceAI': 'Model prediction positions (with systematic offsets)'
                },
                'strand_handling': {
                    'GTF': 'Genomic coordinates',
                    'OpenSpliceAI': 'Sequence-relative with strand transformation',
                    'SpliceAI': 'Requires post-prediction adjustment'
                }
            },
            'compatibility_solution': 'Use format adapter for automatic coordinate transformation'
        }


def main():
    """Main analysis script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare splice site coordinate definitions")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='splice_definition_analysis',
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (0=quiet, 1=normal, 2=verbose)'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = SpliceSiteDefinitionAnalyzer(verbose=args.verbose)
    
    # Perform comprehensive analysis
    analysis_results = analyzer.analyze_splice_site_definitions()
    
    # Create test comparison
    test_results = analyzer.create_test_comparison(args.output_dir)
    
    # Save complete analysis
    analysis_file = os.path.join(args.output_dir, "complete_splice_definition_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump({
            'analysis': analysis_results,
            'test_comparison': test_results,
            'summary': {
                'main_finding': 'Coordinate system differences require format adapter',
                'recommendation': 'Use OpenSpliceAI adapter for automatic handling',
                'compatibility': 'Fully compatible with adapter'
            }
        }, f, indent=2, default=str)
    
    print(f"\n\nComplete analysis saved: {analysis_file}")
    print(f"Test comparison saved: {test_results['test_file']}")
    
    return analysis_results


if __name__ == "__main__":
    main()
