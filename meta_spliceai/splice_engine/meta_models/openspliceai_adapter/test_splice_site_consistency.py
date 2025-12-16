#!/usr/bin/env python3
"""
Practical Test: Splice Site Consistency Between MetaSpliceAI and OpenSpliceAI

This script performs a concrete comparison using your actual splice_sites.tsv
data to identify any position differences that would arise from using OpenSpliceAI
preprocessing versus your current GTF-derived annotations.
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

class PracticalSpliceSiteComparison:
    """
    Performs practical comparison of splice site positions using real data.
    """
    
    def __init__(self, verbose: int = 1):
        self.verbose = verbose
        
        # Your known SpliceAI adjustments
        self.spliceai_adjustments = {
            'donor': {'plus': 2, 'minus': 1},
            'acceptor': {'plus': 0, 'minus': -1}
        }
    
    def load_and_analyze_splice_sites(self, splice_sites_file: str) -> Dict[str, Any]:
        """Load your splice_sites.tsv and analyze the coordinate patterns."""
        
        if self.verbose >= 1:
            print("="*60)
            print("PRACTICAL SPLICE SITE CONSISTENCY TEST")
            print("="*60)
            print(f"\nLoading splice sites from: {splice_sites_file}")
        
        # Load the data
        df = pd.read_csv(splice_sites_file, sep='\t')
        
        if self.verbose >= 1:
            print(f"Loaded {len(df)} splice sites")
            print(f"Site types: {df['site_type'].value_counts().to_dict()}")
            print(f"Strands: {df['strand'].value_counts().to_dict()}")
            print(f"Chromosomes: {len(df['chrom'].unique())} unique")
        
        # Analyze coordinate patterns
        analysis = {
            'total_sites': len(df),
            'site_distribution': df['site_type'].value_counts().to_dict(),
            'strand_distribution': df['strand'].value_counts().to_dict(),
            'chromosome_distribution': df['chrom'].value_counts().head(10).to_dict(),
            'coordinate_analysis': self.analyze_coordinate_patterns(df),
            'sample_sites': self.extract_sample_sites(df)
        }
        
        return analysis
    
    def analyze_coordinate_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze coordinate patterns in your splice site data."""
        
        patterns = {}
        
        # Analyze position vs start/end relationships
        df['position_vs_start'] = df['position'] - df['start']
        df['position_vs_end'] = df['position'] - df['end']
        
        # Group by site type and strand
        for site_type in ['donor', 'acceptor']:
            patterns[site_type] = {}
            site_df = df[df['site_type'] == site_type]
            
            for strand in ['+', '-']:
                strand_df = site_df[site_df['strand'] == strand]
                if len(strand_df) > 0:
                    patterns[site_type][strand] = {
                        'count': len(strand_df),
                        'position_vs_start_stats': {
                            'mean': strand_df['position_vs_start'].mean(),
                            'std': strand_df['position_vs_start'].std(),
                            'min': strand_df['position_vs_start'].min(),
                            'max': strand_df['position_vs_start'].max()
                        },
                        'position_vs_end_stats': {
                            'mean': strand_df['position_vs_end'].mean(),
                            'std': strand_df['position_vs_end'].std(),
                            'min': strand_df['position_vs_end'].min(),
                            'max': strand_df['position_vs_end'].max()
                        }
                    }
        
        return patterns
    
    def extract_sample_sites(self, df: pd.DataFrame, n_samples: int = 10) -> Dict[str, List[Dict]]:
        """Extract sample splice sites for detailed analysis."""
        
        samples = {}
        
        # Get samples for each site type and strand combination
        for site_type in ['donor', 'acceptor']:
            samples[site_type] = {}
            site_df = df[df['site_type'] == site_type]
            
            for strand in ['+', '-']:
                strand_df = site_df[site_df['strand'] == strand]
                if len(strand_df) > 0:
                    sample_df = strand_df.head(n_samples)
                    samples[site_type][strand] = sample_df.to_dict('records')
        
        return samples
    
    def simulate_openspliceai_coordinates(self, sample_sites: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate what OpenSpliceAI coordinate calculation would produce
        for the same splice sites.
        """
        
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("SIMULATING OPENSPLICEAI COORDINATE CALCULATION")
            print("="*60)
        
        simulated = {}
        
        for site_type in ['donor', 'acceptor']:
            simulated[site_type] = {}
            
            for strand in ['+', '-']:
                if strand in sample_sites[site_type]:
                    original_sites = sample_sites[site_type][strand]
                    simulated_sites = []
                    
                    for site in original_sites:
                        # Simulate OpenSpliceAI calculation
                        openspliceai_pos = self.calculate_openspliceai_position(site, site_type, strand)
                        
                        simulated_site = site.copy()
                        simulated_site['openspliceai_position'] = openspliceai_pos
                        simulated_site['position_difference'] = openspliceai_pos - site['position']
                        
                        simulated_sites.append(simulated_site)
                    
                    simulated[site_type][strand] = simulated_sites
        
        return simulated
    
    def calculate_openspliceai_position(self, site: Dict[str, Any], site_type: str, strand: str) -> int:
        """
        Calculate what OpenSpliceAI would produce for this splice site.
        
        Based on OpenSpliceAI logic:
        - Donor: one base after end of current exon
        - Acceptor: at start of next exon
        """
        
        # For this simulation, we'll use the GTF coordinates and apply
        # the known differences between GTF and OpenSpliceAI approaches
        
        original_pos = site['position']
        
        if site_type == 'donor':
            # OpenSpliceAI: "one base after end of current exon"
            # Your GTF: "last nucleotide of exon"
            # Difference: OpenSpliceAI is typically +1 from GTF for donors
            openspliceai_pos = original_pos + 1
        else:  # acceptor
            # OpenSpliceAI: "at start of next exon"
            # Your GTF: "first nucleotide of exon"
            # These should be the same
            openspliceai_pos = original_pos
        
        return openspliceai_pos
    
    def analyze_position_differences(self, simulated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the position differences between your data and OpenSpliceAI simulation."""
        
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("POSITION DIFFERENCE ANALYSIS")
            print("="*60)
        
        differences = {}
        
        for site_type in ['donor', 'acceptor']:
            differences[site_type] = {}
            
            for strand in ['+', '-']:
                if strand in simulated_data[site_type]:
                    sites = simulated_data[site_type][strand]
                    
                    position_diffs = [site['position_difference'] for site in sites]
                    
                    differences[site_type][strand] = {
                        'sample_count': len(sites),
                        'position_differences': position_diffs,
                        'mean_difference': np.mean(position_diffs),
                        'std_difference': np.std(position_diffs),
                        'unique_differences': list(set(position_diffs)),
                        'consistent_offset': len(set(position_diffs)) == 1
                    }
                    
                    if self.verbose >= 1:
                        print(f"\n{site_type.upper()} sites on {strand} strand:")
                        print(f"  Sample count: {len(sites)}")
                        print(f"  Position differences: {position_diffs}")
                        print(f"  Mean difference: {np.mean(position_diffs):.2f}")
                        print(f"  Consistent offset: {len(set(position_diffs)) == 1}")
        
        return differences
    
    def compare_with_spliceai_adjustments(self, differences: Dict[str, Any]) -> Dict[str, Any]:
        """Compare the observed differences with your known SpliceAI adjustments."""
        
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("COMPARISON WITH SPLICEAI ADJUSTMENTS")
            print("="*60)
        
        comparison = {}
        
        for site_type in ['donor', 'acceptor']:
            comparison[site_type] = {}
            
            for strand in ['+', '-']:
                strand_key = 'plus' if strand == '+' else 'minus'
                
                if strand in differences[site_type]:
                    observed_diff = differences[site_type][strand]['mean_difference']
                    expected_adjustment = self.spliceai_adjustments[site_type][strand_key]
                    
                    comparison[site_type][strand] = {
                        'observed_openspliceai_difference': observed_diff,
                        'known_spliceai_adjustment': expected_adjustment,
                        'total_expected_difference': observed_diff + expected_adjustment,
                        'interpretation': self.interpret_difference(observed_diff, expected_adjustment)
                    }
                    
                    if self.verbose >= 1:
                        print(f"\n{site_type.upper()} sites on {strand} strand:")
                        print(f"  OpenSpliceAI vs Your GTF: {observed_diff:+.1f} nt")
                        print(f"  Your SpliceAI adjustment: {expected_adjustment:+d} nt")
                        print(f"  Net difference: {observed_diff + expected_adjustment:+.1f} nt")
                        print(f"  Interpretation: {comparison[site_type][strand]['interpretation']}")
        
        return comparison
    
    def interpret_difference(self, openspliceai_diff: float, spliceai_adjustment: int) -> str:
        """Interpret the combined difference."""
        
        total_diff = openspliceai_diff + spliceai_adjustment
        
        if abs(total_diff) < 0.5:
            return "Excellent agreement - positions are essentially identical"
        elif abs(total_diff) < 1.5:
            return "Good agreement - minor offset that can be easily handled"
        elif abs(total_diff) < 2.5:
            return "Moderate difference - requires careful coordinate mapping"
        else:
            return "Significant difference - needs thorough validation"
    
    def generate_recommendations(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on the analysis."""
        
        recommendations = {
            'overall_compatibility': 'good',
            'action_required': [],
            'format_adapter_needed': True,
            'specific_recommendations': {}
        }
        
        max_difference = 0
        
        for site_type in ['donor', 'acceptor']:
            for strand in ['+', '-']:
                if strand in comparison[site_type]:
                    total_diff = abs(comparison[site_type][strand]['total_expected_difference'])
                    max_difference = max(max_difference, total_diff)
                    
                    if total_diff > 1.5:
                        recommendations['action_required'].append(
                            f"Validate {site_type} sites on {strand} strand (difference: {total_diff:.1f}nt)"
                        )
        
        if max_difference < 1.0:
            recommendations['overall_compatibility'] = 'excellent'
        elif max_difference < 2.0:
            recommendations['overall_compatibility'] = 'good'
        else:
            recommendations['overall_compatibility'] = 'requires_attention'
        
        recommendations['specific_recommendations'] = {
            'use_format_adapter': True,
            'validate_coordinates': max_difference > 1.0,
            'test_with_subset': True,
            'monitor_predictions': max_difference > 0.5
        }
        
        return recommendations
    
    def run_comprehensive_test(
        self,
        splice_sites_file: str,
        output_dir: str = "splice_consistency_test"
    ) -> Dict[str, Any]:
        """Run comprehensive splice site consistency test."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Load and analyze your data
        analysis = self.load_and_analyze_splice_sites(splice_sites_file)
        
        # Step 2: Simulate OpenSpliceAI coordinates
        simulated = self.simulate_openspliceai_coordinates(analysis['sample_sites'])
        
        # Step 3: Analyze position differences
        differences = self.analyze_position_differences(simulated)
        
        # Step 4: Compare with SpliceAI adjustments
        comparison = self.compare_with_spliceai_adjustments(differences)
        
        # Step 5: Generate recommendations
        recommendations = self.generate_recommendations(comparison)
        
        # Compile results
        results = {
            'input_analysis': analysis,
            'openspliceai_simulation': simulated,
            'position_differences': differences,
            'spliceai_comparison': comparison,
            'recommendations': recommendations,
            'summary': {
                'total_sites_analyzed': analysis['total_sites'],
                'overall_compatibility': recommendations['overall_compatibility'],
                'format_adapter_recommended': recommendations['format_adapter_needed'],
                'action_items': recommendations['action_required']
            }
        }
        
        # Save results
        results_file = os.path.join(output_dir, "splice_site_consistency_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("FINAL RECOMMENDATIONS")
            print("="*60)
            print(f"Overall compatibility: {recommendations['overall_compatibility'].upper()}")
            print(f"Format adapter needed: {'YES' if recommendations['format_adapter_needed'] else 'NO'}")
            
            if recommendations['action_required']:
                print(f"\nAction items:")
                for i, action in enumerate(recommendations['action_required'], 1):
                    print(f"  {i}. {action}")
            else:
                print("\nNo specific action items - good compatibility!")
            
            print(f"\nDetailed results saved: {results_file}")
        
        return results


def main():
    """Main test script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test splice site consistency between MetaSpliceAI and OpenSpliceAI")
    parser.add_argument(
        '--splice-sites-file',
        type=str,
        default='data/ensembl/splice_sites.tsv',
        help='Path to your splice_sites.tsv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='splice_consistency_test',
        help='Output directory for test results'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (0=quiet, 1=normal, 2=verbose)'
    )
    
    args = parser.parse_args()
    
    # Run test
    tester = PracticalSpliceSiteComparison(verbose=args.verbose)
    results = tester.run_comprehensive_test(
        splice_sites_file=args.splice_sites_file,
        output_dir=args.output_dir
    )
    
    # Exit with appropriate code
    compatibility = results['recommendations']['overall_compatibility']
    exit_code = 0 if compatibility in ['excellent', 'good'] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
