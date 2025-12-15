#!/usr/bin/env python3
"""
Generate per-chromosome validation statistics for any VCF dataset.

This is a general-purpose tool that can validate coordinate consistency
for any VCF file against any reference FASTA, breaking down results by chromosome.

Usage:
    python generate_per_chromosome_validation.py --vcf input.vcf.gz --fasta reference.fa
    python generate_per_chromosome_validation.py --vcf data/ensembl/clinvar/vcf/clinvar_20250831_main_chroms.vcf.gz --fasta data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa
"""

import argparse
import pandas as pd
import subprocess
import sys
from pathlib import Path

def get_variants_per_chromosome(vcf_path, max_per_chrom=10):
    """Get a sample of variants from each chromosome."""
    
    print("Analyzing chromosome distribution in VCF...")
    
    # Get chromosome distribution
    cmd = f"bcftools view -H {vcf_path} | cut -f1 | sort | uniq -c | sort -nr"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error getting chromosome distribution: {result.stderr}")
        return None
    
    # Parse chromosome counts
    chrom_counts = {}
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            parts = line.strip().split()
            count = int(parts[0])
            chrom = parts[1]
            chrom_counts[chrom] = count
    
    print(f"Found variants on {len(chrom_counts)} chromosomes")
    
    # Sample variants from each chromosome
    sampled_variants = []
    
    for chrom, count in chrom_counts.items():
        sample_size = min(max_per_chrom, count)
        print(f"Sampling {sample_size} variants from chromosome {chrom} (total: {count})")
        
        # Extract variants from this chromosome
        cmd = f"bcftools view -H -r {chrom} {vcf_path} | head -{sample_size}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    fields = line.split('\t')
                    if len(fields) >= 5:
                        sampled_variants.append({
                            'chromosome': fields[0],
                            'position': int(fields[1]),
                            'ref': fields[3],
                            'alt': fields[4]
                        })
    
    return sampled_variants

def validate_variants_individually(variants, fasta_path, verbose=True):
    """Validate each variant individually using the coordinate verifier."""
    
    results = []
    
    for i, variant in enumerate(variants):
        if verbose:
            print(f"Validating variant {i+1}/{len(variants)}: {variant['chromosome']}:{variant['position']} {variant['ref']}→{variant['alt']}")
        
        # Use the coordinate verifier for single position validation
        pos_string = f"{variant['chromosome']}:{variant['position']}:{variant['ref']}:{variant['alt']}"
        
        # Determine the path to vcf_coordinate_verifier.py
        script_dir = Path(__file__).parent
        verifier_path = script_dir / "vcf_coordinate_verifier.py"
        
        cmd = [
            "python", str(verifier_path),
            "--verify-position", pos_string,
            "--fasta", fasta_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse result
        success = result.returncode == 0 and "✅ VERIFICATION SUCCESSFUL!" in result.stdout
        
        results.append({
            'chromosome': variant['chromosome'],
            'position': variant['position'],
            'expected_ref': variant['ref'],
            'alt': variant['alt'],
            'match': success,
            'status': 'SUCCESS' if success else 'FAILED'
        })
        
        if verbose:
            if not success:
                print(f"  ❌ Failed: {result.stdout}")
            else:
                print(f"  ✅ Success")
    
    return results

def generate_chromosome_statistics(results_df):
    """Generate per-chromosome statistics from validation results."""
    
    # Per-chromosome statistics
    chrom_stats = results_df.groupby('chromosome').agg({
        'match': ['count', 'sum', 'mean']
    }).round(3)
    
    chrom_stats.columns = ['Total_Variants', 'Matches', 'Consistency_Rate']
    chrom_stats['Consistency_Percent'] = (chrom_stats['Consistency_Rate'] * 100).round(1)
    chrom_stats['Status'] = chrom_stats['Consistency_Percent'].apply(
        lambda x: '🎯 Perfect' if x == 100.0 else '✅ Excellent' if x >= 95.0 else '⚠️ Good' if x >= 80.0 else '❌ Issues'
    )
    
    return chrom_stats

def print_chromosome_report(chrom_stats, total_variants, total_matches):
    """Print formatted per-chromosome validation report."""
    
    print("\nPer-Chromosome Validation Results:")
    print("=" * 70)
    print(f"{'Chromosome':<12} {'Variants':<8} {'Matches':<8} {'Consistency':<12} {'Status':<15}")
    print("-" * 70)
    
    # Sort chromosomes naturally (1, 2, ..., 10, 11, ..., X, Y, MT)
    def chrom_sort_key(chrom):
        if chrom.isdigit():
            return (0, int(chrom))
        elif chrom in ['X', 'Y']:
            return (1, ord(chrom))
        elif chrom == 'MT':
            return (2, 0)
        else:
            return (3, chrom)
    
    for chrom in sorted(chrom_stats.index, key=chrom_sort_key):
        row = chrom_stats.loc[chrom]
        print(f"{chrom:<12} {int(row['Total_Variants']):<8} {int(row['Matches']):<8} {row['Consistency_Percent']:<11.1f}% {row['Status']:<15}")
    
    # Overall statistics
    overall_consistency = (total_matches / total_variants * 100) if total_variants > 0 else 0
    overall_status = '🎯 Perfect' if overall_consistency == 100.0 else '✅ Excellent' if overall_consistency >= 95.0 else '⚠️ Good' if overall_consistency >= 80.0 else '❌ Issues'
    
    print("-" * 70)
    print(f"{'Overall':<12} {total_variants:<8} {total_matches:<8} {overall_consistency:<11.1f}% {overall_status:<15}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-chromosome validation statistics for VCF datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate ClinVar dataset
  python generate_per_chromosome_validation.py \\
      --vcf data/ensembl/clinvar/vcf/clinvar_20250831_main_chroms.vcf.gz \\
      --fasta data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa

  # Validate custom VCF with more samples per chromosome
  python generate_per_chromosome_validation.py \\
      --vcf my_variants.vcf.gz \\
      --fasta reference.fa \\
      --samples-per-chromosome 10 \\
      --output my_validation_results
        """
    )
    
    parser.add_argument("--vcf", required=True,
                       help="VCF file to validate (supports relative paths)")
    parser.add_argument("--fasta", required=True,
                       help="Reference FASTA file (supports relative paths)")
    parser.add_argument("--samples-per-chromosome", type=int, default=5,
                       help="Number of variants to sample per chromosome (default: 5)")
    parser.add_argument("--output", default="chromosome_validation",
                       help="Output file prefix (default: chromosome_validation)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode - minimal output")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose mode - detailed progress")
    
    args = parser.parse_args()
    
    # Resolve file paths (use the same robust resolution as vcf_coordinate_verifier)
    vcf_path = Path(args.vcf)
    fasta_path = Path(args.fasta)
    
    # Auto-detect project root for relative paths
    project_root = Path.cwd()
    for parent in [project_root] + list(project_root.parents):
        if (parent / "meta_spliceai").exists() and (parent / "pyproject.toml").exists():
            project_root = parent
            break
    
    # Resolve VCF path
    if not vcf_path.is_absolute():
        if not vcf_path.exists():
            # Try project-relative path
            project_vcf = project_root / vcf_path
            if project_vcf.exists():
                vcf_path = project_vcf
    
    # Resolve FASTA path
    if not fasta_path.is_absolute():
        if not fasta_path.exists():
            # Try project-relative path
            project_fasta = project_root / fasta_path
            if project_fasta.exists():
                fasta_path = project_fasta
    
    # Check if files exist
    if not vcf_path.exists():
        print(f"Error: VCF file not found: {vcf_path}")
        return 1
    
    if not fasta_path.exists():
        print(f"Error: FASTA file not found: {fasta_path}")
        return 1
    
    if not args.quiet:
        print(f"Validating VCF: {vcf_path}")
        print(f"Against FASTA: {fasta_path}")
        print(f"Samples per chromosome: {args.samples_per_chromosome}")
    
    # Sample variants from each chromosome
    if not args.quiet:
        print("\nStep 1: Sampling variants from each chromosome...")
    
    variants = get_variants_per_chromosome(str(vcf_path), max_per_chrom=args.samples_per_chromosome)
    
    if not variants:
        print("No variants found!")
        return 1
    
    if not args.quiet:
        print(f"\nStep 2: Validating {len(variants)} variants...")
    
    results = validate_variants_individually(variants, str(fasta_path), verbose=args.verbose)
    
    # Create DataFrame and analyze
    df = pd.DataFrame(results)
    
    if not args.quiet:
        print("\nStep 3: Generating per-chromosome statistics...")
    
    chrom_stats = generate_chromosome_statistics(df)
    
    # Print report
    print_chromosome_report(chrom_stats, len(df), df['match'].sum())
    
    # Save detailed results
    detailed_file = f"{args.output}_detailed.tsv"
    summary_file = f"{args.output}_summary.tsv"
    
    df.to_csv(detailed_file, sep='\t', index=False)
    chrom_stats.to_csv(summary_file, sep='\t')
    
    if not args.quiet:
        print(f"\nDetailed results saved to: {detailed_file}")
        print(f"Summary results saved to: {summary_file}")
    
    # Return appropriate exit code
    overall_consistency = df['match'].mean() * 100
    if overall_consistency >= 95:
        return 0  # Success
    elif overall_consistency >= 80:
        return 1  # Warning
    else:
        return 2  # Error

if __name__ == "__main__":
    sys.exit(main())
