#!/usr/bin/env python3
"""
ClinVar Dataset Validation Script

Comprehensive validation script for the ClinVar dataset, including:
- File integrity checks
- Coordinate system validation  
- Statistical profiling
- Quality assurance metrics
- Compatibility testing

Usage:
    # Single file validation
    python clinvar_dataset_validator.py --vcf data/ensembl/clinvar/vcf/clinvar_20250831_main_chroms.vcf.gz --reference data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa
    
    # Directory validation  
    python clinvar_dataset_validator.py --vcf-dir data/ensembl/clinvar/vcf/ --reference data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import subprocess
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClinVarDatasetValidator:
    """Comprehensive ClinVar dataset validator."""
    
    def __init__(self, vcf_input: Path, reference_fasta: Path):
        """
        Initialize validator.
        
        Args:
            vcf_input: Path to VCF file or directory containing ClinVar VCF files
            reference_fasta: Path to reference FASTA file
        """
        self.vcf_input = Path(vcf_input)
        self.reference_fasta = Path(reference_fasta)
        self.validation_results = {}
        
        # Determine if input is file or directory
        if self.vcf_input.is_file():
            self.vcf_dir = self.vcf_input.parent
            self.single_file_mode = True
            self.target_file = self.vcf_input.name
            self.expected_files = [self.target_file]
        else:
            self.vcf_dir = self.vcf_input
            self.single_file_mode = False
            self.target_file = None
            # Expected files in ClinVar directory
            self.expected_files = [
                "clinvar_20250831.vcf.gz",
                "clinvar_20250831_reheadered.vcf.gz", 
                "clinvar_20250831_main_chroms.vcf.gz"
            ]
        
    def validate_file_integrity(self) -> Dict:
        """Validate file integrity and basic properties."""
        logger.info("Validating file integrity...")
        
        results = {
            'files_present': {},
            'file_sizes': {},
            'index_files': {},
            'compression_valid': {}
        }
        
        for filename in self.expected_files:
            if self.single_file_mode:
                # In single file mode, check the actual target file
                file_path = self.vcf_input
            else:
                # In directory mode, check expected files in directory
                file_path = self.vcf_dir / filename
            
            # Check file existence
            results['files_present'][filename] = file_path.exists()
            
            if file_path.exists():
                # File size
                results['file_sizes'][filename] = file_path.stat().st_size
                
                # Check for index files
                tbi_path = Path(str(file_path) + '.tbi')
                csi_path = Path(str(file_path) + '.csi')
                results['index_files'][filename] = {
                    'tbi': tbi_path.exists(),
                    'csi': csi_path.exists(),
                    'has_index': tbi_path.exists() or csi_path.exists()
                }
                
                # Validate compression
                try:
                    cmd = ['bcftools', 'view', '-h', str(file_path)]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    results['compression_valid'][filename] = result.returncode == 0
                except Exception as e:
                    logger.error(f"Error validating compression for {filename}: {e}")
                    results['compression_valid'][filename] = False
            else:
                results['file_sizes'][filename] = 0
                results['index_files'][filename] = {'tbi': False, 'csi': False, 'has_index': False}
                results['compression_valid'][filename] = False
        
        return results
    
    def validate_vcf_format(self, vcf_file: str) -> Dict:
        """Validate VCF format compliance."""
        logger.info(f"Validating VCF format for {vcf_file}...")
        
        if self.single_file_mode:
            file_path = self.vcf_input
        else:
            file_path = self.vcf_dir / vcf_file
            
        if not file_path.exists():
            return {'valid': False, 'error': 'File not found'}
        
        try:
            # Check header
            cmd = ['bcftools', 'view', '-h', str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return {'valid': False, 'error': 'Cannot read VCF header'}
            
            header_lines = result.stdout.strip().split('\n')
            
            # Basic format checks
            checks = {
                'has_fileformat': any(line.startswith('##fileformat=VCF') for line in header_lines),
                'has_contigs': any(line.startswith('##contig=') for line in header_lines),
                'has_column_header': any(line.startswith('#CHROM') for line in header_lines),
                'contig_count': sum(1 for line in header_lines if line.startswith('##contig='))
            }
            
            # Check first few variants
            cmd = ['bcftools', 'view', '-H', str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                first_variant = result.stdout.strip().split('\n')[0]
                fields = first_variant.split('\t')
                checks['correct_field_count'] = len(fields) >= 8
                checks['has_variants'] = True
            else:
                checks['correct_field_count'] = False
                checks['has_variants'] = False
            
            return {
                'valid': all([checks['has_fileformat'], checks['has_column_header'], checks['correct_field_count']]),
                'checks': checks
            }
            
        except Exception as e:
            logger.error(f"Error validating VCF format: {e}")
            return {'valid': False, 'error': str(e)}
    
    def get_variant_statistics(self, vcf_file: str) -> Dict:
        """Get basic variant statistics."""
        logger.info(f"Getting variant statistics for {vcf_file}...")
        
        if self.single_file_mode:
            file_path = self.vcf_input
        else:
            file_path = self.vcf_dir / vcf_file
            
        if not file_path.exists():
            return {'error': 'File not found'}
        
        try:
            # Get bcftools stats
            cmd = ['bcftools', 'stats', str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                return {'error': 'bcftools stats failed'}
            
            stats = {}
            for line in result.stdout.split('\n'):
                if line.startswith('SN'):
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        key = parts[2].strip(':').replace(' ', '_')
                        try:
                            value = int(parts[3])
                        except ValueError:
                            value = parts[3]
                        stats[key] = value
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting variant statistics: {e}")
            return {'error': str(e)}
    
    def validate_coordinates(self, vcf_file: str, sample_size: int = 100) -> Dict:
        """Validate coordinate system consistency using subprocess."""
        logger.info(f"Validating coordinates for {vcf_file} (sample size: {sample_size})...")
        
        try:
            # Use subprocess to call the coordinate verifier
            if self.single_file_mode:
                vcf_path = str(self.vcf_input)
            else:
                vcf_path = str(self.vcf_dir / vcf_file)
            
            # Auto-detect project root
            project_root = Path.cwd()
            for parent in [project_root] + list(project_root.parents):
                if (parent / "meta_spliceai").exists() and (parent / "pyproject.toml").exists():
                    project_root = parent
                    break
            
            # Path to coordinate verifier
            verifier_script = project_root / "meta_spliceai" / "splice_engine" / "case_studies" / "tools" / "vcf_coordinate_verifier.py"
            
            # Run coordinate validation
            cmd = [
                "python", str(verifier_script),
                "--vcf", vcf_path,
                "--fasta", str(self.reference_fasta),
                "--variants", str(sample_size)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return {'error': f"Coordinate verifier failed: {result.stderr}"}
            
            # Parse output to extract consistency score
            output = result.stdout
            consistency_score = 0
            total_variants = 0
            matches = 0
            
            # Look for consistency score in output
            for line in output.split('\n'):
                if "COORDINATE SYSTEM CONSISTENCY:" in line:
                    try:
                        consistency_score = float(line.split(':')[1].strip().replace('%', ''))
                    except:
                        pass
                elif "Total variants processed:" in line:
                    try:
                        total_variants = int(line.split(':')[1].strip())
                    except:
                        pass
                elif "Reference allele matches:" in line:
                    try:
                        matches = int(line.split(':')[1].strip())
                    except:
                        pass
            
            # Classify result
            if consistency_score >= 95:
                assessment = "CONSISTENT"
            elif consistency_score >= 80:
                assessment = "MOSTLY_CONSISTENT"
            else:
                assessment = "INCONSISTENT"
            
            return {
                'total_variants': total_variants,
                'successful_verifications': matches,  # Assuming matches = successful
                'coordinate_matches': matches,
                'consistency_score': consistency_score,
                'assessment': assessment,
                'raw_output': output
            }
            
        except Exception as e:
            logger.error(f"Error validating coordinates: {e}")
            return {'error': str(e)}
    
    def compare_file_versions(self) -> Dict:
        """Compare different versions of ClinVar files."""
        logger.info("Comparing file versions...")
        
        comparison = {}
        
        for filename in self.expected_files:
            file_path = self.vcf_dir / filename
            if file_path.exists():
                stats = self.get_variant_statistics(filename)
                if 'error' not in stats:
                    comparison[filename] = {
                        'variant_count': stats.get('number_of_records', 0),
                        'snp_count': stats.get('number_of_SNPs', 0),
                        'indel_count': stats.get('number_of_indels', 0),
                        'multiallelic_sites': stats.get('number_of_multiallelic_sites', 0)
                    }
        
        # Calculate differences
        if len(comparison) >= 2:
            files = list(comparison.keys())
            differences = {}
            
            for i, file1 in enumerate(files):
                for file2 in files[i+1:]:
                    diff_key = f"{file1}_vs_{file2}"
                    differences[diff_key] = {
                        'variant_diff': comparison[file1]['variant_count'] - comparison[file2]['variant_count'],
                        'snp_diff': comparison[file1]['snp_count'] - comparison[file2]['snp_count'],
                        'indel_diff': comparison[file1]['indel_count'] - comparison[file2]['indel_count']
                    }
            
            comparison['differences'] = differences
        
        return comparison
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report."""
        logger.info("Generating comprehensive validation report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset': 'ClinVar_20250831',
            'vcf_directory': str(self.vcf_dir),
            'reference_fasta': str(self.reference_fasta),
            'validation_results': {}
        }
        
        # File integrity validation
        report['validation_results']['file_integrity'] = self.validate_file_integrity()
        
        # VCF format validation for each file
        format_results = {}
        for filename in self.expected_files:
            if self.single_file_mode:
                if self.vcf_input.exists():
                    format_results[filename] = self.validate_vcf_format(filename)
            else:
                if (self.vcf_dir / filename).exists():
                    format_results[filename] = self.validate_vcf_format(filename)
        report['validation_results']['vcf_format'] = format_results
        
        # Variant statistics for each file
        stats_results = {}
        for filename in self.expected_files:
            if self.single_file_mode:
                if self.vcf_input.exists():
                    stats_results[filename] = self.get_variant_statistics(filename)
            else:
                if (self.vcf_dir / filename).exists():
                    stats_results[filename] = self.get_variant_statistics(filename)
        report['validation_results']['variant_statistics'] = stats_results
        
        # Coordinate validation for main analysis file
        if self.single_file_mode:
            report['validation_results']['coordinate_validation'] = self.validate_coordinates(self.target_file)
        else:
            main_file = "clinvar_20250831_main_chroms.vcf.gz"
            if (self.vcf_dir / main_file).exists():
                report['validation_results']['coordinate_validation'] = self.validate_coordinates(main_file)
        
        # File comparison
        report['validation_results']['file_comparison'] = self.compare_file_versions()
        
        # Overall assessment
        report['overall_assessment'] = self._assess_overall_quality(report['validation_results'])
        
        return report
    
    def _assess_overall_quality(self, results: Dict) -> Dict:
        """Assess overall dataset quality."""
        
        assessment = {
            'file_integrity_score': 0,
            'format_compliance_score': 0,
            'coordinate_consistency_score': 0,
            'overall_score': 0,
            'recommendation': '',
            'issues': [],
            'strengths': []
        }
        
        # File integrity score
        integrity = results['file_integrity']
        files_present = sum(integrity['files_present'].values())
        files_with_index = sum(1 for f in integrity['index_files'].values() if f['has_index'])
        files_valid_compression = sum(integrity['compression_valid'].values())
        
        assessment['file_integrity_score'] = (files_present + files_with_index + files_valid_compression) / (3 * len(self.expected_files)) * 100
        
        # Format compliance score
        format_results = results['vcf_format']
        valid_formats = sum(1 for f in format_results.values() if f.get('valid', False))
        assessment['format_compliance_score'] = (valid_formats / len(format_results)) * 100 if format_results else 0
        
        # Coordinate consistency score
        coord_results = results.get('coordinate_validation', {})
        assessment['coordinate_consistency_score'] = coord_results.get('consistency_score', 0)
        
        # Overall score
        scores = [assessment['file_integrity_score'], assessment['format_compliance_score'], assessment['coordinate_consistency_score']]
        assessment['overall_score'] = sum(scores) / len(scores)
        
        # Recommendation
        if assessment['overall_score'] >= 95:
            assessment['recommendation'] = "EXCELLENT - Ready for production use"
            assessment['strengths'].append("High-quality dataset with excellent consistency")
        elif assessment['overall_score'] >= 85:
            assessment['recommendation'] = "GOOD - Suitable for most analyses with minor issues"
        elif assessment['overall_score'] >= 70:
            assessment['recommendation'] = "ACCEPTABLE - May require additional preprocessing"
        else:
            assessment['recommendation'] = "POOR - Significant issues require resolution"
        
        # Identify specific issues and strengths
        if assessment['coordinate_consistency_score'] >= 95:
            assessment['strengths'].append("Excellent coordinate system consistency")
        elif assessment['coordinate_consistency_score'] < 80:
            assessment['issues'].append("Coordinate system inconsistencies detected")
        
        if assessment['file_integrity_score'] == 100:
            assessment['strengths'].append("All files present with valid indices")
        elif assessment['file_integrity_score'] < 90:
            assessment['issues'].append("Missing files or index issues")
        
        return assessment
    
    def save_report(self, report: Dict, output_path: Path):
        """Save validation report to file."""
        logger.info(f"Saving validation report to {output_path}")
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def print_summary(self, report: Dict):
        """Print validation summary."""
        
        print("\n" + "="*80)
        print("CLINVAR DATASET VALIDATION SUMMARY")
        print("="*80)
        
        assessment = report['overall_assessment']
        
        print(f"\nOverall Quality Score: {assessment['overall_score']:.1f}%")
        print(f"Recommendation: {assessment['recommendation']}")
        
        print(f"\nDetailed Scores:")
        print(f"  File Integrity:        {assessment['file_integrity_score']:.1f}%")
        print(f"  Format Compliance:     {assessment['format_compliance_score']:.1f}%")
        print(f"  Coordinate Consistency: {assessment['coordinate_consistency_score']:.1f}%")
        
        if assessment['strengths']:
            print(f"\n✅ Strengths:")
            for strength in assessment['strengths']:
                print(f"   • {strength}")
        
        if assessment['issues']:
            print(f"\n⚠️  Issues:")
            for issue in assessment['issues']:
                print(f"   • {issue}")
        
        # File status
        print(f"\nFile Status:")
        for filename in self.expected_files:
            file_path = self.vcf_dir / filename
            status = "✅ Present" if file_path.exists() else "❌ Missing"
            print(f"  {filename}: {status}")
        
        # Coordinate validation results
        coord_results = report['validation_results'].get('coordinate_validation', {})
        if coord_results and 'error' not in coord_results:
            print(f"\nCoordinate Validation:")
            print(f"  Consistency Score: {coord_results['consistency_score']:.1f}%")
            print(f"  Assessment: {coord_results['assessment']}")
            print(f"  Sample Size: {coord_results['total_variants']} variants")
        
        print("\n" + "="*80)


def resolve_paths(vcf_input: str, reference: str):
    """Resolve file paths with project root detection."""
    
    # Auto-detect project root
    project_root = Path.cwd()
    for parent in [project_root] + list(project_root.parents):
        if (parent / "meta_spliceai").exists() and (parent / "pyproject.toml").exists():
            project_root = parent
            break
    
    # Resolve VCF input path
    vcf_path = Path(vcf_input)
    if not vcf_path.is_absolute():
        if not vcf_path.exists():
            # Try project-relative path
            project_vcf = project_root / vcf_path
            if project_vcf.exists():
                vcf_path = project_vcf
    
    # Resolve reference path
    ref_path = Path(reference)
    if not ref_path.is_absolute():
        if not ref_path.exists():
            # Try project-relative path
            project_ref = project_root / ref_path
            if project_ref.exists():
                ref_path = project_ref
    
    return vcf_path, ref_path


def main():
    parser = argparse.ArgumentParser(
        description="Validate ClinVar dataset integrity and quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single VCF file
  python clinvar_dataset_validator.py \\
      --vcf data/ensembl/clinvar/vcf/clinvar_20250831_main_chroms.vcf.gz \\
      --reference data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa

  # Validate entire ClinVar directory
  python clinvar_dataset_validator.py \\
      --vcf-dir data/ensembl/clinvar/vcf/ \\
      --reference data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa
        """
    )
    
    # Support both single file and directory modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--vcf", help="Single VCF file to validate")
    group.add_argument("--vcf-dir", help="Directory containing ClinVar VCF files")
    
    parser.add_argument("--reference", required=True,
                       help="Reference FASTA file")
    parser.add_argument("--output", 
                       help="Output JSON report file (default: clinvar_validation_report.json)")
    parser.add_argument("--sample-size", type=int, default=100,
                       help="Sample size for coordinate validation (default: 100)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine input path
    vcf_input = args.vcf if args.vcf else args.vcf_dir
    
    # Resolve paths
    vcf_path, ref_path = resolve_paths(vcf_input, args.reference)
    
    # Check if resolved paths exist
    if not vcf_path.exists():
        logger.error(f"VCF input not found: {vcf_path}")
        return 2
    
    if not ref_path.exists():
        logger.error(f"Reference FASTA not found: {ref_path}")
        return 2
    
    # Initialize validator
    validator = ClinVarDatasetValidator(vcf_path, ref_path)
    
    # Generate report
    report = validator.generate_validation_report()
    
    # Print summary
    validator.print_summary(report)
    
    # Save report
    output_path = Path(args.output) if args.output else Path("clinvar_validation_report.json")
    validator.save_report(report, output_path)
    
    print(f"\nDetailed report saved to: {output_path}")
    
    # Exit with appropriate code
    overall_score = report['overall_assessment']['overall_score']
    if overall_score >= 95:
        sys.exit(0)  # Success
    elif overall_score >= 70:
        sys.exit(1)  # Warning
    else:
        sys.exit(2)  # Error


if __name__ == "__main__":
    main()
