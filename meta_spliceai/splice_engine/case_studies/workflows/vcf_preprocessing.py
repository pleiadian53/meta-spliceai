"""
VCF preprocessing workflow with bcftools normalization.

Implements proper VCF normalization including:
- Multiallelic splitting (-m -both)
- Left-alignment and trimming of indels (-f reference.fa)
- Indexing with tabix for efficient querying

Left-alignment moves indels to the leftmost position that yields the same alternate sequence.
This only occurs in repetitive contexts (homopolymers, tandem repeats).
It uses reference genome 5'→3' coordinates, independent of gene strand.

Example (homopolymer):
  Reference: ...AAAAAATCG...
  Before: chr1:1005 A -> - (delete one A)
  After:  chr1:1001 A -> - (same deletion, left-aligned)
  
Example (tandem repeat):
  Reference: ...ATCATCATCG...
  Before: chr1:1006 ATC -> - (delete one ATC)
  After:  chr1:1003 ATC -> - (same deletion, left-aligned)

Why this matters for splice analysis:
- Ensures consistent variant representation across different callers/datasets
- Critical for accurate WT/ALT sequence construction around splice sites
- Prevents duplicate variants at different positions that represent same biological change
- Standardizes coordinates for splice site proximity calculations
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..data_sources.resource_manager import CaseStudyResourceManager


@dataclass
class VCFPreprocessingConfig:
    """Configuration for VCF preprocessing."""
    
    # Input/output paths
    input_vcf: Path
    output_vcf: Path
    reference_fasta: Optional[Path] = None
    
    # Normalization options
    split_multiallelics: bool = True
    left_align: bool = True
    trim_alleles: bool = True
    
    # Quality filtering
    min_qual: Optional[float] = None
    exclude_filters: List[str] = None
    
    # Performance options
    threads: int = 1
    memory_gb: int = 4
    
    # Validation
    validate_output: bool = True
    create_index: bool = True


class VCFPreprocessor:
    """VCF preprocessing workflow using bcftools."""
    
    def __init__(self, config: VCFPreprocessingConfig):
        """
        Initialize VCF preprocessor.
        
        Parameters
        ----------
        config : VCFPreprocessingConfig
            Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.resource_manager = CaseStudyResourceManager()
        
        # Resolve reference FASTA if not provided
        if not self.config.reference_fasta:
            self.config.reference_fasta = self._get_reference_fasta()
    
    def _get_reference_fasta(self) -> Path:
        """Get reference FASTA path from resource manager."""
        try:
            return self.resource_manager.get_reference_fasta()
        except Exception as e:
            self.logger.warning(f"Could not get reference FASTA from resource manager: {e}")
            # Fallback to common paths
            common_paths = [
                Path("/data/reference/GRCh38.fa"),
                Path("/data/reference/hg38.fa"),
                Path("data/reference/GRCh38.fa")
            ]
            
            for path in common_paths:
                if path.exists():
                    return path
            
            raise FileNotFoundError("Reference FASTA not found. Please specify reference_fasta in config.")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required tools are available."""
        tools = {}
        
        # Check bcftools
        try:
            result = subprocess.run(['bcftools', '--version'], 
                                  capture_output=True, text=True, check=True)
            tools['bcftools'] = True
            self.logger.info(f"bcftools available: {result.stdout.split()[1]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            tools['bcftools'] = False
            self.logger.error("bcftools not found")
        
        # Check tabix
        try:
            result = subprocess.run(['tabix', '--version'], 
                                  capture_output=True, text=True, check=True)
            tools['tabix'] = True
            self.logger.info(f"tabix available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            tools['tabix'] = False
            self.logger.error("tabix not found")
        
        # Check bgzip
        try:
            subprocess.run(['bgzip', '--version'], 
                          capture_output=True, text=True, check=True)
            tools['bgzip'] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            tools['bgzip'] = False
            self.logger.error("bgzip not found")
        
        return tools
    
    def normalize_vcf(self) -> Path:
        """
        Normalize VCF file using bcftools norm.
        
        Returns
        -------
        Path
            Path to normalized VCF file
        """
        self.logger.info(f"Normalizing VCF: {self.config.input_vcf}")
        
        # Check dependencies
        tools = self.check_dependencies()
        if not all(tools.values()):
            missing = [tool for tool, available in tools.items() if not available]
            raise RuntimeError(f"Missing required tools: {missing}")
        
        # Ensure output directory exists
        self.config.output_vcf.parent.mkdir(parents=True, exist_ok=True)
        
        # First, try to filter out problematic contigs that aren't in the reference
        filtered_vcf = self._filter_problematic_contigs()
        
        # Build bcftools norm command using filtered VCF
        cmd = self._build_norm_command(filtered_vcf)
        
        # Run normalization
        self.logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info("VCF normalization completed successfully")
            
            if result.stderr:
                self.logger.info(f"bcftools norm stderr: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"VCF normalization failed: {e}")
            self.logger.error(f"Command: {' '.join(cmd)}")
            self.logger.error(f"Stderr: {e.stderr}")
            
            # If normalization fails due to contig issues, try without left-alignment
            if "was not found" in str(e.stderr):
                self.logger.warning("Normalization failed due to missing contigs, retrying without left-alignment...")
                return self._normalize_without_left_alignment(filtered_vcf)
            raise
        
        # Compress and index if requested
        if self.config.create_index:
            self._compress_and_index()
        
        # Validate output if requested
        if self.config.validate_output:
            self._validate_output()
        
        return self.config.output_vcf
    
    def _filter_problematic_contigs(self) -> Path:
        """Filter out contigs that aren't in the reference genome."""
        # Standard chromosome contigs that should be in the reference
        standard_contigs = [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
            '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT',
            'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
            'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
            'chr20', 'chr21', 'chr22', 'chrX', 'chrY', 'chrM'
        ]
        
        # Check if input VCF has problematic contigs
        try:
            result = subprocess.run(['bcftools', 'view', '-H', str(self.config.input_vcf)], 
                                 capture_output=True, text=True, check=True)
            contigs = set(line.split('\t')[0] for line in result.stdout.strip().split('\n') if line.strip())
            
            problematic_contigs = contigs - set(standard_contigs)
            
            if problematic_contigs:
                self.logger.warning(f"Found {len(problematic_contigs)} non-standard contigs: {list(problematic_contigs)[:5]}...")
                
                # Create filtered VCF with only standard contigs
                filtered_vcf = self.config.input_vcf.parent / f"{self.config.input_vcf.stem}_filtered.vcf.gz"
                contig_list = ','.join(standard_contigs)
                
                cmd = ['bcftools', 'view', '-r', contig_list, '-Oz', '-o', str(filtered_vcf), str(self.config.input_vcf)]
                subprocess.run(cmd, check=True)
                
                # Create index for filtered VCF
                subprocess.run(['bcftools', 'index', str(filtered_vcf)], check=True)
                
                self.logger.info(f"Filtered VCF created: {filtered_vcf}")
                return filtered_vcf
            else:
                self.logger.info("No problematic contigs found, using original VCF")
                return self.config.input_vcf
                
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Could not check contigs: {e}, using original VCF")
            return self.config.input_vcf
    
    def _normalize_without_left_alignment(self, input_vcf: Path) -> Path:
        """Normalize VCF without left-alignment to avoid contig issues."""
        self.logger.info("Attempting normalization without left-alignment...")
        
        # Build command without reference FASTA
        cmd = ['bcftools', 'norm']
        
        # Multiallelic splitting only
        if self.config.split_multiallelics:
            cmd.extend(['-m', '-both'])
        
        # Quality filtering
        if self.config.min_qual is not None:
            cmd.extend(['-i', f'QUAL>={self.config.min_qual}'])
        
        # Exclude filters
        if self.config.exclude_filters:
            filter_expr = ' || '.join([f'FILTER="{f}"' for f in self.config.exclude_filters])
            cmd.extend(['-e', filter_expr])
        
        # Threading
        if self.config.threads > 1:
            cmd.extend(['--threads', str(self.config.threads)])
        
        # Input/output
        cmd.extend(['-o', str(self.config.output_vcf)])
        cmd.append(str(input_vcf))
        
        # Run normalization
        self.logger.info(f"Running without left-alignment: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if result.stderr:
            self.logger.info(f"bcftools norm stderr: {result.stderr}")
        
        # Compress and index if requested
        if self.config.create_index:
            self._compress_and_index()
        
        # Validate output if requested
        if self.config.validate_output:
            self._validate_output()
        
        self.logger.info("VCF normalization completed without left-alignment")
        return self.config.output_vcf

    def _build_norm_command(self, input_vcf: Optional[Path] = None) -> List[str]:
        """Build bcftools norm command."""
        cmd = ['bcftools', 'norm']
        
        # Use provided input VCF or default
        input_file = input_vcf if input_vcf else self.config.input_vcf
        
        # Multiallelic splitting
        if self.config.split_multiallelics:
            cmd.extend(['-m', '-both'])
        
        # Reference FASTA for left-alignment
        if self.config.left_align and self.config.reference_fasta:
            cmd.extend(['-f', str(self.config.reference_fasta)])
        
        # Quality filtering
        if self.config.min_qual is not None:
            cmd.extend(['-i', f'QUAL>={self.config.min_qual}'])
        
        # Exclude filters
        if self.config.exclude_filters:
            filter_expr = ' || '.join([f'FILTER="{f}"' for f in self.config.exclude_filters])
            cmd.extend(['-e', filter_expr])
        
        # Threading
        if self.config.threads > 1:
            cmd.extend(['--threads', str(self.config.threads)])
        
        # Input/output
        cmd.extend(['-o', str(self.config.output_vcf)])
        cmd.append(str(input_file))
        
        return cmd
    
    def _compress_and_index(self):
        """Compress VCF with bgzip and create tabix index."""
        if not str(self.config.output_vcf).endswith('.gz'):
            # Compress with bgzip
            compressed_path = Path(str(self.config.output_vcf) + '.gz')
            cmd = ['bgzip', '-c', str(self.config.output_vcf)]
            
            self.logger.info(f"Compressing VCF: {compressed_path}")
            with open(compressed_path, 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)
            
            # Update config to point to compressed file
            self.config.output_vcf = compressed_path
        
        # Create tabix index
        self.logger.info("Creating tabix index")
        cmd = ['tabix', '-p', 'vcf', str(self.config.output_vcf)]
        subprocess.run(cmd, check=True)
        
        self.logger.info(f"Index created: {self.config.output_vcf}.tbi")
    
    def _validate_output(self):
        """Validate normalized VCF output."""
        self.logger.info("Validating normalized VCF")
        
        # Basic validation with bcftools
        cmd = ['bcftools', 'view', '-H', str(self.config.output_vcf)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Count variants
            variant_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            self.logger.info(f"Normalized VCF contains {variant_count} variants")
            
            # Check for multiallelic sites (should be none after normalization)
            multiallelic_cmd = ['bcftools', 'view', '-H', '-i', 'N_ALT>1', str(self.config.output_vcf)]
            multiallelic_result = subprocess.run(multiallelic_cmd, capture_output=True, text=True, check=True)
            
            multiallelic_count = len(multiallelic_result.stdout.strip().split('\n')) if multiallelic_result.stdout.strip() else 0
            
            if multiallelic_count > 0:
                self.logger.warning(f"Found {multiallelic_count} multiallelic sites after normalization")
            else:
                self.logger.info("✓ No multiallelic sites found - normalization successful")
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"VCF validation failed: {e}")
            raise
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get statistics about the normalization process."""
        if not self.config.output_vcf.exists():
            return {"error": "Output VCF not found"}
        
        stats = {}
        
        try:
            # Get basic stats
            cmd = ['bcftools', 'stats', str(self.config.output_vcf)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse key statistics
            for line in result.stdout.split('\n'):
                if line.startswith('SN'):
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        key = parts[2].strip(':')
                        value = parts[3]
                        stats[key] = value
            
            # Check for specific normalization indicators
            multiallelic_cmd = ['bcftools', 'view', '-H', '-i', 'N_ALT>1', str(self.config.output_vcf)]
            multiallelic_result = subprocess.run(multiallelic_cmd, capture_output=True, text=True, check=True)
            
            multiallelic_count = len(multiallelic_result.stdout.strip().split('\n')) if multiallelic_result.stdout.strip() else 0
            stats['multiallelic_sites_remaining'] = multiallelic_count
            
        except subprocess.CalledProcessError as e:
            stats['error'] = f"Failed to get stats: {e}"
        
        return stats


def preprocess_clinvar_vcf(input_vcf: Path, output_dir: Path, 
                          reference_fasta: Optional[Path] = None) -> Path:
    """
    Convenience function to preprocess ClinVar VCF with standard settings.
    
    Parameters
    ----------
    input_vcf : Path
        Input ClinVar VCF file
    output_dir : Path
        Output directory for processed files
    reference_fasta : Path, optional
        Reference FASTA file (auto-detected if not provided)
    
    Returns
    -------
    Path
        Path to normalized VCF file
    """
    output_vcf = output_dir / f"{input_vcf.stem}_normalized.vcf.gz"
    
    config = VCFPreprocessingConfig(
        input_vcf=input_vcf,
        output_vcf=output_vcf,
        reference_fasta=reference_fasta,
        split_multiallelics=True,
        left_align=True,
        trim_alleles=True,
        create_index=True,
        validate_output=True
    )
    
    preprocessor = VCFPreprocessor(config)
    return preprocessor.normalize_vcf()


def preprocess_vcf_for_splice_analysis(input_vcf: Path, output_dir: Path,
                                     min_qual: float = 10.0) -> Path:
    """
    Preprocess VCF specifically for splice site analysis.
    
    Applies quality filtering and normalization optimized for splice variants.
    
    Parameters
    ----------
    input_vcf : Path
        Input VCF file
    output_dir : Path
        Output directory
    min_qual : float
        Minimum variant quality score
    
    Returns
    -------
    Path
        Path to processed VCF file
    """
    output_vcf = output_dir / f"{input_vcf.stem}_splice_ready.vcf.gz"
    
    config = VCFPreprocessingConfig(
        input_vcf=input_vcf,
        output_vcf=output_vcf,
        split_multiallelics=True,
        left_align=True,
        trim_alleles=True,
        min_qual=min_qual,
        exclude_filters=['LowQual', 'FAIL'],
        create_index=True,
        validate_output=True
    )
    
    preprocessor = VCFPreprocessor(config)
    return preprocessor.normalize_vcf()
