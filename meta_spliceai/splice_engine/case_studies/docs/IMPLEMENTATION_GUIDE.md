# Implementation Guide: Alternative Splicing Case Study Framework

**Document Version**: 1.0  
**Date**: 2025-07-28  
**Purpose**: Practical implementation guide for the Q1-Q7 system design solutions

---

## 🎯 **QUICK START IMPLEMENTATION**

This guide provides step-by-step implementation instructions for the comprehensive system design solutions outlined in `SYSTEM_DESIGN_ANALYSIS_Q1_Q7.md`.

---

## 📋 **IMPLEMENTATION CHECKLIST**

### **Phase 1: Foundation Infrastructure** ✅
- [ ] Create `genomic_resources` package
- [ ] Implement `OutputManager` class
- [ ] Update `Config` class with centralized paths
- [ ] Create quality control threshold definitions

### **Phase 2: Alternative Splicing Framework** 🧬
- [ ] Implement `AlternativeSpliceEvent` enumeration
- [ ] Create `DiseaseInducedSplicing` dataclass
- [ ] Develop comprehensive splice site annotation
- [ ] Build `AlternativeGenomeBuilder` class

### **Phase 3: Case Study Integration** 🔗
- [ ] Integrate variant databases
- [ ] Create disease-specific validation workflows
- [ ] Implement reporting and visualization
- [ ] Validate against known mutations

---

## 🔧 **DETAILED IMPLEMENTATION STEPS**

### **Step 1: Create Genomic Resources Package**

#### **1.1 Create Package Structure**
```bash
mkdir -p meta_spliceai/system/genomic_resources
touch meta_spliceai/system/genomic_resources/__init__.py
```

#### **1.2 Implement Core Classes**
```python
# meta_spliceai/system/genomic_resources/core.py

from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from ..config import Config

class CoordinateSystem(Enum):
    """Standardized coordinate system conventions"""
    ZERO_BASED = "0-based"
    ONE_BASED = "1-based"

class SpliceSiteDefinition:
    """Standardized splice site motif requirements"""
    
    DONOR_MOTIFS = ["GT", "GC"]
    ACCEPTOR_MOTIFS = ["AG"]
    MIN_SCORE_THRESHOLD = 0.1
    CONTEXT_WINDOW = 2

@dataclass
class StandardizedGenome:
    """GRCh38.112 specification with coordinate system definitions"""
    
    GENOME_BUILD: str = "GRCh38"
    ENSEMBL_RELEASE: str = "112"
    COORDINATE_SYSTEM: CoordinateSystem = CoordinateSystem.ONE_BASED
    
    @property
    def gtf_path(self) -> Path:
        return Path(Config.DATA_DIR) / "ensembl" / "Homo_sapiens.GRCh38.112.gtf"
    
    @property
    def fasta_path(self) -> Path:
        return Path(Config.DATA_DIR) / "ensembl" / "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
    @property
    def annotations_db_path(self) -> Path:
        return Path(Config.DATA_DIR) / "ensembl" / "annotations.db"
    
    def validate_paths(self) -> bool:
        """Validate that all required files exist"""
        required_files = [self.gtf_path, self.fasta_path]
        return all(path.exists() for path in required_files)

class GenomicResourceManager:
    """Centralized management of all genomic inputs and outputs"""
    
    def __init__(self, genome: Optional[StandardizedGenome] = None):
        self.genome = genome or StandardizedGenome()
        if not self.genome.validate_paths():
            raise FileNotFoundError("Required genomic files not found")
    
    def get_foundation_model_config(self, model_name: str) -> dict:
        """Get foundation model configuration"""
        configs = {
            "spliceai": {
                "context_length": 10000,
                "batch_size": 32,
                "model_path": "spliceai"
            },
            "mmsplice": {
                "context_length": 5000,
                "batch_size": 16,
                "model_path": "mmsplice"
            }
        }
        return configs.get(model_name, {})
    
    def get_case_study_database_path(self, database: str) -> Path:
        """Get path to case study database files"""
        base_path = Path(Config.DATA_DIR) / "case_studies"
        database_paths = {
            "splicevardb": base_path / "splicevardb",
            "clinvar": base_path / "clinvar", 
            "mutsplicedb": base_path / "mutsplicedb",
            "dbass": base_path / "dbass"
        }
        return database_paths.get(database, base_path / database)
```

#### **1.3 Update Config Class**
```python
# meta_spliceai/system/config.py (additions)

from .genomic_resources import GenomicResourceManager

class Config:
    # ... existing code ...
    
    # Add genomic resource manager
    @classmethod
    def get_genomic_resources(cls):
        """Get genomic resource manager instance"""
        if not hasattr(cls, '_genomic_resources'):
            cls._genomic_resources = GenomicResourceManager()
        return cls._genomic_resources
```

### **Step 2: Implement Output Management**

#### **2.1 Create Output Manager**
```python
# meta_spliceai/system/output_manager.py

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class OutputManager:
    """Systematic tracking of all generated artifacts"""
    
    def __init__(self, base_output_dir: Path):
        self.base_dir = Path(base_output_dir)
        self.registry_file = self.base_dir / "artifact_registry.json"
        self.artifact_registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load existing artifact registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"artifacts": [], "created": datetime.now().isoformat()}
    
    def _save_registry(self):
        """Save artifact registry to disk"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.artifact_registry, f, indent=2)
    
    def register_artifact(self, artifact_type: str, path: Path, metadata: dict):
        """Register generated artifacts with metadata"""
        artifact_entry = {
            "type": artifact_type,
            "path": str(path),
            "created": datetime.now().isoformat(),
            "metadata": metadata
        }
        self.artifact_registry["artifacts"].append(artifact_entry)
        self._save_registry()
    
    def get_artifacts_by_type(self, artifact_type: str) -> List[Path]:
        """Retrieve artifacts by type"""
        matching_artifacts = [
            Path(artifact["path"]) 
            for artifact in self.artifact_registry["artifacts"]
            if artifact["type"] == artifact_type
        ]
        return matching_artifacts
    
    def cleanup_temporary_artifacts(self, max_age_days: int = 7):
        """Clean up temporary analysis artifacts"""
        # Implementation for cleanup logic
        pass
```

### **Step 3: Implement Alternative Splicing Framework**

#### **3.1 Create Alternative Splicing Types**
```python
# meta_spliceai/splice_engine/case_studies/alternative_splicing.py

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from .data_sources.base import SpliceMutation, ClinicalSignificance

class AlternativeSpliceEvent(Enum):
    """Comprehensive classification of mutation-induced splice events"""
    
    CRYPTIC_DONOR_ACTIVATION = "cryptic_donor_activation"
    CRYPTIC_ACCEPTOR_ACTIVATION = "cryptic_acceptor_activation"
    CANONICAL_SITE_LOSS = "canonical_site_loss"
    EXON_SKIPPING = "exon_skipping"
    INTRON_RETENTION = "intron_retention"
    PSEUDOEXON_ACTIVATION = "pseudoexon_activation"
    PARTIAL_EXON_DELETION = "partial_exon_deletion"
    COMPLEX_REARRANGEMENT = "complex_rearrangement"

class SpliceCategory(Enum):
    """Categories for comprehensive splice site annotation"""
    
    CANONICAL = "canonical"
    CRYPTIC_ACTIVATED = "cryptic_activated"
    CANONICAL_DISRUPTED = "canonical_disrupted"
    DISEASE_ASSOCIATED = "disease_associated"
    PATHOGENIC_VARIANT = "pathogenic_variant"
    PREDICTED_ALTERNATIVE = "predicted_alternative"
    HIGH_CONFIDENCE_CRYPTIC = "high_confidence_cryptic"

@dataclass
class CanonicalSplicesite:
    """Canonical splice site representation"""
    position: int
    site_type: str  # "donor" or "acceptor"
    strength: float
    gene_id: str
    transcript_id: str

@dataclass
class AlternativeSplicesite:
    """Alternative splice site representation"""
    position: int
    site_type: str
    strength: float
    splice_category: SpliceCategory
    variant_id: Optional[str] = None

@dataclass
class DiseaseInducedSplicing:
    """Formal representation of disease-associated alternative splicing"""
    
    # Disease context
    disease_category: str
    disease_name: str
    
    # Mutation context
    mutation_context: SpliceMutation
    splice_event_type: AlternativeSpliceEvent
    
    # Splice site effects
    canonical_site_affected: Optional[CanonicalSplicesite]
    alternative_sites_created: List[AlternativeSplicesite]
    
    # Functional consequences
    functional_consequence: str
    protein_impact: str
    
    # Clinical annotations
    clinical_significance: ClinicalSignificance
    validation_evidence: List[str]
    
    # Quantitative measures
    splice_strength_change: Optional[float] = None
    inclusion_level_change: Optional[float] = None
    expression_impact: Optional[float] = None
```

#### **3.2 Create Alternative Genome Builder**
```python
# meta_spliceai/splice_engine/case_studies/alternative_genome_builder.py

from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from pyfaidx import Fasta
from .data_sources.base import SpliceMutation
from .alternative_splicing import AlternativeSplicesite

class AlternativeGenomeBuilder:
    """Generate variant-aware genomic resources"""
    
    def __init__(self, reference_gtf: Path, reference_fasta: Path):
        self.reference_gtf = reference_gtf
        self.reference_fasta = reference_fasta
        self.fasta = Fasta(str(reference_fasta))
    
    def create_variant_fasta(self, 
                           variants: List[SpliceMutation],
                           output_path: Path,
                           context_window: int = 10000) -> Path:
        """Generate alternative FASTA with variant-induced sequences"""
        
        variant_sequences = {}
        
        for variant in variants:
            # Extract sequence context around variant
            chrom = variant.chrom
            start = max(1, variant.position - context_window // 2)
            end = variant.position + context_window // 2
            
            # Get reference sequence
            ref_seq = str(self.fasta[chrom][start:end])
            
            # Apply variant
            variant_seq = self._apply_variant_to_sequence(
                ref_seq, variant, start
            )
            
            # Store variant sequence
            variant_id = f"{variant.chrom}_{variant.position}_{variant.ref_allele}_{variant.alt_allele}"
            variant_sequences[variant_id] = variant_seq
        
        # Write variant FASTA
        self._write_fasta(variant_sequences, output_path)
        return output_path
    
    def _apply_variant_to_sequence(self, 
                                 reference_seq: str, 
                                 variant: SpliceMutation,
                                 start_pos: int) -> str:
        """Apply variant to reference sequence"""
        
        seq_list = list(reference_seq)
        relative_pos = variant.position - start_pos
        
        if len(variant.ref_allele) == 1 and len(variant.alt_allele) == 1:
            # Substitution
            seq_list[relative_pos] = variant.alt_allele
        elif len(variant.ref_allele) > len(variant.alt_allele):
            # Deletion
            del_length = len(variant.ref_allele) - len(variant.alt_allele)
            del seq_list[relative_pos:relative_pos + del_length]
            if variant.alt_allele:
                seq_list[relative_pos] = variant.alt_allele
        else:
            # Insertion
            ins_seq = variant.alt_allele[len(variant.ref_allele):]
            seq_list.insert(relative_pos + 1, ins_seq)
        
        return ''.join(seq_list)
    
    def _write_fasta(self, sequences: Dict[str, str], output_path: Path):
        """Write sequences to FASTA file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for seq_id, sequence in sequences.items():
                f.write(f">{seq_id}\n")
                # Write sequence in 80-character lines
                for i in range(0, len(sequence), 80):
                    f.write(sequence[i:i+80] + "\n")
    
    def create_alternative_gtf(self,
                              alternative_splice_sites: pd.DataFrame,
                              output_path: Path) -> Path:
        """Generate alternative GTF with variant-induced isoforms"""
        
        # Load reference GTF
        reference_gtf_df = pd.read_csv(
            self.reference_gtf, 
            sep='\t', 
            comment='#',
            names=['seqname', 'source', 'feature', 'start', 'end', 
                   'score', 'strand', 'frame', 'attribute']
        )
        
        # Create alternative transcripts
        alternative_transcripts = self._create_alternative_transcripts(
            reference_gtf_df, alternative_splice_sites
        )
        
        # Combine reference and alternative
        combined_gtf = pd.concat([reference_gtf_df, alternative_transcripts])
        
        # Write combined GTF
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_gtf.to_csv(output_path, sep='\t', index=False, header=False)
        
        return output_path
    
    def _create_alternative_transcripts(self,
                                      reference_gtf: pd.DataFrame,
                                      alternative_sites: pd.DataFrame) -> pd.DataFrame:
        """Create new transcript isoforms based on alternative splice sites"""
        
        alternative_transcripts = []
        
        for gene_id in alternative_sites['gene_id'].unique():
            gene_alt_sites = alternative_sites[
                alternative_sites['gene_id'] == gene_id
            ]
            
            # Get reference transcripts for this gene
            gene_transcripts = reference_gtf[
                reference_gtf['attribute'].str.contains(f'gene_id "{gene_id}"')
            ]
            
            # Create alternative isoforms
            for _, alt_site in gene_alt_sites.iterrows():
                if alt_site['splice_category'] in ['cryptic_activated', 'disease_associated']:
                    new_transcript = self._modify_transcript_for_alternative_site(
                        gene_transcripts, alt_site
                    )
                    alternative_transcripts.extend(new_transcript)
        
        return pd.DataFrame(alternative_transcripts)
    
    def _modify_transcript_for_alternative_site(self,
                                              transcript_features: pd.DataFrame,
                                              alternative_site: pd.Series) -> List[dict]:
        """Modify transcript features to incorporate alternative splice site"""
        
        # This is a simplified implementation
        # In practice, this would need sophisticated logic to:
        # 1. Identify affected exons
        # 2. Modify exon boundaries
        # 3. Update transcript coordinates
        # 4. Handle different splice event types
        
        modified_features = []
        
        # For now, create a placeholder alternative transcript
        for _, feature in transcript_features.iterrows():
            if feature['feature'] == 'transcript':
                # Create alternative transcript entry
                alt_transcript = feature.copy()
                alt_transcript['attribute'] = alt_transcript['attribute'].replace(
                    'transcript_id "', 'transcript_id "ALT_'
                )
                modified_features.append(alt_transcript.to_dict())
        
        return modified_features
```

### **Step 4: Integration with Existing Infrastructure**

#### **4.1 Update AlignedSpliceExtractor**
```python
# Add to aligned_splice_extractor.py

from ..case_studies.alternative_splicing import SpliceCategory

class AlignedSpliceExtractor:
    # ... existing code ...
    
    def extract_comprehensive_splice_sites(self,
                                         gtf_file: str,
                                         fasta_file: str,
                                         alternative_sites_file: Optional[str] = None,
                                         gene_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Extract both canonical and alternative splice sites"""
        
        # Extract canonical sites (existing functionality)
        canonical_sites = self.extract_splice_sites(gtf_file, fasta_file, gene_ids)
        canonical_sites['splice_category'] = SpliceCategory.CANONICAL.value
        
        # Add alternative sites if provided
        if alternative_sites_file:
            alternative_sites = pd.read_csv(alternative_sites_file, sep='\t')
            comprehensive_sites = pd.concat([canonical_sites, alternative_sites])
        else:
            comprehensive_sites = canonical_sites
        
        return comprehensive_sites
```

#### **4.2 Create Integration Script**
```python
# meta_spliceai/splice_engine/case_studies/create_comprehensive_annotation.py

import pandas as pd
from pathlib import Path
from typing import List
from .data_sources.base import BaseIngester
from .alternative_splicing import SpliceCategory
from ..meta_models.openspliceai_adapter import AlignedSpliceExtractor

def create_comprehensive_splice_annotation(
    canonical_sites_file: Path,
    variant_databases: List[BaseIngester],
    output_file: Path
) -> pd.DataFrame:
    """Create unified splice site annotation with canonical + alternative"""
    
    # Load canonical sites
    canonical_sites = pd.read_csv(canonical_sites_file, sep='\t')
    canonical_sites['splice_category'] = SpliceCategory.CANONICAL.value
    canonical_sites['variant_id'] = None
    canonical_sites['clinical_significance'] = None
    
    comprehensive_sites = [canonical_sites]
    
    # Add variant-induced sites from each database
    for database in variant_databases:
        print(f"Processing {database.__class__.__name__}...")
        
        # Ingest data from database
        result = database.ingest()
        
        # Convert to splice site format
        variant_sites = database.create_splice_sites_annotation(result.mutations)
        variant_sites['splice_category'] = SpliceCategory.DISEASE_ASSOCIATED.value
        
        comprehensive_sites.append(variant_sites)
    
    # Combine all sites
    final_annotation = pd.concat(comprehensive_sites, ignore_index=True)
    
    # Remove duplicates (same position, same gene)
    final_annotation = final_annotation.drop_duplicates(
        subset=['chrom', 'position', 'gene_id', 'site_type']
    )
    
    # Save result
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_annotation.to_csv(output_file, sep='\t', index=False)
    
    print(f"Created comprehensive annotation with {len(final_annotation)} splice sites")
    return final_annotation

if __name__ == "__main__":
    from .data_sources.splicevardb import SpliceVarDBIngester
    from .data_sources.clinvar import ClinVarIngester
    
    # Example usage
    canonical_sites = Path("data/ensembl/splice_sites.tsv")
    output_file = Path("data/ensembl/alternative_splice_sites.tsv")
    
    # Initialize variant databases
    databases = [
        SpliceVarDBIngester(output_dir=Path("case_studies/splicevardb")),
        ClinVarIngester(output_dir=Path("case_studies/clinvar"))
    ]
    
    # Create comprehensive annotation
    result = create_comprehensive_splice_annotation(
        canonical_sites, databases, output_file
    )
```

---

## 🧪 **TESTING AND VALIDATION**

### **Unit Tests**
```python
# tests/test_genomic_resources.py

import pytest
from meta_spliceai.system.genomic_resources import GenomicResourceManager, StandardizedGenome

def test_genomic_resource_manager():
    """Test genomic resource manager functionality"""
    manager = GenomicResourceManager()
    
    # Test foundation model config
    spliceai_config = manager.get_foundation_model_config("spliceai")
    assert "context_length" in spliceai_config
    assert spliceai_config["context_length"] == 10000
    
    # Test database paths
    splicevardb_path = manager.get_case_study_database_path("splicevardb")
    assert "splicevardb" in str(splicevardb_path)

def test_standardized_genome():
    """Test standardized genome specification"""
    genome = StandardizedGenome()
    
    assert genome.GENOME_BUILD == "GRCh38"
    assert genome.ENSEMBL_RELEASE == "112"
    assert "Homo_sapiens.GRCh38.112.gtf" in str(genome.gtf_path)
```

### **Integration Tests**
```python
# tests/test_alternative_splicing_integration.py

import pytest
import pandas as pd
from meta_spliceai.splice_engine.case_studies.create_comprehensive_annotation import (
    create_comprehensive_splice_annotation
)

def test_comprehensive_annotation_creation():
    """Test creation of comprehensive splice site annotation"""
    
    # Create mock canonical sites
    canonical_sites = pd.DataFrame({
        'chrom': ['1', '1'],
        'position': [1000, 2000],
        'site_type': ['donor', 'acceptor'],
        'gene_id': ['ENSG001', 'ENSG001']
    })
    
    # Test annotation creation (with mock databases)
    result = create_comprehensive_splice_annotation(
        canonical_sites_file=None,  # Mock input
        variant_databases=[],       # Empty for testing
        output_file=Path("test_output.tsv")
    )
    
    assert len(result) >= len(canonical_sites)
    assert 'splice_category' in result.columns
```

---

## 📊 **MONITORING AND METRICS**

### **Performance Monitoring**
```python
# meta_spliceai/system/monitoring.py

import time
import psutil
from typing import Dict, Any

class PerformanceMonitor:
    """Monitor system performance during processing"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.metrics['start_memory'] = psutil.virtual_memory().used
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics"""
        if self.start_time:
            self.metrics['duration'] = time.time() - self.start_time
            self.metrics['end_memory'] = psutil.virtual_memory().used
            self.metrics['memory_delta'] = (
                self.metrics['end_memory'] - self.metrics['start_memory']
            )
        
        return self.metrics
```

---

## 🚀 **DEPLOYMENT CHECKLIST**

### **Pre-Deployment Validation**
- [ ] All unit tests pass
- [ ] Integration tests pass  
- [ ] Performance benchmarks meet requirements
- [ ] Documentation is complete and accurate
- [ ] Error handling is robust

### **Production Deployment**
- [ ] Create production configuration
- [ ] Set up monitoring and logging
- [ ] Deploy to production environment
- [ ] Run end-to-end validation
- [ ] Monitor initial performance

### **Post-Deployment**
- [ ] Validate case study results
- [ ] Monitor system performance
- [ ] Collect user feedback
- [ ] Plan future enhancements

---

## 📚 **ADDITIONAL RESOURCES**

### **Documentation Links**
- [System Design Analysis (Q1-Q7)](./SYSTEM_DESIGN_ANALYSIS_Q1_Q7.md)
- [Case Studies README](../README.md)
- [AlignedSpliceExtractor Documentation](../../meta_models/openspliceai_adapter/README_ALIGNED_EXTRACTOR.md)

### **Code Examples**
- [Variant Database Ingesters](../data_sources/)
- [Format Handling](../formats/)
- [Case Study Workflows](../workflows/)

This implementation guide provides the practical steps needed to realize the comprehensive system design outlined in the Q1-Q7 analysis, ensuring a smooth transition from design to working implementation.
