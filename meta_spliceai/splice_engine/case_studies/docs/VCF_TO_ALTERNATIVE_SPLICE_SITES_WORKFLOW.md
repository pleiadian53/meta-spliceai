# From VCF to Alternative Splice Sites: Complete Training Data Workflow

**Document Version**: 1.0  
**Date**: 2025-07-28  
**Purpose**: Bridge the gap from VCF-based variant analysis to alternative splice site annotation for meta-model training

---

## 🎯 **EXECUTIVE SUMMARY**

This document addresses the critical workflow gap: **How do we go from VCF variant analysis to properly represented alternative splice sites for meta-model training?**

The answer involves a **multi-stage data transformation pipeline** that converts variant-level information into splice site-level training data, enabling the meta-model to learn patterns of alternative splicing and predict new alternative splicing patterns under similar mutations or diseases.

**Key Insight**: VCF is the standardized input format, but we need to transform variant impacts into splice site annotations that can be integrated with canonical splice sites for comprehensive meta-model training.

---

## 📋 **THE COMPLETE WORKFLOW PIPELINE**

### **Stage 1: VCF Standardization** ✅
```
Disease Databases → Standardized VCF Format
```

### **Stage 2: Variant Impact Analysis** 🔬
```
VCF + Reference Genome → Delta Scores + Impact Predictions
```

### **Stage 3: Alternative Splice Site Extraction** 🧬
```
Delta Scores → Alternative Splice Site Coordinates
```

### **Stage 4: Training Data Integration** 🎯
```
Alternative Sites + Canonical Sites → alternative_splice_sites.tsv
```

### **Stage 5: Meta-Model Training** 🚀
```
alternative_splice_sites.tsv → Enhanced Meta-Model
```

---

## 🔬 **STAGE 2-3: THE CRITICAL TRANSFORMATION**

### **From Delta Scores to Alternative Splice Sites**

This is the **key missing piece** you've identified! Here's how we transform OpenSpliceAI delta scores into alternative splice site annotations:

#### **Input: OpenSpliceAI Delta Scores**
```
# Example OpenSpliceAI output format:
# ALT|GENE|AG_score|AG_pos|AL_score|AL_pos|DG_score|DG_pos|DL_score|DL_pos
T|CFTR|0.85|+3|0.02|-1|0.12|+5|0.91|-2
```

#### **Transformation Logic**
```python
def extract_alternative_splice_sites_from_delta_scores(
    openspliceai_results: pd.DataFrame,
    reference_gtf: str,
    threshold: float = 0.2
) -> pd.DataFrame:
    """
    Transform OpenSpliceAI delta scores into alternative splice site annotations
    
    This is the CRITICAL function that bridges VCF analysis to training data!
    """
    
    alternative_sites = []
    
    for _, variant_result in openspliceai_results.iterrows():
        # Parse delta score components
        delta_data = parse_delta_score_string(variant_result['SpliceAI'])
        
        # Extract significant alternative splice sites
        for splice_type in ['AG', 'DG']:  # Focus on gains (new sites)
            score = delta_data[f'{splice_type}_score']
            position = delta_data[f'{splice_type}_pos']
            
            if abs(score) >= threshold:
                # Calculate absolute genomic coordinate
                variant_pos = variant_result['POS']
                splice_site_pos = variant_pos + position
                
                # Create alternative splice site record
                alt_site = {
                    'chrom': variant_result['CHROM'],
                    'start': splice_site_pos - 1,  # 0-based start
                    'end': splice_site_pos,        # 1-based end
                    'position': splice_site_pos,   # 1-based position
                    'strand': get_gene_strand(delta_data['GENE']),
                    'site_type': 'donor' if splice_type == 'DG' else 'acceptor',
                    'gene_id': delta_data['GENE'],
                    'transcript_id': get_primary_transcript(delta_data['GENE']),
                    
                    # Alternative splicing specific fields
                    'splice_category': determine_splice_category(score, position),
                    'variant_id': f"{variant_result['CHROM']}_{variant_result['POS']}_{variant_result['REF']}_{variant_result['ALT']}",
                    'delta_score': score,
                    'distance_from_variant': position,
                    'clinical_significance': variant_result.get('CLNSIG', 'Unknown'),
                    'validation_evidence': 'computational_prediction'
                }
                
                alternative_sites.append(alt_site)
    
    return pd.DataFrame(alternative_sites)

def determine_splice_category(delta_score: float, distance: int) -> str:
    """Classify alternative splice sites based on delta score and distance"""
    
    if abs(delta_score) >= 0.8:
        return 'high_confidence_cryptic'
    elif abs(delta_score) >= 0.5:
        return 'cryptic_activated'
    elif abs(delta_score) >= 0.2:
        return 'predicted_alternative'
    else:
        return 'low_confidence_prediction'
```

---

## 🧬 **STAGE 4: COMPREHENSIVE TRAINING DATA INTEGRATION**

### **Merging Canonical and Alternative Splice Sites**

```python
def create_comprehensive_training_annotation(
    canonical_sites_file: str,
    alternative_sites_df: pd.DataFrame,
    output_file: str
) -> pd.DataFrame:
    """
    Create unified training annotation combining canonical and alternative sites
    
    This creates the alternative_splice_sites.tsv for meta-model training!
    """
    
    # Load canonical splice sites
    canonical_sites = pd.read_csv(canonical_sites_file, sep='\t')
    canonical_sites['splice_category'] = 'canonical'
    canonical_sites['variant_id'] = None
    canonical_sites['delta_score'] = None
    canonical_sites['distance_from_variant'] = None
    canonical_sites['clinical_significance'] = None
    canonical_sites['validation_evidence'] = 'reference_annotation'
    
    # Ensure consistent column structure
    required_columns = [
        'chrom', 'start', 'end', 'position', 'strand', 'site_type', 
        'gene_id', 'transcript_id', 'splice_category', 'variant_id',
        'delta_score', 'distance_from_variant', 'clinical_significance',
        'validation_evidence'
    ]
    
    # Standardize canonical sites
    canonical_standardized = canonical_sites.reindex(columns=required_columns)
    
    # Combine canonical and alternative sites
    comprehensive_sites = pd.concat([
        canonical_standardized,
        alternative_sites_df
    ], ignore_index=True)
    
    # Remove duplicates (same genomic position)
    comprehensive_sites = comprehensive_sites.drop_duplicates(
        subset=['chrom', 'position', 'site_type', 'gene_id']
    )
    
    # Sort by genomic coordinates
    comprehensive_sites = comprehensive_sites.sort_values([
        'chrom', 'position', 'gene_id'
    ])
    
    # Save comprehensive annotation
    comprehensive_sites.to_csv(output_file, sep='\t', index=False)
    
    print(f"Created comprehensive annotation with {len(comprehensive_sites)} splice sites:")
    print(f"  - Canonical sites: {len(canonical_standardized)}")
    print(f"  - Alternative sites: {len(alternative_sites_df)}")
    
    return comprehensive_sites
```

---

## 🎯 **STAGE 5: META-MODEL TRAINING DATA PREPARATION**

### **From alternative_splice_sites.tsv to Training Features**

```python
def prepare_alternative_splicing_training_data(
    alternative_splice_sites_file: str,
    reference_fasta: str,
    output_dir: str,
    context_length: int = 10000
) -> dict:
    """
    Prepare training data from comprehensive alternative splice site annotation
    
    This creates the actual training dataset for meta-model learning!
    """
    
    # Load comprehensive splice site annotation
    splice_sites = pd.read_csv(alternative_splice_sites_file, sep='\t')
    
    training_data = {
        'sequences': [],
        'labels': [],
        'features': [],
        'metadata': []
    }
    
    fasta = Fasta(reference_fasta)
    
    for _, site in splice_sites.iterrows():
        # Extract sequence context around splice site
        chrom = site['chrom']
        position = site['position']
        
        # Get sequence context
        start_pos = max(1, position - context_length // 2)
        end_pos = position + context_length // 2
        sequence = str(fasta[chrom][start_pos:end_pos])
        
        # Create label based on splice category
        label = create_training_label(site)
        
        # Extract features
        features = extract_splice_site_features(site, sequence)
        
        # Create metadata
        metadata = {
            'splice_category': site['splice_category'],
            'variant_id': site['variant_id'],
            'delta_score': site['delta_score'],
            'clinical_significance': site['clinical_significance'],
            'gene_id': site['gene_id']
        }
        
        training_data['sequences'].append(sequence)
        training_data['labels'].append(label)
        training_data['features'].append(features)
        training_data['metadata'].append(metadata)
    
    # Save training data
    save_training_data(training_data, output_dir)
    
    return training_data

def create_training_label(site: pd.Series) -> dict:
    """Create training label from splice site annotation"""
    
    # Multi-task learning labels
    label = {
        'is_splice_site': 1 if site['site_type'] in ['donor', 'acceptor'] else 0,
        'site_type': encode_site_type(site['site_type']),
        'splice_category': encode_splice_category(site['splice_category']),
        'pathogenicity': encode_clinical_significance(site['clinical_significance']),
        'delta_score_regression': site['delta_score'] if pd.notna(site['delta_score']) else 0.0
    }
    
    return label

def encode_splice_category(category: str) -> int:
    """Encode splice category for training"""
    category_mapping = {
        'canonical': 0,
        'cryptic_activated': 1,
        'canonical_disrupted': 2,
        'disease_associated': 3,
        'predicted_alternative': 4,
        'high_confidence_cryptic': 5
    }
    return category_mapping.get(category, 0)
```

---

## 🚀 **COMPLETE END-TO-END WORKFLOW IMPLEMENTATION**

### **Master Workflow Class**

```python
class AlternativeSplicingTrainingPipeline:
    """
    Complete pipeline from VCF variant analysis to meta-model training data
    
    This is the COMPLETE SOLUTION to your question!
    """
    
    def __init__(self, 
                 reference_fasta: str,
                 reference_gtf: str,
                 canonical_splice_sites: str,
                 openspliceai_path: str):
        self.reference_fasta = reference_fasta
        self.reference_gtf = reference_gtf
        self.canonical_splice_sites = canonical_splice_sites
        self.openspliceai_path = openspliceai_path
    
    def process_disease_database(self, 
                               database_name: str,
                               database_file: str,
                               output_dir: str) -> dict:
        """
        Complete workflow from disease database to training data
        
        This answers your question: How do we go from VCF to training data?
        """
        
        results = {}
        
        # Stage 1: Convert database to VCF
        print(f"Stage 1: Converting {database_name} to VCF format...")
        vcf_file = self.convert_database_to_vcf(database_file, database_name)
        results['vcf_file'] = vcf_file
        
        # Stage 2: Run OpenSpliceAI variant analysis
        print("Stage 2: Running OpenSpliceAI variant analysis...")
        delta_scores_file = self.run_openspliceai_analysis(vcf_file)
        results['delta_scores_file'] = delta_scores_file
        
        # Stage 3: Extract alternative splice sites
        print("Stage 3: Extracting alternative splice sites...")
        alternative_sites = self.extract_alternative_splice_sites(delta_scores_file)
        results['alternative_sites'] = alternative_sites
        
        # Stage 4: Create comprehensive annotation
        print("Stage 4: Creating comprehensive splice site annotation...")
        comprehensive_annotation = self.create_comprehensive_annotation(
            alternative_sites, output_dir
        )
        results['comprehensive_annotation'] = comprehensive_annotation
        
        # Stage 5: Prepare training data
        print("Stage 5: Preparing meta-model training data...")
        training_data = self.prepare_training_data(
            comprehensive_annotation, output_dir
        )
        results['training_data'] = training_data
        
        print(f"✅ Complete pipeline finished for {database_name}")
        print(f"📊 Training data: {len(training_data['sequences'])} examples")
        print(f"📁 Output directory: {output_dir}")
        
        return results
    
    def convert_database_to_vcf(self, database_file: str, database_name: str) -> str:
        """Convert disease database to standardized VCF format"""
        
        if database_name == 'clinvar':
            return self.convert_clinvar_to_vcf(database_file)
        elif database_name == 'splicevardb':
            return self.convert_splicevardb_to_vcf(database_file)
        elif database_name == 'mutsplicedb':
            return self.convert_mutsplicedb_to_vcf(database_file)
        else:
            raise ValueError(f"Unknown database: {database_name}")
    
    def run_openspliceai_analysis(self, vcf_file: str) -> str:
        """Run OpenSpliceAI variant analysis on VCF file"""
        
        output_file = vcf_file.replace('.vcf', '_openspliceai.vcf')
        
        cmd = [
            'python', '-m', 'openspliceai', 'variant',
            '-R', self.reference_fasta,
            '-A', 'grch38',
            '-I', vcf_file,
            '-O', output_file,
            '-D', '50',
            '--flanking-size', '5000'
        ]
        
        subprocess.run(cmd, check=True)
        return output_file
    
    def extract_alternative_splice_sites(self, delta_scores_file: str) -> pd.DataFrame:
        """Extract alternative splice sites from OpenSpliceAI delta scores"""
        
        # Parse OpenSpliceAI VCF output
        openspliceai_results = self.parse_openspliceai_vcf(delta_scores_file)
        
        # Transform to alternative splice sites
        alternative_sites = extract_alternative_splice_sites_from_delta_scores(
            openspliceai_results, self.reference_gtf
        )
        
        return alternative_sites
    
    def create_comprehensive_annotation(self, 
                                      alternative_sites: pd.DataFrame,
                                      output_dir: str) -> str:
        """Create comprehensive splice site annotation file"""
        
        output_file = os.path.join(output_dir, 'alternative_splice_sites.tsv')
        
        comprehensive_annotation = create_comprehensive_training_annotation(
            self.canonical_splice_sites,
            alternative_sites,
            output_file
        )
        
        return output_file
    
    def prepare_training_data(self, 
                            comprehensive_annotation_file: str,
                            output_dir: str) -> dict:
        """Prepare final training data for meta-model"""
        
        training_data = prepare_alternative_splicing_training_data(
            comprehensive_annotation_file,
            self.reference_fasta,
            output_dir
        )
        
        return training_data
```

---

## 📊 **EXAMPLE USAGE**

### **Complete Workflow Execution**

```python
# Initialize pipeline
pipeline = AlternativeSplicingTrainingPipeline(
    reference_fasta='data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
    reference_gtf='data/ensembl/Homo_sapiens.GRCh38.112.gtf',
    canonical_splice_sites='data/ensembl/splice_sites.tsv',
    openspliceai_path='meta_spliceai/openspliceai'
)

# Process ClinVar database
clinvar_results = pipeline.process_disease_database(
    database_name='clinvar',
    database_file='data/case_studies/clinvar/clinvar_variants.txt',
    output_dir='data/case_studies/clinvar/training_data'
)

# Process SpliceVarDB database
splicevardb_results = pipeline.process_disease_database(
    database_name='splicevardb',
    database_file='data/case_studies/splicevardb/splicevardb_variants.tsv',
    output_dir='data/case_studies/splicevardb/training_data'
)

# Combine training data from multiple databases
combined_training_data = combine_training_datasets([
    clinvar_results['training_data'],
    splicevardb_results['training_data']
])

print(f"✅ Combined training dataset ready:")
print(f"📊 Total examples: {len(combined_training_data['sequences'])}")
print(f"🧬 Canonical sites: {sum(1 for m in combined_training_data['metadata'] if m['splice_category'] == 'canonical')}")
print(f"🔬 Alternative sites: {sum(1 for m in combined_training_data['metadata'] if m['splice_category'] != 'canonical')}")
```

---

## 🎯 **KEY INSIGHTS FOR META-MODEL TRAINING**

### **What the Meta-Model Learns**

With this comprehensive training data, the meta-model can learn to:

1. **Distinguish Splice Site Types**: Canonical vs. cryptic vs. disease-associated
2. **Predict Alternative Splicing**: Given a variant, predict new splice sites
3. **Assess Clinical Impact**: Link splice site changes to pathogenicity
4. **Generalize Across Diseases**: Learn patterns that apply to new mutations

### **Training Data Characteristics**

```python
# Example training data structure
training_example = {
    'sequence': 'ATCGATCG...',  # 10kb sequence context
    'labels': {
        'is_splice_site': 1,
        'site_type': 'donor',
        'splice_category': 'cryptic_activated',
        'pathogenicity': 'pathogenic',
        'delta_score_regression': 0.85
    },
    'features': {
        'position_features': [...],
        'sequence_features': [...],
        'conservation_features': [...]
    },
    'metadata': {
        'variant_id': 'chr7_117559590_T_G',
        'gene_id': 'ENSG00000001626',
        'clinical_significance': 'Pathogenic',
        'disease': 'cystic_fibrosis'
    }
}
```

---

## 🚀 **CONCLUSION**

**Answer to your question**: Yes, VCF is the standardized input format, but the key is the **transformation pipeline** that converts variant-level delta scores into splice site-level training annotations.

**The complete workflow**:
1. **VCF Standardization**: Disease databases → VCF format
2. **Delta Score Analysis**: VCF → OpenSpliceAI → Delta scores
3. **Site Extraction**: Delta scores → Alternative splice site coordinates
4. **Data Integration**: Alternative sites + Canonical sites → `alternative_splice_sites.tsv`
5. **Training Preparation**: Comprehensive annotation → Meta-model training data

**This enables the meta-model to**:
- Learn from both canonical and alternative splice sites
- Predict new alternative splicing patterns under similar mutations
- Generalize across different diseases and mutation types
- Provide clinical impact assessment for novel variants

The key insight is that **variants create alternative splice sites**, and by systematically extracting these sites from variant analysis, we create comprehensive training data that teaches the meta-model to recognize and predict alternative splicing patterns! 🧬🎯
