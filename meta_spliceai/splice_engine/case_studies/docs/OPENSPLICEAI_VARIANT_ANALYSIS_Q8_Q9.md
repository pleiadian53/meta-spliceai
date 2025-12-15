# OpenSpliceAI Variant Analysis Capabilities (Q8-Q9)

**Document Version**: 1.0  
**Date**: 2025-07-28  
**Purpose**: Comprehensive analysis of OpenSpliceAI's variant analysis capabilities and ClinVar integration potential

---

## 📋 **EXECUTIVE SUMMARY**

This document addresses questions Q8-Q9 regarding OpenSpliceAI's variant analysis capabilities, ClinVar integration status, and the specific mechanisms for variant scoring and impact prediction. Based on comprehensive code analysis of the integrated OpenSpliceAI subpackage (`meta_spliceai/openspliceai`), we provide detailed answers about current capabilities and integration opportunities.

**Key Findings**:
- ❌ **No Direct ClinVar Integration**: OpenSpliceAI does not currently integrate ClinVar data directly
- ✅ **Robust Variant Analysis Framework**: Comprehensive VCF-based variant scoring system
- ✅ **Delta Score Calculation**: Sophisticated mechanism for computing variant impacts
- ✅ **Integration Opportunity**: Clear pathway for ClinVar integration through existing infrastructure

---

## 🎯 **Q8: ClinVar Integration Analysis**

### **Question**
Does OpenSpliceAI directly integrate ClinVar data for variant effect evaluation on splicing?

### **Answer: NO - No Direct ClinVar Integration**

#### **Evidence from Code Analysis**
```bash
# Search Results:
grep -r "ClinVar" meta_spliceai/openspliceai/
# No results found

find meta_spliceai/openspliceai/ -name "*clinvar*"  
# No ClinVar-specific files found
```

#### **Current Variant Analysis Architecture**
OpenSpliceAI processes variants through a **VCF-based workflow** without direct database integration:

```python
# From openspliceai/variant/variant.py
def variant(args):
    """
    Annotates variants in a VCF file using SpliceAI-toolkit
    - Reads input VCF file
    - Annotates each variant with delta scores and delta positions  
    - Writes annotated variants to output VCF file
    """
    
    # Required inputs:
    # - input_vcf: VCF file with variants
    # - ref_genome: Reference genome FASTA
    # - annotation: Gene annotation (grch37/grch38)
    # - model: SpliceAI model path
```

#### **Variant Processing Workflow**
```
1. VCF Input → 2. Sequence Extraction → 3. Model Prediction → 4. Delta Score Calculation → 5. Annotated VCF Output
```

### **ClinVar Integration Opportunities**

#### **Option 1: Preprocessing Integration**
```python
# Proposed ClinVar preprocessing workflow
def preprocess_clinvar_to_vcf(clinvar_file: str, output_vcf: str):
    """Convert ClinVar data to VCF format for OpenSpliceAI processing"""
    
    clinvar_df = pd.read_csv(clinvar_file, sep='\t')
    
    # Filter for splice-affecting variants
    splice_variants = clinvar_df[
        clinvar_df['ClinicalSignificance'].isin(['Pathogenic', 'Likely pathogenic']) &
        clinvar_df['MolecularConsequence'].str.contains('splice', case=False, na=False)
    ]
    
    # Convert to VCF format
    vcf_records = []
    for _, variant in splice_variants.iterrows():
        vcf_record = create_vcf_record(
            chrom=variant['Chromosome'],
            pos=variant['Start'],
            ref=variant['ReferenceAllele'], 
            alt=variant['AlternateAllele'],
            info={'CLNSIG': variant['ClinicalSignificance']}
        )
        vcf_records.append(vcf_record)
    
    write_vcf(vcf_records, output_vcf)
```

#### **Option 2: Post-Processing Integration**
```python
# Proposed ClinVar annotation enrichment
def enrich_openspliceai_with_clinvar(openspliceai_results: str, 
                                   clinvar_annotations: str,
                                   output_file: str):
    """Enrich OpenSpliceAI results with ClinVar clinical significance"""
    
    # Load OpenSpliceAI delta scores
    spliceai_df = pd.read_csv(openspliceai_results, sep='\t')
    
    # Load ClinVar annotations  
    clinvar_df = pd.read_csv(clinvar_annotations, sep='\t')
    
    # Merge on genomic coordinates
    enriched_results = spliceai_df.merge(
        clinvar_df, 
        on=['chrom', 'pos', 'ref', 'alt'],
        how='left'
    )
    
    # Add clinical interpretation
    enriched_results['clinical_impact'] = enriched_results.apply(
        lambda row: interpret_clinical_impact(
            row['delta_score'], row['ClinicalSignificance']
        ), axis=1
    )
    
    enriched_results.to_csv(output_file, sep='\t', index=False)
```

#### **Option 3: Direct Integration (Recommended)**
```python
# Proposed direct ClinVar integration in Annotator class
class ClinVarEnhancedAnnotator(Annotator):
    """Extended Annotator with ClinVar integration"""
    
    def __init__(self, ref_fasta, annotations, clinvar_file=None, **kwargs):
        super().__init__(ref_fasta, annotations, **kwargs)
        
        if clinvar_file:
            self.clinvar_data = self._load_clinvar_data(clinvar_file)
        else:
            self.clinvar_data = None
    
    def _load_clinvar_data(self, clinvar_file: str) -> pd.DataFrame:
        """Load and index ClinVar data for rapid lookup"""
        clinvar_df = pd.read_csv(clinvar_file, sep='\t')
        
        # Create genomic coordinate index
        clinvar_df['coord_key'] = (
            clinvar_df['Chromosome'].astype(str) + '_' +
            clinvar_df['Start'].astype(str) + '_' +
            clinvar_df['ReferenceAllele'] + '_' +
            clinvar_df['AlternateAllele']
        )
        
        return clinvar_df.set_index('coord_key')
    
    def get_clinvar_annotation(self, chrom: str, pos: int, ref: str, alt: str) -> dict:
        """Retrieve ClinVar annotation for a variant"""
        coord_key = f"{chrom}_{pos}_{ref}_{alt}"
        
        if self.clinvar_data is not None and coord_key in self.clinvar_data.index:
            clinvar_record = self.clinvar_data.loc[coord_key]
            return {
                'clinical_significance': clinvar_record['ClinicalSignificance'],
                'review_status': clinvar_record['ReviewStatus'],
                'molecular_consequence': clinvar_record['MolecularConsequence'],
                'disease_name': clinvar_record['PhenotypeList']
            }
        
        return {'clinical_significance': 'Unknown'}
```

---

## 🔬 **Q9: Variant Analysis Subcommands and Delta Score Computation**

### **Question**
Please point out the location of the variant analysis subcommands and delta score computation mechanisms.

### **Answer: Comprehensive Variant Analysis Framework Located**

#### **🎯 Primary Variant Analysis Locations**

##### **1. Main Variant Command Entry Point**
```
📁 meta_spliceai/openspliceai/openspliceai.py
├── parse_args_variant() [Lines 123-141]
└── main() [Lines 162-197]
```

**Command Line Interface**:
```bash
# OpenSpliceAI variant analysis command
python -m openspliceai variant \
    -R reference.fa \
    -A grch38 \
    -I input.vcf \
    -O output.vcf \
    -D 50 \
    --model SpliceAI \
    --flanking-size 5000
```

##### **2. Core Variant Processing Module**
```
📁 meta_spliceai/openspliceai/variant/
├── variant.py [Main variant processing workflow]
├── utils.py [Delta score calculation and utilities]
└── get_anno.py [Annotation retrieval functions]
```

##### **3. Delta Score Calculation Engine**
```
📍 meta_spliceai/openspliceai/variant/utils.py
└── get_delta_scores() [Lines 351-540]
```

#### **🧬 Delta Score Computation Mechanism**

##### **Core Algorithm Overview**
```python
def get_delta_scores(record, ann, dist_var, mask, flanking_size=5000, precision=2):
    """
    Calculate delta scores for variant impacts on splice sites.
    
    Process:
    1. Extract reference and alternative sequences
    2. One-hot encode sequences  
    3. Run SpliceAI prediction on both sequences
    4. Calculate delta scores (alternative - reference)
    5. Identify gained/lost splice sites within distance threshold
    """
```

##### **Step-by-Step Delta Score Calculation**

###### **Step 1: Sequence Extraction**
```python
# Define coverage and window size around the variant
cov = 2 * dist_var + 1  # Coverage window (default: 101bp for dist_var=50)
wid = 2 * flanking_size + cov  # Total window (default: 10,101bp for flanking_size=5000)

# Extract reference sequence around variant
seq = ann.ref_fasta[chrom][record.pos - wid // 2 - 1 : record.pos + wid // 2].seq

# Create alternative sequence by applying variant
x_ref = 'N' * pad_size[0] + seq[pad_size[0]: wid - pad_size[1]] + 'N' * pad_size[1]
x_alt = x_ref[: wid // 2] + str(record.alts[j]) + x_ref[wid // 2 + ref_len:]
```

###### **Step 2: Model Prediction**
```python
# One-hot encode sequences
x_ref = one_hot_encode(x_ref)[None, :]
x_alt = one_hot_encode(x_alt)[None, :]

# Handle strand orientation
if strands[i] == '-':
    x_ref = x_ref[:, ::-1, ::-1]  # Reverse complement
    x_alt = x_alt[:, ::-1, ::-1]

# Predict splice probabilities using ensemble of models
y_ref = np.mean([ann.models[m].predict(x_ref) for m in range(len(ann.models))], axis=0)
y_alt = np.mean([ann.models[m].predict(x_alt) for m in range(len(ann.models))], axis=0)
```

###### **Step 3: Delta Score Calculation**
```python
# Calculate delta scores for each splice type
y_diff = y_alt[0, :] - y_ref[0, :]  # Alternative - Reference

# Extract delta scores for specific splice site types:
# - Acceptor Gain (AG): y_diff[:, 1] 
# - Acceptor Loss (AL): -y_diff[:, 1]
# - Donor Gain (DG): y_diff[:, 2]
# - Donor Loss (DL): -y_diff[:, 2]

# Find maximum delta scores within distance threshold
idx_pa = (y_diff[cov // 2 - dist_var : cov // 2 + dist_var + 1, 1] > 0.1).nonzero()[0]
idx_na = (y_diff[cov // 2 - dist_var : cov // 2 + dist_var + 1, 1] < -0.1).nonzero()[0]
idx_pd = (y_diff[cov // 2 - dist_var : cov // 2 + dist_var + 1, 2] > 0.1).nonzero()[0]
idx_nd = (y_diff[cov // 2 - dist_var : cov // 2 + dist_var + 1, 2] < -0.1).nonzero()[0]
```

###### **Step 4: Impact Assessment**
```python
# Calculate maximum delta scores and positions
if len(idx_pa) > 0:
    max_idx = idx_pa[np.argmax(y_diff[cov // 2 - dist_var + idx_pa, 1])]
    acceptor_gain_score = y_diff[cov // 2 - dist_var + max_idx, 1]
    acceptor_gain_position = max_idx - dist_var
else:
    acceptor_gain_score = 0.0
    acceptor_gain_position = 0

# Similar calculations for acceptor_loss, donor_gain, donor_loss
```

##### **Output Format**
```python
# Delta scores are formatted as:
# "ALT|GENE|AG_score|AG_pos|AL_score|AL_pos|DG_score|DG_pos|DL_score|DL_pos"

delta_score_string = f"{record.alts[j]}|{genes[i]}|{ag:.2f}|{ag_pos}|{al:.2f}|{al_pos}|{dg:.2f}|{dg_pos}|{dl:.2f}|{dl_pos}"
```

#### **🎯 Variant Scoring Parameters**

##### **Key Configuration Options**
```python
# Command line parameters affecting delta score calculation:
parser_variant.add_argument('-D', '--distance', default=50, type=int,
    help='maximum distance between variant and gained/lost splice site')

parser_variant.add_argument('-M', '--mask', default=0, type=int, choices=[0, 1],
    help='mask scores representing annotated acceptor/donor gain')

parser_variant.add_argument('--flanking-size', '-f', type=int, default=80,
    help='Sum of flanking sequence lengths on each side of input')

parser_variant.add_argument('--precision', '-p', type=int, default=2,
    help='Number of decimal places to round the output scores')
```

##### **Model Support**
```python
# Supported model types:
parser_variant.add_argument('--model-type', '-t', type=str, 
    choices=['keras', 'pytorch'], default='pytorch')

# Model path options:
parser_variant.add_argument('--model', '-m', default="SpliceAI", type=str,
    help='Path to SpliceAI model file, directory of models, or "SpliceAI" for default')
```

#### **🔧 Functional Consequence Assessment**

##### **Delta Score Interpretation**
```python
# Interpretation guidelines (from SpliceAI literature):
def interpret_delta_score(delta_score: float) -> str:
    """Interpret delta score clinical significance"""
    
    if abs(delta_score) >= 0.8:
        return "high_impact"      # Likely pathogenic
    elif abs(delta_score) >= 0.5:
        return "moderate_impact"  # Possibly pathogenic  
    elif abs(delta_score) >= 0.2:
        return "low_impact"       # Uncertain significance
    else:
        return "minimal_impact"   # Likely benign

# Positional impact assessment:
def assess_positional_impact(delta_position: int, distance_threshold: int = 50) -> str:
    """Assess impact based on distance from variant"""
    
    if abs(delta_position) <= 2:
        return "direct_impact"    # Variant directly affects splice site
    elif abs(delta_position) <= 10:
        return "proximal_impact"  # Variant affects nearby regulatory elements
    elif abs(delta_position) <= distance_threshold:
        return "distal_impact"    # Variant affects distant regulatory elements
    else:
        return "no_impact"        # Beyond detection threshold
```

##### **Splice Site Type Classification**
```python
# Delta score components:
class SpliceImpactType(Enum):
    ACCEPTOR_GAIN = "AG"    # Creation of new acceptor site
    ACCEPTOR_LOSS = "AL"    # Disruption of existing acceptor site  
    DONOR_GAIN = "DG"       # Creation of new donor site
    DONOR_LOSS = "DL"       # Disruption of existing donor site

# Combined impact assessment:
def assess_combined_impact(ag: float, al: float, dg: float, dl: float) -> dict:
    """Assess combined splicing impact from all delta scores"""
    
    max_gain = max(ag, dg)
    max_loss = max(abs(al), abs(dl))
    
    return {
        'primary_effect': 'gain' if max_gain > max_loss else 'loss',
        'effect_magnitude': max(max_gain, max_loss),
        'complexity': 'simple' if (ag > 0.1) + (al < -0.1) + (dg > 0.1) + (dl < -0.1) == 1 else 'complex'
    }
```

---

## 🔗 **INTEGRATION WITH METASPLICEAI CASE STUDIES**

### **Leveraging OpenSpliceAI for Case Study Validation**

#### **Current Integration Points**
```python
# Existing integration through AlignedSpliceExtractor
from meta_spliceai.splice_engine.meta_models.openspliceai_adapter import AlignedSpliceExtractor

# Proposed variant analysis integration
class VariantAnalysisWorkflow:
    """Integrate OpenSpliceAI variant analysis with case study framework"""
    
    def __init__(self, openspliceai_path: str, case_study_databases: List[str]):
        self.openspliceai = openspliceai_path
        self.databases = case_study_databases
    
    def analyze_disease_variants(self, database_name: str) -> pd.DataFrame:
        """Analyze variants from specific disease database"""
        
        # Step 1: Convert database to VCF format
        vcf_file = self.convert_database_to_vcf(database_name)
        
        # Step 2: Run OpenSpliceAI variant analysis
        results_file = self.run_openspliceai_analysis(vcf_file)
        
        # Step 3: Parse and interpret results
        delta_scores = self.parse_delta_scores(results_file)
        
        # Step 4: Integrate with clinical annotations
        enriched_results = self.enrich_with_clinical_data(delta_scores, database_name)
        
        return enriched_results
```

#### **Case Study Database Integration**
```python
# Integration with existing case study databases
database_integrations = {
    'splicevardb': {
        'converter': 'convert_splicevardb_to_vcf',
        'enricher': 'enrich_with_experimental_validation'
    },
    'clinvar': {
        'converter': 'convert_clinvar_to_vcf', 
        'enricher': 'enrich_with_clinical_significance'
    },
    'mutsplicedb': {
        'converter': 'convert_mutsplicedb_to_vcf',
        'enricher': 'enrich_with_cancer_context'
    },
    'dbass': {
        'converter': 'convert_dbass_to_vcf',
        'enricher': 'enrich_with_cryptic_site_data'
    }
}
```

---

## 📊 **IMPLEMENTATION RECOMMENDATIONS**

### **Phase 1: Direct VCF Integration** (Immediate)
- ✅ Use existing OpenSpliceAI variant analysis with VCF inputs
- ✅ Convert case study databases to VCF format
- ✅ Process through OpenSpliceAI variant command
- ✅ Parse delta scores for case study validation

### **Phase 2: Enhanced ClinVar Integration** (Short-term)
- 🔧 Implement ClinVar preprocessing pipeline
- 🔧 Create clinical significance enrichment workflow
- 🔧 Develop automated ClinVar update mechanism
- 🔧 Integrate with existing case study framework

### **Phase 3: Native Database Integration** (Long-term)
- 🚀 Extend Annotator class with database integration
- 🚀 Implement real-time clinical annotation lookup
- 🚀 Create unified variant analysis API
- 🚀 Develop comprehensive validation framework

---

## 🎯 **CONCLUSION**

### **Q8 Summary: ClinVar Integration Status**
- **❌ No Direct Integration**: OpenSpliceAI does not currently integrate ClinVar data
- **✅ Clear Integration Path**: VCF-based workflow provides straightforward integration opportunity
- **🔧 Multiple Options**: Preprocessing, post-processing, or direct integration approaches available

### **Q9 Summary: Variant Analysis Capabilities**
- **📍 Location**: `meta_spliceai/openspliceai/variant/` directory contains all variant analysis functionality
- **🧬 Delta Scores**: Sophisticated calculation using reference vs. alternative sequence predictions
- **🎯 Impact Assessment**: Comprehensive evaluation of acceptor/donor gain/loss with positional information
- **🔧 Configurable**: Flexible parameters for distance thresholds, flanking sizes, and precision

### **Integration Opportunity**
The existing OpenSpliceAI variant analysis framework provides an excellent foundation for comprehensive case study validation. By implementing the proposed ClinVar integration strategies, we can create a powerful system that combines:

- **🧬 Computational Predictions**: OpenSpliceAI delta scores
- **🏥 Clinical Annotations**: ClinVar pathogenicity classifications  
- **📊 Experimental Validation**: SpliceVarDB experimental evidence
- **🎯 Meta-Learning Enhancement**: MetaSpliceAI meta-model improvements

This integration will enable unprecedented validation of splice site analysis against real-world clinical variants, providing the confidence needed for clinical deployment of the MetaSpliceAI meta-learning framework.
