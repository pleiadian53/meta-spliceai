# OpenSpliceAI vs MetaSpliceAI: Data Generation Comparison

**Date**: October 15, 2025  
**Purpose**: Compare `create_datafile.py` (OpenSpliceAI) vs MetaSpliceAI's splice site annotation generation

---

## Executive Summary

Both systems generate splice site annotations from GTF files, but with **fundamentally different purposes and outputs**:

| Aspect | OpenSpliceAI | MetaSpliceAI |
|--------|-------------|----------------|
| **Purpose** | Training data for ML model | Meta-learning feature extraction |
| **Output Format** | HDF5 with per-nucleotide labels | TSV with splice site positions |
| **Granularity** | Dense sequence + label arrays | Sparse site annotations |
| **Scope** | Per-gene sequences | Genome-wide splice sites |
| **Use Case** | Train base splice predictor | Train meta-learner on base predictions |

**Key Insight**: OpenSpliceAI creates **training data** for the base model, while MetaSpliceAI creates **annotations** for meta-learning.

---

## 1. OpenSpliceAI's `create_datafile.py`

### **Purpose**: Generate Training Data for ML Model

Creates HDF5 files containing:
- Full gene sequences
- Per-nucleotide labels (0=no site, 1=acceptor, 2=donor)
- One entry per gene

### **Workflow**

```
GTF + FASTA → gffutils DB → Per-Gene Processing → HDF5 Training Data
```

#### **Step-by-Step Process**:

```python
# 1. Load GTF into gffutils database
db = gffutils.create_db(gff_file, dbfn=db_file)

# 2. For each protein-coding gene:
for gene in db.features_of_type('gene'):
    if gene.attributes["gene_biotype"][0] == "protein_coding":
        
        # 3. Extract full gene sequence
        gene_seq = seq_dict[gene.seqid].seq[gene.start-1:gene.end].upper()
        
        # 4. Initialize labels (all zeros)
        labels = [0] * len(gene_seq)
        
        # 5. Select longest transcript
        transcripts = list(db.children(gene, featuretype='mRNA'))
        max_trans = max(transcripts, key=lambda t: t.end - t.start + 1)
        
        # 6. Get exons and label splice sites
        exons = list(db.children(max_trans, featuretype='exon'))
        for i in range(len(exons) - 1):
            # Donor: one base after exon end
            first_site = exons[i].end - gene.start + 1
            # Acceptor: at next exon start  
            second_site = exons[i + 1].start - gene.start
            
            # Label based on strand
            if gene.strand == '+':
                labels[first_site] = 2       # Donor
                labels[second_site - 2] = 1  # Acceptor
            elif gene.strand == '-':
                labels[len(labels) - first_site - 2] = 1  # Acceptor
                labels[len(labels) - second_site] = 2      # Donor
        
        # 7. For minus strand: reverse complement sequence
        if gene.strand == '-':
            gene_seq = gene_seq.reverse_complement()
        
        # 8. Store in HDF5
        NAME.append(gene_id)
        SEQ.append(str(gene_seq))
        LABEL.append(''.join(str(num) for num in labels))

# 9. Save to HDF5 file
h5f.create_dataset('NAME', data=NAME)
h5f.create_dataset('SEQ', data=SEQ)
h5f.create_dataset('LABEL', data=LABEL)
```

### **Output Format**: HDF5

```python
# Structure of datafile_train.h5:
{
    'NAME': ['ENSG00000001', 'ENSG00000002', ...],     # Gene IDs
    'CHROM': ['chr1', 'chr1', ...],                     # Chromosomes
    'STRAND': ['+', '-', ...],                          # Strands
    'TX_START': ['100', '5000', ...],                   # Gene starts
    'TX_END': ['5000', '8000', ...],                    # Gene ends
    'SEQ': ['ATCGATCG...', 'GCTAGCTA...', ...],        # Full gene sequences
    'LABEL': ['000002000100...', '000100020...', ...]  # Per-nucleotide labels
}

# Label encoding:
# 0 = no splice site
# 1 = acceptor site
# 2 = donor site
```

### **Key Characteristics**

1. **Per-Gene Granularity**: One entry per gene
2. **Dense Labeling**: Every nucleotide has a label
3. **Sequence Included**: Full gene sequence stored
4. **Strand Normalization**: Minus strand sequences are reverse-complemented
5. **Gene-Relative Coordinates**: Labels are relative to gene start
6. **Training-Optimized**: Format designed for ML model training

---

## 2. MetaSpliceAI's Splice Site Generation

### **Purpose**: Extract Splice Site Annotations for Meta-Learning

Creates TSV files containing:
- Genomic coordinates of each splice site
- Site type (donor/acceptor)
- Gene/transcript associations
- Additional metadata (biotype, exon rank, etc.)

### **Workflow**

```
GTF → Extract Exon Boundaries → Identify Splice Sites → TSV Annotations
```

#### **Step-by-Step Process**:

```python
# From meta_spliceai/splice_engine/extract_genomic_features.py

# 1. Load GTF and extract splice sites
def prepare_splice_site_annotations(gtf_file_path, output_file, ...):
    
    # 2. Parse GTF
    gtf_df = pr.read_gtf(gtf_file_path)
    
    # 3. Extract exons
    exons = gtf_df[gtf_df['Feature'] == 'exon']
    
    # 4. Group by transcript
    for transcript_id, transcript_exons in exons.groupby('transcript_id'):
        
        # 5. Sort exons by position
        transcript_exons = transcript_exons.sort_values('Start')
        
        # 6. Identify splice sites from exon boundaries
        for i in range(len(transcript_exons) - 1):
            curr_exon = transcript_exons.iloc[i]
            next_exon = transcript_exons.iloc[i + 1]
            
            # Donor: last position of current exon (GTF coordinate)
            donor = {
                'chrom': curr_exon['Chromosome'],
                'position': curr_exon['End'],  # GTF 1-based
                'strand': curr_exon['Strand'],
                'site_type': 'donor',
                'gene_id': curr_exon['gene_id'],
                'transcript_id': transcript_id
            }
            
            # Acceptor: first position of next exon (GTF coordinate)
            acceptor = {
                'chrom': next_exon['Chromosome'],
                'position': next_exon['Start'],  # GTF 1-based
                'strand': next_exon['Strand'],
                'site_type': 'acceptor',
                'gene_id': next_exon['gene_id'],
                'transcript_id': transcript_id
            }
            
            splice_sites.append(donor)
            splice_sites.append(acceptor)
    
    # 7. Save to TSV
    df = pd.DataFrame(splice_sites)
    df.to_csv(output_file, sep='\t', index=False)
```

### **Output Format**: TSV

```
chrom    start    end    position    strand    site_type    gene_id           transcript_id
1        100      100    100         +         donor        ENSG00000001      ENST00000001
1        500      500    500         +         acceptor     ENSG00000001      ENST00000001
1        600      600    600         +         donor        ENSG00000001      ENST00000001
1        1000     1000   1000        +         acceptor     ENSG00000001      ENST00000001
```

### **Enhanced Version** (`splice_sites_enhanced.tsv`):

```
chrom position strand site_type gene_id gene_name transcript_id exon_id exon_rank gene_biotype transcript_biotype
1     100      +      donor     ENSG001 BRCA1     ENST001       ENSE001 1         protein_coding mRNA
1     500      +      acceptor  ENSG001 BRCA1     ENST001       ENSE002 2         protein_coding mRNA
```

### **Key Characteristics**

1. **Per-Site Granularity**: One row per splice site
2. **Sparse Representation**: Only splice sites, not every nucleotide
3. **Genomic Coordinates**: Absolute genomic positions (not gene-relative)
4. **No Sequence**: Sequences extracted separately when needed
5. **Strand Preservation**: Genomic coordinates, not reverse-complemented
6. **Analysis-Optimized**: Format designed for querying and meta-learning

---

## 3. Detailed Comparison

### **3.1 Data Structure**

| Aspect | OpenSpliceAI | MetaSpliceAI |
|--------|-------------|----------------|
| **File Format** | HDF5 (binary) | TSV (text) |
| **Data Structure** | Per-gene arrays | Per-site rows |
| **Sequence Storage** | Full sequences included | Extracted on demand |
| **Coordinate System** | Gene-relative (0-based) | Genomic absolute (1-based) |
| **Strand Handling** | Reverse complement for `-` | Genomic coordinates preserved |

### **3.2 Splice Site Coordinate Definitions**

#### **OpenSpliceAI** (Lines 90-104):

```python
# Donor: one base AFTER exon end
first_site = exons[i].end - gene.start + 1

# Acceptor: AT next exon start  
second_site = exons[i + 1].start - gene.start

# Plus strand labeling:
labels[first_site] = 2              # Donor at first_site
labels[second_site - 2] = 1         # Acceptor at second_site - 2 (!)

# Minus strand labeling (reverse complement applied):
labels[len(labels) - first_site - 2] = 1   # Acceptor
labels[len(labels) - second_site] = 2       # Donor
```

**Key Points**:
- Donor is `exon.end + 1` (first base of intron)
- Acceptor labeling uses `second_site - 2` (2 bases before exon start)
- Coordinates are **gene-relative**
- Sequences are **reverse-complemented** for minus strand

#### **MetaSpliceAI** (from GTF exon boundaries):

```python
# Donor: LAST base of exon (T in GT dinucleotide)
donor_position = exon.end  # GTF 1-based coordinate

# Acceptor: FIRST base of next exon (A in AG dinucleotide)  
acceptor_position = next_exon.start  # GTF 1-based coordinate

# Coordinates are genomic (not gene-relative)
# No reverse complement (genomic strand preserved)
```

**Key Points**:
- Donor is `exon.end` (last base of exon, T in GT)
- Acceptor is `exon.start` (first base of exon, A in AG)
- Coordinates are **genomic absolute**
- Sequences are **not reverse-complemented**

### **3.3 Offset Summary**

| Site Type | Strand | OpenSpliceAI Position | MetaSpliceAI Position | Offset |
|-----------|--------|----------------------|------------------------|--------|
| Donor | + | `exon.end - gene.start + 1` (gene-relative) | `exon.end` (genomic) | +1 nt (relative) |
| Donor | - | Reverse complement applied | Genomic coordinate | Different frame |
| Acceptor | + | `(exon.start - gene.start) - 2` (labeled) | `exon.start` (genomic) | Context dependent |
| Acceptor | - | Reverse complement applied | Genomic coordinate | Different frame |

**Critical Difference**: OpenSpliceAI uses **gene-relative coordinates with reverse complement** while MetaSpliceAI uses **genomic absolute coordinates**.

### **3.4 Use Case Optimization**

#### **OpenSpliceAI - Training Data Generation**

**Optimized For**:
- Feeding sequences directly to ML models
- Context window extraction (model sees surrounding sequence)
- Batch processing during training
- Efficient random access by gene

**Design Choices**:
- HDF5 for fast array access
- Per-gene organization for mini-batch creation
- Dense labeling for cross-entropy loss
- Reverse complement for consistent 5'→3' orientation

**Training Workflow**:
```python
# Load gene sequence and labels
gene_seq = h5f['SEQ'][gene_idx]
labels = h5f['LABEL'][gene_idx]

# Extract context window around splice site
window = gene_seq[pos-context:pos+context]

# Train model to predict label from sequence
loss = cross_entropy(model(window), labels[pos])
```

#### **MetaSpliceAI - Meta-Learning Feature Extraction**

**Optimized For**:
- Querying splice sites by genomic region
- Joining with external databases (ClinVar, SpliceVarDB)
- Feature engineering for meta-models
- Cross-validation by chromosome

**Design Choices**:
- TSV for easy querying and filtering
- Per-site organization for sparse access
- Genomic coordinates for database integration
- Metadata-rich for feature engineering

**Meta-Learning Workflow**:
```python
# Load splice sites
splice_sites = pd.read_csv('splice_sites_enhanced.tsv', sep='\t')

# Get base model predictions for these sites
base_predictions = spliceai.predict(splice_sites)

# Extract meta-features
meta_features = engineer_features(splice_sites, base_predictions)

# Train meta-model to correct base model
meta_model.fit(meta_features, corrected_labels)
```

---

## 4. Similarities

Despite different purposes, both systems:

1. **Start with GTF + FASTA**: Both use Ensembl GTF and reference genome
2. **Use gffutils**: Both leverage `gffutils` for GTF parsing
3. **Extract Exon Boundaries**: Both identify splice sites from exon junctions
4. **Filter by Gene Type**: Both support filtering (OpenSpliceAI: protein_coding only)
5. **Handle Multiple Transcripts**: Both must choose which transcript(s) to use
6. **Validate Motifs**: Both check for canonical GT-AG dinucleotides

---

## 5. Key Differences

### **5.1 Philosophical Differences**

| Aspect | OpenSpliceAI | MetaSpliceAI |
|--------|-------------|----------------|
| **Paradigm** | "Sequence → Label" (supervised learning) | "Base Prediction → Corrected Prediction" (meta-learning) |
| **Granularity** | Per-nucleotide (dense) | Per-site (sparse) |
| **Reference Frame** | Gene-centric | Genome-centric |
| **Orientation** | 5'→3' normalized | Genomic strand preserved |

### **5.2 Technical Differences**

#### **Coordinate Systems**

```python
# Example: Exon at chr1:1000-2000 on plus strand

# OpenSpliceAI (if gene starts at 500):
donor_position = 2000 - 500 + 1 = 1501  # Gene-relative, 0-based-ish
label_array[1501] = 2                    # Dense array

# MetaSpliceAI:
donor_position = 2000                    # Genomic, 1-based
splice_sites_df.loc[i] = {'chrom': '1', 'position': 2000, ...}  # Sparse table
```

#### **Transcript Selection**

```python
# OpenSpliceAI: Uses longest transcript ONLY
max_trans = max(transcripts, key=lambda t: t.end - t.start + 1)

# MetaSpliceAI: Can use ALL transcripts (creates multiple splice site entries)
for transcript in transcripts:
    extract_splice_sites(transcript)
```

#### **Output Size**

```
# Example: 20,000 protein-coding genes, avg 10 exons per gene

# OpenSpliceAI:
# - 20,000 HDF5 entries (one per gene)
# - Each entry: full gene sequence (10-100kb) + labels
# - Total: ~1-2 GB HDF5 file

# MetaSpliceAI:
# - 20,000 genes × 10 exons × 2 sites (donor+acceptor) = 400,000 rows
# - Each row: ~150 bytes metadata
# - Total: ~60 MB TSV file (plus ~350 MB for enhanced version)
```

### **5.3 Extensibility**

| Feature | OpenSpliceAI | MetaSpliceAI |
|---------|-------------|----------------|
| **Add New Genes** | Regenerate entire HDF5 | Append rows to TSV |
| **Query by Region** | Load all genes in region | Direct SQL-like filtering |
| **Join with Variants** | Complex (need genomic coords) | Simple (genomic coords native) |
| **Multi-Transcript** | Only longest | All transcripts |
| **Feature Engineering** | Requires sequence extraction | Metadata readily available |

---

## 6. Integration in MetaSpliceAI

### **Why We Need Both Approaches**

```
OpenSpliceAI create_datafile.py → Train Base Model → Generate Predictions
                                                               ↓
MetaSpliceAI splice_sites.tsv ← ← ← ← ← ← ← ← ← ← Meta-Learning Features
                                                               ↓
                                                         Train Meta-Model
```

1. **OpenSpliceAI's approach** is needed to **train the base splice site predictor**
   - Creates training data for SpliceAI-like models
   - Optimized for sequence-to-label learning

2. **MetaSpliceAI's approach** is needed for **meta-learning on top of base predictions**
   - Creates annotations for feature engineering
   - Optimized for prediction correction and variant analysis

### **Actual Usage in MetaSpliceAI**

We use **MetaSpliceAI's approach** almost exclusively because:

1. We use **pre-trained** SpliceAI models (don't need to train from scratch)
2. Our focus is **meta-learning** (correcting base model predictions)
3. We need **genomic coordinates** for variant analysis
4. We need **metadata** for feature engineering

**Exception**: If we wanted to **retrain** the base model from scratch, we would use OpenSpliceAI's approach.

---

## 7. Coordinate Reconciliation

### **Converting Between Systems**

```python
from meta_spliceai.splice_engine.meta_models.openspliceai_adapter import (
    SpliceCoordinateReconciler
)

reconciler = SpliceCoordinateReconciler()

# Convert OpenSpliceAI gene-relative → MetaSpliceAI genomic
def openspliceai_to_metaspliceai(gene_relative_pos, gene_start):
    return gene_start + gene_relative_pos - 1  # Adjust for 0/1-based difference

# Convert MetaSpliceAI genomic → OpenSpliceAI gene-relative  
def metaspliceai_to_openspliceai(genomic_pos, gene_start):
    return genomic_pos - gene_start + 1
```

### **Handling Strand Differences**

```python
# OpenSpliceAI: Reverse complement for minus strand
if strand == '-':
    sequence = reverse_complement(sequence)
    # Coordinates are now relative to RC sequence

# MetaSpliceAI: Preserve genomic coordinates
# No reverse complement
# Extract sequence on-demand with strand info
```

---

## 8. Best Practices

### **When to Use OpenSpliceAI Approach**

✅ Training a new base splice site predictor from scratch  
✅ Need dense per-nucleotide labels for CNN/transformer training  
✅ Working with gene-centric sequence context  
✅ Optimizing for training data loading speed  

### **When to Use MetaSpliceAI Approach**

✅ Working with pre-trained base models (SpliceAI, Pangolin, etc.)  
✅ Need to join splice sites with variant databases  
✅ Performing meta-learning or prediction correction  
✅ Need sparse, queryable splice site annotations  
✅ Integrating multiple data sources  

### **Using Both Together**

```python
# 1. Train base model with OpenSpliceAI approach
create_datafile.py → datafile_train.h5 → Train SpliceAI model

# 2. Generate genome-wide predictions
spliceai_model.predict(genome) → base_predictions.tsv

# 3. Use MetaSpliceAI approach for meta-learning
splice_sites.tsv + base_predictions.tsv → meta_features.tsv → Train Meta-Model
```

---

## 9. Summary Table

| Feature | OpenSpliceAI `create_datafile.py` | MetaSpliceAI `derive.py` |
|---------|----------------------------------|---------------------------|
| **Purpose** | Base model training data | Meta-learning annotations |
| **Output** | HDF5 (dense arrays) | TSV (sparse table) |
| **Granularity** | Per-gene sequences | Per-site rows |
| **Coordinates** | Gene-relative, 0-based | Genomic absolute, 1-based |
| **Strand** | Reverse complement | Preserved |
| **Transcripts** | Longest only | All transcripts |
| **Size** | ~1-2 GB | ~60-400 MB |
| **Query Speed** | Fast for gene access | Fast for region/site access |
| **Extensibility** | Regenerate for updates | Append for updates |
| **Integration** | Training pipeline | Analysis pipeline |
| **Use Case** | Train base predictor | Meta-learning & variant analysis |

---

## 10. Conclusion

**OpenSpliceAI's `create_datafile.py`** and **MetaSpliceAI's splice site generation** are **complementary tools** designed for different stages of the splice prediction pipeline:

- **OpenSpliceAI**: Creates training data for **base models**
- **MetaSpliceAI**: Creates annotations for **meta-learning**

Both start from GTF + FASTA, but produce fundamentally different outputs optimized for their specific use cases. Understanding these differences is critical for:

1. **Correct coordinate reconciliation** when integrating predictions
2. **Choosing the right approach** for your specific task
3. **Avoiding confusion** about coordinate system differences

In the MetaSpliceAI system, we primarily use our own approach because we focus on **meta-learning on top of pre-trained base models**, not training base models from scratch.

---

**Document Version**: 1.0  
**Last Updated**: October 15, 2025  
**Related**: `BASE_MODEL_SPLICE_SITE_DEFINITIONS.md`, `coordinate_reconciliation.py`

