# Aberrant Splice Site Annotations: Augmenting Canonical Annotations

**Purpose**: Roadmap for deriving and integrating variant-induced splice site positions  
**Status**: 🔬 Research & Data Collection Phase  
**Priority**: ⭐⭐⭐ HIGH - Would fundamentally improve training signal quality  
**Created**: December 2025

---

## Executive Summary

### The Problem

Current training approaches have a fundamental limitation:

```
Current: Variant → Base Model → Delta Score → Train Meta Model
                      ↑
              Circular dependency: Meta model learns what base model predicts
```

We lack **ground truth aberrant splice site positions** - the precise genomic coordinates where variants induce cryptic/alternative splice sites.

### The Solution

Augment `splice_sites_enhanced.tsv` with a new file:

```
splice_sites_induced.tsv
├── Variant-induced cryptic donors
├── Variant-induced cryptic acceptors  
├── Exon skipping junction coordinates
├── Intron retention boundaries
└── Evidence source and confidence
```

This enables:
```
Improved: Variant → Ground Truth Positions → Train Meta Model (no base model dependency)
```

---

## Why This Matters

### Current Approach Limitations

| Limitation | Impact | How Aberrant Annotations Help |
|------------|--------|-------------------------------|
| Base model quality | Meta model learns base model errors | Direct ground truth, no propagation |
| No position specificity | Only know "splice-altering", not WHERE | Precise genomic coordinates |
| Cryptic sites missed | Base model may not detect novel sites | RNA-seq validated positions |
| Circular training | Model may just memorize base model | Independent supervision signal |

### What Precise Positions Enable

1. **Position Localization Training** (Multi-Step Step 3)
   - Current: Derive from base model delta (indirect)
   - Improved: Direct coordinate supervision

2. **Binary Mask Targets**
   - Current: Gaussian smoothing around estimated positions
   - Improved: Exact ±2bp around validated sites

3. **Novel Junction Detection**
   - Current: Not possible without ground truth
   - Improved: Train to predict junction coordinates

---

## Data Sources for Aberrant Splice Sites

### 1. MutSpliceDB (NCI) ⭐ PRIMARY

**URL**: https://brb.nci.nih.gov/splicing/  
**Documentation**: https://brb.nci.nih.gov/splicing/documentation.html  
**Publication**: PMID 33600011

| Attribute | Details |
|-----------|---------|
| Coverage | TCGA + CCLE samples |
| Evidence | Manual review of RNA-seq BAM files |
| Data Fields | Gene, HGVS, Allele Registry ID, Splicing Effect |
| Export | CSV/Excel via web interface |
| API | ❌ No public API |
| Contact | dmitriy.sonkin@nih.gov |

**What MutSpliceDB Provides:**
```
Gene: TP53
Variant: NM_000546.5:c.375+5G>A
Effect: "Exon 4 skipping confirmed"
Sample: TCGA-LUAD-xxxxx
Evidence: IGV screenshot + mini BAM
```

**What We Need to Derive:**
```
From "Exon 4 skipping" → Junction coordinates:
  - Canonical: Exon 3 end (chr17:7,675,123) — Exon 4 start (chr17:7,674,890)
  - Aberrant: Exon 3 end (chr17:7,675,123) — Exon 5 start (chr17:7,673,456)
  
This gives us the cryptic junction: chr17:7,675,123-7,673,456
```

**Data Extraction Pipeline:**
```python
# Conceptual pipeline to extract coordinates from MutSpliceDB
import pandas as pd
from meta_spliceai.system.genomic_resources import Registry

def extract_junction_from_effect(effect_description: str, gene: str, gtf):
    """
    Parse effect description to derive junction coordinates.
    
    Effect types:
    - "Exon X skipping" → Novel junction skipping exon X
    - "Intron retention" → Intron included in transcript
    - "Cryptic donor activation at position Y" → New donor site
    - "Cryptic acceptor activation at position Y" → New acceptor site
    """
    if "exon" in effect_description.lower() and "skipping" in effect_description.lower():
        # Extract skipped exon number
        exon_num = parse_exon_number(effect_description)
        exons = get_exons_for_gene(gene, gtf)
        
        # Novel junction: exon before skipped → exon after skipped
        novel_donor = exons[exon_num - 1].end
        novel_acceptor = exons[exon_num + 1].start
        
        return {
            'type': 'exon_skipping',
            'novel_donor_pos': novel_donor,
            'novel_acceptor_pos': novel_acceptor,
            'skipped_exon': exon_num
        }
    
    # ... handle other effect types
```

---

### 2. Snaptron (JHU) ⭐ SECONDARY

**URL**: http://snaptron.cs.jhu.edu/  
**Publication**: PMID 27655932

| Attribute | Details |
|-----------|---------|
| Coverage | >400 million junctions from >50,000 samples |
| Data | GTEx, TCGA, SRAv2 compilations |
| Query | REST API + bulk download |
| Format | Junction coordinates with sample counts |

**What Snaptron Provides:**
```
Junction: chr17:7675123-7673456
Strand: -
Samples: 5
Coverage: 12,8,3,15,7
Annotation: novel
```

**API Query Example:**
```bash
# Query junctions in a region
curl "http://snaptron.cs.jhu.edu/gtex/snaptron?regions=chr17:7673000-7676000"

# Response includes:
# chrom, start, end, strand, samples, coverage, annotation_status
```

**Use Case**: Find rare junctions near known splice-altering variants
```python
def find_rare_junctions_near_variant(chrom, pos, window=1000, max_samples=10):
    """
    Query Snaptron for rare junctions near a variant position.
    
    Rare junctions (few samples) near splice-altering variants
    are likely induced by those variants.
    """
    region = f"{chrom}:{pos-window}-{pos+window}"
    response = requests.get(
        f"http://snaptron.cs.jhu.edu/gtex/snaptron?regions={region}"
    )
    
    junctions = parse_snaptron_response(response.text)
    
    # Filter to rare junctions (not in most samples)
    rare = [j for j in junctions if j['sample_count'] <= max_samples]
    
    return rare
```

---

### 3. GTEx sQTLs ⭐ SECONDARY

**URL**: https://gtexportal.org/  
**Data**: Splice quantitative trait loci

| Attribute | Details |
|-----------|---------|
| Coverage | 54 tissues, ~1,000 donors |
| Data Type | Variant → Splicing effect associations |
| Format | BED + association statistics |
| Download | Bulk download available |

**What sQTLs Provide:**
```
Variant: rs12345 (chr17:7675000:G:A)
Phenotype: Intron excision ratio at chr17:7674890-7675123
Tissue: Lung
Effect Size: -0.35 (reduced canonical splicing)
P-value: 1.2e-15
```

**Use Case**: Variants with significant sQTL effects have validated splicing impact
```python
def get_sqtl_affected_junctions(variant_id):
    """
    Get junctions affected by a variant from GTEx sQTL data.
    
    sQTL phenotypes are typically:
    - Intron excision ratios (LeafCutter)
    - Percent spliced in (PSI)
    - Junction read counts
    """
    sqtl_data = load_gtex_sqtl(variant_id)
    
    affected_junctions = []
    for hit in sqtl_data:
        junction = parse_phenotype_id(hit['phenotype_id'])
        affected_junctions.append({
            'junction': junction,
            'effect_size': hit['slope'],
            'tissue': hit['tissue'],
            'pvalue': hit['pvalue']
        })
    
    return affected_junctions
```

---

### 4. DBASS3/DBASS5 (Legacy)

**URL**: http://www.dbass.org.uk/ (may be outdated)  
**Publication**: PMID 17576680

| Attribute | Details |
|-----------|---------|
| Coverage | Curated aberrant 3' and 5' splice sites |
| Data | Disease-causing splice mutations |
| Format | Web database |
| Status | ⚠️ May not be actively maintained |

---

### 5. Long-Read Transcriptome Data (Emerging)

**Sources**:
- LRGASP (Long-Read Genome Annotation and Sequencing Project)
- PacBio IsoSeq datasets
- Oxford Nanopore direct RNA-seq

| Attribute | Details |
|-----------|---------|
| Coverage | Limited but growing |
| Advantage | Full-length transcript isoforms |
| Format | GTF with novel isoforms |
| Challenge | Need matched variant data |

---

## Proposed Data Schema

### `splice_sites_induced.tsv`

```tsv
# Variant-Induced Aberrant Splice Sites
# Format: TSV with header
# Coordinates: 1-based, GRCh38

chrom	position	strand	site_type	inducing_variant	effect_type	gene	evidence_source	evidence_samples	confidence
chr17	7673456	-	acceptor_cryptic	chr17:7675000:G:A	exon_skipping	TP53	MutSpliceDB	TCGA-LUAD-001,TCGA-LUAD-002	high
chr17	7675123	-	donor_canonical_lost	chr17:7675000:G:A	exon_skipping	TP53	MutSpliceDB	TCGA-LUAD-001,TCGA-LUAD-002	high
chr7	117559650	+	donor_cryptic	chr7:117559590:G:A	cryptic_activation	CFTR	Snaptron+SpliceVarDB	GTEx_Lung_x3	medium
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `chrom` | string | Chromosome (chr1-chr22, chrX, chrY) |
| `position` | int | 1-based genomic position |
| `strand` | char | + or - |
| `site_type` | enum | `donor_cryptic`, `donor_canonical_lost`, `acceptor_cryptic`, `acceptor_canonical_lost` |
| `inducing_variant` | string | Variant that induces this site (chrom:pos:ref:alt) |
| `effect_type` | enum | `exon_skipping`, `intron_retention`, `cryptic_activation`, `canonical_disruption` |
| `gene` | string | Gene symbol |
| `evidence_source` | string | Database/study providing evidence |
| `evidence_samples` | string | Comma-separated sample IDs |
| `confidence` | enum | `high` (RNA-seq validated), `medium` (computational), `low` (inferred) |

---

## Implementation Roadmap

### Phase 1: Data Collection (2-4 weeks)

```
□ Download MutSpliceDB entries (CSV export)
□ Parse effect descriptions to extract affected exons
□ Map to GTF coordinates to get junction positions
□ Cross-reference with Snaptron for junction validation
□ Create initial splice_sites_induced.tsv (~1,000 entries)
```

### Phase 2: Data Integration (1-2 weeks)

```
□ Create SpliceSiteInducedLoader class
□ Integrate with existing genomic_resources.Registry
□ Add coordinate conversion (if needed for GRCh37)
□ Validate coordinates against reference genome
```

### Phase 3: Model Training (2-4 weeks)

```
□ Update PositionLocalizer to use induced sites as targets
□ Create binary mask targets from precise coordinates
□ Compare model performance: delta-derived vs induced-site targets
□ Evaluate on held-out variants
```

### Phase 4: Expansion (Ongoing)

```
□ Process GTEx sQTL data for additional sites
□ Incorporate long-read data as available
□ Build automated pipeline to update annotations
□ Consider contributing back to public databases
```

---

## Integration with Existing Code

### Current Splice Site Annotation

```python
# Current: data/mane/GRCh38/splice_sites_enhanced.tsv
# Contains: Canonical splice sites from GTF + some common alternatives

from meta_spliceai.system.genomic_resources import Registry

registry = Registry(build='GRCh38')
splice_sites = registry.get_splice_sites()  # Returns DataFrame
```

### Proposed Extension

```python
# New: data/mane/GRCh38/splice_sites_induced.tsv
# Contains: Variant-induced aberrant splice sites

from meta_spliceai.system.genomic_resources import Registry

registry = Registry(build='GRCh38')

# Get all splice sites (canonical + common alternative)
canonical_sites = registry.get_splice_sites()

# Get variant-induced sites (new)
induced_sites = registry.get_induced_splice_sites()

# Filter by inducing variant
variant_sites = induced_sites.query("inducing_variant == 'chr7:117559590:G:A'")

# Get all sites for a region
all_sites = registry.get_all_splice_sites(
    chrom='chr7',
    start=117559000,
    end=117560000,
    include_induced=True
)
```

---

## Expected Impact

### Model Training Improvements

| Metric | Current (Delta-Based) | Expected (Induced Sites) | Improvement |
|--------|----------------------|--------------------------|-------------|
| Position Accuracy | ~65% within 10bp | ~85% within 5bp | +20% |
| Cryptic Detection | Limited | Significantly better | TBD |
| Generalization | Base model dependent | Independent | Better on novel variants |

### Research Contributions

1. **Novel Dataset**: First comprehensive induced splice site annotation
2. **Better Benchmarking**: Ground truth for position localization
3. **Clinical Utility**: Direct mapping variant → affected sites

---

## Open Questions

1. **Coverage**: How many variants have RNA-seq evidence?
   - Estimate: ~5,000-10,000 from MutSpliceDB + Snaptron

2. **Precision**: Can we reliably derive coordinates from effect descriptions?
   - Need: Robust parsing + GTF coordinate mapping

3. **Generalization**: Will models trained on observed sites generalize?
   - Hypothesis: Yes, if we capture diverse effect types

4. **Maintenance**: How to keep annotations updated?
   - Proposal: Quarterly updates from data sources

---

## Contact and Resources

### Data Source Contacts

| Resource | Contact | Purpose |
|----------|---------|---------|
| MutSpliceDB | dmitriy.sonkin@nih.gov | Bulk data access |
| Snaptron | Ben Langmead (JHU) | API questions |
| GTEx | gtex-help@broadinstitute.org | sQTL data |

### Internal Resources

- `meta_spliceai/splice_engine/meta_layer/data/induced_splice_loader.py` (to be created)
- `data/mane/GRCh38/splice_sites_induced.tsv` (to be created)
- `docs/data/INDUCED_SPLICE_SITES.md` (detailed data documentation)

---

## See Also

- [SPLICEVARDB.md](../data/SPLICEVARDB.md) - Current variant data source
- [MULTI_STEP_FRAMEWORK.md](../methods/MULTI_STEP_FRAMEWORK.md) - Position localization
- [HGVS_TUTORIAL.md](../data/HGVS_TUTORIAL.md) - Understanding variant notation
- [VALIDATED_DELTA_PREDICTION.md](../methods/VALIDATED_DELTA_PREDICTION.md) - Current approach

