# Splice Site Analysis in the Human Genome

This document summarizes key findings from analyzing alternative splicing patterns across different gene types in the human genome using the MetaSpliceAI enhanced annotation utilities.

## Overview

The analysis examines splice site annotations enriched with gene and transcript features to understand:
1. Distribution of splice sites across gene types
2. Alternative splicing patterns
3. Genes with exceptional alternative splicing complexity

## Analysis Methods

The analysis was performed using the `enhance_splice_sites_with_features` utility, which merges splice site annotations with gene and transcript information from the Ensembl database. The utility enables filtering of splice sites by gene type, making it possible to analyze splicing patterns across different gene biotypes.

## Key Findings

### Protein-Coding Genes

Protein-coding genes show extensive splicing complexity:

```
Total genes with splice sites: 19,087
Average transcripts per gene: 8.86
Genes with alternative splicing: 16,883 (88.5%)
```

**Splice Site Distribution:**
| Site Type | Count    |
|-----------|----------|
| Acceptor  | 1,235,403|
| Donor     | 1,235,403|

**Top 10 Alternatively Spliced Protein-Coding Genes:**

| Gene ID         | Gene Name | Transcript Count |
|-----------------|-----------|------------------|
| ENSG00000109339 | MAPK10    | 192              |
| ENSG00000115392 | FANCL     | 156              |
| ENSG00000145362 | ANK2      | 129              |
| ENSG00000107862 | GBF1      | 118              |
| ENSG00000125124 | BBS2      | 109              |
| ENSG00000068400 | GRIPAP1   | 107              |
| ENSG00000163913 | IFT122    | 107              |
| ENSG00000006071 | ABCC8     | 100              |
| ENSG00000185359 | HGS       | 99               |
| ENSG00000127990 | SGCE      | 98               |

### Long Non-coding RNAs (lncRNAs)

lncRNAs also exhibit significant alternative splicing, though to a lesser degree than protein-coding genes:

```
Total lncRNA genes with splice sites: 16,055
Average transcripts per lncRNA gene: 3.40
```

When combined with protein-coding genes for analysis:
```
Total genes with splice sites (protein-coding + lncRNA): 35,142
Total genes with alternative splicing: 23,231 (66.1%)
```

**Splice Site Distribution in lncRNAs:**
| Site Type | Count   |
|-----------|---------|
| Acceptor  | 165,376 |
| Donor     | 165,376 |

### Top 10 Most Alternatively Spliced Genes (All Types)

When analyzing both protein-coding and lncRNA genes together, lncRNAs surprisingly dominate the top positions for alternative transcript complexity:

| Gene ID         | Gene Name  | Gene Type      | Transcript Count |
|-----------------|------------|----------------|------------------|
| ENSG00000179818 | PCBP1-AS1  | lncRNA         | 295              |
| ENSG00000215386 | MIR99AHG   | lncRNA         | 257              |
| ENSG00000109339 | MAPK10     | protein_coding | 192              |
| ENSG00000249859 | PVT1       | lncRNA         | 190              |
| ENSG00000226674 | TEX41      | lncRNA         | 189              |
| ENSG00000241469 | LINC00635  | lncRNA         | 164              |
| ENSG00000115392 | FANCL      | protein_coding | 156              |
| ENSG00000227195 | MIR663AHG  | lncRNA         | 145              |
| ENSG00000242086 | MUC20-OT1  | lncRNA         | 142              |
| ENSG00000229140 | CCDC26     | lncRNA         | 141              |

## Implications for SpliceAI Analysis

These findings have important implications for SpliceAI analysis:

1. **Coverage considerations**: SpliceAI traditionally focuses on protein-coding genes (19,087 genes), but this represents only ~54% of genes with splice sites when including lncRNAs.

2. **Alternative splicing complexity**: Protein-coding genes have on average 8.86 transcripts per gene compared to 3.40 for lncRNAs, indicating greater splicing complexity that SpliceAI needs to model.

3. **Specialized genes**: Some genes have extraordinary alternative splicing complexity (>100 transcripts), which may require special attention in splice site prediction models.

4. **lncRNA splicing patterns**: lncRNAs also demonstrate significant alternative splicing, with 8 of the top 10 most alternatively spliced genes being lncRNAs rather than protein-coding genes.

## Usage in MetaSpliceAI

The enhanced splice site annotations can be filtered for specific gene types using the `enhance_splice_sites_with_features` utility:

```python
from meta_spliceai.splice_engine.meta_models.utils import enhance_splice_sites_with_features

# Get protein-coding splice sites only
protein_coding_sites = enhance_splice_sites_with_features(
    splice_sites_path="data/ensembl/splice_sites.tsv",
    gene_features_path="data/ensembl/spliceai_analysis/gene_features.tsv",
    gene_types_to_keep=["protein_coding"]
)

# Get multiple gene types
coding_and_noncoding_sites = enhance_splice_sites_with_features(
    splice_sites_path="data/ensembl/splice_sites.tsv",
    gene_features_path="data/ensembl/spliceai_analysis/gene_features.tsv",
    gene_types_to_keep=["protein_coding", "lncRNA", "processed_pseudogene"]
)
```

This filtering capability is particularly useful for:
- Training models on specific gene subsets
- Evaluating model performance across different gene biotypes
- Creating tailored datasets for specialized analysis
