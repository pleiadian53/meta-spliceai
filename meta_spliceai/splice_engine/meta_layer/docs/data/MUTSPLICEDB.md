# MutSpliceDB Data Integration

**Source**: https://brb.nci.nih.gov/splicing/  
**Documentation**: https://brb.nci.nih.gov/splicing/documentation.html  
**Publication**: PMID 33600011  
**Local Data**: `data/mutsplicedb/` (gitignored)

---

## Overview

MutSpliceDB is a database developed by NCI's Biometric Research Program that documents 
the effects of splice site mutations on RNA splicing, using RNA-seq evidence from TCGA 
and CCLE samples.

We use MutSpliceDB to derive **precise aberrant splice site coordinates** for training 
position localization models.

---

## How to Download

MutSpliceDB does not have a public API. Data must be exported manually:

1. Go to: https://brb.nci.nih.gov/splicing/
2. Click "**Access MutSpliceDB**"
3. Use the export buttons (top right of table) to download CSV or Excel
4. Save as: `data/mutsplicedb/mutsplicedb_export.csv`

---

## Data Fields

Based on MutSpliceDB documentation:

| Field | Description | Example |
|-------|-------------|---------|
| Gene Symbol | Gene name | TP53 |
| Entrez Gene ID | NCBI Gene ID | 7157 |
| HGVS Notation | Transcript-based variant | NM_000546.5:c.375+5G>A |
| Allele Registry ID | ClinGen ID | CA388161720 |
| Splicing Effect | Description of outcome | "Exon 4 skipping" |
| Sample Name | Sample identifier | TCGA-LUAD-xxxxx |
| Sample Source | Data source | TCGA, CCLE |
| BAM File | RNA-seq evidence | Link to BAM |

---

## Processing Pipeline

After downloading, run:

```bash
python scripts/data_processing/parse_mutsplicedb.py \
    --input data/mutsplicedb/mutsplicedb_export.csv \
    --output data/mutsplicedb/splice_sites_induced.tsv \
    --gtf data/mane/GRCh38/MANE.GRCh38.v1.3.ensembl_genomic.gtf
```

### What the Parser Does

1. **Loads GTF** to get exon coordinates per transcript
2. **Parses effect descriptions** (e.g., "Exon 4 skipping")
3. **Derives junction coordinates**:
   - Exon skipping → Novel junction between flanking exons
   - Intron retention → Canonical sites marked as "lost"
   - Cryptic activation → Flagged for manual review

### Output Schema

```tsv
chrom	position	strand	site_type	inducing_variant	effect_type	gene	...
chr17	7673456	-	acceptor_novel_junction	NM_000546.5:c.375+5G>A	exon_skipping	TP53	...
```

---

## Effect Parsing Examples

| Effect Description | Parsed Type | Derived Sites |
|-------------------|-------------|---------------|
| "Exon 4 skipping" | exon_skipping | Novel junction: exon3_end → exon5_start |
| "Exon 3-5 skipping" | exon_skipping | Novel junction: exon2_end → exon6_start |
| "Intron 5 retention" | intron_retention | Canonical donor/acceptor marked as lost |
| "Cryptic donor activation" | cryptic_donor | Flagged - needs position from BAM |

---

## Integration with Aberrant Site Annotations

The output `splice_sites_induced.tsv` feeds into:
- Position localization training (Multi-Step Step 3)
- Direct supervision without base model dependency
- Validation of base model delta predictions

See: `meta_spliceai/splice_engine/meta_layer/docs/wishlist/ABERRANT_SPLICE_SITE_ANNOTATIONS.md`

---

## Files and Locations

| File | Location | Description |
|------|----------|-------------|
| Parser script | `scripts/data_processing/parse_mutsplicedb.py` | Converts export to coordinates |
| Raw export | `data/mutsplicedb/mutsplicedb_export.csv` | Manual download (gitignored) |
| Parsed output | `data/mutsplicedb/splice_sites_induced.tsv` | Final induced sites (gitignored) |
| This doc | `docs/data/MUTSPLICEDB.md` | Documentation |

---

## Contact

For bulk data access or API questions, contact:
- **Dr. Dmitriy Sonkin**: dmitriy.sonkin@nih.gov

