# Genomic Build and Coordinate Handling Guide
**Package:** openspliceai_recalibration  
**Critical Topic:** Ensuring coordinate accuracy across different genomic builds

## 🎯 The Problem

SpliceVarDB provides coordinates in **both hg19 (GRCh37) and hg38 (GRCh38)**. Using the wrong coordinates with OpenSpliceAI will cause:

1. **Incorrect sequence extraction** from reference genome
2. **Wrong splice site predictions** (coordinates don't match features)
3. **Invalid alternative splice site annotations**
4. **Silent failures** (predictions run but are meaningless)

## 📊 SpliceVarDB Data Format

Your downloaded file (`splicevardb.download.tsv`) has **both** builds:

```tsv
variant_id  hg19                hg38                gene    hgvs
1           1-100573238-T-C     1-100107682-T-C     SASS6   NM_194292.3:c.1092A>G
2           1-100576040-C-A     1-100110484-C-A     SASS6   NM_194292.3:c.670-1G>T
```

**Note the coordinate differences:**
- Same variant has different positions in hg19 vs hg38
- Example: `100573238` (hg19) → `100107682` (hg38)
- Difference: ~465kb shift due to genome assembly updates

## ✅ Correct Build Selection

### For OpenSpliceAI → Use hg38

```python
# OpenSpliceAI was trained on GRCh38 (MANE)
config = PipelineConfig(
    genome_build="GRCh38",
    coordinate_column="hg38",  # ← Use this column
    reference_genome="data/mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna",
    base_model="openspliceai"
)
```

### For SpliceAI → Use hg19

```python
# SpliceAI was trained on GRCh37
config = PipelineConfig(
    genome_build="GRCh37",
    coordinate_column="hg19",  # ← Use this column
    reference_genome="data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa",
    base_model="spliceai"
)
```

## 🔄 Coordinate Parsing

**Format in SpliceVarDB:** `"CHROM-POS-REF-ALT"` (quoted string)

**Example:** `"1-100107682-T-C"`

### Parsing Code

```python
def parse_variant(variant_str):
    """
    Parse variant string from SpliceVarDB.
    
    Args:
        variant_str: String like '"1-100107682-T-C"'
    
    Returns:
        dict with chrom, pos, ref, alt
    """
    # Remove quotes
    clean = variant_str.strip('"')
    
    # Split on hyphens
    parts = clean.split('-')
    
    return {
        'chrom': parts[0],          # "1", "2", ..., "X", "Y"
        'pos': int(parts[1]),        # Genomic position (1-based)
        'ref': parts[2],             # Reference allele
        'alt': parts[3]              # Alternative allele
    }

# Usage
import pandas as pd

variants = pd.read_csv('splicevardb_download.tsv', sep='\t')

# For OpenSpliceAI: parse hg38
variants['parsed_coords'] = variants['hg38'].apply(parse_variant)
variants[['chrom', 'pos', 'ref', 'alt']] = pd.DataFrame(
    variants['parsed_coords'].tolist(), index=variants.index
)

print(f"Using GRCh38 coordinates for {len(variants)} variants")
```

## ⚠️ Coordinate Validation

**Always validate** that coordinates match your reference genome:

### 1. Chromosome Name Compatibility

**GRCh38 (MANE) uses RefSeq naming:**
- SpliceVarDB: `"1"`, `"2"`, ..., `"X"`
- MANE FASTA: `NC_000001.11`, `NC_000002.12`, ..., `NC_000023.11` (X)

**Need mapping:**
```python
# Chromosome name mapping for GRCh38 MANE
CHROM_MAP_GRCH38 = {
    '1': 'NC_000001.11', '2': 'NC_000002.12', '3': 'NC_000003.12',
    '4': 'NC_000004.12', '5': 'NC_000005.10', '6': 'NC_000006.12',
    '7': 'NC_000007.14', '8': 'NC_000008.11', '9': 'NC_000009.12',
    '10': 'NC_000010.11', '11': 'NC_000011.10', '12': 'NC_000012.12',
    '13': 'NC_000013.11', '14': 'NC_000014.9', '15': 'NC_000015.10',
    '16': 'NC_000016.10', '17': 'NC_000017.11', '18': 'NC_000018.10',
    '19': 'NC_000019.10', '20': 'NC_000020.11', '21': 'NC_000021.9',
    '22': 'NC_000022.11', 'X': 'NC_000023.11', 'Y': 'NC_000024.10'
}

# Apply mapping
variants['chrom_refseq'] = variants['chrom'].map(CHROM_MAP_GRCH38)
```

### 2. Reference Allele Validation

**Verify reference allele matches genome:**

```python
from pysam import FastaFile

def validate_reference(chrom, pos, ref_allele, fasta_path):
    """
    Validate that reference allele matches genome.
    
    Args:
        chrom: Chromosome (e.g., 'NC_000001.11')
        pos: Position (1-based)
        ref_allele: Expected reference (e.g., 'T')
        fasta_path: Path to reference FASTA
    
    Returns:
        bool: True if matches
    """
    fasta = FastaFile(fasta_path)
    
    # Fetch sequence (pysam uses 0-based coordinates)
    genome_base = fasta.fetch(chrom, pos-1, pos).upper()
    
    matches = (genome_base == ref_allele)
    
    if not matches:
        print(f"⚠️  Mismatch at {chrom}:{pos}")
        print(f"   Expected: {ref_allele}")
        print(f"   Found:    {genome_base}")
    
    return matches

# Validate all variants
fasta_path = "data/mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
variants['ref_valid'] = variants.apply(
    lambda row: validate_reference(
        row['chrom_refseq'], row['pos'], row['ref'], fasta_path
    ),
    axis=1
)

mismatches = (~variants['ref_valid']).sum()
print(f"Validated {len(variants)} variants: {mismatches} mismatches")

if mismatches > 0:
    print("⚠️  WARNING: Reference mismatches detected!")
    print("   This may indicate wrong build or coordinate issues")
```

## 🎯 Alternative Splice Site Annotation

When predicting alternative splice sites with OpenSpliceAI:

### Workflow

1. **Load variant with hg38 coordinates**
   ```python
   variant_coord = parse_variant(row['hg38'])  # GRCh38
   ```

2. **Extract sequence from GRCh38 reference**
   ```python
   reference = "data/mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
   sequence = extract_sequence(variant_coord, reference, window=5000)
   ```

3. **Run OpenSpliceAI prediction**
   ```python
   # OpenSpliceAI internally uses GRCh38 coordinates
   predictions = openspliceai.predict(sequence)
   ```

4. **Annotate alternative splice sites in GRCh38**
   ```python
   # All output coordinates are in GRCh38
   alt_donor = {
       'chrom': variant_coord['chrom'],  # GRCh38
       'pos': predicted_donor_pos,        # GRCh38
       'build': 'GRCh38',
       'score': donor_score
   }
   ```

5. **Save with explicit build annotation**
   ```python
   results = pd.DataFrame({
       'variant_id': variant_id,
       'chrom_grch38': alt_donor['chrom'],
       'pos_grch38': alt_donor['pos'],
       'build': 'GRCh38',
       'alt_donor_score': alt_donor['score']
   })
   
   results.to_parquet('alternative_splice_sites_GRCh38.parquet')
   ```

## 📋 Checklist for Coordinate Accuracy

Before running your OpenSpliceAI experiments:

- [ ] Confirmed SpliceVarDB has both hg19 and hg38 columns
- [ ] Selected **hg38** column for OpenSpliceAI
- [ ] Reference genome is **GRCh38 MANE**: `data/mane/GRCh38/...`
- [ ] GTF/GFF is **MANE v1.3**: `MANE.GRCh38.v1.3.refseq_genomic.gff`
- [ ] Chromosome names mapped to RefSeq format (NC_000001.11, etc.)
- [ ] Reference alleles validated against genome
- [ ] All output coordinates labeled as **GRCh38**
- [ ] Configuration explicitly sets `genome_build="GRCh38"`
- [ ] Configuration explicitly sets `coordinate_column="hg38"`

## 🚨 Common Mistakes to Avoid

### ❌ Using hg19 with OpenSpliceAI
```python
# WRONG - Coordinate mismatch!
config = PipelineConfig(
    coordinate_column="hg19",  # ← GRCh37 coordinates
    reference_genome="data/mane/GRCh38/..."  # ← GRCh38 genome
)
# Result: Sequences extracted from wrong positions!
```

### ❌ Forgetting Chromosome Name Mapping
```python
# WRONG - Chromosome name mismatch!
chrom = "1"  # From SpliceVarDB
sequence = fasta.fetch(chrom, pos-1, pos)  # ← Won't find "1" in RefSeq FASTA
# Need: chrom = "NC_000001.11"
```

### ❌ Not Validating Reference Alleles
```python
# WRONG - Assuming coordinates are correct
predictions = openspliceai.predict(sequence)
# Should: Validate ref allele first to catch coordinate issues
```

### ❌ Mixing Builds in Output
```python
# WRONG - Ambiguous build in output
results = pd.DataFrame({
    'chrom': variant_coord['chrom'],
    'pos': predicted_pos,
    # No build annotation! Which genome?
})

# RIGHT - Explicit build annotation
results = pd.DataFrame({
    'chrom_grch38': variant_coord['chrom'],
    'pos_grch38': predicted_pos,
    'build': 'GRCh38'
})
```

## 📚 References

- **DATA_LAYOUT_MASTER_GUIDE.md** - Current data organization
- **Base Model Data Mapping** - Build-specific requirements
- **OpenSpliceAI Documentation** - GRCh38 MANE specifics
- **SpliceVarDB Paper** - Data format and build information

## 💡 Summary

**Key Principle:** Always match variant coordinates to the base model's training genome.

For OpenSpliceAI experiments with SpliceVarDB:
1. ✅ Use `hg38` column from SpliceVarDB
2. ✅ Use `data/mane/GRCh38/` reference files
3. ✅ Map chromosome names to RefSeq format
4. ✅ Validate reference alleles
5. ✅ Annotate all outputs with explicit build (GRCh38)

**Result:** Accurate alternative splice site predictions with correct genomic coordinates! 🎯





