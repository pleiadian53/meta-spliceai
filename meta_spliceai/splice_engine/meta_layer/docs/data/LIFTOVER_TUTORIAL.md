# Liftover: Converting Coordinates Between Genome Builds

**Purpose**: Convert genomic coordinates between different genome assemblies (e.g., hg19 → hg38)

---

## Why Liftover is Needed

### The Problem

Different genomic datasets use different genome assembly versions:

| Assembly | UCSC Name | Year | Notes |
|----------|-----------|------|-------|
| GRCh37 | hg19 | 2009 | Legacy, still common in older datasets |
| GRCh38 | hg38 | 2013 | Current standard, used by OpenSpliceAI |

**Coordinates differ between builds** because:
- Contigs were reordered/merged
- Gaps were filled with new sequences
- Errors were corrected
- New alternative haplotypes added

### Example

The same variant can have different coordinates:

| Build | TP53 exon 4 start |
|-------|-------------------|
| hg19 | chr17:7578371 |
| hg38 | chr17:7675053 |

**Difference: ~96,682 bp shift!**

---

## How Liftover Works

### Chain Files

UCSC provides "chain files" that map regions between assemblies:

```
chain 4900 chr17 81195210 + 7571719 7590856 chr17 83257441 + 7668401 7687538 1
...
```

Each chain describes:
- Source region (old build)
- Target region (new build)
- Alignment between them

### The Algorithm

1. **Find** which chain covers your coordinate
2. **Map** through the alignment
3. **Return** new coordinate (or None if unmappable)

---

## Using pyliftover in Python

### Installation

```bash
pip install pyliftover
```

### Basic Usage

```python
from pyliftover import LiftOver

# Initialize converter (downloads chain file on first use)
lo = LiftOver('hg19', 'hg38')

# Convert a single position
result = lo.convert_coordinate('chr17', 7578371)

if result:
    new_chrom, new_pos, new_strand, chain_score = result[0]
    print(f"hg19 chr17:7578371 → hg38 {new_chrom}:{new_pos}")
    # Output: hg19 chr17:7578371 → hg38 chr17:7675053
else:
    print("Position could not be lifted over")
```

### Batch Conversion

```python
import pandas as pd
from pyliftover import LiftOver

lo = LiftOver('hg19', 'hg38')

def liftover_position(chrom, pos):
    """Convert a single position, return None if failed."""
    # Ensure chr prefix
    if not chrom.startswith('chr'):
        chrom = 'chr' + chrom
    
    result = lo.convert_coordinate(chrom, pos)
    
    if result and len(result) > 0:
        return result[0][1]  # Return new position
    return None

# Apply to DataFrame
df['position_hg38'] = df.apply(
    lambda row: liftover_position(row['chrom'], row['position']),
    axis=1
)

# Filter failed conversions
successful = df[df['position_hg38'].notna()]
failed = df[df['position_hg38'].isna()]

print(f"Converted: {len(successful)}, Failed: {len(failed)}")
```

---

## Available Conversions

### Common Chain Files

| From | To | Chain File |
|------|-----|------------|
| hg19 | hg38 | hg19ToHg38.over.chain.gz |
| hg38 | hg19 | hg38ToHg19.over.chain.gz |
| hg18 | hg38 | hg18ToHg38.over.chain.gz |
| mm9 | mm10 | mm9ToMm10.over.chain.gz |

pyliftover auto-downloads from UCSC on first use.

### Other Species

```python
# Mouse
lo_mouse = LiftOver('mm9', 'mm10')

# Requires manual chain file for non-UCSC assemblies
lo_custom = LiftOver('/path/to/custom.chain')
```

---

## When Liftover Fails

### Common Failure Reasons

1. **Position in deleted region**: The sequence was removed in the new build
2. **Position in new sequence**: Region was duplicated or rearranged
3. **Position in gap**: Original position was in an assembly gap
4. **Centromere/telomere regions**: Often poorly assembled

### Failure Rate

For human hg19 → hg38:
- **Exonic regions**: <0.1% failure rate
- **Intronic regions**: ~0.5% failure rate
- **Intergenic**: ~1-2% failure rate
- **Repetitive elements**: 5-10% failure rate

### Handling Failures

```python
def safe_liftover(chrom, pos, liftover_obj, fallback='drop'):
    """
    Safe liftover with multiple strategies.
    
    Parameters
    ----------
    fallback : str
        'drop' - Return None for failed conversions
        'keep' - Keep original coordinate (risky!)
        'nearest' - Try nearby positions
    """
    result = liftover_obj.convert_coordinate(chrom, pos)
    
    if result and len(result) > 0:
        return result[0][1]
    
    if fallback == 'drop':
        return None
    elif fallback == 'keep':
        return pos  # Risky: coordinates will be wrong!
    elif fallback == 'nearest':
        # Try positions within ±10bp
        for offset in range(1, 11):
            for delta in [offset, -offset]:
                result = liftover_obj.convert_coordinate(chrom, pos + delta)
                if result:
                    return result[0][1]
        return None
    
    return None
```

---

## Use in MutSpliceDB Parser

Our parser supports automatic liftover:

```bash
python scripts/data_processing/parse_mutsplicedb.py \
    --input data/mutsplicedb/MutSpliceDB_BRP_2025-12-18.csv \
    --output data/mutsplicedb/splice_sites_induced.tsv \
    --gtf data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf \
    --liftover \
    --verbose
```

This:
1. Processes both hg38 and hg19 entries
2. Converts hg19 coordinates to hg38
3. Marks converted entries: `genome hg19→hg38 (lifted)`
4. Marks failures: `[LIFTOVER_FAILED]` with confidence=low

---

## Alternative Tools

### Command-Line: UCSC liftOver

```bash
# Download chain file
wget https://hgdownload.cse.ucsc.edu/goldenpath/hg19/liftOver/hg19ToHg38.over.chain.gz

# Convert BED file
liftOver input.bed hg19ToHg38.over.chain.gz output.bed unmapped.bed
```

### CrossMap (Python CLI)

```bash
pip install CrossMap

CrossMap.py bed hg19ToHg38.over.chain.gz input.bed output.bed
```

### Web Interface

UCSC LiftOver: https://genome.ucsc.edu/cgi-bin/hgLiftOver

---

## Best Practices

### 1. Always Track Original Coordinates

```python
df['original_chrom'] = df['chrom']
df['original_pos'] = df['position']
df['original_build'] = 'hg19'

# Then liftover
df['position'] = df.apply(liftover_func, axis=1)
df['build'] = 'hg38'
```

### 2. Validate After Liftover

```python
# Check that lifted positions are within expected gene regions
def validate_lifted_position(row, gene_coords):
    gene = gene_coords.get(row['gene'])
    if gene:
        if gene['start'] <= row['position'] <= gene['end']:
            return True
    return False
```

### 3. Report Failure Statistics

```python
total = len(df)
failed = df['position_lifted'].isna().sum()

print(f"Liftover success rate: {(total-failed)/total*100:.1f}%")
print(f"Failed conversions: {failed}")
```

### 4. Use Appropriate Confidence Levels

| Source | Confidence |
|--------|------------|
| Native hg38 | high |
| Lifted hg19→hg38 | medium |
| Lifted with nearby fallback | low |
| Failed liftover | exclude or very low |

---

## References

- UCSC LiftOver: https://genome.ucsc.edu/cgi-bin/hgLiftOver
- pyliftover: https://github.com/konstantint/pyliftover
- CrossMap: http://crossmap.sourceforge.net/
- Chain file format: https://genome.ucsc.edu/goldenPath/help/chain.html

---

## See Also

- [MUTSPLICEDB.md](MUTSPLICEDB.md) - Using liftover with MutSpliceDB parser
- [SPLICEVARDB.md](SPLICEVARDB.md) - SpliceVarDB dataset (hg38 native)

