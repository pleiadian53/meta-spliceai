# Position Coordinate Systems

**Module:** `meta_models/core/position_types.py`  
**Created:** December 2025  
**Purpose:** Explicit handling of position column semantics to prevent coordinate misinterpretation bugs

---

## The Problem

The `position` column appears in multiple contexts with **different semantics**:

| Context | Position Meaning | Example |
|---------|-----------------|---------|
| GTF/GFF annotations | Absolute genomic coordinates | chr17:41,196,312 |
| `splice_sites_enhanced.tsv` | Absolute genomic coordinates | 41196312 |
| `nucleotide_scores.tsv` | Strand-dependent relative position | 1 (5' end) |
| Meta-model training artifacts | Strand-dependent relative position | 1, 2, 3, ... |

**Bug discovered (Dec 2025):** The nucleotide_scores generation code in `splice_prediction_workflow.py` treated absolute positions as relative, then added `gene_start` again, doubling the coordinate values.

---

## Coordinate System Definitions

### ABSOLUTE Coordinates

Genomic coordinates from the reference assembly.

```
Characteristics:
- Independent of strand orientation
- Always increasing from lower to higher values on the reference
- Used in annotation files (GTF, GFF, BED)
- Example: BRCA1 on chr17 spans 41,196,312 - 41,277,500 (GRCh37)
```

### RELATIVE Coordinates

Strand-dependent positions within a gene, 1-indexed from the 5' end.

```
Characteristics:
- Position 1 = transcription start (5' end)
- Increases in transcription direction (5' → 3')
- Strand-dependent mapping to absolute coordinates
- Used in prediction outputs and training artifacts
```

### Strand-Dependent Mapping

```
POSITIVE STRAND (+):
  Transcription: 5' -----> 3'
  Genomic:       gene_start -----> gene_end
  
  Position 1 = gene_start (lowest coordinate)
  Position N = gene_end (highest coordinate)
  
  Formula: absolute = gene_start + relative - 1
           relative = absolute - gene_start + 1

NEGATIVE STRAND (-):
  Transcription: 5' -----> 3'
  Genomic:       gene_end <----- gene_start
  
  Position 1 = gene_end (highest coordinate)
  Position N = gene_start (lowest coordinate)
  
  Formula: absolute = gene_end - relative + 1
           relative = gene_end - absolute + 1
```

---

## API Reference

### PositionType Enum

```python
from meta_spliceai.splice_engine.meta_models.core.position_types import PositionType

PositionType.ABSOLUTE  # Genomic coordinates
PositionType.RELATIVE  # Strand-dependent gene positions
```

### Core Conversion Functions

#### `absolute_to_relative()`

Convert absolute genomic coordinate(s) to relative position(s).

```python
from meta_spliceai.splice_engine.meta_models.core.position_types import absolute_to_relative

# Single position
rel = absolute_to_relative(
    41277500,           # absolute position
    gene_start=41196312,
    gene_end=41277500,
    strand='-'
)
# Returns: 1 (5' end for negative strand)

# Batch conversion
positions = [41277500, 41277499, 41277498]
rel_batch = absolute_to_relative(positions, gene_start=41196312, gene_end=41277500, strand='-')
# Returns: [1, 2, 3]
```

#### `relative_to_absolute()`

Convert relative position(s) to absolute genomic coordinate(s).

```python
from meta_spliceai.splice_engine.meta_models.core.position_types import relative_to_absolute

# Single position
abs_pos = relative_to_absolute(
    1,                  # relative position
    gene_start=41196312,
    gene_end=41277500,
    strand='-'
)
# Returns: 41277500

# Batch conversion
positions = [1, 2, 3]
abs_batch = relative_to_absolute(positions, gene_start=41196312, gene_end=41277500, strand='-')
# Returns: [41277500, 41277499, 41277498]
```

### Helper Classes and Functions

#### `GeneCoordinates` Dataclass

```python
from meta_spliceai.splice_engine.meta_models.core.position_types import GeneCoordinates

coords = GeneCoordinates(
    gene_start=41196312,
    gene_end=41277500,
    strand='-',
    gene_id='ENSG00000012048'  # optional
)

# Properties
coords.length  # 81189 (gene length in nucleotides)
```

#### `validate_position_range()`

Validate that a position is within expected range.

```python
from meta_spliceai.splice_engine.meta_models.core.position_types import (
    validate_position_range, PositionType
)

# Returns True/False
is_valid = validate_position_range(
    position=50000,
    position_type=PositionType.RELATIVE,
    gene_start=41196312,
    gene_end=41277500
)

# With strict=True, raises ValueError for invalid positions
validate_position_range(position=100000, ..., strict=True)
# Raises: ValueError("Relative position 100000 outside range [1, 81189]")
```

#### `infer_position_type()`

Heuristic helper to detect coordinate type (for debugging/migration).

```python
from meta_spliceai.splice_engine.meta_models.core.position_types import infer_position_type

# Detect from position values
pos_type, confidence = infer_position_type(
    positions=[41196312, 41196313, 41196314],
    gene_start=41196312,
    gene_end=41277500
)
# Returns: (PositionType.ABSOLUTE, 1.0)

pos_type, confidence = infer_position_type(
    positions=[1, 2, 3, 4, 5],
    gene_start=41196312,
    gene_end=41277500
)
# Returns: (PositionType.RELATIVE, 1.0)
```

---

## Usage Guidelines

### When to Use ABSOLUTE Coordinates

- Reading/writing annotation files (GTF, GFF, BED)
- Cross-referencing with external databases (Ensembl, UCSC)
- Displaying genomic positions to users
- Joining with splice_sites_enhanced.tsv

### When to Use RELATIVE Coordinates

- Generating prediction outputs (nucleotide_scores.tsv)
- Creating training data for meta-models
- Position-based feature engineering
- Comparing positions across different genes

### Best Practices

```python
# ✅ DO: Explicitly label coordinate types in DataFrames
df = pl.DataFrame({
    'position': relative_positions,        # RELATIVE (1-indexed, 5' to 3')
    'genomic_position': absolute_positions  # ABSOLUTE genomic coordinate
})

# ✅ DO: Use conversion functions from position_types module
from meta_spliceai.splice_engine.meta_models.core.position_types import absolute_to_relative

relative = absolute_to_relative(absolute, gene_start, gene_end, strand)

# ❌ DON'T: Manually compute with unclear semantics
# This is error-prone and the bug that was fixed
position = gene_start + p - 1  # What is p? Absolute or relative?
```

---

## Real-World Example: BRCA1

BRCA1 is on the **negative strand** of chromosome 17.

```
Gene coordinates (GRCh37):
  gene_start = 41,196,312 (lower coordinate, but 3' end!)
  gene_end = 41,277,500 (higher coordinate, and 5' end!)
  strand = '-'
  length = 81,189 bp

Position mapping:
  Relative 1     → Absolute 41,277,500 (5' end, transcription start)
  Relative 2     → Absolute 41,277,499
  Relative 3     → Absolute 41,277,498
  ...
  Relative 81189 → Absolute 41,196,312 (3' end)

First exon starts at the 5' end:
  In relative coordinates: position ~1
  In absolute coordinates: position ~41,277,500
```

---

## Historical Bug Reference

### The Position Doubling Bug (Dec 2025)

**File:** `splice_prediction_workflow.py`

**Symptom:** BRCA1 genomic_position showed ~82 million instead of ~41 million.

**Root Cause:**
```python
# BUG: positions already contains absolute values
positions = pred_data.get('positions', ...)  # Contains 41196312, 41196313, ...

# This doubles the value!
'genomic_position': [gene_start + p - 1 for p in positions]
# Result: 41196312 + 41196312 - 1 = 82392623 ❌
```

**Fix:**
```python
absolute_positions = pred_data.get('positions', ...)

# Convert to relative for 'position' column
relative_positions = absolute_to_relative(
    absolute_positions, gene_start, gene_end, strand
)

# Use absolute values directly for 'genomic_position'
'position': relative_positions,          # RELATIVE
'genomic_position': absolute_positions   # ABSOLUTE (already correct)
```

---

## See Also

- `meta_models/core/position_types.py` - Source code with full docstrings
- `meta_models/workflows/splice_prediction_workflow.py` - Primary user of these utilities
- `docs/base_models/NUCLEOTIDE_SCORES_DESIGN_RATIONALE.md` (project-level) - Why nucleotide scores exist
- `docs/base_models/GENOME_BUILD_COMPATIBILITY.md` (project-level) - Build-specific coordinate considerations

