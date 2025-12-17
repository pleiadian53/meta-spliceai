# HGVS Notation Tutorial: Position Hints for Splice Site Analysis

**Purpose**: Understanding HGVS notation for deriving position labels  
**Status**: 📚 Educational Reference  
**Created**: December 2025

---

## Overview

**HGVS (Human Genome Variation Society)** is a standardized nomenclature for describing 
sequence variants in DNA, RNA, and protein. For splice variant analysis, HGVS notation 
provides valuable *position hints* about where a variant is located relative to exon boundaries.

However, **HGVS labels are considered "weak" compared to delta-based labels** because they 
encode *variant location*, not *actual splice effect*. This document explains the 
distinction and when each approach is appropriate.

---

## Why HGVS is a "Weak Label" Source

### The Core Problem

HGVS tells you **WHERE** a variant is, but not **WHAT EFFECT** it has:

| Information Source | What It Tells You | What It Doesn't Tell You |
|--------------------|-------------------|--------------------------|
| **HGVS notation** | Variant is at donor site +1 position | Whether splicing is actually disrupted |
| **Delta from base model** | Donor probability changed by -0.85 | N/A (directly measures effect) |

### Example: Same Position, Different Effects

```
Variant A: c.123+1G>A  (HGVS: canonical donor +1)
Variant B: c.123+1G>C  (HGVS: canonical donor +1)

HGVS analysis: Both variants are at canonical donor site → Both "should" disrupt splicing

Reality:
- Variant A: Complete exon skipping (GT→AT destroys donor)
- Variant B: Partial effect (GT→CT may have residual function due to cryptic site usage)

Base model delta:
- Variant A: Δ_donor = -0.92 at position 123
- Variant B: Δ_donor = -0.45 at position 123, Δ_donor = +0.38 at position 178 (cryptic)
```

### Why This Matters for Training

If you train a model using HGVS-derived labels:
- **Label**: "Position 123 is affected" (from HGVS `c.123+1G>A`)
- **Reality**: Position 123 AND position 178 are affected (cryptic activation)

The model learns an incomplete picture. Delta-based labels capture the full picture.

---

## HGVS Notation Structure

### Basic Format

```
Reference:Position.Change
    │        │      │
    │        │      └── What changed (e.g., G>A)
    │        └── Where (coding position + intronic offset)
    └── Reference sequence (e.g., NM_001234.5)
```

### Coordinate Types

| Prefix | Meaning | Example |
|--------|---------|---------|
| `c.` | Coding DNA (CDS position) | `c.123G>A` (exonic) |
| `g.` | Genomic (chromosome position) | `g.123456G>A` (absolute position) |
| `r.` | RNA | `r.123g>a` (after transcription) |
| `p.` | Protein | `p.Arg123Cys` (amino acid change) |

---

## Understanding c. Notation: What Does c.123 Mean?

### c. = CDS (Coding DNA Sequence) Position

The `c.` notation refers to the **position within the coding sequence**:

```
c.123 = The 123rd nucleotide in the CDS
      = Position 123 counting from the first base of the START codon (ATG)
```

**Key points:**
- It is **NOT** a genomic coordinate (that would be `g.`)
- It is **NOT** an exon number
- It is **transcript-relative** (different transcripts have different c. numbering)
- Only counts nucleotides that will be translated to protein

### How Do We Know Which Gene?

The gene is specified by the **reference sequence prefix**:

```
NM_000492.4:c.1521+1G>A
│         │ │
│         │ └── CDS position 1521
│         └── Transcript version
└── RefSeq transcript accession (this is CFTR)
```

| Prefix | Meaning | Example |
|--------|---------|---------|
| `NM_` | mRNA (coding transcript) | `NM_000492.4` (CFTR) |
| `NR_` | Non-coding RNA | `NR_046018.2` |
| `NP_` | Protein | `NP_000483.3` |
| `NC_` | Genomic (chromosome) | `NC_000007.14` (chr7) |
| `NG_` | Genomic (gene region) | `NG_016465.4` (CFTR gene) |

**Without the prefix, the gene is ambiguous.** Many databases omit it when context is clear.

---

## Exon vs CDS: Understanding the Relationship

**Critical distinction: Exons ≠ CDS**

### Visual Overview

```
GENE STRUCTURE (DNA → mRNA → Protein)
══════════════════════════════════════════════════════════════════════════════

GENOMIC DNA:
5' ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3'

TRANSCRIPT (after splicing):
     ┌───────────────────────── mRNA ─────────────────────────┐
     │                                                         │
     │  ┌───┐   ┌─────────┐   ┌───────────┐   ┌───────┐  ┌──┐ │
     │  │ E1│   │   E2    │   │    E3     │   │  E4   │  │E5│ │
     │  └───┘   └─────────┘   └───────────┘   └───────┘  └──┘ │
     │    │         │              │              │        │   │
     │  5'UTR    CODING        CODING         CODING    3'UTR  │
     │    │         │              │              │        │   │
     │    │    ┌────┴──────────────┴──────────────┴────┐   │   │
     │    │    │           CDS (translated)           │   │   │
     │    │    │      ATG ─────────────────── STOP    │   │   │
     │    │    └──────────────────────────────────────┘   │   │
     │    │         │                              │      │   │
     │ Not CDS    c.1                           c.END  Not CDS │
     └─────────────────────────────────────────────────────────┘

Legend:
  ───── Introns (spliced out, not in mRNA)
  ┌───┐ Exons (included in mRNA)
  CDS   Coding sequence (translated to protein)
  UTR   Untranslated regions (in mRNA but not translated)
```

### Detailed Breakdown

```
            EXON 1          EXON 2              EXON 3           EXON 4
           ┌────────┐     ┌──────────┐        ┌──────────┐     ┌────────┐
mRNA:      │5'UTR│C│     │ CODING   │        │ CODING   │     │C│3'UTR │
           └────────┘     └──────────┘        └──────────┘     └────────┘
              │  │             │                   │            │
              │  │             │                   │            │
              │ c.1          c.150               c.450        c.600
              │ (ATG)                                         (STOP)
              │
           Not in CDS
           (5' UTR)

Key insight:
- Exon 1 contains 5'UTR (untranslated) + start of CDS
- Exons 2-3 are entirely within CDS  
- Exon 4 contains end of CDS + 3'UTR (untranslated)
- c. numbering ONLY covers the ATG-to-STOP region
```

### What Each Region Contains

| Region | In Exons? | In CDS? | c. Notation | Translated? |
|--------|-----------|---------|-------------|-------------|
| 5' UTR | ✅ Yes | ❌ No | `c.-N` (negative) | ❌ No |
| Coding | ✅ Yes | ✅ Yes | `c.1` to `c.END` | ✅ Yes |
| 3' UTR | ✅ Yes | ❌ No | `c.*N` (asterisk) | ❌ No |
| Introns | ❌ No | ❌ No | `c.X+N` or `c.X-N` | ❌ No |

---

## Intronic Offset Notation Explained

The `+` and `-` offsets indicate distance from the **CDS exon boundary**:

### Donor Site Region (5' end of intron)

```
                     EXON (CDS)                        INTRON
              ...━━━━━━━━━━━━━━━━━━━━┫ ┃━━━━━━━━━━━━━━━━━━━━━━...
                                    ┃ ┃
                     c.123          ┃ ┃ c.123+1   c.123+2   c.123+3
                    (last CDS       ┃ ┃  (1bp      (2bp      (3bp
                     base of        ┃ ┃   into     into      into
                     this exon)     ┃ ┃  intron)  intron)   intron)
                                    ┃ ┃
                               Donor site
                               (GT dinucleotide)
```

### Acceptor Site Region (3' end of intron)

```
                     INTRON                            EXON (CDS)
              ...━━━━━━━━━━━━━━━━━━━━┫ ┃━━━━━━━━━━━━━━━━━━━━━━...
                                    ┃ ┃
     c.456-3   c.456-2   c.456-1    ┃ ┃ c.456
      (3bp      (2bp      (1bp      ┃ ┃  (first CDS
      before    before    before    ┃ ┃   base of
      exon)     exon)     exon)     ┃ ┃   this exon)
                                    ┃ ┃
                               Acceptor site
                               (AG dinucleotide)
```

---

## Complete Real-World Example

```
Gene: CFTR (Cystic Fibrosis Transmembrane Conductance Regulator)
Transcript: NM_000492.4

Genomic location: chr7:117,480,025-117,668,665 (188 kb span)
CDS length: c.1 to c.4443 (1480 amino acids × 3 = 4440, plus stop)
Structure: 27 exons

Example variant: NM_000492.4:c.1521+1G>A

Breakdown:
├── NM_000492.4  → CFTR transcript (RefSeq accession)
├── c.1521       → CDS position 1521 (this falls in exon 10)
├── +1           → 1bp INTO the intron (after exon 10 ends)
├── G>A          → Reference G changed to alternate A
└── Effect       → Disrupts donor splice site of exon 10
                   (destroys the canonical GT → AT)
```

---

## Converting Between Coordinate Systems

### c. (CDS) → Genomic Coordinate

This requires:
1. **Transcript ID** (tells you which gene/isoform)
2. **Genome build** (GRCh37 vs GRCh38 have different coordinates)
3. **Transcript annotation** (exon coordinates from GTF/GFF)

```python
# Python example using biocommons.hgvs library
from hgvs.parser import Parser
from hgvs.dataproviders.uta import connect
from hgvs.mapper import AssemblyMapper

# Connect to UTA (Universal Transcript Archive)
hdp = connect()
am = AssemblyMapper(hdp, assembly_name="GRCh38")

# Parse HGVS
hp = Parser()
var_c = hp.parse("NM_000492.4:c.1521+1G>A")

# Map to genomic coordinates
var_g = am.c_to_g(var_c)
print(var_g)  # NC_000007.14:g.117559590C>T
```

### Quick Reference Table

| Notation | What It Means | Coordinate System |
|----------|---------------|-------------------|
| `c.123` | CDS position 123 | Transcript-relative, coding only |
| `c.123+5` | 5bp into intron after CDS pos 123 | Transcript-relative, intronic |
| `c.123-5` | 5bp before exon containing CDS pos 123 | Transcript-relative, intronic |
| `c.-10` | 10bp into 5'UTR (before ATG) | Transcript-relative, non-coding |
| `c.*10` | 10bp into 3'UTR (after STOP) | Transcript-relative, non-coding |
| `g.12345678` | Genomic position 12345678 | Chromosome-absolute |

---

## Using the Existing HGVS Parser

We have an HGVS parser at `meta_spliceai/splice_engine/case_studies/formats/hgvs_parser.py`:

```python
from meta_spliceai.splice_engine.case_studies.formats.hgvs_parser import (
    HGVSParser,
    HGVSVariant
)

parser = HGVSParser()

# Parse a splice site variant
variant = parser.parse("NM_001234.5:c.670-1G>T")

print(f"Position: {variant.start_position}")         # 670
print(f"Intronic offset: {variant.intronic_offset}") # -1
print(f"Valid: {variant.is_valid}")                  # True
print(f"Is splice site: {parser.is_splice_site_variant(variant)}")  # True
print(f"Site type: {parser.get_splice_site_type(variant)}")         # "acceptor"
```

### Supported Patterns

| Pattern | Example | Description |
|---------|---------|-------------|
| Substitution | `c.123G>A` | Single nucleotide change |
| Intronic | `c.123+1G>A` | Donor site region |
| Intronic | `c.123-2A>G` | Acceptor site region |
| Deletion | `c.123_125del` | Multi-base deletion |
| Insertion | `c.123_124insGGG` | Insertion |
| Delins | `c.123_125delinsAT` | Deletion-insertion |

---

## Position Hint Derivation from HGVS

### Basic Approach

```python
def derive_position_hints_from_hgvs(hgvs_string: str) -> dict:
    """
    Extract position hints from HGVS notation.
    
    Returns HINTS, not definitive labels. Use for:
    1. Pre-filtering variants
    2. Providing weak supervision signals
    3. Sanity checking delta-based predictions
    """
    parser = HGVSParser()
    variant = parser.parse(hgvs_string)
    
    if not variant.is_valid:
        return {'confidence': 'unparseable', 'hints': []}
    
    hints = {
        'confidence': 'low',  # HGVS is always "low" confidence for effect prediction
        'hints': []
    }
    
    offset = variant.intronic_offset
    
    if offset is None:
        # Exonic variant - could affect ESE/ESS, less direct splice effect
        hints['hints'].append({
            'type': 'exonic',
            'mechanism': 'ESE/ESS disruption possible',
            'position_confidence': 'very_low'
        })
    
    elif offset > 0:  # Donor region (positive offset)
        if offset <= 2:
            hints['confidence'] = 'medium'  # Canonical position, more predictable
            hints['hints'].append({
                'type': 'donor_loss',
                'mechanism': 'Canonical donor site disruption',
                'canonical_region': True,
                'offset': offset
            })
        elif offset <= 8:
            hints['hints'].append({
                'type': 'donor_loss_possible',
                'mechanism': 'Extended splice region',
                'canonical_region': False,
                'offset': offset
            })
        else:
            hints['hints'].append({
                'type': 'cryptic_possible',
                'mechanism': 'Deep intronic - cryptic site creation?',
                'offset': offset
            })
    
    else:  # Acceptor region (negative offset)
        if abs(offset) <= 2:
            hints['confidence'] = 'medium'
            hints['hints'].append({
                'type': 'acceptor_loss',
                'mechanism': 'Canonical acceptor site disruption',
                'canonical_region': True,
                'offset': offset
            })
        elif abs(offset) <= 25:
            hints['hints'].append({
                'type': 'acceptor_loss_possible',
                'mechanism': 'Polypyrimidine tract or branch point region',
                'canonical_region': False,
                'offset': offset
            })
        else:
            hints['hints'].append({
                'type': 'cryptic_possible',
                'mechanism': 'Deep intronic - cryptic site creation?',
                'offset': offset
            })
    
    return hints
```

### Confidence Levels

| Confidence | Meaning | When to Use |
|------------|---------|-------------|
| `high` | Strong evidence of effect location | **Never from HGVS alone** |
| `medium` | Reasonable guess at effect location | Canonical splice sites (±1, ±2) |
| `low` | Uncertain, use with caution | Extended regions (±3 to ±20) |
| `very_low` | Speculative | Deep intronic, exonic |

---

## When to Use HGVS vs Delta-Based Labels

### Use HGVS Labels When:

1. **Pre-filtering variants** before running base model
   ```python
   # Quick filter: only process variants near splice sites
   if parser.is_splice_site_variant(variant):
       # Worth running expensive delta computation
       delta = compute_base_model_delta(variant)
   ```

2. **No base model available** (fallback only)
   ```python
   if base_model_unavailable:
       # Fall back to HGVS hints (worse but better than nothing)
       hints = derive_position_hints_from_hgvs(variant.hgvs)
   ```

3. **Sanity checking** delta predictions
   ```python
   hgvs_hint = derive_position_hints_from_hgvs(variant.hgvs)
   delta_label = derive_position_labels_from_delta(ref, alt, models)
   
   if hgvs_hint['type'] == 'donor_loss' and delta_label.effect_type == 'acceptor_gain':
       # Unusual: HGVS says donor, delta says acceptor
       # Flag for manual review
       flag_for_review(variant)
   ```

4. **Weak supervision** in semi-supervised learning
   ```python
   # Use HGVS as weak labels for unlabeled variants
   weak_labels = {v.id: derive_position_hints_from_hgvs(v.hgvs) for v in unlabeled}
   ```

### Use Delta-Based Labels When:

1. **Training position localization models** (primary recommendation)
2. **Need accurate effect location**, not just variant location
3. **Multiple effects possible** (gain + loss, cryptic activation)
4. **Quantitative analysis** (how much effect, not just where)

---

## HGVS Pattern Reference

### Splice Site Classification Table

| HGVS Pattern | Distance | Region | Likely Effect | Confidence |
|--------------|----------|--------|---------------|------------|
| `c.XXX+1` | +1 | Canonical donor GT | Donor loss | Medium |
| `c.XXX+2` | +2 | Canonical donor GT | Donor loss | Medium |
| `c.XXX+3..+8` | +3 to +8 | Extended donor | Donor loss possible | Low |
| `c.XXX-1` | -1 | Canonical acceptor AG | Acceptor loss | Medium |
| `c.XXX-2` | -2 | Canonical acceptor AG | Acceptor loss | Medium |
| `c.XXX-3..-25` | -3 to -25 | Polypyrimidine tract | Acceptor weakening | Low |
| `c.XXX-26..-50` | -26 to -50 | Branch point region | Acceptor weakening | Very low |
| `c.XXX+N` (N > 10) | Deep 5' | Deep intronic | Cryptic creation? | Very low |
| `c.XXX-N` (N > 50) | Deep 3' | Deep intronic | Cryptic creation? | Very low |
| `c.XXX` (no offset) | 0 | Exonic | ESE/ESS disruption? | Very low |

### Canonical Splice Site Sequences

```
                    EXON          INTRON
Donor site:    ...MAG|gtaagt...
                  -2-1+1+2+3+4+5+6

                    INTRON         EXON
Acceptor site: ...yyyyyyyyag|G...
               -30        -2-1+1
```

Where:
- `M` = A or C
- `gt` = canonical donor (rarely GC)
- `ag` = canonical acceptor
- `y` = pyrimidine (C or T) in polypyrimidine tract

---

## Integration with Position Labels Module

The `data/position_labels.py` module provides an HGVS-based derivation function:

```python
from meta_spliceai.splice_engine.meta_layer.data.position_labels import (
    derive_position_from_hgvs
)

# Parse HGVS and get position hint
affected_pos = derive_position_from_hgvs("c.123+1G>A")

if affected_pos:
    print(f"Effect type: {affected_pos.effect_type}")  # 'donor_loss'
    print(f"Channel: {affected_pos.channel}")          # 2 (donor)
    print(f"Confidence: weak (from HGVS)")
```

**Important**: This returns a position hint with **placeholder delta value**. 
The actual delta must come from base model computation.

---

## Alternative Data Sources for Position Labels

### Compared to ClinVar

| Aspect | HGVS (from SpliceVarDB) | ClinVar |
|--------|-------------------------|---------|
| Position specificity | Variant location only | Variant location + some consequence annotation |
| Effect annotation | Implicit (inferred from location) | `MC` field with consequence type |
| Actual isoform data | **No** | **No** |
| Ground truth quality | Experimentally validated | Mixed (computational + clinical) |

**Key insight**: Neither SpliceVarDB nor ClinVar provide **actual isoform data** (i.e., 
GTF annotations of aberrant transcripts resulting from the variant). Both provide:
- Variant genomic coordinates
- Classification of effect type (splice-altering vs normal)
- But NOT the precise positions of cryptic splice sites or alternative exons

### For True Position Ground Truth

To get **actual splice isoform data** (alternative transcripts), you would need:
1. **RNA-seq data** from variant carriers showing aberrant transcripts
2. **Long-read sequencing** (PacBio/ONT) with full transcript isoforms
3. **Minigene assay results** with splicing outcomes

These are typically available in:
- **Individual papers** (case studies)
- **DBASS** (Database of Aberrant 3' and 5' Splice Sites) - has some
- **SpliceAI supplementary data** - has some validated examples

---

## Summary

| Label Source | Pros | Cons | Best Use |
|--------------|------|------|----------|
| **Delta from base model** | Quantitative, captures all effects | Depends on model quality | Primary training labels |
| **HGVS notation** | Fast, no model needed | Location ≠ effect, misses cryptics | Pre-filtering, weak supervision |
| **SpliceVarDB classification** | Ground truth filtering | Binary (altering/normal), no positions | Validating delta labels |
| **ClinVar MC field** | Consequence annotation | Coarse categories | Filtering by consequence |

**Recommended approach**: Use delta-based labels (validated by SpliceVarDB) as primary 
training signal, with HGVS hints for pre-filtering and sanity checking.

---

## See Also

- `case_studies/formats/hgvs_parser.py` - HGVS parser implementation
- `data/position_labels.py` - Position label derivation utilities
- `docs/methods/MULTI_STEP_FRAMEWORK.md` - Multi-Step Framework documentation
- `docs/data/SPLICEVARDB.md` - SpliceVarDB dataset documentation

