# Alternative Splicing Pipeline: From Scores to Exon Predictions

**Last Updated**: December 2025  
**Status**: Design Document

---

## Overview

This document explains the pipeline that converts **recalibrated per-nucleotide splice site scores** into **predicted exon-intron structures** with proper donor-acceptor pairing.

```
Per-nucleotide scores  →  Splice site calls  →  Donor-acceptor pairs  →  Exon predictions
    P(donor)                [chr1:12345]         (D1 → A1)               Exon 1: 100-12345
    P(acceptor)             [chr1:12500]         (D2 → A2)               Exon 2: 12500-15000
    P(neither)              [chr1:15000]         ...                     ...
```

---

## The Complete Pipeline

### Stage 1: Score Recalibration (Meta-Layer)

**Input**: Base layer artifacts (sequence, features)  
**Output**: Recalibrated per-nucleotide probabilities

```python
# Meta-layer recalibrates base model scores
recalibrated_scores = meta_layer.predict(
    sequence=context_501nt,
    base_scores=base_model_features
)

# Output: [donor_prob, acceptor_prob, neither_prob] per nucleotide
```

### Stage 2: Splice Site Calling

**Input**: Recalibrated probabilities  
**Output**: Called splice site positions

```
┌─────────────────────────────────────────────────────────────────┐
│                      SPLICE SITE CALLING                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Thresholding                                           │
│  ───────────────────────────────────────────────────────────    │
│  Filter positions where P(donor) or P(acceptor) > threshold     │
│  Default threshold: 0.5                                          │
│                                                                  │
│  Step 2: Peak Detection                                          │
│  ───────────────────────────────────────────────────────────    │
│  Find local maxima to avoid calling overlapping sites           │
│  Use window-based peak detection (±5nt)                         │
│                                                                  │
│  Step 3: Consensus Filtering                                     │
│  ───────────────────────────────────────────────────────────    │
│  Check for canonical dinucleotide motifs:                       │
│  • Donor: GT at 5' end of intron (positions +1, +2)             │
│  • Acceptor: AG at 3' end of intron (positions -2, -1)          │
│                                                                  │
│  Step 4: Score-Based Ranking                                     │
│  ───────────────────────────────────────────────────────────    │
│  Rank sites by confidence for downstream pairing                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 3: Donor-Acceptor Pairing

**Input**: Called splice sites  
**Output**: Intron predictions (donor-acceptor pairs)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DONOR-ACCEPTOR PAIRING                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Rules for Valid Pairs:                                          │
│  ───────────────────────────────────────────────────────────    │
│  1. Strand consistency:                                          │
│     • Forward strand (+): Donor position < Acceptor position     │
│     • Reverse strand (-): Donor position > Acceptor position     │
│                                                                  │
│  2. Distance constraints:                                        │
│     • Minimum intron length: 20 nt (GT-AG consensus)            │
│     • Maximum intron length: ~1M nt (biological limit)          │
│     • Typical: 100 nt - 100 kb                                  │
│                                                                  │
│  3. Gene boundary:                                               │
│     • Pairs must be within the same gene                        │
│     • Use gene annotations to enforce boundaries                │
│                                                                  │
│  4. No crossing introns:                                         │
│     • Introns should not overlap in non-nested manner           │
│     • (Nested introns are rare but possible)                    │
│                                                                  │
│  5. Score threshold:                                             │
│     • Combined pair score: sqrt(P_donor * P_acceptor)           │
│     • Minimum pair score: 0.1                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 4: Exon Prediction

**Input**: Intron predictions  
**Output**: Exon structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      EXON PREDICTION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Gene: BRCA1 (chr17:43044295-43170245, strand -)                │
│                                                                  │
│  Predicted Introns:                                              │
│  ─────────────────                                               │
│  Intron 1: D(43125364) → A(43124017)   [score: 0.95]            │
│  Intron 2: D(43115779) → A(43106455)   [score: 0.92]            │
│  ...                                                             │
│                                                                  │
│  Derived Exons:                                                  │
│  ─────────────                                                   │
│  Exon 1:  43170245 - 43125365   (First exon to first donor)     │
│  Exon 2:  43124016 - 43115780   (Acceptor to next donor)        │
│  Exon 3:  43106454 - 43091435   (Acceptor to next donor)        │
│  ...                                                             │
│  Exon N:  43057052 - 43044295   (Last acceptor to gene end)     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Splice Site Caller

```python
# In meta_spliceai/splice_engine/meta_layer/inference/splice_site_caller.py

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class SpliceSite:
    """Represents a called splice site."""
    position: int           # Genomic position (1-based)
    site_type: str          # 'donor' or 'acceptor'
    score: float            # Probability from meta-layer
    chrom: str
    strand: str
    gene_id: str
    has_consensus: bool     # GT/AG motif present
    
    @property
    def is_reliable(self) -> bool:
        """Check if this is a high-confidence call."""
        return self.score >= 0.5 and self.has_consensus


class SpliceSiteCaller:
    """
    Call splice sites from per-nucleotide probabilities.
    
    Parameters
    ----------
    donor_threshold : float
        Minimum probability for donor sites (default: 0.5)
    acceptor_threshold : float
        Minimum probability for acceptor sites (default: 0.5)
    peak_window : int
        Window size for local maximum detection (default: 5)
    require_consensus : bool
        Whether to require GT/AG consensus motifs (default: True)
    """
    
    def __init__(
        self,
        donor_threshold: float = 0.5,
        acceptor_threshold: float = 0.5,
        peak_window: int = 5,
        require_consensus: bool = True
    ):
        self.donor_threshold = donor_threshold
        self.acceptor_threshold = acceptor_threshold
        self.peak_window = peak_window
        self.require_consensus = require_consensus
    
    def call_sites(
        self,
        positions: np.ndarray,    # [N] genomic positions
        donor_probs: np.ndarray,  # [N] P(donor) per position
        acceptor_probs: np.ndarray,  # [N] P(acceptor) per position
        sequence: str,            # Full gene sequence
        gene_start: int,          # Gene start position
        strand: str,              # '+' or '-'
        chrom: str,
        gene_id: str
    ) -> List[SpliceSite]:
        """
        Call splice sites from probabilities.
        
        Returns
        -------
        list of SpliceSite
            Called splice sites, sorted by position.
        """
        sites = []
        
        # Call donors
        donor_peaks = self._find_peaks(donor_probs, self.donor_threshold)
        for idx in donor_peaks:
            pos = positions[idx]
            site = SpliceSite(
                position=int(pos),
                site_type='donor',
                score=float(donor_probs[idx]),
                chrom=chrom,
                strand=strand,
                gene_id=gene_id,
                has_consensus=self._check_donor_consensus(
                    sequence, pos - gene_start, strand
                )
            )
            
            if not self.require_consensus or site.has_consensus:
                sites.append(site)
        
        # Call acceptors
        acceptor_peaks = self._find_peaks(acceptor_probs, self.acceptor_threshold)
        for idx in acceptor_peaks:
            pos = positions[idx]
            site = SpliceSite(
                position=int(pos),
                site_type='acceptor',
                score=float(acceptor_probs[idx]),
                chrom=chrom,
                strand=strand,
                gene_id=gene_id,
                has_consensus=self._check_acceptor_consensus(
                    sequence, pos - gene_start, strand
                )
            )
            
            if not self.require_consensus or site.has_consensus:
                sites.append(site)
        
        # Sort by position
        sites.sort(key=lambda s: s.position)
        
        return sites
    
    def _find_peaks(
        self,
        probs: np.ndarray,
        threshold: float
    ) -> List[int]:
        """Find local maxima above threshold."""
        peaks = []
        
        for i in range(len(probs)):
            if probs[i] < threshold:
                continue
            
            # Check if local maximum
            start = max(0, i - self.peak_window)
            end = min(len(probs), i + self.peak_window + 1)
            
            if probs[i] == probs[start:end].max():
                peaks.append(i)
        
        return peaks
    
    def _check_donor_consensus(
        self,
        sequence: str,
        rel_pos: int,
        strand: str
    ) -> bool:
        """Check for GT consensus at donor site."""
        try:
            if strand == '+':
                # GT should be at positions +1, +2 (right of exon)
                dinuc = sequence[rel_pos:rel_pos + 2].upper()
                return dinuc == 'GT'
            else:
                # Reverse complement: AC at positions -2, -1
                dinuc = sequence[rel_pos - 2:rel_pos].upper()
                return dinuc == 'AC'
        except (IndexError, TypeError):
            return False
    
    def _check_acceptor_consensus(
        self,
        sequence: str,
        rel_pos: int,
        strand: str
    ) -> bool:
        """Check for AG consensus at acceptor site."""
        try:
            if strand == '+':
                # AG should be at positions -2, -1 (left of exon)
                dinuc = sequence[rel_pos - 2:rel_pos].upper()
                return dinuc == 'AG'
            else:
                # Reverse complement: CT at positions +1, +2
                dinuc = sequence[rel_pos:rel_pos + 2].upper()
                return dinuc == 'CT'
        except (IndexError, TypeError):
            return False
```

### Exon Predictor

```python
# In meta_spliceai/splice_engine/meta_layer/inference/exon_predictor.py

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Intron:
    """Represents a predicted intron (donor-acceptor pair)."""
    donor: SpliceSite
    acceptor: SpliceSite
    length: int
    score: float  # Combined confidence
    
    @property
    def is_valid(self) -> bool:
        """Check biological validity."""
        return (
            20 <= self.length <= 1_000_000 and
            self.donor.site_type == 'donor' and
            self.acceptor.site_type == 'acceptor'
        )


@dataclass
class Exon:
    """Represents a predicted exon."""
    start: int
    end: int
    chrom: str
    strand: str
    gene_id: str
    upstream_acceptor: Optional[SpliceSite]  # None for first exon
    downstream_donor: Optional[SpliceSite]   # None for last exon
    
    @property
    def length(self) -> int:
        return abs(self.end - self.start) + 1


class ExonPredictor:
    """
    Predict exon-intron structure from called splice sites.
    
    Parameters
    ----------
    min_intron_length : int
        Minimum intron length (default: 20)
    max_intron_length : int
        Maximum intron length (default: 1,000,000)
    min_exon_length : int
        Minimum exon length (default: 10)
    min_pair_score : float
        Minimum combined donor-acceptor score (default: 0.1)
    """
    
    def __init__(
        self,
        min_intron_length: int = 20,
        max_intron_length: int = 1_000_000,
        min_exon_length: int = 10,
        min_pair_score: float = 0.1
    ):
        self.min_intron_length = min_intron_length
        self.max_intron_length = max_intron_length
        self.min_exon_length = min_exon_length
        self.min_pair_score = min_pair_score
    
    def predict_exons(
        self,
        splice_sites: List[SpliceSite],
        gene_start: int,
        gene_end: int,
        strand: str,
        chrom: str,
        gene_id: str
    ) -> Tuple[List[Intron], List[Exon]]:
        """
        Predict exon-intron structure from splice sites.
        
        Parameters
        ----------
        splice_sites : list of SpliceSite
            Called splice sites within the gene.
        gene_start : int
            Gene start position.
        gene_end : int
            Gene end position.
        strand : str
            Gene strand ('+' or '-').
        chrom : str
            Chromosome.
        gene_id : str
            Gene identifier.
        
        Returns
        -------
        introns : list of Intron
            Predicted introns.
        exons : list of Exon
            Predicted exons.
        """
        # Separate donors and acceptors
        donors = [s for s in splice_sites if s.site_type == 'donor']
        acceptors = [s for s in splice_sites if s.site_type == 'acceptor']
        
        # Find valid pairs
        introns = self._pair_donors_acceptors(donors, acceptors, strand)
        
        # Derive exons from introns
        exons = self._derive_exons(
            introns, gene_start, gene_end, strand, chrom, gene_id
        )
        
        return introns, exons
    
    def _pair_donors_acceptors(
        self,
        donors: List[SpliceSite],
        acceptors: List[SpliceSite],
        strand: str
    ) -> List[Intron]:
        """
        Pair donors with acceptors to form introns.
        
        Uses greedy pairing: for each donor, find the best matching acceptor.
        """
        introns = []
        used_acceptors = set()
        
        # Sort by position
        donors = sorted(donors, key=lambda s: s.position)
        acceptors = sorted(acceptors, key=lambda s: s.position)
        
        for donor in donors:
            best_acceptor = None
            best_score = -1
            
            for acceptor in acceptors:
                if id(acceptor) in used_acceptors:
                    continue
                
                # Check strand-specific ordering
                if strand == '+':
                    # Forward: donor < acceptor
                    if donor.position >= acceptor.position:
                        continue
                    intron_length = acceptor.position - donor.position
                else:
                    # Reverse: donor > acceptor
                    if donor.position <= acceptor.position:
                        continue
                    intron_length = donor.position - acceptor.position
                
                # Check length constraints
                if not (self.min_intron_length <= intron_length <= self.max_intron_length):
                    continue
                
                # Compute pair score
                pair_score = np.sqrt(donor.score * acceptor.score)
                
                if pair_score < self.min_pair_score:
                    continue
                
                if pair_score > best_score:
                    best_score = pair_score
                    best_acceptor = acceptor
            
            if best_acceptor is not None:
                intron = Intron(
                    donor=donor,
                    acceptor=best_acceptor,
                    length=abs(donor.position - best_acceptor.position),
                    score=best_score
                )
                introns.append(intron)
                used_acceptors.add(id(best_acceptor))
        
        # Sort introns by position
        introns.sort(key=lambda i: min(i.donor.position, i.acceptor.position))
        
        return introns
    
    def _derive_exons(
        self,
        introns: List[Intron],
        gene_start: int,
        gene_end: int,
        strand: str,
        chrom: str,
        gene_id: str
    ) -> List[Exon]:
        """
        Derive exons from intron boundaries.
        
        Exons are the regions between introns (and gene boundaries).
        """
        exons = []
        
        if not introns:
            # No introns = single exon gene
            exons.append(Exon(
                start=gene_start,
                end=gene_end,
                chrom=chrom,
                strand=strand,
                gene_id=gene_id,
                upstream_acceptor=None,
                downstream_donor=None
            ))
            return exons
        
        # Get all splice boundaries
        if strand == '+':
            # Forward strand: exon = [acceptor+1, donor-1]
            # First exon: [gene_start, first_donor-1]
            first_donor = min(i.donor.position for i in introns)
            exons.append(Exon(
                start=gene_start,
                end=first_donor - 1,
                chrom=chrom,
                strand=strand,
                gene_id=gene_id,
                upstream_acceptor=None,
                downstream_donor=introns[0].donor
            ))
            
            # Internal exons
            for i, intron in enumerate(introns[:-1]):
                next_intron = introns[i + 1]
                exon_start = intron.acceptor.position + 1
                exon_end = next_intron.donor.position - 1
                
                if exon_end - exon_start >= self.min_exon_length:
                    exons.append(Exon(
                        start=exon_start,
                        end=exon_end,
                        chrom=chrom,
                        strand=strand,
                        gene_id=gene_id,
                        upstream_acceptor=intron.acceptor,
                        downstream_donor=next_intron.donor
                    ))
            
            # Last exon
            last_acceptor = max(i.acceptor.position for i in introns)
            exons.append(Exon(
                start=last_acceptor + 1,
                end=gene_end,
                chrom=chrom,
                strand=strand,
                gene_id=gene_id,
                upstream_acceptor=introns[-1].acceptor,
                downstream_donor=None
            ))
        
        else:
            # Reverse strand: similar but reversed
            # (Details omitted for brevity, but logic is mirrored)
            pass
        
        return exons
```

---

## Filtering Decoy Sites

### Decoy Detection Criteria

Not all high-scoring positions are true splice sites. Decoys can be filtered by:

```
┌─────────────────────────────────────────────────────────────────┐
│                       DECOY DETECTION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Missing Consensus Motif                                      │
│     ─────────────────────────                                    │
│     • Donor without GT                                           │
│     • Acceptor without AG                                        │
│     • Minor motifs: GC-AG, AT-AC (rare)                         │
│                                                                  │
│  2. Orphan Sites                                                 │
│     ─────────────                                                │
│     • Donor with no matching acceptor                           │
│     • Acceptor with no matching donor                           │
│     • Outside gene boundaries                                    │
│                                                                  │
│  3. Low Confidence                                               │
│     ───────────────                                              │
│     • Score below threshold                                      │
│     • High entropy in probability distribution                  │
│     • Inconsistent with neighboring positions                   │
│                                                                  │
│  4. Biological Implausibility                                    │
│     ──────────────────────────                                   │
│     • Intron < 20 nt                                            │
│     • Exon < 10 nt                                              │
│     • Overlapping exons (non-nested)                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Confidence Scoring

```python
def compute_site_confidence(
    site: SpliceSite,
    all_sites: List[SpliceSite]
) -> float:
    """
    Compute confidence score for a splice site.
    
    Considers:
    - Raw probability score
    - Consensus motif presence
    - Pairing potential (can it form valid intron?)
    - Local context (is it isolated or part of a pattern?)
    """
    confidence = site.score
    
    # Boost for consensus motif
    if site.has_consensus:
        confidence *= 1.2
    else:
        confidence *= 0.5  # Penalize missing consensus
    
    # Check pairing potential
    if site.site_type == 'donor':
        # How many acceptors could pair with this?
        potential_pairs = sum(
            1 for s in all_sites 
            if s.site_type == 'acceptor' and _can_pair(site, s)
        )
    else:
        potential_pairs = sum(
            1 for s in all_sites 
            if s.site_type == 'donor' and _can_pair(s, site)
        )
    
    if potential_pairs == 0:
        confidence *= 0.3  # Penalize orphan sites
    
    return min(confidence, 1.0)
```

---

## Complete Workflow

```python
from meta_spliceai.splice_engine.meta_layer.inference import (
    MetaLayerPredictor,
    SpliceSiteCaller,
    ExonPredictor
)

def predict_alternative_splicing(
    gene_id: str,
    config: MetaLayerConfig,
    model_path: str
) -> Dict:
    """
    Complete pipeline: scores → exons.
    
    Returns
    -------
    dict with keys:
        - 'recalibrated_scores': Per-nucleotide probabilities
        - 'splice_sites': Called splice sites
        - 'introns': Predicted introns (donor-acceptor pairs)
        - 'exons': Predicted exon structure
    """
    
    # Step 1: Load meta-layer model
    predictor = MetaLayerPredictor(model_path, config)
    
    # Step 2: Get recalibrated scores
    scores = predictor.predict_gene(gene_id)
    
    # Step 3: Call splice sites
    caller = SpliceSiteCaller(
        donor_threshold=0.5,
        acceptor_threshold=0.5,
        require_consensus=True
    )
    
    splice_sites = caller.call_sites(
        positions=scores['positions'],
        donor_probs=scores['donor_probs'],
        acceptor_probs=scores['acceptor_probs'],
        sequence=scores['sequence'],
        gene_start=scores['gene_start'],
        strand=scores['strand'],
        chrom=scores['chrom'],
        gene_id=gene_id
    )
    
    # Step 4: Predict exons
    exon_predictor = ExonPredictor(
        min_intron_length=20,
        max_intron_length=1_000_000
    )
    
    introns, exons = exon_predictor.predict_exons(
        splice_sites=splice_sites,
        gene_start=scores['gene_start'],
        gene_end=scores['gene_end'],
        strand=scores['strand'],
        chrom=scores['chrom'],
        gene_id=gene_id
    )
    
    return {
        'recalibrated_scores': scores,
        'splice_sites': splice_sites,
        'introns': introns,
        'exons': exons
    }
```

---

## Summary

| Stage | Input | Output | Key Algorithm |
|-------|-------|--------|---------------|
| **1. Recalibration** | Base features + sequence | P(donor), P(acceptor), P(neither) | Meta-layer model |
| **2. Site Calling** | Probabilities | Splice site list | Peak detection + thresholding |
| **3. Pairing** | Splice sites | Intron list | Greedy pairing with constraints |
| **4. Exon Prediction** | Introns + gene bounds | Exon structure | Boundary derivation |
| **5. Filtering** | All predictions | Reliable predictions | Decoy detection |

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [LABELING_STRATEGY.md](LABELING_STRATEGY.md) - How labels are created
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Step-by-step training

---

*Last Updated: December 2025*

