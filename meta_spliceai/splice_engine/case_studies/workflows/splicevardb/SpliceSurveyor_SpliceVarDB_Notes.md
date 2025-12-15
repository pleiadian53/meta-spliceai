# MetaSpliceAI – SpliceVarDB Integration

This document summarizes how to use **SpliceVarDB** as a training dataset for the **MetaSpliceAI meta-learning layer**.  
It includes workflow diagrams, column schema, and notes.

---

## Workflow Overview

```mermaid
flowchart TD
  A[SpliceVarDB\n(>50k variants, assays, outcomes)] --> B[Ingest & Harmonize\n(build, fields, PMIDs)]
  B --> C[Normalize to VCF\nleft-align, multiallelic split\nbcftools + hg38.fa]
  C --> D[OpenSpliceAI Variant Mode\nWT vs ALT within ±50nt/±400nt]
  D --> E[Delta Feature Builder\nALG/ALL/DLG/DLL + offsets\nnearest baseline scores\nregion & distance features]
  B -->|assay type, evidence| E
  B -->|class labels| F[Labeler\nsplice-altering = 1\nnot splice-altering = 0\nlow-frequency = weak/holdout]
  E --> G[Meta-Learning Layer\n(MetaSpliceAI)]
  F --> G
  G --> H[Calibrated Predictions\nsite gain/loss, confidence\ncandidate cryptic sites]
  H --> I[Reports & Triage\nper-variant rationale\nlinks to PMIDs/assays]
  G --> J[Evaluation & QA\nby region (canonical/cryptic/deep intronic)\nby gene holdout]
  J -->|thresholds & weights| G
```

---

## Training Table Schema

```mermaid
flowchart LR
  subgraph Inputs
    X1[Variant (CHROM, POS, REF, ALT)]
    X2[Base Scores\nSpliceAI/OpenSpliceAI\nWT & ALT]
    X3[Delta Metrics\nALG, ALL, DLG, DLL\nALG_POS, DLG_POS]
    X4[Context Features\n±k-mers, GC, distance to junction\nregion class: canonical/near/PPT/BP/deep]
    X5[Evidence & Assay\nRNA-seq / minigene / RT-PCR\nPMIDs, evidence strength]
  end

  subgraph Training Row
    T1[features.parquet row]
    T2[label]
  end

  X1-->T1
  X2-->T1
  X3-->T1
  X4-->T1
  X5-->T1
  Y1[SpliceVarDB Class\nsplice-altering / not / low-freq]-->T2

  T1-->M[Meta Model\n(LogReg/XGB/NN)]
  T2-->M
  M-->O[Pred: splice impact + site gain/loss + score]
```

---

## Column Specification (train-ready)

- `chrom, pos, ref, alt, gene, build`  
- `class` → labels:  
  - `splice-altering` → **1 (positive)**  
  - `not splice-altering` → **0 (negative)**  
  - `low-frequency splice-altering` → **weak/holdout**  
- `donor_gain, donor_loss, acceptor_gain, acceptor_loss, dlg_pos, alg_pos`  
- `spliceai_wt_donor, spliceai_wt_acceptor, spliceai_alt_donor, spliceai_alt_acceptor` (optional but useful)  
- `region_class` (canonical / near / PPT / BP / deep_intronic)  
- `dist_to_nearest_junction`  
- `assay_type` (RNA-seq / minigene / RT-PCR / MPRA)  
- `evidence_strength`  
- `pmids` (literature provenance)  

---

## Training Strategy Notes

1. **Positives vs Negatives:** clear separation; use SpliceVarDB curated classes.  
2. **Weak Class:** hold out `low-frequency` as evaluation or semi-supervised samples.  
3. **Weights:** weight by assay strength (RNA-seq > minigene > RT-PCR).  
4. **Generalization:**  
   - Stratify by region (canonical vs deep intronic vs cryptic).  
   - Hold out entire genes (e.g., CFTR, BRCA1/2) for external validation.  
5. **Calibration:** use SpliceVarDB negatives to tune thresholds and reduce false positives.  

---

**Next Steps:**  
- Integrate this training schema into MetaSpliceAI’s feature builder.  
- Benchmark against baseline SpliceAI delta-thresholding.  
- Document improvements by variant class (canonical, deep intronic, cryptic).  

