# Splice Surveyor Meta-Models Design Document

## Multi-Class (3-Way) Meta-Model Architecture

This document outlines the design and implementation plan for the Splice Surveyor meta-model architecture, which aims to improve splice site prediction accuracy by recalibrating SpliceAI predictions through a unified meta-learning approach.

### 🎯 Overview

The meta-model approach integrates with the existing MetaSpliceAI pipeline by recalibrating SpliceAI predictions through a single unified meta-model capable of adjusting splice-site probabilities across three distinct classes (donor, acceptor, neither).

### 🧬 Workflow

#### ① Training Phase

- **Inputs**:
  - Base SpliceAI predictions: (`p_donor`, `p_acceptor`, `p_neither`)
  - Additional genomic/transcriptomic context features (k-mers, gene-level attributes, regulatory elements)
- **Labels**:
  - Ground-truth splice-site annotations from genome (GTF/GFF3):
    - Donor (class 0)
    - Acceptor (class 1)
    - Neither (class 2)
- **Training Strategy**:
  - Multi-class classification model trained via categorical cross-entropy (or similar loss) to produce a corrected probability distribution (`p_donor'`, `p_acceptor'`, `p_neither'`) with probabilities summing to 1.

#### ② Inference Phase

- **Base Model (SpliceAI)**:
  - Generates initial probability predictions for each nucleotide position.
- **Candidate Filtering (Optional)**:
  - Discard positions with very low base probabilities to streamline meta-model computation.
- **Meta-Model Correction**:
  - For each candidate position:
    - Extract features: Base probabilities + genomic context.
    - Apply the trained meta-model to output corrected probabilities.
- **Final Predictions**:
  - Meta-model outputs coherent, corrected probabilities summing to 1 at every candidate site.

### 🧰 Implementation Components

#### A. Feature Set

Core features for the meta-model include:
- **Base Model Outputs** (critical):
  - SpliceAI probabilities (`p_donor`, `p_acceptor`, `p_neither`).
- **Genomic Context Features**:
  - Gene-level: GC content, exon counts, gene length.
  - Transcript-level: exon/intron structures, exon ratios, distances to boundaries.
  - Sequence-level: k-mer frequencies or embeddings (e.g., 4-mers, 6-mers).
- **Advanced Regulatory Features** (optional, recommended if available):
  - DNA methylation, chromatin accessibility (ATAC-seq), RBP motifs, ESE/ESS annotations.

Implementation strategy: Start with a minimal, highly interpretable set (e.g., SpliceAI outputs + transcript-level features). Expand incrementally based on validation-set performance.

#### B. Label Definitions

Three distinct classes per position:
- 0: Donor splice site
- 1: Acceptor splice site
- 2: Neither
- Labels implemented as categorical integers ({0,1,2}), optimized with categorical cross-entropy.
- Alternatively, labels can be stored in one-hot format (less common but valid).

#### C. Training Dataset Construction

- **Candidate Selection**:
  - Select positions likely relevant to splice prediction (e.g., threshold-based on base probabilities to ensure balanced training).
- **Class Balancing**:
  - Control sampling to mitigate class imbalance (oversampling minor classes, undersampling abundant negatives).
- **Strict Dataset Splitting**:
  - Train/Validation/Test splits with complete isolation of the final hold-out test set.

#### D. Training Strategy and Avoiding Data Leakage

- **Base Model Freeze**:
  - SpliceAI predictions generated once across entire dataset and kept fixed.
- **Meta-Model Training Protocol**:
  - Training on isolated training set.
  - Validation set used strictly for hyperparameter tuning.
  - Hold-out test set used exclusively for final unbiased evaluation.

#### E. Inference Workflow

- **Clear inference pipeline**:
- **Initial SpliceAI Run**:
  - Predict initial probabilities across entire new sequence.
- **Candidate Filtering** (optional):
  - Select positions exceeding a minimal probability threshold.
- **Meta-model Prediction**:
  - Generate corrected distribution (`p_donor'`, `p_acceptor'`, `p_neither'`) for candidate sites.
- **Final Decision**:
  - Threshold or argmax on corrected probabilities determines final splice site classification.

### 🚨 Advantages of Single Multi-Class Meta-Model

- **Unified Correction**: Single meta-model corrects all splice classes simultaneously, avoiding contradictions inherent in multiple binary models.
- **Simplified Logic**: Easier implementation and inference workflow.
- **Minimized Data Leakage**: Clear separation of roles, datasets, and frozen base predictions ensures unbiased evaluation.
- **Future-Proof & Scalable**: Easily incorporates new features or refined annotations.

### 📌 Implementation Roadmap

#### Phase 1: Foundation (Current)
- ✅ Enhanced prediction workflow with all three probability scores
- ✅ Automatic splice site adjustment detection
- ✅ Comprehensive error analysis framework

#### Phase 2: Feature Engineering
- 🔄 Extract and normalize features from SpliceAI predictions
- 🔄 Implement sequence-level feature extraction (k-mers, sequence windows)
- 🔄 Design gene and transcript-level feature calculations

#### Phase 3: Model Development
- 🔄 Implement dataset construction with class balancing
- 🔄 Develop model architecture (initial candidates: XGBoost, Neural Networks)
- 🔄 Create training workflow with proper validation strategy

#### Phase 4: Evaluation & Deployment
- 🔄 Design comprehensive evaluation metrics and benchmarks
- 🔄 Implement inference pipeline
- 🔄 Create visualization tools for model interpretation

## Relationship to Current Components

The current enhanced evaluation and workflow components serve as the foundation for this meta-model approach by:

1. **Preserving all three probability scores** from SpliceAI, which are the critical base features for the meta-model
2. **Providing standardized error analysis** that will be used to evaluate meta-model performance
3. **Implementing automatic adjustment detection** that provides insights into systematic biases in the base model
4. **Creating a flexible workflow** that can be extended to incorporate the meta-model prediction step

## Next Steps

With the foundation in place, the next implementation steps include:

1. Creating feature extraction modules for genomic context features
2. Implementing dataset construction with proper class balancing
3. Developing initial model architectures (XGBoost and neural network versions)
4. Designing the training and validation protocol
5. Implementing the inference pipeline
