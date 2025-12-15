# Splice Site Prediction Foundation Model

This module implements a deep learning-based splice site predictor capable of accurately identifying splice donor and acceptor sites from genomic sequences. It provides foundational predictions for MetaSpliceAI workflows.

## Key Features
- Processes genomic sequence data (FASTA) and annotation data (GTF)
- Implements multiple model architectures:
  - Transformer-based models (DNABERT, HyenaDNA)
  - CNN with dilated convolutions
- Binary classification for splice site probability prediction
- Advanced optimization techniques (dropout, LayerNorm, residual connections)
- Interpretability features (attention maps, feature importance visualization)

## Usage
See `tutorial.ipynb` for detailed examples of training and using the model.
