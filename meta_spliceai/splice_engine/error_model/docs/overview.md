# Error Analysis and Modeling Overview

## Introduction

Error analysis is a crucial stage in the MetaSpliceAI workflow that identifies and investigates false positives (FPs) and false negatives (FNs) in splice site predictions. This process helps uncover patterns—such as sequence motifs or genomic features—that contribute to prediction errors, ultimately determining when and why the base model makes mistakes.

## Purpose

The error analysis stage serves several key purposes:

1. **Error Pattern Identification**: Discover systematic patterns in prediction errors
2. **Feature Importance Analysis**: Determine which genomic or sequence features correlate with errors
3. **Model Interpretability**: Provide insights into the base model's decision-making process
4. **Error Correction Strategy**: Guide the development of meta-models to correct these errors

## Components

The error analysis framework consists of two complementary modeling approaches:

### Classical ML Models

These models use traditional machine learning techniques like XGBoost to classify errors based on:
- Sequence context features
- Gene-level features
- Transcript-level features

SHAP (SHapley Additive exPlanations) value analysis is then applied to identify important features, such as sequence motifs and exon characteristics, that contribute to prediction errors.

### Transformer-Based Models

Pre-trained models like DNABERT or HyenaDNA are fine-tuned to classify errors, leveraging:
- Attention weights to analyze sequence motifs
- Region importance analysis to identify critical areas
- Complex pattern recognition for long-range dependencies

This approach is particularly effective for capturing complex sequence patterns that classical models might miss.

## Workflow

The error analysis workflow typically follows these steps:

1. **Data Preparation**: Extract sequences and features from error sites and correct predictions
2. **Feature Engineering**: Generate relevant features for analysis
3. **Model Training**: Train error classification models
4. **Model Evaluation**: Assess model performance and error reduction potential
5. **Feature Importance**: Analyze which features contribute most to errors
6. **Visualization**: Create visualizations to interpret model decisions

## Integration with MetaSpliceAI

Error analysis is positioned between the base prediction stage and the meta-model stage in the MetaSpliceAI pipeline:

```
Base Prediction (SpliceAI) → Error Analysis → Meta-Model → Isoform Discovery
```

The insights gained from error analysis directly inform the development of meta-models that improve overall splice site prediction accuracy.

## Getting Started

See the [workflow documentation](workflow.md) for detailed information on running the error analysis pipeline.
