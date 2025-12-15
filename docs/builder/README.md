# Training Dataset Builder Documentation

## Overview

This directory contains **user-facing documentation** for the incremental training dataset builder, which generates training data for the meta-model by running base models (SpliceAI) and extracting features.

## Documentation in This Directory

### 📘 [USAGE_GUIDE.md](USAGE_GUIDE.md)
**Complete user guide for the incremental builder**

Topics covered:
- Quick start examples
- Two-phase architecture (Base Model Pass → Dataset Building)
- Complete command-line option reference
- Common workflows (from scratch, with existing artifacts, custom genes, etc.)
- Performance optimization (memory, speed, disk)
- Logging best practices
- Output structure and gene manifests
- Verification procedures
- Troubleshooting guide

**Start here if**: You want to build a training dataset

---

### 📋 [COMMAND_EXAMPLES.md](COMMAND_EXAMPLES.md)
**Copy-paste ready command-line examples**

Topics covered:
- Complete production example (recommended starting point)
- Quick test examples (100, 10 genes)
- Specialized examples (error-focused, disease-specific, custom lists)
- Memory-constrained environments (laptop vs server)
- Multi-kmer datasets
- Large-scale production builds
- Parameter selection guidelines
- Common patterns and verification

**Start here if**: You want example commands to run immediately

---

### 🧪 [TEST_GUIDE.md](TEST_GUIDE.md)
**Testing procedures for the builder**

Topics covered:
- Quick validation test (100 genes, 5-15 min)
- Test script usage (`./scripts/testing/test_incremental_builder.sh`)
- Verification procedures (output, logs, manifests, datasets)
- Expected outputs and success criteria
- Troubleshooting test failures
- Advanced testing scenarios

**Start here if**: You want to verify your setup works correctly

---

## Quick Start

### 1. Test Your Setup (5-15 minutes)
```bash
cd /Users/pleiadian53/work/meta-spliceai
./scripts/testing/test_incremental_builder.sh
```

### 2. Review the Output
```bash
ls -lh data/train_test_100genes_3mers/
cat data/train_test_100genes_3mers/gene_manifest.csv
```

### 3. Run Production Build (40-60 minutes)
```bash
./scripts/builder/run_builder_resumable.sh tmux
```

## Related Documentation

### Base Model Integration
- **[Base Model Pass Workflow](../base_models/BASE_MODEL_PASS_WORKFLOW.md)** - How the `--run-workflow` option works
  - SpliceAI model loading and inference
  - Feature engineering pipeline (~40-50 features)
  - Integration with the builder
  - Performance considerations

### Developer Documentation
For developers who want to understand or modify the builder code:
- **Package docs**: `meta_spliceai/splice_engine/meta_models/builder/docs/`
  - Architecture and design decisions
  - API reference
  - Contributing guidelines

### System Configuration
- **Genomic Resources**: `meta_spliceai/system/genomic_resources/`
  - How genomic data (GTF, FASTA, splice sites) is managed
  - Registry-based path resolution

### Scripts
- **Builder scripts**: `scripts/builder/`
  - `run_builder_resumable.sh` - Production build with resumable execution
- **Testing scripts**: `scripts/testing/`
  - `test_incremental_builder.sh` - Quick validation test

## Documentation Level: User-Facing

**This is USER documentation** (how to USE the builder), not developer documentation (how it WORKS internally).

| This Directory (`docs/builder/`) | Package Docs (`meta_spliceai/.../builder/docs/`) |
|----------------------------------|--------------------------------------------------|
| **How to USE the builder** | **How the builder WORKS internally** |
| Command-line examples | Architecture diagrams |
| Troubleshooting | API reference |
| Common workflows | Design decisions |
| Testing procedures | Contributing guidelines |

For the distinction between user and developer docs, see:
- [Documentation Architecture](../DOCUMENTATION_ARCHITECTURE.md)

## Most Common Questions

### Q: How do I build a training dataset?
**A**: See [USAGE_GUIDE.md](USAGE_GUIDE.md) → "Common Workflows" section

### Q: What commands should I run?
**A**: See [COMMAND_EXAMPLES.md](COMMAND_EXAMPLES.md) → "Complete Production Example"

### Q: How do I test if it works?
**A**: See [TEST_GUIDE.md](TEST_GUIDE.md) → "Quick Validation Test"

### Q: What does `--run-workflow` do?
**A**: See [../base_models/BASE_MODEL_PASS_WORKFLOW.md](../base_models/BASE_MODEL_PASS_WORKFLOW.md)

### Q: It's not working, help!
**A**: See [USAGE_GUIDE.md](USAGE_GUIDE.md) → "Troubleshooting" section

### Q: How does the builder work internally?
**A**: See `meta_spliceai/splice_engine/meta_models/builder/docs/` (developer docs)

## Feedback and Contributions

If you find issues with the documentation or have suggestions:
1. Open an issue describing the problem
2. Suggest improvements
3. Contribute documentation updates

---

**Last Updated**: October 17, 2025

