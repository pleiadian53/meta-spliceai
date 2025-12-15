# Meta-SpliceAI Documentation Guide

This document explains the documentation structure for the Meta-SpliceAI project.

---

## 📁 Documentation Structure

### 1. Project-Level Documentation: `docs/`

**Purpose**: Public documentation for users and contributors (on GitHub)

**Location**: `docs/`

**Contents**:
- Installation guides
- User tutorials
- High-level architecture
- Base model documentation
- Training workflows
- Testing procedures
- Public development guidelines

**Example Structure**:
```
docs/
├── README.md                     # Documentation index
├── installation/                 # How to install and setup
│   ├── INSTALLATION.md
│   └── test_installation.sh
├── tutorials/                    # User guides and tutorials
├── base_models/                  # Base model documentation
│   ├── BASE_MODEL_COMPARISON_GUIDE.md
│   └── ...
├── training/                     # Training documentation
│   ├── TN_SAMPLING_FIX_IMPLEMENTATION.md
│   └── BASE_MODEL_ARTIFACTS_VERIFICATION.md
├── testing/                      # Testing guides
└── development/                  # PUBLIC dev guidelines only
    ├── README.md
    ├── MANE_VS_ENSEMBL_SPLICE_SITES.md
    └── ...
```

---

### 2. Package-Level Documentation: `<package>/docs/`

**Purpose**: Technical documentation specific to a package/module

**Location**: Within the package directory (e.g., `meta_spliceai/splice_engine/docs/`)

**Contents**:
- Module-specific implementation details
- Algorithm descriptions
- Internal API documentation
- Code examples for that module

**Example Structure**:
```
meta_spliceai/
├── splice_engine/
│   ├── docs/                     # Splice engine documentation
│   │   ├── consensus_analysis.md
│   │   └── ...
│   │
│   ├── meta_layer/               # ⭐ Multimodal meta-learning
│   │   └── docs/
│   │       ├── ARCHITECTURE.md           # System design
│   │       ├── LABELING_STRATEGY.md      # Label derivation
│   │       ├── TRAINING_VS_INFERENCE.md  # Data format differences
│   │       ├── methods/                  # Methodology docs
│   │       │   ├── ROADMAP.md
│   │       │   └── GPU_REQUIREMENTS.md
│   │       └── experiments/              # Experiment results
│   │
│   └── meta_models/
│       └── builder/
│           └── docs/             # Builder-specific docs
│               ├── artifact_validation.md
│               └── training_dataset_workflows.md
```

---

## 🔍 Finding Documentation

### As a User
Start at: `docs/README.md`

### As a Developer
- **Project overview**: `docs/`
- **Module details**: `<package>/docs/`

---

## 🚀 Getting Started

- **New Users**: Start with [`docs/README.md`](docs/README.md)
- **Installation**: Follow the installation guide
- **Tutorials**: Check `docs/tutorials/`
- **Development**: See [`docs/development/`](docs/development/)
- **Contributing**: Read [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## 📋 Documentation Status

| Category | Status | Location |
|----------|--------|----------|
| **Installation** | ✅ Available | `docs/installation/` |
| **Base Models** | ✅ Complete | `docs/base_models/` |
| **Training** | ✅ Complete | `docs/training/` |
| **Testing** | ✅ Available | `docs/testing/` |
| **Meta-Layer** | ✅ Complete | `meta_spliceai/splice_engine/meta_layer/docs/` |
| **API Reference** | ⏸️ Planned | TBD |
| **Tutorials** | ⏸️ Planned | `docs/tutorials/` |

---

## 📞 Questions & Contributions

- **Questions**: Open a [GitHub issue](https://github.com/pleiadian53/meta-spliceai/issues)
- **Contributions**: See [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Documentation feedback**: We welcome improvements!

---

**Meta-SpliceAI** - Meta-learning framework for splice site prediction

