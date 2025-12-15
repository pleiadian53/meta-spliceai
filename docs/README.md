# Meta-SpliceAI Documentation

**Project-level documentation for the Meta-SpliceAI framework**

---

## Documentation Structure

### 📚 Project-Level Documentation (`docs/`)
High-level documentation for users and contributors:

```
docs/
├── README.md                    # This file - documentation index
├── installation/                # Installation guides
├── tutorials/                   # User tutorials and examples
├── base_models/                 # Base model documentation
├── training/                    # Training workflow documentation
├── testing/                     # Testing procedures
└── development/                 # Development guidelines (public)
```

### 🔧 Package-Level Documentation
Technical documentation within packages:
- `meta_spliceai/splice_engine/docs/` - Splice engine internals
- `meta_spliceai/splice_engine/meta_models/builder/docs/` - Dataset builder
- etc.

---

## Quick Links

### For Users
- [Installation Guide](installation/INSTALLATION.md)
- [Quick Start Tutorial](tutorials/)
- [Base Model Comparison](base_models/BASE_MODEL_COMPARISON_GUIDE.md)

### For Developers
- [Development Guidelines](development/) *(if public)*
- [Testing Guide](testing/)
- [Training Documentation](training/)

### For Contributors
- [CONTRIBUTING.md](../CONTRIBUTING.md) *(when created)*
- [LICENSE](../LICENSE) *(when created)*

---

## Documentation Guidelines

### What Goes Where

#### Project-Level (`docs/`)
- Installation instructions
- User guides and tutorials
- High-level architecture
- API documentation
- Testing procedures
- Public development guidelines

#### Package-Level (`<package>/docs/`)
- Technical implementation details
- Module-specific documentation
- Algorithm descriptions
- Code examples specific to that module

---

## Contributing to Documentation

When adding documentation:

1. **User-facing**: Put in `docs/`
2. **Technical/module-specific**: Put in package's `docs/` subdirectory

### Style Guidelines
- Use Markdown format
- Include code examples where relevant
- Keep sections focused and concise
- Update this index when adding new docs

---

## Current Documentation Status

| Category | Status | Location |
|----------|--------|----------|
| **Installation** | ✅ Available | `docs/installation/` |
| **Base Models** | ✅ Complete | `docs/base_models/` |
| **Training** | ✅ Complete | `docs/training/` |
| **Testing** | ✅ Available | `docs/testing/` |
| **Tutorials** | ⏸️ Planned | `docs/tutorials/` |
| **API Reference** | ⏸️ Planned | TBD |
| **Contributing Guide** | ⏸️ Planned | `CONTRIBUTING.md` |

---

## Getting Started

1. **New users**: Start with [Installation Guide](installation/INSTALLATION.md)
2. **Developers**: See [Development Guidelines](development/) (when public)
3. **Contributors**: Read [CONTRIBUTING.md](../CONTRIBUTING.md) (when created)

---

**Meta-SpliceAI** - Meta-learning framework for splice site prediction
