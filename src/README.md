# MetaSpliceAI - Production Source (Future)

**Status**: 🚧 Placeholder - For Future Refactoring  
**Current Development**: See `meta_spliceai/` directory

---

## Purpose

This directory is reserved for the **production-ready, refactored codebase** of MetaSpliceAI.

Once we achieve a viable, generalizable solution for:
- ✅ Detecting alternative splice sites induced by genetic variants
- ✅ Predicting splice effects across disease contexts (cancer, neurological, etc.)
- ✅ Position localization with validated ground truth

...we will refactor the essential packages and modules from `meta_spliceai/` into this 
clean `src/` structure.

---

## Current Development Location

All active development is in:

```
meta_spliceai/
├── splice_engine/
│   ├── meta_layer/          # Meta-learning layer (ACTIVE)
│   │   ├── models/          # Delta predictor, position localizer, etc.
│   │   ├── data/            # SpliceVarDB loader, variant datasets
│   │   ├── docs/            # Comprehensive documentation
│   │   └── tests/           # Training scripts and experiments
│   ├── models/              # Base models (OpenSpliceAI, SpliceAI)
│   └── case_studies/        # Data source integrations
└── system/                  # Genomic resources, config management
```

---

## Planned Structure (Post-Refactoring)

```
src/
├── metaspliceai/
│   ├── __init__.py
│   ├── core/
│   │   ├── base_model.py         # Unified base model interface
│   │   ├── meta_model.py         # Production meta-layer model
│   │   └── ensemble.py           # Base + Meta ensemble
│   │
│   ├── models/
│   │   ├── delta_predictor.py    # Validated delta prediction
│   │   ├── position_localizer.py # Aberrant site localization
│   │   └── effect_classifier.py  # Effect type classification
│   │
│   ├── data/
│   │   ├── variant_loader.py     # Unified variant loading
│   │   ├── splice_sites.py       # Canonical + induced sites
│   │   └── genome.py             # Reference genome interface
│   │
│   ├── inference/
│   │   ├── predictor.py          # Main prediction interface
│   │   ├── batch.py              # Batch prediction
│   │   └── vcf_annotator.py      # VCF annotation pipeline
│   │
│   └── utils/
│       ├── encoding.py           # Sequence encoding
│       └── coordinates.py        # Coordinate conversion
│
├── tests/
│   └── ...
│
└── examples/
    ├── predict_variant.py
    ├── annotate_vcf.py
    └── train_custom_model.py
```

---

## Refactoring Criteria

Before moving to `src/`, we need:

### Must Have
- [ ] Correlation r > 0.7 on validated delta prediction
- [ ] Position localization accuracy > 80% (within 10bp)
- [ ] Ground truth aberrant splice site annotations
- [ ] Robust evaluation on held-out variants

### Should Have
- [ ] Multi-task model (classification + localization + delta)
- [ ] Disease-specific fine-tuning capability
- [ ] Long-context support (>10kb)

### Nice to Have
- [ ] Pre-trained models for common diseases
- [ ] API server for predictions
- [ ] Integration with clinical pipelines

---

## Timeline

| Phase | Status | Target |
|-------|--------|--------|
| Validated Delta Prediction | ✅ Working (r=0.61) | Done |
| Position Localization | 🔄 In Progress | Q1 2026 |
| Aberrant Site Annotations | 📋 Planned | Q1-Q2 2026 |
| Production Refactor | ⏳ Future | Q2-Q3 2026 |

---

## See Also

- `meta_spliceai/splice_engine/meta_layer/docs/` - Current documentation
- `meta_spliceai/splice_engine/meta_layer/docs/wishlist/` - Future experiments
- `docs/` - Project-wide documentation

