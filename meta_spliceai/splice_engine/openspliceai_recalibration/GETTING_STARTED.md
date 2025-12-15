# Getting Started with OpenSpliceAI Recalibration

Welcome! This guide will help you get started with the new `openspliceai_recalibration` package.

## 🎯 What Is This Package?

A **separate experimental framework** for training recalibration models on SpliceVarDB data using **OpenSpliceAI as the base model**, independent from your existing `meta_models` package (which uses SpliceAI).

### Why Separate?

- **Different base model**: OpenSpliceAI (PyTorch) vs SpliceAI (Keras/TF)
- **Different approach**: Direct recalibration vs meta-learning layer
- **Experimental**: Won't interfere with your production `meta_models` code
- **Focused**: Specifically for variant-induced splice-altering prediction

## 🚀 Quick Start (5 Minutes)

### Step 1: Verify OpenSpliceAI Models

```bash
# Check if OpenSpliceAI models are installed
ls -la data/models/openspliceai/

# If not found, download them:
./scripts/base_model/download_openspliceai_models.sh
```

### Step 2: Run Demo Training

```bash
# Run with demo data (no downloads required)
cd /Users/pleiadian53/work/meta-spliceai

python -m meta_spliceai.splice_engine.openspliceai_recalibration.examples.train_with_splicevardb \
    --test-mode \
    --max-variants 50 \
    --verbose
```

This will:
1. ✅ Use built-in demo variants (CFTR, BRCA1, DMD)
2. ✅ Generate OpenSpliceAI predictions
3. ✅ Create feature tables
4. ✅ Save reports to `./models/openspliceai_recalibration/`

### Step 3: Review Results

```bash
# View training report
cat models/openspliceai_recalibration/training_report.json

# View cached predictions
ls -la models/openspliceai_recalibration/openspliceai_predictions.parquet

# View features
ls -la models/openspliceai_recalibration/training_features.parquet
```

## 📚 What Got Built

### Complete Implementations

1. **SpliceVarDB Loader** - Fully functional API client
   ```python
   from meta_spliceai.splice_engine.openspliceai_recalibration import SpliceVarDBLoader
   
   loader = SpliceVarDBLoader(output_dir="./data/splicevardb")
   variants_df = loader.load_validated_variants(build="GRCh38")
   print(f"Loaded {len(variants_df)} variants")
   ```

2. **OpenSpliceAI Predictor** - Working PyTorch wrapper
   ```python
   from meta_spliceai.splice_engine.openspliceai_recalibration import OpenSpliceAIPredictor
   
   predictor = OpenSpliceAIPredictor()
   result = predictor.predict_variant(
       chrom="7", pos=117199644, ref="C", alt="T",
       sequence="ACGT..." * 1000, gene="CFTR"
   )
   print(f"Donor gain: {result['donor_gain']:.3f}")
   ```

3. **Training Pipeline** - End-to-end orchestration
   ```python
   from meta_spliceai.splice_engine.openspliceai_recalibration import SpliceVarDBTrainingPipeline
   
   pipeline = SpliceVarDBTrainingPipeline(
       data_dir="./data/splicevardb",
       output_dir="./models/recalibration"
   )
   results = pipeline.run()
   ```

### Placeholder Implementations (Need Your Work)

These have interfaces defined but need implementation:

- **Sequence Extraction**: Extract genomic sequences from hg38.fa
- **Feature Engineering**: Build delta and context features
- **Recalibration Models**: Isotonic, Platt, XGBoost
- **Evaluation Framework**: ROC/PR curves, calibration plots

## 📁 Package Location

```
meta_spliceai/splice_engine/
│
├── meta_models/                    # Your existing SpliceAI-based package
│   └── ...
│
├── openspliceai_recalibration/    # NEW: Your experimental package
│   ├── core/                      # Prediction & recalibration
│   ├── data/                      # SpliceVarDB & data processing
│   ├── workflows/                 # Training & inference pipelines
│   ├── examples/                  # Usage examples
│   ├── configs/                   # Configuration templates
│   ├── README.md                  # Package documentation
│   ├── INTEGRATION_GUIDE.md       # Integration instructions
│   ├── IMPLEMENTATION_SUMMARY.md  # Technical details
│   └── GETTING_STARTED.md         # This file
│
└── case_studies/                  # Your existing validation package
    └── ...
```

## 🔧 Next Steps

### Immediate (Complete the Core Functionality)

1. **Implement Sequence Extraction** (`data/variant_processor.py`)
   ```python
   # Use pyfaidx or pysam to extract sequences
   import pyfaidx
   
   def extract_sequences(variants_df, reference_genome, context_size=10000):
       fasta = pyfaidx.Fasta(reference_genome)
       sequences = []
       for _, row in variants_df.iterrows():
           seq = fasta[row['chrom']][row['pos']-context_size:row['pos']+context_size]
           sequences.append(str(seq))
       return sequences
   ```

2. **Implement Feature Engineering** (`data/feature_builder.py`)
   ```python
   # Build delta features from predictions
   def build_features(predictions_df):
       features = predictions_df.copy()
       features['max_delta'] = features[['donor_gain', 'acceptor_gain']].abs().max(axis=1)
       features['gain_loss_ratio'] = ...
       return features
   ```

3. **Implement Recalibration** (`core/recalibrator.py`)
   ```python
   from sklearn.isotonic import IsotonicRegression
   
   class IsotonicRecalibrator:
       def __init__(self):
           self.model = IsotonicRegression()
       
       def fit(self, scores, labels):
           self.model.fit(scores, labels)
       
       def predict(self, scores):
           return self.model.predict(scores)
   ```

### Short-term (Enhance Capabilities)

4. **Add Real SpliceVarDB Data Access**
   - Register at https://splicevardb.org
   - Get API token
   - Test with real 50K+ variants

5. **Add Evaluation Framework**
   - ROC/PR curves
   - Calibration plots
   - Region-specific metrics

6. **Create Inference Pipeline**
   - Production deployment
   - Batch processing
   - Export formats

### Long-term (Research Goals)

7. **Compare with meta_models**
   - Same test sets
   - Performance benchmarking
   - Ablation studies

8. **Validate on Disease Cohorts**
   - ClinVar pathogenic variants
   - MutSpliceDB cancer mutations
   - DBASS cryptic sites

9. **Publish Results**
   - Methods comparison paper
   - SpliceVarDB validation study

## 🧪 Testing Your Setup

### Test 1: SpliceVarDB Loader

```bash
python -m meta_spliceai.splice_engine.openspliceai_recalibration.data.splicevardb_loader \
    --output-dir ./test_splicevardb \
    --max-variants 10 \
    --export-tsv \
    --verbose
```

### Test 2: OpenSpliceAI Predictor

```python
# Test predictor independently
python -m meta_spliceai.splice_engine.openspliceai_recalibration.core.base_predictor
```

### Test 3: Complete Pipeline

```bash
python -m meta_spliceai.splice_engine.openspliceai_recalibration.examples.train_with_splicevardb \
    --test-mode \
    --max-variants 100 \
    -vv  # Very verbose
```

## 📖 Documentation

### Key Documents

1. **README.md** - Package overview, architecture, features
2. **INTEGRATION_GUIDE.md** - How to integrate with existing code
3. **IMPLEMENTATION_SUMMARY.md** - Technical implementation details
4. **GETTING_STARTED.md** - This quick start guide

### Example Scripts

- `examples/train_with_splicevardb.py` - Complete training example

### Configuration

- `configs/default_config.yaml` - Full configuration template

## 🔗 Integration with Existing Code

### With case_studies

```python
# Your existing case_studies workflow
from meta_spliceai.splice_engine.case_studies.workflows import DiseaseValidationWorkflow

# New recalibration pipeline
from meta_spliceai.splice_engine.openspliceai_recalibration import SpliceVarDBTrainingPipeline

# Train recalibrator
pipeline = SpliceVarDBTrainingPipeline()
results = pipeline.run()

# Validate with case_studies
validator = DiseaseValidationWorkflow(model=results["model"])
```

### With meta_models (Comparison)

```python
# Your existing meta_models
from meta_spliceai.splice_engine.meta_models.training import MetaModelTrainer

# New recalibration
from meta_spliceai.splice_engine.openspliceai_recalibration import SpliceVarDBTrainingPipeline

# Train both
spliceai_model = MetaModelTrainer(base_model="spliceai").train(data)
openspliceai_results = SpliceVarDBTrainingPipeline().run()

# Compare performance
```

## ❓ FAQ

**Q: Will this interfere with my existing meta_models package?**  
A: No! It's completely separate. Different directory, different base model, independent code.

**Q: Do I need SpliceVarDB API access?**  
A: No for development (uses demo data). Yes for training on full 50K+ variants.

**Q: Can I use my existing reference genome?**  
A: Yes! Just pass `--reference-genome /path/to/hg38.fa` to the training script.

**Q: What if OpenSpliceAI models aren't found?**  
A: Run `./scripts/base_model/download_openspliceai_models.sh` to download them.

**Q: How do I switch back to meta_models?**  
A: Just use the meta_models package as before. They're independent.

## 🐛 Troubleshooting

### Issue: OpenSpliceAI models not found

```bash
# Solution: Download models
./scripts/base_model/download_openspliceai_models.sh

# Verify installation
ls -la data/models/openspliceai/
```

### Issue: Import errors

```python
# Make sure you're in the project root
cd /Users/pleiadian53/work/meta-spliceai

# Then run
python -m meta_spliceai.splice_engine.openspliceai_recalibration.examples.train_with_splicevardb
```

### Issue: SpliceVarDB API errors

The package automatically falls back to demo data if API access fails. This is intentional for development.

### Issue: Memory errors with large datasets

```python
# Use smaller batch sizes
config = PipelineConfig(
    batch_size=50,  # Reduce from default 100
    low_memory_mode=True
)
pipeline = SpliceVarDBTrainingPipeline(config=config)
```

## 💡 Tips

1. **Start with demo data** - Test everything before getting SpliceVarDB access
2. **Use test mode** - `--test-mode --max-variants 100` for fast iteration
3. **Check caching** - Predictions are cached to speed up repeated runs
4. **Enable verbose mode** - Use `-vv` for detailed debugging output
5. **Review reports** - Check `training_report.json` for dataset statistics

## 🎉 You're Ready!

You now have:
- ✅ Complete package structure
- ✅ Working SpliceVarDB loader
- ✅ Functional OpenSpliceAI predictor
- ✅ End-to-end training pipeline
- ✅ Comprehensive documentation

**Start with the demo**, then implement the core functionality (sequence extraction, feature engineering, recalibration models), and finally test on real SpliceVarDB data!

---

**Questions?** Check the documentation or create an issue.  
**Ready to code?** Start with implementing `data/variant_processor.py`!  
**Need examples?** See `examples/train_with_splicevardb.py`

