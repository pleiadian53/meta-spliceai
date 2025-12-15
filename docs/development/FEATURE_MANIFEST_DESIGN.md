## Feature Manifest Design: Single Enriched Manifest

### Executive Summary

**Recommendation: Use a single, enriched manifest that documents both raw and encoded schemas.**

This design provides the best balance of:
- ✅ **Simplicity**: One manifest file (with optional legacy format for compatibility)
- ✅ **Completeness**: Documents encoding transformations
- ✅ **Reproducibility**: Full encoding specifications included
- ✅ **Backward Compatibility**: Can generate legacy format (feature names only)
- ✅ **Debugging Support**: Rich metadata for troubleshooting

---

## Design Options Considered

### ❌ Option 1: Two Separate Manifests (NOT RECOMMENDED)

**Approach**: Maintain two manifests:
1. `feature_manifest_raw.csv` - Pre-encoding feature schema
2. `feature_manifest_encoded.csv` - Post-encoding feature schema

**Pros**:
- Clear separation of concerns
- Easy to see "before" and "after"

**Cons**:
- ❌ **Complexity**: Two files to maintain and keep synchronized
- ❌ **Confusion**: Which one to use for inference?
- ❌ **Redundancy**: Most features appear in both files
- ❌ **Maintenance burden**: Must update both when adding features
- ❌ **Schema drift risk**: Files can get out of sync

**Verdict**: **TOO COMPLICATED** - the cons outweigh the pros.

---

### ✅ Option 2: Single Enriched Manifest (RECOMMENDED)

**Approach**: One manifest with rich metadata documenting encoding transformations.

**Structure**:
```
feature_manifest.json       # Complete manifest (primary)
feature_manifest.csv         # Enriched CSV view
feature_manifest_legacy.csv  # Legacy format (backward compatibility)
```

**Pros**:
- ✅ **Single source of truth**: One file to maintain
- ✅ **Complete information**: Documents both raw and encoded schemas
- ✅ **Backward compatible**: Can generate legacy format
- ✅ **Debugging friendly**: Rich metadata for troubleshooting
- ✅ **Extensible**: Easy to add new metadata fields
- ✅ **Machine-readable**: JSON format for programmatic use
- ✅ **Human-readable**: CSV format for quick inspection

**Cons**:
- Slightly more complex structure (but manageable)
- Larger file size (but negligible for typical datasets)

**Verdict**: **RECOMMENDED** - provides the best balance.

---

## Implementation Details

### 1. Manifest Structure

#### **`feature_manifest.json`** (Primary Format)

```json
{
  "features": [
    {
      "name": "chrom",
      "original_name": "chrom",
      "dtype": "Int64",
      "feature_type": "categorical",
      "is_encoded": true,
      "encoding_method": "custom",
      "encoding_mapping": {
        "1": 1,
        "chr1": 1,
        "X": 23,
        "chrX": 23,
        "Y": 24,
        "MT": 25,
        "GL000225.1": 100
      },
      "statistics": {
        "count": 99470,
        "unique_count": 54,
        "min": 1,
        "max": 125,
        "mean": 12.4
      },
      "description": "Chromosome identifier with biological ordering"
    },
    {
      "name": "AAA",
      "dtype": "Int64",
      "feature_type": "kmer",
      "is_encoded": false,
      "statistics": {
        "count": 99470,
        "unique_count": 42,
        "min": 0,
        "max": 87,
        "mean": 5.2
      },
      "description": "K-mer count for AAA motif"
    },
    {
      "name": "donor_score",
      "dtype": "Float64",
      "feature_type": "probability",
      "is_encoded": false,
      "statistics": {
        "count": 99470,
        "min": 0.0,
        "max": 1.0,
        "mean": 0.045
      },
      "description": "SpliceAI donor site probability"
    }
  ],
  "metadata": {
    "creation_date": "2025-10-27T14:30:00",
    "total_features": 121,
    "encoded_features": 1,
    "kmer_features": 64,
    "numerical_features": 45,
    "probability_features": 11,
    "version": "1.0.0"
  },
  "encoding_specs": {
    "chrom": {
      "encoding_type": "custom",
      "handle_unknown": "use_default",
      "default_value": 100,
      "description": "Chromosome identifier with biological ordering"
    }
  },
  "excluded_features": [
    "splice_type",
    "pred_type",
    "true_position",
    "predicted_position",
    "position",
    "gene_id",
    "transcript_id",
    "strand",
    "sequence"
  ]
}
```

#### **`feature_manifest.csv`** (Enriched CSV View)

```csv
feature,dtype,feature_type,is_encoded,original_name,encoding_method,description
chrom,Int64,categorical,True,chrom,custom,Chromosome identifier
AAA,Int64,kmer,False,,,K-mer count for AAA motif
ACG,Int64,kmer,False,,,K-mer count for ACG motif
donor_score,Float64,probability,False,,,SpliceAI donor probability
acceptor_score,Float64,probability,False,,,SpliceAI acceptor probability
...
```

#### **`feature_manifest_legacy.csv`** (Legacy Format)

```csv
feature
chrom
AAA
ACG
donor_score
acceptor_score
...
```

---

### 2. Key Features

#### **Feature Type Classification**

Each feature is automatically classified as:
- `kmer`: K-mer count features (e.g., `AAA`, `ACG`)
- `probability`: Probability-derived features (e.g., `donor_score`, `entropy`)
- `context`: Context score features (e.g., `context_score_p1`)
- `genomic`: Genomic coordinate features (e.g., `gene_start`, `gene_length`)
- `categorical`: Categorical features (e.g., `chrom`)
- `numerical`: Generic numerical features

#### **Encoding Documentation**

For encoded features, the manifest documents:
- `is_encoded`: Boolean flag
- `encoding_method`: Type of encoding ('ordinal', 'custom', 'onehot')
- `encoding_mapping`: Complete mapping (original → encoded values)
- `original_name`: Original feature name (if different)

#### **Statistics**

Basic statistics for each feature:
- Count, null count, unique count
- Min, max, mean, std (for numerical features)
- Percentiles (25th, 50th, 75th)

#### **Metadata**

Global metadata:
- Creation timestamp
- Feature counts by type
- Version information
- Encoding specifications used

---

### 3. Usage Examples

#### **Creating a Manifest (Training)**

```python
from meta_spliceai.splice_engine.meta_models.builder.feature_manifest_utils import (
    save_feature_manifests
)

# After preparing training data
X_df, y = prepare_training_data(df, encode_chrom=True)

# Save manifests
save_feature_manifests(
    df=X_df,
    output_dir="results/my_model",
    excluded_features=LEAKAGE_COLUMNS + METADATA_COLUMNS,
    include_statistics=True,
    save_legacy_csv=True  # For backward compatibility
)
```

**Output**:
```
📄 Feature manifests saved to results/my_model/
   - feature_manifest.json (complete metadata)
   - feature_manifest.csv (enriched view)
   - feature_manifest_legacy.csv (legacy format)
```

#### **Loading a Manifest (Inference)**

```python
from meta_spliceai.splice_engine.meta_models.builder.feature_manifest_utils import (
    FeatureManifest
)

# Load manifest
manifest = FeatureManifest.from_json("results/my_model/feature_manifest.json")

# Get feature names (for model input)
feature_names = manifest.get_feature_names()

# Get encoded features (to apply same encoding)
encoded_features = manifest.get_encoded_features()
for feature in encoded_features:
    print(f"Feature: {feature.name}")
    print(f"  Encoding: {feature.encoding_method}")
    print(f"  Mapping: {feature.encoding_mapping}")
```

#### **Comparing Manifests**

```python
from meta_spliceai.splice_engine.meta_models.builder.feature_manifest_utils import (
    FeatureManifest,
    compare_manifests
)

# Load training and inference manifests
train_manifest = FeatureManifest.from_json("results/training/feature_manifest.json")
infer_manifest = FeatureManifest.from_json("results/inference/feature_manifest.json")

# Compare
comparison = compare_manifests(train_manifest, infer_manifest)

print(f"Matching features: {len(comparison['matching_features'])}")
print(f"Missing in inference: {comparison['only_in_first']}")
print(f"Extra in inference: {comparison['only_in_second']}")
print(f"Type mismatches: {len(comparison['type_mismatches'])}")
```

---

### 4. Integration with Existing Code

#### **Update Training Scripts**

```python
# In training_strategies.py

from meta_spliceai.splice_engine.meta_models.builder.feature_manifest_utils import (
    save_feature_manifests
)

# Replace this:
# feature_manifest = pd.DataFrame({'feature': feature_names})
# feature_manifest.to_csv(out_dir / "feature_manifest.csv", index=False)

# With this:
save_feature_manifests(
    df=X_df,  # Polars or pandas DataFrame
    output_dir=out_dir,
    excluded_features=excluded_features,
    include_statistics=True,
    save_legacy_csv=True
)
```

#### **Update Inference Scripts**

```python
# In enhanced_selective_inference.py

from meta_spliceai.splice_engine.meta_models.builder.feature_manifest_utils import (
    FeatureManifest
)

# Load manifest to get encoding specs
manifest_path = model_dir / "feature_manifest.json"
if manifest_path.exists():
    manifest = FeatureManifest.from_json(manifest_path)
    
    # Get encoded features
    for feature_info in manifest.get_encoded_features():
        # Apply same encoding as training
        print(f"Encoding {feature_info.name} using {feature_info.encoding_method}")
else:
    # Fallback to legacy format
    legacy_path = model_dir / "feature_manifest.csv"
    # ... handle legacy format
```

---

### 5. Migration Strategy

#### **Phase 1: Add New Format (Backward Compatible)**

1. ✅ Implement `feature_manifest_utils.py`
2. ✅ Update training scripts to generate both formats:
   - `feature_manifest.json` (new)
   - `feature_manifest.csv` (enriched)
   - `feature_manifest_legacy.csv` (old format)
3. ✅ Update inference scripts to prefer new format, fallback to legacy

#### **Phase 2: Adopt New Format**

1. Use new format for all new training runs
2. Document benefits and usage
3. Update existing models' manifests (optional)

#### **Phase 3: Deprecate Legacy (Future)**

1. Remove `feature_manifest_legacy.csv` generation
2. Keep only `feature_manifest.json` and `feature_manifest.csv`

---

### 6. Benefits Over Two-Manifest Design

| Aspect | Two Manifests | Single Enriched Manifest |
|--------|---------------|-------------------------|
| **Simplicity** | ❌ Two files to maintain | ✅ One file (+ optional views) |
| **Completeness** | ⚠️ Information split across files | ✅ All info in one place |
| **Sync Risk** | ❌ Files can drift | ✅ Single source of truth |
| **Debugging** | ❌ Must cross-reference files | ✅ All info together |
| **Extensibility** | ❌ Must update both | ✅ Update one manifest |
| **Backward Compat** | ❌ Breaking change | ✅ Legacy format supported |
| **Machine Readable** | ⚠️ Must parse both | ✅ JSON primary format |
| **Human Readable** | ✅ CSV is readable | ✅ CSV view available |

---

### 7. Addressing Your Concerns

#### **"Would two versions be too complicated?"**

**Yes.** Here's why:

1. **Maintenance Burden**: Every feature addition/removal requires updating both manifests
2. **Synchronization Risk**: Files can get out of sync, causing hard-to-debug errors
3. **User Confusion**: Which manifest should inference use?
4. **Redundancy**: Most features appear identically in both files
5. **Implementation Cost**: More code to maintain two formats

#### **"Should we document raw vs encoded features?"**

**Yes, absolutely!** But in a **single enriched manifest**, not separate files.

**Benefits of Single Enriched Manifest**:
- ✅ All information in one place
- ✅ Explicit `is_encoded` flag per feature
- ✅ Encoding mapping documented inline
- ✅ Statistics for both raw and encoded values
- ✅ No synchronization issues

---

### 8. Real-World Example

#### **Training Output**

```bash
results/meta_model_1000genes_3mers_fresh/
├── model_multiclass.pkl
├── feature_manifest.json          # ← Complete manifest (PRIMARY)
├── feature_manifest.csv            # ← Enriched CSV view
├── feature_manifest_legacy.csv     # ← Legacy format (optional)
├── global_excluded_features.txt
└── model_metadata.json
```

#### **Inference Usage**

```python
# Load model
model = joblib.load("results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl")

# Load manifest
manifest = FeatureManifest.from_json(
    "results/meta_model_1000genes_3mers_fresh/feature_manifest.json"
)

# Check encoding for chrom
chrom_info = next(f for f in manifest.features if f.name == 'chrom')
print(f"Chrom is encoded: {chrom_info.is_encoded}")
print(f"Encoding method: {chrom_info.encoding_method}")
print(f"Mapping: {chrom_info.encoding_mapping}")

# Apply same encoding to inference data
if chrom_info.is_encoded:
    inference_df = encode_categorical_features(
        inference_df,
        features_to_encode=['chrom'],
        verbose=True
    )
```

---

## Conclusion

**Recommendation: Implement a single, enriched manifest design.**

### Summary

- ✅ **One manifest to rule them all**: `feature_manifest.json`
- ✅ **Backward compatible**: Generate legacy CSV if needed
- ✅ **Complete documentation**: Raw and encoded schemas in one file
- ✅ **Easy to use**: Clear APIs for reading and comparing manifests
- ✅ **Debugging friendly**: Rich metadata for troubleshooting
- ❌ **NOT two manifests**: Too complex, too much synchronization risk

### Implementation Status

- ✅ `feature_manifest_utils.py` implemented
- ✅ Comprehensive manifest creation and loading
- ✅ Backward compatibility with legacy format
- ✅ Manifest comparison utilities
- ⏳ Integration with training scripts (next step)
- ⏳ Integration with inference scripts (next step)

### Next Steps

1. Update `training_strategies.py` to use new manifest utilities
2. Update `enhanced_selective_inference.py` to load enriched manifests
3. Add tests for manifest creation and comparison
4. Document migration path for existing models
5. (Optional) Create conversion script for legacy manifests

**The single enriched manifest design provides the best balance of simplicity, completeness, and maintainability.**

