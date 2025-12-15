# Feature Manifest Design: Visual Comparison

## Quick Decision Guide

**Question**: Should we have two feature manifests (raw + encoded) or one enriched manifest?

**Answer**: **One enriched manifest** (with optional legacy format for compatibility)

---

## Side-by-Side Comparison

### ❌ Option 1: Two Separate Manifests

```
results/my_model/
├── feature_manifest_raw.csv      # Pre-encoding schema
│   feature,dtype,type
│   chrom,string,categorical
│   AAA,int64,kmer
│   donor_score,float64,probability
│
├── feature_manifest_encoded.csv  # Post-encoding schema
│   feature,dtype,type
│   chrom,int64,categorical_encoded  ← Now numeric!
│   AAA,int64,kmer
│   donor_score,float64,probability
│
└── chrom_encoding.json           # Separate encoding map
    {"1": 1, "X": 23, "MT": 25, ...}
```

**Problems**:
- ❌ Which file for inference? (confusion)
- ❌ Must keep 2-3 files synchronized (error-prone)
- ❌ Encoding mapping separate from schema (fragmented)
- ❌ Must update both when adding features (maintenance burden)

---

### ✅ Option 2: Single Enriched Manifest (RECOMMENDED)

```
results/my_model/
├── feature_manifest.json         # Complete manifest (PRIMARY)
│   {
│     "features": [
│       {
│         "name": "chrom",
│         "dtype": "Int64",
│         "feature_type": "categorical",
│         "is_encoded": true,          ← Flag indicates encoding
│         "encoding_method": "custom",
│         "encoding_mapping": {        ← Mapping embedded
│           "1": 1, "X": 23, "MT": 25, ...
│         }
│       },
│       {
│         "name": "AAA",
│         "dtype": "Int64",
│         "feature_type": "kmer",
│         "is_encoded": false          ← Clearly not encoded
│       }
│     ],
│     "encoding_specs": {...},
│     "metadata": {...}
│   }
│
├── feature_manifest.csv          # CSV view (for humans)
│   feature,dtype,type,is_encoded,encoding_method
│   chrom,Int64,categorical,true,custom
│   AAA,Int64,kmer,false,
│   donor_score,Float64,probability,false,
│
└── feature_manifest_legacy.csv   # Legacy format (optional)
    feature
    chrom
    AAA
    donor_score
```

**Benefits**:
- ✅ Single source of truth (no synchronization issues)
- ✅ All information in one place (no fragmentation)
- ✅ Clear encoding status for each feature
- ✅ Backward compatible with legacy tools

---

## Practical Example: Chromosome Feature

### Two-Manifest Approach

**File 1: `feature_manifest_raw.csv`**
```csv
feature,dtype,type
chrom,string,categorical
```

**File 2: `feature_manifest_encoded.csv`**
```csv
feature,dtype,type
chrom,int64,categorical_encoded
```

**File 3: `chrom_encoding.json`**
```json
{
  "1": 1,
  "chr1": 1,
  "X": 23,
  "chrX": 23,
  "MT": 25
}
```

**Usage (Inference)**:
```python
# Which file to use? 🤔
# How to apply encoding? Must cross-reference 3 files! 😰

# Step 1: Load raw schema
raw_df = pd.read_csv("feature_manifest_raw.csv")

# Step 2: Load encoded schema
encoded_df = pd.read_csv("feature_manifest_encoded.csv")

# Step 3: Load encoding map
with open("chrom_encoding.json") as f:
    chrom_map = json.load(f)

# Step 4: Apply encoding manually
# ... complex code to match features and apply encoding
```

---

### Single-Manifest Approach

**File: `feature_manifest.json`**
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
        "1": 1, "chr1": 1,
        "X": 23, "chrX": 23,
        "MT": 25
      }
    }
  ]
}
```

**Usage (Inference)**:
```python
# Clear and simple! ✅

# Step 1: Load manifest
manifest = FeatureManifest.from_json("feature_manifest.json")

# Step 2: Apply encoding
for feature in manifest.get_encoded_features():
    # Encoding info is embedded!
    encode_categorical_features(
        df,
        features_to_encode=[feature.name],
        verbose=True
    )
```

---

## Decision Matrix

| Criterion | Two Manifests | Single Enriched |
|-----------|--------------|-----------------|
| **Files to maintain** | 2-3 | 1 (+ optional views) |
| **Synchronization risk** | ❌ High | ✅ None |
| **Information fragmentation** | ❌ Split across files | ✅ All in one place |
| **Inference clarity** | ❌ Ambiguous | ✅ Clear |
| **Debugging** | ❌ Must cross-reference | ✅ Self-contained |
| **Backward compatibility** | ❌ Breaking change | ✅ Legacy format supported |
| **Encoding documentation** | ⚠️ Separate file | ✅ Embedded |
| **Maintenance burden** | ❌ High | ✅ Low |
| **Learning curve** | ❌ Steeper | ✅ Intuitive |
| **Extensibility** | ❌ Update multiple files | ✅ Update one manifest |

**Score**: Two Manifests: 1/10 | Single Enriched: 10/10

---

## Real-World Scenarios

### Scenario 1: Adding a New Categorical Feature

**Two-Manifest Approach**:
```python
# Must update 3 files! 😰

# 1. Add to raw manifest
raw_manifest.append({
    'feature': 'gene_type',
    'dtype': 'string',
    'type': 'categorical'
})
raw_manifest.to_csv('feature_manifest_raw.csv')

# 2. Add to encoded manifest
encoded_manifest.append({
    'feature': 'gene_type',
    'dtype': 'int64',
    'type': 'categorical_encoded'
})
encoded_manifest.to_csv('feature_manifest_encoded.csv')

# 3. Create encoding map
with open('gene_type_encoding.json', 'w') as f:
    json.dump({'protein_coding': 1, 'lncRNA': 2, ...}, f)

# Risk: Forget to update one file → schema drift! 💥
```

**Single-Manifest Approach**:
```python
# Just update one manifest! ✅

manifest.features.append(FeatureInfo(
    name='gene_type',
    dtype='Int64',
    feature_type='categorical',
    is_encoded=True,
    encoding_method='custom',
    encoding_mapping={'protein_coding': 1, 'lncRNA': 2, ...}
))
manifest.to_json('feature_manifest.json')

# No synchronization issues! ✨
```

---

### Scenario 2: Debugging a Feature Mismatch

**Two-Manifest Approach**:
```python
# Complex debugging workflow 😰

# 1. Check raw manifest
print("Raw chrom type:", raw_manifest[raw_manifest['feature']=='chrom']['dtype'])
# Output: string

# 2. Check encoded manifest  
print("Encoded chrom type:", encoded_manifest[encoded_manifest['feature']=='chrom']['dtype'])
# Output: int64

# 3. Check encoding map
print("Chrom encoding:", chrom_encoding_map)
# Output: {'1': 1, 'X': 23, ...}

# 4. Manually verify encoding was applied
# ... complex verification code

# Question: Did we apply the encoding? 🤔
# Answer: Must check multiple files and inference code! 😰
```

**Single-Manifest Approach**:
```python
# Simple debugging! ✅

manifest = FeatureManifest.from_json('feature_manifest.json')
chrom_info = next(f for f in manifest.features if f.name == 'chrom')

# All info in one place!
print(f"Feature: {chrom_info.name}")
print(f"Type: {chrom_info.feature_type}")
print(f"Is encoded: {chrom_info.is_encoded}")
print(f"Encoding method: {chrom_info.encoding_method}")
print(f"Mapping: {chrom_info.encoding_mapping}")

# Clear answer immediately! ✨
```

---

## Migration Path

### Phase 1: Add New Format (Backward Compatible)

```python
# Training script outputs:
save_feature_manifests(
    df=X_df,
    output_dir=out_dir,
    save_legacy_csv=True  # ← Keep old format for compatibility
)

# Generates:
# - feature_manifest.json (new, enriched)
# - feature_manifest.csv (new, human-readable)
# - feature_manifest_legacy.csv (old format)
```

### Phase 2: Update Inference

```python
# Inference preferentially loads new format:
manifest_path = model_dir / "feature_manifest.json"
if manifest_path.exists():
    manifest = FeatureManifest.from_json(manifest_path)
else:
    # Fallback to legacy
    legacy_path = model_dir / "feature_manifest_legacy.csv"
    # ... load legacy format
```

### Phase 3: Deprecate Legacy (Future)

```python
# Eventually, only generate new format:
save_feature_manifests(
    df=X_df,
    output_dir=out_dir,
    save_legacy_csv=False  # ← No longer needed
)
```

---

## Conclusion

### **Recommendation: Single Enriched Manifest**

**Why?**
1. ✅ **Simplicity**: One file to maintain (vs 2-3)
2. ✅ **Clarity**: All information in one place
3. ✅ **Safety**: No synchronization risk
4. ✅ **Debugging**: Self-contained and complete
5. ✅ **Compatibility**: Supports legacy format
6. ✅ **Extensibility**: Easy to add metadata

**Two manifests would be:**
- ❌ More complex to maintain
- ❌ Error-prone (synchronization)
- ❌ Confusing for users
- ❌ Harder to debug
- ❌ More implementation work

**The choice is clear: Single enriched manifest wins by every metric.**

---

## Quick Reference

### Use Single Enriched Manifest When:
- ✅ You want simplicity
- ✅ You want completeness
- ✅ You want easy debugging
- ✅ You want backward compatibility
- ✅ **You want a production-ready solution**

### Use Two Manifests When:
- ❌ You enjoy complexity
- ❌ You want more files to maintain
- ❌ You want synchronization headaches
- ❌ **Never** (seriously, don't do this)

---

**Final Verdict**: **Single Enriched Manifest** is the clear winner. ✅

