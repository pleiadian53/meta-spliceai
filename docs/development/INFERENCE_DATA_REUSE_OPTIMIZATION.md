# Inference Workflow Data Reuse Optimization

## Problem

During inference testing, we observed this message:
```
Saving sequences by chromosome:
 ...
```

This indicates that **genomic datasets are being regenerated during inference**, which is inefficient and wasteful.

### Why This Is a Problem

1. **Performance**: Regenerating genomic datasets (gene sequences, splice sites, annotations) is **very slow**
2. **Disk Space**: Creates duplicate data in inference output directories
3. **Resource Waste**: These datasets were already created during the **base model pass** (training)
4. **Inference Goal**: The inference workflow should **make predictions**, not **recreate datasets**

---

## Root Cause

### Current Workflow Behavior

The `prepare_genomic_sequences()` function in `data_preparation.py` has this logic:

```python
def _process_gene_sequences(..., do_extract, force_overwrite, ...):
    # Check if files exist
    if do_extract and not force_overwrite:
        files_present = check_if_files_exist(...)  # Lines 77-99
        
        if files_present:
            print("[skip] Sequence files already present")
            do_extract = False  # ✅ Skip extraction
    
    # Extract sequences if still requested
    if do_extract:  # ❌ Problem: This still executes!
        gene_sequence_retrieval_workflow(...)  # Regenerates everything
```

### Why It's Being Called

The inference workflow calls:
```python
seq_result = prepare_genomic_sequences(
    local_dir=inference_output_dir,  # ❌ Wrong directory!
    gtf_file=gtf_file,
    genome_fasta=genome_fasta,
    do_extract=True,  # ❌ Always tries to extract
    ...
)
```

**Problem**: It's looking for files in the **inference output directory**, not the **shared data directory** where they already exist!

---

## Solution: Centralized Genomic Resources

### Design Principles

1. **Single Source of Truth**: All genomic datasets in `data/ensembl/`
2. **Reuse, Don't Recreate**: Inference loads from shared location
3. **Registry-Based Paths**: Use `Registry` to find datasets
4. **Explicit Control**: Clear flags for when to extract vs. load

### Implementation Strategy

#### **Option 1: Enhanced Registry Integration (RECOMMENDED)**

Modify the inference workflow to use the **centralized genomic resources** via `Registry`:

```python
# In enhanced_selective_inference.py

from meta_spliceai.system.genomic_resources import Registry

def _get_genomic_datasets(self):
    """Load genomic datasets from centralized location."""
    registry = Registry()
    
    # Check if datasets exist
    datasets_status = {
        'splice_sites': registry.resolve("splice_sites"),
        'gene_features': registry.resolve("gene_features"),
        'gene_sequences': registry.resolve("gene_sequences"),
    }
    
    self.logger.info("  Genomic dataset status:")
    for name, path in datasets_status.items():
        exists = "✅" if (path and Path(path).exists()) else "❌"
        self.logger.info(f"    {exists} {name}")
    
    # If datasets don't exist, create them ONCE in centralized location
    if not all(path and Path(path).exists() for path in datasets_status.values()):
        self.logger.warning("  ⚠️  Some genomic datasets missing - will create them")
        self._create_genomic_datasets()  # Creates in data/ensembl/
    else:
        self.logger.info("  ✅ Found existing genomic datasets - reusing them")
    
    return datasets_status
```

#### **Option 2: Add `use_existing_datasets` Flag**

Add a flag to control whether to create or reuse datasets:

```python
def prepare_genomic_sequences(
    local_dir: str,
    gtf_file: str,
    genome_fasta: str,
    do_extract: bool = True,
    force_overwrite: bool = False,
    use_existing_datasets: bool = True,  # ← NEW FLAG
    shared_data_dir: Optional[str] = None,  # ← NEW PARAMETER
    ...
):
    """
    Prepare genomic sequences.
    
    Parameters
    ----------
    use_existing_datasets : bool, default=True
        If True, look for existing datasets in shared_data_dir first
    shared_data_dir : str, optional
        Path to shared genomic data directory (e.g., data/ensembl/)
        If None, uses Registry to find it
    """
    
    # Try to use existing datasets first
    if use_existing_datasets:
        if shared_data_dir is None:
            # Use Registry to find shared location
            from meta_spliceai.system.genomic_resources import Registry
            registry = Registry()
            shared_data_dir = registry.resolve_base_dir("ensembl_data")
        
        # Check if files exist in shared location
        shared_files_exist = check_files_in_directory(shared_data_dir, ...)
        
        if shared_files_exist:
            print("[info] Using existing genomic datasets from:", shared_data_dir)
            return {
                'success': True,
                'sequences_file': shared_data_dir,
                'reused_existing': True
            }
    
    # Fall back to extraction/creation
    if do_extract:
        gene_sequence_retrieval_workflow(...)
```

#### **Option 3: Inference-Specific Configuration**

Create a separate configuration for inference that explicitly points to shared datasets:

```python
# In EnhancedSelectiveInferenceConfig

@dataclass
class EnhancedSelectiveInferenceConfig:
    """Configuration for inference workflow."""
    
    # Genomic data sources
    use_shared_genomic_data: bool = True
    shared_data_dir: Optional[Path] = None  # If None, use Registry
    
    # Extraction control
    allow_dataset_creation: bool = False  # Prevent accidental creation
    
    ...
```

---

## Recommended Implementation

### **Use Option 1: Registry Integration** ✅

This is the cleanest and most maintainable approach.

### **Step 1: Update Inference Workflow**

```python
# In enhanced_selective_inference.py

def _ensure_genomic_datasets(self):
    """Ensure genomic datasets are available, reusing existing ones."""
    
    from meta_spliceai.system.genomic_resources import Registry
    
    registry = Registry()
    
    # Check what's available
    self.logger.info("  Checking genomic dataset availability...")
    
    datasets = {
        'splice_sites': registry.resolve("splice_sites"),
        'gene_features': registry.resolve("gene_features"),
        'transcript_features': registry.resolve("transcript_features"),
        'exon_features': registry.resolve("exon_features"),
        'gene_sequences': registry.resolve("gene_sequences"),
    }
    
    # Log status
    all_present = True
    for name, path in datasets.items():
        if path and Path(path).exists():
            self.logger.info(f"    ✅ {name}")
        else:
            self.logger.warning(f"    ❌ {name} (missing)")
            all_present = False
    
    if not all_present:
        raise FileNotFoundError(
            "Some genomic datasets are missing. "
            "Please run the base model pass first to create them:\n"
            "  python scripts/run_base_model_pass.py"
        )
    
    self.logger.info("  ✅ All genomic datasets available - will reuse them")
    
    return datasets
```

### **Step 2: Skip Sequence Extraction During Inference**

```python
# In splice_prediction_workflow.py or enhanced_selective_inference.py

# During inference, set do_extract=False and provide shared data directory
seq_result = prepare_genomic_sequences(
    local_dir=shared_data_dir,  # ✅ Use shared location
    gtf_file=gtf_file,
    genome_fasta=genome_fasta,
    do_extract=False,  # ✅ Don't extract, just load
    force_overwrite=False,  # ✅ Never overwrite shared data
    use_existing_datasets=True,  # ✅ Explicit flag
    ...
)
```

### **Step 3: Add Safety Checks**

```python
def _validate_inference_mode(self):
    """Validate that inference is not creating new datasets."""
    
    # Ensure output directory != shared data directory
    if self.config.inference_base_dir == self.shared_data_dir:
        raise ValueError(
            "Inference output directory cannot be the same as shared data directory!"
        )
    
    # Ensure we're in read-only mode for shared data
    if self.config.allow_dataset_creation:
        self.logger.warning(
            "  ⚠️  Dataset creation is enabled during inference - this is unusual!"
        )
```

---

## Benefits

### **Performance**
- ⚡ **10-100x faster**: No need to re-extract sequences from FASTA
- 💾 **Reduced disk usage**: No duplicate datasets
- 🔄 **Immediate startup**: Datasets are already available

### **Correctness**
- ✅ **Consistency**: All workflows use the same datasets
- 🔒 **Immutability**: Shared datasets are read-only during inference
- 🎯 **Clear separation**: Training creates, inference uses

### **Maintainability**
- 📁 **Single location**: All genomic data in `data/ensembl/`
- 🗺️ **Registry-based**: Automatic path resolution
- 🧪 **Testability**: Easy to verify dataset reuse

---

## Testing

### **Test 1: Verify No Redundant Extraction**

```python
# Should NOT print "Saving sequences by chromosome"
workflow = EnhancedSelectiveInferenceWorkflow(config)
results = workflow.run()

# Verify no new sequence files in inference output directory
inference_seq_files = list(config.inference_base_dir.glob("gene_sequence_*.parquet"))
assert len(inference_seq_files) == 0, "Should not create sequence files during inference!"
```

### **Test 2: Verify Registry Integration**

```python
from meta_spliceai.system.genomic_resources import Registry

registry = Registry()

# Should find existing datasets
splice_sites = registry.resolve("splice_sites")
assert splice_sites and Path(splice_sites).exists()

gene_features = registry.resolve("gene_features")
assert gene_features and Path(gene_features).exists()
```

### **Test 3: Performance Comparison**

```bash
# Before optimization
time python scripts/run_inference.py --genes ENSG00000141736
# Expected: ~5-10 minutes (includes sequence extraction)

# After optimization
time python scripts/run_inference.py --genes ENSG00000141736
# Expected: ~30 seconds (loads existing sequences)
```

---

## Migration Plan

### **Phase 1: Add Registry Integration** (Immediate)

1. ✅ Update `enhanced_selective_inference.py` to use `Registry`
2. ✅ Add `_ensure_genomic_datasets()` method
3. ✅ Add safety checks to prevent accidental dataset creation
4. ✅ Test with existing inference workflow

### **Phase 2: Update Data Preparation Functions** (Short-term)

1. Add `use_existing_datasets` flag to `prepare_genomic_sequences()`
2. Add `shared_data_dir` parameter
3. Modify file existence checks to look in shared location first
4. Update documentation

### **Phase 3: Deprecate Local Dataset Creation** (Long-term)

1. Remove `do_extract` flag from inference configurations
2. Make all inference workflows read-only for genomic datasets
3. Add warnings if datasets don't exist in shared location
4. Update all documentation and examples

---

## Related Files

- `meta_spliceai/splice_engine/meta_models/workflows/data_preparation.py` - Dataset preparation functions
- `meta_spliceai/splice_engine/meta_models/workflows/sequence_data_utils.py` - Sequence extraction helpers
- `meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py` - Inference workflow
- `meta_spliceai/system/genomic_resources/Registry.py` - Path resolution
- `meta_spliceai/splice_engine/extract_genomic_features.py` - Sequence extraction (called during base model pass)

---

## Summary

**Problem**: Inference workflow was regenerating genomic datasets instead of reusing existing ones.

**Root Cause**: Looking for datasets in inference output directory instead of shared data directory.

**Solution**: Use `Registry` to find and reuse datasets from centralized location (`data/ensembl/`).

**Impact**: 
- ⚡ 10-100x faster inference startup
- 💾 Significantly reduced disk usage
- ✅ Consistent datasets across workflows
- 🎯 Clear separation between training (creates) and inference (uses)

**Next Steps**: Implement Registry integration in inference workflow (Option 1).

