# Systematic Resource Management Integration for Training Workflows

## 🎯 **Problem Analysis**

The `run_gene_cv_sigmoid.py` script contained **4 hardcoded paths** that should be managed systematically:

1. **`--splice-sites-path`**: `"data/ensembl/splice_sites.tsv"`
2. **`--transcript-features-path`**: `"data/ensembl/spliceai_analysis/transcript_features.tsv"`
3. **`--gene-features-path`**: `"data/ensembl/spliceai_analysis/gene_features.tsv"`
4. **`--exclude-features`**: `"configs/exclude_features.txt"`

These hardcoded paths prevent consistent resource management across workflows and don't integrate with the existing `meta_spliceai.system.genomic_resources` system.

## 🔧 **Solution: Systematic Resource Management**

### **1. Created Training Resource Manager**

**File**: `meta_spliceai/splice_engine/meta_models/training/resource_manager.py`

**Key Features**:
- **Integrates with `genomic_resources`** for consistent path resolution
- **Multi-environment support** (Development, Production, Lakehouse)
- **Multi-genome support** (GRCh37, GRCh38) with automatic detection
- **Graceful fallback** to hardcoded paths if systematic management fails
- **Resource validation** before training workflows

**Usage**:
```python
from .resource_manager import create_training_resource_manager

manager = create_training_resource_manager()
defaults = manager.get_training_defaults()
# Returns systematic paths for all training resources
```

### **2. Created Systematic Defaults Provider**

**File**: `meta_spliceai/splice_engine/meta_models/training/systematic_defaults.py`

**Key Features**:
- **Drop-in replacement** for hardcoded defaults in argument parsers
- **Automatic integration** with `genomic_resources` system
- **Fallback mechanism** if systematic management unavailable
- **Validation utilities** for systematic paths

**Usage**:
```python
from .systematic_defaults import get_systematic_defaults

defaults = get_systematic_defaults()
p.add_argument("--splice-sites-path", default=defaults["splice_sites_path"])
```

### **3. Integration Example**

**File**: `meta_spliceai/splice_engine/meta_models/training/run_gene_cv_sigmoid_systematic.py`

**Demonstrates**:
- How to replace hardcoded defaults with systematic resource management
- Enhanced resource validation before training
- Graceful fallback mechanisms
- Backward compatibility preservation

## 📋 **Integration Pattern**

### **Before (Hardcoded)**:
```python
p.add_argument("--splice-sites-path", 
               default="data/ensembl/splice_sites.tsv",
               help="Path to splice site annotations file")
```

### **After (Systematic)**:
```python
from .systematic_defaults import get_systematic_defaults
defaults = get_systematic_defaults()

p.add_argument("--splice-sites-path", 
               default=defaults.get("splice_sites_path", "data/ensembl/splice_sites.tsv"),
               help="Path to splice site annotations file")
```

## 🔗 **Integration with genomic_resources**

The training resource manager leverages the existing `genomic_resources` system:

```python
from meta_spliceai.system.genomic_resources import create_systematic_manager
self.genomic_manager = create_systematic_manager(str(self.project_root))

# Systematic directory structure
self.directories = {
    "ensembl_base": self.genomic_manager.genome.get_source_dir("ensembl"),
    "analysis": self.genomic_manager.genome.get_source_dir("ensembl") / "spliceai_analysis",
    # ...
}
```

## 📊 **Systematic Path Resolution**

| Resource Type | Systematic Path | Source |
|---------------|-----------------|--------|
| **Splice Sites** | `data/ensembl/splice_sites.tsv` | GTF exon boundaries |
| **Gene Features** | `data/ensembl/spliceai_analysis/gene_features.tsv` | GTF analysis |
| **Transcript Features** | `data/ensembl/spliceai_analysis/transcript_features.tsv` | GTF analysis |
| **Feature Exclusion** | `configs/exclude_features.txt` | Training configuration |
| **GTF File** | Via `genomic_resources` | Systematic genome management |
| **FASTA File** | Via `genomic_resources` | Systematic genome management |

## ✅ **Benefits**

1. **Consistent Paths**: All workflows use the same systematic organization
2. **Multi-Environment**: Automatic adaptation to Development/Production/Lakehouse
3. **Multi-Genome**: GRCh37/GRCh38 support with automatic detection
4. **Validation**: Resource availability checked before training
5. **Backward Compatible**: Existing usage patterns continue to work
6. **Graceful Degradation**: Falls back to hardcoded paths if needed

## 🚀 **Implementation Steps**

### **Step 1: Add Resource Managers to Training Package**
```bash
# Copy the new resource management modules
cp resource_manager.py meta_spliceai/splice_engine/meta_models/training/
cp systematic_defaults.py meta_spliceai/splice_engine/meta_models/training/
```

### **Step 2: Update run_gene_cv_sigmoid.py**
Replace the argument parser section (lines ~150-158) with systematic defaults:

```python
# Replace hardcoded defaults
from .systematic_defaults import get_systematic_defaults
defaults = get_systematic_defaults()

p.add_argument("--splice-sites-path", 
               default=defaults.get("splice_sites_path", "data/ensembl/splice_sites.tsv"),
               help="Path to splice site annotations file")
# ... (similar for other paths)
```

### **Step 3: Add Resource Validation**
Add validation step in main function:

```python
from .systematic_defaults import validate_systematic_paths
path_validation = validate_systematic_paths()

missing_critical = [res for res in ["splice_sites", "gene_features", "transcript_features"] 
                   if not path_validation.get(res, False)]
if missing_critical:
    print(f"⚠️ Missing critical resources: {missing_critical}")
```

### **Step 4: Update Documentation**
- Replace hardcoded default values with "*(systematic path resolution)*"
- Add systematic resource management section
- Document integration benefits and usage patterns

## 🔍 **Validation and Testing**

### **Resource Validation**:
```bash
python -m meta_spliceai.splice_engine.meta_models.training.systematic_defaults --validate
```

### **Show Systematic Defaults**:
```bash
python -m meta_spliceai.splice_engine.meta_models.training.systematic_defaults --show-defaults
```

### **Training Resource Validation**:
```bash
python -m meta_spliceai.splice_engine.meta_models.training.resource_manager --validate
```

## 📝 **Documentation Updates**

Updated `gene_cv_sigmoid.md` to reflect:
- Systematic resource management integration
- Resource validation capabilities
- Manual override options
- Integration benefits and usage patterns

## 🎉 **Result**

The training workflows now integrate seamlessly with the `genomic_resources` system, providing:
- **Consistent path management** across all workflows
- **Multi-environment and multi-genome support**
- **Automatic resource validation**
- **Graceful fallback mechanisms**
- **Backward compatibility** with existing usage

This ensures that training workflows find genomic resources in consistent locations and integrate properly with the systematic data management approach used throughout the MetaSpliceAI system.





