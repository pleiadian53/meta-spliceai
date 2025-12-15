# AI Agent Porting Guide: Meta-SpliceAI Base Layer

**Last Updated**: November 27, 2025  
**Purpose**: Step-by-step instructions for AI agents to port the base prediction layer to a new project  
**Audience**: AI coding assistants, autonomous agents, code analysis systems

---

## Overview for AI Agents

This guide provides **explicit, stage-by-stage instructions** for analyzing and porting the Meta-SpliceAI base layer. The base layer is a model-agnostic framework that can run splice site predictions with any compatible model.

**What you'll learn**:
1. How to trace the codebase from entry points to core logic
2. Which files are essential vs. optional
3. How the genomic resources system standardizes data layout
4. How to verify each stage of the port

**Estimated Steps**: 6 stages, ~50 files to analyze

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Stage 1: Understand Entry Points](#stage-1-understand-entry-points)
3. [Stage 2: Trace Core Workflow](#stage-2-trace-core-workflow)
4. [Stage 3: Map Data Dependencies](#stage-3-map-data-dependencies)
5. [Stage 4: Understand Genomic Resources System](#stage-4-understand-genomic-resources-system)
6. [Stage 5: Identify Essential vs. Optional Components](#stage-5-identify-essential-vs-optional-components)
7. [Stage 6: Create Minimal Port](#stage-6-create-minimal-port)
8. [Verification Checklist](#verification-checklist)
9. [Common Pitfalls](#common-pitfalls)

---

## Prerequisites

### What AI Agents Should Know

Before starting, ensure you have:
- ✅ Access to the Meta-SpliceAI repository
- ✅ Ability to read Python files and trace imports
- ✅ Understanding of Python dataclasses, type hints, and decorators
- ✅ Familiarity with Polars DataFrames (similar to pandas)
- ✅ Basic genomics knowledge (genes, chromosomes, splice sites)

### Key Terminology

| Term | Definition |
|------|------------|
| **Base Model** | Pre-trained model (SpliceAI, OpenSpliceAI) that produces per-nucleotide splice scores |
| **Splice Site** | Genomic position where splicing occurs (donor/acceptor) |
| **Chunk** | Group of 500 genes processed together for memory efficiency |
| **Mini-batch** | Group of 50 genes within a chunk for further memory optimization |
| **Artifact** | Output file (predictions, errors, sequences) |
| **GTF/GFF3** | Gene annotation file formats |

---

## Stage 1: Understand Entry Points

**Goal**: Identify the main entry points and understand their relationships

### Step 1.1: Analyze the User-Facing Entry Point

**File to read**: `meta_spliceai/run_base_model.py`

**What to look for**:
1. The main functions: `run_base_model_predictions()` and `predict_splice_sites()`
2. Input parameters: What does a user provide?
3. Configuration object: `BaseModelConfig` (alias for `SpliceAIConfig`)
4. The delegation: Where does it forward the request?

**Key observation**:
```python
# Line 36-37: The main delegation
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)
```

**Action**: Note that this is a **thin wrapper**. The real logic is in `splice_prediction_workflow.py`.

**Verification**: Can you answer these questions?
- What parameters does `run_base_model_predictions()` accept?
- What does it return?
- What is `BaseModelConfig` an alias for?

---

### Step 1.2: Analyze the CLI Entry Point

**File to read**: `meta_spliceai/splice_engine/cli/run_base_model.py`

**What to look for**:
1. How command-line arguments map to Python parameters
2. What defaults are used
3. How it calls the underlying workflow

**Pattern to recognize**:
```
CLI args → Parse → Create config → Call run_enhanced_splice_prediction_workflow()
```

**Action**: Identify that **both entry points** (Python API and CLI) converge to the same core workflow.

---

### Step 1.3: Map the Entry Point Hierarchy

**Create this mental map**:

```
User Entry Points:
├── meta_spliceai/run_base_model.py
│   └── run_base_model_predictions()
│       └── BaseModelConfig (wrapper)
│           └── SpliceAIConfig (actual class)
│
├── meta_spliceai/splice_engine/cli/run_base_model.py
│   └── main() [CLI entry]
│       └── argparse → dict → SpliceAIConfig
│
└── scripts/training/process_chromosomes_sequential_smart.sh
    └── Sequential chromosome orchestration
        └── Calls CLI: run_base_model

All converge to:
    └── run_enhanced_splice_prediction_workflow()
```

**Verification**: Can you trace a call from CLI to core workflow?

---

## Stage 2: Trace Core Workflow

**Goal**: Understand the main prediction workflow and its dependencies

### Step 2.1: Analyze the Core Workflow

**File to read**: `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`

**This is the MOST IMPORTANT file. Read it in sections.**

**Section 1: Imports (Lines 1-72)**

**Action**: List all imported modules. Categorize them:

| Category | Example Modules | Purpose |
|----------|----------------|---------|
| **Core Workflow** | `run_spliceai_workflow`, `evaluate_models` | Base prediction engine |
| **Data Preparation** | `data_preparation` module | Load GTF, extract sequences |
| **Data Types** | `SpliceAIConfig`, `GeneManifest` | Configuration and tracking |
| **I/O Handlers** | `MetaModelDataHandler` | Save/load artifacts |
| **Model Utils** | `model_utils`, `sequence_utils` | Load models, process sequences |
| **Enhancement** | `enhanced_process_predictions_with_all_scores` | Extended evaluation |

**Verification**: Can you categorize each import?

---

### Step 2.2: Trace the Workflow Function

**Function to analyze**: `run_enhanced_splice_prediction_workflow()` (lines 74-1202)

**Break it into logical sections**:

#### Section A: Configuration (Lines 74-210)

**What happens**:
- Accept config or create from kwargs
- Extract parameters (gtf_file, genome_fasta, eval_dir, etc.)
- Set up artifact manager
- Set up data handlers

**Key classes to understand**:
- `SpliceAIConfig`: Configuration dataclass
- `MetaModelDataHandler`: Artifact I/O handler
- `ArtifactManager`: Production/test mode artifact management

**Action**: Identify all configuration parameters and their defaults.

---

#### Section B: Data Preparation (Lines 216-450)

**What happens** (in order):
1. **Prepare gene annotations** (line 217-228): Extract from GTF/GFF3
2. **Derive gene features** (line 229-274): Gene types, lengths, biotypes
3. **Prepare splice sites** (line 276-308): Extract known splice sites
4. **Standardize schema** (line 299-308): Ensure consistent column names
5. **Prepare sequences** (line 317-336): Extract gene sequences from FASTA
6. **Handle overlapping genes** (line 339-348): Detect gene overlaps
7. **Determine chromosomes** (line 350-361): Decide which chromosomes to process
8. **Load models** (line 363-370): Load base model ensemble
9. **Prepare adjustments** (line 373-450): Calculate position adjustments

**Key functions to trace**:

| Function | File | Purpose |
|----------|------|---------|
| `prepare_gene_annotations()` | `data_preparation.py` | Load and filter GTF/GFF3 |
| `prepare_splice_site_annotations()` | `data_preparation.py` | Extract splice sites |
| `prepare_genomic_sequences()` | `data_preparation.py` | Extract gene sequences |
| `handle_overlapping_genes()` | `data_preparation.py` | Find overlaps |
| `load_spliceai_models()` | `data_preparation.py` | Load base model |
| `standardize_splice_sites_schema()` | `genomic_resources/schema.py` | Standardize columns |

**Action**: For each function, note its inputs and outputs.

---

#### Section C: Main Processing Loop (Lines 500-1005)

**What happens**:
- **Outer loop**: Iterate over chromosomes
- **Middle loop**: Iterate over 500-gene chunks
- **Inner loop**: Process 50-gene mini-batches

**The nested structure**:

```python
for chromosome in chromosomes:
    # Load sequences for this chromosome
    
    for chunk_start in range(0, n_genes, 500):  # 500-gene chunks
        # Checkpoint: Skip if chunk already processed
        
        for mini_batch_idx in range(0, 500, 50):  # 50-gene mini-batches
            # 1. Predict splice sites
            # 2. Evaluate predictions (TP/FP/FN/TN)
            # 3. Extract sequences (±250nt windows)
            # 4. Accumulate results
            # 5. Free memory (gc.collect)
        
        # Consolidate mini-batches into chunk
        # Save chunk artifacts
        # Free chunk memory
```

**Key pattern**: **Mini-batching for memory efficiency**

**Action**: Identify where predictions, evaluation, and sequence extraction happen.

---

#### Section D: Aggregation and Save (Lines 1018-1202)

**What happens**:
- Optionally aggregate all chunks into full genome files
- Save gene manifest (processing metadata)
- Return results dictionary

**Key insight**: Aggregation is **optional** (can be disabled with `--no-final-aggregate`).

---

### Step 2.3: Identify Critical Dependencies

**From the workflow analysis, you need**:

1. **Prediction Engine**: `run_spliceai_workflow.predict_splice_sites_for_genes()`
2. **Evaluation**: `enhanced_process_predictions_with_all_scores()`
3. **Sequence Extraction**: `extract_analysis_sequences()`
4. **Data Preparation**: All functions in `data_preparation.py`
5. **Model Loading**: `load_base_model_ensemble()`
6. **Data Handlers**: `MetaModelDataHandler`, `ModelEvaluationFileHandler`
7. **Schema Standardization**: `standardize_splice_sites_schema()`, `standardize_gene_features_schema()`

**Verification**: Can you draw a dependency graph of these functions?

---

## Stage 3: Map Data Dependencies

**Goal**: Understand what data files are needed and where they come from

### Step 3.1: Identify Required Data Files

**From the workflow, you need**:

| Data Type | File Format | Example | Purpose |
|-----------|-------------|---------|---------|
| **Reference Genome** | FASTA (.fa/.fna) | `GRCh38.fna` | Extract gene sequences |
| **Gene Annotations** | GTF or GFF3 | `MANE.v1.3.gff` | Define genes and transcripts |
| **Base Model Weights** | PyTorch (.pt) | `openspliceai_1.pt` | Load pre-trained model |
| **Splice Sites** (derived) | TSV | `splice_sites_enhanced.tsv` | Reference for evaluation |
| **Gene Sequences** (derived) | Parquet | `gene_sequence_1.parquet` | Pre-extracted for speed |

**Action**: List all file paths from the config and trace where they're used.

---

### Step 3.2: Understand Data Flow

**Data preparation creates derived files**:

```
Input:
├── genome.fa (reference FASTA)
└── annotations.gtf (gene definitions)

Processing:
├── Extract splice sites → splice_sites_enhanced.tsv
├── Extract sequences → gene_sequence_*.parquet
└── Create database → annotations.db

Usage:
├── splice_sites_enhanced.tsv → Evaluation (TP/FP/FN)
├── gene_sequence_*.parquet → Prediction input
└── annotations.db → Gene metadata queries
```

**Key insight**: Many files can be **reused** across runs (set `do_extract_*=False`).

**Verification**: Can you identify which files are inputs vs. derived?

---

## Stage 4: Understand Genomic Resources System

**Goal**: Learn how Meta-SpliceAI standardizes data layout and schema

### Step 4.1: Read the Genomic Resources Overview

**Files to read** (in order):

1. **`meta_spliceai/system/genomic_resources/README.md`**
   - Overview of the system
   - Directory conventions
   - Build naming

2. **`meta_spliceai/system/genomic_resources/__init__.py`**
   - Public API
   - Key functions: `standardize_splice_sites_schema()`, `Registry`

3. **`meta_spliceai/system/genomic_resources/registry.py`**
   - `Registry` class: Path resolution for different builds
   - How to map build names to data directories

**Key concept**: **Build-specific paths**

```python
from meta_spliceai.system.genomic_resources import Registry

# GRCh38 MANE (for OpenSpliceAI)
registry = Registry(build='GRCh38_MANE')
paths = registry.get_paths()
# Returns: {
#   'data_dir': 'data/mane/GRCh38/',
#   'genome_fasta': 'data/mane/GRCh38/...genomic.fna',
#   'gtf_file': 'data/mane/GRCh38/MANE...gff',
#   ...
# }

# GRCh37 (for SpliceAI)
registry = Registry(build='GRCh37', release='87')
paths = registry.get_paths()
# Returns: {
#   'data_dir': 'data/ensembl/GRCh37/',
#   'genome_fasta': 'data/ensembl/GRCh37/...assembly.fa',
#   'gtf_file': 'data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf',
#   ...
# }
```

**Action**: Understand how `Registry` resolves paths based on build.

---

### Step 4.2: Understand Schema Standardization

**File to read**: `meta_spliceai/system/genomic_resources/schema.py`

**Problem it solves**: Different data sources use different column names:
- Ensembl GTF: `seqname`, `gene_id`, `transcript_id`
- RefSeq GFF3: `seqid`, `gene`, `transcript`
- SpliceAI output: `site_type`
- OpenSpliceAI output: `splice_type`

**Solution**: Standardize to a canonical schema:

```python
from meta_spliceai.system.genomic_resources import standardize_splice_sites_schema

# Input: DataFrame with 'site_type' column
df_input = pl.DataFrame({'site_type': ['donor', 'acceptor']})

# Standardize: 'site_type' → 'splice_type'
df_standard = standardize_splice_sites_schema(df_input)
# Output: DataFrame with 'splice_type' column

# Now all downstream code uses 'splice_type' consistently
```

**Key functions**:
- `standardize_splice_sites_schema()`: Splice site DataFrames
- `standardize_gene_features_schema()`: Gene annotation DataFrames
- `get_standard_splice_sites_schema()`: Get canonical column list

**Action**: Identify all synonymous column names that are standardized.

---

### Step 4.3: Understand Directory Conventions

**File to read**: `docs/data/DATA_LAYOUT_MASTER_GUIDE.md`

**Standard layout**:

```
data/
├── <annotation_source>/     # "ensembl" or "mane"
│   └── <build>/             # "GRCh37" or "GRCh38"
│       ├── *.gtf or *.gff   # Gene annotations
│       ├── *.fa or *.fna    # Reference genome
│       │
│       ├── splice_sites_enhanced.tsv      # Derived: splice sites
│       ├── annotations.db                 # Derived: annotation DB
│       ├── gene_sequence_*.parquet        # Derived: sequences per chr
│       │
│       └── <base_model>_eval/             # "spliceai_eval" or "openspliceai_eval"
│           └── meta_models/               # Meta-model artifacts
│               ├── analysis_sequences_*_chunk_*.tsv
│               ├── error_analysis_*_chunk_*.tsv
│               ├── splice_positions_*_chunk_*.tsv
│               └── gene_manifest.tsv
│
└── models/                                # Model weights
    ├── spliceai/
    └── openspliceai/
```

**Key principle**: **Build-specific directories** prevent mixing incompatible data.

**Action**: Memorize this structure. All file paths follow this convention.

---

## Stage 5: Identify Essential vs. Optional Components

**Goal**: Determine the minimal set of files needed for a basic port

### Essential Components (Must Port)

**Core Workflow** (7 files):
1. `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`
   - Main workflow orchestration
2. `meta_spliceai/splice_engine/meta_models/workflows/data_preparation.py`
   - Data loading and preparation
3. `meta_spliceai/splice_engine/meta_models/workflows/sequence_data_utils.py`
   - Sequence extraction utilities
4. `meta_spliceai/splice_engine/run_spliceai_workflow.py`
   - Base prediction engine
5. `meta_spliceai/splice_engine/evaluate_models.py`
   - Evaluation logic (TP/FP/FN)
6. `meta_spliceai/splice_engine/meta_models/core/enhanced_workflow.py`
   - Enhanced evaluation with all scores
7. `meta_spliceai/run_base_model.py`
   - User-facing entry point

**Data Types** (2 files):
8. `meta_spliceai/splice_engine/meta_models/core/data_types.py`
   - `SpliceAIConfig`, `GeneManifest`
9. `meta_spliceai/splice_engine/meta_models/core/artifact_manager.py`
   - Production/test artifact management

**I/O Handlers** (2 files):
10. `meta_spliceai/splice_engine/meta_models/io/handlers.py`
    - `MetaModelDataHandler`
11. `meta_spliceai/splice_engine/model_evaluator.py`
    - `ModelEvaluationFileHandler`

**Model Loading** (1 file):
12. `meta_spliceai/splice_engine/meta_models/utils/model_utils.py`
    - `load_base_model_ensemble()`

**Genomic Resources** (5 files):
13. `meta_spliceai/system/genomic_resources/__init__.py`
    - Public API
14. `meta_spliceai/system/genomic_resources/registry.py`
    - Path resolution
15. `meta_spliceai/system/genomic_resources/schema.py`
    - Schema standardization
16. `meta_spliceai/system/genomic_resources/splice_sites.py`
    - Splice site extraction
17. `meta_spliceai/system/genomic_resources/derive.py`
    - Gene feature derivation

**Utilities** (3 files):
18. `meta_spliceai/splice_engine/analysis_utils.py`
    - Analysis utilities
19. `meta_spliceai/splice_engine/workflow_utils.py`
    - Workflow utilities
20. `meta_spliceai/splice_engine/utils_df.py`
    - DataFrame utilities

**Total Essential Files**: ~20 files

---

### Optional Components (Can Skip for Minimal Port)

**CLI** (1 file):
- `meta_spliceai/splice_engine/cli/run_base_model.py`
  - Only needed if you want CLI interface

**Position Adjustments** (1 file):
- `meta_spliceai/splice_engine/meta_models/utils/infer_splice_site_adjustments.py`
  - Only needed for auto-adjustment detection

**Sequence Utilities** (1 file):
- `meta_spliceai/splice_engine/meta_models/utils/sequence_utils.py`
  - Additional sequence utilities (mostly duplicates)

**Validators** (1 file):
- `meta_spliceai/system/genomic_resources/validators.py`
  - Data validation (useful but not essential)

**Shell Scripts** (all files in `scripts/`):
- Orchestration scripts
- Only needed for batch processing

---

## Stage 6: Create Minimal Port

**Goal**: Extract essential components into a new project

### Step 6.1: Set Up Project Structure

**Create this directory structure in your new project**:

```
my_project/
├── base_layer/                    # Ported base layer
│   ├── __init__.py
│   │
│   ├── core/                      # Core workflow
│   │   ├── workflow.py            # Main workflow
│   │   ├── data_prep.py           # Data preparation
│   │   ├── config.py              # Configuration
│   │   └── handlers.py            # I/O handlers
│   │
│   ├── genomic/                   # Genomic resources
│   │   ├── registry.py            # Path resolution
│   │   ├── schema.py              # Schema standardization
│   │   └── splice_sites.py        # Splice site extraction
│   │
│   ├── models/                    # Model integration
│   │   ├── loader.py              # Model loading
│   │   └── custom_model.py        # Your custom model (if any)
│   │
│   └── utils/                     # Utilities
│       ├── dataframe.py           # DataFrame utilities
│       └── file_io.py             # File I/O utilities
│
├── data/                          # Data directory
│   ├── mane/GRCh38/              # OpenSpliceAI data
│   └── models/                    # Model weights
│
└── run_predictions.py             # Entry point
```

---

### Step 6.2: Port Files with Adaptations

**For each essential file**:

1. **Copy the file** to your new project
2. **Update imports** to match your new structure
3. **Remove dependencies** on non-essential components
4. **Test the import** to verify no missing dependencies

**Example: Porting the main workflow**

Original:
```python
# meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py
from meta_spliceai.splice_engine.run_spliceai_workflow import predict_splice_sites_for_genes
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
```

Ported:
```python
# my_project/base_layer/core/workflow.py
from my_project.base_layer.core.prediction import predict_splice_sites_for_genes
from my_project.base_layer.core.config import SpliceAIConfig
```

**Action**: For each of the 20 essential files, copy and update imports.

---

### Step 6.3: Create Entry Point

**File to create**: `my_project/run_predictions.py`

```python
"""
Minimal entry point for base layer predictions.
"""
from my_project.base_layer.core.workflow import run_enhanced_splice_prediction_workflow
from my_project.base_layer.core.config import SpliceAIConfig

def run_predictions(genes=None, chromosomes=None):
    """Run predictions on specified genes or chromosomes."""
    config = SpliceAIConfig(
        base_model='openspliceai',
        mode='test',
        coverage='sample'
    )
    
    results = run_enhanced_splice_prediction_workflow(
        config=config,
        target_genes=genes,
        target_chromosomes=chromosomes,
        verbosity=1
    )
    
    return results

if __name__ == '__main__':
    # Test with a single gene
    results = run_predictions(genes=['UNC13A'])
    print(f"Processed: {results['manifest_summary']['processed_genes']} genes")
    print(f"Positions: {results['positions'].height}")
```

---

### Step 6.4: Verify the Port

**Test 1: Import Test**
```python
# Test all imports work
from my_project.base_layer.core.workflow import run_enhanced_splice_prediction_workflow
from my_project.base_layer.core.config import SpliceAIConfig
from my_project.base_layer.genomic.registry import Registry
print("✓ All imports successful")
```

**Test 2: Configuration Test**
```python
# Test configuration
config = SpliceAIConfig(base_model='openspliceai')
print(f"✓ Config created: {config.base_model}")
```

**Test 3: Registry Test**
```python
# Test path resolution
registry = Registry(build='GRCh38_MANE')
paths = registry.get_paths()
print(f"✓ Registry resolves paths: {paths['data_dir']}")
```

**Test 4: End-to-End Test**
```python
# Test full workflow on a single gene
results = run_predictions(genes=['UNC13A'])
assert results['success'] == True
assert results['positions'].height > 0
print("✓ End-to-end test passed")
```

---

## Verification Checklist

After completing all stages, verify:

- [ ] **Stage 1**: Can identify all entry points and their relationships
- [ ] **Stage 2**: Can trace the workflow from entry to prediction to output
- [ ] **Stage 3**: Can list all required data files and their purpose
- [ ] **Stage 4**: Can explain the genomic resources system and schema standardization
- [ ] **Stage 5**: Can distinguish essential from optional components
- [ ] **Stage 6**: Can run a minimal port end-to-end

**Self-test questions**:
1. What does `run_enhanced_splice_prediction_workflow()` return?
2. Where are splice sites extracted from?
3. What is the purpose of schema standardization?
4. What is the difference between a chunk and a mini-batch?
5. Why does the system use build-specific directories?

---

## Common Pitfalls

### Pitfall 1: Missing Schema Standardization

**Problem**: Ported code breaks because column names don't match

**Example**:
```python
# This fails if DataFrame has 'site_type' instead of 'splice_type'
df.filter(pl.col('splice_type') == 'donor')
```

**Solution**: Always call `standardize_splice_sites_schema()` after loading data

---

### Pitfall 2: Wrong Data Directory Structure

**Problem**: Files not found because path resolution fails

**Example**:
```python
# This fails if data is in wrong location
genome_fasta = 'data/GRCh38/genome.fa'  # ❌ Wrong
genome_fasta = 'data/mane/GRCh38/genome.fna'  # ✓ Correct
```

**Solution**: Use `Registry` for all path resolution

---

### Pitfall 3: Mixing Genome Builds

**Problem**: Using GRCh37 data with OpenSpliceAI (trained on GRCh38)

**Example**:
```python
# This gives poor results
config = SpliceAIConfig(
    base_model='openspliceai',  # Trained on GRCh38
    genome_fasta='data/ensembl/GRCh37/genome.fa'  # ❌ Wrong build!
)
```

**Solution**: Always match base model to correct build:
- SpliceAI → GRCh37 (Ensembl)
- OpenSpliceAI → GRCh38 (MANE)

---

### Pitfall 4: Memory Issues from Missing Mini-Batching

**Problem**: Processing 500 genes at once causes OOM

**Example**:
```python
# This can use 5-10 GB memory
predictions = predict_splice_sites_for_genes(chunk_500_genes)
```

**Solution**: Use mini-batching (already implemented in workflow)

---

### Pitfall 5: Ignoring Checkpointing

**Problem**: Re-processing completed chunks after interruption

**Solution**: The workflow checks for existing chunk artifacts automatically. Don't disable this feature.

---

## Configuration Architecture: Design Decision Explained

### Why Inheritance Instead of Alias?

**Historical Context**: Originally, the codebase had:
```python
BaseModelConfig = SpliceAIConfig  # Just an alias
```

**Problem**: This is confusing and not extensible:
- `SpliceAIConfig` used for OpenSpliceAI (naming mismatch)
- Can't add model-specific parameters easily
- Misleading for AI agents and developers

**Solution**: Proper inheritance hierarchy:
```python
BaseModelConfig (abstract)  # Real base class
├── SpliceAIConfig         # SpliceAI-specific
└── OpenSpliceAIConfig     # OpenSpliceAI-specific
```

### How It Works Now

**Abstract Base Class**:
```python
@dataclass
class BaseModelConfig(ABC):
    # Common parameters (all models need these)
    genome_fasta: str
    gtf_file: str
    eval_dir: str
    mode: str = 'test'
    # ...
    
    @property
    @abstractmethod
    def base_model(self) -> str:
        """Each subclass returns its model identifier."""
        pass
    
    @abstractmethod
    def get_model_specific_params(self) -> Dict[str, Any]:
        """Each subclass returns its specific parameters."""
        pass
```

**Model-Specific Configs**:
```python
@dataclass
class SpliceAIConfig(BaseModelConfig):
    threshold: float = 0.5
    consensus_window: int = 2
    # SpliceAI-specific parameters
    
    @property
    def base_model(self) -> str:
        return 'spliceai'

@dataclass
class OpenSpliceAIConfig(BaseModelConfig):
    threshold: float = 0.5
    consensus_window: int = 2
    # OpenSpliceAI-specific parameters
    
    @property
    def base_model(self) -> str:
        return 'openspliceai'
```

**Factory Pattern**:
```python
def create_model_config(base_model: str, **kwargs) -> BaseModelConfig:
    if base_model == 'spliceai':
        return SpliceAIConfig(**kwargs)
    elif base_model == 'openspliceai':
        return OpenSpliceAIConfig(**kwargs)
    else:
        return _load_custom_model_config(base_model, **kwargs)
```

### Benefits

✅ **Clear naming**: `SpliceAIConfig` for SpliceAI, `OpenSpliceAIConfig` for OpenSpliceAI  
✅ **Extensible**: Easy to add new model configs with unique parameters  
✅ **Type-safe**: IDEs autocomplete model-specific parameters correctly  
✅ **Self-documenting**: Config class name indicates which model  
✅ **Backward compatible**: Old code still works during migration  

### Extensibility Example

Adding a transformer-based model with unique parameters:

```python
@dataclass
class TransformerSpliceConfig(BaseModelConfig):
    # Common parameters inherited
    # Model-specific parameters
    num_heads: int = 8
    num_layers: int = 12
    dropout: float = 0.1
    attention_window: int = 512
    
    @property
    def base_model(self) -> str:
        return 'transformer_splice'
    
    def get_model_specific_params(self) -> Dict[str, Any]:
        return {
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'attention_window': self.attention_window
        }
```

Usage:
```python
# Factory automatically creates correct config
config = create_model_config(
    base_model='transformer_splice',
    num_heads=16,
    num_layers=24
)
```

### Migration Status

**Current State** (as of Nov 2025):
- ✅ Model config classes implemented (`model_config.py`)
- ✅ Tests written and passing
- ⏳ Integration with existing code (backward compatible)
- ⏳ Documentation updated

**Backward Compatibility**:
All existing code continues to work. No breaking changes.

**References**:
- **Refactoring Plan**: `dev/CONFIG_REFACTORING_PLAN.md`
- **Implementation**: `meta_spliceai/splice_engine/meta_models/core/model_config.py`
- **Tests**: `tests/test_config_refactoring.py`

---

## Advanced Topics (Optional)

### Adding a Custom Model

After completing the basic port, you can add custom models:

1. **Create model wrapper** (see `BASE_LAYER_INTEGRATION_GUIDE.md` Section "Custom Model Integration")
2. **Create config class** inheriting from `BaseModelConfig`
3. **Register in `model_utils.py`**
4. **Use immediately** via factory

**Example**:
```python
# Step 1: Create config (in meta_spliceai/splice_engine/models/my_model/config.py)
@dataclass
class MyModelConfig(BaseModelConfig):
    my_param: float = 1.0
    
    @property
    def base_model(self) -> str:
        return 'my_model'
    
    def get_model_specific_params(self) -> Dict[str, Any]:
        return {'my_param': self.my_param}

# Step 2: Use via factory
config = create_model_config(base_model='my_model', my_param=2.0)
```

### Extending Evaluation

To add custom evaluation metrics:

1. Modify `enhanced_process_predictions_with_all_scores()` in `enhanced_workflow.py`
2. Add your metric calculation
3. Include in returned DataFrames

### Custom Feature Extraction

To add custom features for meta-learning:

1. Modify `extract_analysis_sequences()` in `sequence_data_utils.py`
2. Add your feature extraction logic
3. Include in `analysis_sequences` DataFrame

---

## Summary: AI Agent Workflow

**The complete porting process**:

```
1. Start → Read entry points (run_base_model.py, CLI)
           ↓
2. Trace → Follow to core workflow (splice_prediction_workflow.py)
           ↓
3. Map → Identify data dependencies (GTF, FASTA, model weights)
           ↓
4. Learn → Understand genomic resources system (Registry, schema)
           ↓
5. Filter → Identify essential vs optional (20 core files)
           ↓
6. Port → Copy, adapt imports, test
           ↓
7. Verify → Run end-to-end test
           ↓
8. Done → Minimal base layer working in new project
```

**Estimated time**: 2-4 hours for a complete port

---

## Getting Help

If you encounter issues:

1. **Check verification checkpoints** after each stage
2. **Review common pitfalls** section
3. **Consult related documentation**:
   - `BASE_LAYER_INTEGRATION_GUIDE.md` - User-facing integration
   - `DATA_LAYOUT_MASTER_GUIDE.md` - Data organization
   - `genomic_resources/README.md` - Genomic resources system
4. **Test incrementally** - don't port everything at once

---

**For AI Agents**: This guide provides a systematic approach to understanding and porting the base layer. Follow the stages in order, verify at each checkpoint, and you'll have a working port.

**Questions at each stage are intentional** - they help verify understanding before proceeding.

