# Base Layer Integration Guide

**Last Updated**: November 27, 2025  
**Purpose**: Guide for integrating the Meta-SpliceAI base prediction layer into other projects

---

## Overview

The Meta-SpliceAI **base layer** is a generalized, model-agnostic framework for:
1. Running splice site predictions with any compatible base model (SpliceAI, OpenSpliceAI, etc.)
2. Evaluating predictions against reference annotations
3. Extracting training features for downstream meta-learning
4. Managing artifacts with intelligent checkpointing

**Key Features**:
- ✅ **Model-agnostic**: Works with any model producing per-nucleotide splice scores
- ✅ **Memory-efficient**: Mini-batch processing for large-scale genome analysis
- ✅ **Robust checkpointing**: Chunk-level resumption
- ✅ **Schema standardization**: Handles GRCh37/GRCh38, GTF/GFF3, Ensembl/RefSeq
- ✅ **Production-ready**: CLI interface and artifact management

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Entry Points](#entry-points)
4. [Integration Scenarios](#integration-scenarios)
5. [API Reference](#api-reference)
6. [Custom Model Integration](#custom-model-integration)
7. [Data Requirements](#data-requirements)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### For End Users (CLI)

**Run base model predictions** on a genome:

```bash
# Activate environment
mamba activate metaspliceai

# Run full genome pass with OpenSpliceAI
run_base_model \
    --base-model openspliceai \
    --mode production \
    --coverage full_genome \
    --verbosity 1

# Run on specific chromosomes
run_base_model \
    --base-model spliceai \
    --mode production \
    --coverage full_genome \
    --chromosomes 21,22 \
    --verbosity 1

# Run in test mode (small subset)
run_base_model \
    --base-model openspliceai \
    --mode test \
    --coverage sample \
    --verbosity 2
```

### For Developers (Python API)

**Use the base layer** in your Python code:

```python
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)

# Configure the workflow
config = SpliceAIConfig(
    base_model='openspliceai',
    genome_fasta='data/mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna',
    gtf_file='data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gff',
    eval_dir='data/mane/GRCh38/openspliceai_eval',
    mode='production',
    coverage='full_genome',
    do_extract_annotations=False,  # Use existing
    do_extract_sequences=False,    # Use existing
    mini_batch_size=50             # Memory-efficient processing
)

# Run the workflow
results = run_enhanced_splice_prediction_workflow(
    config=config,
    target_chromosomes=['1', '2'],  # Optional: specific chromosomes
    verbosity=1
)

# Access results
print(f"Processed genes: {results['manifest_summary']['processed_genes']}")
print(f"Artifacts: {results['paths']['artifacts_dir']}")
```

---

## Architecture Overview

### Core Components

```
meta_spliceai/splice_engine/
├── meta_models/
│   ├── core/
│   │   ├── data_types.py              # SpliceAIConfig, GeneManifest
│   │   ├── enhanced_workflow.py       # Enhanced evaluation with all scores
│   │   └── artifact_manager.py        # Production/test artifact management
│   │
│   ├── workflows/
│   │   ├── splice_prediction_workflow.py  # ⭐ MAIN ENTRY POINT
│   │   ├── data_preparation.py        # Data loading and standardization
│   │   └── sequence_data_utils.py     # Sequence extraction
│   │
│   ├── io/
│   │   └── handlers.py                # MetaModelDataHandler (artifact I/O)
│   │
│   └── utils/
│       ├── model_utils.py             # Base model loading
│       ├── sequence_utils.py          # Genomic sequence utilities
│       └── infer_splice_site_adjustments.py  # Position adjustments
│
├── run_spliceai_workflow.py          # Base prediction engine
├── evaluate_models.py                 # Evaluation logic
└── cli/
    └── run_base_model.py              # ⭐ CLI ENTRY POINT
```

### Data Flow

```
1. Configuration
   ↓
2. Data Preparation
   ├── Load annotations (GTF/GFF3)
   ├── Extract splice sites
   ├── Extract gene sequences
   └── Detect overlapping genes
   ↓
3. Model Loading
   └── Load base model ensemble (SpliceAI/OpenSpliceAI/Custom)
   ↓
4. Chromosome Iteration
   ↓
5. Chunk Processing (500 genes/chunk)
   ├── Mini-batch prediction (50 genes/mini-batch)
   ├── Evaluation (TP/FP/FN/TN)
   ├── Sequence extraction (±250nt windows)
   └── Save artifacts (chunk-level)
   ↓
6. Aggregation (optional)
   └── Combine all chunks into full genome files
```

---

## Entry Points

### 1. CLI Entry Point (Recommended for End Users)

**File**: `meta_spliceai/splice_engine/cli/run_base_model.py`

**Purpose**: Command-line interface for running base model passes

**Usage**:
```bash
run_base_model --help
```

**Key Features**:
- Automatic data path resolution (GRCh37/GRCh38)
- Production/test mode switching
- Checkpoint-aware resumption
- Progress monitoring

**When to Use**:
- Running full genome passes
- Production deployments
- Automated pipelines
- Non-interactive workflows

---

### 2. Python API Entry Point (Recommended for Developers)

**File**: `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`

**Function**: `run_enhanced_splice_prediction_workflow()`

**Purpose**: Programmatic access to the base layer

**Key Features**:
- Full configuration control
- Target gene filtering
- In-memory result access
- Integration with custom workflows

**When to Use**:
- Custom analysis workflows
- Research prototyping
- Integration with other tools
- Interactive analysis

---

### 3. Shell Script Entry Point (Recommended for Sequential Processing)

**File**: `scripts/training/process_chromosomes_sequential_smart.sh`

**Purpose**: Orchestrate sequential chromosome processing with logging

**Usage**:
```bash
cd scripts/training/
./process_chromosomes_sequential_smart.sh
```

**Key Features**:
- Sequential chromosome processing
- Comprehensive logging
- Error recovery
- Resource monitoring

**When to Use**:
- Long-running genome passes
- Resource-constrained systems
- Background processing
- Production deployments

---

## Integration Scenarios

### Scenario 1: Add to Existing Python Project

**Goal**: Use Meta-SpliceAI predictions in your Python application

**Steps**:

1. **Install Meta-SpliceAI**:
```bash
pip install meta-spliceai
# OR clone and install from source
git clone https://github.com/yourusername/meta-spliceai.git
cd meta-spliceai
pip install -e .
```

2. **Prepare Data**:
```python
from meta_spliceai.system.genomic_resources import Registry

# Set up genomic resources
registry = Registry(build='GRCh38_MANE')
registry.setup_data_directory()  # Downloads and prepares data
```

3. **Run Predictions**:
```python
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)

# Run on specific genes
results = run_enhanced_splice_prediction_workflow(
    target_genes=['UNC13A', 'STMN2', 'TARDBP'],
    verbosity=1
)

# Access predictions
positions_df = results['positions']  # Polars DataFrame
print(positions_df.filter(pl.col('splice_type') == 'donor'))
```

4. **Use Results**:
```python
# Export to pandas for compatibility
positions_pd = positions_df.to_pandas()

# Filter high-confidence predictions
high_conf = positions_df.filter(pl.col('donor_score') > 0.9)

# Save to your preferred format
high_conf.write_csv('predictions.csv')
high_conf.write_parquet('predictions.parquet')
```

---

### Scenario 2: Integrate with AI Agent Workflow

**Goal**: Provide AI agent with a turnkey splice prediction system

**Recommended Approach**: Share the **CLI + minimal data setup guide**

**Quick Start for AI Agent**:

```markdown
# Quick Start for AI Agents

## 1. Environment Setup
```bash
mamba create -n spliceai python=3.10
mamba activate spliceai
pip install meta-spliceai
```

## 2. Data Setup (One-time)
```bash
# Download and prepare GRCh38 MANE data
python -m meta_spliceai.system.genomic_resources.setup --build GRCh38_MANE
```

## 3. Run Predictions
```bash
# Full genome
run_base_model --base-model openspliceai --mode production --coverage full_genome

# Specific genes
run_base_model --base-model openspliceai --mode test --coverage sample --genes UNC13A,STMN2
```

## 4. Access Results
```python
import polars as pl

# Load predictions
df = pl.read_csv('data/mane/GRCh38/openspliceai_eval/meta_models/full_splice_positions_enhanced.tsv', separator='\t')
print(df.head())
```

## 5. Key Files
- **Predictions**: `data/mane/GRCh38/openspliceai_eval/meta_models/analysis_sequences_*.tsv`
- **Errors**: `data/mane/GRCh38/openspliceai_eval/meta_models/error_analysis_*.tsv`
- **Manifest**: `data/mane/GRCh38/openspliceai_eval/meta_models/gene_manifest.tsv`

## 6. Troubleshooting
- Missing sequences? → Set `--do-extract-sequences`
- Out of memory? → Reduce `--mini-batch-size`
- Wrong build? → Check `--build` matches your data
```

---

### Scenario 3: Add Custom Base Model

**Goal**: Integrate your own splice prediction model into the framework

**Requirements**:
1. Model produces per-nucleotide scores: `(donor_prob, acceptor_prob, neither_prob)`
2. Model accepts DNA sequences as input
3. Model is pre-trained (weights available)

**Steps**:

1. **Create Model Wrapper**:

Create `meta_spliceai/splice_engine/models/my_custom_model.py`:

```python
import torch
from typing import Dict, List, Tuple

class MyCustomSpliceModel:
    """Wrapper for custom splice prediction model."""
    
    def __init__(self, model_path: str, context: int = 10000):
        """
        Parameters
        ----------
        model_path : str
            Path to model weights
        context : int
            Context window size (default: 10000)
        """
        self.context = context
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load your model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint."""
        # YOUR MODEL LOADING CODE HERE
        # Example:
        from my_model_package import MySpliceModel
        model = MySpliceModel.from_pretrained(model_path)
        return model
    
    def predict(self, sequence: str) -> Dict[str, List[float]]:
        """
        Predict splice sites for a DNA sequence.
        
        Parameters
        ----------
        sequence : str
            DNA sequence (ACGT)
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'donor_prob': List of donor probabilities (one per nucleotide)
            - 'acceptor_prob': List of acceptor probabilities
            - 'neither_prob': List of neither probabilities (optional)
            - 'positions': List of genomic positions (1-indexed)
        """
        # YOUR PREDICTION CODE HERE
        # Must return per-nucleotide probabilities
        
        with torch.no_grad():
            # Convert sequence to tensor
            inputs = self._encode_sequence(sequence)
            inputs = inputs.to(self.device)
            
            # Get predictions
            outputs = self.model(inputs)
            
            # Extract probabilities (adjust based on your model output)
            donor_prob = outputs[:, 0].cpu().numpy().tolist()
            acceptor_prob = outputs[:, 1].cpu().numpy().tolist()
            neither_prob = outputs[:, 2].cpu().numpy().tolist()
        
        return {
            'donor_prob': donor_prob,
            'acceptor_prob': acceptor_prob,
            'neither_prob': neither_prob,
            'positions': list(range(1, len(donor_prob) + 1))
        }
    
    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode DNA sequence to tensor."""
        # YOUR ENCODING LOGIC HERE
        # Example one-hot encoding:
        encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        encoded = [encoding.get(base.upper(), 0) for base in sequence]
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
```

2. **Register Model**:

Update `meta_spliceai/splice_engine/meta_models/utils/model_utils.py`:

```python
def load_base_model_ensemble(
    base_model: str,
    context: int = 10000,
    verbosity: int = 1
) -> Tuple[List, Dict]:
    """
    Load base model ensemble.
    
    Parameters
    ----------
    base_model : str
        Model name: 'spliceai', 'openspliceai', 'my_custom_model'
    ...
    """
    if base_model == 'my_custom_model':
        from meta_spliceai.splice_engine.models.my_custom_model import MyCustomSpliceModel
        
        model = MyCustomSpliceModel(
            model_path='data/models/my_custom_model/weights.pt',
            context=context
        )
        
        metadata = {
            'base_model': 'my_custom_model',
            'genome_build': 'GRCh38',  # Adjust as needed
            'framework': 'pytorch',
            'context': context
        }
        
        return [model], metadata
    
    elif base_model == 'spliceai':
        # ... existing SpliceAI code ...
    # ... rest of function ...
```

3. **Use Custom Model**:

```bash
run_base_model --base-model my_custom_model --mode test --coverage sample
```

---

## API Reference

### Core Configuration: `SpliceAIConfig`

```python
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig

config = SpliceAIConfig(
    # Model selection
    base_model='openspliceai',  # 'spliceai', 'openspliceai', or custom
    
    # Data paths
    genome_fasta='path/to/genome.fa',
    gtf_file='path/to/annotations.gtf',
    eval_dir='path/to/output',
    
    # Processing mode
    mode='production',  # 'production' or 'test'
    coverage='full_genome',  # 'full_genome' or 'sample'
    
    # Data preparation flags
    do_extract_annotations=False,  # Extract from GTF/GFF3
    do_extract_sequences=False,    # Extract gene sequences
    do_extract_splice_sites=False, # Extract splice sites
    
    # Performance tuning
    mini_batch_size=50,  # Genes per mini-batch (memory control)
    chunk_size=500,      # Genes per chunk (artifact granularity)
    
    # Evaluation parameters
    threshold=0.1,       # Minimum probability threshold
    consensus_window=10, # Window for consensus scoring
    error_window=10,     # Window for error tolerance
    
    # Chromosome selection
    chromosomes=['1', '2'],  # Specific chromosomes (optional)
    test_mode=False,         # Test mode (small subset)
    
    # Output control
    save_nucleotide_scores=False,  # Save full nucleotide scores (large!)
    no_final_aggregate=False        # Skip final aggregation (memory)
)
```

### Main Workflow Function

```python
def run_enhanced_splice_prediction_workflow(
    config: Optional[SpliceAIConfig] = None,
    target_genes: Optional[List[str]] = None,
    target_chromosomes: Optional[List[str]] = None,
    verbosity: int = 1,
    no_final_aggregate: bool = False,
    no_tn_sampling: bool = False,
    position_id_mode: str = 'genomic',
    **kwargs
) -> Dict[str, pl.DataFrame]:
    """
    Run the enhanced SpliceAI prediction workflow.
    
    Parameters
    ----------
    config : SpliceAIConfig, optional
        Configuration object. If None, uses defaults from kwargs.
    target_genes : List[str], optional
        Gene IDs or symbols to process. If None, processes all genes.
    target_chromosomes : List[str], optional
        Chromosomes to process. If None, processes all chromosomes.
    verbosity : int, default=1
        0 = minimal, 1 = normal, 2 = detailed
    no_final_aggregate : bool, default=False
        Skip memory-heavy final aggregation
    no_tn_sampling : bool, default=False
        Preserve all true negative positions (memory-heavy)
    position_id_mode : str, default='genomic'
        Position identification strategy
    **kwargs
        Additional config parameters
    
    Returns
    -------
    dict
        Results dictionary containing:
        - 'success': bool
        - 'error_analysis': pl.DataFrame (TP/FP/FN classifications)
        - 'positions': pl.DataFrame (splice site predictions)
        - 'analysis_sequences': pl.DataFrame (±250nt windows)
        - 'gene_manifest': pl.DataFrame (processing metadata)
        - 'nucleotide_scores': pl.DataFrame (full scores, if enabled)
        - 'paths': dict (artifact paths)
        - 'artifact_manager': dict (artifact management info)
        - 'manifest_summary': dict (processing statistics)
    """
```

---

## Data Requirements

### Minimum Requirements

To use the base layer, you need:

1. **Reference Genome** (FASTA):
   - GRCh37: `Homo_sapiens.GRCh37.dna.primary_assembly.fa`
   - GRCh38: `GCF_000001405.40_GRCh38.p14_genomic.fna`

2. **Gene Annotations** (GTF or GFF3):
   - GRCh37: Ensembl GTF (release 87)
   - GRCh38: MANE RefSeq GFF3 (v1.3)

3. **Base Model Weights**:
   - SpliceAI: 5 PyTorch models
   - OpenSpliceAI: 5 PyTorch models
   - Custom: Your model weights

### Directory Structure

The system expects:

```
data/
├── ensembl/GRCh37/              # For SpliceAI
│   ├── Homo_sapiens.GRCh37.87.gtf
│   ├── Homo_sapiens.GRCh37.dna.primary_assembly.fa
│   └── spliceai_eval/           # Output directory
│
├── mane/GRCh38/                 # For OpenSpliceAI
│   ├── MANE.GRCh38.v1.3.refseq_genomic.gff
│   ├── GCF_000001405.40_GRCh38.p14_genomic.fna
│   └── openspliceai_eval/       # Output directory
│
└── models/                      # Model weights
    ├── spliceai/
    │   ├── spliceai_1.pt
    │   ├── spliceai_2.pt
    │   └── ... (5 models)
    └── openspliceai/
        ├── openspliceai_1.pt
        └── ... (5 models)
```

### Automatic Data Setup

Use the genomic resources system:

```python
from meta_spliceai.system.genomic_resources import Registry

# GRCh38 MANE (for OpenSpliceAI)
registry = Registry(build='GRCh38_MANE')
registry.setup_data_directory()

# GRCh37 (for SpliceAI)
registry = Registry(build='GRCh37', release='87')
registry.setup_data_directory()
```

---

## Configuration

### Environment Variables

```bash
# Data directory (optional - default: ./data)
export METASPLICEAI_DATA_DIR=/path/to/data

# Logging (optional)
export TQDM_MININTERVAL=300  # Update progress every 5 min
export TQDM_MAXINTERVAL=600  # Force update every 10 min
```

### Configuration Files

Create `configs/my_config.yaml`:

```yaml
base_model: openspliceai
mode: production
coverage: full_genome

# Data paths (relative to METASPLICEAI_DATA_DIR)
genome_fasta: mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna
gtf_file: mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gff
eval_dir: mane/GRCh38/openspliceai_eval

# Performance
mini_batch_size: 50
chunk_size: 500

# Data preparation (set to false to use pre-extracted)
do_extract_annotations: false
do_extract_sequences: false
do_extract_splice_sites: false

# Evaluation
threshold: 0.1
consensus_window: 10
error_window: 10
```

Load config:

```python
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
import yaml

with open('configs/my_config.yaml') as f:
    config_dict = yaml.safe_load(f)

config = SpliceAIConfig(**config_dict)
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution 1**: Reduce mini-batch size
```python
config.mini_batch_size = 25  # Default: 50
```

**Solution 2**: Disable final aggregation
```bash
run_base_model --no-final-aggregate
```

**Solution 3**: Process fewer chromosomes at once
```bash
run_base_model --chromosomes 21,22
```

---

### Issue: Missing Sequences

**Error**: `FileNotFoundError: gene_sequence_1.parquet not found`

**Solution**: Enable sequence extraction
```bash
run_base_model --do-extract-sequences
```

Or in Python:
```python
config.do_extract_sequences = True
```

---

### Issue: Model Not Found

**Error**: `Base model 'my_model' not recognized`

**Solution**: Register custom model (see [Custom Model Integration](#custom-model-integration))

---

### Issue: Checkpoint Not Resuming

**Symptom**: Re-processes already completed chromosomes

**Solution**: Check chunk artifacts exist in correct directory
```bash
ls data/mane/GRCh38/openspliceai_eval/meta_models/analysis_sequences_*_chunk_*.tsv
```

Verify script uses chunk-level checkpointing:
```bash
grep "checkpoint" scripts/training/process_chromosomes_sequential_smart.sh
```

---

## Best Practices

### 1. Start Small

Always test with a small subset first:

```bash
# Test mode
run_base_model --mode test --coverage sample --verbosity 2

# Single chromosome
run_base_model --chromosomes 21 --verbosity 2
```

### 2. Use Pre-extracted Data

For repeated runs, extract data once and reuse:

```bash
# First run: extract everything
run_base_model --do-extract-annotations --do-extract-sequences --do-extract-splice-sites

# Subsequent runs: reuse
run_base_model  # No extraction flags needed
```

### 3. Monitor Memory

Use resource monitoring:

```bash
# In separate terminal
watch -n 5 'ps aux | grep run_base_model | grep -v grep'
```

Or enable memory profiling:
```python
config = SpliceAIConfig(verbosity=2)  # Verbose includes memory stats
```

### 4. Checkpoint Frequently

Use chunk-level artifacts for checkpointing:
- Default chunk size: 500 genes
- Each chunk saved separately
- Resume automatically from last completed chunk

### 5. Version Control Artifacts

Use artifact manager's mode system:
```bash
# Production: never overwrite
run_base_model --mode production

# Test: overwrite freely
run_base_model --mode test
```

---

## Summary: Recommended Entry Points

| Use Case | Entry Point | When to Use |
|----------|-------------|-------------|
| **Production genome pass** | `run_base_model` CLI | Full genome, automated pipelines |
| **Custom Python workflow** | `run_enhanced_splice_prediction_workflow()` | Integration, custom analysis |
| **Sequential processing** | `process_chromosomes_sequential_smart.sh` | Long-running, resource-constrained |
| **Interactive analysis** | Python API + Jupyter | Exploration, visualization |
| **AI agent integration** | CLI + minimal setup guide | Turnkey solution for agents |
| **Custom model** | Model wrapper + registration | New splice prediction models |

---

## Additional Resources

### For Developers
- **Data Organization**: `docs/data/DATA_LAYOUT_MASTER_GUIDE.md`
- **Base Model Comparison**: `docs/base_models/BASE_MODEL_DATA_MAPPING.md`
- **Schema Standardization**: `meta_spliceai/system/genomic_resources/docs/SCHEMA_STANDARDIZATION.md`
- **Memory Optimization**: `dev/MEMORY_OPTIMIZATION_COMPLETE.md`
- **Checkpointing**: `dev/CHUNK_LEVEL_CHECKPOINTING.md`

### For AI Agents
- **AI Agent Porting Guide**: `docs/base_models/AI_AGENT_PORTING_GUIDE.md` - Systematic stage-by-stage instructions
- **AI Agent Prompts**: `docs/base_models/AI_AGENT_PROMPTS.md` - Copy-paste prompts for instructing AI assistants

---

**Questions?** Open an issue or consult the documentation in `docs/`.


