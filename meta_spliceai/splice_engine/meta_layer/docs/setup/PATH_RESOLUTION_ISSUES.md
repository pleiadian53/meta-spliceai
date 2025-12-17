# Path Resolution Issues on RunPods

**Created**: December 17, 2025  
**Issue**: Project root resolves incorrectly on remote environments  
**Status**: Workaround documented

---

## Problem Summary

When running Meta-SpliceAI on RunPods, the project root may resolve to the **local development path** instead of the **remote workspace path**, causing `FileNotFoundError` for configuration files.

### Symptoms

```python
FileNotFoundError: Config file not found: /root/work/meta-spliceai/configs/genomic_resources.yaml
Please create configs/genomic_resources.yaml in the project root: /root/work/meta-spliceai
```

### Root Cause

The `meta_spliceai/system/config.py` uses path detection logic that may resolve to hardcoded or cached paths from the development environment. On local machines, projects are typically under `~/work/` (e.g., `/Users/username/work/meta-spliceai`), while on RunPods they are under `/workspace/`.

| Environment | Typical Project Root |
|-------------|---------------------|
| Local (Mac) | `~/work/meta-spliceai` or `/Users/<user>/work/meta-spliceai` |
| Local (Linux) | `~/work/meta-spliceai` or `/home/<user>/work/meta-spliceai` |
| **RunPods** | `/workspace/meta-spliceai` |
| Azure/Synapse | `/mnt/nfs1/meta-spliceai` |

---

## Workaround: Symlink

The simplest fix is to create a symlink that maps the expected path to the actual path:

```bash
# Create the expected directory structure
mkdir -p /root/work

# Create symlink from expected path to actual path
ln -sf /workspace/meta-spliceai /root/work/meta-spliceai

# Verify
ls -la /root/work/
# Should show: meta-spliceai -> /workspace/meta-spliceai
```

### Add to Shell Profile (Persistent)

To make this persistent across pod restarts, add to `~/.bashrc`:

```bash
echo '# Fix Meta-SpliceAI path resolution for RunPods
mkdir -p /root/work 2>/dev/null
[ ! -L /root/work/meta-spliceai ] && ln -sf /workspace/meta-spliceai /root/work/meta-spliceai 2>/dev/null
' >> ~/.bashrc
```

---

## Environment Variables

The system supports environment variable overrides for path configuration:

### Project Root Override

```bash
export META_SPLICEAI_ROOT=/workspace/meta-spliceai
```

### Genomic Resources Configuration

```bash
# Override species/build/release
export SS_SPECIES=homo_sapiens
export SS_BUILD=GRCh38
export SS_RELEASE=112

# Override data root directory
export SS_DATA_ROOT=/workspace/meta-spliceai/data

# Explicit file paths (highest priority)
export SS_GTF_PATH=/workspace/meta-spliceai/data/ensembl/GRCh38/Homo_sapiens.GRCh38.112.gtf
export SS_FASTA_PATH=/workspace/meta-spliceai/data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

### Recommended `.bashrc` for RunPods

```bash
# Meta-SpliceAI environment setup
export META_SPLICEAI_ROOT=/workspace/meta-spliceai
export SS_DATA_ROOT=/workspace/meta-spliceai/data

# Create symlink for backwards compatibility
mkdir -p /root/work 2>/dev/null
[ ! -L /root/work/meta-spliceai ] && ln -sf /workspace/meta-spliceai /root/work/meta-spliceai 2>/dev/null

# Activate conda environment
eval "$(/workspace/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate metaspliceai
```

---

## Files Involved in Path Resolution

### 1. `meta_spliceai/system/config.py`

Contains `find_project_root()` which detects the project root by looking for marker files (`.git`, `pyproject.toml`, etc.).

### 2. `meta_spliceai/system/config.ini`

Contains hardcoded paths for different deployment environments:

```ini
[DataSource]
source = ensembl
version = GRCh38.106
# data_prefix is determined dynamically by SystemConfig.PROJ_DIR
```

### 3. `configs/genomic_resources.yaml`

Main configuration for genomic resources. Must exist at `<PROJECT_ROOT>/configs/genomic_resources.yaml`.

### 4. `meta_spliceai/system/genomic_resources/config.py`

Loads configuration from YAML with environment variable overrides.

---

## Debugging Path Resolution

### Check Current Project Root

```python
from meta_spliceai.system.config import find_project_root
import os

proj_root = find_project_root(os.getcwd())
print(f"Project root: {proj_root}")
print(f"Config exists: {os.path.exists(f'{proj_root}/configs/genomic_resources.yaml')}")
```

### Check Environment Variables

```bash
echo "META_SPLICEAI_ROOT: $META_SPLICEAI_ROOT"
echo "SS_DATA_ROOT: $SS_DATA_ROOT"
```

### Verify Configuration Loading

```python
from meta_spliceai.system.genomic_resources import load_config

try:
    cfg = load_config()
    print(f"Config loaded successfully")
    print(f"  Build: {cfg.build}")
    print(f"  Data root: {cfg.data_root}")
except FileNotFoundError as e:
    print(f"Config error: {e}")
```

---

## Best Practices for Multi-Environment Development

### 1. Use Relative Paths in Code

Prefer relative paths from project root rather than hardcoded absolute paths:

```python
# Good
from pathlib import Path
project_root = Path(__file__).resolve().parents[4]
data_path = project_root / "data" / "splicevardb"

# Avoid
data_path = Path("/Users/me/work/meta-spliceai/data/splicevardb")  # Hardcoded!
```

### 2. Use Environment Variables for Overrides

```python
import os
from pathlib import Path

# Allow environment override
data_root = Path(os.getenv("SS_DATA_ROOT", "data"))
```

### 3. Document Environment-Specific Setup

Each deployment environment should have its own setup documentation:

- `RUNPODS_COMPLETE_SETUP.md` - RunPods
- `LOCAL_SETUP.md` - Local development
- `AZURE_SETUP.md` - Azure/Synapse

### 4. Test Path Resolution Early

Add path validation at the start of scripts:

```python
def validate_environment():
    """Validate that required paths exist."""
    from pathlib import Path
    
    required = [
        Path("configs/genomic_resources.yaml"),
        Path("data/splicevardb/splicevardb.download.tsv"),
    ]
    
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
    
    print("✅ Environment validated")
```

---

## Quick Fix Summary

For a new RunPods pod, run these commands once:

```bash
# 1. Create symlink for path resolution
mkdir -p /root/work
ln -sf /workspace/meta-spliceai /root/work/meta-spliceai

# 2. Set environment variables
export META_SPLICEAI_ROOT=/workspace/meta-spliceai
export SS_DATA_ROOT=/workspace/meta-spliceai/data

# 3. Verify
python -c "from meta_spliceai.system.genomic_resources import load_config; print('✅ Config loads successfully')"
```

---

## Related Documentation

- [RUNPODS_COMPLETE_SETUP.md](./RUNPODS_COMPLETE_SETUP.md) - Full setup guide
- [RUNPODS_STORAGE_REQUIREMENTS.md](./RUNPODS_STORAGE_REQUIREMENTS.md) - Disk space requirements
- [../../../system/genomic_resources/README.md](../../../system/genomic_resources/README.md) - Genomic resources system

---

*This document helps troubleshoot path resolution issues when moving between development environments.*

