# Base Model Scripts

Scripts for downloading and managing base model resources.

## Available Scripts

| Script | Purpose |
|--------|---------|
| `download_openspliceai_models.sh` | Download OpenSpliceAI pre-trained models |

## Quick Start

### Download OpenSpliceAI Models

```bash
./scripts/base_model/download_openspliceai_models.sh
```

This downloads 5 PyTorch models (~13MB total) to `data/models/openspliceai/`.

### Verify Installation

```bash
# Check SpliceAI models
ls -la data/models/spliceai/

# Check OpenSpliceAI models  
ls -la data/models/openspliceai/
```

## Model Locations

| Model | Location | Format |
|-------|----------|--------|
| SpliceAI | `data/models/spliceai/` | Keras HDF5 (.h5) |
| OpenSpliceAI | `data/models/openspliceai/` | PyTorch (.pt) |

## Related Documentation

- `meta_spliceai/splice_engine/base_models/docs/SPLICEAI.md` - SpliceAI details
- `meta_spliceai/splice_engine/base_models/docs/OPENSPLICEAI.md` - OpenSpliceAI details
- `docs/base_models/` - User-facing base model guides

