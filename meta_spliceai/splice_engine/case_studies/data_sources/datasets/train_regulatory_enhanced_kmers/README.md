# Dataset: train_regulatory_enhanced_kmers

## Quick Overview

**Purpose**: Multi-gene-type dataset including regulatory non-coding genes for enhanced splice pattern analysis  
**Size**: *TBD* (~10,000 genes, estimated ~15M records)  
**Features**: Enhanced k-mer analysis (3-mer + 5-mer) with regulatory elements  
**Status**: 🚧 **In Development**  

## Planned Dataset Location
```
/home/bchiu/work/meta-spliceai/train_regulatory_enhanced_kmers/
```

## Dataset Specifications (Planned)

### Gene Selection Strategy
- **Total Genes**: ~10,000 genes
- **Selection Policy**: `meta_optimized` for strategic regulatory coverage
- **Gene Types**: 
  - `protein_coding` - Core protein-coding genes
  - `lncRNA` - Long non-coding RNAs with regulatory functions
  - `miRNA` - MicroRNAs for post-transcriptional regulation
  - `snoRNA` - Small nucleolar RNAs for RNA modification
  - `snRNA` - Small nuclear RNAs for splicing machinery

### Enhanced Features
- **K-mer Analysis**: Both 3-mer and 5-mer composition analysis
- **Regulatory Elements**: Enhanced detection of regulatory splice patterns
- **Multi-Gene-Type Support**: Specialized features for non-coding gene analysis
- **Cross-Gene-Type Comparisons**: Features enabling comparative analysis

### Processing Parameters
- **Batch Size**: 200 genes per batch (optimized for diverse gene types)
- **Batch Rows**: 15,000 rows per batch (adjusted for regulatory complexity)
- **K-mer Sizes**: 3,5 (dual k-mer analysis)
- **Gene IDs File**: `strategic_regulatory_genes.txt`

## Expected Use Cases

- **Regulatory Variant Assessment**: Impact analysis for variants in regulatory regions
- **Non-coding Splice Pattern Analysis**: Understanding splicing in regulatory genes
- **Multi-Gene-Type Meta-Learning**: Training models across diverse gene biotypes
- **Comparative Splicing Research**: Cross-gene-type splicing mechanism studies
- **Disease-Associated Regulatory Analysis**: Pathogenic variants in regulatory elements

## Generation Command (Planned)

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 10000 \
    --subset-policy meta_optimized \
    --gene-types protein_coding,lncRNA,miRNA,snoRNA,snRNA \
    --gene-ids-file strategic_regulatory_genes.txt \
    --output-dir train_regulatory_enhanced_kmers \
    --batch-size 200 \
    --batch-rows 15000 \
    --run-workflow \
    --kmer-sizes 3,5 \
    --verbose 2>&1 | tee -a logs/train_regulatory_enhanced.log
```

## Documentation Files (To Be Created)

| File | Status | Description |
|------|--------|-------------|
| `train_regulatory_enhanced_kmers_profile.md` | 📝 Pending | Comprehensive dataset overview and characteristics |
| `train_regulatory_enhanced_kmers_technical_spec.md` | 📝 Pending | Detailed schema and technical specifications |
| `validate_train_regulatory_enhanced_kmers.py` | 📝 Pending | Dataset validation and usage demonstration script |

## Development Progress

- [x] Directory structure created
- [x] Initial planning and specifications
- [ ] Gene selection strategy finalization
- [ ] Strategic regulatory genes file preparation
- [ ] Dataset generation execution
- [ ] Quality validation and testing
- [ ] Comprehensive documentation creation
- [ ] Integration testing with meta-models

## Key Differences from train_pc_7000_3mers_opt

### Enhanced Scope
- **Multi-Gene-Type**: Includes 5 different gene biotypes vs. protein-coding only
- **Dual K-mer Analysis**: 3-mer + 5-mer vs. 3-mer only
- **Regulatory Focus**: Specialized features for regulatory element analysis
- **Strategic Selection**: `meta_optimized` policy vs. random selection

### Technical Improvements
- **Optimized Batching**: Adjusted batch parameters for regulatory gene complexity
- **Enhanced Feature Set**: Additional features for non-coding gene analysis
- **Cross-Type Analysis**: Features enabling comparative analysis across gene types

## Integration with Case Studies

This dataset will enable:
- **Advanced Variant Analysis**: Regulatory variant impact assessment
- **Multi-Modal Learning**: Training across diverse gene types
- **Regulatory Disease Studies**: Analysis of disease-associated regulatory variants
- **Comparative Genomics**: Cross-gene-type splicing pattern analysis

---

**Dataset Version**: *TBD*  
**Documentation Version**: 0.1 (Planning)  
**Last Updated**: 2025-08-23  
**Next Milestone**: Gene selection strategy finalization
