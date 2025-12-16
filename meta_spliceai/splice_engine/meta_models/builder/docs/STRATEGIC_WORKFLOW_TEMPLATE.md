# Strategic Training Workflow - Copy-Paste Template

**Ready-to-use commands for creating strategically enhanced training datasets**

## Template Variables

**Customize these values for your specific use case:**

```bash
# Configuration
GENE_TYPE="protein_coding"                    # protein_coding, lncRNA, etc.
BASE_GENES=5000                              # Random genes from policy
STRATEGIC_GENES=2000                         # Strategic enhancement genes  
TOTAL_GENES=$((BASE_GENES + STRATEGIC_GENES)) # ~7000 (after deduplication)
DATASET_NAME="train_pc_7000_strategic"       # Output directory name
STUDY_NAME="pc_strategic_evaluation"         # Evaluation study name
KMER_SIZE=3                                  # k-mer feature size
```

---

## Phase 1: Strategic Gene Selection

```bash
# Activate environment
mamba activate surveyor

# Create strategic gene lists directory
mkdir -p strategic_genes
cd strategic_genes

echo "🎯 Creating strategic gene selections for ${GENE_TYPE} genes..."

# 1. Meta-optimized genes (best for meta-model performance)
python -m meta_spliceai.splice_engine.meta_models.builder.strategic_gene_selector \
    meta-optimized \
    --count 1000 \
    --gene-types ${GENE_TYPE} \
    --output meta_optimized_${GENE_TYPE}.txt \
    --verbose

# 2. High splice density genes (splice-rich regions)
python -m meta_spliceai.splice_engine.meta_models.builder.strategic_gene_selector \
    high-density \
    --count 600 \
    --min-density 12.0 \
    --gene-types ${GENE_TYPE} \
    --output high_density_${GENE_TYPE}.txt \
    --verbose

# 3. Length-stratified genes (balanced size distribution)
python -m meta_spliceai.splice_engine.meta_models.builder.strategic_gene_selector \
    length-strata \
    --ranges 20000,50000 50000,150000 150000,500000 \
    --counts 200,200,200 \
    --gene-types ${GENE_TYPE} \
    --output-dir length_strata_${GENE_TYPE} \
    --verbose

# 4. Combine all strategic selections
echo "📋 Combining strategic gene selections..."
cat meta_optimized_${GENE_TYPE}.txt \
    high_density_${GENE_TYPE}.txt \
    length_strata_${GENE_TYPE}/all_length_strata.txt > combined_strategic_${GENE_TYPE}.txt

# 5. Remove duplicates and create final list
sort combined_strategic_${GENE_TYPE}.txt | uniq > final_strategic_${STRATEGIC_GENES}_${GENE_TYPE}.txt

# 6. Verify strategic selection
ACTUAL_STRATEGIC=$(wc -l < final_strategic_${STRATEGIC_GENES}_${GENE_TYPE}.txt)
echo "✅ Strategic gene selection complete: ${ACTUAL_STRATEGIC} genes (requested: ${STRATEGIC_GENES})"

# Return to project root
cd ..
```

---

## Phase 2: Training Dataset Creation

Choose one of the following approaches:

### **Option A: Strategic + Random (Recommended)**

```bash
echo "🏗️ Creating strategic + random dataset: ${DATASET_NAME}..."

# Create training dataset with strategic enhancement
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes ${TOTAL_GENES} \
    --subset-policy random \
    --gene-types ${GENE_TYPE} \
    --gene-ids-file strategic_genes/final_strategic_${STRATEGIC_GENES}_${GENE_TYPE}.txt \
    --output-dir ${DATASET_NAME} \
    --batch-size 500 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes ${KMER_SIZE} \
    --verbose

echo "✅ Strategic + random dataset complete: ${DATASET_NAME} (~${TOTAL_GENES} genes)"
```

### **Option B: All Available Genes (Maximum Coverage)**

```bash
echo "🌐 Creating comprehensive dataset with ALL ${GENE_TYPE} genes..."

# Use all available genes of specified type
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --subset-policy all \
    --gene-types ${GENE_TYPE} \
    --output-dir train_${GENE_TYPE}_all_${KMER_SIZE}mers \
    --batch-size 500 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes ${KMER_SIZE} \
    --verbose

echo "✅ Comprehensive dataset complete with all available ${GENE_TYPE} genes"
```

### **Option C: Strategic Genes Only**

```bash
echo "🎯 Creating strategic-only dataset..."

# Use only strategic genes (custom selection)
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --gene-types ${GENE_TYPE} \
    --gene-ids-file strategic_genes/final_strategic_${STRATEGIC_GENES}_${GENE_TYPE}.txt \
    --output-dir train_${GENE_TYPE}_strategic_only_${KMER_SIZE}mers \
    --batch-size 500 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes ${KMER_SIZE} \
    --verbose

echo "✅ Strategic-only dataset complete"
```

---

## Phase 3: Verification

```bash
echo "🔍 Verifying training dataset composition..."

# Comprehensive verification
python -c "
import polars as pl
import sys

try:
    manifest = pl.read_csv('${DATASET_NAME}/gene_manifest.csv')
    
    print('📊 TRAINING DATASET VERIFICATION')
    print('=' * 60)
    print(f'Dataset: ${DATASET_NAME}')
    print(f'Expected gene type: ${GENE_TYPE}')
    print(f'Expected total genes: ~${TOTAL_GENES} (${BASE_GENES} random + ${STRATEGIC_GENES} strategic)')
    print()
    
    # Basic stats
    print(f'✅ Actual total genes: {len(manifest):,}')
    
    # Gene type verification
    print('\\n🧬 Gene type distribution:')
    type_counts = manifest['gene_type'].value_counts()
    all_expected_type = True
    for row in type_counts.iter_rows():
        gene_type, count = row
        is_expected = gene_type == '${GENE_TYPE}'
        status = '✅' if is_expected else '❌'
        print(f'  {status} {gene_type}: {count:,}')
        if not is_expected:
            all_expected_type = False
    
    if all_expected_type:
        print('\\n✅ All genes match expected type: ${GENE_TYPE}')
    else:
        print('\\n❌ WARNING: Found unexpected gene types!')
    
    # Length statistics
    print('\\n📏 Gene length statistics:')
    length_stats = manifest['gene_length']
    print(f'  Mean: {length_stats.mean():.0f} bp')
    print(f'  Median: {length_stats.median():.0f} bp')
    print(f'  Range: {length_stats.min():,} - {length_stats.max():,} bp')
    
    # Splice density statistics  
    print('\\n🧬 Splice density statistics:')
    density_stats = manifest['splice_density_per_kb']
    print(f'  Mean: {density_stats.mean():.2f} sites/kb')
    print(f'  Median: {density_stats.median():.2f} sites/kb')
    print(f'  Range: {density_stats.min():.2f} - {density_stats.max():.2f} sites/kb')
    
    # Strategic enhancement verification
    with open('strategic_genes/final_strategic_${STRATEGIC_GENES}_${GENE_TYPE}.txt') as f:
        strategic_genes = set(line.strip() for line in f)
    
    manifest_genes = set(manifest['gene_id'].to_list())
    strategic_in_manifest = strategic_genes.intersection(manifest_genes)
    
    print('\\n🎯 Strategic enhancement verification:')
    print(f'  Strategic genes requested: {len(strategic_genes):,}')
    print(f'  Strategic genes in dataset: {len(strategic_in_manifest):,}')
    print(f'  Strategic inclusion rate: {len(strategic_in_manifest)/len(strategic_genes)*100:.1f}%')
    
    if len(strategic_in_manifest) == len(strategic_genes):
        print('  ✅ All strategic genes successfully included')
    else:
        missing = len(strategic_genes) - len(strategic_in_manifest)
        print(f'  ⚠️ {missing} strategic genes missing (likely due to workflow constraints)')
    
    print('\\n' + '=' * 60)
    print('📋 DATASET SUMMARY')
    print('=' * 60)
    print(f'✅ Dataset: ${DATASET_NAME}')
    print(f'✅ Total genes: {len(manifest):,}')
    print(f'✅ Gene type consistency: {\"Yes\" if all_expected_type else \"No\"}')
    print(f'✅ Strategic enhancement: {len(strategic_in_manifest):,}/{len(strategic_genes):,} genes')
    print(f'✅ Enhanced manifest: Available with comprehensive characteristics')
    
except Exception as e:
    print(f'❌ Error during verification: {e}')
    sys.exit(1)
"

echo ""
echo "📁 Training dataset files:"
ls -la ${DATASET_NAME}/
echo ""
echo "📋 Enhanced manifest preview:"
head -5 ${DATASET_NAME}/gene_manifest.csv
```

---

## Phase 4: Evaluation Gene Lists

```bash
echo "🧪 Creating evaluation gene lists..."

# Create evaluation gene lists (⚠️ Note: no gene type filtering available)
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --training 20 \
    --unseen 30 \
    --study-name "${STUDY_NAME}" \
    --training-dataset ${DATASET_NAME} \
    --output-dir evaluation_genes \
    --prefix ${STUDY_NAME} \
    --verbose

echo "⚠️ Note: prepare_gene_lists.py does not filter by gene type"
echo "   Evaluation genes may include types other than ${GENE_TYPE}"
echo "   Consider manual filtering if gene type consistency is critical"
```

---

## Phase 5: Ready-to-Use Inference Commands

```bash
echo "🚀 READY-TO-USE INFERENCE COMMANDS"
echo "=================================="
echo ""
echo "# Create logs directory"
echo "mkdir -p logs"
echo ""
echo "# Training genes evaluation (should show strong meta-model performance)"
echo "python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \\"
echo "    --model results/gene_cv_pc_1000_3mers_run_4 \\"
echo "    --training-dataset ${DATASET_NAME} \\"
echo "    --genes-file evaluation_genes/${STUDY_NAME}_training_genes.txt \\"
echo "    --output-dir results/${STUDY_NAME}_training_meta_only \\"
echo "    --inference-mode meta_only \\"
echo "    --enable-chunked-processing \\"
echo "    --chunk-size 5000 \\"
echo "    --verbose \\"
echo "    --mlflow-enable \\"
echo "    --mlflow-experiment \"${STUDY_NAME}_training\" \\"
echo "    2>&1 | tee logs/${STUDY_NAME}_training_meta_only.log"
echo ""
echo "# Unseen genes evaluation (tests generalization)"  
echo "python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \\"
echo "    --model results/gene_cv_pc_1000_3mers_run_4 \\"
echo "    --training-dataset ${DATASET_NAME} \\"
echo "    --genes-file evaluation_genes/${STUDY_NAME}_unseen_genes.txt \\"
echo "    --output-dir results/${STUDY_NAME}_unseen_meta_only \\"
echo "    --inference-mode meta_only \\"
echo "    --enable-chunked-processing \\"
echo "    --chunk-size 5000 \\"
echo "    --verbose \\"
echo "    --mlflow-enable \\"
echo "    --mlflow-experiment \"${STUDY_NAME}_unseen\" \\"
echo "    2>&1 | tee logs/${STUDY_NAME}_unseen_meta_only.log"
```

---

## Documentation Template

```bash
# Create dataset documentation
cat > ${DATASET_NAME}/DATASET_INFO.md << EOF
# Training Dataset: ${DATASET_NAME}

## Configuration
- **Base genes**: ${BASE_GENES} (${GENE_TYPE}, random selection)
- **Strategic genes**: ${STRATEGIC_GENES} (${GENE_TYPE}, characteristic-based)
- **Total genes**: ~${TOTAL_GENES} (after deduplication)
- **Gene types**: ${GENE_TYPE} only
- **k-mer sizes**: ${KMER_SIZE}
- **Created**: $(date)
- **Workflow**: Strategic enhancement with incremental builder

## Strategic Selection Breakdown
- **Meta-optimized**: 1000 genes (optimized for meta-model performance)
- **High splice density**: 600 genes (>12 sites/kb)
- **Length-stratified**: 600 genes (balanced size distribution)
  - Medium (20-50kb): 200 genes
  - Large (50-150kb): 200 genes  
  - Very large (150-500kb): 200 genes

## Verification Results
\`\`\`
$(python -c "
import polars as pl
manifest = pl.read_csv('${DATASET_NAME}/gene_manifest.csv')
print(f'Actual total genes: {len(manifest):,}')
print('Gene type distribution:')
type_counts = manifest['gene_type'].value_counts()
for row in type_counts.iter_rows():
    gene_type, count = row
    print(f'  {gene_type}: {count:,}')
print(f'Mean gene length: {manifest[\"gene_length\"].mean():.0f} bp')
print(f'Mean splice density: {manifest[\"splice_density_per_kb\"].mean():.2f} sites/kb')
")
\`\`\`

## Files
- \`master/\`: Partitioned training dataset (Parquet files)
- \`gene_manifest.csv\`: Enhanced gene manifest with characteristics
- \`DATASET_INFO.md\`: This documentation file

## Usage
This dataset is ready for meta-model training with transcript-aware position identification.

## Evaluation
Use \`prepare_gene_lists.py\` to create evaluation gene lists for inference workflow testing.
EOF

echo "📋 Dataset documentation created: ${DATASET_NAME}/DATASET_INFO.md"
```

---

## Customization Examples

### **Example 1: lncRNA-Focused Dataset**
```bash
GENE_TYPE="lncRNA"
BASE_GENES=3000
STRATEGIC_GENES=1500
DATASET_NAME="train_lnc_4500_strategic"
STUDY_NAME="lnc_strategic_evaluation"
```

### **Example 2: Multi-Type Dataset**
```bash
GENE_TYPE="protein_coding lncRNA"  # Multiple types
BASE_GENES=4000
STRATEGIC_GENES=2000
DATASET_NAME="train_pc_lnc_6000_strategic"
STUDY_NAME="multi_type_evaluation"
```

### **Example 3: Large-Scale Dataset**
```bash
GENE_TYPE="protein_coding"
BASE_GENES=8000
STRATEGIC_GENES=3000
DATASET_NAME="train_pc_11000_strategic"
STUDY_NAME="large_scale_evaluation"
KMER_SIZE=6  # Larger k-mers for more complex features
```

---

## Troubleshooting Checklist

- [ ] **Environment activated**: `mamba activate surveyor`
- [ ] **Gene type consistency**: Same `--gene-types` in both tools
- [ ] **Strategic genes created**: Check `strategic_genes/` directory
- [ ] **Training dataset exists**: Check `${DATASET_NAME}/master/` directory
- [ ] **Enhanced manifest available**: Check `${DATASET_NAME}/gene_manifest.csv`
- [ ] **Gene type verification passed**: All genes match expected type
- [ ] **Strategic inclusion verified**: Most strategic genes included
- [ ] **Evaluation genes created**: Check `evaluation_genes/` directory
- [ ] **Documentation created**: Check `${DATASET_NAME}/DATASET_INFO.md`

---

## Next Steps

1. **Customize the template variables** at the top of this file
2. **Run the phases sequentially** (copy-paste each phase)
3. **Verify results** after each phase
4. **Proceed to model training** with your strategically enhanced dataset
5. **Use evaluation commands** to test meta-model performance

**Happy strategic training! 🎯**
