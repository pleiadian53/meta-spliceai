# Generalized Analysis Tools for Inference Workflow

## Overview
This document outlines the design for generalized analysis tools that can handle arbitrary numbers of genes (from tens to thousands) in the splice site prediction inference workflow.

## Core Components

### 1. `inference_analyzer.py` - Main Analysis Framework
A modular, scalable analysis tool that can:
- Handle results from 1 to 10,000+ genes
- Work with streaming/partial results
- Support multiple inference modes (base_only, hybrid, meta_only)
- Generate both summary and detailed per-gene reports

**Key Features:**
```python
class InferenceAnalyzer:
    def __init__(self, output_dirs, batch_size=100):
        """
        Args:
            output_dirs: Dict mapping modes to output directories
            batch_size: Process genes in batches for memory efficiency
        """
        
    def analyze_batch(self, gene_ids):
        """Process a batch of genes"""
        
    def stream_analysis(self):
        """Analyze results as they become available"""
        
    def generate_summary_report(self):
        """Create high-level summary for all genes"""
        
    def generate_detailed_report(self, gene_id=None):
        """Create detailed report for specific gene(s)"""
```

### 2. `inference_monitor.py` - Real-time Progress Monitoring
Monitor long-running inference jobs with:
- Progress tracking across multiple parallel runs
- ETA estimation based on current throughput
- Resource usage monitoring (CPU, memory, disk)
- Alert system for failures or anomalies

**Features:**
- Dashboard-style display (TUI using Rich/Textual)
- Export progress to JSON for web monitoring
- Slack/email notifications for completion
- Automatic log rotation for large runs

### 3. `batch_comparator.py` - Multi-mode Comparison Tool
Compare results across different inference modes:
- Statistical significance testing
- Performance improvement metrics
- Gene-level comparative analysis
- Identify genes where meta-model helps most

**Output Formats:**
- Interactive HTML reports (using Plotly)
- CSV exports for further analysis
- MLFlow integration for experiment tracking
- LaTeX tables for publications

### 4. `performance_aggregator.py` - Large-scale Metrics
Efficiently compute metrics for thousands of genes:
- Streaming computation (doesn't load all data at once)
- Hierarchical aggregation (gene → chromosome → genome)
- Support for distributed computation
- Cache intermediate results

**Metrics:**
- Standard: Precision, Recall, F1, AP, ROC-AUC
- Custom: Splice-site specific metrics
- Stratified: By gene size, chromosome, GC content
- Temporal: Performance over processing time

### 5. `error_analyzer.py` - Systematic Error Analysis
Identify patterns in prediction errors:
- Cluster genes by error patterns
- Identify systematic biases
- Suggest model improvements
- Generate error heat maps

### 6. `resource_profiler.py` - Performance Profiling
Profile computational requirements:
- Time per gene/chunk
- Memory usage patterns
- I/O bottlenecks
- Optimization recommendations

## Command-line Interface

### Basic Usage
```bash
# Analyze results for any number of genes
python -m workflows.inference.inference_analyzer \
    --base-dir results/test_scenario2b_base \
    --hybrid-dir results/test_scenario2b_hybrid \
    --meta-dir results/test_scenario2b_meta \
    --output-dir analysis_results \
    --batch-size 100

# Monitor running inference
python -m workflows.inference.inference_monitor \
    --log-files "*.log" \
    --update-interval 30 \
    --dashboard

# Compare modes
python -m workflows.inference.batch_comparator \
    --reference base_only \
    --compare-with hybrid meta_only \
    --gene-list genes.txt \
    --output comparison_report.html
```

### Advanced Usage
```bash
# Stream analysis for ongoing inference
python -m workflows.inference.inference_analyzer \
    --stream-mode \
    --watch-dir results/ \
    --update-interval 60

# Distributed analysis for large datasets
python -m workflows.inference.performance_aggregator \
    --distributed \
    --workers 8 \
    --cache-dir /tmp/inference_cache

# Error pattern analysis
python -m workflows.inference.error_analyzer \
    --predictions-dir results/predictions \
    --clustering-method kmeans \
    --n-clusters 10
```

## Data Management

### Input Handling
- Support for multiple file formats (Parquet, CSV, TSV)
- Lazy loading for large datasets
- Automatic format detection
- Compressed file support

### Output Organization
```
analysis_results/
├── summary/
│   ├── overview.json          # High-level metrics
│   ├── performance_table.csv  # Tabular results
│   └── key_findings.md        # Executive summary
├── per_gene/
│   ├── gene_metrics.parquet   # Detailed per-gene metrics
│   └── outliers.json          # Genes with unusual performance
├── comparisons/
│   ├── mode_comparison.html   # Interactive comparison
│   └── significance_tests.csv # Statistical tests
├── visualizations/
│   ├── roc_curves.pdf         # ROC/PR curves
│   ├── performance_dist.html  # Performance distributions
│   └── error_heatmap.png      # Error pattern visualization
└── logs/
    └── analysis.log            # Processing logs
```

## Scalability Considerations

### Memory Management
- Process genes in configurable batches
- Use memory-mapped files for large datasets
- Implement garbage collection strategies
- Support for out-of-core computation

### Parallel Processing
- Multi-threading for I/O operations
- Multi-processing for CPU-intensive tasks
- Support for cluster computing (Dask/Ray)
- GPU acceleration where applicable

### Caching Strategy
- Cache expensive computations
- Incremental updates for new genes
- Distributed cache for cluster setups
- Automatic cache invalidation

## Integration Points

### MLFlow Integration
- Automatic metric logging
- Artifact storage
- Experiment comparison
- Model versioning

### Workflow Integration
- Hooks for post-inference analysis
- Automatic report generation
- CI/CD pipeline integration
- Notification systems

## Implementation Timeline

1. **Phase 1** (After 20-gene test): Core analyzers
   - `inference_analyzer.py`
   - `batch_comparator.py`

2. **Phase 2**: Monitoring and profiling
   - `inference_monitor.py`
   - `resource_profiler.py`

3. **Phase 3**: Advanced analysis
   - `error_analyzer.py`
   - `performance_aggregator.py`

4. **Phase 4**: Optimizations
   - Distributed computing support
   - GPU acceleration
   - Web-based dashboard

## Testing Strategy

### Unit Tests
- Test each component independently
- Mock large datasets
- Edge case handling

### Integration Tests
- End-to-end workflow testing
- Multi-mode comparison
- Performance benchmarks

### Scalability Tests
- Test with 10, 100, 1000, 10000 genes
- Memory usage profiling
- Performance regression tests

## Documentation

### User Guide
- Quick start guide
- Common workflows
- Troubleshooting

### API Documentation
- Module documentation
- Function signatures
- Usage examples

### Architecture Documentation
- System design
- Data flow diagrams
- Performance considerations
