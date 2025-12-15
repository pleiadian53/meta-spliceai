# Test Runner Scripts

This directory contains wrapper scripts for running test suites.

## Available Runners

### OpenSpliceAI Tests

**Script**: `run_openspliceai_tests.sh`

**Purpose**: Run the OpenSpliceAI integration test suite

**Usage**:
```bash
# From project root
./scripts/testing/runners/run_openspliceai_tests.sh
```

**What it does**:
1. Activates the `surveyor` conda environment
2. Runs the simple 5-gene test (`test_openspliceai_simple.py`)
3. Optionally runs the full 30-gene test (`test_openspliceai_gene_categories.py`)
4. Logs all output to `logs/` directory

**Requirements**:
- `surveyor` conda environment must be set up
- OpenSpliceAI models must be downloaded
- GRCh38 MANE data must be available

---

## Adding New Test Runners

When creating new test runner scripts:

1. **Place them here**: `scripts/testing/runners/`
2. **Make them executable**: `chmod +x script_name.sh`
3. **Use absolute paths**: For project directory references
4. **Activate environment**: Ensure proper conda/mamba activation
5. **Log output**: Use `tee` to save logs to `logs/` directory
6. **Document here**: Add entry to this README

---

*Last Updated: November 7, 2025*


