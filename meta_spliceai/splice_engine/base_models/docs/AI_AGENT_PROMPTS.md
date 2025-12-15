# AI Agent Prompts: Meta-SpliceAI Base Layer Porting

**Last Updated**: November 27, 2025  
**Purpose**: Ready-to-use prompts for instructing AI agents to port the base layer  
**Audience**: Humans instructing AI coding assistants

---

## Overview

This document provides **copy-paste prompts** you can use to instruct AI coding assistants (like Claude, GPT-4, or other AI agents) to analyze and port the Meta-SpliceAI base layer.

**Use Case**: You want an AI agent to understand the codebase and help you:
- Port the base layer to a new project
- Integrate it into an existing system
- Understand specific components
- Debug or extend functionality

---

## Table of Contents

1. [Initial Context Setting](#initial-context-setting)
2. [Stage-by-Stage Prompts](#stage-by-stage-prompts)
3. [Verification Prompts](#verification-prompts)
4. [Troubleshooting Prompts](#troubleshooting-prompts)
5. [Advanced Integration Prompts](#advanced-integration-prompts)

---

## Initial Context Setting

### Prompt 1: Provide Overview and Goals

```
I need you to help me port the Meta-SpliceAI base layer to a new project. 

The base layer is a model-agnostic framework for splice site prediction that:
1. Runs predictions with any compatible base model (SpliceAI, OpenSpliceAI, custom models)
2. Evaluates predictions against reference annotations
3. Extracts training features for meta-learning
4. Manages artifacts with intelligent checkpointing

Key characteristics:
- Model-agnostic: Works with any model producing per-nucleotide splice scores
- Memory-efficient: Mini-batch processing (50 genes/batch within 500-gene chunks)
- Production-ready: Checkpointing, artifact management, schema standardization

Please follow the systematic approach in docs/base_models/AI_AGENT_PORTING_GUIDE.md. 
We'll go through this in 6 stages:
1. Understand entry points
2. Trace core workflow
3. Map data dependencies
4. Understand genomic resources system
5. Identify essential vs optional components
6. Create minimal port

Are you ready to begin Stage 1?
```

---

## Stage-by-Stage Prompts

### Stage 1: Understand Entry Points

#### Prompt 1.1: Analyze User-Facing Entry Point

```
Stage 1, Step 1: Analyze the user-facing Python API entry point.

Please read and analyze: meta_spliceai/run_base_model.py

Focus on:
1. The main functions: run_base_model_predictions() and predict_splice_sites()
2. What parameters does a user provide?
3. What is the configuration system? (BaseModelConfig, SpliceAIConfig, OpenSpliceAIConfig)
4. Where does the main delegation happen? (What function is called?)

After analyzing, please answer:
- What parameters does run_base_model_predictions() accept?
- What does it return?
- What core workflow function does it delegate to?

Create a simple call flow diagram from this entry point to the core workflow.

**Important Note on Configuration Design**:
The configuration system uses inheritance:
- BaseModelConfig (abstract base class)
  ├── SpliceAIConfig (for SpliceAI/GRCh37)
  └── OpenSpliceAIConfig (for OpenSpliceAI/GRCh38)

You may see `BaseModelConfig = SpliceAIConfig` alias in older code - this is DEPRECATED.
The refactored design uses proper inheritance. See dev/CONFIG_REFACTORING_PLAN.md.
```

#### Prompt 1.2: Analyze CLI Entry Point

```
Stage 1, Step 2: Analyze the CLI entry point.

Please read: meta_spliceai/splice_engine/cli/run_base_model.py

Focus on:
1. How do command-line arguments map to Python parameters?
2. What defaults are used?
3. Does this also call the same core workflow?

After analyzing, confirm:
- Do both entry points (Python API and CLI) converge to the same core workflow function?
- What is that function?
```

#### Prompt 1.3: Create Entry Point Map

```
Stage 1, Step 3: Create a complete entry point hierarchy map.

Based on your analysis of:
- meta_spliceai/run_base_model.py (Python API)
- meta_spliceai/splice_engine/cli/run_base_model.py (CLI)

Please create a hierarchical diagram showing:
1. All user entry points
2. Configuration classes they use
3. The convergence point (core workflow function)
4. Any shell script orchestration

Format as a tree structure with clear arrows showing delegation.
```

---

### Stage 2: Trace Core Workflow

#### Prompt 2.1: Analyze Core Workflow Structure

```
Stage 2, Step 1: Analyze the core workflow file structure.

Please read: meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py

This is the MOST IMPORTANT file. Please analyze it in sections:

Section 1: Imports (lines 1-72)
- List all imported modules
- Categorize them into: Core Workflow, Data Preparation, Data Types, I/O Handlers, Model Utils, Enhancement
- Create a table with: Module Name | Category | Purpose

Section 2: Main function signature (lines 74-133)
- What is the function name?
- What parameters does it accept?
- What does it return?
- What are the key configuration options?

Please provide your analysis for Section 1 first.
```

#### Prompt 2.2: Break Down Workflow Logic

```
Stage 2, Step 2: Break down the main workflow function into logical sections.

For run_enhanced_splice_prediction_workflow() in splice_prediction_workflow.py:

Please identify and describe each major section:
1. Configuration (roughly lines 74-210): What happens here?
2. Data Preparation (lines 216-450): What are the 9 preparation steps?
3. Main Processing Loop (lines 500-1005): What is the nested loop structure?
4. Aggregation and Save (lines 1018-1202): What gets saved?

For each section, list:
- Key functions called
- Key classes/objects created
- Main logic flow

Start with Section 1 (Configuration). What key setup happens here?
```

#### Prompt 2.3: Trace Data Preparation Functions

```
Stage 2, Step 3: Trace the data preparation functions.

The workflow calls these functions in data_preparation.py:
1. prepare_gene_annotations()
2. prepare_splice_site_annotations()
3. prepare_genomic_sequences()
4. handle_overlapping_genes()
5. determine_target_chromosomes()
6. load_spliceai_models()

For each function, please analyze and document:
- Input parameters
- What it does
- Output/return value
- Files it reads or creates

Please start with prepare_gene_annotations(). What does it do?
```

#### Prompt 2.4: Understand the Processing Loop

```
Stage 2, Step 4: Understand the nested processing loop.

In splice_prediction_workflow.py, lines 500-1005, there's a three-level nested loop:

for chromosome in chromosomes:
    for chunk (500 genes):
        for mini_batch (50 genes):
            # Process here

Please explain:
1. Why this nested structure? What does each level achieve?
2. What happens at each level? (chromosome loading, chunk checkpointing, mini-batch processing)
3. Where does prediction happen?
4. Where does evaluation happen?
5. Where does sequence extraction happen?
6. Where is memory freed?

Create a flowchart showing the complete loop structure with key operations at each level.
```

---

### Stage 3: Map Data Dependencies

#### Prompt 3.1: Identify Required Data Files

```
Stage 3, Step 1: Identify all required data files.

Based on your analysis of the workflow, please create a comprehensive table of:

| Data Type | File Format | Example Filename | Purpose | How Generated |
|-----------|-------------|------------------|---------|---------------|
| ... | ... | ... | ... | ... |

Include:
- Reference genome
- Gene annotations
- Base model weights
- Derived files (splice sites, sequences, databases)

For each, specify if it's an INPUT or DERIVED file.
```

#### Prompt 3.2: Create Data Flow Diagram

```
Stage 3, Step 2: Create a data flow diagram.

Please show:
1. Input files (user provides)
2. Processing steps that create derived files
3. Derived files (system generates)
4. How derived files are used in predictions

Use this format:
Input → Processing → Derived → Usage

Be specific about file names and processing functions.
```

---

### Stage 4: Understand Genomic Resources System

#### Prompt 4.1: Analyze the Registry

```
Stage 4, Step 1: Analyze the genomic resources Registry.

Please read: meta_spliceai/system/genomic_resources/registry.py

Focus on the Registry class:
1. What is its purpose?
2. How does it map build names (like 'GRCh38_MANE') to data directories?
3. What paths does get_paths() return?
4. Show example usage for both GRCh37 and GRCh38

Then explain: Why is build-specific path resolution important?
```

#### Prompt 4.2: Understand Schema Standardization

```
Stage 4, Step 2: Understand schema standardization.

Please read: meta_spliceai/system/genomic_resources/schema.py

Explain:
1. What problem does schema standardization solve?
2. What are some examples of synonymous column names?
   (e.g., 'site_type' vs 'splice_type', 'seqname' vs 'chrom')
3. What functions perform standardization?
4. Show a code example of using standardize_splice_sites_schema()

Then explain: When must schema standardization be called in the workflow?
```

#### Prompt 4.3: Explain Directory Conventions

```
Stage 4, Step 3: Explain the standard directory layout.

Please read: docs/data/DATA_LAYOUT_MASTER_GUIDE.md

Then create a visual directory tree showing:
1. The standard data/ directory structure
2. Where SpliceAI data lives vs OpenSpliceAI data
3. Where derived files are stored
4. Where model weights are stored
5. Where output artifacts are saved

Explain why this structure prevents data mixing between builds.
```

---

### Stage 5: Identify Essential Components

#### Prompt 5.1: List Essential Files

```
Stage 5, Step 1: Identify the minimal set of essential files.

Based on your complete analysis of the workflow, data preparation, and genomic resources:

Please create two lists:

ESSENTIAL FILES (must port):
- File path
- One-sentence purpose
- Dependencies (what it imports)

OPTIONAL FILES (can skip for minimal port):
- File path
- Why it's optional
- When you'd need it

Aim for ~20 essential files maximum.
```

#### Prompt 5.2: Create Dependency Graph

```
Stage 5, Step 2: Create a dependency graph of essential components.

For the ~20 essential files you identified:

Please create a directed graph showing:
- Each file as a node
- Import relationships as edges
- Group by category (Core Workflow, Data Types, I/O, Genomic Resources, Utils)

Identify:
- Which files are "leaf nodes" (no dependencies)?
- Which files are "root nodes" (many things depend on them)?
- What is the critical path through the dependency graph?
```

---

### Stage 6: Create Minimal Port

#### Prompt 6.1: Design Port Structure

```
Stage 6, Step 1: Design the directory structure for the minimal port.

Based on the ~20 essential files, design a new project structure that:
1. Is simpler than the original (fewer nested directories)
2. Maintains logical grouping (core, genomic, models, utils)
3. Uses clear naming
4. Follows Python package conventions

Propose a directory structure for "my_project" that ports the essential components.
Include what files go where and why.
```

#### Prompt 6.2: Plan Import Refactoring

```
Stage 6, Step 2: Plan how to refactor imports.

For each essential file in the port:
1. Original path in meta-spliceai
2. New path in my_project
3. Import statements that need updating
4. Dependencies that need to be resolved

Create a refactoring table showing the transformation for each file.
```

#### Prompt 6.3: Create Minimal Entry Point

```
Stage 6, Step 3: Create a minimal entry point for the ported base layer.

Please write a simple run_predictions.py that:
1. Imports the core workflow function (using new import paths)
2. Creates a simple configuration
3. Runs predictions on a test gene
4. Returns and prints results

Keep it under 50 lines. Include:
- Imports
- Simple function wrapper
- if __name__ == '__main__': test block
- Inline comments explaining each step
```

---

## Verification Prompts

### Verify Stage 1

```
Verification: Stage 1 complete?

Please confirm you can answer:
1. What are the three main entry points to the base layer?
2. What function do they all converge to?
3. What configuration class is used?
4. Draw the entry point hierarchy from memory.

If you can answer all four, we can proceed to Stage 2.
```

### Verify Stage 2

```
Verification: Stage 2 complete?

Please confirm you understand:
1. The main workflow function name and what it does
2. The 9 data preparation steps (list them)
3. The 3-level nested loop structure (chromosome → chunk → mini-batch)
4. Where prediction, evaluation, and sequence extraction happen
5. Why mini-batching is used

If yes, proceed to Stage 3.
```

### Verify Stage 3

```
Verification: Stage 3 complete?

Please confirm you can:
1. List the 5 required input files
2. List the 3 main derived files
3. Draw the data flow from inputs → derived → usage
4. Explain which files can be reused vs must be regenerated

If yes, proceed to Stage 4.
```

### Verify Stage 4

```
Verification: Stage 4 complete?

Please confirm you understand:
1. What the Registry class does
2. Why schema standardization is necessary
3. The standard data directory structure
4. Why build-specific directories prevent data mixing

Show me an example of using Registry to get paths for GRCh38_MANE.
```

### Verify Stage 5

```
Verification: Stage 5 complete?

Please confirm you:
1. Identified ~20 essential files
2. Can categorize them (Core Workflow, Data Types, I/O, etc.)
3. Created a dependency graph
4. Distinguished essential from optional components

List the 5 most critical files (highest in dependency graph).
```

### Verify Stage 6

```
Verification: Stage 6 complete?

Please confirm you:
1. Designed a new project structure
2. Mapped all essential files to new locations
3. Created a minimal entry point
4. Understand how to update imports

Show me the proposed directory structure and entry point code.
```

---

## Troubleshooting Prompts

### Debug: Import Errors

```
I'm getting import errors after porting. Please help me debug.

Error: [paste error message]

Affected file: [file path]

Please:
1. Identify what the import is trying to find
2. Check if that dependency was ported
3. Suggest how to fix the import path
4. If the dependency is missing, is it essential or optional?
```

### Debug: Data File Not Found

```
The ported code can't find a data file.

Error: FileNotFoundError: [file path]

Please:
1. What was this file supposed to be?
2. Is it an input file I need to provide?
3. Or is it a derived file that should be generated?
4. How do I generate it? (what function creates it?)
5. Is the directory structure correct according to conventions?
```

### Debug: Schema Issues

```
I'm getting KeyError or column not found errors.

Error: [paste error message]

Please:
1. Is this a schema standardization issue?
2. What column is it expecting vs what's present?
3. Where should I call standardize_*_schema()?
4. Show me the exact code to fix it.
```

### Debug: Wrong Results

```
The ported code runs but gives different results than the original.

Please help me check:
1. Am I using the correct genome build for the base model?
2. Are schema standardizations being applied?
3. Is the data directory structure correct?
4. Are there any hard-coded paths I missed?
5. What are the most common causes of result discrepancies?
```

---

## Advanced Integration Prompts

### Add Custom Model

```
I want to add my own custom splice prediction model to the ported base layer.

My model:
- Framework: [PyTorch/TensorFlow/Other]
- Input: DNA sequences (ACGT)
- Output: Per-nucleotide probabilities for donor, acceptor, neither
- Weights location: [path]

Please:
1. Show me the model wrapper template I need to implement
2. What methods must the wrapper have?
3. Where do I register the model?
4. How do I specify model metadata (build, framework)?
5. Show me a complete example wrapper for my model

Then show me how to use it: run_predictions(base_model='my_custom_model')
```

### Optimize for Large Genomes

```
I need to optimize the ported code for processing very large genomes (>3GB).

Current issues:
- [describe memory/performance issues]

Please help me:
1. Identify memory bottlenecks in the workflow
2. Suggest optimizations (mini-batch size, chunk size, etc.)
3. Should I disable final aggregation?
4. How can I monitor memory usage during processing?
5. Can I process chromosomes in parallel?
```

### Integrate with Existing Pipeline

```
I want to integrate the ported base layer into my existing genomics pipeline.

My pipeline:
- Input format: [describe]
- Current workflow: [describe]
- Desired integration point: [describe]

Please help me:
1. What's the best integration approach?
2. Can I use the Python API programmatically?
3. How do I configure paths for my existing data structure?
4. Can I customize the output format?
5. Show me an integration example
```

---

## Complete Example Session

Here's how a complete AI agent porting session might look:

```
Human: [Initial Context Setting Prompt - from above]

AI: Ready to begin Stage 1. I understand we're porting the Meta-SpliceAI base 
    layer using a 6-stage systematic approach. Let's start with analyzing the 
    entry points.
