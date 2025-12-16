# Python Dataclasses in MetaSpliceAI

This document outlines how dataclasses are used throughout the MetaSpliceAI codebase and provides best practices for working with them.

## Overview

Python's `@dataclass` decorator (introduced in Python 3.7) is used extensively in MetaSpliceAI to create clean, type-safe configuration and data container classes. Dataclasses reduce boilerplate code while providing robust type annotations and automatic methods.

## Key Benefits

1. **Reduced Boilerplate**: Automatic generation of `__init__`, `__repr__`, and other special methods
2. **Type Annotations**: Explicit type declarations for better IDE support and code readability
3. **Immutability Option**: Support for read-only (frozen) data structures when needed
4. **Default Values**: Clean syntax for default values and calculations
5. **Post-Processing**: Ability to customize initialization with `__post_init__`

## Usage Patterns in MetaSpliceAI

### Configuration Classes

The most common use of dataclasses in MetaSpliceAI is for configuration objects like `SpliceAIConfig`:

```python
@dataclass
class SpliceAIConfig:
    """Configuration for enhanced SpliceAI prediction workflow."""
    
    # File paths with smart defaults from Analyzer class
    gtf_file: Optional[str] = field(default_factory=lambda: Analyzer.gtf_file)
    genome_fasta: Optional[str] = field(default_factory=lambda: Analyzer.genome_fasta)
    eval_dir: str = field(default_factory=lambda: Analyzer.eval_dir)
    
    # Simple default values
    threshold: float = 0.5
    consensus_window: int = 2
    
    # Derived attributes with post-initialization
    local_dir: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived values after dataclass initialization."""
        # If local_dir is not explicitly provided, derive it from eval_dir
        if self.local_dir is None:
            self.local_dir = os.path.dirname(self.eval_dir)
```

### Data Container Classes

Dataclasses are also used for structured data containers that hold related fields:

```python
@dataclass
class SpliceRecord:
    """Container for splice site information."""
    transcript_id: str
    chrom: str
    position: int
    strand: str
    site_type: str
    is_annotated: bool = False
    probability: float = 0.0
```

## Best Practices

### 1. Use Type Annotations Consistently

Always include type annotations for all fields. This improves IDE support, makes code more readable, and enables static type checking:

```python
# Good
@dataclass
class GeneFeature:
    gene_id: str
    start: int
    end: int
    strand: str
    
# Avoid
@dataclass
class GeneFeature:
    gene_id = ""  # Missing type annotation
    start = 0
    end = 0
    strand = ""
```

### 2. Use `field()` for Complex Defaults

For mutable defaults or calculated values, use `field(default_factory=...)` instead of direct assignment:

```python
# Good
@dataclass
class AnalysisConfig:
    chromosomes: List[str] = field(default_factory=list)
    
# Bad - Shared mutable default across all instances!
@dataclass
class AnalysisConfig:
    chromosomes: List[str] = []  # All instances will share the same list!
```

### 3. Leverage `__post_init__` for Derived Values

Use `__post_init__` to calculate or validate fields after initialization:

```python
@dataclass
class GenomicRegion:
    start: int
    end: int
    length: Optional[int] = None
    
    def __post_init__(self):
        # Automatically calculate length if not provided
        if self.length is None:
            self.length = self.end - self.start
        
        # Validate field values
        if self.start > self.end:
            raise ValueError(f"Invalid region: start ({self.start}) > end ({self.end})")
```

### 4. Consider Frozen Dataclasses for Immutable Data

For data that should not change after creation, use `frozen=True`:

```python
@dataclass(frozen=True)
class GeneIdentifier:
    """Immutable gene identifier that can be used as a dictionary key."""
    gene_id: str
    version: str
```

### 5. Add Helper Methods for Related Functionality

Dataclasses can contain methods just like regular classes:

```python
@dataclass
class SpliceAIConfig:
    eval_dir: str = field(default_factory=lambda: Analyzer.eval_dir)
    output_subdir: str = "meta_models"
    
    def get_full_eval_dir(self) -> str:
        """Get the full evaluation directory path."""
        return os.path.join(self.eval_dir, self.output_subdir)
```

### 6. Document Fields with Clear Docstrings

Good documentation is essential for dataclasses that serve as configurations:

```python
@dataclass
class OverlappingGeneConfig:
    """Configuration for overlapping gene analysis.
    
    Attributes
    ----------
    min_exons : int
        Minimum number of exons for a gene to be considered
    filter_valid_splice_sites : bool
        Whether to filter for valid splice sites
    """
    min_exons: int = 2
    filter_valid_splice_sites: bool = True
```

## Migration from Dictionary Configurations

When migrating from dictionary-based configurations to dataclasses, follow these steps:

1. Identify all keys used in the dictionary
2. Create appropriate type annotations for each key
3. Add default values matching the original defaults
4. Add validation in `__post_init__` if needed
5. Update code that accessed dictionary values to use attribute access

```python
# Before: Dictionary-based configuration
config = {
    'threshold': 0.5,
    'window_size': 10,
    'do_extract': False
}

# After: Dataclass-based configuration
@dataclass
class AnalysisConfig:
    threshold: float = 0.5
    window_size: int = 10
    do_extract: bool = False
```

This transformation makes code more maintainable, self-documenting, and enables IDE features like auto-completion and type checking.
