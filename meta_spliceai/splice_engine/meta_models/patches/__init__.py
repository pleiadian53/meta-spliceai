"""
Patches for compatibility issues in the splice_engine package.

This subpackage contains patched versions of functions to handle specific
edge cases and compatibility issues when running the original code.
"""

from meta_spliceai.splice_engine.meta_models.patches.genomic_features import (
    patched_compute_intron_lengths
)
