"""
Post-Training Analysis Tools

This package contains analysis scripts and tools that can be run after completing
CV training runs to perform additional diagnostics, comparisons, and insights.

Available Tools:
    - compare_cv_runs: Compare results from multiple CV runs
    - (future tools will be added here)

Usage:
    from meta_spliceai.splice_engine.meta_models.analysis.post_training import compare_cv_runs
"""

from .compare_cv_runs import CVRunComparator, main as compare_cv_runs_main

__all__ = ['CVRunComparator', 'compare_cv_runs_main'] 