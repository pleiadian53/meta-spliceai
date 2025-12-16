"""
Meta-model feature extraction and processing modules.
"""

from .kmer_features import (
    make_kmer_features,
    harmonize_feature_sets,
    make_kmer_featurized_dataset
)

from .kmer_filtering import (
    KmerFilterManager,
    MotifBasedFilter,
    MutualInformationFilter,
    SparsityAwareFilter,
    VarianceFilter,
    EnsembleFilter,
    create_filter_manager,
    filter_dataset_kmers,
    get_kmer_filtering_stats
)

from .kmer_filter_config import (
    KmerFilterConfig,
    add_kmer_filtering_args,
    create_preset_configs,
    get_preset_config,
    print_filtering_stats,
    validate_filtering_config,
    integrate_kmer_filtering_in_cv
)

__all__ = [
    # K-mer feature extraction
    'make_kmer_features',
    'harmonize_feature_sets', 
    'make_kmer_featurized_dataset',
    
    # K-mer filtering
    'KmerFilterManager',
    'MotifBasedFilter',
    'MutualInformationFilter',
    'SparsityAwareFilter',
    'VarianceFilter',
    'EnsembleFilter',
    'create_filter_manager',
    'filter_dataset_kmers',
    'get_kmer_filtering_stats',
    
    # K-mer filtering configuration
    'KmerFilterConfig',
    'add_kmer_filtering_args',
    'create_preset_configs',
    'get_preset_config',
    'print_filtering_stats',
    'validate_filtering_config',
    'integrate_kmer_filtering_in_cv'
]
