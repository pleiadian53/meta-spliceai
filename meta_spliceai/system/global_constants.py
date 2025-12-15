"""
Global constants for the SpliceMediator project.

This module contains constants that are used across multiple packages
in the SpliceMediator project. Package-specific constants should still
be defined within their respective packages.
"""

# Common column names used across packages
COL_TID = "transcript_id" # "consensus_tx_id"
# NOTE: Other names for the same column:
# "tx_id" in redundant_transcripts  
# "transcript_id" in Gencode

COL_PRIMARY_GID = "primary_gene_id"  # used in txdb.transcripts
COL_GID = "gene_id"  # "gene_id", "consensus_gene_id"
COL_GN = "gene_name"
COL_EID = "exon_id"

# Common gene and transcript identifiers
GENE_ID_PREFIX = "GN."

# Expression metrics
COL_TPM = "consensus_tpm"
COL_SID = "sample_id"
COL_SN = "sample_name"
COL_DATASET_ID = COL_DATASET = "dataset_id"

# Common biotype columns
COL_LABEL = 'label'
COL_PRED = 'prediction'
COL_BTYPE = COL_BIOTYPE = COL_BTYPE_REF = 'tx_type_ref'  # 'transcript_biotype' in Gencode
COL_BTYPE_PRED = 'tx_type_pred'

# Sequence-related column names
COL_SEQ = "sequence"
COL_MARKER = "marker"

# Dataset identifiers
TEST_FILE_ID = 'testset'
TRAIN_FILE_ID = 'trainset'
QUANTIFIED_FILE_ID = 'quantified'
