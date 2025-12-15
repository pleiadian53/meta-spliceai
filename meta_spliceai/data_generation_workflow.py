import os, re, sys
import csv
import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
from tabulate import tabulate

import argparse, configparser
from .system import sys_config as config
from .system.sys_config import Txdb
from .system.model_config import BiotypeModel, NMDEffModel

from .utils.utils_sys import highlight
from .sphere_pipeline.process_trpts_from_synapse import (
    # retrieve_nmd_tx_ids, 
    retrieve_tx_ids_matching_biotypes,
    retrieve_noncoding_tx_ids, 
    retrieve_random_genes, 
    run_batched_query_generic,
    retrieve_sample_matched_tx_ids_for_testset,
    retrieve_sample_matched_protein_coding_tx_ids_for_testset,
    # retrieve_protein_coding_tx_ids,
    # lookup_appris_annotation,
    select_samples_by_dataset,
    run_batched_query,
    map_tx_source_id, map_gene_source_id, 
    filter_samples_in_tpm_matrix,
    construct_sparse_tpm_matrix, construct_tpm_matrix, structure_tpm_matrix, 
    remove_duplicate_exons,
    retrieve_transcript_sequences_incrementally, 
    extract_features,
)

from .sphere_pipeline import utils_data as ud 
from .sphere_pipeline import utils_test as ut
from .sphere_pipeline import constants as const

from .sphere_pipeline import data_model as dm
from .sphere_pipeline.data_model import (DataSource, TranscriptIO, 
                                           SequenceDescriptor, 
                                           Sequence, 
                                           Concept)

from .sphere_pipeline import metadata_model as mm
from .sphere_pipeline.metadata_model import GEMatrix
from .sphere_pipeline import query_templates
from .sphere_pipeline.feature_extractor import (
    harmonize_test_set, 
)

from .mllib.model_tracker import ModelTracker
# from .biotype_classifier import train_biotype_3way_classifier
from .mllib.evaluator import (
    calculate_all_metrics, 
)
from .utils.utils_doc import print_emphasized


def has_unknown_biotypes(tx_io=None, **kargs):

    col_btype_ref = TranscriptIO.col_btype
    col_btype_pred = TranscriptIO.col_btype_pred

    # target_biotype, target_suffix = concept.biotype, concept.suffix
    if tx_io is None: 
        biotype = kargs.get("biotype", 'nmd_eff')
        suffix = kargs.get("suffix", 'testset')
        tx_io = TranscriptIO(biotype=biotype, suffix=suffix)
    
    print(f"[predicate] Loading test set transcript data with ID={tx_io.ID} ...")
    df_tx = tx_io.load_tx()  # Load test set transcript

    n_tx_type_pred_unknown = df_tx[col_btype_pred].isna().sum()
    n_tx_type_ref_unknown = df_tx[col_btype_ref].isna().sum()

    return (n_tx_type_pred_unknown > 0) or (n_tx_type_ref_unknown > 0)

# def load_or_train_model(overwrite=False, **kargs):

#     # Source data IDs, which references the transcriptomic dataset used to train the biotype classifier
#     source_biotype = kargs.get('source_biotype', BiotypeModel.source_biotype)
#     source_suffix = kargs.get('source_suffix', BiotypeModel.source_suffix)   

#     model_name = kargs.get("model_name", BiotypeModel.model_name) # the ML algorithm used as the classifier (e.g. XGBoost)
#     model_suffix = kargs.get("model_suffix", BiotypeModel.model_suffix) # Supplementary model identifier 

#     tracker = kargs.get("tracker", 
#                             ModelTracker(experiment=BiotypeModel.model_output_dir, 
#                                 model_type=BiotypeModel.model_type, 
#                                     model_name=model_name, model_suffix=model_suffix)) 

#     model = None 
#     try:                               
#         if not overwrite:
#             print("[I/O] Loading pre-trained biotype classifier ...")
#             model = tracker.load(verbose=1) # Load the pre-trained model
#         else:
#             msg = "[action] Training new a biotype model ..."
#             raise FileNotFoundError(msg)

#     except FileNotFoundError as e: 
#         if not overwrite:
#             print(e)
#         # Train the model from scratch
#         print("(load_or_train_model) Training a new instance of biotype classifier ...")
        
#         train_biotype_3way_classifier(labeling_concept=BiotypeModel.labeling_concept, 
#                                 experiment=BiotypeModel.model_output_dir,  
#                                     model_name=model_name, 
#                                     model_suffix=model_suffix,
#                                         n_fold=BiotypeModel.n_folds, 
#                                         use_nested_cv=BiotypeModel.use_nested_cv, 
#                                             biotype=source_biotype, 
#                                             suffix=source_suffix)
#         print("[info] Model training complete.")
#         print("[I/O] Loading the newly trained biotype classifier ...")
#         model = tracker.load(verbose=1) # Load the pre-trained model

#     model_state = tracker.load_metadata() # format: json, pkl

#     return model, model_state

def data_survey(df, msg=''): 
    res = {}

    if msg: print_emphasized(msg)
    print(f"...... data shape: {df.shape}")
    
    col_gid = const.col_gid
    col_tid = const.col_tid
    col_btype = const.col_btype

    if col_gid in df.columns: 
        res['n_genes'] = df[col_gid].nunique()
        print(f"...... n(genes): {res['n_genes']}")
    if col_tid in df.columns: 
        res['n_trpts'] = df[col_tid].nunique()
        print(f"...... n(transcripts): {res['n_trpts']}")

    if col_btype in df.columns:
        biotypes = df[col_btype].unique()
        print(f"...... biotypes: {biotypes}")

    return res

def predict_biotypes(concept, **kargs):
    from .mllib.model_evaluation import evaluate_multiclass_classifier
    from .sphere_pipeline.utils_data import dummify_and_verify
    
    # Identify genes with well-defined treatment and control
    # Identfy genes with 'dangling' transcripts 
    # - NMD targets without protein-coding transcripts in control group 
    # - No NMD targets found but protein-coding transcripts exist
    
    # Identify 'dangling' genes
    # - genes with transcripts of other biotypes but no NMD-fated and protein-coding

    # Source data IDs, which references the transcriptomic dataset used to train the biotype classifier
    train_model = kargs.get('train_model', False)
    source_biotype = kargs.get('source_biotype', BiotypeModel.source_biotype)
    source_suffix = kargs.get('source_suffix', BiotypeModel.source_suffix)   

    model_name = kargs.get("model_name", BiotypeModel.model_name) # the ML algorithm used as the classifier (e.g. XGBoost)
    model_suffix = kargs.get("model_suffix", BiotypeModel.model_suffix) # Supplementary model identifier 
     
    concept_trained_classifier = \
        Concept(concept=BiotypeModel.labeling_concept, 
                        biotype=source_biotype, suffix=source_suffix)

    ##################################################
    model, model_state = \
        load_or_train_model(overwrite=train_model, 
            source_biotype=source_biotype, source_suffix=source_suffix,
                model_name=model_name, model_suffix=model_suffix)
    ##################################################

    fs_train = model_state['feature_names_cat_encoded'] 
    print(f"(predict_biotypes) n={len(fs_train)} features used for training the biotype classifier.")
    
    # -----------------------------------------------

    col_tid = TranscriptIO.col_tid
    col_btype_known = col_btype_ref = TranscriptIO.col_btype
    col_btype_pred = TranscriptIO.col_btype_pred
    col_label = TranscriptIO.col_label

    # Test set concept and 
    labeling_concept = concept.labeling_concept
    assert labeling_concept == concept_trained_classifier.concept, \
        f"Inconsistent concept between training set and test set: {labeling_concept} <> {concept_trained_classifier.concept}"
    
    target_biotype = concept.biotype  # biotype ID for the test set
    target_suffix = concept.suffix    # supplementary ID for the test set; by default "testset"

    tx_io = TranscriptIO(biotype=target_biotype, suffix=target_suffix) # test set transcript ID: (target_biotype, target_suffix)

    print(f"(predict_biotypes) Loading test set transcript data with ID={(target_biotype), (target_suffix)} ...")
    df_tx = tx_io.load_tx(verbose=1)  # Load test set transcript
    data_survey(df_tx, "(predict_biotypes) Before biotype predictions")

    # Identify transcripts for which the biotype annotation values are null (i.e. 'tx_type_ref' and 'tx_type_pred')
    # has_unknown_biotypes(tx_io)
    for col in [col_btype_ref, col_btype_pred, ]:
        tx_set = df_tx[df_tx[col].isna()][col_tid].unique()
        print(f"(predict_biotypes) Found n={len(tx_set)} transcripts with NULL values in column={col}")

    #-------------------------
    ftype = BiotypeModel.ftype

    print(f"[I/O] Loading featurized transcriptomic TEST data given ID=({target_biotype}, {target_suffix}), ftype={ftype} ...")
    dtor_test = SequenceDescriptor(biotype=target_biotype, suffix=target_suffix) # concept=labeling_concept
    df_test = dtor_test.load_transcript_features(ftype=ftype)
    assert df_test[col_tid].nunique() == df_test[col_tid].shape[0]

    # assert set(df_tx[col_tid].values) >= set(df_test[col_tid].values)
    print(f"... found n={df_test[col_tid].nunique()} (=?= {df_tx[col_tid].nunique()}) unique transcripts in featurized dataframe")

    print(f"[I/O] Loading featurized transcriptomic TRAINING data given ID=({source_biotype}, {source_suffix}), ftype={ftype} ...")
    dtor_train = SequenceDescriptor(biotype=source_biotype, suffix=source_suffix) # Training transcript data
    df_train = dtor_train.load_transcript_features(ftype=ftype, drop_labels=True)
    X_train_prime, _ = SequenceDescriptor.to_xy(df_train, dummify=True)
    print(f"... original shape(X): {X_train_prime.shape}")
    assert not col_label in df_train.columns

    # The original feature sets may not be consistent 
    tid_as_index = kargs.get('tid_as_index', True)
    X_test, _ = SequenceDescriptor.to_xy(df_test, dummify=True, tid_as_index=tid_as_index)
    n_test = X_test.shape[0]
    X_train, _ = SequenceDescriptor.to_xy(df_train, dummify=True, tid_as_index=tid_as_index)
    n_train = X_train.shape[0]

    if set(X_test.columns) != set(fs_train): 
        highlight("[action] Harmonizing the test set features with the training set ...")

        X_test, X_test_meta, X_train, X_train_meta = \
            harmonize_test_set(
                df_test, 
                df_train=df_train, fs_train=fs_train,
                get_non_feature_columns=SequenceDescriptor.get_non_feature_columns, 
                cat_encoder=dummify_and_verify, 
                tid_as_index=tid_as_index,
                merge_meta_data=False)  
        # NOTE: The harmonized test data X_test does not have transcript IDs as index

    # --- Test ---
    assert X_train.shape == X_train_prime.shape
    assert X_train.shape[0] == n_train
    
    # --- Test --- 
    print(f"[test] columns in X_test but not in fs_train:\n{set(X_test.columns)-set(fs_train)}\n")
    print(f"[test] columns in fs_train but not in X_test:\n{set(fs_train)-set(X_test.columns)}\n")
    # NOTE: E.g. {'chromosome_chrY', 'chromosome_chr21'} if not dummified prior to canonicalizing
    assert set(X_test.columns) == set(fs_train)    

    y_pred = model.predict(X_test)

    # Todo: Evaluate the prediction
    # evaluate_multiclass_classifier(model, X_test, y_test, metrics=['accuracy', 'f1_macro', 'roc_auc_macro', 'mcc'])
    # concept = Concept(concept=labeling_concept, biotype=test_biotype, suffix=test_suffix) 

    # Map label prediction (classes in integers) back to their names
    print("(predict_biotypes) Mapping label prediction in integer to their names ...")
    print(f"> labeling concept: {labeling_concept}")
    print(f"=> concept.label_names:\n{concept.label_names}\n")
    predictions = [concept.label_names[yp] for yp in y_pred]
    df_test[col_label] = predictions
    data_survey(df_test, "(predict_biotypes) df_test, after biotype predictions:")

    df_tx = df_tx.merge(df_test[[col_tid, col_label]], on=col_tid, how='inner')
    df_tx[col_btype_pred] = df_tx[col_label]
    df_tx.drop(col_label, axis=1, inplace=True)

    # Fill in predicted biotypes where there are no existing annotations from known sources
    df_tx[col_btype_known] = df_tx[col_btype_known].fillna(df_tx[col_btype_pred])

    # --- Test ---
    dm.gene_survey(df_tx, "(predict_biotypes) After biotype predictions")

    # Save the transcript dataframe with predicted labeles
    save = kargs.get('save', True)

    if save: 
        print(f"(predict_biotypes) Saving transcript test set with ID=({target_biotype}, {target_suffix})")
        tx_io = TranscriptIO(biotype=target_biotype, suffix=target_suffix)
        tx_io.save_tx(df_tx, verbose=1)

    # --- Test --- 
    n_tx_type_pred_unknown = df_tx[col_btype_pred].isna().sum()
    n_tx_type_ref_unknown = df_tx[col_btype_ref].isna().sum()
    print(f"[test] After filling in biotype predictions, we still have n={n_tx_type_pred_unknown} unknown entries")
    print(f"... n(unknown) for {col_btype_known}: {n_tx_type_ref_unknown} =?= {n_tx_type_pred_unknown}")

    return df_tx # labeled transcript dataframe 


# Refactor: sphere_pipeline.data_model_utils
def randomly_subset_transcripts(df_tx, max_tx_per_gene=100, **kargs):
    """
    Randomly subset transcripts for each gene in the dataframe.

    Parameters:
    - df: DataFrame containing columns 'tx_id' and 'gene_id'
    - max_tx_per_gene: Maximum number of transcripts per gene

    Returns:
    - DataFrame with randomly subsetted transcripts
    """
    import random
    
    # Assume df_tx has columns: ['tx_id', 'gene_id', ...]
    col_tid = kargs.get('col_tid', 'tx_id') 
    col_gid = kargs.get('col_gid', 'gene_id') 
    random_state = kargs.get('random_state', None)

    # Group transcripts by gene_id
    grouped = df_tx.groupby(col_gid)[col_tid].agg(list).reset_index()

    # Set random seed for reproducibility
    random.seed(random_state)

    # Randomly select up to max_tx_per_gene transcripts per gene
    # grouped[col_tid] = grouped[col_tid].apply(lambda x: random.sample(x, min(len(x), max_tx_per_gene)))
    
    def subset_transcripts(tx_list):
        if len(tx_list) > max_tx_per_gene:
            return random.sample(tx_list, max_tx_per_gene)
        else:
            return tx_list
    grouped[col_tid] = grouped[col_tid].apply(subset_transcripts)

    # Initialize a list to store the sampled transcripts
    sampled_tx = []
    
    # For each gene, sample up to max_transcripts transcripts
    for tx_ids in grouped[col_tid]:
        sampled_tx.extend(tx_ids)

    # Filter df_tx to keep only the sampled transcripts
    df_sampled = df_tx[df_tx[col_tid].isin(sampled_tx)]
    
    return df_sampled

def gene_names_to_ids(names, tx_io=None, **kargs): 

    col_gn = TranscriptIO.col_gn
    col_gid = TranscriptIO.col_gid
    if names is None: names = []

    gene_id_prefix = kargs.get('gene_id_prefix', const.gene_id_prefix) # configuration

    # Check if the input genes are already in the form of gene IDs
    # [Todo]

    if tx_io is None: 
        biotype = kargs.get("biotype", 'nmd_eff')
        suffix = kargs.get("suffix", 'testset')
        tx_io = TranscriptIO(biotype=biotype, suffix=suffix)

    sql_template = kargs.get('sql_template', 
        query_templates['map_gene_id_constraint_optional'])

    if len(names) > 0:
        gene_condition = f"WHERE g.gene_name IN ({','.join([repr(gene) for gene in names])})"
    else:
        gene_condition = ""

    template_params = {'gene_condition': gene_condition, "txdb_genes": Txdb.genes}
    # NOTE: 'tx_id_condition'? run_batched_query() will handle the IN-predicate for tx_set

    query_result_path = tx_io.target_genes_path
    query_cache_dir = tx_io.cache_dir
    run_batched_query(None,  # Set to None to disable filtering by tx_id
        sql_template, template_params, 
            tx_table_ref='g', in_predicate_first=False,
            batch_size=25000, 
                output_path=query_result_path,
                temp_dir=query_cache_dir, verbose=2)
    df_genes = ud.load_data(query_result_path, delimiter='\t', use_dask=False)
    
    return df_genes.set_index(col_gn)[col_gid].to_dict()

def retrieve_testset_given_transcripts(tx_io=None, tx_set=[], **kargs):

    def display_genes_map(genes_dict, n=20):
        # Get the first n items of the dictionary
        print(f"[info] Displaying the first n={n} gene names and IDs ...")
        items = list(genes_dict.items())[:n]

        # Print each item
        for name, id in items:
            print(f"... name: {name} => ID: {id}")

    # from sphere_pipeline.data_model import gene_survey

    col_gid = TranscriptIO.col_gid
    col_tid = TranscriptIO.col_tid
    gene_id_prefix = kargs.get('gene_id_prefix', const.gene_id_prefix) # configuration

    if tx_io is None:
        target_biotype = kargs.get("target_biotype", "nmd_eff")
        target_suffix = kargs.get("target_suffix", "testset")
        tx_io = TranscriptIO(biotype=target_biotype, suffix=target_suffix)
    else: 
        target_biotype = tx_io.biotype
        target_suffix = tx_io.suffix

    # Retrieve transcript attributes from txdb.transcripts for those transcripts meeting appris-specific constraints
    # Remove nan values from tx_set
    # tx_set = [x for x in tx_set if not pd.isna(x)]

    # Raise an exception if tx_set contains nan
    if any(pd.isna(x) for x in tx_set):
        raise ValueError("tx_set contains nan")

    print_emphasized(f"[retrieval] Number of input transripts: {len(set(tx_set))}")

    sql_template = query_templates['select_transcripts_given_tx_id']
    template_params = {'txdb_transcripts': Txdb.transcripts, }
    # query_result_path = tx_io.tx_appris_path
    query_result_path = os.path.join(tx_io.output_dir, f"tx-{target_biotype}-{target_suffix}-input.csv")
    run_batched_query(tx_set, 
        sql_template, template_params,
            tx_table_ref='t', in_predicate_first=True, 
            batch_size=20000, use_dask=False,
                output_path=query_result_path)
    df_tx = ud.load_data(query_result_path, delimiter='\t', use_dask=False)
    print(f"... query output n={df_tx[col_tid].nunique()} transcripts")

    # --- Test ---
    # N0 = df_tx[col_tid].nunique() 
    # df_tx = df_tx[df_tx[col_tid].isin(tx_set)]
    # N1 = df_tx[col_tid].nunique()
    # print(f"[test] Before filtering by tx_set, n_transcripts={N0} => n_transcripts={N1} after filtering by tx_set")

    # Drop duplicates based on ['tx_id', 'gene_id'] 
    # NOTE that this will not prevent transcripts from being associated with multiple genes in SequenceSphere
    # E.g. tx1 --- g1
    #      tx1 --- g2
    #      This is still consider non-duplicates
    print_emphasized("[retrieval] Filtering out duplicates based on transcript and gene IDs ...")    
    df_tx.drop_duplicates(subset=[col_tid, col_gid], inplace=True)  # Drop duplicates based on tx_id and gene_id
    data_survey(df_tx, msg="[retrieval] Transcripts dataframe after removing duplicates ...")

    #################################################
    genes = kargs.get('gene_set', [])  # Assuming that these input genes are in the form of gene IDs
    if genes is None: 
        genes = []

    if len(genes) > 0: 
        # Check if the input genes are in the form of gene names (not IDs)
        gene_names = set()
        gene_ids = set()
        for gene in genes:
            if gene.startswith(gene_id_prefix): 
                gene_ids.add(gene)
            else: 
                gene_names.add(gene)

        if gene_names: 
            genes_dict = gene_names_to_ids(gene_names, tx_io=tx_io)
            gene_ids.update(genes_dict.values())

            display_genes_map(genes_dict, n=20)
            genes_unmatched = set(gene_names) - set(genes_dict.keys())
            if len(genes_unmatched) > 0: 
                print(f"[test] n={len(genes_unmatched)} genes not found in DB:\n{genes_unmatched}\n")
        
        gene_set = list(gene_ids)
        n_genes = len(genes)
        print_emphasized(f"[retrieval] Ensuring transcript dataframe contains only user-specified genes n={n_genes} ..")

        # Filter the transcript dataframe by the input gene set
        N0 = df_tx[col_tid].nunique()
        df_tx = df_tx[df_tx[col_gid].isin(gene_set)]
        N1 = df_tx[col_tid].nunique()
        print(f"[test] Before filtering by gene set, n_transcripts={N0} => n_transcripts={N1} after filtering by gene set")

    #################################################

    print_emphasized("[retrieval] Ensuring unique transcript-to-gene mapping ...") 
    # NOTE: If this is not the case, the feature extraction process may fail
    df_tx = ensure_unique_transcript_to_gene_mapping(df_tx)
    data_survey(df_tx, msg="[retrieval] Transcripts dataframe after ensuring unique transcript-to-gene mapping ...")
    # data_survey(df_tx, msg="[retrieval] Transcripts dataframe after removing duplicates and filtering by gene set ...")
    
    # ------------------------------------------------

    print(f"[output] Saving transcript dataframe (ID={tx_io.ID}) ...")
    tx_io.save_tx(df_tx, verbose=1)

    df_txid = df_tx[[col_tid, col_gid]]

    print(f"[output] Saving transcript & gene IDs ...")
    tx_io.save_tx_id(df_txid) # Initial set of tx_ids in the test data 

    return df_tx

def retrieve_testset_given_genes(tx_io=None, genes=[], **kargs): 
    """

    Parameters: 
    - query_template 
    - n_genes
    - filter_by_cds
    
    Memo: 
    1. (Pre-)filter transcripts by CDS
       - Genes in SequenceSphere can be associated with a very large number of transcripts
       - Prioritize those with CDS because they are more likely to be relevant for NMD efficiency
    """
    # from sphere_pipeline.data_model import gene_survey

    col_gid = const.col_gid
    col_tid = const.col_tid
    col_btype = const.col_btype

    # Transcript data ID
    if genes is None: genes = []

    if tx_io is None:
        target_biotype = kargs.get("target_biotype", "nmd_eff")
        target_suffix = kargs.get("target_suffix", "testset")
        tx_io = TranscriptIO(biotype=target_biotype, suffix=target_suffix)
    else: 
        target_biotype = tx_io.biotype
        target_suffix = tx_io.suffix

    def display_genes_map(genes_dict, n=20):
        # Get the first n items of the dictionary
        print(f"[info] Displaying the first n={n} gene names and IDs ...")
        items = list(genes_dict.items())[:n]

        # Print each item
        for name, id in items:
            print(f"... name: {name} => ID: {id}")

    # --------------------------------------------

    if len(genes) == 0: 
        query_template = kargs.get('query_template', 
            query_templates['select_n_random_transcripts_with_unknonwn_biotypes'])
        # NOTE: Use this default query template to priorize genes without biotype annotations

        n_genes = kargs.get('n_genes', 1000)
        df_genes = retrieve_random_genes(n_genes=n_genes, query_template=query_template) 
        gene_set = df_genes[col_gid].unique()
        print(f"[testset retrieval] Retrieved n_genes={len(gene_set)}(=?= {n_genes}) from SequenceSphere")
        print(f"... shape(df_genes): {df_genes.shape}")
        print(f"... columns: {list(df_genes.columns)}")

        assert df_genes.shape[0] == len(gene_set), "Each gene_id should be unique."
    else: 
        # Todo: What if the input genes are already in the form of gene IDs?
        gene_id_prefix = const.gene_id_prefix # configuration

        # Check if the input genes are already in the form of gene IDs
        gene_names = set()
        gene_ids = set()
        for gene in genes:
            if gene.startswith(gene_id_prefix): 
                gene_ids.add(gene)
            else: 
                gene_names.add(gene)

        if gene_names: 
            genes_dict = gene_names_to_ids(gene_names, tx_io=tx_io)
            gene_ids.update(genes_dict.values())

            display_genes_map(genes_dict, n=20)
            genes_unmatched = set(gene_names) - set(genes_dict.keys())
            if len(genes_unmatched) > 0: 
                print(f"[test] n={len(genes_unmatched)} genes not found in DB:\n{genes_unmatched}\n")

        gene_set = list(gene_ids)
        n_genes = len(genes)
        print(f"[testset retrieval] User provided n_genes={n_genes}")

    target_genes = gene_set
    sql_template = query_templates['select_transcripts_with_in_predicate']
    template_params = {"txdb_transcripts": Txdb.transcripts, }

    query_result_path = tx_io.tx_path

    run_batched_query_generic(target_genes,   # break this large gene set into batches
        sql_template, template_params, 
            table_ref='t', in_predicate_first=True, target_col=col_gid,
            batch_size=max(100, n_genes//5), 
                use_dask=False, temp_dir = tx_io.cache_dir,
                output_path=query_result_path)
    
    # --- Test --- 
    df_tx = tx_io.load_tx(verbose=1)
    N0 = df_tx.shape[0]
    dm.gene_survey(df_tx, "[testset retrieval] Transcripts data given specified or randomly sampled genes")

    # Example biotype tally
    # protein_coding                        1871
    # lncRNA                                 860
    # retained_intron                        309
    # nonsense_mediated_decay                296
    # protein_coding_CDS_not_defined         230
    # processed_pseudogene                   102
    # miRNA                                   33
    # transcribed_unprocessed_pseudogene      20
    # snRNA                                   16
    # misc_RNA                                15
    # unprocessed_pseudogene                  11
    # processed_transcript                    11
    # transcribed_processed_pseudogene        10
    # snoRNA                                   9
    # TEC                                      8
    # IG_V_gene                                5
    # rRNA_pseudogene                          3
    # TR_V_gene                                3
    # IG_V_pseudogene                          2
    # scaRNA                                   1
    # IG_J_gene                                1
    # IG_C_gene                                1
    # non_stop_decay                           1
    # transcribed_unitary_pseudogene           1
    
    #################################################

    # Filter transcripts that do not have CDS 
    filter_by_cds = kargs.get('filter_by_cds', True)
    
    filter_by_predicted_biotypes = kargs.get('use_tx_type_pred', False)
    # NOTE: False by default because many values in tx_type_pred can be null
    
    max_tx = kargs.get('max_tx_per_gene', 100)
    random_subset_transcripts = kargs.get('random_subset_transcripts', True)
    if max_tx is None: random_subset_transcripts = False

    include_predicted_nmd = kargs.get('include_predicted_nmd', True)
    include_principal_isoforms = kargs.get('include_principal_isoforms', True) # Include principal isoforms for the candidate genes

    if filter_by_cds: 
        cond_has_cds = (
            (df_tx['cds_start'].notnull()) & 
            (df_tx['cds_end'].notnull()) & 
            (df_tx['cds_start'] != 0) & 
            (df_tx['cds_end'] != 0) & 
            (df_tx['cds_end']-df_tx['cds_start'] > 0)    
        )
        # df_tx['has_cds'] = cond_has_cds.astype(int)
        n0 = df_tx.shape[0]
        df_tx = df_tx[cond_has_cds]
        
        print(f"[info] Filtering by CDS: nrow={n0} -> {df_tx.shape[0]}")
        dm.gene_survey(df_tx, "[testset retrieval] Transcripts data after filtered by CDS")

    if filter_by_predicted_biotypes:  # False by default because many values in tx_type_pred can be null
        df_tx = subset_transcripts_by_tx_type_pred(df_tx, tx_types=['nmd', 'nonsense_mediated_decay', 'protein_coding'])
        # NOTE: tx_type_pred from SequenceSphere will predict 'nmd' for transcripts that are NMD-fated
        dm.gene_survey(df_nmd, "[testset retrieval] Transcripts predicted to be either NMD-fated or protein-coding")

    if random_subset_transcripts:
        n0 = df_tx.shape[0]

        if include_predicted_nmd:  # Include NMD-fated transcripts by default
            df_nmd = subset_transcripts_by_tx_type_pred(df_tx, tx_types=['nmd', 'nonsense_mediated_decay'])
            # NOTE: tx_type_pred from SequenceSphere will predict 'nmd' for transcripts that are NMD-fated
            dm.gene_survey(df_nmd, "[testset retrieval] Transcripts predicted to be NMD-fated")

            print("[condition] Preserving all transcripts predicted to be NMD-fated ...")
            # Remove the rows in df_tx that are in df_nmd
            df_tx_not_nmd = df_tx[~df_tx[col_tid].isin(df_nmd[col_tid])]

            # Randomly subset df_tx_not_nmd
            df_tx_not_nmd = randomly_subset_transcripts(df_tx_not_nmd, max_tx_per_gene=max_tx)

            # Append df_nmd back to df_tx_not_nmd
            df_tx = pd.concat([df_tx_not_nmd, df_nmd])

            print(f"[info] Randomly subsetting transcripts per gene (+ NMD): nrow={n0} -> {df_tx.shape[0]}")
            dm.gene_survey(df_tx, "[testset retrieval] Transcripts data after randomly subsetting transcripts for each gene (+ NMD)")
        else:            
            
            df_tx = randomly_subset_transcripts(df_tx, max_tx_per_gene=max_tx)

            print(f"[info] Randomly subsetting transcripts per gene: nrow={n0} -> {df_tx.shape[0]}")
            dm.gene_survey(df_tx, "[testset retrieval] Transcripts data after randomly subsetting transcripts for each gene")

    if include_principal_isoforms:
        highlight("[testset retrieval] Incorporating principal isoforms 'as much as possible' ...")
        df_appris = retrieve_principle_isoforms(genes=target_genes, suffix=target_suffix) # by default, overwrite_result_set=True
        negative_scores = df_appris[df_appris['appris_score']<0]['appris_score'].values
        assert len(negative_scores) == 0, \
            f"A subset of the appris-matched tx have negative appris scores:\n{negative_scores}\n"
        
        # Include transcripts in df_tx that satisfy the appris-specific constraints 
        # - Use the (col_gid, col_tid) for all genes that can be found in df_appris, 
        #   which are associated with principal isoforms
        # - For genes in df_id_ctrl that do not find a match in df_id_ctrl, use the existing transcripts as they are
        tx_appris = df_appris[col_tid].unique()
        genes_appris = df_appris[col_gid].unique()
        print(f"... Found n={len(genes_appris)} genes and m={len(tx_appris)} tx that match appris criteria")
        print(f"...... given n(target genes)={len(target_genes)}")

        # Retrieve transcript attributes from txdb.transcripts for those transcripts meeting appris-specific constraints
        sql_template = query_templates['select_transcripts_given_tx_id']
        template_params = {'tb_transcripts': 'transcripts', }
        # query_result_path = tx_io.tx_appris_path
        query_result_path = os.path.join(tx_io.output_dir, f"tx-{target_biotype}-{target_suffix}-appris.csv")
        run_batched_query(tx_appris, 
            sql_template, template_params,
                tx_table_ref='t', in_predicate_first=True, 
                batch_size=25000, 
                    output_path=query_result_path)
        df_tx_appris = ud.load_data(query_result_path, delimiter='\t', use_dask=False)
        # NOTE: Transcripts in tx_appris can be mapped to more genes in txdb.transcripts (more than the initial target genes)

        # --- Test ---
        genes_appris_matched = df_tx_appris[col_gid].unique()
        tx_appris_matched = df_tx_appris[col_tid].unique()
        print(f"[test] Out of n={len(genes_appris)} genes that match appris criteria <? {len(genes_appris_matched)} found in txdb.transcripts")
        print(f"[test] Out of m={len(tx_appris)} tx that match appris criteria =?= {len(tx_appris_matched)} found in txdb.transcripts")

        # The part of the original control set that does not have a match in df_appris
        genes_appris_unmatched = set(target_genes)-set(genes_appris_matched)
        print(f"[test] Number of initial target genes that do not have a match by appris criteria: {len(genes_appris_unmatched)}")

        print("[action] Combining the random set with the appris-qualified set ...")
        # df_tx_appris = df_tx_appris[df_tx_appris[col_gid].isin(target_genes)]
        
        df_tx = pd.concat([df_tx_appris, df_tx], ignore_index=True)
        dm.gene_survey(df_tx, "[testset retrieval] After including principal isoforms")
        # NOTE: Include df_tx because it contains the original set of transcripts for the target genes, which may 
        #       also include NMD-fated transcripts
        
        df_tx = df_tx[df_tx[col_gid].isin(target_genes)]
        dm.gene_survey(df_tx, "[testset retrieval] Ensure txdb.transcripts matched transcripts are associated with target genes")

    N0 = df_tx.shape[0]

    # Some transcripts may be mapped to multiple genes in SequenceSphere ... 
    # To address the issue of transcripts being mapped to multiple genes in our database, we will:
    #   - Drop duplicates based on tx_id and gene_id
    #   - Further address the issue of transcripts being associated with multiple genes
    #   - While dropping duplicates, also ensure that all unique genes are retained in the DataFrame.
    print("[action] Dropping duplicates based on tx_id, gene_id  while ensuring that all unique genes are retained ...")

    # Drop duplicates solely based on ['tx_id', 'gene_id'] is not sufficient because 
    # a transcript can be associated with multiple genes in SequenceSphere
    # E.g. tx1 --- g1
    #      tx1 --- g2
    #      This is still consider non-duplicates
    df_tx.drop_duplicates(subset=[col_tid, col_gid], inplace=True)  # Drop duplicates based on tx_id and gene_id

    print("[action] Ensuring that transcripts are associated with only one gene ...") 
    # Identify transcripts that are associated with multiple genes
    transcript_gene_counts = df_tx.groupby(col_tid)[col_gid].nunique()
    multiple_genes = transcript_gene_counts[transcript_gene_counts > 1]
    print("... number of transcripts associated with multiple genes: {}".format(multiple_genes.shape[0]))

    # --------------------------------------------------------
    # For each transcript that is associated with multiple genes
    n_associated_genes = []
    
    # Initialize a counter for transcripts where none of the associated genes are associated with another transcript
    counter = 0 
    
    for tx_id in multiple_genes.index:
        # Get the associated genes
        associated_genes = df_tx[df_tx[col_tid] == tx_id][col_gid].unique()

        n_associated_genes.append(len(associated_genes))

        # Initialize a flag for whether any of the associated genes are associated with another transcript(s)
        flag = False
        
        # For each associated gene
        for gene_id in associated_genes:
            # If the gene is also associated with another transcript
            if df_tx[(df_tx[col_tid] != tx_id) & (df_tx[col_gid] == gene_id)].shape[0] > 0:
                
                # Remove the association between the transcript and the gene
                df_tx = df_tx[~((df_tx[col_tid] == tx_id) & (df_tx[col_gid] == gene_id))]
                # NOTE: Filtering df_tx to exclude the row where col_tid equals tx_id and col_gid equals gene_id. 
                #       If the combination of tx_id and gene_id is unique in df_tx, then the line above will 
                #       remove exactly one row from df_tx
                
                # If the transcript is now associated with only one gene, stop checking
                if df_tx[df_tx[col_tid] == tx_id][col_gid].nunique() == 1:
                    break
        
        # If none of the associated genes are associated with another transcript
        if not flag:
            counter += 1

    print(f"[info] Number of transcripts uniquely associated with their genes: {counter}(/{multiple_genes.shape[0]})")
    # tx1 --- g1
    #     --- g2 (x)
    # tx2 --- g2 (v)
    #     --- g3 (?)
    # if we keep tx1-g1, tx1-g2, tx2-g2, tx2-g3, then we will have 2 transcripts associated with multiple genes
    # but if we keep tx1-g1, tx2-g2, tx2-g3, then we will have 1 transcripts associated with multiple genes, 
    # assuming that g3 is not associated with any other genes, then we'll have no way of eliminating the association 
    # between tx2 and g3

    max_n = max(n_associated_genes) if len(n_associated_genes) > 0 else 0
    print(f"[info] Maximum number of associated genes for a transcript: {max_n}")

    dm.gene_survey(df_tx, "[testset retrieval] After dropping duplicates while ensuring all unique genes are retained ...")
    
    # After this process, there might still be transcripts that are associated with multiple genes 
    # (if none of the associated genes are associated with another transcript).

    df_tx = ensure_unique_transcript_to_gene_mapping(df_tx)
    # NOTE: This may remove genes 

    # transcript_gene_counts = df_tx.groupby(col_tid)[col_gid].nunique()
    # multiple_genes = transcript_gene_counts[transcript_gene_counts > 1]
    # if multiple_genes.shape[0] > 0:
    #     print(f"[info] Still found non-zero transcripts associated with multiple genes after de-dupilcation step: n={multiple_genes.shape[0]}")
    #     df_tx.drop_duplicates(subset=[col_tid, ], inplace=True)

    # Group by transcript ID and keep the first gene ID for each transcript
    # df_tx = df_tx.groupby(col_gid).first().reset_index()
    print(f"[info] Found any duplicates in gene-tx combination? {df_tx.shape[0]} <? {N0}")
    dm.gene_survey(df_tx, "[testset retrieval] After dropping all duplicates to ensure a one-to-one mapping between genes and transcripts")

    # --- Output ---

    print(f"[output] Saving transcript dataframe (ID={tx_io.ID}) ...")
    tx_io.save_tx(df_tx, verbose=1)

    df_txid = df_tx[[col_tid, col_gid]]

    print(f"[output] Saving transcript & gene IDs ...")
    tx_io.save_tx_id(df_txid) # Initial set of tx_ids in the test data 

    return df_tx

def retrieve_and_featurize_testset(concept, **kargs): 

    target_biotype = concept.biotype
    target_suffix = concept.suffix
    col_tid = TranscriptIO.col_tid
    col_gid = TranscriptIO.col_gid

    txio_testset = TranscriptIO(biotype=target_biotype, suffix=target_suffix) # nmd_eff, testset

    # Action parameters 
    retrieve_transcripts = kargs.get("retrieve_transcripts", True)
    retrieve_exons = kargs.get("retrieve_exons", False)
    retrieve_sequences = kargs.get("retrieve_sequences", False)
    retrieve_tx_data = (retrieve_transcripts 
                        or retrieve_exons 
                        or retrieve_sequences)

    featurize = kargs.get("featurize", False)
    test = kargs.get("test", featurize)
    
    # Input modes
    genes = kargs.get("genes", [])
    transcript_df = kargs.get("transcript_df", None)
    
    if retrieve_transcripts: 
        max_n_trpts = kargs.get("max_tx_per_gene", 200)

        if len(genes) > 0:
            print("[input] Given a list of gene names ...")
            df_tx = retrieve_testset_given_genes(txio_testset,
                        genes=genes,
                        random_subset_transcripts=True, 
                        max_tx_per_gene=max_n_trpts,  # if max_n_trpts is None, then do not subset
                            include_principal_isoforms=True)
        elif transcript_df is not None:
            print("[input] Given a transcript dataframe ...")

            gene_set = transcript_df[col_gid].unique() if col_gid in transcript_df else None

            df_tx = retrieve_testset_given_transcripts(txio_testset, 
                        tx_set=transcript_df[col_tid].unique(), 
                        gene_set =gene_set)
        else: 
            n_genes = kargs.get("n_genes", 100) 
            print(f"[input] Randomly sampling genes (n={n_genes}) ...")

            df_tx = retrieve_testset_given_genes(txio_testset, 
                        n_genes=n_genes, 
                        max_tx_per_gene=max_n_trpts, 
                        random_subset_transcripts=True, 
                            include_principal_isoforms=True)
            genes = df_tx[col_gid].unique()

    test_data_pipeline_feature_extraction(concept,
            retrieve_tx_exon_data=retrieve_exons, 
            run_sequence_retrieval=retrieve_sequences,  # NOTE: marker sequences are generated in this step
                run_feature_extraction=featurize, 
                    run_extract_features=True,
                    featurize_sequences=True,
                        run_pos_hoc_ops=retrieve_tx_data and featurize)

    # After the above step, we should have the featurized transcriptomic test set
    if test: 
        descriptor = SequenceDescriptor(biotype=target_biotype, suffix=target_suffix)
        # df_dtor = descriptor.load_transcript_features()
        df_dtor, filepath = descriptor.load_full_transcript_features(return_data_path=True)
        # assert df_dtor is not None and not df_dtor.empty
        if df_dtor is not None and not df_dtor.empty: 
            dm.gene_survey(df_dtor, "[featurize] After fully featurize test set transcripts")

        # Design: Predict biotypes here? 

    return

def run_gene_expression_analysis(concept, **kargs):
    # Enable the following action parameters
    if not 'determine_treatment_and_control' in kargs: 
        kargs['determine_treatment_and_control'] = True
    if not 'prepare_tpm_matrix' in kargs: 
        kargs['prepare_tpm_matrix'] = True

    # Disable the following action paramters
    kargs['retrieve_tx_exon_data'] = False
    kargs['run_feature_extraction'] = False
    kargs['run_sequence_retrieval'] = False
    kargs['auto_labeling'] = False
    kargs['run_pos_hoc_ops'] = False 
    return test_data_pipeline(concept, **kargs)

def test_data_pipeline_feature_extraction(concept, *, 
        retrieve_tx_exon_data=True,
        run_sequence_retrieval=True, # retrieve_sequences=True, 
        run_feature_extraction=True, # featurize_sequences=True, 
        run_pos_hoc_ops=True, 
                **kargs): 
    # from sphere_pipeline.data_model import gene_survey
    import gc

    test = kargs.get("test", False)
    verbose = kargs.get("verbose", 1)
    col_btype = const.col_btype
        
    base_concept = 'nmd_eff'
    if concept is None: 
        suffix = txio_id = kargs.get("txio_id", "testset") # "suffix" serves as part of TranscriptIO's ID
        dataset = kargs.get("dataset", "normal-gtex")
        concept = Concept(concept=base_concept, suffix=txio_id)
    else: 
        suffix = txio_id = concept.suffix 
        dataset = concept.dataset

    pos_biotypes = kargs.get("pos_biotypes", concept.pos_biotypes)  # constituent biotypes associated with ... 
    # ... the treatment group with the base concept (e.g. nmd_eff)
    neg_biotypes = kargs.get("neg_biotypes", concept.neg_biotypes) # constituent biotypes associated with ... 
    # ... the control group with the base concept (e.g. nmd_eff)

    treatment_biotype = concept.treatment_biotype  # collective name/biotype for the treatment group
    control_biotype = concept.control_biotype # collective name/biotype for the control group

    highlight(f"[prediction] Gathering transcriptomic data with labeling concept={concept.concept} ...")
    print(f"... treament biotype: {treatment_biotype}")
    print(f"...... constituents: {pos_biotypes}")
    print(f"... control biotype:  {control_biotype}")
    print(f"...... constituents: {neg_biotypes}")

    # Final transcript dataset ID
    target_biotype = kargs.get("target_biotype", concept.biotype) # "combined" by default
    target_suffix = kargs.get("target_suffix", concept.suffix) # "testset" by default
    new_source = {base_concept: {'biotype': target_biotype, 'suffix': target_suffix}}
    assert not DataSource.has_conflict_with_reserved_source_ids(new_source)

    print(f"[prediction] Final test set transcript data can be identified via:")
    print(f"... target biotype: {target_biotype}")
    print(f"... target suffix: {target_suffix}")
    
    col_tid = TranscriptIO.col_tid 
    col_gid = TranscriptIO.col_gid
    query_processed = False

    ###############################################
    output_files = {}
    txio_testset = TranscriptIO(biotype=target_biotype, suffix=target_suffix) # nmd_eff, testset
    
    highlight(f"Loading the initial test set in transcript dataframe ...")
    print(f"... transcript data ID: {txio_testset.ID}")
    df_tx = txio_testset.load_tx()

    if df_tx is None or df_tx.empty: 
        genes = kargs.get("genes", [])
        n_genes = kargs.get("n_genes", 1000)
        df_tx = retrieve_testset_given_genes(txio_testset, genes=genes, n_genes=n_genes) 

    n_genes_init = df_tx[col_gid].nunique()
    n_trpts_init = df_tx[col_tid].nunique()
    dm.gene_survey(df_tx, f"[input] Initial test set in df_tx")
    # print(f"... n(genes) in the initial test set: {n_genes_init}")
    # print(f"... n(trpts) in the initial test set: {n_trpts_init}")
    
    # --- Test ---
    columns_to_check = [col_tid, col_gid, ]
    assert all(col in df_tx.columns for col in columns_to_check)
    assert df_tx[col_tid].nunique() == df_tx.shape[0], "Each transcript should be unique but duplicates are found."
    assert not has_transcripts_mapped_to_multiple_genes(df_tx), "Each transcript should be associated with only one gene."
        
    # Predict biotypes with results going to tx_type_pred 
    # Design: This is delegated to retrieve_and_featurize_testset()
    # NOTE: To predict transcript biotypes, we need to featurize them first but to featurize them, we need tx_exon data

    # Fill in tx_pred_ref if annotation not available from the known source  
    # Design: This is delegated to retrieve_and_featurize_testset()

    df_tx_id = txio_testset.load_tx_id() # Initial set of tx_ids in the test data 
    assert set(df_tx_id[col_tid]) == set(df_tx[col_tid])

    ###############################################

    if retrieve_tx_exon_data:  
        highlight("[data pipeline] Retrieving transcript-exon data given the target transcripts (in df_tx) ...")

        assert df_tx is not None and not df_tx.empty

        tx_set = df_tx[col_tid].unique()
        assert len(tx_set) > 0
        N0 = len(tx_set)
        print(f"[info] Found n={len(tx_set)} unique transcripts")
        print(f"... n_genes: {df_tx[col_gid].nunique()}")
        
        sql_template = query_templates['select_from_tx_ex_given_tx_set']
        template_params = {'txdb_transcripts': Txdb.transcripts, 'txdb_exons': Txdb.exons, }
        query_result_path = txio_testset.tx_ex_path
        run_batched_query(tx_set,  # break this large transcript set into batches
            sql_template, template_params, 
                tx_table_ref='t', in_predicate_first=True,
                batch_size=25000, 
                    output_path=query_result_path, use_dask=False)

        # df_tx_ex may have duplicates in terms of tx_id and exon_id combinations
        df_tx_ex = txio_testset.load_tx_ex(is_featurized=False) 
        tx_ex_set = df_tx_ex[col_tid].unique()
        print(f"... found n={len(tx_ex_set)} =?= {N0} unique transcripts")
        # NOTE: If len(tx_ex_set) < len(tx_set), then a subset of the transcripts are not consistently documented 
        #       in txdb.exons
        df_tx_ex = remove_duplicate_exons(df_tx_ex)

        output_files['df_tx_tx'] = txio_testset.save_tx_ex(df_tx_ex, is_featurized=False)
        query_processed = True

    ### End retrieve_tx_exon_data
    if query_processed: 
        # Sleep for a few seconds
        time.sleep(10)
            
    if run_sequence_retrieval: 
        # import sphere_pipeline.seq_analyzer as sa
        highlight("[data pipeline] Retrieving transcript sequences ...")

        retrieve_sequences = kargs.get("retrieve_sequences", True)
        download_sequences = kargs.get("download_sequences_from_blob", False)
        generate_markers = kargs.get("generate_markers", True)
        trim_introns = kargs.get("trim_introns", True) # save the trimmed data in a separate file

        if retrieve_sequences:
            to_str = kargs.get("to_str", True) # convert Bio.Seq to regular string 

            highlight("[retrieval] Retrieving transcript sequences ...")
            print(f"... download sequences from blob? {download_sequences}")

            temp_dir = txio_testset.cache_dir
            retrieve_transcript_sequences_incrementally(txio_testset, batch_size=30000, temp_dir=temp_dir, 
                download_sequences_from_blob=download_sequences, 
                    generate_markers=generate_markers, trim_introns=trim_introns, 
                        load_tx_id_target=False, to_str=to_str)

            # --- Test ---
            # print("[data] Reading sequence dataframe ...")
            # is_dask_dataframe = False
            # df_seq = txio_testset.load_tx_seq()

            # if isinstance(df_seq, dd.DataFrame): 
            #     is_dask_dataframe = True
    
    # Release memory
    # print("[info] Optionally release memory associated with transcript dataframe (from txdb.transcripts)...")
    # del df_tx 
    # gc.collect()    
    
    if run_feature_extraction: 
        highlight("[feature extraction] Running feature extraction on transcript-exon data ...")

        df_tx = txio_testset.load_tx()  # Load transcript data
        res_tx = data_survey(df_tx, f"[feature extraction] Transcript dataframe with ID={txio_testset.ID}")
        print(f"... columns(df_tx):\n{list(df_tx.columns)}\n")

        # df_tx.drop_duplicates(subset=['tx_id', 'gene_id'], inplace=True)  # Drop duplicates based on tx_id and gene_id
        # res_tx = data_survey(df_tx, f"[feature extraction] De-duplicated transcript dataframe with ID={txio_testset.ID}")

        # Action parameters 
        run_extract_features = kargs.get("run_extract_features", True)
        featurize_sequences = kargs.get("featurize_sequences", True) 
        combine_by_biotype = kargs.get("combine_by_biotype", True)

        if run_extract_features: 
     
            # txio_testset = TranscriptIO(biotype=target_biotype, suffix=target_suffix)
            # tx_set = df_tx[col_tid].unique() 

            df_tx_ex = txio_testset.load_tx_ex()  # Load transcript-exon data
            res_tx_ex = data_survey(df_tx_ex, f"[feature extraction] Transcript-exon data with ID={txio_testset.ID}")

            print(f"[feature extraction] Found n={res_tx_ex['n_trpts']} =?= ({res_tx['n_trpts']}) unique transcripts at tx-exon level")

            extract_features(txio_testset, df_tx_ex=df_tx_ex, consolidate_transcripts=False)

        print("[I/O] Loading featurized transcript data ...")
        df_featurized = txio_testset.load_tx_ex(is_featurized=True, verbose=1) 
        data_survey(df_featurized, f"[info] Featurized transcript data (df_featurized)")
        print("... shape(df_featurized): {}".format(df_featurized.shape))
        print("... n(trpts): {}".format(df_featurized[col_tid].nunique()))

        df_tx_filtered = df_tx[df_tx[col_tid].isin(df_featurized[col_tid])]
        data_survey(df_tx_filtered, f"[info] Filtered transcript dataframe (df_tx_filtered)") 

        # Add genes and biotype info
        if not col_gid in df_featurized.columns: 
            print("[action] Adding gene information ...")
            df_featurized = df_featurized.merge(df_tx[[col_tid, col_gid]], on=col_tid, how='inner')
            print("... n(genes): {}".format(df_featurized[col_gid].nunique()))

        if not col_btype in df_featurized.columns: 
            contains_nan = df_tx[col_btype].isna().any()
            
            # NOTE: At this stage, some of the transcripts may not have biotype information
            # assert not contains_nan, \
            #     "Biotype information should be available for all transcripts. Called predict_biotypes()?"
            if contains_nan: 
                # Count the number of unique transcripts with NaN biotype information
                num_tx = df_tx[col_tid].nunique()
                num_nan_biotypes = df_tx[df_tx[col_btype].isna()][col_tid].nunique()
                print(f"... n={num_nan_biotypes}(/{num_tx}) transcripts with NaN biotype information")
                print("[decision] Skippig biotypes at this stage ...")
            else: 
                print("[action] Adding biotype information ...")
                # df_tx_filtered = df_tx[df_tx[col_tid].isin(df_featurized[col_tid])]
                
                shape0 = df_featurized.shape[0]
                df_featurized = df_featurized.merge(df_tx[[col_tid, col_btype]], on=col_tid, how='inner')
                assert shape0 == df_featurized.shape[0], "Mismatch in the number of rows after merging biotype information"
                # NOTE: assertion error may be raised if there are duplicate tx_ids in df_tx

                # df_featurized = df_featurized.merge(df_tx[[col_tid, col_btype]], on=col_tid, how='inner')
                # df_featurized = ud.merge_in_chunks(df_featurized, df_tx[[col_tid, col_btype]], col_tid, chunk_size=10000)

        # Reorder columns to ensure that dense columns are first
        feature_columns = TranscriptIO.feature_columns
        meta_columns = [col for col in df_featurized.columns if not col in feature_columns]
        column_order = meta_columns + feature_columns # [col for col in df_featurized.columns if col not in meta_columns]
        df_featurized = df_featurized[column_order] 

        print("(data_pipeline) Saving updated featurized tx-exon data")
        output_files['df_tx_ex_featurized'] = \
            txio_testset.save_tx_ex(df_featurized, is_featurized=True, sep='\t')
 
        # --- Test --- 
        n_genes_featurized = df_featurized[col_gid].nunique()
        n_trpts_featurized = df_featurized[col_tid].nunique()    
        print(f"... n_gene_featurized: {n_genes_featurized} =?= {n_genes_init}")
        print(f"... n_trpts_featurized: {n_trpts_featurized} =?= {n_trpts_init}")
        print(f"... shape(df_featurized): {df_featurized.shape}")  
        print(f"... example tx_ids:\n{list(df_featurized.sample(n=5)[col_tid])}\n")  
        print(f"[info] List of transcript features")
        for i, col in enumerate(df_featurized.columns): 
            print(f"... [{i+1}] {col}") 

        if featurize_sequences: 
            overwrite = kargs.get("overwrite", True)
            trim_features = kargs.get("trim_features", False) # Don't trim features by default (e.g. removing constant features)

            descriptor = SequenceDescriptor(biotype=target_biotype, suffix=target_suffix) # concept=labeling_concept

            df_dtor = descriptor.load_transcript_features()
            if overwrite or (df_dtor is None or df_dtor.empty): 
                # Load sequence-featurized dataset or if doesn't already exist, then run feature extraction on the transcript's sequence
                print("> Featurizing transcripts via their sequences ...")
                df_dtor = dm.featurize_transcript_sequences(descriptor, 
                                use_cached=not overwrite, 
                                    trim_features=trim_features, 
                                    verbose=verbose)
        
                # Test
                df_dtor_prime = descriptor.load_transcript_features()
                assert df_dtor.shape == df_dtor_prime.shape 
        
    if run_pos_hoc_ops:  
        
        print(f"[info] Mapping transcript source IDs (biotype={target_biotype}, suffix={target_suffix}) ...")
        map_tx_source_id(txio_testset)

        # Assigning each transcript (ID) to its associated gene
        print(f"[info] Mapping gene source IDs (biotype={target_biotype}, suffix={target_suffix}) ...")
        map_gene_source_id(txio_testset) 

    print("#" * 80); print()    

    return output_files


def add_gene_info(df, tx_io=None, **kargs):
    col_gid = TranscriptIO.col_gid
    col_tid = TranscriptIO.col_tid

    if tx_io is None: 
        biotype = kargs.get("biotype", 'nmd_eff')
        suffix = kargs.get("suffix", 'testset')
        tx_io = TranscriptIO(biotype=biotype, suffix=suffix)

    # Load the total set of transcripts (from the input of the pipeline)
    df_txid = tx_io.load_tx_id()

    if col_gid in df.columns:
        print(f"[info] Column {col_gid} already exists in the dataframe. Overwriting it ...")
        df.drop(columns=[col_gid], inplace=True)
        
    print("[action] Adding gene information ...")
    df = df.merge(df_txid[[col_tid, col_gid]], on=col_tid, how='inner')

    return df

def add_biotype_info(df, tx_io=None, **kargs):
    col_gid = TranscriptIO.col_gid
    col_tid = TranscriptIO.col_tid
    col_btype = TranscriptIO.col_btype

    raise_exception = kargs.get("raise_exception", False)
    # overwrite = kargs.get("overwrite", True)

    if tx_io is None: 
        biotype = kargs.get("biotype", 'nmd_eff')
        suffix = kargs.get("suffix", 'testset')
        tx_io = TranscriptIO(biotype=biotype, suffix=suffix)

    # Load the total set of transcripts (from the input of the pipeline)
    df_tx = tx_io.load_tx(use_dask=False) 
    assert isinstance(df_tx, pd.DataFrame)

    # Test: Why are there still duplicates in the transcript dataframe?
    df_tx.drop_duplicates(subset=[col_tid, col_gid], inplace=True)  # Drop duplicates based on tx_id and gene_id

    data_survey(df_tx, f"[add_biotype_info] Transcript dataframe to include biotype information (ID={tx_io.ID})")

    if col_btype in df.columns:
        print(f"[info] Column {col_btype} already exists in the dataframe. Overwriting it ...")
        df.drop(columns=[col_btype], inplace=True)

    print("(add_biotype_info) Adding biotype information ...")
    contains_nan = df_tx[col_btype].isna().any()

    if contains_nan: 
        # Count the number of unique transcripts with NaN biotype information
        num_tx = df_tx[col_tid].nunique()
        num_nan_biotypes = df_tx[df_tx[col_btype].isna()][col_tid].nunique()
        msg = f"... n={num_nan_biotypes}(/{num_tx}) transcripts with NaN biotype information"

        if raise_exception:
            raise ValueError(msg)
        else: 
            print(msg)

    df = df.merge(df_tx[[col_tid, col_btype]], on=col_tid, how='inner')

    return df 

def lookup_appris_annotation(tx_set=None, gene_set=None, biotype='protein_coding', **kargs): 
    """
    
    Memo
    ----
    1. This function is slightly different than the version in process_trpts_from_synapse
    """
    # from synapse.query_templates import query_templates
    if tx_set is None: tx_set = []
    if gene_set is None: gene_set = []

    suffix = kargs.get("suffix", None)
    sql_template = kargs.get('sql_template', 
        query_templates['retrieve_tx_gene_set_with_appris_constraints_given_genes'])
    # NOTE: Alternative template options: 
    #   - query_templates['lookup_ann_given_tx_set']
    #   - query_templates['retrieve_tx_gene_set_with_appris_constraints']

    tx_io = TranscriptIO(biotype=biotype, suffix=suffix)

    # if tx_set is None: 
    #     df_tx = tx_io.load_tx_id()
    #     tx_set = df_tx['tx_id'].unique()

    if len(gene_set) > 0:
        gene_id_condition = f"WHERE t.gene_id IN ({','.join([repr(gene) for gene in gene_set])})"
    else:
        gene_id_condition = ""

    template_params = {'gene_id_condition': gene_id_condition, 
                       "txdb_transcripts": Txdb.transcripts, "txdb_tx_ann": Txdb.tx_ann}
    # NOTE: 'tx_id_condition'? run_batched_query() will handle the IN-predicate for tx_set

    query_result_path = tx_io.tx_appris_path
    query_cache_dir = tx_io.cache_dir
    run_batched_query(tx_set,  
        sql_template, template_params, 
            tx_table_ref='a', in_predicate_first=False,
            batch_size=25000, 
                output_path=query_result_path,
                temp_dir=query_cache_dir, verbose=2, use_dask=False)
    
    return

def subset_transcripts_by_tx_type_pred(df_tx, tx_types=None):
    """
    Subset transcripts by the predicted biotype from SequenceSphere 

    Parameters:
    - df_tx: DataFrame containing columns 'tx_id' and 'tx_type_pred'

    Returns:
    - DataFrame with subsetted transcripts
    """
    col_tid = TranscriptIO.col_tid
    col_btype_pred = TranscriptIO.col_btype_pred

    if tx_types is None: tx_types = ['nmd', 'protein_coding']
    if isinstance(tx_types, str): tx_types = [tx_types, ]
    
    df_tx = df_tx[df_tx[col_btype_pred].isin(tx_types)]
    return df_tx

def retrieve_predicted_nmd_fated_transcripts(genes, **kargs):
    # See process_trpts_from_synpase.select_transcripts_matching_biotypes()
    pass 

def retrieve_principle_isoforms(genes, **kargs): 

    target_biotype = kargs.get('biotype', 'protein_coding')
    target_suffix = kargs.get('suffix', 'testset')
    coding_biotypes = ['protein_coding', ]  # concept.neg_biotypes # constituent biotypes associated with ... 
    if genes is None: genes = []

    target_genes = genes
    target_principal_flags = kargs.get("target_principal_flags", [1, 2, 6, ])
    process_treatment_on_gene_level = kargs.get("process_treatment_on_gene_level", True)
    col_tid = TranscriptIO.col_tid
    col_gid = TranscriptIO.col_gid

    df_appris_set = []
    appris_dict = {}
    sql_template = kargs.get('sql_template', 
        query_templates['retrieve_tx_gene_set_with_appris_constraints_given_genes'])

    # NOTE: Other template options: 
    #       - query_templates['retrieve_tx_gene_set_with_appris_constraints']
    overwrite = kargs.get('overwrite_result_set', True)

    for biotype in coding_biotypes: 
        tx_io = TranscriptIO(biotype=biotype, suffix=target_suffix)

        df_appris = tx_io.load_tx_appris()
        if overwrite or (df_appris is None or df_appris.empty): 
            print(f"[action] Fetching APPRIS annotation for ID=({biotype}, {target_suffix}) ...")

            lookup_appris_annotation(
                tx_set=None, # Set to None to consider all possible transcripts
                gene_set=genes,
                biotype=biotype, 
                suffix=target_suffix, 
                sql_template=sql_template
            )  
            # By default, will retrieve all tx in SequenceSphere that satisfy ...
            # ... APPRIS-specific constraint (non-null principal flags, appris_score > 0)
            # NOTE: At the moment, genes are not part of the where-clause constraint in lookup_appris_annotation

            df_appris = tx_io.load_tx_appris()
            print(f"... lookup_appris_annotation() -> df_appris with shape: {df_appris.shape}")

            # Test 
            n = df_appris[(df_appris['appris_score'] < 0)].shape[0]
            print(f"... n(rows) for which appris score < 0: {n} =?= 0")

        # print(f"(retrieve_principle_isoforms) Loading qualified constrol set of data ID: ({biotype}, {target_suffix}) ...")
        # df_tx = tx_io.load_tx_id()
        # dm.gene_survey(df_tx, f"Initial qualified control set ({biotype} by default)")
        
        print("(retrieve_principle_isoforms) filtering initial tx_ids by target gene set ...")

        if len(target_genes) > 0: 
            df_appris = df_appris[df_appris[col_gid].isin(target_genes)] 
            dm.gene_survey(df_appris, "[principal_isoforms] Target genes with tx satisfying APPRIS constraints")

        ########################################
        # Secondary filtering logic for principal isoforms
        # NOTE: The SQL query will give us an initial gene_ids and tx_ids that satisfy APPRIS-specific constraints
        #       but if we wish to be more stringent in the constraints, then we can apply them here on the Pandas level
            
        df_appris['principal_flag'] = df_appris['principal_flag'].astype(int)
        # NOTE: Convert 'principal_flag' column to integer, keeping NaN as NaN
        #       df_appris['principal_flag'] = pd.to_numeric(df_appris['principal_flag'], errors='coerce').astype('Int64') 
            
        # Filter df_appris based on the conditions
        df_appris = df_appris[(df_appris['appris_score'] > 0) & (df_appris['principal_flag'].isin(target_principal_flags))]

        # Get the transcripts that meet the conditions
        filtered_tx_ids = df_appris[col_tid].unique()
        print(f"... Number of unique transcripts in df_appris: {len(filtered_tx_ids)}")

        # Filter df_tx based on the transcriptss
        # df_tx = df_tx[df_tx[col_tid].isin(filtered_tx_ids)]

        # dm.gene_survey(df_tx, msg=f"Protein-coding trpts with principal flag in {target_principal_flags}")

        df_appris_set.append(df_appris)
        ########################################

        # Now create another dictionary mapping tx_id to its appris score (the higher the more reliable)
        appris_dict.update(df_appris.set_index(col_tid)['appris_score'].to_dict())
        print(f"[info] Example APPRIS scores:\n{ut.sample_dict(appris_dict, 20)}\n")

    df_appris = pd.concat(df_appris_set, ignore_index=True)
    print(f"... columns(df_appris):\n{list(df_appris.columns)}\n")

    return df_appris


def filter_principal_isoforms_with_common_genes(df_id_treat, df_id_ctrl, combine_df=False):
    # Identify common genes in both treatment and control groups
    common_genes = set(df_id_treat[col_gid]).intersection(set(df_id_ctrl[col_gid]))
    
    # Filter treatment and control dataframes to include only common genes
    df_id_treat = df_id_treat[df_id_treat[col_gid].isin(common_genes)]
    df_id_ctrl = df_id_ctrl[df_id_ctrl[col_gid].isin(common_genes)]
    
    # Initialize an empty DataFrame for the filtered control group
    filtered_ctrl = pd.DataFrame(columns=df_id_ctrl.columns)
    
    # Process each gene
    for gene in common_genes:
        ctrl_group = df_id_ctrl[df_id_ctrl[col_gid] == gene]

        # Check if there are principal isoforms for this gene
        if 'is_principal_isoform' in ctrl_group.columns and ctrl_group['is_principal_isoform'].any():
            # Keep only principal isoforms
            principal_isoforms = ctrl_group[ctrl_group['is_principal_isoform'] == True]
            filtered_ctrl = pd.concat([filtered_ctrl, principal_isoforms], ignore_index=True)
        else:
            # Keep all control transcripts if no principal isoform is present
            filtered_ctrl = pd.concat([filtered_ctrl, ctrl_group], ignore_index=True)

    if combine_df: 
        # Combine the treatment and filtered control groups
        final_df = pd.concat([df_id_treat, filtered_ctrl], ignore_index=True)
        return final_df
    return df_id_treat, filtered_ctrl

def filter_principal_isoforms(df_id_ctrl):
    """
    Filter the control group transcripts to keep principal isoforms when available.
    If a gene does not have any principal isoforms, retain all regular protein-coding transcripts.

    Parameters:
    df_id_ctrl : DataFrame
        A DataFrame consisting of at least two columns: tx_id and gene_id,
        and a column 'is_principal_isoform' indicating if the transcript is a principal isoform.

    Returns:
    DataFrame
        A DataFrame containing only the desired protein-coding transcripts for each gene.
    """

    # Function to process each group
    def filter_isoforms(group):
        if group['is_principal_isoform'].any():
            return group[group['is_principal_isoform']]
        return group

    # Apply the function to each gene group
    filtered_ctrl = df_id_ctrl.groupby('gene_id').apply(filter_isoforms).reset_index(drop=True)

    return filtered_ctrl

def has_transcripts_mapped_to_multiple_genes(df, predicate_mode=True):
    col_tid = const.col_tid
    col_gid = const.col_gid    

    # Check how many transcripts are mapped to multiple genes
    transcript_gene_counts = df.groupby(col_tid)[col_gid].nunique()
    multi_gene_transcripts = transcript_gene_counts[transcript_gene_counts > 1]
    # NOTE: multi_gene_transcripts is a Series that includes only the transcripts that are mapped to more than one gene

    # Print the number of transcripts mapped to multiple genes
    print(f"[validate] Number of transcripts mapped to multiple genes: {len(multi_gene_transcripts)}")

    if not predicate_mode:  
        return multi_gene_transcripts

    return len(multi_gene_transcripts) > 0

def ensure_unique_transcript_to_gene_mapping(df, **kargs):
    col_tid = const.col_tid
    col_gid = const.col_gid    

    multi_gene_transcripts = has_transcripts_mapped_to_multiple_genes(df, predicate_mode=False)

    # If any transcripts are mapped to multiple genes, choose one gene for each
    if len(multi_gene_transcripts) > 0:
        print("[resolve] Resolving transcripts mapped to multiple genes...")
        df = df.groupby(col_tid).first().reset_index()

    # Check that each transcript is now mapped to only one gene
    assert df[col_tid].nunique() == df.shape[0], "Each transcript should be unique but duplicates are found."

    return df

def gene_level_jaccard_similarity(df_id_treat, df_id_ctrl, col_gid='gene_id', verbose=0):

    res = {}
    res['union'] = set(df_id_treat[col_gid]).union(set(df_id_ctrl[col_gid]))

    # Identify common genes in both treatment and control groups
    res['intersection'] = res['common_genes'] = set(df_id_treat[col_gid]).intersection(set(df_id_ctrl[col_gid]))

    # Genes in treatment group but not in control group
    res['treat_not_ctrl'] = set(df_id_treat[col_gid]).difference(df_id_ctrl[col_gid])

    # Genes in control group but not in treatment group
    res['ctrl_not_treat'] = set(df_id_ctrl[col_gid]).difference(df_id_treat[col_gid])

    n_genes = len(res['union'])
    n_genes_common = len(res['intersection'])
    jaccard_index = n_genes_common/ n_genes

    if verbose:
        print_emphasized("(gene_level_jaccard_similarity) Gene-level Jaccard similarity ...")
        print("... jaccard index (gene-level): %.3f" % jaccard_index)
        print("... size(union): %d, size(intersection): %d" % (len(res['union']), len(res['intersection'])))
        print("... n(genes) in treatment but not control: %d" % len(res['treat_not_ctrl']))
        print("... n(genes) in control but not treatment: %d" % len(res['ctrl_not_treat']))

    return jaccard_index, res
                       
def test_data_pipeline(concept=None, *, 
        determine_treatment_and_control=True, 
        prepare_tpm_matrix=True, 
        retrieve_tx_exon_data=True, 
        run_sequence_retrieval=True, # retrieve_sequences=True, 
        run_feature_extraction=True, # featurize_sequences=True, 
        auto_labeling=False, run_pos_hoc_ops=False, 
        # gather_data_for_protein_coding_prediction=False,
                **kargs):
    """
    Gather the transcriptomic test dataset(s) that satisfy NMD efficiency-specific constraints: 
    - Genes must have well-defined treatment and control groups, where the treatment group consists of 
        NMD targets and the control group consists of protein-coding transcripts
    - All transcripts must have valid TPM values
    - APPRIS-specific constraints for protein-coding transcripts are dropped at the moment because 
      the genes and transcripts in the test set probably do not have APPRIS annotation available.

    Memo
    ----
    TPM matrix preparation
        Create an intermediate dataframe that simplifies the extraction of TPM values by sample IDs for each transcript. 
        For this purpose, we need to focus on three columns: tx_id, sample_id, and tpm. The goal is to ensure 
        that each combination of tx_id and sample_id is unique, which can be achieved by dropping duplicates.

    """
    # from sphere_pipeline.data_model import gene_survey
    from itertools import chain

    test = kargs.get("test", False)
    verbose = kargs.get("verbose", 1)
    col_btype = kargs.get("col_biotype", "tx_type_ref")
        
    base_concept = 'nmd_eff'
    if concept is None: 
        suffix = txio_id = kargs.get("txio_id", "testset") # "suffix" serves as part of TranscriptIO's ID
        dataset = kargs.get("dataset", "normal-gtex")
        concept = Concept(concept=base_concept, suffix=txio_id)
    else: 
        suffix = txio_id = concept.suffix 
        dataset = concept.dataset

    pos_biotypes = kargs.get("pos_biotypes", concept.pos_biotypes)  # constituent biotypes associated with ... 
    # ... the treatment group with the base concept (e.g. nmd_eff)
    neg_biotypes = kargs.get("neg_biotypes", concept.neg_biotypes) # constituent biotypes associated with ... 
    # ... the control group with the base concept (e.g. nmd_eff)

    treatment_biotype = concept.treatment_biotype  # collective name/biotype for the treatment group
    control_biotype = concept.control_biotype # collective name/biotype for the control group
    biotype_groups = concept.biotype_groups
    print(f"[test] biotype_groups:\n{biotype_groups}\n")

    highlight(f"[prediction] Gathering transcriptomic data with labeling concept={concept.concept} ...")
    print(f"... treament biotype: {treatment_biotype}")
    print(f"...... constituents: {biotype_groups[treatment_biotype]} =?= {pos_biotypes}")
    print(f"... control biotype:  {control_biotype}")
    print(f"...... constituents: {biotype_groups[control_biotype]} =?= {neg_biotypes}")

    # Transcriptomic dataset ID used for the model training phase
    source_biotype = kargs.get("source_biotype", NMDEffModel.source_biotype)
    source_suffix = kargs.get("source_suffix", NMDEffModel.source_suffix)

    # Final transcript dataset ID used for inference (which identifies the test set)
    target_biotype = kargs.get("target_biotype", concept.biotype) # "combined" by default
    target_suffix = kargs.get("target_suffix", concept.suffix) # "testset" by default
    new_source = {base_concept: {'biotype': target_biotype, 'suffix': target_suffix}}
    assert not DataSource.has_conflict_with_reserved_source_ids(new_source)
    
    col_tid = TranscriptIO.col_tid 
    col_gid = TranscriptIO.col_gid

    # TPM parameters 
    sparsified_tpm = kargs.get("sparsify_tpm", False) # Storing TPM matrix in sparse format? 
    
    ###############################################
    output_files = {} # Keep track of all the output files
    txio_testset = TranscriptIO(biotype=target_biotype, suffix=target_suffix) # nmd_eff, testset
    
    highlight("[pipeline] Loading the initial test set in transcript dataframe ...")
    df_tx = txio_testset.load_tx()

    if df_tx is None or df_tx.empty: 
        genes = kargs.get("genes", [])
        n_genes = kargs.get("n_genes", 1000)
        df_tx = retrieve_testset_given_genes(txio_testset, genes=genes, n_genes=n_genes) 

    n_genes_init = df_tx[col_gid].nunique()
    n_trpts_init = df_tx[col_tid].nunique()
    print(f"... n(genes) in the initial test set: {n_genes_init}")
    print(f"... n(trpts) in the initial test set: {n_trpts_init}")

    # --- Test ---
    columns_to_check = [col_tid, col_gid, ]
    assert all(col in df_tx.columns for col in columns_to_check)
        
    # Predict biotypes with results going to tx_type_pred 
    # Design: This is delegated to retrieve_and_featurize_testset()
    # NOTE: To predict transcript biotypes, we need to featurize them first but to featurize them, we need tx_exon data

    # Fill in tx_pred_ref if annotation not available from the known source  
    # Design: This is delegated to retrieve_and_featurize_testset()

    df_txid = df_tx[[col_tid, col_gid]]
    output_files['df_txid'] = txio_testset.save_tx_id(df_txid) # Initial set of tx_ids in the test data 
    
    ###############################################
    highlight(f"> Determining the sample set for TPM matrix (dataset={dataset}) ...")
    use_samples_in_trainset = kargs.get("use_samples_in_trainset", False)

    # ------------------------------------

    # Meta data parameters
    samples = kargs.get("samples", [])
    user_provided_samples = True

    df_meta = select_samples_by_dataset(keyword=dataset, return_dataframe=True, verbose=1)
    sample_id_to_name = df_meta.set_index('sample_id')['sample_name'].to_dict()
    # NOTE: Todo: selecting samples via tissue_type

    if samples is None or len(samples) == 0: 
        if use_samples_in_trainset: 

            gem = GEMatrix(dataset=dataset, biotype=source_biotype, suffix=source_suffix) # source='synapse'

            # Find the samples used in the trained model(s)
            df_ge = gem.load(as_dense=True, sparsified_tpm=False)
            cols_dict = gem.get_cols_dict(df_ge)
            num_cols = cols_dict['num_cols']
            cat_cols = cols_dict['cat_cols']
            feature_cols = gem.get_feature_columns(df_ge) 
            assert set(feature_cols) == set(num_cols).union(cat_cols)
            training_samples = num_cols
            print(f"... training data had used n={len(training_samples)} samples (with dataset={dataset})")
            print(f"... example samples:\n{np.random.choice(training_samples, 5)}\n")  # training TPM matrix uses sample names
            del df_ge

            samples = {sid for sid, sn in sample_id_to_name.items() if sn in training_samples} # TPM matrix uses sample names
            print(f"=> Selecting n={len(samples)} samples consistent with the training data")
        else: 
            samples = list(sample_id_to_name.keys())
        user_provided_samples = False
    assert len(samples) > 0
    print_emphasized(f"[decision] Will prepare TPM matrix based on n={len(samples)} samples (given dataset={dataset})")

    ###############################################

    # tx_io = txio_testset = TranscriptIO(biotype=target_biotype, suffix=target_suffix) # nmd_eff, testset
        
    print("[I/O] Loading the initial test set consisting of transcript IDs and other attributes ...")
    df_tx = txio_testset.load_tx()
    nrows0 = df_tx.shape[0]

    constituent_biotypes = pos_biotypes + neg_biotypes # list(chain.from_iterable(biotype_groups.values()))
    # NOTE: Don't use [treatment_biotype, control_biotype]

    print(f"[decision] Selecting only genes and transcripts for which biotypes are in {constituent_biotypes}")
    df_tx_target = df_tx.loc[df_tx[col_btype].isin(constituent_biotypes)]
    # NOTE: Assuming that values in df_tx[col_btype] are not null at this point
    print(f"... n(rows) in the initial test set: {nrows0} -> {df_tx_target.shape[0]} after biotype filtering")
    n_genes = df_tx_target[col_gid].nunique()
    n_trpts = df_tx_target[col_tid].nunique()
    print(f"... after biotype filtered: n(genes)={n_genes}, n(trpts)={n_trpts}")

    print("[action] Splitting the transcript candidates into treatment and control groups ...") 
    df_id_treat = df_tx_target[df_tx_target[col_btype].isin(pos_biotypes)][[col_gid, col_tid]]
    df_id_ctrl = df_tx_target[df_tx_target[col_btype].isin(neg_biotypes)][[col_gid, col_tid]]

    n_genes_treat = df_id_treat[col_gid].nunique()
    n_genes_ctrl = df_id_ctrl[col_gid].nunique()
    n_trpts_treat = df_id_treat[col_tid].nunique()
    n_trpts_ctrl = df_id_ctrl[col_tid].nunique()
    print(f"... treatment group: n(genes)={n_genes_treat}, n(trpts)={n_trpts_treat}")
    print(f"... control group: n(genes)={n_genes_ctrl}, n(trpts)={n_trpts_ctrl}")
    # NOTE: Due to a transcript being associated with multiple genes, the number of genes in the treatment and control groups 
    #       may sum to be greater than unique number of genes (n_genes)
    
    # --- Test --- 
    print(f"... total number of unique genes: {n_genes} <? {n_genes_treat+n_genes_ctrl}")
    print(f"... n_genes intersection (treat ^ ctrl): {len(set(df_id_treat[col_gid]).intersection(set(df_id_ctrl[col_gid])))}")

    if determine_treatment_and_control:
        highlight("[treatment & control] Determining NMD-fated transcripts and their corresponding control set ...")

        prioritize_principal_isoforms = kargs.get("prioritize_principal_isoforms", True)
        match_samples = kargs.get("match_samples", True)
        # process_treatment_on_gene_level = kargs.get("process_treatment_on_gene_level", True)
 
        # ------------------------------------
        print("[info] Looking into APPRIS annotation ...")

        # Target genes 
        target_genes = df_tx[col_gid].unique()
        ng0 = df_id_ctrl[col_gid].nunique()
        nt0 = df_id_ctrl[col_tid].nunique()
        print(f"... initial df_id_ctrl: n(genes)={ng0}, n(trpts)={nt0}")

        # Prioritize principal isoforms (as determined by APPRIS) as the control set; use protein-coding tx ...
        # ... in df_id_ctrl for those genes that do not have principal isoforms identified
        df_appris = retrieve_principle_isoforms(genes=target_genes, suffix=target_suffix) # by default, overwrite_result_set=True
        negative_scores = df_appris[df_appris['appris_score']<0]['appris_score'].values
        assert len(negative_scores) == 0, \
            f"A subset of the appris-matched tx have negative appris scores:\n{negative_scores}\n"

        target_genes_appris = df_appris[col_gid].unique()
        print(f"... Found n={len(target_genes_appris)} (/{len(target_genes)}) genes that match appris criteria")
        
        # Replace transcripts in df_id_ctrl that satisfy the appris-specific constraints 
        # - Use the (col_gid, col_tid) for all genes that can be found in df_appris, 
        #   which are associated with principal isoforms
        # - For genes in df_id_ctrl that do not find a match in df_id_ctrl, use the existing transcripts as they are
        columns_ctrl = df_id_ctrl.columns

        if prioritize_principal_isoforms:
            print_emphasized("[decision] Using principal isoforms as the control set when available")

            tx_appris = df_appris[col_tid].unique()
            df_id_ctrl['is_principal_isoform'] = False
            df_id_ctrl.loc[df_id_ctrl[col_tid].isin(tx_appris), 'is_principal_isoform'] = True
            # n_is_principal = df_id_ctrl['is_principal_isoform'].sum()
            n_is_principal = df_id_ctrl[df_id_ctrl['is_principal_isoform']==True][col_tid].nunique()

            print(f"[info] Among n={n_trpts_ctrl} control transcripts ...")
            print(f"... n={n_is_principal} are APPRIS principal isoforms")
            print(f"... n={n_trpts_ctrl-n_is_principal} are not APPRIS principal isoforms")

            # df_id_ctrl_appris = df_appris[[col_gid, col_tid]]

            # The part of the original control set that does not have a match in df_appris
            # df_id_ctrl_bar = df_id_ctrl[~df_id_ctrl[col_gid].isin(target_genes_appris)] 

            # df_id_ctrl = pd.concat([df_id_ctrl_appris, df_id_ctrl_bar], ignore_index=True)
            # NOTE: This could alter the original transcript set in df_id_ctrl

            df_id_ctrl = filter_principal_isoforms(df_id_ctrl)
            ng = df_id_ctrl[col_gid].nunique()
            nt = df_id_ctrl[col_tid].nunique()
            n_is_principal_prime = df_id_ctrl[df_id_ctrl['is_principal_isoform']==True][col_tid].nunique()
            assert n_is_principal_prime == n_is_principal, \
                f"Mismatch in the number of principal isoforms: n(filtered)={n_is_principal_prime} <> n0={n_is_principal}"
            
            print(f"[info] After principal isoform search:") 
            print(f"... n(genes) in ctrl: {ng} (prior: {ng0}), n(trpts)={nt} (<? {nt0} previously)")

            df_id_ctrl = df_id_ctrl.drop(columns=['is_principal_isoform'])

        ### [condition] df_id_ctrl adjusted
        assert set(df_id_ctrl.columns) == set(columns_ctrl), \
            f"Columns in df_id_ctrl have changed:\n{df_id_ctrl.columns}\n"

        # --- Test ---
        print("[test] Prior to sample-matching, the similarity of gene sets in the treatment and control groups:")
        jaccard_index, res = gene_level_jaccard_similarity(df_id_treat, df_id_ctrl, col_gid='gene_id', verbose=1)
            
        n_genes_matched = 0
        if match_samples: 
            print_emphasized("[info] Prior to matching samples, the number of genes in the treatment and control groups:")
            print(f"... treatment group: n(genes)={df_id_treat[col_gid].nunique()}, n(trpts)={df_id_treat[col_tid].nunique()}")
            print(f"... control group: n(genes)={df_id_ctrl[col_gid].nunique()}, n(trpts)={df_id_ctrl[col_tid].nunique()}")

            # Select valid treatment transcripts (e.g. NMD): "valid" in the sense of valid TPM values with matched samples
            # - TPM values > 1
            # - Todo: flexible threshold
            df_id_treat = \
                retrieve_sample_matched_tx_ids_for_testset(
                    TranscriptIO(biotype=treatment_biotype, suffix=target_suffix),  # 'protein_coding', 'testset' 
                    df_tx_init=df_id_treat,
                    col_biotype=col_btype, 
                        dataset=dataset, 
                        samples=samples,  # Only consider these samples when checking availability of TPM values
                            select_from_tx_ann=True,
                                save=True)  # If True, save the resulting transcript IDs (and their gene IDs)
            print_emphasized(f"[treatment & control] After TPM-valid, sample-matched treatment transcripts are selected ...")
            print(f"... treatment group: n(genes)={df_id_treat[col_gid].nunique()}, n(trpts)={df_id_treat[col_tid].nunique()}")
            
            # Select valid control transcripts (e.g., protein-coding)
            df_id_ctrl = \
                retrieve_sample_matched_protein_coding_tx_ids_for_testset(
                    TranscriptIO(biotype=control_biotype, suffix=target_suffix),  # 'protein_coding', 'testset' 
                    df_tx_init=df_id_ctrl, 
                    col_biotype=col_btype, 
                        dataset=dataset, 
                        samples=samples,  # Only consider these samples when checking availability of TPM values
                            bypass_appris_criteria=True, # True by default because test data usually don't have APPRIS annotation
                            select_from_tx_ann=True,
                                save=True)  # If True, save the resulting transcript IDs (and their gene IDs)
            print_emphasized(f"[treatment & control] After TPM-valid, sample-matched control transcripts are selected ...")
            print(f"... control group: n(genes)={df_id_ctrl[col_gid].nunique()}, n(trpts)={df_id_ctrl[col_tid].nunique()}")
        
        print("[treatment & control] Ensuring that each gene has well-defined treatment group and control group ...")

        print("[test] After sample-matching, the similarity of gene sets in the treatment and control groups:")
        jaccard_index, res = gene_level_jaccard_similarity(df_id_treat, df_id_ctrl, col_gid='gene_id', verbose=1)

        union_genes = res['union']
        n_genes = len(union_genes)

        # Identify common genes in both treatment and control groups
        common_genes = res['intersection']
        n_genes_matched = len(common_genes)
        n_gene_unmatched = n_genes - n_genes_matched
        
        # Filter treatment and control dataframes to include only common genes
        df_id_treat = df_id_treat[df_id_treat[col_gid].isin(common_genes)]
        df_id_ctrl = df_id_ctrl[df_id_ctrl[col_gid].isin(common_genes)]
        df_id_combined = pd.concat([df_id_treat, df_id_ctrl], ignore_index=True)

        # --- Test ---
        print(f"[treatment & control] Final mapping: shape={df_id_combined.shape}")
        print(f"... n(genes): {n_genes} >? n(matched): {n_genes_matched}")
        print(f"... out of n={n_genes} genes, {n_gene_unmatched} did not match.")

        # # Create mappings for each dataframe (mapping from gene_id to a list of tx_ids)
        # treatment_mapping = df_id_treat.groupby(col_gid)[col_tid].unique().reset_index()
        # control_mapping = df_id_ctrl.groupby(col_gid)[col_tid].unique().reset_index()

        # # --- Test ---
        # print("[info] Example genes and transcripts in the treatment and the control group")
        # print(treatment_mapping[[col_gid, col_tid]].head())
        # print(control_mapping[[col_gid, col_tid]].head())

        # # Merge the mappings on 'gene_id'
        # merged_mappings = pd.merge(treatment_mapping, control_mapping, on=col_gid, how='inner', suffixes=('_pos', '_neg'))

        # # Filter out rows where either list is empty
        # final_mapping = merged_mappings[(merged_mappings['tx_id_pos'].str.len() > 0) & 
        #                                 (merged_mappings['tx_id_neg'].str.len() > 0)]

    else:
        print("[data_pipeline] Skipping the determine-treatment-and-control step ...")
        jaccard_index, res = gene_level_jaccard_similarity(df_id_treat, df_id_ctrl, col_gid='gene_id', verbose=1)

        n_genes = len(res['union'])
        common_genes = res['intersection']
        n_genes_matched = len(common_genes)
        n_gene_unmatched = n_genes - n_genes_matched

        # df_appris = retrieve_principle_isoforms(genes=target_genes, suffix=target_suffix) 
        # tx_appris = df_appris[col_tid].unqiue()
        # df_id_treat = df_tx_target[df_tx_target[col_btype].isin(pos_biotypes)][[col_gid, col_tid]]
        # df_id_ctrl = df_tx_target[df_tx_target[col_btype].isin(neg_biotypes)][[col_gid, col_tid]]
        print(f"... treatment group: n(genes)={df_id_treat[col_gid].nunique()}, n(trpts)={df_id_treat[col_tid].nunique()}")
        print(f"... control group: n(genes)={df_id_ctrl[col_gid].nunique()}, n(trpts)={df_id_ctrl[col_tid].nunique()}")
        df_id_combined = pd.concat([df_id_treat, df_id_ctrl], ignore_index=True)

    # Also save biotype-specific datasets
    print("[I/O] Saving biotype-specific target IDs ...")
    txio_treat = TranscriptIO(biotype=treatment_biotype, suffix=target_suffix) # nmd, testset
    output_files['df_txid_treatment_matched'] = \
        txio_treat.save_tx_id_target(df_id_treat)
    txio_ctrl = TranscriptIO(biotype=control_biotype, suffix=target_suffix) # protein_coding, testset
    output_files['df_txid_control_matched'] = \
        txio_ctrl.save_tx_id_target(df_id_ctrl)

    print("[I/O] Saving the combined biotype-specific tx_ids with well-defined treatment and control ...")
    output_files['df_txid_combined'] = \
        txio_testset.save_tx_id_target(df_id_combined) # Default ID: 'nmd_eff', 'testset'

    # --- Test ---
    print("[info] At the end of the determine-treatment-and-control stage ...")
    print(f"... n(genes) with both treatment and control transcripts: {len(common_genes)}")
    print(f"... n(trpts): {df_id_combined[col_tid].nunique()}")

    # Create an extra column in the transcript dataframe that marks the transcripts that are associated with ...
    # ... genes with well-defined treatment and control groups
    print("[action] Saving updated transcript dataframe with a match indicator ...")

    df_tx['is_matched'] = df_tx[col_tid].isin(df_id_combined[col_tid]).astype(int)
    assert df_tx.shape[0] == nrows0
    
    df_tx['is_principal_isoform'] = 0
    # Create an extra colum in the transcript dataframe to marks principle isoforms
    if df_appris is not None: 
        df_tx['is_principal_isoform'] = df_tx[col_tid].isin(df_appris[col_tid]).astype(int)
    assert df_tx.shape[0] == nrows0

    output_files['df_txid_matched'] = \
        txio_testset.save_tx(df_tx, verbose=1)
    data_survey(df_tx, "Transcript dataframe after the treatment-control stage")

    # --- Test --- 
    n_genes_matched_prime = df_tx[df_tx['is_matched']==1][col_gid].nunique()
    n_trpts_matched_prime = df_tx[df_tx['is_matched']==1][col_tid].nunique()
    print(f"... n(genes) matched: {n_genes_matched_prime} =?= {n_genes_matched} (/{df_tx[col_gid].nunique()})")
    print(f"... n(trpts) matched: {n_trpts_matched_prime} (/{df_tx[col_tid].nunique()})")

    # === Output ===
    # df_tx with additional column, is_matched, as treatment-control match indicator 
    # df_txid with target gene and transcript IDs
    #################################

    # tx_io = TranscriptIO(biotype=target_biotype, suffix=target_suffix) # nmd_eff, testset
    
    # Only include genes with well-defined treatment and control groups in the TPM matrix (because constructing ...
    # ... TPM matrix is a relatively expensive operation)
        
    df_txid = txio_testset.load_tx_id_target()
    if df_txid is None or df_txid.empty: 
        print("[data_pipeline] Could not find treatment-control matched transcripts, set prepare_tpm_matrix to False ...")
        prepare_tpm_matrix = False
    
    # Sleep for a while
    time.sleep(10)

    if prepare_tpm_matrix: 
        highlight("[data_pipeline] Preparing TPM matices (for only those with well-defined treatment and control) ...")

        # sio = MetaIO(dataset=dataset, suffix=suffix)
        # sample_dict = sio.map_sample_id_to_name()
        print(f"... Given n={len(sample_id_to_name)} unique samples")

        meta_columns = ['tx_id', 'gene_id']  # other meta data: label, biotype ('tx_type_ref' or 'tx_type_pred' in Synapse)

        # Action parameters
        collect_tpm_data= kargs.get("collect_tpm_data", True)
        construct_matrix = kargs.get("construct_tpm_matrix", True)
        
        filter_samples = kargs.get("filter_samples", False)
        target_samples = kargs.get("target_samples", []) 
        
        map_internal_to_src_id = kargs.get("map_internal_to_src_id", False)
        add_label = kargs.get("add_label", True)
        add_biotype = kargs.get("add_biotype", True)

        # Sparse TPM? 
        # sparsified_tpm = kargs.get("sparsify_tpm", False)
        time.sleep(5)

        for biotype in [treatment_biotype, control_biotype, ]: 
            print(f"[tpm_matrix] Processing biotype={biotype}, suffix={target_suffix}")
            
            tx_io = TranscriptIO(biotype=biotype, suffix=target_suffix)

            df_tx = tx_io.load_tx_id_target()  # Select only those transcripts given by the determine_treatment_and_control step
            assert df_tx is not None and not df_tx.empty

            tx_set = df_tx['tx_id'].unique()
            assert len(tx_set) > 0
            print(f"[info] Found n={len(tx_set)} unique transcripts for biotype={biotype}")

            if collect_tpm_data: 
                sql_template = query_templates['select_from_tx_ann_given_tx_set']
                # NOTE: In this template, only select rows where tpm value is not null and > 1 (Todo: customizable threshold)
                template_params = {"txdb_tx_ann": Txdb.tx_ann, }
                query_result_path = tx_io.tx_ann_path  
                query_cache_dir = tx_io.cache_dir
                run_batched_query(tx_set,  # break this large transcript set into batches
                    sql_template, template_params, 
                        tx_table_ref='a', in_predicate_first=False,
                        batch_size=25000, 
                            output_path=query_result_path,
                            temp_dir=query_cache_dir, use_dask=False)
                # NOTE: If you use /tmp as the temp directory, it may run out of space before the datasets are consolidated

                # Mapping from internal ID to source ID (if possible)
                print(f"[tpm_matrix] Mapping transcript source IDs (biotype={biotype}) ...")
                map_tx_source_id(biotype=biotype, suffix=suffix, id_method='load_tx_id_target')

                # Assigning each transcript (ID) to its associated gene
                print(f"[tpm_matrix] Mapping gene source IDs (biotype={biotype}) ...")
                map_gene_source_id(biotype=biotype, suffix=suffix, id_method='load_tx_id_target') 

            # df_tx_ann = ud.load_data(query_result_path, delimiter='\t', use_dask=True)
            
            if construct_matrix:  
                use_dask = kargs.get("use_dask", True)

                # Test: Check columns
                df_tx_ann = tx_io.load_tx_ann(use_dask=use_dask, verbose=1, 
                                    dtype={'tx_alias_base': 'object'}) 

                # Keep only those genes and transcripts with proper treatment-control groups
                print(f"[tpm_matrix] Biotype={tx_io.biotype}: filtering tx according to proper treatment-control groups")
                
                df_tc = tx_io.load_tx_id_target() 
                print(f"... number of unique genes in df_tc: {len(df_tc['gene_id'].unique())}")
                print(f"... number of unique transcripts in df_tc: {len(df_tc['tx_id'].unique())}")
                
                df_tx_ann = df_tx_ann[df_tx_ann['tx_id'].isin(df_tc['tx_id'])]
                # NOTE: We could move this step further upstream to the collect_tpm_data step

                if filter_samples: 
                    # Todo: Selecting samples by tissue_type
                    df_tx_ann = filter_samples_in_tpm_matrix(df_tx_ann, samples=target_samples)

                if test: 
                    tx_set_ann = df_tx_ann['tx_id'].unique()
                    n_null_tpm_values = df_tx_ann['tpm'].isna().sum()  # not df_tx_ann['tpm'].isna().any()
                    assert n_null_tpm_values.compute() == 0
                    
                    # if set(tx_set_ann.compute()) != set(tx_set): 
                    #     print(f"[info] Found n(tx_ann)={len(tx_set_ann)} !=? size(tx_set): {len(tx_set)}")
                    #     # NOTE: 
                    #     #    NMD 
                    #     #       Found n(tx_ann)=17669 !=? size(tx_set): 22664
                    #     #    protein coding 
                    #     #       Found n(tx_ann)=76937 !=? size(tx_set): 81932
                    # print(f"(data_pipeline) df_tx_ann columns:\n{list(df_tx_ann.columns)}\n")
                    # NOTE: columns include 
                    #       'tx_id', 'tx_alias_base', 'tx_alias_version', 'sample_id', 'tpm' 
                    print(f"[test] With biotype={biotype}, after filtering by control-set criteria ...")
                    print(f"... n(trpts) in df_tx_ann: {len(tx_set_ann)}")
                    print(f"... approx shape of filtered df_tx_ann: {ud.get_approx_shape(df_tx_ann)}")

                print(f"[tpm_matrix] Formulating TPM matrix ...")

                gem = GEMatrix(dataset=dataset, biotype=biotype, suffix=suffix) # source='synapse'
                print(f"... path to tpm matrix:\n{gem.tpm_matrix_path}\n")

                tx_to_gene = df_tx.set_index('tx_id')['gene_id'].to_dict()
                npartitions = kargs.get("npartitions", 100 if biotype in neg_biotypes else 10)

                if sparsified_tpm: 
                    # dense_file_path, sparse_file_path = gem.sparse_tpm_matrix_paths
                    tpm_matrix = construct_sparse_tpm_matrix(df_tx_ann, 
                                    tx_to_gene=tx_to_gene, aggfunc='mean', 
                                        output_path=gem.sparse_tpm_matrix_paths, npartitions=npartitions)
                    print(f"... TPM sparse matrix:\n{tpm_matrix.head()}\n")
                else:
                    tpm_matrix = construct_tpm_matrix(df_tx_ann, 
                                    tx_to_gene=tx_to_gene, aggfunc='mean', sparsify=False, 
                                        use_dask=use_dask,
                                        output_path=gem.tpm_matrix_path, npartitions=npartitions)
                    print(f"... TPM dense matrix:\n{tpm_matrix.head()}\n")
                
                if test: 
                    if sparsified_tpm: 
                        sparsity_data = ut.calculate_sparsity_from_annotation_dataframe(df_tx_ann, col_sample='sample_id')
                        sparsity_matrix = ut.calculate_sparsity(tpm_matrix, dense_columns=meta_columns)

                        # Test: Sparsity
                        print("... Sparsity in original data:", sparsity_data)
                        print("... Sparsity in TPM matrix:", sparsity_matrix)

                    # Validate aggregation
                    # is_aggregation_correct = ut.validate_aggregation(df_tx_ann, tpm_matrix)
                    # print("... Is aggregation correct:", is_aggregation_correct)

                if map_internal_to_src_id: 
                    print(f"[tpm_matrix] Mapping internal to source IDs in the TPM matrix (biotype={biotype})")
                    tx_src_id = tx_io.load_tx_src_id() # Convert tx_id from internal ID to source ID (e.g. gencode, refseq)
                    gene_src_id = tx_io.load_gene_src_id() # Convert gene_id from internal ID to source ID

                    tx_src_id_dict = tx_src_id.set_index('tx_id')['tx_alias_base'].to_dict()
                    def map_tx(tx_id):
                        return tx_src_id_dict.get(tx_id, tx_id)
                    tpm_matrix['tx_id'] = tpm_matrix['tx_id'].map(map_tx)

                    gene_src_id_dict = gene_src_id.set_index('gene_id')['gene_alias_base'].to_dict()
                    def map_gene(gene_id):
                        return gene_src_id_dict.get(gene_id, gene_id)
                    tpm_matrix['gene_id'] = tpm_matrix['gene_id'].map(map_gene)

                    # print(f"... TPM matrix after mapping to source IDs:\n{tpm_matrix[['tx_id', 'gene_id', ]].head()}\n")

                print("[tpm_matrix] Ordering and renaming TPM matrix ...") # group & order rows by gene_id and tx_id ...
                # ... and renaming sample IDs by their sample names
                tpm_matrix = structure_tpm_matrix(tpm_matrix, sample_id_to_name, meta_columns=meta_columns, verbose=1)
                msg = "user provided" if user_provided_samples else f"dataset={dataset}"
                print(f"... after filterning by samples ({msg}), ncols(tpm_matrix): {tpm_matrix.shape[1]}")
                
                if sparsified_tpm:
                    # print(f"[output] Saving structured TPM matrix to (biotype={biotype}):\n{gem.tpm_matrix_path}\n")
                    output_paths = gem.save_sparse_tpm_matrix(tpm_matrix, dense_columns=meta_columns)
                    # ud.save_sparse_df_to_parquet(tpm_matrix, gem.tpm_matrix_path)

                    dense_file_path, sparse_file_path = output_paths
                    print(f"[output] Saving dense part of TPM matrix to:\n{dense_file_path}\n")
                    print(f"... Saving sparse part of TPM matrix to:\n{sparse_file_path}\n")
                else: 
                    # print(f"[output] Saving structured TPM matrix to (biotype={biotype}):\n{gem.tpm_matrix_path}\n")
                    output_path = gem.save_tpm_matrix(tpm_matrix, format="parquet")
                    # ud.save_df_to_parquet(tpm_matrix, gem.tpm_matrix_path) 
                    print(f"[output] Saving TPM matrix with src IDs to:\n{output_path}\n")  

                # Keep track of the output
                output_files['tpm_matrix'] = output_path  
                           
        ### End collecting TPM values foreach biotypes
        
        col_label = GEMatrix.col_label
        # col_btype = GEMatrix.col_biotype

        # Load NMD TPM
        gem = GEMatrix(dataset=dataset, biotype=treatment_biotype, suffix=target_suffix)  # Default ID: 'nmd', 'testset'
        tpm_matrix_treat = gem.load_sparse_tpm_matrix() if sparsified_tpm else gem.load_tpm_matrix()
        print(f"[data] shape(tpm_matrix_treat): {tpm_matrix_treat.shape}")  

        # Test
        txio_treat = TranscriptIO(biotype=treatment_biotype, suffix=target_suffix)      
        df_txid_treat = txio_treat.load_tx_id_target()
        tx_in_treat_tpm = set(tpm_matrix_treat[col_tid])
        tpm_has_target_set = tx_in_treat_tpm == set(df_txid_treat[col_tid])
        print(f"[test] Does the TPM treatment-group target transcripts? {tpm_has_target_set}")
        print(f"... n(genes) in treatment TPM ({treatment_biotype}): {tpm_matrix_treat[col_gid].nunique()}")
        print(f"... n(trpts) in treatment TPM ({treatment_biotype}): {len(tx_in_treat_tpm)}")

        # -----------------------------------------------------------------------

        # Load contorl TPM
        gem = GEMatrix(dataset=dataset, biotype=control_biotype, suffix=target_suffix) 
        tpm_matrix_ctrl = gem.load_sparse_tpm_matrix() if sparsified_tpm else gem.load_tpm_matrix()
        print(f"[data] shape(tpm_matrix_ctrl): {tpm_matrix_ctrl.shape}")

        # Test 
        txio_ctrl = TranscriptIO(biotype=control_biotype, suffix=target_suffix)      
        df_txid_ctrl = txio_ctrl.load_tx_id_target()
        tx_in_ctrl_tpm = set(tpm_matrix_ctrl[col_tid])
        tpm_has_target_set = tx_in_ctrl_tpm == set(df_txid_ctrl[col_tid])
        print(f"[test] Does the TPM has the control-group target transcripts? {tpm_has_target_set}")
        print(f"... n(genes) in control TPM ({control_biotype}): {tpm_matrix_ctrl[col_gid].nunique()}")
        print(f"... n(trpts) in control TPM ({control_biotype}): {len(tx_in_ctrl_tpm)}")
        
        # Configure the baseline label: NMD-fated transcripts are labeled as 1 
        # while control set (e.g. protein-coding transcripts) are labeled as 0
        if add_label: 
            print("... adding baseline labels: NMD-fataed as positive (1), control as negative (0)")
            concept = Concept(concept='is_nmd')
            tpm_matrix_treat[col_label] = concept.assign_label(treatment_biotype)
            tpm_matrix_ctrl[col_label] = concept.assign_label(control_biotype)
            meta_columns.append(col_label)

        if add_biotype: 
            print(f"... adding biotype: {treatment_biotype} representing NMD, {control_biotype} representing control")
            # Add biotype as a feature (used to faciliate NMD efficiency calculation)
            tpm_matrix_treat[col_btype] = treatment_biotype 
            tpm_matrix_ctrl[col_btype] = control_biotype  
            meta_columns.append(col_btype)

        sample_columns = set(tpm_matrix_treat).union(tpm_matrix_ctrl) - set(meta_columns)
        print(f"[info] Found n={len(sample_columns)} combined sample columns")
        
        # Combine both TPM matrices 
        gem_combined = GEMatrix(dataset=dataset, biotype=target_biotype, suffix=target_suffix) 
        tpm_matrix = mm.combine_tpm_matrices(tpm_matrix_treat, tpm_matrix_ctrl, dense_columns=meta_columns)
        print(f"[output] Shape of the combined TPM matrix: {tpm_matrix.shape}") 

        if sparsified_tpm:
            # print(f"[output] Saving combined sparse TPM matrix:\n{gem_combined.tpm_matrix_path}\n")
            output_paths = gem_combined.save_sparse_tpm_matrix(tpm_matrix, dense_columns=meta_columns)
            # ud.save_sparse_df_to_parquet(tpm_matrix, gem.tpm_matrix_path)

            dense_file_path, sparse_file_path = output_paths
            print(f"[output] Saving dense part of TPM matrix to:\n{dense_file_path}\n")
            print(f"... Saving sparse part of TPM matrix to:\n{sparse_file_path}\n")
        else: 
            # print(f"[output] Saving TPM matrix with src IDs to:\n{gem_combined.tpm_matrix_path}\n")
            output_path = gem_combined.save_tpm_matrix(tpm_matrix, format="parquet")
            # ud.save_df_to_parquet(tpm_matrix, gem.tpm_matrix_path)  
            print(f"[output] Saving TPM matrix with src IDs to:\n{output_path}\n") 

        output_files['combined_tpm_matrix'] = output_path 

    # Sleep for a few seconds
    time.sleep(10)

    if retrieve_tx_exon_data:  
        highlight("[data pipeline] Retrieving transcript-exon data ...")
        # NOTE: Creating marker sequence depends on transcript-exon data as well

        for biotype in [treatment_biotype, control_biotype, ]:  # foreach treatment and control (collective) biotypes
            print(f"\n[feature extraction] Processing biotype={biotype} ...")
            
            tx_io = TranscriptIO(biotype=biotype, suffix=target_suffix)

            # df_tx = tx_io.load_tx_target() 
            df_tx = tx_io.load_tx_id_target() # Select only transcripts with well-defined treatment and control

            assert df_tx is not None and not df_tx.empty

            tx_set = df_tx[col_tid].unique()
            assert len(tx_set) > 0
            N0 = len(tx_set)
            print(f"[info] Found n={len(tx_set)} unique transcripts")
            
            sql_template = query_templates['select_from_tx_ex_given_tx_set']
            template_params = {'txdb_transcripts': Txdb.transcripts, 'txdb_exons': Txdb.exons, }
            query_result_path = tx_io.tx_ex_path
            run_batched_query(tx_set,  # break this large transcript set into batches
                sql_template, template_params, 
                    tx_table_ref='t', in_predicate_first=True,
                    batch_size=25000, 
                        output_path=query_result_path)

            # df_tx_ex may have duplicates in terms of tx_id and exon_id combinations
            df_tx_ex = tx_io.load_tx_ex(is_featurized=False) 
            tx_ex_set = df_tx_ex[col_tid].unique()
            print(f"... found n={len(tx_ex_set)} =?= {N0} unique transcripts")
            # NOTE: If len(tx_ex_set) < len(tx_set), then a subset of the transcripts are not consistently documented 
            #       in txdb.exons
            df_tx_ex = remove_duplicate_exons(df_tx_ex)
            output_files['df_tx_ex'] = tx_io.save_tx_ex(df_tx_ex, is_featurized=False)
        ### End retrieve_tx_exon_data

    # Sleep for a few seconds
    time.sleep(10)
            
    if run_sequence_retrieval: 
        # import sphere_pipeline.seq_analyzer as sa
        highlight("[data pipeline] Retrieving transcript sequences ...")

        retrieve_sequences = kargs.get("retrieve_sequences", True)
        download_sequences = kargs.get("download_sequences_from_blob", False)
        generate_markers = kargs.get("generate_markers", True)
        trim_introns = kargs.get("trim_introns", True) # save the trimmed data in a separate file

        if retrieve_sequences:
            to_str = kargs.get("to_str", True) # convert Bio.Seq to regular string 

            highlight("[data pipeline] Retrieving transcript sequences ...")

            for biotype in [treatment_biotype, control_biotype, ]: # foreach treatment and control (collective) biotypes
                tx_io = TranscriptIO(biotype=biotype, suffix=target_suffix)

                temp_dir = tx_io.cache_dir
                retrieve_transcript_sequences_incrementally(tx_io, batch_size=25000, temp_dir=temp_dir, 
                    download_sequences_from_blob=download_sequences, 
                        generate_markers=generate_markers, trim_introns=trim_introns, 
                            load_tx_id_target=True, to_str=to_str)
                
                # retrieve_transcript_sequences(tx_io, download_sequences_from_blob=False, load_tx_id_target=True, to_str=to_str)

            # Combine sequences from treatment and control
            sequences = []
            is_dask_dataframe = False
            for biotype in [treatment_biotype, control_biotype, ]: # pos_biotypes + neg_biotypes:
                tx_io = TranscriptIO(biotype=biotype, suffix=target_suffix)
                df_seq = tx_io.load_tx_seq()
                sequences.append(df_seq)

                if isinstance(df_seq, dd.DataFrame): 
                    is_dask_dataframe = True
            
            print("(data_pipeline) Processing and combining sequences ...")
            # txio_testset = TranscriptIO(biotype=target_biotype, suffix=target_suffix)
            # NOTE: Biotype-combined dataset uses 'target_biotype' as the ID

            if is_dask_dataframe: 
                df_seq = dd.concat(sequences, ignore_index=True)
            else: 
                df_seq = pd.concat(sequences, ignore_index=True)

            output_files['df_seq'] = txio_testset.save_tx_seq(df_seq)

    if run_feature_extraction: 
        highlight("(data_pipeline) Running feature extraction on transcript-exon data ...")

        # Action parameters 
        run_extract_features = kargs.get("run_extract_features", True)
        featurize_sequences = kargs.get("featurize_sequences", True) 
        combine_by_biotype = kargs.get("combine_by_biotype", True)

        if run_extract_features: 
     
            for biotype in [treatment_biotype, control_biotype, ]: # pos_biotypes + neg_biotypes: 
                tx_io = TranscriptIO(biotype=biotype, suffix=target_suffix)
                # df_tx = tx_io.load_tx_id()
                df_tx = tx_io.load_tx_id_target() # target transcripts selected by the determine_treatment_and_control step
                tx_set = df_tx[col_tid].unique() 

                # df_tx_ann = ud.load_data(tx_io.tx_ex_path, delimiter='\t', use_dask=False) # Load annotation data
                df_tx_ex = tx_io.load_tx_ex()  # Load transcript-exon data

                tx_set_ex = df_tx_ex[col_tid].unique()
                N = len(tx_set_ex)
                print(f"(data_pipeline) ID=({biotype}, {target_suffix}), found n={N} =?= ({len(tx_set)}) unique transcripts at tx-exon level")

                extract_features(tx_io, df_tx_ex, consolidate_transcripts=False)

        # Todo: Add gene_id? 

        # Load feature files by biotype and combine them
        # Todo: dealing with big dataset
        df_trpt = dm.combine_featurized_transcripts_by_biotypes([treatment_biotype, control_biotype, ], 
                        suffix=target_suffix, 
                        add_gene_id=True, add_biotype=True
                        # is_featurized=True  # Always True
                        )
 
        # Biotype-combined dataset (use target_biotype as the ID)
        # txio_testset = TranscriptIO(biotype=target_biotype, suffix=target_suffix)
        output_files['df_tx_ex_featurized'] = \
            txio_testset.save_tx_ex(df_trpt, is_featurized=True, sep='\t')
        # df_tx_ex_featurized.to_csv(output_path, index=False, sep='\t')  # unprocessed

        # Test 
        df_featurized = txio_testset.load_tx_ex(is_featurized=True) 
        assert df_featurized.shape == df_trpt.shape  
        print(f"... shape(df_featurized): {df_featurized.shape}")  
        print(f"... example tx_ids:\n{list(df_featurized.sample(n=5)[col_tid])}\n")  
        print(f"[info] List of transcript features")
        for i, col in enumerate(df_featurized.columns): 
            print(f"... [{i+1}] {col}") 

        if featurize_sequences: 
            descriptor = SequenceDescriptor(biotype=target_biotype, suffix=target_suffix) # concept=labeling_concept
            # NOTE: Biotype-combined dataset uses 'target_biotype' as the ID

            df_dtor = descriptor.load_transcript_features()
            if df_dtor is None or df_dtor.empty: 
                # Load sequence-featurized dataset or if doesn't already exist, then run feature extraction on the transcript's sequence
                df_dtor = dm.featurize_transcript_sequences(descriptor, verbose=verbose)
        
                # Test
                df_dtor_prime = descriptor.load_transcript_features()
                assert df_dtor.shape == df_dtor_prime.shape 
        
    if run_pos_hoc_ops:  
        
        combine_gene_src_ids = True

        if combine_gene_src_ids: 

            if not prepare_tpm_matrix: 
                for i, biotype in enumerate([treatment_biotype, control_biotype, ]):
                    # Mapping from internal ID to source ID (if possible)
                    print(f"[info] Mapping transcript source IDs (biotype={biotype}) ...")
                    map_tx_source_id(biotype=biotype, suffix=target_suffix)

                    # Assigning each transcript (ID) to its associated gene
                    print(f"[info] Mapping gene source IDs (biotype={biotype}) ...")
                    map_gene_source_id(biotype=biotype, suffix=target_suffix) 

            highlight("[data_pipeline] Combining gene-specific source IDs ...")
            df_idx = []
            for biotype in [treatment_biotype, control_biotype, ]: #  pos_btypes + neg_btypes: 
                tx_io = TranscriptIO(biotype=biotype, suffix=target_suffix)
                gene_src_id = tx_io.load_gene_src_id()
                print(f"[test] gene_src_id:\n{gene_src_id.head()}\n")
                if gene_src_id is not None and not gene_src_id.empty: 
                    df_idx.append(gene_src_id) # Convert gene_id from internal ID to source ID
                else: 
                    msg = f"[pipeline] Could not load gene data from:\n{tx_io.gene_src_id_path}\n"
                    msg += "> Hint: Run process_trpts_from_synapse.map_gene_source_id()"
                    raise FileNotFoundError(msg)
                    # print(msg)
            
            df_idx = pd.concat(df_idx, ignore_index=True)
            print(f"[info] Gene ID and name dataframe: shape(df_idx): {df_idx.shape}")
            # gene_dict = df_idx.set_index(col_gid)['gene_name'].to_dict()
            
            # Combine treatment and control 
            # txio_testset = TranscriptIO(biotype=target_biotype, suffix=target_suffix)
            # NOTE: Biotype-combined dataset uses 'target_biotype' as the ID

            output_files['gene_src_id'] = txio_testset.save_gene_src_id(df_idx)

    print("#" * 80); print()    

    return output_files


def process_testset_for_nmd_eff_prediction(concept, **kargs):

    target_biotype = concept.biotype
    target_suffix = concept.suffix
    txio_testset = TranscriptIO(biotype=target_biotype, suffix=target_suffix) # nmd_eff, testset
    
    highlight("(process_testset_for_nmd_eff_prediction) Loading the initial test set in transcript dataframe ...")
    df_tx = txio_testset.load_tx()
    assert 'is_matched' in df_tx.columns

    test_data_pipeline(concept,
            determine_treatment_and_control=True, # Enable the treatment-control selection step
            prepare_tpm_matrix=True,  # Enable the TPM matrix construction step 
            retrieve_tx_exon_data=False, 
            run_sequence_retrieval=False, 
            run_feature_extraction=False, 
            run_pos_hoc_ops=False)

    return

def predict_biotype_only_action(args):
    args.retrieve_transcripts = False
    args.featurize_transcripts = False
    args.do_gene_expr_analysis = False

def parse_arguments():
    description_txt = "Retrieve transcriptomic test datasets from SequenceSphere"
    parser = argparse.ArgumentParser(description=description_txt)
    
    parser.add_argument('--concept', dest='concept', default='exon_prediction', 
                        help="The concept (name) associated with the transcriptomic data") 
    parser.add_argument('--biotype', default='uorf', 
                        help='Transcript data ID that captures biotype-specific properties')
    parser.add_argument('--suffix', default='testset', 
                        help='Transcript supplementary ID (e.g. trainset, testset, benchmark)')
    parser.add_argument('--dataset', type=str, default='normal-gtex', help='Dataset name for constructing TPM matrix')

    parser.add_argument('--gene-path', '--path-to-genes', dest='genes_path', default=None,  # required=True, 
                        help='Path to the file containing the set of genes')
    parser.add_argument('--gene-string', default='', 
                        help='String specifying a list of genes')
    parser.add_argument('--tx-path', dest='tx_path', default=None, 
                        help='Path to the transcript dataframe')

    # For ALL data
    parser.add_argument('-r', '--retrieve', dest='retrieve_all', action='store_true', default=False, 
                        help='Retrieve all transcriptomic data from SeqSphere including exons and transcript sequences')
    parser.add_argument('--no-retrieve', dest='retrieve_all', action='store_false',
                        help='Bypass all transcriptomic data retrieval from SeqSphere')

    # For transcript data
    parser.add_argument('--retrieve-tx', '--retrieve-transcripts', 
                        dest='retrieve_transcripts', action='store_true', default=True,
                        help='Retrieve transcript data')

    # For exon data
    parser.add_argument('--retrieve-exons', dest='retrieve_exons', action='store_true', default=False,
                        help='Retrieve exon data')

    # For sequence data
    parser.add_argument('--retrieve-seq', '--retrieve-sequences', 
                        dest='retrieve_sequences', action='store_true', default=False,
                        help='Retrieve sequence data')
                        
    parser.add_argument('-f', "--featurize", dest='featurize_transcripts', action='store_true', default=False, 
                        help="Run feature extraction on transcriptomic data")
    parser.add_argument('--no-featurize', dest='featurize_transcripts', action='store_false',
                        help="Bypass feature extraction")
    
    parser.add_argument('--analyze', dest='do_gene_expr_analysis', action='store_true', default=False,
                        help="Perform gene analysis and construct TPM matrices")
    parser.add_argument('--no-analyze', dest='do_gene_expr_analysis', action='store_false',
                        help="Bypass gene analysis and TPM matrix construction")
    # NOTE: If neither --analyze nor --no-analyze is specified, do_gene_expr_analysis will be False by default

    parser.add_argument('--predict-biotype-only', action='store_true', 
                    help='Only predict biotype, bypassing transcript retrieval, feature extraction, and gene analysis')
    # NOTE: The default value for the --predict-biotype-only option, if not specified, is False. 
    #       This is because the action='store_true' option in argparse sets the default value to False 
    #       when the argument is not provided on the command line.
    
    
    args = parser.parse_args()

    # Transcriptomic data retrieval 
    # If retrieve_all is True, set all other flags to True unless they've been explicitly set to False
    if args.retrieve_all:
        if args.retrieve_transcripts is None:
            args.retrieve_transcripts = True
        if args.retrieve_exons is None:
            args.retrieve_exons = True
        if args.retrieve_sequences is None:
            args.retrieve_sequences = True
    # If retrieve_all is False, set all other flags to False
    else:
        args.retrieve_transcripts = False
        args.retrieve_exons = False
        args.retrieve_sequences = False

    # Biotype classification
    if args.predict_biotype_only:
        predict_biotype_only_action(args)

    return args

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def read_genes(filepath_or_string):
    # import re 

    # Initialize an empty list to hold the genes
    genes = []

    # Check if filepath_or_string is a file path
    if os.path.isfile(filepath_or_string):
        # If it's a file path, open the file and read the genes into the list
        with open(filepath_or_string, 'r') as f:
            file_content = f.read()
    else:
        # If it's not a file path, assume it's a string of genes
        file_content = filepath_or_string

    # Split the string on commas, newlines, or spaces to get the list of genes
    genes = re.split(',|\n| ', file_content)

    # Remove leading/trailing whitespace from each gene and remove empty strings
    genes = [gene.strip() for gene in genes if gene.strip()]

    return genes

# Some input handling utilities ---------------------------------------------------

def detect_separator(filename):
    # import csv
    with open(filename, 'r') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
    return dialect.delimiter

def read_file_into_dataframe(filepath):
    if os.path.isdir(filepath):
        df = pd.read_parquet(filepath) # assume it's a directory containing parquet files
    else:
        _, file_extension = os.path.splitext(filepath)
        if file_extension in ['.csv', '.tsv']:
            sep = ',' if file_extension == '.csv' else '\t'
            df = pd.read_csv(filepath, sep=sep)
        elif file_extension == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    return df

####################################################################################################    

def main(concept, biotype, suffix, dataset='normal-gtex', 
         retrieve_transcripts=True, retrieve_exons=False, retrieve_sequences=False, 
         featurize_transcripts=False, analyze_transcripts=False, run_biotype_classifier=False, 
         train_biotype_model=False, genes_path=None, gene_string=None, tx_path=None, df_tx=None): 
    import time

    args = parse_arguments()

    target_biotype = biotype 
    target_suffix = suffix
    new_source = {concept: {'biotype': target_biotype, 'suffix': target_suffix}}
    gene_expr_dataset = dataset # 'normal-gtex' by default

    print_emphasized("(data_generation_workflow) Command-line options given by:")
    print(f"... concept: {concept}")
    print(f"... transcript data ID:")
    print(f"...... target biotype: {target_biotype}")
    print(f"...... target suffix: {target_suffix}")
    print(f"... gene expression dataset keyword: {gene_expr_dataset}")

    # Optional parameters 
    # retrieve_transcripts = args.retrieve_transcripts  # True by default
    # retrieve_exons = args.retrieve_exons # disabled via --no-retrieve-exons
    # retrieve_sequences = args.retrieve_sequences # disbabled via --no-retrieve-seq

    # featurize_transcripts = args.featurize_transcripts
    
    # run_biotype_classifier = True # True by default
    # train_biotype_model = False
    # NOTE: When set to True and if the biotype model is not trained, then the model will be trained during the biotype classification step
    #       Alternatively, run the following command to train the biotype model separately
    #       python biotype_classifier.py --concept=biotype_3way --biotype=3way --suffix=trainset

    # analyze_transcripts = args.do_gene_expr_analysis
    print(f"... retrieve transcripts? {retrieve_transcripts}")
    print(f"... retrieve exons? {retrieve_exons}")
    print(f"... retrieve sequences? {retrieve_sequences}")

    print(f"... featurize transcripts? {featurize_transcripts}")
    print(f"... analyze transcripts (treatment vs control, TPM)? {analyze_transcripts}")
    print(f"... run biotype classifier? {run_biotype_classifier}")
    print(f"...... train biotype model? {train_biotype_model}")
    print('-' * 85)

    txio_testset = TranscriptIO(biotype=target_biotype, suffix=target_suffix) 
    # genes_path = args.genes_path
    # gene_string = args.gene_string
    # tx_path = args.tx_path

    # --- Retrieve genes from the file (if provided) ---

    # Initialize an empty list to hold the genes
    genes = []
    input_mode = "random"

    if genes_path is not None: # If the path to a file containing the genes is provided ...
        # If genes_path is not a file path, prepend txio_testset.data_dir
        if not os.path.dirname(genes_path):
            print(f"... using default data path (ID={txio_testset.ID}):\n{txio_testset.gene_dir}\n")
            txio_testset.create_gene_dir()

            genes_path = os.path.join(txio_testset.gene_dir, genes_path)

        # Check if the file exists
        if os.path.isfile(genes_path):
            genes = read_genes(genes_path)
        else:
            print(f"[main] Invalid file path: {genes_path}")
    elif gene_string: # If a string of genes is provided ...
        print("[main] Given the input genes in the form of a string ...")
        genes = read_genes(gene_string)
    else: 
        print("[main] No input genes provided. Will look for the transcript-based input file ...")

    # ---- Retrieve transcript data from the file (if provided) ----

    # If a list of gene names is provided, print them out
    if len(genes) > 0:  
        input_mode = "genes"

        print("[input] Found the following genes:")
        for i, gene in enumerate(genes):
            print(f"... [{i+1}] {gene}")

    else: 
        col_tid = TranscriptIO.col_tid
        col_gid = TranscriptIO.col_gid

        if tx_path is not None:
            # If tx_path is not a file path, prepend txio_testset.data_dir
            if not os.path.dirname(tx_path):
                print(f"... using default data path (ID={txio_testset.ID}):\n{txio_testset.tx_dir}\n")
                txio_testset.create_tx_dir()

                tx_path = os.path.join(txio_testset.tx_dir, tx_path)

            # Check if the file exists
            if os.path.exists(tx_path):
                # sep = detect_separator(tx_path)
                print(f"[test] Reading from file:\n{tx_path}\n")

                # Read the file into a DataFrame
                df = read_file_into_dataframe(tx_path)
                print(f"[main] Found the following columns in the transcript file:\n{list(df.columns)}")

                # Keep only the 'tx_id' and 'gene_id' columns, if they exist
                if col_tid in df.columns and col_gid in df.columns:
                    df = df[[col_tid, col_gid]]

                    num_samples = min(10, len(df[col_gid]))
                    sample_values = df[col_gid].sample(num_samples)
                    print(f"[test] Example genes:\n{sample_values}\n")
                elif col_tid in df.columns:
                    df = df[[col_tid]]
                else:
                    raise ValueError("The file must contain a 'tx_id' column")   

                df_tx = df 
                n_genes = df_tx[col_gid].nunique() if col_gid in df_tx.columns else 0
                n_trpts = df_tx[col_tid].nunique()
                print(f"... Given n={n_genes} genes and {n_trpts} transcripts in the file")

        if df_tx is not None and not df_tx.empty:
            input_mode = "transcripts"

    ############################

    start_time = time.time()  # Record the start time

    concept_obj = \
        Concept(concept=concept, 
            biotype=target_biotype, suffix=target_suffix, 
                dataset=gene_expr_dataset)
    
    retrieve_tx_data = retrieve_transcripts or retrieve_exons or retrieve_sequences
    if retrieve_tx_data or featurize_transcripts :

        if len(genes) > 0:
            # Retrieve and featurize testset from user-previded genes
            print(f"[main] Processing n={len(genes)} user-specified genes ...")
            retrieve_and_featurize_testset(
                concept_obj, 
                    genes=genes, max_tx_per_gene=1000,
                        # retrieve=retrieve_transcripts, 
                        retrieve_transcripts = retrieve_transcripts, 
                        retrieve_exons = retrieve_exons, 
                        retrieve_sequences = retrieve_sequences,
                            featurize=featurize_transcripts, 
                                test=featurize_transcripts)
        elif df_tx is not None and not df_tx.empty:
            # Retrieve and featurize testset from user-previded transcript data
            print("[main] Processing user-specified transcript data ...")
            retrieve_and_featurize_testset(
                concept_obj, 
                    transcript_df=df_tx,
                        # retrieve=retrieve_transcripts, 
                        retrieve_transcripts = retrieve_transcripts, 
                        retrieve_exons = retrieve_exons, 
                        retrieve_sequences = retrieve_sequences,
                            featurize=featurize_transcripts, 
                                test=featurize_transcripts)
        else:
            # Retrieve and featurize testset from random genes
            print("[main] Processing random genes ...")
            retrieve_and_featurize_testset(
                concept_obj, 
                    n_genes=100, max_tx_per_gene=200, 
                        # retrieve=retrieve_transcripts, 
                        retrieve_transcripts = retrieve_transcripts, 
                        retrieve_exons = retrieve_exons, 
                        retrieve_sequences = retrieve_sequences,
                            featurize=featurize_transcripts, 
                                test=featurize_transcripts)
    
    if run_biotype_classifier: 
        labeing_concept = concept
        
        concept_biotype_classifier = \
            Concept(concept=BiotypeModel.labeling_concept,  
                    biotype=target_biotype, suffix=target_suffix) # Same transcript data IDs as the NMD-efficiency concept
        print("[main] Predicting biotypes ...")
        # if has_unknown_biotypes(biotype=target_biotype, suffix=target_suffix):
        #     predict_biotypes(concept_biotype_classifier, train_model=train_biotype_model)
        predict_biotypes(concept_biotype_classifier, train_model=train_biotype_model)

        # Fill in other biotype-specific information to relevant dataframes ...
        # ... Re-visit featurized transcript-exon data
        df_featurized = txio_testset.load_tx_ex(is_featurized=True) 
        shape0 = df_featurized.shape
        df_featurized = add_biotype_info(df_featurized, txio_testset, raise_exception=True)
        assert df_featurized.shape[0] == shape0[0], f"Shape mismatch: {df_featurized.shape[0]} != {shape0[0]}"

        filepath = txio_testset.save_tx_ex(df_featurized, is_featurized=True, sep='\t')
        print(f"(add_biotype) Saved updated featurized tx-exon data to:\n{filepath}\n")

    if analyze_transcripts: 

        run_gene_expression_analysis(concept_obj, 
                determine_treatment_and_control=True,
                prioritize_principal_isoforms=True, 
                match_samples= True, # False if input_mode == 'transcripts' else True,
                    prepare_tpm_matrix=True)
        # NOTE: If the input mode is 'transcripts', then the treatment-control selection step is skipped because
        #       we assume that the transcript dataset has been pre-determined for the analysis
    
    ############################
    
    end_time = time.time()    # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    formatted_time = format_time(elapsed_time)  # Format the time for display
    print(f"The test data pipeline took {formatted_time} to complete.")


if __name__ == "__main__": 
    # Parse command-line arguments
    args = parse_arguments()

    # Call main() with the parsed arguments
    main(args.concept, args.biotype, args.suffix, args.dataset, args.retrieve_transcripts, args.retrieve_exons, 
         args.retrieve_sequences, args.featurize_transcripts, args.do_gene_expr_analysis, args.genes_path, 
         args.gene_string, args.tx_path)