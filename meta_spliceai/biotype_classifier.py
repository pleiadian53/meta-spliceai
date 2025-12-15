
import os, sys
from pathlib import Path
import pandas as pd

import argparse
import meta_spliceai.system.sys_config as config
from meta_spliceai.system.model_config import BiotypeModel
from meta_spliceai.utils.utils_sys import highlight

from meta_spliceai.sphere_pipeline.utils_data import count_unique_values
from meta_spliceai.sphere_pipeline.data_model_utils import (
    concept_to_labels,
    create_descriptor_based_training_data,
    create_descriptor_based_training_data_multiclass,
    mini_ml_pipeline
)
from meta_spliceai.sphere_pipeline.data_model import (
    # DataSource,
    # TranscriptIO, 
    SequenceDescriptor,
    # Sequence,
    Concept
)

# Todo
# from mllib.model_trainer import mini_ml_pipeline
from meta_spliceai.mllib.model_tracker import ModelTracker


def train_biotype_3way_classifier(labeling_concept="biotype_3way", model_name='xgboost', **kargs): 
    """

    Memo
    ----
    1. See feature_extractor.py for an example of how to create training data for the 3-way biotype classifier
    """
    pass

def train_biotype_classifier(labeling_concept='is_nmd', model_name='xgboost', **kargs):
    pass

def parse_arguments():
    description_txt = "Transcript biotype classifiers"
    parser = argparse.ArgumentParser(description=description_txt)
    
    parser.add_argument('--concept', dest='labeling_concept', default='nmd_vs_coding', 
                        help="The concept (name) associated with the labels or labeling concept") 
    parser.add_argument('--experiment', dest='experiment_name', default=None,
                        help='The name of the experiment (default: the same as the labeling concept)')
    parser.add_argument('--source-biotype', '--biotype', dest='biotype', default='combined', help='Source transcript biotype')
    parser.add_argument('--source-suffix', '--suffix', dest='suffix', default='test', help='Source transcript data suffix')

    parser.add_argument('-m', '--model', dest='model_name', type=str, default='xgboost', help='Name of the model')
    parser.add_argument('--model-suffix', type=str, default='bohb', help='Model suffix')

    parser.add_argument('--run-nested-cv', action='store_true', default=False, help='Run nested CV')

    args = parser.parse_args()

    return args

def main(): 

    args = parse_arguments()

    labeling_concept = args.labeling_concept
    # NOTE: 'nmd_vs_coding', 'nmd_vs_noncoding', 'coding_vs_noncoding', 'biotype_3way'
    experiment_name = args.experiment_name
    if experiment_name is None: experiment_name = labeling_concept

    source_biotype = args.biotype 
    source_suffix = args.suffix
    model_name = args.model_name
    model_suffix = args.model_suffix
    run_nested_cv = args.run_nested_cv # False by default (expensive)

    concept = {'labeling_concept': labeling_concept, 'experiment': experiment_name}
    model = {'name': model_name, 'suffix': model_suffix}
    tx_source_ids = {'biotype': source_biotype, 'suffix': source_suffix}

    print("[classifier] key parameters:")
    print(f"... what is being classified? {labeling_concept}")
    print(f"... using model = {model_name} with {model_suffix}")
    print(f"... training data coming from ID=({source_biotype}, {source_suffix})")

    if labeling_concept.find('3way') >= 0: 
        train_biotype_3way_classifier(labeling_concept=BiotypeModel.labeling_concept, # 'biotype_3way'
                                        experiment= BiotypeModel.model_output_dir,  # 'biotype_3way'
                                            model_name=model['name'], 
                                            model_suffix=model['suffix'],
                                            n_fold=5, use_nested_cv=run_nested_cv, 
                                                biotype=tx_source_ids['biotype'], 
                                                suffix=tx_source_ids['suffix']
                                        )
    else: 
        train_biotype_classifier(labeling_concept=concept['labeling_concept'], 
                                experiment=concept['experiment'],
                                    model_name=model['name'], 
                                    model_suffix=model['suffix'],
                                    n_fold=5, use_nested_cv=run_nested_cv, 
                                        biotype=tx_source_ids['biotype'], 
                                        suffix=tx_source_ids['suffix']
                                )


    return

if __name__ == "__main__": 
    main()