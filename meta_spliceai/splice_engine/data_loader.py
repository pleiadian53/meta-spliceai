from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator, 
    display, 
    display_dataframe_in_chunks
)   

from meta_spliceai.splice_engine.utils_doc import (
    display_feature_set
)

from meta_spliceai.splice_engine.error_analyzer import (
    ErrorAnalyzer
)

from meta_spliceai.splice_engine.model_evaluator import (
    ModelEvaluationFileHandler
)

from meta_spliceai.splice_engine.model_utils import (
    analyze_data_labels
)

def load_error_classifier_dataset(pred_type='FP', **kargs):
    verbose = kargs.get('verbose', 1)
    col_label = kargs.get('col_label', 'label')
    to_pandas = kargs.get('to_pandas', True)
    subject = kargs.get('subject', None)
    splice_type = kargs.get('splice_type', None)
    return_feature_analysis = kargs.get('return_feature_analysis', False)

    error_label = kargs.get("error_label", pred_type)
    correct_label = kargs.get("correct_label", "TP")

    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')

    # Load pre-computed featurized dataset [1]
    if subject is None: 
        df_trainset = mefd.load_featurized_dataset(
            aggregated=True, 
            error_label=error_label, 
            correct_label=correct_label, 
            splice_type=splice_type
        )
    else: 
        print_emphasized(f"[action] Loading featurized dataset for subject={subject} ...")
        df_trainset = mefd.load_featurized_dataset(
            aggregated=True, 
            subject=subject, 
            error_label=error_label, 
            correct_label=correct_label, 
            splice_type=splice_type
        )
    
    print_emphasized(f"[action] Training error classifier for {pred_type} ...")
    if verbose > 0: 
        print_with_indent(f"Training set: {df_trainset.shape}", indent_level=1)
        print_with_indent(f"Columns: {display_feature_set(df_trainset, max_kmers=100)}", indent_level=1)
        analysis_result = \
            analyze_data_labels(df_trainset, label_col=col_label, verbose=verbose, handle_missing=None)

    if to_pandas:  
        if isinstance(df_trainset, pl.DataFrame):
            df_trainset = df_trainset.to_pandas()
        
        if not isinstance(df_trainset, pd.DataFrame):
            raise ValueError(f"Invalid type: {type(df_trainset)}")
        
    if return_feature_analysis: 
        return df_trainset, analysis_result

    return df_trainset