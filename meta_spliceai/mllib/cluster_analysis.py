import random
import os, sys, time
from pathlib import Path
from functools import partial
from tabulate import tabulate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
import umap
import tensorflow as tf
# import tensorflow_hub as hub

# from sentence_transformers import SentenceTransformer 
# NOTE: A lot of dependencies including nvidia's libraries

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm.notebook import trange
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials

from meta_spliceai.system.sys_config import SequenceIO
from meta_spliceai.utils.utils_sys import highlight


pd.set_option("display.max_rows", 600)
pd.set_option("display.max_columns", 500)
pd.set_option("max_colwidth", 400)

# Modular level configuration
class ClusterAnalysis(object): 

    data_params = {
        "source": 'ensembl', 
        "version": "GRCh38.106", 

        "data_prefix": SequenceIO.get_data_dir(), 
        "experiment": "cluster_analysis", 
        "artifact_dir": 'sequence', 
        # "output_dir": os.getcwd(), 
    }

    def __init__(self, data_params={}, exp_root="experiments"):

        # Data parameters
        if data_params: 
            ClusterAnalysis.data_params.update(data_params)
        self.data_params = ClusterAnalysis.data_params
        self.exp_root = exp_root

    def get_output_directory(self, data_params={}, verbose=1): 
        data_prefix = self.data_params.get('data_prefix', SequenceIO.get_data_dir())
        experiment = self.data_params.get("experiment", "cluster_analysis")
        exp_root = os.path.join(data_prefix, self.exp_root)    # <project_dir>/data/experiments/<specific_experiment_dir>
        exp_dir = os.path.join(exp_root, experiment)
        Path(exp_dir).mkdir(parents=True, exist_ok=True)
        output_dir = data_params.get('output_dir', exp_dir)   # experiment output directory
        if verbose: print(f"[ClusterAnalysis] Output directory:\n{output_dir}\n")  
        return output_dir
    def get_experiment_output_directory(self, data_params={}, verbose=1): 
        return self.get_output_directory(data_params=data_params, verbose=verbose)
    def get_data_output_directory(self, data_params={}, verbose=1): 
        pass

def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components, 
                      min_cluster_size,
                      random_state = None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """
    
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                metric='cosine', 
                                random_state=random_state).fit_transform(message_embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size,
                               metric='euclidean', 
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters

def generate_n_clusters(message_embeddings,
                       n_clusters, 
                       n_neighbors,
                       n_components, 
                       min_cluster_size,
                       random_state = None): 

    from hdbscan import flat
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                metric='cosine', 
                                random_state=random_state).fit_transform(message_embeddings))

    clusterer = flat.HDBSCAN_flat(umap_embeddings, n_clusters, prediction_data=True)
    # flat.approximate_predict_flat(clusterer, points_to_predict, n_clusters)

    return

def score_clusters(clusters, prob_threshold = 0.05):
    """
    Returns the label count and cost of a given cluster supplied from running hdbscan
    """
    
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num)
    
    return label_count, cost

def random_search(embeddings, space, num_evals):
    """
    Randomly search hyperparameter space and limited number of times 
    and return a summary of the results
    """
    
    results = []
    
    for i in trange(num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])
        
        clusters = generate_clusters(embeddings, 
                                     n_neighbors = n_neighbors, 
                                     n_components = n_components, 
                                     min_cluster_size = min_cluster_size, 
                                     random_state = 42)
    
        label_count, cost = score_clusters(clusters, prob_threshold = 0.05)
                
        results.append([i, n_neighbors, n_components, min_cluster_size, 
                        label_count, cost])
    
    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components', 
                                               'min_cluster_size', 'label_count', 'cost'])
    
    return result_df.sort_values(by='cost')


def objective(params, embeddings, label_lower, label_upper):
    """
    Objective function for hyperopt to minimize

    Arguments:
        params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'random_state' and
               their values to use for evaluation
        embeddings: embeddings to use
        label_lower: int, lower end of range of number of expected clusters
        label_upper: int, upper end of range of number of expected clusters

    Returns:
        loss: cost function result incorporating penalties for falling
              outside desired range for number of clusters
        label_count: int, number of unique cluster labels, including noise
        status: string, hypoeropt status

        """
    
    clusters = generate_clusters(embeddings, 
                                 n_neighbors = params['n_neighbors'], 
                                 n_components = params['n_components'], 
                                 min_cluster_size = params['min_cluster_size'],
                                 random_state = params['random_state'])
    
    label_count, cost = score_clusters(clusters, prob_threshold = 0.05)
    
    #15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.15 
    else:
        penalty = 0
    
    loss = cost + penalty
    
    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}

def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayesian search on hyperparameter space using hyperopt

    Arguments:
        embeddings: embeddings to use
        space: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', and 'random_state' and
               values that use built-in hyperopt functions to define
               search spaces for each
        label_lower: int, lower end of range of number of expected clusters
        label_upper: int, upper end of range of number of expected clusters
        max_evals: int, maximum number of parameter combinations to try

    Saves the following to instance variables:
        best_params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'min_samples', and 'random_state' and
               values associated with lowest cost scenario tested
        best_clusters: HDBSCAN object associated with lowest cost scenario
                       tested
        trials: hyperopt trials object for search

        """
    
    trials = Trials()
    fmin_objective = partial(objective, 
                             embeddings=embeddings, 
                             label_lower=label_lower,
                             label_upper=label_upper)
    
    best = fmin(fmin_objective, 
                space = space, 
                algo=tpe.suggest,
                max_evals=max_evals, 
                trials=trials)

    best_params = space_eval(space, best)
    print ('best:')
    print (best_params)
    print (f"label count: {trials.best_trial['result']['label_count']}")
    
    best_clusters = generate_clusters(embeddings, 
                                      n_neighbors = best_params['n_neighbors'], 
                                      n_components = best_params['n_components'], 
                                      min_cluster_size = best_params['min_cluster_size'],
                                      random_state = best_params['random_state'])
    
    return best_params, best_clusters, trials

def combine_results(df_ground, cluster_dict):
    """
    Returns dataframe of all documents and each model's assigned cluster

    Arguments:
        df_ground: dataframe of original documents with associated ground truth
                   labels
        cluster_dict: dict, keys as column name for specific model and value as
                      best clusters HDBSCAN object

    Returns:
        df_combined: dataframe of all documents with labels from
                     best clusters for each model

    """
    df_combined = df_ground.copy()
    
    for key, value in cluster_dict.items():
        df_combined[key] = value.labels_
    
    return df_combined

def summarize_results(results_dict, results_df):
    """
    Returns a table summarizing each model's performance compared to ground
    truth labels and the model's hyperparametes

    Arguments:
        results_dict: dict, key is the model name and value is a list of: 
                      model column name in combine_results output, best_params and best_clusters 
                      for each model (e.g. ['label_use', best_params_use, trials_use])
        results_df: dataframe output of combine_results function; dataframe of all documents 
                    with labels from best clusters for each model

    Returns:
        df_final: dataframe with each row including a model name, calculated ARI and NMI,
                  loss, label count, and hyperparameters of best model

    """
    summary = []

    for key, value in results_dict.items():
        ground_label = results_df['category'].values
        predicted_label = results_df[value[0]].values
        
        ari = np.round(adjusted_rand_score(ground_label, predicted_label), 3)
        nmi = np.round(normalized_mutual_info_score(ground_label, predicted_label), 3)
        loss = value[2].best_trial['result']['loss']
        label_count = value[2].best_trial['result']['label_count']
        n_neighbors = value[1]['n_neighbors']
        n_components = value[1]['n_components']
        min_cluster_size = value[1]['min_cluster_size']
        random_state = value[1]['random_state']
        
        summary.append([key, ari, nmi, loss, label_count, n_neighbors, n_components, 
                        min_cluster_size, random_state])

    df_final = pd.DataFrame(summary, columns=['Model', 'ARI', 'NMI', 'loss', 
                                              'label_count', 'n_neighbors',
                                              'n_components', 'min_cluster_size',
                                              'random_state'])
    
    return df_final.sort_values(by='NMI', ascending=False)

def plot_clusters(embeddings, clusters, n_neighbors=15, min_dist=0.1, **kargs):
    """
    Reduce dimensionality of best clusters and plot in 2D

    Arguments:
        embeddings: embeddings to use
        clusteres: HDBSCAN object of clusters
        n_neighbors: float, UMAP hyperparameter n_neighbors
        min_dist: float, UMAP hyperparameter min_dist for effective
                  minimum distance between embedded points

    """
    from meta_spliceai.utils.utils_misc import savefig

    umap_data = umap.UMAP(n_neighbors=n_neighbors, 
                          n_components=2, 
                          min_dist = min_dist,  
                          #metric='cosine',
                          random_state=42).fit_transform(embeddings)

    point_size = 100.0 / np.sqrt(embeddings.shape[0])
    
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = clusters.labels_

    
    fig, ax = plt.subplots(figsize=(14, 8))
    outliers = result[result.labels == -1]
    clustered = result[result.labels != -1]
    plt.scatter(outliers.x, outliers.y, color = 'lightgrey', s=point_size)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=point_size, cmap='jet')
    plt.colorbar()
    # plt.show()

    # Save plot
    # --------------------------------------------------------
    ext = 'tif'
    default_output_file = f'umap_hdbscan.{ext}'
    default_output_dir = os.path.join(os.getcwd(), 'plot')
    output_path = kargs.get("output_path", os.path.join(default_output_dir, default_output_file))

    print(f"[transcript_embedding] Saving transcript embedding cluster analysis file to:\n{output_path}\n")   
    savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)
    # --------------------------------------------------------

    return plt

def get_label_counts(df, col_label='label', dtype=None, verbose=0): 
    table = tabulate(df[col_label].value_counts().to_frame(), headers='keys', tablefmt='psql')

    if verbose: 
        if dtype is not None: 
            print(f"> In {dtype} set, label counts:\n{table}\n")
        else: 
            print(f"> Label counts:\n{table}\n")
    return table

def encode_labels(df, col_label='label', label_names={}, mode="to_int", verbose=0): 
    import numbers
    # maps label names to integers
    if not label_names: 
        # For convenience of the NMD project, assign a default labeling map 
        label_names = {0: 'nmd_ineff', 1: 'nmd_eff'}  # Todo: configuration

    if verbose: 
        print("> Before label encoding, label counts:") 
        print(get_label_counts(df))

    # NOTE: NO need to use groupby
    # for r, dfg in df.groupby(col_label): 
    #   ...
    labels = df[col_label].unique()

    if mode == "to_int": # convert label names to integers
        if isinstance(labels[0], str): 
            assert set(labels) == set(label_names.values())

            for class_code, class_name in label_names.items(): 
                df.loc[df[col_label]==class_name, col_label] = class_code
        else: 
            # otherwise, the labels must have been encoded, in which case the work is done
            assert isinstance(labels[0], numbers.Number), f"unusual label dtype: {type(labels[0])}"
    else: 
        # print(f"[debug] type(labels[0]): {type(labels[0])}") # numpy.int64
        if isinstance(labels[0], numbers.Number):
            for class_code, class_name in label_names.items(): 
                df.loc[df[col_label]==class_code, col_label] = class_name
        else: 
            assert isinstance(labels[0], str)

    if verbose:
        print("> After label encoding, label counts:") 
        print(get_label_counts(df))
    
    return df

def load_sgt_embeddings(**kargs):
    # Source dataframe: e.g. GRCh38.106.nmd.transcripts.partial_intron.nmd_eff-t0.9.embed-sgt1.csv 
    from meta_spliceai.sequence_model import sgt_model
    from sklearn.preprocessing import LabelEncoder

    print("> Loading SGT embedding ...")

    verbose = kargs.get("verbose", 1)
    col_label = 'label'  # TransformerModelConfig.col_label
    pos_label = 1
    neg_label = 0

    ca = ClusterAnalysis() # data_params
    output_dir = ca.get_output_directory()

    return_file_id = kargs.get("return_file_id", True)
    threshold_eff = kargs.get('threshold_eff', 0.9)

    eff_scoring_method = kargs.get("eff_scoring_method", "median") # "log_ratio", "median"
    concept_default = f"nmd_eff-t{threshold_eff}" if eff_scoring_method == 'log_ratio' else "nmd_eff-median"
    labeling_concept = kargs.get("labeling_concept", concept_default)  # "biotype"
    
    tissue_type = kargs.get("tissue_type", "all")
    # NOTE: f"nmd_eff-t{threshold_eff}", "nmd_eff-median"  # "biotype"    

    # Model parameters 
    kappa = kargs.get("kappa", 1)
    suffix = kargs.get("suffix", f"embed-sgt{kappa}")

    labeling_params = {
        "labeling_concept": labeling_concept, 
        "tissue_type": tissue_type, 
        "label_names": {0: 'nmd_ineff', 1: 'nmd_eff'}, 
    }
    labeling_params.update(kargs.get("labeling_params", {}))

    df, cols_dict, *rest = \
        sgt_model.load_embeddings(labeling_params=labeling_params, 
            suffix=suffix, 
            return_data_path=True)
    print(f"... shape(df): {df.shape}")  # for NMD-fated transcripts: (16818, 27)
    print(f"... columns: {df.columns.values}")
    # colums: embed_{0-24}, transcript_id, label
    print(f"... num_cols: {cols_dict['num_cols'][:10]}")

    assert len(rest) > 0
    data_path = rest[0]
    print(f"... data path:\n{data_path}\n")
    # NOTE: 
    #   ctype: partial_intron
    #   threshold_eff: 0.9
    #   suffix: embed-sgt1
    #       => GRCh38.106.nmd.transcripts.partial_intron.nmd_eff-t0.9.embed-sgt1.csv
    input_dir = os.path.dirname(data_path)
    input_file = os.path.basename(data_path)
    print(f"... embedding file:\n{input_file}\n")
    print(f"... original label counts:\n{get_label_counts(df, col_label)}\n")
    file_id = '.'.join(input_file.split('.')[:-1])

    # Update labeling 
    target_biotypes = ['nonsense_mediated_decay', ]
    use_class_names = True
    label_names = labeling_params['label_names']

    y_count = get_label_counts(df)
    print(f"... label counts:\n{y_count}\n")
    # print(f"... cols_dict:\n{cols_dict}\n")

    kappa = kargs.get("kappa", 1)
    suffix = kargs.get("suffix", f"embed-sgt{kappa}")

    y = df[col_label].values
    if isinstance(y[0], str): 
        # NOTE: problem with LabelEncoder is no explicit control over which class goes with which integer
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_y = encoder.transform(y)
        y = encoded_y
        print(f"... classes: {encoder.classes_}")
    
    # cond = df[col_label]=='nmd_eff'
    # df.loc[cond, col_label] = pos_label
    # cond = df[col_label]=='nmd_ineff'
    # df.loc[cond, col_label] = neg_label
    df[col_label] = y
    y_count = get_label_counts(df)
    print(f"... label counts (encoded):\n{y_count}\n")
    
    n_classes = len(np.unique(y))

    num_cols = cols_dict['num_cols']
    X = df[num_cols]
    # y is already encoded

    print("> Embedding data spec:")
    print(f"... shape(X): {X.shape}")
    print(f"... label counts:\n{y_count}\n")

    if return_file_id: 
        return (X, y, file_id)

    return (X, y)

def load_nt_embeddings(**kargs): 
    from sklearn.preprocessing import LabelEncoder
    from meta_spliceai.sequence_model import transformer_model as tm
    from tabulate import tabulate

    print("> Loading NT embedding ...")

    verbose = kargs.get("verbose", 1)
    col_label = 'label'  # TransformerModelConfig.col_label
    pos_label = 1
    neg_label = 0

    # Data parameters
    # data_params = {
    #     "data_prefix": SequenceIO.get_data_dir(), 
    #     "experiment": "cluster_analysis", 

    # }
    ca = ClusterAnalysis() # data_params
    output_dir = ca.get_output_directory()
    return_dataframe = kargs.get("return_dataframe", False)

    return_file_id = kargs.get("return_file_id", True)
    threshold_eff = kargs.get('threshold_eff', 0.8)
    layer_index = kargs.get("layer_index", 20)  # from which layer of the transformer was the embedding extracted? 
    
    eff_scoring_method = kargs.get("eff_scoring_method", "median") # "log_ratio", "median"
    concept_default = f"nmd_eff-t{threshold_eff}" if eff_scoring_method == 'log_ratio' else "nmd_eff-median"
    labeling_concept = kargs.get("labeling_concept", concept_default)  # "biotype"
    # NOTE: f"nmd_eff-t{threshold_eff}", "nmd_eff-median"  # "biotype"

    sequence_content_type = kargs.get("sequence_content_type", "partial_intron")
    tissue_type = kargs.get("tissue_type", "all")  # "all", "brain"

    labeling_params = {
        "labeling_concept": labeling_concept, 
        "tissue_type": tissue_type, 
        "label_names": {0: 'nmd_ineff', 1: 'nmd_eff'}, 
    }
    labeling_params.update(kargs.get("labeling_params", {}))

    df, cols_dict, *rest = \
        tm.load_embeddings(labeling_params=labeling_params, 
            layer_index=layer_index, 
                sequence_content_type=sequence_content_type,
                # suffix="embed-nt20",  # this serves as part of the embedding file ID
                return_data_path=True)

    print("> Profiling NT embedding dataframe ...")
    data_path = rest[0] if len(rest) > 0 else 'n/a'
    print(f"... shape(df): {df.shape}")
    print(f"... columns: {df.columns.values}")
    print(f"... data path: {data_path}")

    fn = os.path.basename(data_path)
    file_id = '.'.join(fn.split('.')[:-1])
    # GRCh38.106.nmd.transcripts.partial_intron.nmd_eff-median.embed-nt20.csv

    y_count = df[col_label].value_counts()
    print(f"... label counts:\n{y_count}\n")
    # print(f"... cols_dict:\n{cols_dict}\n")

    # layer_index = kargs.get("layer_index", 20)
    suffix = kargs.get("suffix", f"embed-nt{layer_index}")

    y = df[col_label].values
    if isinstance(y[0], str): 
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_y = encoder.transform(y)
        y = encoded_y
        print(f"... classes: {encoder.classes_}")
    
    # cond = df[col_label]=='nmd_eff'
    # df.loc[cond, col_label] = pos_label
    # cond = df[col_label]=='nmd_ineff'
    # df.loc[cond, col_label] = neg_label
    df[col_label] = y
    y_count = df[col_label].value_counts()
    print(f"... label counts:\n{y_count}\n")
    
    n_classes = y_count.shape[0]

    num_cols = cols_dict['num_cols']
    X = df[num_cols]
    # y is already encoded
    y_counts = get_label_counts(df, dtype=None, verbose=0)

    print("> Embedding data spec:")
    print(f"... shape(X): {X.shape}")
    print(f"... label counts:\n{y_counts}\n")

    if return_dataframe: 
        return df, cols_dict

    if return_file_id: 
        return (X, y, file_id)

    return (X, y)

def run_umap_hdbscan(X, n_clusters=2, n_components=2, **kargs): 

    return

def run_pca_kmeans(X, n_clusters=2, n_components=2, **kargs): 
    import sklearn.metrics
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from meta_spliceai.utils.utils_misc import savefig

    pca = PCA(n_components=n_components)
    pca.fit(X)

    X_pca = pca.transform(X)

    print("> Explained variance ratio:", np.sum(pca.explained_variance_ratio_))
    df = pd.DataFrame(data=X_pca, columns=['x1', 'x2'])
    print(df.head())

    kmeans = KMeans(n_clusters=n_clusters, max_iter =300)
    kmeans.fit(df)

    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_

    fig = plt.figure(figsize=(10, 8))

    colmap = {1: 'r', 2: 'b'} # Todo: {1: 'r', 2: 'g', 3: 'b', 4: 'c'}
    colors = list(map(lambda x: colmap[x+1], labels))
    plt.scatter(df['x1'], df['x2'], color=colors, alpha=0.5, edgecolor=colors)

    # Save plot
    # --------------------------------------------------------
    ext = 'tif'
    output_file = kargs.get("output_file", f"pca_kmeans.{ext}") 
    suffix = kargs.get("suffix", "")
    if suffix: 
        output_file_stem = output_file.split('.')[:-1]
        output_file = f"{output_file_stem}.{suffix}.{ext}"

    # NOTE: 'labeling_concept' should also be in the naming of the output because it determines not only 
    #       the labeling but also the transcript subset
    output_dir = kargs.get("output_dir", os.getcwd())
    output_path = os.path.join(output_dir, output_file)

    print(f"[cluster] Saving cluster analysis to:\n{output_path}\n")   
    savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)
    # --------------------------------------------------------

    return labels

def demo_clustering_pca_kmeans(**kargs): 
    from meta_spliceai.sequence_model import transformer_model as tm
    from meta_spliceai.utils.utils_misc import savefig

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    import sklearn.metrics
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # Data parameters 
    ca = ClusterAnalysis() 
    output_dir = ca.get_output_directory()

    # Model parameters
    layer_index = 20
    ctype = sequence_content_type = "partial_intron"
    tissue_type = 'all'
    eff_scoring_method = 'median'
    labeling_concept = "nmd_eff-median"

    labeling_params = {
            "labeling_concept": labeling_concept, 
            "tissue_type": tissue_type, 
            "eff_scoring_method": eff_scoring_method, 
            "label_names": {0: 'nmd_ineff', 1: 'nmd_eff'}, 
    }

    # Action parameters 
    test_clustering = True

    # Embedding type 
    embedding_type = 'nt'  
    # NOTE: 
    #    nt: nucleotide transformer
    #   sgt: 

    # (X, y)
    embeddings_nt, label, *rest = \
        load_nt_embeddings(
            layer_index=layer_index, 

            sequence_content_type=ctype,  # partial_intron, flanking_sc
            labeling_params = labeling_params, 
            
            return_file_id = True
            # tissue_type=tissue_type,  # e.g. "all", "brain", ...
            #     eff_scoring_method=eff_scoring_method,  # "median", "log_ratio"
            #     labeling_concept=labeling_concept   # NOTE: can be automatically determined
    )
    
    file_id = ''
    if len(rest) > 0: file_id = rest[0]

    trpt_embedding = embeddings_nt

    if test_clustering: 

        highlight("> Sequence clustering via PCA followed by k-means")

        pca = PCA(n_components=2)
        pca.fit(trpt_embedding)

        X = pca.transform(trpt_embedding)

        print(np.sum(pca.explained_variance_ratio_))
        df = pd.DataFrame(data=X, columns=['x1', 'x2'])
        print(df.head())

        kmeans = KMeans(n_clusters=4, max_iter =300)
        kmeans.fit(df)

        labels = kmeans.predict(df)
        centroids = kmeans.cluster_centers_

        fig = plt.figure(figsize=(10, 8))
        colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'c'}
        colors = list(map(lambda x: colmap[x+1], labels))
        plt.scatter(df['x1'], df['x2'], color=colors, alpha=0.5, edgecolor=colors)

        # Save plot
        # --------------------------------------------------------
        ext = 'tif'
        suffix = kargs.get("suffix", "pca_kmeans") 
        if file_id: 
            output_file = f"{file_id}.{suffix}.{ext}"
        else: 
            output_file = f"{suffix}.{ext}"

        # NOTE: 'labeling_concept' should also be in the naming of the output because it determines not only 
        #       the labeling but also the transcript subset
        output_path = os.path.join(output_dir, output_file)

        print(f"[transcript_embedding] Saving transcript embedding cluster analysis to:\n{output_path}\n")   
        savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)
        # --------------------------------------------------------

def demo_clustering_umap_hdbscan(**kargs): 
    import meta_spliceai.nmd_concept.seq_model as sm
    # from meta_spliceai.sequence_model import sgt_model

    # Data parameters
    data_params = ClusterAnalysis.data_params
    ca = ClusterAnalysis() 
    output_dir = ca.get_output_directory()

    sequence_content_type = ctype = 'partial_intron'
    tissue_type = ttype = "all" # "all", "brain"
    eff_scoring_method = "median"
    labeling_concept = "nmd_eff-median"

    labeling_params = {
            "labeling_concept": labeling_concept, 
            "tissue_type": tissue_type, 
            "eff_scoring_method": eff_scoring_method, 
            "label_names": {0: 'nmd_ineff', 1: 'nmd_eff'}, 
    }

    # (X, y)
    print("> Loading NT-based embeddings ...")
    embeddings_nt, label, *rest = \
        load_nt_embeddings(
            layer_index=20, 
            sequence_content_type=ctype,  # partial_intron, flanking_sc
            labeling_params=labeling_params, 
            return_file_id = True
            # tissue_type=ttype,  # e.g. "all", "brain", ...
            #     eff_scoring_method=eff_scoring_method,  # "median", "log_ratio"
            #     labeling_concept=labeling_concept   # NOTE: can be automatically determined
    )
    
    # Load other embeddings for comparison
    print("> Loading SGT embeddings ...")
    embeddings_sgt, label, *rest = \
        load_sgt_embeddings(
            kappa = 1, 
            sequence_content_type=ctype,  # partial_intron, flanking_sc
            labeling_params=labeling_params, 
            return_file_id=True
    )

    file_id = "" # f'umap_hdbscan-C{ctype}-T{ttype}-{eff_scoring_method}'
    if len(rest) > 0: file_id = rest[0]

    # Results with default hyperparameters
    clusters_default = generate_clusters(embeddings_nt, 
                                     n_neighbors = 15, 
                                     n_components = 5, 
                                     min_cluster_size = 10,
                                     random_state= 42)
    labels_def, cost_def = score_clusters(clusters_default)
    print("> Labels:", labels_def) # 2
    print("> Cost:", cost_def) # 0

    # Bayesian optimization with Hyperopt
    hspace = {
        "n_neighbors": hp.choice('n_neighbors', range(3,16)),
        "n_components": hp.choice('n_components', range(3,16)),
        "min_cluster_size": hp.choice('min_cluster_size', range(2,16)),
        "random_state": 42
    }

    label_lower = 30
    label_upper = 100
    max_evals = 100

    best_params_nt, best_clusters_nt, trials_nt = bayesian_search(embeddings_nt, 
                                                                 space=hspace, 
                                                                 label_lower=label_lower, 
                                                                 label_upper=label_upper, 
                                                                 max_evals=max_evals)

    # Best parameters for embeddings_nt
    # {'min_cluster_size': 9, 'n_components': 12, 'n_neighbors': 14, 'random_state': 42}
    # ... best clusters:
    # HDBSCAN(min_cluster_size=9)
    # ... trials: <hyperopt.base.Trials object at 0x7f919f2f5c70>

    embed_type = "nt"
    print(f"> Bayesian Search for embedding type: {embed_type}")
    print(f"... best params:\n{best_params_nt}\n")
    print(f"... best clusters:\n{best_clusters_nt}\n")
    print(f"... trials: {trials_nt}")
    print("> Best trial and its parameters? ")
    print(trials_nt.best_trial)

    # best_params_sgt, best_clusters_sgt, trials_sgt = bayesian_search(embeddings_sgt, 
    #                                                              space=hspace, 
    #                                                              label_lower=label_lower, 
    #                                                              label_upper=label_upper, 
    #                                                              max_evals=max_evals)

    # embed_type = "sgt"
    # print(f"> Bayesian Search for embedding type: {embed_type}")
    # print(f"... best params:\n{best_params_sgt}\n")
    # print(f"... best clusters:\n{best_clusters_sgt}\n")
    # print(f"... trials: {trials_sgt}")
    # print("> Best trial and its parameters? ")
    # print(trials_sgt.best_trial)

    ### 
    # Dummy
    embeddings_sgt = embeddings_nt
    best_params_sgt = best_params_nt
    best_clusters_sgt = best_clusters_nt

    # Output
    ext = "tif"
    embed_types = {'nt': embeddings_nt , 
                   'sgt': embeddings_sgt, 
                   }
    best_clusters = {'nt': best_clusters_nt, 
                    'sgt': best_clusters_sgt, 
                    }
    for embed_type in embed_types:
        suffix = kargs.get("suffix", f"umap_hdbscan-{embed_type}")
        if file_id: 
            output_path = os.path.join(output_dir, f"{file_id}.{suffix}.{ext}")
        else: 
            output_path = os.path.join(output_dir, f"cluster_analysis.{suffix}.{ext}")

        plt = plot_clusters(embed_types[embed_type], best_clusters[embed_type], output_path=output_path)
        
    # Relation between clusters and labels? 
    
    # Load transcript sequence data
    concept = "nmd"
    df_seq = sm.load_transcript_sequence_training_data(
                            # concept=concept, 
                            sequence_content_type=sequence_content_type, 
                                labeling=labeling_concept,
                                labeling_params=labeling_params,
                                    apply_chunking=False,  
                                        data_prefix=data_params['data_prefix'], 
                                        source=data_params['source'], version=data_params['version'], 
                                        artifact_dir=data_params['artifact_dir'])
    # NOTE: ensembl downloaded transcripts 
    #       size: 1283.63 Mb
    print("> Loaded sequence data:")
    print(f"... dimension: {df_seq}")
    print(f"... columns: {list(df_seq.columns)}")

    # cluster_dict = {
    #             'label_nt': best_clusters_nt,
    #             'label_svg': best_clusters_svg, 
    #             }

    # results_df = combine_results(df_seq[['sequence', 'label']], cluster_dict)
    # results_df.head()


    return

def run_hdbscan_flat_clustering(X, **kargs): 
    from hdbscan import HDBSCAN
    from hdbscan.flat import (HDBSCAN_flat,
                          approximate_predict_flat,
                          membership_vector_flat,
                          all_points_membership_vectors_flat)
    from meta_spliceai.utils.utils_misc import savefig

    def save_plot(n_clusters, output_dir=None, output_file=None, ext='tif'): 
        # ext = 'tif'
        if not output_dir: output_dir = os.getcwd()
        if not output_file: output_file = f"flat_hdbscan-nC{n_clusters}.{ext}"

        # NOTE: 'labeling_concept' should also be in the naming of the output because it determines not only 
        #       the labeling but also the transcript subset
        output_path = os.path.join(output_dir, output_file)

        print(f"[clustering] Saving cluster analysis (flat hdbscan) to:\n{output_path}\n")   
        savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)

    # Data parameters
    data_params = ClusterAnalysis.data_params
    ca = ClusterAnalysis() 
    output_dir = kargs.get("output_dir", ca.get_output_directory())

    # Bayesian optimization with Hyperopt
    hspace = {
        "n_neighbors": hp.choice('n_neighbors', range(3,16)),
        "n_components": hp.choice('n_components', range(3,16)),
        "min_cluster_size": hp.choice('min_cluster_size', range(2,16)),
        "random_state": 42
    }

    label_lower = 2
    label_upper = 20
    max_evals = 100

    # Find the optimal parameters for UMAP coupled with HDBSCAN
    best_params, best_clusters, trials = bayesian_search(X, 
                                                            space=hspace, 
                                                            label_lower=label_lower, 
                                                            label_upper=label_upper, 
                                                            max_evals=max_evals)

    min_dist = kargs.get("mid_dist", 0.1)

    X_umap = umap.UMAP(n_neighbors=best_params['n_neighbors'], 
                          n_components=best_params['n_components'], 
                          min_dist = min_dist,  
                          #metric='cosine',
                          random_state=42).fit_transform(X)


    # Train the base HDBSCAN class
    clusterer = HDBSCAN(cluster_selection_method='eom', 
                           min_cluster_size=best_params['min_cluster_size']).fit(X_umap)

    n_clusters_arr = np.arange(2, 11).astype(int)

    labels = proba = None

    # Alternatively, expect a few n_clusters ... 
    # for n_clusters in [2, ]: # n_clusters_arr:
    
    n_clusters = kargs.get('n_clusters', 2)

    clusterer = HDBSCAN_flat(X_umap, 
                                clusterer=clusterer,
                                n_clusters=n_clusters)
    # This does not re-train the clusterer;
    #    instead, it extracts flat clustering from the existing hierarchy
    labels = clusterer.labels_
    proba = clusterer.probabilities_

    plt.figure(figsize=(7, 3))
    plt.title(f"Flat clustering for {n_clusters} clusters")
    plt.scatter(X[labels>=0, 0], X[labels>=0, 1], c=labels[labels>=0], s=5,
                cmap=plt.cm.jet)
    plt.scatter(X[labels<0, 0], X[labels<0, 1], c='k', s=3, marker='x', alpha=0.2)
    # plt.show()
    print(f"Unique labels (-1 for outliers): {np.unique(labels)}")

    ext = "tif"
    suffix = kargs.get("suffix", "")
    output_file = f"flat_hdbscan-nC{n_clusters}-{suffix}.{ext}" if suffix else f"flat_hdbscan-nC{n_clusters}.{ext}"
    save_plot(n_clusters, output_dir=output_dir, output_file=output_file) # output_file=None

    return labels

def transforma_and_cluster_efficiency_matrix(): 
    pass

def demo_flat_hdbscan_clustering(**kargs): 
    """
    
    Related
    -------
    1. transform_and_cluster_efficiency_matrix()
    """
    from meta_spliceai.utils.utils_misc import savefig
    import meta_spliceai.nmd_concept.gene_expr_analyzer as gea
    from meta_spliceai.utils.utils_data import to_dataframe
    
    import seaborn as sns
    from sklearn.datasets import make_blobs, make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Todo: configuration
    col_gid = 'gene_id'
    col_tid = 'transcript_id'
    col_label = 'label'
    col_btype = "transcript_biotype"
    col_measure = "eff_measure"
    col_corr = "corr"

    data_params = ClusterAnalysis.data_params
    ca = ClusterAnalysis() 
    output_dir = kargs.get("output_dir", ca.get_output_directory())

    use_case = kargs.get('use_case', "mock")  # mock, nt (nucleotide transformer), eff (NMD efficiency matrix)
    use_umap_hdbscan_pipeline = kargs.get("use_umap_hdbscan", True)

    n_clusters = kargs.get("n_clusters", 2)

    # Create mock data
    df = None
    num_cols = []
    n_upper = n_lower = None
    if use_case == 'mock': 
        centers = [(0, 2), (-0.2, 0), (0.2, 0),
           (1.5, 0), (2., 1.), (2.5, 0.)]
        std = [0.5, 0.08, 0.06, 0.25, 0.25, 0.25]
        X, y = make_blobs(n_samples=[700, 300, 800, 1000, 400, 1500],
                        centers=centers,
                        cluster_std=std)
        X1, y1 = make_moons(n_samples=5000, noise=0.07)
        X1 += 3.
        y1 += len(centers)
        X = np.vstack((X, X1))
        y = np.concatenate((y, y1))
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        df = to_dataframe(X, y)
        num_cols = list(df.columns.drop(col_label))

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.2,
                                                random_state=42)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].set_title("Training set")
        axes[0].scatter(X[:, 0], X[:, 1], c=y, s=5)
        axes[1].set_title("Test set")
        axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=5)
        plt.suptitle("Dataset used for illustration")

        # save plot
        ext = 'tif'
        subject = "blob_data"
        default_output_file = f'{subject}.{ext}'
        default_output_dir = os.path.join(os.getcwd(), 'plot')
        output_path = kargs.get("output_path", os.path.join(output_dir, default_output_file))

        print(f"[umap-hdbscan] Saving synthetic data to:\n{output_path}\n")   
        savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)

    elif use_case == "eff":
        threshold_eff = kargs.get("threshold_eff", 0.50)
        threshold_expr = kargs.get("threshold_expr", [-1.0, 1.0])
        tissue_type = kargs.get("tissue_type", "all")
        eff_scoring_method = kargs.get("eff_scoring_method", "log_ratio")
        label_names = kargs.get("label_names", {0: 'nmd_ineff', 1: 'nmd_eff'} )  

        highlight(f"> Computing efficiency matrix at th_eff={threshold_eff}, ttype={tissue_type}, eff_method={eff_scoring_method}")
        df_eff = gea.load_nmd_eff_matrix(
            # df_trpt=df_trpt  # Set to None by using default transcript dataframe
            threshold_eff=threshold_eff, threshold_expr=threshold_expr, 
                tissue_type=tissue_type, 
                eff_scoring_method=eff_scoring_method, 
                return_eff_measure=True, # if True, will include eff_measure in `df_eff` (efficiency matrix dataframe)`
                use_cached=False # if True, will use cached/pre-computed labeling data; otherwise, recompute the labeling data
        )

        highlight("> Ranking transcripts by NMD efficiency measure ...")
        is_corr_specific = False
        if col_measure in df_eff.columns: 
            # then just use the default NMD efficiency score to sort/rank the transcripts 
            print(f"> Found NMD eff measure in efficiency matrix ...")

            if col_corr in df_eff.columns: 
                df_eff = df_eff.sort_values(by=[col_measure, col_corr, ], ascending=False) 
                is_corr_specific = True
                print("... efficiency metric is correlation specific")
            else: 
                df_eff = df_eff.sort_values(by=col_measure, ascending=False)
        else:  
            # Use percentile score as the default 
            print(f"> Using percentile scoring by default (th_eff={threshold_eff})")
            percentile_scores = []
            for i in range(X.shape[0]): 
                percentile_scores.append(np.percentile(X[i, :], threshold_eff*100)) 
            df_eff[col_measure] = percentile_scores
            df_eff = df_eff.sort_values(by=col_measure, ascending=False)

            percentile_scores = []
            for i in range(X.shape[0]): 
                percentile_scores.append(np.percentile(X[i, :], threshold_eff*100)) 
            df_eff['p_score'] = percentile_scores
            df_eff = df_eff.sort_values(by='p_score', ascending=False)

        # Test
        # ------------------------------------------------------------------
        n = 15
        print(f"> top {n} by percentile={threshold_eff} ...")

        if is_corr_specific: 
            print(df_eff[[col_gid, col_tid, col_label, col_measure, col_corr]].head(n))
        else: 
            print(df_eff[[col_gid, col_tid, col_label, col_measure]].head(n))

        print(f"> last {n} by percentile={threshold_eff} ...")
        if is_corr_specific: 
            print(df_eff[[col_gid, col_tid, col_label, col_measure, col_corr]].tail(n))
        else: 
            print(df_eff[[col_gid, col_tid, col_label, col_measure]].tail(n))
        # ------------------------------------------------------------------

        print("... selecting a subset of transcripts to facilitate cluster analysis")
        n = n_upper = n_lower = kargs.get("topn", 500)
        if n is None: 
            # no-op: preserving all transcripts
            pass
        else:
            # Fitler the transcripts to preserve only sufficient sample size for viz 
            #  - Select top 100 NMD efficient transcripts by percentile
            assert df_eff.shape[0] > n * 2
            df_eff_topn = df_eff.iloc[:n, :]
            df_eff_lastn = df_eff.iloc[-n:, :]
            df_eff = pd.concat([df_eff_topn, df_eff_lastn], ignore_index=True)


        print("... singling out numeric columns")    
        cols_meta = [col_gid, col_tid, col_label]  # Todo: configuration
        # cols_meta = [col_gid, col_tid, col_label, col_measure, col_corr]
        
        # num_cols = list(df_eff.columns.drop(cols_meta))
        # NOTE: cols_meta could also contain additional columns like 
        #       col_meausre, col_corr
        num_cols = list(df.columns[df.columns.str.startswith('SRR')])

        # new, filtered, ordered X
        X = df_eff[num_cols].values
        assert np.sum(np.isnan(X)) == 0
        assert not np.isnan(X).any(), f"Efficiency matrix has null values n={np.sum(np.isnan(X))}"

        print(f"... shape of efficiency matrix (topn={n}): {X.shape}, N={X.shape[0]}")
        
        # Standardize the matrix
        epsilon = 1e-6
        X = (X-X.mean(axis=0))/(X.std(axis=0)+epsilon) 
        assert np.sum(np.isnan(X)) == 0
        
        df_eff = encode_labels(df_eff, label_names=label_names, mode='to_str')
        y = df_eff[col_label].values

        df = to_dataframe(X, y, feature_cols=num_cols)   
        print(f"> final df shape: {df.shape}") 

    elif use_case == "nt": 
        # Model parameters
        layer_index = 20
        ctype = sequence_content_type = "partial_intron"
        tissue_type = 'all'
        eff_scoring_method = 'log_ratio'
        threshold_eff = 0.8

        concept_base = 'nmd_eff'
        concept_default = f"{concept_base}-t{threshold_eff}" if eff_scoring_method == 'log_ratio' else f"{concept_base}-{eff_scoring_method}"
        labeling_concept = concept_default

        labeling_params = {
                "labeling_concept": labeling_concept, 
                "tissue_type": tissue_type, 
                "eff_scoring_method": eff_scoring_method, 
                "label_names": {0: 'nmd_ineff', 1: 'nmd_eff'}, 
        }

        # Embedding type 
        embedding_type = 'nt'  
        # NOTE: 
        #    nt: nucleotide transformer
        #   sgt: 

        # (X, y)
        df_embed, cols_dict, *rest = \
            load_nt_embeddings(
                layer_index=layer_index, 

                sequence_content_type=ctype,  # partial_intron, flanking_sc
                labeling_params = labeling_params, 
                
                return_file_id = True, 
                return_dataframe = True, 
                # tissue_type=tissue_type,  # e.g. "all", "brain", ...
                #     eff_scoring_method=eff_scoring_method,  # "median", "log_ratio"
                #     labeling_concept=labeling_concept   # NOTE: can be automatically determined
        )
        # E.g. <prefix>/ensembl/GRCh38.106/sequence/GRCh38.106.nmd.transcripts.partial_intron.nmd_eff-median.embed-nt20.csv

        X_nt = df_embed[cols_dict['num_cols']].values
        num_cols = cols_dict['num_cols']
        print(f"> Shape of embedding matrix: {X_nt.shape}")

        X = (X_nt-X_nt.mean(axis=0))/(X_nt.std(axis=0))

        df_embed = encode_labels(df_embed, label_names=labeling_params['label_names'], mode='to_str')
        y = df_embed[col_label].values

        # Dimensionality reduction 
        # pca = PCA(n_components=n_components)
        # pca.fit(X_ref)
        # X_ref_pca = pca.transform(X_ref)

        df = to_dataframe(X, y, feature_cols=num_cols)
    ### end data retrieval

    highlight(f"> Generating cluster map (N={n_upper}x2)...")
    
    # Subsampling the data: too many data points will clutter the heatmap
    # ---------------------------------------
    # df = df.sample(n=min(df.shape[0], 200))
    X_subset = df[num_cols].values
    print(f"> number of NaN in X_subset: {np.sum(np.isnan(X_subset))}")
    y_subset = df[col_label].values
    pca = PCA(n_components=15)
    X_subset = pca.fit_transform(X_subset)
    # ---------------------------------------

    labels = pd.Series(y_subset)
    color_symbols = dict(zip(np.unique(labels), "rb")) # r: 0, b: 1
    row_colors = labels.map(color_symbols)
    print(f"... shape(X_subset): {X_subset.shape}")
    print(f"... classes -> colors: {color_symbols}")
    print(f"... row colors: {row_colors}")
    fig = sns.clustermap(X_subset, row_colors=row_colors.values, cmap="mako", vmin=0, vmax=10)
    # NOTE: Since X is an numpy array, row_colors needs to be converted to numpy array as well 
    #       row_colors.to_numpy()
    #       row_colors.values
    
    ext = 'tif'
    output_file = f"heatmap-{use_case}" # .{ext}"
    if n_upper is not None: output_file = f"{output_file}-N{n_upper}"
    output_file = f"{output_file}-nC{n_clusters}.{ext}"  # NOTE: no N- because ALL data points are considered
    
    output_path = os.path.join(output_dir, output_file)
    print(f"> Saving cluster map to:\n{output_path}\n")
    fig.savefig(output_path)
    
    if use_umap_hdbscan_pipeline: 
        highlight(f"> Running UMAP-HDBSCAN pipeline (n_clusters={n_clusters})")
        suffix = use_case if n_upper is None else f"N{n_upper}-{use_case}"
        run_hdbscan_flat_clustering(X, n_clusters=n_clusters, suffix=suffix)
    else: 
        highlight(f"> Running (flat) HDBSCAN (n_clusters={n_clusters})")

        from hdbscan.flat import (HDBSCAN_flat,
                          approximate_predict_flat,
                          membership_vector_flat,
                          all_points_membership_vectors_flat)

        clusterer = HDBSCAN_flat(X,
                            cluster_selection_method='eom',
                            n_clusters=n_clusters, min_cluster_size=10)
        labels = clusterer.labels_
        proba = clusterer.probabilities_

        plt.scatter(X[labels>=0, 0], X[labels>=0, 1], c=labels[labels>=0], s=5,
                    cmap=plt.cm.jet)
        plt.scatter(X[labels<0, 0], X[labels<0, 1], c='k', s=3, marker='x', alpha=0.2)
        # plt.show()

        ext = "tif"
        output_file = f"flat_hdbscan-{use_case}-N{n_upper}-nC{n_clusters}.{ext}"
        output_path = os.path.join(output_dir, output_file)
        print(f"[clustering] Saving simple cluster analysis (flat hdbscan) to:\n{output_path}\n")   
        savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)

        print(f"Unique labels (-1 for outliers): {np.unique(labels)}")
    
    return

def demo_spectrual_biclustering(**kargs): 
    from sklearn.datasets import make_biclusters
    from sklearn.cluster import SpectralCoclustering
    from sklearn.metrics import consensus_score
    from sklearn.decomposition import PCA

    import meta_spliceai.nmd_concept.gene_expr_analyzer as gea
    from meta_spliceai.utils.utils_misc import savefig

    # Todo: configuration
    col_gid = 'gene_id'
    col_tid = 'transcript_id'
    col_label = 'label'

    useMockData = kargs.get('use_mock_data', False)
    data = data_ref = None
    has_cluster_labels = False

    n_clusters = kargs.get("n_clusters", 5)
    
    # PCA 
    n_components = kargs.get("n_components", 30)

    if useMockData: 
        data, rows, columns = make_biclusters(
            shape=(300, 300), n_clusters=n_clusters, noise=5, shuffle=False, random_state=0
        )
        print(f"> shape(data): {data.shape}")
        
        # shuffle clusters
        rng = np.random.RandomState(0)
        row_idx = rng.permutation(data.shape[0])
        col_idx = rng.permutation(data.shape[1])
        data = data[row_idx][:, col_idx]

        has_cluster_labels = True
    else: 
        threshold_eff = kargs.get("threshold_eff", 0.8)
        threshold_expr = kargs.get("threshold_expr", [-1.0, 1.0])
        tissue_type = kargs.get("tissue_type", "all")
        eff_scoring_method = kargs.get("eff_scoring_method", "log_ratio")

        print(f"> Computing efficiency matrix at th_eff={threshold_eff}, ttype={tissue_type}, eff_method={eff_scoring_method}")
        df_eff = gea.load_nmd_eff_matrix(
            # df_trpt=df_trpt  # Set to None by using default transcript dataframe
            threshold_eff=threshold_eff, threshold_expr=threshold_expr, 
                tissue_type=tissue_type, 
                eff_scoring_method=eff_scoring_method)

        # Remove meta data
        cols_meta = [col_gid, col_tid, col_label]  # Todo: configuration
        X = df_eff.drop(cols_meta, axis=1).values

        print(f"> shape of efficiency matrix: {X.shape}")

        # Standardize the matrix
        X_std = (X-X.mean(axis=0))/(X.std(axis=0))

        # Dimensionality reduction 
        pca = PCA(n_components=n_components)
        pca.fit(X_std)
        X_pca = pca.transform(X_std)

        data = X_pca

        # Cluster membership
        rows = columns = None # NMD efficiency matrix is not annotated by default

        highlight("> Loading reference data: NT-based transcript embeddings ...")
        
        # Model parameters
        layer_index = 20
        ctype = sequence_content_type = "partial_intron"
        tissue_type = 'all'
        eff_scoring_method = 'median'
        labeling_concept = "nmd_eff-median"

        labeling_params = {
                "labeling_concept": labeling_concept, 
                "tissue_type": tissue_type, 
                "eff_scoring_method": eff_scoring_method, 
                "label_names": {0: 'nmd_ineff', 1: 'nmd_eff'}, 
        }

        # Embedding type 
        embedding_type = 'nt'  
        # NOTE: 
        #    nt: nucleotide transformer
        #   sgt: 

        # (X, y)
        df_embed, cols_dict, *rest = \
            load_nt_embeddings(
                layer_index=layer_index, 

                sequence_content_type=ctype,  # partial_intron, flanking_sc
                labeling_params = labeling_params, 
                
                return_file_id = True, 
                return_dataframe = True, 
                # tissue_type=tissue_type,  # e.g. "all", "brain", ...
                #     eff_scoring_method=eff_scoring_method,  # "median", "log_ratio"
                #     labeling_concept=labeling_concept   # NOTE: can be automatically determined
        )
        # E.g. <prefix>/ensembl/GRCh38.106/sequence/GRCh38.106.nmd.transcripts.partial_intron.nmd_eff-median.embed-nt20.csv

        X_nt = df_embed[cols_dict['num_cols']].values
        print(f"> Shape of embedding matrix: {X_nt.shape} =?= {X.shape} (eff matrix)")

        X_ref = (X_nt-X_nt.mean(axis=0))/(X_nt.std(axis=0))

        # Dimensionality reduction 
        pca = PCA(n_components=n_components)
        pca.fit(X_ref)
        X_ref_pca = pca.transform(X_ref)

        # Standardize the embeddings
        data_ref = X_ref_pca

        assert data_ref.shape == data.shape

    # plt.matshow(data, cmap=plt.cm.Blues)
    # plt.title("Original dataset")

    # plt.matshow(data, cmap=plt.cm.Blues)
    # plt.title("Shuffled dataset")

    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
    model.fit(data)

    if has_cluster_labels: 
        assert rows is not None and columns is not None
        print("> Given known cluster memberships as a priori")
        score = consensus_score(model.biclusters_, (rows[:, row_idx], columns[:, col_idx]))

        print("... consensus score (wrt the labeling): {:.3f}".format(score))
    else: 
        # The source data dosen't have labels so we'll use a reference dataset, from which another biclustering is computed
        assert data_ref is not None
        model_ref = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
        model_ref.fit(data_ref)
        
        # if data.shape == data_ref.shape: 
        #     score = consensus_score(model.biclusters_, model_ref.biclusters_)

        #     print("... consensus score (wrt ref data): {:.3f}".format(score))
        # NOTE: matrix contains invalid numeric entries

    fit_data = data[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]

    fig = plt.figure(figsize=(100, 50))

    plt.matshow(fit_data, cmap=plt.cm.Blues)
    plt.title("After biclustering; rearranged to show biclusters")

    # plt.show()

    # Save plot: Configure output
    data_params = ClusterAnalysis.data_params
    ca = ClusterAnalysis() 
    output_dir = kargs.get("output_dir", ca.get_output_directory())
    # --------------------------------------------------------
    ext = 'tif'
    cluster_method = "spectral_bicluster"
    default_output_file = f'{cluster_method}.{ext}'
    default_output_dir = os.path.join(os.getcwd(), 'plot')
    output_path = kargs.get("output_path", os.path.join(output_dir, default_output_file))

    print(f"[biclustering] Saving cluster analysis ({cluster_method}) file to:\n{output_path}\n")   
    savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)
    # --------------------------------------------------------

    if data_ref is not None: 
        plt.clf()

        fit_data = data_ref[np.argsort(model_ref.row_labels_)]
        fit_data = fit_data[:, np.argsort(model_ref.column_labels_)]

        plt.matshow(fit_data, cmap=plt.cm.Blues)
        plt.title("After biclustering; rearranged to show biclusters")

        ext = 'tif'
        cluster_method = "spectral_bicluster"
        suffix = 'ref'
        default_output_file = f'{cluster_method}-{suffix}.{ext}'
        default_output_dir = os.path.join(os.getcwd(), 'plot')
        output_path = kargs.get("output_path", os.path.join(output_dir, default_output_file))

        print(f"[biclustering] Saving reference cluster analysis ({cluster_method}) file to:\n{output_path}\n")   
        savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)

    return


def test(): 
    from meta_spliceai.sequence_model import transformer_model as tm

    start_time = time.time()

    SequenceIO.proj_name = 'nmd'    
    data_prefix = SequenceIO.data_prefix = "/mnt/SpliceMediator/splice-mediator"
    # Options: SequenceIO.get_data_dir(), 
    #          "/mnt/SpliceMediator/splice-mediator" 
    #          "/mnt/nfs1/splice-mediator"
    print(f"[test] data prefix:\n{SequenceIO.get_data_dir()}\n")

    # Remember ot also configure the local analysis object
    ClusterAnalysis.data_params['data_prefix'] = data_prefix  

    # PCA + Kmeans 
    # demo_clustering_pca_kmeans()

    # UMAP + HDBSCAN
    # demo_clustering_umap_hdbscan()
    demo_flat_hdbscan_clustering(use_case='eff',  # 'eff', 'nt'
                use_umap_hdbscan=True, 
                n_clusters=2, 
                threshold_eff=0.95, 
                topn=1000)  
    # NOTE: "flat" HDBSCAN: allows for choosing number of clusters specifically while using HDBSCAN

    # Spectral biclustering (on NMD efficiency score matrix, etc)
    # demo_spectrual_biclustering()

    # NT (nucleotide transformer) vs SGT 

    delta_t = time.time() - start_time
    print(f"[demo] Elapsed {delta_t} seconds ...")

    return

if __name__ == "__main__": 
    test()