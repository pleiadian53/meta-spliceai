
import os, sys
# sys.path.append('..')

import numpy as np
# np.random.seed(123)
import pandas as pd

import shap  

# from .utils import highlight
from meta_spliceai.mllib.utils import highlight

def get_top_features(shap_values, feature_names, N=5, **kargs):
    # Check if the shap_values is of type Explanation and extract values if needed
    if isinstance(shap_values, shap.Explanation):
        shap_values = shap_values.values
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Get the indices of the top features
    top_indices = np.argsort(mean_abs_shap)[-N:]
    
    # Get the names of the top features
    top_feature_names = [feature_names[i] for i in top_indices]

    return_as_dataframe = kargs.get('return_as_dataframe', False) 
    if return_as_dataframe: 
        adict = {'Feature': top_feature_names, 'Importance': mean_abs_shap[top_indices]}
        return pd.DataFrame(adict)
    
    return top_feature_names

def xgboost_plot_importance(model, importance_type="gain", topn=15, **kargs): 
    import matplotlib.pyplot as plt
    import pandas as pd

    feature_importance = model.get_booster().get_score(importance_type=importance_type)
    sorted_idx = sorted(feature_importance, key=feature_importance.get, reverse=True)[:topn]

    plt.figure(figsize=(12, 8))

    plt.barh(range(len(sorted_idx[::-1])), [feature_importance[i] for i in sorted_idx[::-1]], align='center')
    plt.yticks(range(len(sorted_idx[::-1])), [i for i in sorted_idx[::-1]])

    for i, v in enumerate([feature_importance[i] for i in sorted_idx[::-1]]):
        plt.text(v + 0.01, i, str(round(v, 2)), color='blue', va='center', fontweight='bold')

    plt.xlabel('Feature Importance')
    plt.title('Feature Importance (cover)')
    
    interactive = kargs.get("interactive", False)

    if interactive: 
        plt.show()

    else: 
        ext = 'pdf'
        output_file = kargs.get('output_file', f"xgboost-feature-importance-{importance_type}.{ext}") 
        output_dir = kargs.get("output_dir", os.getcwd())
        output_path = os.path.join(output_dir, output_file)

        plt.savefig(output_path)

    return

def demo_shap(**kargs): 
    # import shap
    import xgboost
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from meta_spliceai.nmd_concept.nmd_targets_analyzer import get_nmd_data

    # Load data
    X, y = shap.datasets.adult()

    # X, y, meta_data = \
    #     get_nmd_data(policy='topn', threshold_eff=0.3, topn=300, 
    #                     verbose=1, 
    #                     return_x_as_dataframe=True, return_meta_data=True) # return dataframe so that we can get feature names / columns
    # feature_names = list(X.columns)
    # X = X.values
    print(f"... shape(X): {X.shape}, shape(y): {y.shape}")
    print(f"... type(X): {type(X)}, type(y): {type(y)}")
    classes = np.unique(y)
    n_classes = len(classes)
    print(f"... classes (n={n_classes}): {classes}")
    feature_names = list(X.columns)
    # f_encoded = list(set(meta_data['features_encoded']) - set(meta_data['features_original']))
    # print(f"... encoded features (n={len(f_encoded)}):\n{f_encoded}\n")

    experiment_id = "test"
    output_dir = kargs.get("output_dir", "../experiments/test")

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)
        
    # Tree explainer
    model = xgboost.XGBClassifier(n_estimators=20)
    model.fit(X, y)

    # Explain model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    print(f"> type(shap_values): {type(shap_values)}")
    if isinstance(shap_values, np.ndarray): 
        print(f"... shape: {shap_values.shape}")

    ### 
    plt.clf()

    shap.summary_plot(shap_values, X_train, plot_type="bar")
    # NOTE: Feeding X_train for model interpretation
    #       If we wish to know how unseen data is predicted by the model and how it values feature importance, 
    #       then feed X_test instead

    ext = "pdf"
    output_file = f"shap_summary_plot-{experiment_id}.{ext}" 
    output_path = os.path.join(output_dir, output_file)

    print(f"[output] Saving SHAP summary plot to:\n{output_path}\n")
    plt.savefig(output_path)
    ### 

    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)  # return type: shap._explanation.Explanation
    print(f"[test] type of shap_values: {type(shap_values)}") # shap._explanation.Explanation
    print(f"... attributes:\n{dir(shap_values)}\n")
    print(f"... shape(shap_values): {shap_values.values.shape}")

    # --- Feature clustering ----
    clustering = shap.utils.hclust(X, y, linkage="complete")
    shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=0.5)
    shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)
    # NOTE: 
    # WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.

    ext = "pdf"
    output_file = f"shap_feature_clustering-{experiment_id}.{ext}" 
    output_path = os.path.join(output_dir, output_file)

    print(f"[output] Saving SHAP feature clustering to:\n{output_path}\n")
    plt.savefig(output_path)

    # Top feature analysis
    n = 3
    topn_features = get_top_features(shap_values, X.columns, N=n)
    print(f"> Top n={n} features:\n{topn_features}\n")

    # Dependence plot
    print(f"> type of input shap_values: {type(shap_values)}")
    ext = "pdf"
    for feature in topn_features: 
        plt.clf()
        shap.plots.scatter(shap_values[:, feature]) # get feature importance scores for that feature
        # this shap_values has to be an Explainer object

        output_file = f"shap_dependence_plot-{feature}-{experiment_id}.{ext}" 
        output_path = os.path.join(output_dir, output_file)

        print(f"[output] Saving dependence plot for {feature} to:\n{output_path}\n")
        plt.savefig(output_path)

    return 

def get_demo_data(**kargs):
    from meta_spliceai.nmd_concept.nmd_targets_analyzer import get_nmd_data
    from sklearn.datasets import load_breast_cancer
    import matplotlib.pyplot as plt

    # Load data
    # X, y = shap.datasets.adult()
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # X, y, meta_data = \
    #     get_nmd_data(policy='topn', threshold_eff=0.3, topn=300, 
    #                     verbose=1, 
    #                     return_x_as_dataframe=True, return_meta_data=True) # return dataframe so that we can get feature names / columns
    # feature_names = list(X.columns)
    # X = X.values
    print(f"... type(X): {type(X)}, type(y): {type(y)}")
    print(f"... shape(X): {X.shape}, shape(y): {y.shape}")
    
    meta_data = {}
    meta_data["classes"] = classes = np.unique(y)
    n_classes = len(classes)
    print(f"... classes (n={n_classes}): {classes}")
    meta_data["feature_names"] = feature_names = list(X.columns)
    # f_encoded = list(set(meta_data['features_encoded']) - set(meta_data['features_original']))
    # print(f"... encoded features (n={len(f_encoded)}):\n{f_encoded}\n")

    return X, y, meta_data

def demo_kernel_explainer(**kargs): 
    import xgboost
    from meta_spliceai.nmd_concept.nmd_targets_analyzer import get_nmd_data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    import matplotlib.pyplot as plt

    experiment_id = "test"
    output_dir = kargs.get("output_dir", "../experiments/test")
    
    # Load data
    X, y = get_demo_data()

    # Train test split
    X = X.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)
    print(f"... type(X_train): {type(X_train)}, type(y_train): {type(y_train)}")
    print(f"... shape(X_train): {X_train.shape}, shape(y_train): {y_train.shape}")

    # Define model 
    model = SGDClassifier(tol=1e-3, eta0=0.001)
    params = {
        'alpha': 0.0001,
        'loss': 'hinge', # 'log_loss',
        'penalty': 'elasticnet',
        'l1_ratio': 0.5,
        'fit_intercept': True, 
        'learning_rate': 'adaptive',
        'average': False, 
        }
    model.set_params(**params)
    model.fit(X_train, y_train)

    if not hasattr(model, "predict_proba"):
        print("> Applying probability calibration ...")
        model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit") # cv default to None which leads to a 5-fold CV  
        model.fit(X_train, y_train)

    highlight("> Testing prediction")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')  #
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"... F1: {f1}, ROCAUC: {roc_auc}")

    highlight("> Starting SHAP analysis ...")
    # X_train_summary = shap.kmeans(X_train, min(50, X_test.shape[0]//2)) 
    # NOTE: this later leads to the following error
    #       Unknown instance type: <class 'shap.utils._legacy.DenseData'>

    # explainer = shap.KernelExplainer(model) # doesn't work this way
    # shap_values = explainer.shap_values(X_test)
    # shap_explanation = explainer(X_test)

    masker = shap.maskers.Independent(X_train, 10)
    print(f"... type(masker.data): {type(masker.data)}")
    explainer = shap.KernelExplainer(model.predict_proba, masker.data, link="logit") # link="logit"
    # NOTE: If the model output is a probability then the LogitLink link function makes 
    #       the feature importance values have log-odds units.

    shap_values = explainer.shap_values(X_test)  # shap_values is a list of numpy arrays
    print(f"[info] type (shap_values): {type(shap_values)}, n={len(shap_values)}")
    # NOTE: model.predict_proba gives probabilities to two classes 
    #       => shap_values, one for each class => list

    # f = lambda x: model.predict_proba(x)[:,1] # take only the positive class
    # explainer = shap.KernelExplainer(f, masker.data, link="logit") # link="logit"
    
    # shap_values = explainer.shap_values(X_test)  # shap_values is a numpy array
    # print(f"[info] type (shap_values): {type(shap_values)}, shape: {shap_values.shape}")

    # Explain a single prediction from the test set
    plt.clf()
    shap_values = explainer.shap_values(X_test[0]) # X_test.iloc[0,:]
    shap.force_plot(explainer.expected_value[0], shap_values[0], X_test[0])  #  X_test.iloc[0,:]

    ext = "svg"
    output_file = f"shap-force_plot-single-{experiment_id}.{ext}" 
    output_path = os.path.join(output_dir, output_file)
    print(f"[output] Saving SHAP force plot (single prediction) to:\n{output_path}\n")
    plt.savefig(output_path)    
    # NOTE: Cannot save force plot like this just yet 
    #       The additive force diagram is in the JS visualization code 
    #       See: https://github.com/shap/shap/issues/27

    # Explain all the predictions in the test set
    plt.clf()
    shap_values = explainer.shap_values(X_test) # l1_reg=f"num_features({n_features})"
    print(f"[info] type (shap_values): {type(shap_values)}, n={len(shap_values)}")
    # NOTE: shap_values is a list of n_classes
    
    shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)

    output_file = f"shap-force_plot-C0-{experiment_id}.{ext}" 
    output_path = os.path.join(output_dir, output_file)
    print(f"[output] Saving SHAP force plot (X_test, class=0) to:\n{output_path}\n")
    plt.savefig(output_path)    


    plt.clf()
    shap.force_plot(explainer.expected_value[1], shap_values[1], X_test)
    # NOTE: explainer = shap.Explainer(f, X_train)
    #       =>  'Permutation' object has no attribute 'expected_value'
    
    output_file = f"shap-force_plot-C1-{experiment_id}.{ext}"  
    output_path = os.path.join(output_dir, output_file)
    print(f"[output] Saving SHAP force plot (X_test, class=1) to:\n{output_path}\n")
    plt.savefig(output_path)


    ### 
    print("[info] Using generic explainer ...")
    # plt.figure(figsize=(12, 12))

    f = lambda x: model.predict_proba(x)[:,1]
    explainer = shap.Explainer(f, X_train, feature_names=feature_names) # link=shap.links.logit => The condensed distance matrix must contain only finite values.
    # NOTE: if passed 'model' => The passed model is not callable and cannot be analyzed directly with the given masker!
    #       if shap does not natively support a given model, then we need to pass the prediction function explicitly
    explanation = explainer(X_test) 
    # NOTE: shap.KernelExplainer's explainer does not take input of this form
    #       If input = masker.datan | X_test 
    #       => 'Kernel' object has no attribute 'masker

    # shap_values2 = explanation.values # explainer.shap_values(X_test)
    # assert shap_values.shape == shap_values2.shape
    # print(f"[info] type(shap_values): {type(shap_values)}")
    # print(f"[info] dim(shap_values): {shap_values.shape}")  # e.g. (120, 156, 2)
    # print(f"[info] dim(shap_values2): {shap_values2.shape}")

    # Beewarm summary plot
    plt.clf()

    _, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(h*4/3, h)
    plt.tight_layout()
    shap.plots.beeswarm(explanation)

    ext = "pdf"
    output_file = f"shap-beewarm-{experiment_id}.{ext}" 
    output_path = os.path.join(output_dir, output_file)
    print(f"[output] Saving SHAP summary plot (beeswarm) to:\n{output_path}\n")
    plt.savefig(output_path, bbox_inches='tight',dpi=100)

    ### 

    plt.clf()
    

    _, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(h*3/2+2, h)
    plt.tight_layout()
    # shap.plots.beeswarm(explanation)
    shap.plots.heatmap(explanation) 

    ext = "pdf"
    output_file = f"shap-heatmap-{experiment_id}.{ext}" 
    output_path = os.path.join(output_dir, output_file)
    print(f"[output] Saving SHAP heatmap to:\n{output_path}\n")
    plt.savefig(output_path, bbox_inches='tight',dpi=100)


    ### 
    plt.clf()
    clustering = shap.utils.hclust(X_test, y_test, linkage="complete")
    shap.plots.bar(explanation, clustering=clustering, clustering_cutoff=0.5)

    _, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(h*4/3, h)
    plt.tight_layout()

    ext = "pdf"
    output_file = f"shap-hcluster-{experiment_id}.{ext}" 
    output_path = os.path.join(output_dir, output_file)
    print(f"[output] Saving SHAP hcluster to:\n{output_path}\n")
    plt.savefig(output_path, bbox_inches='tight',dpi=100)

    return

def demo_tree_explainer(**kargs):
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV
    import matplotlib.pyplot as plt

    experiment_id = "test"
    output_dir = kargs.get("output_dir", "../experiments/test")
    
    # Load data
    X, y, meta_data = get_demo_data()
    feature_names = meta_data['feature_names']

    # Train test split
    X = X.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)
    print(f"... type(X_train): {type(X_train)}, type(y_train): {type(y_train)}")
    print(f"... shape(X_train): {X_train.shape}, shape(y_train): {y_train.shape}")

    model = XGBClassifier(random_state=0, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Explain model's predictions using SHAP values
    f = lambda x: model.predict_proba(x)[:, target_class]
    explainer = shap.TreeExplainer(model, feature_names=feature_names)
    # NOTE: Must pass the model itself, not a function like "f()"

    # shap_values = explainer.shap_values(X_test)
    explanation = explainer(X_test)  # <- returns Explanation object

    highlight(f"[shap] Using TreeExplainer:\n{explainer}\n")
    print(f"... type(explanation)={type(explanation)}")

    if isinstance(explanation, shap.Explanation):
        shap_values = explanation.values
    else: 
        assert isinstance(explanation, np.ndarray)
        shap_values = explanation

    print(f"[info] TreeExplainer: type(shap_values)={type(shap_values)}")
    if isinstance(shap_values, list): 
        print(f"... n(shap_values)={len(shap_values)}")
        print(f"... shape: {shap_values[target_class].shape}")
    else: 
        assert isinstance(shap_values, np.ndarray)
        print(f"... shape: {shap_values.shape}")

    return

def demo(): 

    # Basic SHAP operations
    # demo_shap()

    # Kernel Explainer 
    # demo_kernel_explainer()

    # Tree Explainer 
    demo_tree_explainer()

    return 


if __name__ == "__main__": 
    # test()
    demo()