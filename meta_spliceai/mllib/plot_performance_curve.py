import os, sys
# sys.path.append('..')
from pathlib import Path

import numpy as np
import pandas as pd
# from scipy import interp 
# NOTE: This is depracated

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# from .utils import savefig
from meta_spliceai.mllib.utils import savefig  


def plot_roc_curve_cv(model, X, y, n_folds=10, figsize=None, **kargs): 
    """
    Plot ROC curve as the model gets trained. 


    Memo
    ----
    1. Saving a figure after invoking pyplot show results in an empty file
       https://stackoverflow.com/questions/21875356/saving-a-figure-after-invoking-pyplot-show-results-in-an-empty-file

    """
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV

    plt.clf()
    verbose = kargs.get("verbose", 1)

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values

    # Initialize a stratified k-fold object
    # n_folds = 10
    # NOTE: 10-fold CV may not provide sufficient training set size for each CV fold 

    # Set up 10-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=n_folds)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Set up the figure to plot the ROC curves
    ax = fig = None
    if figsize is None:

        # Adjust width more aggressively for higher number of folds to accommodate the legend
        base_width = 10
        width_per_fold = 0.5  # Increment width by this amount for each fold
        width = base_width + n_folds * width_per_fold
        
        # Adjust height to ensure the plot area and legend are not cluttered
        base_height = 8
        height_increment = 0.3  # Increment height by this amount for each fold beyond a threshold
        height_threshold = 10  # Start increasing height only if n_folds is greater than this threshold
        height = base_height + max(0, (n_folds - height_threshold) * height_increment)
        
        fig, ax = plt.subplots(figsize=(width, height))
        print(f"[ROCAUC] figure size: {width}x{height}")

        # NOTE: will need (10, 10) for 10-fold CV, otherwise lengend will be too big
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Loop over the folds
    for i, (train, test) in enumerate(cv.split(X, y)):

        # Clone the classifier to make sure the model gets freshly initialized for each fold
        model_clone = clone(model)

        X_train = X.iloc[train] if is_dataframe else X[train]
        X_test = X.iloc[test] if is_dataframe else X[test]
        y_train = y[train]
        y_test = y[test]

        # Standardize the features
        if kargs.get("standardize", True):
            scaler = StandardScaler().fit(X_train) # MinMaxScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        # Fit the model and make predictions
        if not hasattr(model_clone, "predict_proba"):
            model_clone.fit(X_train, y_train)
            model_clone = CalibratedClassifierCV(model_clone, method="sigmoid", cv="prefit")  
            # NOTE: by passing “prefit”, it is assumed that estimator has been fitted already and all data is used for calibration.
            #       - For each fold, we're fitting the model (model_clone) on the training portion 
            #         and just want to calibrate the probabilities on the same data without further splitting.
            probas_ = model_clone.fit(X_train, y_train).predict_proba(X_test)
        else: 
            if i == 0 and verbose: 
                print(f"[info] model has predict_proba method")

            probas_ = model_clone.fit(X_train, y_train).predict_proba(X_test)
        
        # Compute the ROC curve points
        fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
        
        # Compute the area under the ROC curve
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Plot the ROC curve for this fold
        if ax is None: 
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        else: 
            ax.plot(fpr, tpr, alpha=0.3,
                    label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
        # Interpolate the ROC curve
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # Plot the mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if ax is None: 
        plt.plot(mean_fpr, mean_tpr, color='b', 
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), 
            lw=2, alpha=.8)
        
        # We can include the key message directly in the legend by adding a specific label to one of your plot elements

        # mean_auc_label = f"SpliceMediator performs on an average AUC of {mean_auc:.2f} $\pm$ {std_auc:.2f}"
        # plt.plot(mean_fpr, mean_tpr, color='b', label=mean_auc_label, lw=2, alpha=.8)
    else: 
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

    # Plot the standard deviation around the mean ROC curve
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # Plot parameters 
    title_text = kargs.pop("title", "Performance in ROCAUC")  # 'Receiver Operating Characteristic' 
    
    # Finalize the plot
    if ax is None: 
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, 
            label=r'$\pm$ 1 std. dev.')

        # Plot random guessing (diagonal)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guessing', alpha=.8)
        
        # Set plot labels and legend
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title(title_text, fontsize=16)
        plt.legend(loc="lower right")
    else: 
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title_text)
        ax.legend(loc="lower right")

    # Add a more detailed description using figtext
    fig_text = kargs.pop("fig_text", "ROCAUC in Cross-Validation Setting")
    
    if fig_text is not None: 
        # Use the provided fig_text string to dynamically create the figure text
        kargs.update({'mean_auc': mean_auc, 'std_auc': std_auc})
        fig_text = fig_text.format(**kargs) # mean_auc=mean_auc, std_auc=std_auc

        if fig is not None:
            fig.text(0.5, 0.03, fig_text, ha="center", fontsize=14)
        else:
            plt.figtext(0.5, 0.03, fig_text, ha="center", fontsize=14)

    # Optional parameters for plot display and persistence
    display = kargs.get("display", True)
    save = kargs.get("save", True)
    verbose = kargs.get("verbose", 1)

    if save: 
        create_if_not_exist = kargs.get("create_output_dir", True)
        output_dir_default = os.path.join(os.getcwd(), "plot")
        output_dir = kargs.get("output_dir", output_dir_default)
        if create_if_not_exist: 
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        ext = kargs.get("ext", "pdf")
        output_file = kargs.get("output_file", f"roc_curve-test.{ext}")
        output_path = os.path.join(output_dir, output_file)
        
        if verbose: 
            print(f"[evaluation] Saving ROC curve to:\n{output_path}\n")
        savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)
        # fig.savefig(output_path)

    if display: 
        plt.show()
    else: 
        plt.close()

    return


def plot_prc_curve_cv(model, X, y, n_folds=10, figsize=None, **kargs): 
    from sklearn.metrics import precision_recall_curve, auc
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV

    verbose = kargs.get("verbose", 1)

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values

    # Initialize arrays to store precision, recall and average precision for each fold
    precisions = []
    recalls = []
    average_precisions = []
    mean_recall = np.linspace(0, 1, 100)

    # Set up the figure to plot the ROC curves
    ax = fig = None
    if figsize is None:

        # Adjust width more aggressively for higher number of folds to accommodate the legend
        base_width = 10
        width_per_fold = 0.5  # Increment width by this amount for each fold
        width = base_width + n_folds * width_per_fold
        
        # Adjust height to ensure the plot area and legend are not cluttered
        base_height = 8
        height_increment = 0.3  # Increment height by this amount for each fold beyond a threshold
        height_threshold = 10  # Start increasing height only if n_folds is greater than this threshold
        height = base_height + max(0, (n_folds - height_threshold) * height_increment)
        
        fig, ax = plt.subplots(figsize=(width, height))
        print(f"[PRCAUC] figure size: {width}x{height}")

        # NOTE: will need (10, 10) for 10-fold CV, otherwise lengend will be too big
    else:
        fig, ax = plt.subplots(figsize=figsize)

    cv = StratifiedKFold(n_splits=n_folds)
    plt.clf()

    # Perform 5-fold cross validation
    for i, (train, test) in enumerate(cv.split(X, y)):
        # Clone the classifier to make sure the model gets freshly initialized for each fold
        model_clone = clone(model)

        X_train = X.iloc[train] if is_dataframe else X[train]
        X_test = X.iloc[test] if is_dataframe else X[test]
        y_train = y[train]
        y_test = y[test]

        # Standardize the features
        if kargs.get("standardize", True):
            scaler = StandardScaler().fit(X_train)   # MinMaxScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        # Fit the model and make predictions
        if not hasattr(model_clone, "predict_proba"):
            model_clone.fit(X_train, y_train)
            model_clone = CalibratedClassifierCV(model_clone, method="sigmoid", cv="prefit")  
            # NOTE: by passing “prefit”, it is assumed that estimator has been fitted already and all data is used for calibration.
            #       - For each fold, we're fitting the model (model_clone) on the training portion 
            #         and just want to calibrate the probabilities on the same data without further splitting.
            probas_ = model_clone.fit(X_train, y_train).predict_proba(X_test)
        else: 
            if i == 0 and verbose: 
                print(f"[info] model has predict_proba method") 
            probas_ = model_clone.fit(X_train, y_train).predict_proba(X_test)
        
        # Compute precision-recall curve and area under the curve
        precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
        average_precision = auc(recall, precision)
        average_precisions.append(average_precision)
        
        # Plot precision-recall curve for this fold
        plt.plot(recall, precision, lw=1, alpha=0.3, label='PR fold %d (AUC = %0.2f)' % (i, average_precision))

    # Compute and plot mean precision-recall curve
    mean_precision = np.mean(precisions, axis=0)
    mean_precision[-1] = 0.0
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(average_precisions)
    plt.plot(mean_recall, mean_precision, color='b', label=r'Mean PR (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    # Compute and plot standard deviation around mean precision-recall curve
    std_precision = np.std(precisions, axis=0)
    precisions_upper = np.minimum(mean_precision + std_precision, 1)
    precisions_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(mean_recall, precisions_lower, precisions_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    ################################

    # Plot parameters 
    title_text = kargs.pop("title", 'Precision-Recall Curve')

    # Set plot labels and legend
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title_text, fontsize=16)
    plt.legend(loc="lower right")

    # Add a more detailed description using figtext
    fig_text = kargs.pop("fig_text", "PRCAUC in Cross-Validation Setting")
    
    if fig_text is not None: 
        # Use the provided fig_text string to dynamically create the figure text
        kargs.update({'mean_auc': mean_auc, 'std_auc': std_auc})  # PRCAUC
        fig_text = fig_text.format(**kargs) # mean_auc=mean_auc, std_auc=std_auc

        if fig is not None:
            fig.text(0.5, 0.03, fig_text, ha="center", fontsize=14)
        else:
            plt.figtext(0.5, 0.03, fig_text, ha="center", fontsize=14)

    # Optional parameters for plot display and persistence
    display = kargs.get("display", True)
    save = kargs.get("save", True)
    verbose = kargs.get("verbose", 1)

    if save: 
        create_if_not_exist = kargs.get("create_output_dir", True)
        output_dir_default = os.path.join(os.getcwd(), "plot")
        output_dir = kargs.get("output_dir", output_dir_default)
        if create_if_not_exist: 
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        ext = kargs.get("ext", "pdf")
        output_file = kargs.get("output_file", f"prc_curve-test.{ext}")

        output_path = os.path.join(output_dir, output_file)
        if verbose: 
            print(f"[evaluation] Saving PR curve to:\n{output_path}\n")
        savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)

    if display: 
        plt.show()
    else: 
        plt.close()

    return plt

def plot_roc_curve_cv_multiclass(model, X, y, n_folds=10, figsize=None, class_names=None, **kargs):
    """
    Plot ROC curve for each class in a multi-class classification problem using one-vs-all approach.
    """
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    import matplotlib.pyplot as plt

    plt.clf()

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values

    # Initialize a stratified k-fold object
    cv = StratifiedKFold(n_splits=n_folds)

    # Compute ROC curve for each class
    mean_fpr = np.linspace(0, 1, 100)

    for class_label in np.unique(y):
        fig, ax = plt.subplots(figsize=figsize)

        class_name = class_names[class_label] if class_names and class_label in class_names else str(class_label)

        for i, (train, test) in tqdm(enumerate(cv.split(X, y))):
            model_clone = clone(model)

            X_train = X.iloc[train] if is_dataframe else X[train]
            X_test = X.iloc[test] if is_dataframe else X[test]
            y_train = y[train]
            y_test = y[test]

            # Standardize the features
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # Fit the model and make predictions
            if not hasattr(model_clone, "predict_proba"):
                model_clone.fit(X_train, y_train)
                model_clone = CalibratedClassifierCV(model_clone, method="sigmoid", cv="prefit")
                probas_ = model_clone.fit(X_train, y_train).predict_proba(X_test)
            else:
                probas_ = model_clone.fit(X_train, y_train).predict_proba(X_test)

            # Compute ROC curve and ROC area for the current class
            fpr, tpr, _ = roc_curve(y_test == class_label, probas_[:, class_label])
            roc_auc = auc(fpr, tpr)

            # class_name = class_names[class_label] if class_names and class_label in class_names else str(class_label)
            ax.plot(fpr, tpr, lw=1, alpha=0.3,
                    label='ROC fold %d class %s (AUC = %0.2f)' % (i, class_name, roc_auc))

        # Plot random guessing (diagonal)
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guessing', alpha=.8)

        # Plot parameters
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f"ROC Curve for Class {class_name}", xlabel="False Positive Rate",
               ylabel="True Positive Rate")
        ax.legend(loc="lower right")

        # Show or save the plot
        if kargs.get("save", True):
            create_if_not_exist = kargs.get("create_output_dir", True)
            output_dir_default = os.path.join(os.getcwd(), "plot")
            output_dir = kargs.get("output_dir", output_dir_default)
            if create_if_not_exist: Path(output_dir).mkdir(parents=True, exist_ok=True)

            # ext = kargs.get("ext", "pdf")
            output_file = kargs.get("output_file", f"roc_curve-class_{class_name}.pdf")
            output_file_stem, ext = os.path.splitext(output_file)  # output_file = "roc_curve", ext = ".pdf"
            ext = ext[1:]

            if not class_name in output_file_stem: 
                # output_file, ext = os.path.splitext(output_file)  # output_file = "roc_curve", ext = ".pdf"
                output_file = f"{output_file_stem}-{class_name}.{ext}"

            output_path = os.path.join(output_dir, output_file)

            plt.savefig(output_path, format=ext, dpi=100)
            print(f"[evaluation] Saving ROC curve for class {class_name} to:\n{output_path}\n")
        else: 
            plt.show()

        plt.close(fig)

    return


def plot_normalized_confusion_matrix_v0(model, X, y, class_labels=None, **kargs):
    """
    Plots a normalized confusion matrix for a trained model on given data.
    
    Parameters:
    - model: Trained classifier.
    - X: Feature matrix.
    - y: True labels.
    - class_labels: List of class labels. If None, labels will be inferred from the data.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Predict the labels
    y_pred = model.predict(X)
    
    # Compute the confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # If class_labels are not provided, infer them from the data
    if class_labels is None:
        class_labels = np.unique(np.concatenate((y, y_pred)))

    title = kargs.get("title", 'Normalized Confusion Matrix')
    
    # Plot the normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
   
    # Show the plot
    if not 'output_path' in kargs: 
        plt.show()
    else: 
        ext = kargs.get("format", "pdf")
        output_path = kargs.get("output_path", f"confusion_matrix.{ext}") # os.path.join(output_dir, output_file)

        dpi = kargs.get("dpi", 120)
        print(f"[output] Saving density comparison plot ({title}) to:\n{output_path}\n")
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)

    return

def plot_normalized_confusion_matrix(model, X, y, label_dict=None, **kargs):
    """
    Plots a normalized confusion matrix for a trained model on given data.
    
    Parameters:
    - model: Trained classifier.
    - X: Feature matrix.
    - y: True labels.
    - label_dict: Dictionary mapping integer labels to class names. If None, integer labels will be used.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Predict the labels
    y_pred = model.predict(X)
    
    # Compute the confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # If label_dict is provided, use it to map integer labels to class names
    if label_dict is not None:
        class_labels = [label_dict[label] for label in np.unique(np.concatenate((y, y_pred)))]
        # NOTE: E.g. label_dict = {0: 'Class A', 1: 'Class B', 2: 'Class C'}
    else:
        class_labels = np.unique(np.concatenate((y, y_pred)))

    title = kargs.get("title", 'Normalized Confusion Matrix')
    
    # Plot the normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)

    # Show the plot
    if not 'output_path' in kargs: 
        plt.show()
    else: 
        ext = kargs.get("format", "pdf")
        output_path = kargs.get("output_path", f"confusion_matrix.{ext}") # os.path.join(output_dir, output_file)

        dpi = kargs.get("dpi", 120)
        print(f"[output] Saving density comparison plot ({title}) to:\n{output_path}\n")
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)

    return

##############################################


def demo_performance_plots(): 
    import os
    import pandas as pd
    import xgboost as xgb
    from sklearn.datasets import make_classification

    # Create a synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, n_classes=2, random_state=42)

    # Convert to DataFrame for convenience
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y

    # Display the first few rows of the dataset
    print(df.head())

    model = xgb.XGBClassifier(random_state=0, eval_metric='logloss') # aucpr, logloss
    # plot_roc_curve_cv(model, X, y, n_folds=5, figsize=None)

    # Load my own datasets 
    from meta_spliceai.sphere_pipeline.data_model import SequenceDescriptor
    import meta_spliceai.sphere_pipeline.utils_data as ud

    col_label = 'label'
    labeling_concept = "nmd_eff-t0.2"
    target_selection_policy = 'topn'
    input_dir = f"/mnt/nfs1/splice-mediator/synapse/hg38.p14.2bit/descriptor/{labeling_concept}"
    # input_file = "tx_ex-combined-test-featurized.topn.csv"
    input_file = "tx_ex-combined-test-seq-featurized.topn.csv"
    input_path = os.path.join(input_dir, input_file)

    df = pd.read_csv(input_path, sep='\t', header=0)
    print(f"> concept={labeling_concept} => shape(df): {df.shape}")
    print(f"> label counts:\n{df[col_label].value_counts()}\n")
    df = df.sample(frac=1.0)

    descriptor = SequenceDescriptor(concept=labeling_concept, 
                biotype='combined',
                suffix='test',
                    stype='sequence', 
                    # sequence_content_type=sequence_content_type, 
                    # tissue_type=tissue_type
                )

    df2, data_path = descriptor.load(suffix=target_selection_policy, return_data_path=True)
    print(f"> data path:\n{data_path}\n")
    # print(f"> concept={labeling_concept}, policy={target_selection_policy} => shape(df): {df2.shape}")

    print("#" * 90); print()
    
    non_feature_cols = SequenceDescriptor.get_non_feature_columns(df)
    X, y = SequenceDescriptor.to_xy(df, dummify=True, tid_as_index=True)
    # y = df[col_label]
    # X = df.drop(columns=non_feature_cols)
    # X = ud.get_dummies_and_verify(X, drop_first=True)
    print(f"> After dummying variables, shape(X)={X.shape}")

    print("[plot] Plotting performance curves ...")
    plot_roc_curve_cv(model, X, y, n_folds=5, figsize=None, output_file=f"roc_{labeling_concept}-test2.pdf")
    plot_prc_curve_cv(model, X, y, n_folds=5, figsize=None, output_file=f"prc_{labeling_concept}-test2.pdf")

    return

def test(): 
    demo_performance_plots()

    return

if __name__ == "__main__": 
    test() 

        