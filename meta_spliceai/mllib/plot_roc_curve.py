import os
import numpy as np
import pandas as pd
# from scipy import interp
# NOTE: This is depracated

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# from .utils import savefig
from meta_spliceai.mllib.utils import savefig 


def plot_roc_curve_cv(model, X, y, n_folds=10, figsize=None, **kargs): 
    """
    Plot ROC curve as the model gets trained. 

    """
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    
    # Initialize a stratified k-fold object
    # n_folds = 10
    # NOTE: 10-fold CV may not provide sufficient training set size for each CV fold 

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values

    # Set up 10-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=n_folds)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Set up the figure to plot the ROC curves
    ax = None
    if figsize: 
        fig, ax = plt.subplots(figsize=figsize) # will need (10, 10) for 10-fold CV, otherwise lengend will be too big

    # Loop over the folds
    for i, (train, test) in enumerate(cv.split(X, y)):

        # Clone the classifier to make sure the model gets freshly initialized for each fold
        model_clone = clone(model)

        X_train = X.iloc[train] if is_dataframe else X[train]
        X_test = X.iloc[test] if is_dataframe else X[test]
        # No need for special indexing for y as it's either an array or series now
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
            # NOTE: by passing “prefit”, it is assumed that estimator has been fitted already and all data is used for calibration.
            #       - For each fold, we're fitting the model (model_clone) on the training portion 
            #         and just want to calibrate the probabilities on the same data without further splitting.
            probas_ = model_clone.fit(X_train, y_train).predict_proba(X_test)
        else: 
            probas_ = model_clone.fit(X_train, y_train).predict_proba(X_test)

        # Train the classifier on the training data
        # model.fit(X_train, y_train)
        
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
    else: 
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

    # Plot the standard deviation around the mean ROC curve
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    # Finalize the plot
    if ax is None: 
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, 
            label=r'$\pm$ 1 std. dev.')

        # Plot random guessing (diagonal)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guessing', alpha=.8)
        
        # Set plot labels and legend
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
    else: 
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver Operating Characteristic")
        ax.legend(loc="lower right")
    plt.show()


def plot_prc_curve_cv(model, X, y, n_folds=10, figsize=None, **kargs): 
    from sklearn.metrics import precision_recall_curve, auc
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV

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
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit the model and make predictions
        if not hasattr(model_clone, "predict_proba"):
            model_clone.fit(X_train, y_train)
            model_clone = CalibratedClassifierCV(model_clone, method="sigmoid", cv="prefit")  
            # NOTE: by passing “prefit”, it is assumed that estimator has been fitted already and all data is used for calibration.
            #       - For each fold, we're fitting the model (model_clone) on the training portion 
            #         and just want to calibrate the probabilities on the same data without further splitting.
            probas_ = model_clone.fit(X_train, y_train).predict_proba(X_test)
        else: 
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

    # Set plot labels and legend
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")

    # Optional parameters for plot display and persistence
    display = kargs.get("display", True)
    save = kargs.get("save", True)

    if display: 
        plt.show()

    if save: 
        ext = kargs.get("ext", "pdf")

        output_dir_default = os.path.join(os.getcwd(), "experiments")
        output_dir = kargs.get("output_dir", output_dir_default)
        output_file = kargs.get("output_file", f"prc_curve-test.{ext}")
        output_path = os.path.join(output_dir, output_file)

        savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)

    return plt

        