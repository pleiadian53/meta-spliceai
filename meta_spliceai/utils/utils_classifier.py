import os
import numpy as np
import pandas as pd
# import common

import matplotlib.pyplot as plt
# plt.style.use('ggplot')
plt.style.use('seaborn')

import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Some basic classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # AdaBoostClassifier, StackingClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score

"""

Reference 
---------
1. sklearn-crfsuite
   pip install sklearn-crfsuite
   https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
"""

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian generative classification based on KDE.
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        """
        predict_proba() returns an array of class probabilities of shape [n_samples, n_classes]

        Entry [i, j] of this array is the posterior probability that sample i is a member of class j, 
        computed by multiplying the likelihood by the class prior and normalizing.
        """
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


class CESClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self): 
        pass 

# Utility function to report best scores
def report(results, n_top=3):
    """
    Params
    ------
    results: am output dictionary from a GridSearchCV or RandomizedSearchCV
             as in grid_search.cv_results_ (which is a dictionary)

    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def is_label_prediction(y_pred, n_classes=2):
    if str(np.array(y_pred).dtype).startswith('int'): 
        return True
    values = np.unique(y_pred)
    if len(values) <= n_classes: 
        return True
    return False

def generate_imbalanced_data(class_ratio=0.95, verbose=1):
    # from sklearn import datasets
    # import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from collections import Counter
    # import utils_classifier as uclf

    # get the dataset
    c_ratio = class_ratio

    def get_dataset(n_samples=5000, noise=True):
        if noise: 
            X,y = make_classification(n_samples=n_samples, n_features=100, n_informative=30, 
                            n_redundant=6, n_repeated=3, n_classes=2, n_clusters_per_class=1,
                                class_sep=2,
                                flip_y=0.2, # <<< 
                                weights=[c_ratio, ], random_state=17)
        else: 
            X,y = make_classification(n_samples=n_samples, n_features=100, n_informative=30, 
                                n_redundant=6, n_repeated=3, n_classes=2, n_clusters_per_class=1,
                                    class_sep=2, 
                                    flip_y=0, weights=[c_ratio, ], random_state=17)
        return X, y

    X, y =  get_dataset(noise=True)

    uniq_labels = np.unique(y)
    n_classes = len(uniq_labels)

    # Turn into a binary classification problem 
    if n_classes > 2: 
        y0 = y
        y, y_map, le = to_binary_classification(y, target_class=2)
        
        if verbose > 1: 
            print('> y before:\n', y0)
            print('> y after:\n', y)

    print(f'> n_classes: {n_classes}\n{uniq_labels}\n')

    counter = Counter(y)
    if verbose: print(f'> counts:\n{counter}\n') 

    # Plot data
    # f, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(20,8))
    # sns.scatterplot(X,hue=y,ax=ax1);
    # ax1.set_title("With Noise");
    # plt.show();
    
    return X, y

def choose_classifier(name, **kargs): 
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import SGDClassifier, LogisticRegression, Lasso, LassoCV, Perceptron
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.svm import SVC
    from pprint import pprint

    verbose = kargs.get("verbose", 0)
    params = kargs.get("params", None)

    if verbose > 1: print('> Classifier name: %s' % name)

    name = name.lower() # standardize name to lower case
    model = None
    if name.startswith('percep'):  # perceptron
        model = SGDClassifier(loss = 'perceptron', n_iter = 50, random_state = 0)   # loss = 'log' => logistic regression
        # model = Perceptron(tol=1e-3, random_state=0, penalty='l2') # does not have predict_probs()
    elif name.startswith('log'):
        model = LogisticRegression(penalty='l2', tol=1e-4, warm_start=False, solver='sag') # max_iter=int(1e4), class_weight='balanced'
    elif name == 'enet':
        if params is None:
            params = {'penalty': 'elasticnet', 'alpha': 0.01, 'loss': 'modified_huber', 
                     'fit_intercept': True, 'random_state': 0, 'tol': 1e-4}   # 'tol': 1e-3, 
        model = SGDClassifier(**params)
        
    elif name == 'lasso':
        # max_iter is useful only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations taken for the solvers to converge.
        model = LogisticRegression(C=0.01, penalty='l1', solver='saga',
                              tol=1e-4, 
                              warm_start=False) # set warm_start to reuse learned coeffs
        
        # model = Lasso(alpha=0.1) # NOTE: this is a regression model

    elif name.startswith(('qda', 'quad')):  # QDA 
        # default: priors=None, reg_param=0.0, store_covariance=False, tol=0.0001
        model = QuadraticDiscriminantAnalysis() 
    
    elif name.startswith(('svm', 'svc', )):  # SVM with linear kernel 
        # default: C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True, probability=False, tol=0.001,
        model = svc = SVC(probability=True, kernel='linear', class_weight='balanced') 

    elif name.startswith(('rf', 'randomf','random_f')):
        model = RandomForestClassifier(n_estimators = 100, 
                    max_features='sqrt', 
                        max_depth = None, bootstrap = True, random_state = 0)

    elif name.startswith(('nai', ) ):   # naive bayes
        model = GaussianNB()          # default: priors=None, var_smoothing=1e-09

    elif name.startswith('ada'): # AdaBoost with base_estimator as decision tree 
        model = AdaBoostClassifier(n_estimators=kargs.get('n_estimators', 100), random_state=0) 

        ### using other base estimator 
        # svc = SVC(probability=True, kernel='linear')
        # model = AdaBoostClassifier(base_estimator=svc, n_estimators=100, random_state=0)
    elif name.startswith( ('tree', 'deci') ):  # decision tree classifier
        # default: criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None
        model = DecisionTreeClassifier(class_weight='balanced', criterion = "gini", random_state = 100,
                       max_depth=4, min_samples_leaf=5)

    elif name.startswith(('grad', 'gb')):  # gradient boost tree 
        model = GradientBoostingClassifier(n_estimators=kargs.get('n_estimators', 50), 
                    learning_rate=0.1, 
                    # random_state=53, 
                    min_samples_split=100, min_samples_leaf=5, max_depth=8,  # prevent overfitting
                    max_features = 'sqrt', # Its a general thumb-rule to start with square root.
                    subsample=0.7) # fraction of samples to be used for fitting the individual base learners
    elif name.startswith(('knn', 'neighbor')): 
        model =  KNeighborsClassifier(n_neighbors=5)

    if isinstance(params, dict) and len(params) > 0: 
        model.set_params(**params)

    if verbose: 
        final_params = model.get_params()
        print(f"[choose_classifier] model parmaters:\n{final_params}\n")
        # assert final_params['class_weight'] == 'balanced'  # ... ok
    # If kargs has other parameters other than 'verbose'
    # if 'verbose' in kargs: kargs.pop('verbose')
    # if len(kargs) > 0: 
    #     model.set_params(**kargs)

    # [test]
    assert callable(getattr(model, 'fit', None)), "Unsupported classifier: {name}".format(name=name)
    # a valid classifier should provide the following operations
    #     model = model.fit(train_df, train_labels)
    #     test_predictions = model.predict_proba(test_df)[:, 1]  # 1/foldCount worth of data

    return model

###############################################################################################
# Probability Thresholding Utilities
###############################################################################################

def auc_threshold(labels, predictions, verbose=0, pos_label=1): 
    """
    Identify that threshold that gives us the upper-left corner of the curve. 
    Mathematically speaking, we want to find

    p* = argmin< p > | tpr(p) + fpr(p) - 1 |

    Memo
    ----
    1. roc_curve(): 
       - drop_intermediate
          Whether to drop some suboptimal thresholds which would not appear on a plotted ROC curve. 
          This is useful in order to create lighter ROC curves.
    """
    # from sklearn.metrics import roc_curve, plot_roc_curve 

    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, drop_intermediate=False, pos_label=pos_label)

    if verbose: 
        # Plot the objective function with respect to the threshold and see where its minimum is
        plt.scatter(thresholds, np.abs(fpr+tpr-1))
        plt.xlabel("Threshold")
        plt.ylabel("|FPR + TPR - 1|")
        plt.show()

    return thresholds[np.argmin(np.abs(fpr+tpr-1))]

def auc_score_threshold(labels, predictions, verbose=0, pos_label=1):
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, drop_intermediate=False, pos_label=pos_label)
    if verbose: 
        # Plot the objective function with respect to the threshold and see where its minimum is
        plt.scatter(thresholds, np.abs(fpr+tpr-1))
        plt.xlabel("Threshold")
        plt.ylabel("|FPR + TPR - 1|")
        plt.show()

    score = metrics.auc(fpr, tpr)
    threshold = thresholds[np.argmin(np.abs(fpr+tpr-1))]

    return (score, threshold)

def acc_max_threshold(labels, predictions, verbose=0, pos_label=1):
    # from sklearn.metrics import balanced_accuracy_score

    thresholds = []
    accuracies = []

    for p in np.unique(predictions):
        thresholds.append(p)
        y_pred = (predictions >= p).astype(int)
        accuracies.append(metrics.balanced_accuracy_score(labels, y_pred))

    if verbose: 
        plt.scatter(thresholds, accuracies)
        plt.xlabel("Threshold")
        plt.ylabel("Balanced accuracy")
        plt.show()

    return thresholds[np.argmax(accuracies)]

def acc_max_score_threshold(labels, predictions, verbose=0, pos_label=1): 
    thresholds = []
    accuracies = []

    for p in np.unique(predictions):
        thresholds.append(p)
        y_pred = (predictions >= p).astype(int)
        accuracies.append(metrics.balanced_accuracy_score(labels, y_pred))

    if verbose: 
        plt.scatter(thresholds, accuracies)
        plt.xlabel("Threshold")
        plt.ylabel("Balanced accuracy")
        plt.show()

    imax = np.argmax(accuracies)
    # score = np.max(accuracies)
    # threshold = thresholds[np.argmax(accuracies)]
    return accuracies[imax], thresholds[imax]


def fmax_score(labels, predictions, beta = 1.0, pos_label = 1):
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.

    Memo
    ---- 
    1. precision and recall tradeoff doesn't take into account true negative

    """
    precision, recall, threshold = metrics.precision_recall_curve(labels, predictions, pos_label=pos_label)

    # the general formula for positive beta
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    # if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    # i = np.nanargmax(f1)

    # return (f1[i], threshold[i])
    return np.nanmax(f1)

def fmax_threshold(labels, predictions, beta = 1.0, pos_label = 1): 
    precision, recall, threshold = metrics.precision_recall_curve(labels, predictions, pos_label=pos_label)

    # the general formula for positive beta
    # ... if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    i = np.nanargmax(f1)  # the position for which f1 is the max 
    th = threshold[i] if i < len(threshold) else 1.0    # len(threshold) == len(precision) -1 
    # assert f1[i] == np.nanmax(f1)
    return th

def fmax_score_threshold(labels, predictions, beta = 1.0, pos_label = 1):
    """

    Memo
    ---- 
    1. precision and recall tradeoff doesn't take into account true negative
       
       precision: Precision values such that element i is the precision of predictions with score >= thresholds[i] and the last element is 1. 
       recall: Decreasing recall values such that element i is the recall of predictions with score >= thresholds[i] and the last element is 0.

    2. example 

    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> precision, recall, thresholds = precision_recall_curve(
    ...     y_true, y_scores)
    >>> precision  
    array([0.66666667, 0.5       , 1.        , 1.        ])
    >>> recall
    array([1. , 0.5, 0.5, 0. ])
    >>> thresholds
    array([0.35, 0.4 , 0.8 ])

    precision[1] = 0.5, for any prediction >= thresholds[1] = 0.4 as positive (assuming that pos_label = 1)

    """
    # from sklearn.metrics import precision_recall_curve
    precision, recall, threshold = metrics.precision_recall_curve(labels, predictions, pos_label=pos_label)

    # the general formula for positive beta
    # ... if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    i = np.nanargmax(f1)  # the position for which f1 is the max 
    th = threshold[i] if i < len(threshold) else 1.0    # len(threshold) == len(precision) -1 
    # assert f1[i] == np.nanmax(f1)
    return (f1[i], th)

def fmax_precision_recall_scores(labels, predictions, beta = 1.0, pos_label = 1):
    ret = {}
    precision, recall, threshold = metrics.precision_recall_curve(labels, predictions, pos_label=pos_label)

    # the general formula for positive beta
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    i = np.nanargmax(f1)
    th = threshold[i] if i < len(threshold) else 1.0    # len(threshold) == len(precision) -1 

    ret['id'] = i 
    ret['threshold'] = th
    ret['precision'] = precision[i] # element i is the precision of predictions with score >= thresholds[i]
    ret['recall'] = recall[i]
    ret['f'] = ret['fmax'] = f1[i]
    
    return ret  # key:  id, precision, recall, f/fmax 

###############################################################################################
# Model selection utilities 
###############################################################################################

def hyperparameter_template(model='rf'):
    
    model_name = model.lower()
    if model_name.startswith( ('rf', 'randomfor', 'randf') ): 
        n_estimators = [25, 50, 75, 100, 120,  ] # [int(x) for x in np.linspace(start = 100, stop = 700, num = 50)]
        max_features = ["sqrt", ] # ['auto', 'log2']  # Number of features to consider at every split
        
        max_depth = [4, 8] # [8, 10] # [int(x) for x in np.linspace(10, 110, num = 11)]   # Maximum number of levels in tree
        max_depth.append(None)

        min_samples_split = [4,  ]  # Minimum number of samples required to split a node
        min_samples_leaf = [1, ]    # Minimum number of samples required at each leaf node
        max_leaf_nodes = [None,  ] # + [10, 25, 50] # [None] + list(np.linspace(10, 50, 500).astype(int)), # 10 to 50 "inclusive"
        bootstrap = [True, ]       # Method of selecting samples for training each tree
        # ... NOTE: Out of bag estimation only available if bootstrap=True

        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'max_leaf_nodes': max_leaf_nodes, 
                       'bootstrap': bootstrap}
    elif model_name.startswith(('logis', 'logit', )): 
        solvers = ['lbfgs', ] # ['newton-cg', 'lbfgs', 'liblinear'] 
        # Note: newton-cg and lbfgs solvers support only l2 penalties.

        penalty = ['l2', ] # 'l1'
        c_values = np.logspace(-3, 2, 6) # [100, 10, 1.0, 0.1, 0.01]
        random_grid = dict(solver=solvers, penalty=penalty, C=c_values)
    elif model_name == "lasso": 

        penalty = ['l1', ] 
        solvers = ['saga', ]
        c_values = [0.01, 0.1, 10, 100] # np.logspace(-3, 2, 6) 
        random_grid = dict(solver=solvers, penalty=penalty, C=c_values)
    else:  
        raise NotImplementedError(f"{model.capitalize()} not supported. Coming soon :)")
    return random_grid

def tune_model(model, grid, cv=None, **kargs): 
    """

    Parameters 
    ---------- 
    model: a classifier to tune for its best hyperparameter settings
    grid: a parameter dictionary
    """ 
    scoring = kargs.get('scoring', 'f1') # Evaluation metric
    verbose = kargs.get('verbose', 1)

    def fit_on_data(X, y): 
        nonlocal cv 
        if cv is None: 
            random_state = kargs.get('random_state', 53)
            n_splits = kargs.get('n_splits', 5)
            n_repeats = kargs.get('n_repeats', 2)
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring, error_score=0, verbose=verbose)
        model_tuned = grid_result = grid_search.fit(X, y)
        
        # summarize results
        if verbose: 
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
        
        return model_tuned # to make predictions: call .predict(X_test) or grid_result.best_estimator_.predict(X_test)
    return fit_on_data

def estimate_oob_error(data=None, **kargs):
    import matplotlib.pyplot as plt

    from collections import OrderedDict
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from utils import savefig

    RANDOM_STATE = 123

    # Generate a binary classification dataset.
    if data is not None: 
        X, y = data
    else: 
        X, y = make_classification(
            n_samples=500,
            n_features=25,
            n_clusters_per_class=1,
            n_informative=15,
            random_state=RANDOM_STATE,
        )

    # NOTE: Setting the `warm_start` construction parameter to `True` disables
    # support for parallelized ensembles but is necessary for tracking the OOB
    # error trajectory during training.
    ensemble_clfs = [
        (
            "RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE,
            ),
        ),
        # (
        #     "RandomForestClassifier, max_features='log2'",
        #     RandomForestClassifier(
        #         warm_start=True,
        #         max_features="log2",
        #         oob_score=True,
        #         random_state=RANDOM_STATE,
        #     ),
        # ),
        (
            "RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 120 # 15
    max_estimators = 150

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1, 5):
            clf.set_params(n_estimators=i)
            clf.fit(X, y)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_ # <<< OOB score
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")

    # Todo: Configuration
    ext = 'tif'
    data_dir = os.path.join(os.getcwd(), "data")
    output_dir = kargs.get("output_dir", data_dir)
    output_file = kargs.get("output_file", f"performance_evaluation.{ext}")
    # os.path.join(os.getcwd(), "result_expr_significance")
    output_path = kargs.get("output_path", os.path.join(output_dir, output_file))
    savefig(plt, output_path, ext=ext, dpi=300, message='', verbose=True)

    return 

###############################################################################################
# Performance evaluation utilities
# 
# Related Modules
# ---------------
# 
# 
###############################################################################################

# Todo
def evaluate_model(train, test):
    """

    Memo
    ----
    1. Derived from this notebook
       - https://drive.google.com/file/d/1eLBU03jW61rGoUezAw-vromrS5Uj1Uwy/view?usp=share_link

    """
    y_train, train_predictions, train_probs = train # order: true labels, predicted labels, probability scores
    y_test, y_pred, probs = test

    baseline = {}
    baseline['recall'] = recall_score(y_test, [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test, [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5

    results = {}
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['roc'] = roc_auc_score(y_test, probs)

    train_results = {}
    train_results['recall'] = recall_score(y_train,       train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)

    for metric in ['recall', 'precision', 'roc']:  
          print(f'{metric.capitalize()} \
                 Baseline: {round(baseline[metric], 2)} \
                 Test: {round(results[metric], 2)} \
                 Train: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();

###############################################################################################
# Demo Functions 
###############################################################################################

# Dfine grid search
def demo_logistic_regression_tuning(): 
    """

    Reference 
    ---------
    1. How to use GridSearchCV ouptut
       https://stackoverflow.com/questions/35388647/how-to-use-gridsearchcv-output-for-a-scikit-prediction
    """

    # example of grid searching key hyperparametres for logistic regression
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    
    # define dataset
    X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
    
    # define models and parameters
    #################################
    model = LogisticRegression()
    solvers = ['lbfgs', ] # ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2', 'l1', ]
    c_values = [100, 10, 1.0, 0.1, 0.01]
    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    #################################

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result

def demo_knn_tuning(): 
    # example of grid searching key hyperparametres for KNeighborsClassifier
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    
    # define dataset
    X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)

    # define models and parameters
    model = KNeighborsClassifier()

    # define grid search
    #################################
    n_neighbors = range(1, 21, 2)
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
    #################################
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X, y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result

def demo_gradient_boosting(): 
    # example of grid searching key hyperparameters for GradientBoostingClassifier
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier
    
    # define dataset
    X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
    
    # define models and parameters
    model = GradientBoostingClassifier()

    # define grid search
    #################################
    n_estimators = [10, 100, 1000]
    learning_rate = [0.001, 0.01, 0.1]
    subsample = [0.5, 0.7, 1.0]
    max_depth = [3, 7, 9]
    grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
    #################################
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result

def demo_boosting(): 
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import time
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    print('make classification ...')
    X,y = make_classification(n_samples=1000000,
                            n_features=50,
                            n_informative=30,
                            n_redundant=5,
                            n_repeated=0,
                            n_classes=2,
                            n_clusters_per_class=2,
                            class_sep=1,
                            flip_y=0.01,
                            weights=[0.5,0.5],
                            random_state=17)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1000)
    print(f'X_train shape: {X_train.shape}')
    print(f'Train LGBM classifier ...')
    clf = LGBMClassifier(n_estimators=100,
                        num_leaves=64,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=1000,
                        n_jobs=-1)
    start = time.time()
    clf.fit(X_train,y_train)
    elapsed = time.time() - start
    print(f'LGBM Training ran in {elapsed:.5f} seconds')
    y_pred = clf.predict(X_test)
    print(f'Test Accuracy: {accuracy_score(y_test,y_pred):.2f}')
    print(f'Train XGB classifier ...')
    clf = XGBClassifier(n_estimators=100,
                        max_depth=5,
                        max_leaves=64,
                        eta=0.1,
                        reg_lambda=0,
                        tree_method='hist',
                        eval_metric='logloss',
                        use_label_encoder=False,
                        random_state=1000,
                        n_jobs=-1)
    start = time.time()
    clf.fit(X_train,y_train)
    elapsed = time.time() - start
    print(f'XGB Training ran in {elapsed:.5f} seconds')
    y_pred = clf.predict(X_test)
    print(f'Test Accuracy: {accuracy_score(y_test,y_pred):.2f}')

    return 

def test(): 

    # Gradient boosting 
    demo_boosting()

    return

if __name__ == "__main__": 
    test()