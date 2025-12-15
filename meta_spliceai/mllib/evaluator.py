
import os
import numpy as np

# from .utils import (highlight, savefig)
from meta_spliceai.mllib.utils import (
    highlight, 
    savefig
)

class Metrics(object): 

    tracked = ['auc', 'fmax', 'fmax_negative', 'sensitivity', 'specificity', 'brier', 'balanced', 'log' ]
    plot_dir = os.path.join(os.getcwd(), "plot")

    def __init__(self, records={}, op=np.mean): 

        # self.records is nomrally a map: metric -> measurements ...
        # ... but in the case of PerformanceMetrics (derived class), this can be used to hold description (when merging multiple instances of PerformanceMetrics)
        self.records = {}
        if len(records) > 0: 
            self.add(records)
        self.op = op   # combiner on bags

    def add(self, item): 
        if isinstance(item, dict): 
            for k, v in item.items(): 
                if not k in self.records: self.records[k] = []
                if hasattr(v, '__iter__'): 
                    self.records[k].extend(list(v))  # this feature is usually used by PerformaceMetrics.add_doc()
                else: 
                    self.records[k].append(v)
        elif hasattr(item, '__iter__'):
            if len(item) == 0: 
                return # do nothing 
            elif len(item) == 1: 
                self.add_value(name=item)
            else: 
                self.add_value(name=item[0], value=item[1])
        else: 
            print('Warning: dubious input: %s' % str(item))
            self.add_value(name=item)         
        return
    def size(self):
        return len(self.records) 
    def size_bags(self):
        return sum(len(v) for v in self.records.values())  

    def clone(self): 
        import copy
        m = Metrics()
        m.records = copy.deepcopy(self.records)
        m.op = self.op 

        return m

    def add_value(self, name, value=None):
        if not name in self.records: 
            self.records[name] = []
        if value is not None: 
            self.records[name].append(value)
        else: 
            pass # do nothing
        return # no return value 

    def do(self, op=None, min_freq=0):  # perform an operation on the bags but does not change the internal represnetation
        if op is not None: self.op = op
        
        if hasattr(self.op, '__call__'): 
            mx = {}
            for k, v in self.records.items():
                # prune the keys with insufficient data points 
                if len(v) >= min_freq:  
                    mx[k] = self.op(v) 
        else: 
            assert isinstance(self.op, str), "Invalid operator: %s" % self.op
            if op.startswith('freq'):
                self.op = len
                mx = {}
                for k, v in self.records.items(): 
                    if len(v) >= min_freq: 
                        mx[k] = self.op(v)
            else: 
                raise NotImplementedError
        return mx
    def apply(self, op=None, min_freq=0):
        return self.do(op=op, min_freq=min_freq) 
    def aggregate(self, op=None, min_freq=0, by=''):
        if len(by) > 0: 
            if by.startswith('freq'): 
                return self.do(op=len, min_freq=min_freq)
            elif by == 'mean': 
                return self.do(op=np.mean, min_freq=min_freq)
        return self.do(op=op, min_freq=min_freq) # precedence op, self.op

    def display(self, by='freq', reverse=True, op=None, formatstr=''): 
        if by.startswith(('freq', 'agg')): 
            records = self.sort(by, reverse=reverse, op=op)
        else: 
            records = list(self.records.items())
        
        if formatstr: 
            for k, bag in records:
                try:  
                    print(formatstr.format(k, bag))
                except: 
                    print(formatstr.format(key=k, value=bag))
        else: 
            # default
            for k, bag in records: 
                print('[%s] %s' % (k, bag))
        return

    def sort_by_freq(self, reverse=True):
        # set op to len
        v = next(iter(self.records.values())) 
        # reduced? 
        if hasattr(v, '__iter__'): 
            return sorted( [(key, len(bag)) for key, bag in self.records.items()], key=lambda x: x[1], reverse=reverse) 
        return list(self.records.keys())  
    def sort_by_aggregate(self, reverse=True, op=None, min_freq=0):
        import operator 
        sorted_bags = self.aggregate(op=op, min_freq=min_freq)
        return sorted(sorted_bags.items(), key=operator.itemgetter(1, 0), reverse=reverse)  # sort by values first and then sort by keys 
    def sort(self, by='aggregate', reverse=True, op=None, min_freq=0):
        if by.startswith('agg'):
            return self.sort_by_aggregate(op=op, reverse=reverse, min_freq=min_freq) 
        elif by.startswith('freq'):  
            # choosing this will ignore what the default aggregate function is
            return self.sort_by_freq(reverse=reverse) 
        elif by == 'mean': 
            return self.sort_by_aggregate(op=np.mean, reverse=reverse, min_freq=min_freq)
        else: 
            raise ValueError("Metrics.sort(), invalid sort mode: {by}".format(by=by))
        # return self.sort_by_aggregate(op=op, reverse=reverse, min_freq=min_freq)

    def is_uniform(self): 
        # check if the meta data has a uniform length
        if not self.records: return True

        values = next(iter(self.records.values())) # python 2: self.records.itervalues().next() # python3: next(iter(self.records.values()))
        n_values = len(values)

        tval = True
        for k, vals in self.records.items(): 
            if n_values != len(vals): 
                tval = False
                break
        return tval

    # to be overridden by derived class
    def report(self, op=np.mean, message='', order='desc', tracked_only=True): 
        if op is not None: self.op = op
        assert hasattr(self.op, '__call__')

        title_msg = 'Performance metrics (aggregate method: %s)' % self.op.__name__
        highlight(message=title_msg, symbol='#', border=1)

        rank = 0; tracked_metrics = []
        if not message: 
            for i, metric in enumerate(Metrics.tracked): 
                val = self.op(self.records[metric]) # if hasattr(adict[metric]) > 1 else adict[metric]
                tracked_metrics.append((metric, val))
                # if tracked_only and not metric in Metrics.tracked: continue  
                rank += 1 
                print('... [%d] %s: %s' % (rank, metric, val))
        else: 
            message = '(*) %s\n' % message
            for i, metric in enumerate(Metrics.tracked): 
                val = self.op(self.records[metric]) # if hasattr(adict[metric]) > 1 else adict[metric]
                tracked_metrics.append((metric, val))
                # if tracked_only and not metric in Metrics.tracked: continue 
                rank += 1 
                message += '... [%d] %s: %s\n' % (rank, metric, val)

            # which performance metric(s) has the most advantange? 
            # metrics_sorted = sorted([(k, v) for k, v in self.do().items()], key=lambda x:x[1], 
            #                             reverse=True if order.startswith('desc') else False)  # best first
            metrics_sorted = sorted(tracked_metrics, key=lambda x: x[1], reverse=True if order.startswith('desc') else False)
            message += '... metrics order: %s\n'  % ' > '.join([m for m, _ in metrics_sorted])
            highlight(message=message, symbol='*', border=2)


        return

    def save(self, columns=[]): # save performance records 
        pass

    def my_shortname(self, context='suite', size=-1, domain='test', meta=''): 
        # domain: the name of the dataset or project 
        # context: the context in which the performance metrics was derived
        # meta: other parameters
        if size == -1: size = self.n_methods()
        
        name = 'performance_metrics-{context}-N{size}-D{domain}'.format(context=context, size=size, domain=domain)
        if meta: 
            name = '{prefix}-M{meta}'.format(prefix=name, meta=meta)
        return name

    @staticmethod
    def my_shortname(context, size=-1, domain='test', meta=''): 
        # domain: the name of the dataset or project 
        # context: the context in which the performance metrics was derived
        # meta: other parameters
        if size != -1:
            name = 'performance_metrics-{context}-N{size}-D{domain}'.format(context=context, size=size, domain=domain)
        else: 
            name = 'performance_metrics-{context}-D{domain}'.format(context=context, domain=domain)
        if meta: 
            name = '{prefix}-M{meta}'.format(prefix=name, meta=meta)
        return name 

    @staticmethod
    def plot_path(name='test', basedir=None, ext='tif', create_dir=True):
        # create the desired path to the plot by its name
        if basedir is None: basedir = Metrics.plot_dir
        if not os.path.exists(basedir) and create_dir:
            print('(plot) Creating plot directory:\n%s\n' % basedir)
            os.mkdir(basedir) 
        return os.path.join(basedir, '%s.%s' % (name, ext))

### end class Metrics

def plot_pr(cv_data, **kargs): 
    return

def plot_roc_curve_cv(model, X, y, n_folds=10): 
    """
    Plot ROC curve as the model gets trained. 

    """
    # Initialize a stratified k-fold object
    # n_folds = 10
    # NOTE: 10-fold CV may not provide sufficient training set size for each CV fold 

    # Set up 10-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=n_folds)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Set up the figure to plot the ROC curves
    fig, ax = plt.subplots(figsize=(10, 10))

    # Loop over the folds
    for i, (train, test) in enumerate(cv.split(X, y)):
        # Train the classifier on the training data
        model.fit(X.iloc[train], y.iloc[train])
        
        # Compute the ROC curve points
        fpr, tpr, _ = roc_curve(y.iloc[test], model.predict_proba(X.iloc[test])[:, 1])
        
        # Compute the area under the ROC curve
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Plot the ROC curve for this fold
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
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # Plot the standard deviation around the mean ROC curve
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    # Finalize the plot
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example")
    ax.legend(loc="lower right")
    plt.show()


def plot_roc(cv_data, **kargs):
    """
    Plot ROC curve given CV data. 
    
    Params
    ------
    cv_data: a list of (y_true, y_score) obtained from a completed CV process (e.g. datasink)

    **kargs
    -------
    file_name

    Memo
    ----
    1.  Run classifier with cross-validation and plot ROC curves
            cv = StratifiedKFold(n_splits=6)
            classifier = svm.SVC(kernel='linear', probability=True,
                            random_state=random_state)
    """
    import matplotlib.pyplot as plt
    # from scipy import interp
    from sklearn.metrics import roc_curve, auc
    # from utils import savefig

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    n_fold = len(cv_data)
    if not n_fold: 
        print('(plot_roc) No CV data. Aborting ...')
        return

    plt.clf()
    for i, (y_true, y_score) in enumerate(cv_data):
        # probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score) # roc_curve(y[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

    ### plotting
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    # plt.show()
    ext = 'tif'
    data_dir = os.path.join(os.getcwd(), f"data")
    output_dir = kargs.get("output_dir", data_dir)
    output_file = kargs.get("output_file", f"ROC_curve.{ext}")
    output_path = kargs.get("output_path", os.path.join(output_dir, output_file))
    savefig(plt, output_path, dpi=100, message='', verbose=True)

    return

def analyze_precision_recall(y_true, y_score, **kargs):
    """

    Params
    ------
    y_true, y_score

    confusion_matrix: assuming that confusion_matrix is a dataframe 

    """
    def eval_entry(y_true, y_pred):   # [todo]
        TP = np.sum((y_pred == pos_label) & (y_true == y_pred))
        TN = np.sum((y_pred == neg_label) & (y_true == y_pred))
        FP = np.sum((y_pred == pos_label) & (y_true != y_pred))
        FN = np.sum((y_pred == neg_label) & (y_true != y_pred))
        return (TP, FP, TN, FN)

    from sklearn.metrics import confusion_matrix

    p_threshold = kargs.get('p_threshold', 0.5)
    ep = 1e-9
    
    # need to convert probability scores to label predictions 
    y_pred = np.zeros(len(y_true))
    for i, p in enumerate(y_score): 
        # if i == 0: print('... p: {0}, pth: {1}'.format(p, p_threshold))
        if p >= p_threshold: 
            y_pred[i] = 1

    # cm = confusion_matrix(y_true, y_pred)
    # FP = cm.sum(axis=0) - np.diag(cm)  
    # FN = cm.sum(axis=1) - np.diag(cm)
    # TP = np.diag(cm)
    # TN = cm.sum() - (FP + FN + TP)  # remove values for numpy array
    
    pos_label = kargs.get('pos_label', 1)
    neg_label = kargs.get('neg_lable', 0)

    TP, FP, TN, FN = eval_entry(y_true, y_pred)

    metrics = {}
    # print('... nTP: %s, nTN: %s, nFP: %s, nFN: %s' % (TP, TN, FP, FN))

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN+ep)
    metrics['sensitivity'] = metrics['TPR'] = metrics['recall'] = TPR 

    # Specificity or true negative rate
    TNR = TN/(TN+FP+ep) 
    metrics['specificity'] = metrics['TNR'] = TNR

    # Precision or positive predictive value
    PPV = TP/(TP+FP+ep)
    metrics['precision'] = metrics['PPV'] = PPV

    # Negative predictive value
    NPV = TN/(TN+FN+ep)
    metrics['NPV'] = NPV

    # Fall out or false positive rate
    FPR = FP/(FP+TN+ep)
    metrics['FPR'] = FPR

    # False negative rate
    FNR = FN/(TP+FN+ep)
    metrics['FNR'] = FNR

    # False discovery rate
    FDR = FP/(TP+FP+ep)
    metrics['FDR'] = FDR

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN+ep) 
    metrics['accuracy'] = ACC

    return metrics

# calculate the brier skill score
def brier_skill_score(y, yhat, brier_ref=None, pos_label=1):
    from sklearn import metrics

    if brier_ref is None: 
        # Use a no-skill classifier as a reference that predicts P(y=1|x) to be the ratio of positive examples
        r_pos = np.sum(yhat == pos_label)/len(yhat)
        probabilities = [r_pos for _ in range(len(yhat))]
        brier_ref = metrics.brier_score_loss(y, probabilities)

    # calculate the brier score
    bs = metrics.brier_score_loss(y, yhat)
    # calculate skill score
    return 1.0 - (bs / brier_ref)

def calculate_all_metrics(y_true, y_pred, p_th=0.5, **kargs):
    """
    
    Parameters 
    ----------
    y_true: a list or a numpy array of ground truth labels
    y_pred: a list or a numpy array of probability scores

    Related 
    -------
    a. getPerformanceScores()
    """
    # Performance measures via probability predicitons
    metrics_scores = calculate_proba_metrics(y_true, y_pred, **kargs)
    
    # Convert probability predictions to labels
    y_pred_label = (y_pred >= p_th).astype(int)
    
    # Performance measures via label predictions
    metrics_labels = calculate_label_metrics(y_true, y_pred_label, **kargs)
    
    return metrics_scores, metrics_labels

def calculate_proba_metrics(y_true, y_score, **kargs): 
    """

    Related
    -------
    1. use evaluate() to evaluate individual metric 
    """
    from sklearn import metrics 
    from meta_spliceai.utils import utils_classifier as uclf

    metrics_table = {}
    tracked_metrics = kargs.get('metrics', Metrics.tracked) 
    verbose = kargs.get('verbose', 0)
    
    # AUC
    metrics_table['auc'] = metrics.roc_auc_score(y_true, y_score)

    # Fmax 
    beta = kargs.get('beta', 1.0)
    pos_label = kargs.get('pos_label', 1)
    metrics_table['fmax'] = uclf.fmax_score(y_true, y_score, beta = beta, pos_label = pos_label)
        
    # Fmax Negative
    metrics_table['fmax_negative'] = uclf.fmax_score(y_true, y_score, beta = beta, pos_label = 1 - pos_label)

    # Brier loss (smaller is better)
    metrics_table['brier_loss'] = metrics.brier_score_loss(y_true, y_score)

    # Brier skill score (larger is better)
    brier_ref = kargs.get('brier_ref', None)
    metrics_table['brier_score'] = metrics_table['brier'] = brier_skill_score(y_true, y_score, brier_ref=brier_ref)

    # Log loss (smaller is better)
    metrics_table['log'] = metrics_table['log_loss'] = metrics.log_loss(y_true, y_score)

    # [todo] Add other metrics here

    if verbose > 1: 
        print("[help] Available performance measures:")
        for k, v in metrics_table.items(): 
            if not tracked_metrics or (k in tracked_metrics): 
                print(f'  - {k}: {v}')

    return metrics_table

def calculate_label_metrics(y_true, y_pred, **kargs): 
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score

    if len(np.unique(y_pred)) > 2: 
        # ys = np.random.choice(y_pred, 5)
        p_threshold = kargs.get('p_threshold', None)
        if isinstance(p_threshold, float):
            y_pred = (y_pred >= p_threshold).astype(int)
        else: 
            msg = f"(calculate_label_metrics) `y_pred` must be a vector of label predictions but given probabilities without a valid threshold."
            raise ValueError(msg) 
            
    verbose = kargs.get('verbose', 0)
    tracked_metrics = kargs.get('metrics', Metrics.tracked) 

    metrics = {}
    metrics['acc'] = metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced'] = metrics['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)

    # Find another set of metrics (with alternative names)
    # NOTE: Remeber to specify a threshold in order to convert probabilities to crips class labels
    metrics2 = analyze_precision_recall(y_true, y_pred, **kargs)

    # [test]
    # assert np.allclose(metrics2['precision'], metrics['precision']), f"{metrics2['precision']} =! {metrics['precision']}"
    # assert np.allclose(metrics2['recall'], metrics['recall']), f"{metrics2['recall']} =! {metrics['recall']}"
    # NOTE: precision, recall, etc. from `analyze_precision_recall()` may be slightly different (nonetheless inconsequential) to avoid division-by-zero errors

    metrics.update(metrics2)

    if verbose > 1: 
        print("[help] Available label-specific performance measures:")
        for k, v in metrics.items(): 
            if not tracked_metrics or (k in tracked_metrics): 
                print(f'  - {k}: {v}')
    return metrics

def calculate_metrics(model, X_test, Y_test):
    '''Get model evaluation metrics on the test set.'''
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    
    # Get model predictions
    y_predict_r = model.predict(X_test)
    
    # Calculate evaluation metrics for assesing performance of the model.
    roc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
    acc = accuracy_score(Y_test, y_predict_r)
    prec = precision_score(Y_test, y_predict_r)
    rec = recall_score(Y_test, y_predict_r)
    f1 = f1_score(Y_test, y_predict_r)
    
    return acc, roc, prec, rec, f1

def train_and_evaluate(X, Y, model, scaler=None):
    '''Train a Random Forest Classifier and get evaluation metrics'''
    from sklearn.model_selection import train_test_split
    
    # Split train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y, random_state = 123)

    # All features of dataset are float values. You normalize all features of the train and test dataset here.
    if scaler is not None: 
        # scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else: 
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Call the fit model function to train the model on the normalized features and the diagnosis values
    model.fit(X_train_scaled, Y_train)

    # Make predictions on test dataset and calculate metrics.
    roc, acc, prec, rec, f1 = calculate_metrics(model, X_test_scaled, Y_test)

    return acc, roc, prec, rec, f1

def demo_confusion_matrix(): 
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from meta_spliceai.mllib.utils import savefig

    from sklearn.metrics import confusion_matrix, precision_recall_curve
    # NOTE: plot_confusion_matrix is no longer in sklearn.metrics

    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

    X, y = fetch_openml(data_id=1464, return_X_y=True, parser="pandas")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    n_classes = len(np.unique(y))

    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    cm_display = ConfusionMatrixDisplay(cm).plot()

    ext = 'tif'
    output_file_default = f"confusion_matrix-nC{n_classes}.{ext}"
    output_dir = os.path.join(os.getcwd(), 'test')
    output_path = os.path.join(output_dir, output_file_default)
    
    # print(dir(cm_display))
    # savefig(cm, output_path, ext=ext, dpi=100, message='', verbose=True)
    # NOTE: 'ConfusionMatrixDisplay' object has no attribute 'savefig'
    #       - plt.savefig(fpath, bbox_inches='tight', dpi=dpi) will not work

    cm_display.figure_.savefig(output_path, dpi=100)

    y_score = clf.decision_function(X_test)
    prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

    output_file_default = f"prc-nC{n_classes}.{ext}"
    output_path = os.path.join(output_dir, output_file_default)    
    pr_display.figure_.savefig(output_path, dpi=100)

    return

def demo_confusion_matrix_multiclass(): 
    """
    Reference 
    ---------
    1. https://medium.com/mlearning-ai/confusion-matrix-for-multiclass-classification-f25ed7173e66
    """
    from sklearn.metrics import confusion_matrix
    
    return

def demo_evaluate_classifier(): 
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import mutual_info_classif
    from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
    from meta_spliceai.utils.utils_data import get_example_dataset
    
    # from .utils import savefig
    from meta_spliceai.mllib.utils import savefig
    

    data = get_example_dataset(name="titanic"); print('-' * 80)

    X = data.drop('survived', axis=1)
    y = data['survived']
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y,
        test_size=0.3, random_state=0)

    model = LogisticRegression(random_state=0).fit(X, y)

    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)

    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob = model.predict_proba(X_test)[:, 1]

    # Performance meetrics 
    # - proba prediction: 'auc', 'fmax', 'fmax_negative'
    # - label prediction: 'acc', 'balanced_acc', 'precision', 'recall', 'f1', 'sensitivity', 'specificity'

    highlight("> Performance scores for the training set ...")
    metrics_scores_train, metrics_labels_train = calculate_all_metrics(y_train, y_prob_train)
    print(f"... performance for probability prediction:\n{metrics_scores_train}\n")
    print(f"... performance for label prediction:\n{metrics_labels_train}\n")

    highlight("> Performance scores for the test set ...")
    metrics_scores, metrics_labels = calculate_all_metrics(y_test, y_prob)
    print(f"... performance for probability prediction:\n{metrics_scores}\n")
    print(f"... performance for label prediction:\n{metrics_labels}\n")


def test(): 

    # Evaluating classifier performance
    # demo_evaluate_classifier()

    demo_confusion_matrix()

    return

if __name__ == "__main__": 
    test()
