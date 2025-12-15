import os, sys

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, f1_score, classification_report, confusion_matrix 
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

import seaborn as sns
import matplotlib.pyplot as plt

# from .utils import savefig
from meta_spliceai.mllib.utils import savefig 

RED = "rgba(245,173,168,0.6)"
GREEN = "rgba(211,255,216,0.6)"

def draw_confusion_matrix(true, preds, target_names=['nmd_ineff', 'nmd_eff'], **kargs): 
    # from sklearn.metrics import confusion_matrix 
    # import seaborn as sns
    
    plt.clf()
    plt.figure(figsize=(10,6))
    conf_matx = confusion_matrix(true, preds)
    fx=sns.heatmap(conf_matx, annot=True, fmt=".2f",cmap="GnBu")
    fx.set_title('Confusion Matrix \n')
    fx.set_xlabel('\n Predicted Values\n')
    fx.set_ylabel('True Values\n')
    fx.xaxis.set_ticklabels(target_names)
    fx.yaxis.set_ticklabels(target_names)

    # Optional parameters for plot display and persistence
    display = kargs.get("display", True)
    save = kargs.get("save", True)
    verbose = kargs.get("verbose", 1)

    if save: 
        output_dir_default = os.path.join(os.getcwd(), "plot")
        output_dir = kargs.get("output_dir", output_dir_default)

        ext = kargs.get("ext", "tif")
        output_file = kargs.get("output_file", f"confusion_matrix-test.{ext}")

        output_path = os.path.join(output_dir, output_file)
        if verbose: print(f"[evaluation] Saving confusion matrix output to:\n{output_path}\n")
        savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)

    if display: 
        plt.show()

    return plt 

def draw_confusion_matrix_sklearn(model, X, y, **kargs): 
    from sklearn.metrics import ConfusionMatrixDisplay

    colormap = kargs.get("cmap", plt.cm.Blues)
    normalize = kargs.get("normalize", 'true')

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_estimator(model, X, y, cmap=colormap, normalize=normalize)

    # Optional parameters for plot display and persistence
    title_text = kargs.get("title", 'Confusion Matrix')
    display = kargs.get("display", True)
    save = kargs.get("save", True)
    verbose = kargs.get("verbose", 1)

    plt.title(title_text)

    if save: 
        output_dir_default = os.path.join(os.getcwd(), "plot")
        output_dir = kargs.get("output_dir", output_dir_default)

        ext = kargs.get("ext", "tif")
        output_file = kargs.get("output_file", f"confusion_matrix-test.{ext}")

        output_path = os.path.join(output_dir, output_file)
        if verbose: print(f"[evaluation] Saving confusion matrix output to:\n{output_path}\n")
        savefig(plt, output_path, ext=ext, dpi=100, message='', verbose=True)

    if display: 
        plt.show()

    return

def evaluate_multiclass_classifier(model, X_test, y_test, metrics=['accuracy', 'f1_macro', 'roc_auc_macro', 'mcc']):
    """


    Memo
    ----
    * Usage example
        metrics = ['accuracy', 'f1_macro', 'roc_auc_macro', 'mcc']
        results = evaluate_multiclass_classifier(model, X_test, y_test, metrics=metrics)
        for metric, value in results.items():
            print(f"{metric.capitalize()}: {value:.4f}")
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, matthews_corrcoef
    from sklearn.preprocessing import label_binarize
    import numpy as np

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Initialize results dictionary
    results = {}

    if 'accuracy' in metrics:
        # Calculate accuracy
        results['accuracy'] = accuracy_score(y_test, y_pred)

    if 'f1_macro' in metrics:
        # Calculate F1 score
        results['f1_macro'] = f1_score(y_test, y_pred, average='macro')

    if 'roc_auc_macro' in metrics:
        # Binarize the labels for ROC AUC calculation
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        # Calculate ROC AUC score
        n_classes = y_test_bin.shape[1]
        roc_auc = 0
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc += auc(fpr, tpr)
        results['roc_auc_macro'] = roc_auc / n_classes

    if 'mcc' in metrics:
        # Calculate Matthews Correlation Coefficient (MCC)
        results['mcc'] = matthews_corrcoef(y_test, y_pred)

    return results

def classification_report_to_dataframe(y_test, y_pred, **kargs): 
    import pandas as pd

    report = classification_report(y_test, y_pred, output_dict=True)

    # Extract accuracy value from the dictionary and then remove it
    accuracy= report['accuracy']
    del report['accuracy']

    df_report = pd.DataFrame(report).transpose()

    label_names = kargs.get("label_names", {})
    if label_names: 
        label_names = {str(k): v for k, v in label_names.items()}
        df_report = df_report.rename(index=label_names)

    # display(df_report)
    return df_report, accuracy

def create_df_from_confusion_matrix(confusion_matrix, class_labels=None):
    
    ## create a dataframe
    if not len(class_labels):
        ## if class labels not received, created dummy headers and index
        df = pd.DataFrame(data=confusion_matrix, 
                          index=[f"True Class-{i+1}" for i in range(confusion_matrix.shape[0])],
                          columns=[f"Predicted Class-{i+1}" for i in range(confusion_matrix.shape[0])])
    else:
        ## create headers and index using class labels
        df = pd.DataFrame(data=confusion_matrix, 
                          index=[f"True {i}" for i in class_labels],
                          columns=[f"Predicted {i}" for i in class_labels])
    
    ## unpivot dataframe and rename columns
    df = df.stack().reset_index()
    df.rename(columns={0:'instances', 'level_0':'actual', 'level_1':'predicted'}, inplace=True)
    """
    >>> df
            actual          predicted     instances
          True Fraud    Predicted Fraud       10
          True Fraud    Predicted Legit       4
          True Legit    Predicted Fraud       2
          True Legit    Predicted Legit       12
    
    """
    
    ## determine classification color based on correct classification or not.
    df["colour"] = df.apply(lambda x: 
                               GREEN if x.actual.split()[1:] == x.predicted.split()[1:] 
                               else RED, axis=1)

    node_labels = pd.concat([df.actual, df.predicted]).unique()
    node_labels_indices = {label:index for index, label in enumerate(node_labels)}
    
    ## map actual and predicted columns to numbers
    df =  df.assign(actual    = df.actual.apply(lambda x: node_labels_indices[x]),
                    predicted = df.predicted.apply(lambda x: node_labels_indices[x]))
    
    ## determine text for hovering on connecting edges of sankey diagram
    def get_link_text(row):
        if row["colour"] == GREEN:
            instance_count = row["instances"]
            source_class = ' '.join(node_labels[row['actual']].split()[1:])
            target_class = ' '.join(node_labels[row['predicted']].split()[1:])
            return f"{instance_count} {source_class} instances correctly classified as {target_class}"
        else:
            instance_count = row["instances"]
            source_class = ' '.join(node_labels[row['actual']].split()[1:])
            target_class = ' '.join(node_labels[row['predicted']].split()[1:])
            return f"{instance_count} {source_class} instances incorrectly classified as {target_class}"
        
    df["link_text"] = df.apply(get_link_text, axis = 1)
    return df, node_labels
    

def plot_confusion_matrix_as_sankey(confusion_matrix, class_labels=None, **kargs):
    
    """
    plots sankey diagram from confusion matrix and class labels
    
    The function acceps:
        - confusion_matrix
                [[TP, FN]
                 [FP, TN]]
        - class_labels:
            class_labels[0]: Label for positive class
            class_labels[1]: Label for negative class
        
    """
    from plotly import graph_objects as go
    
    df, labels = create_df_from_confusion_matrix(confusion_matrix,  class_labels)
    
    fig = go.Figure(data=[go.Sankey(
    
    node = dict(
      pad = 20,
      thickness = 20,
      line = dict(color = "gray", width = 1.0),
      label = labels,
      hovertemplate = "%{label} has total %{value:d} instances<extra></extra>"
    ),
    link = dict(
      source = df.actual, 
      target = df.predicted,
      value = df.instances,
      color = df.colour,
      customdata = df['link_text'], 
      hovertemplate = "%{customdata}<extra></extra>"  
    ))])

    fig.update_layout(title_text="Confusion Matrix Sankey Diagram", font_size=15,
                      width=500, height=400)
    
    return fig


def demo_multiclass_evaluation(): 
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    # Create a synthetic 3-class dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=0, n_classes=3, random_state=42
    )

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit an XGBoost classifier
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Evaluate the classifier using the custom function
    metrics = ['accuracy', 'f1_macro', 'roc_auc_macro', 'mcc']
    results = evaluate_multiclass_classifier(model, X_test, y_test, metrics=metrics)

    # Print the results
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    return

def demo(): 

    demo_multiclass_evaluation()

    return

if __name__ == "__main__": 
    demo()