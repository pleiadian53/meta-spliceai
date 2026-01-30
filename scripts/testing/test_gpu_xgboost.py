# run inside `conda run -n surveyor python -`
import numpy as np, xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
dtrain = xgb.DMatrix(X.astype(np.float32), label=y)

params = dict(
    objective="multi:softprob",
    num_class=3,
    tree_method="gpu_hist",   # request GPU
    max_bin=16,
    eval_metric="mlogloss",
)
bst = xgb.train(params, dtrain, num_boost_round=10)

pred = bst.predict(dtrain).argmax(axis=1)
print("GPU test accuracy", accuracy_score(y, pred))
