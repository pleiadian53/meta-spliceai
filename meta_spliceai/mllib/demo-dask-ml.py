

def demo_cv(): 
    import dask.array as da
    from dask_ml.datasets import make_regression
    # from sklearn.datasets import make_regression
    from dask_ml.model_selection import train_test_split

    X, y = make_regression(n_samples=125, n_features=4, random_state=0, chunks=50)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(f"> Shape(X_train): {X_train.shape}")

    return

def demo_distributed_xgboost(): 
    import xgboost as xgb
    import dask.array as da
    import dask.distributed

    cluster = dask.distributed.LocalCluster()
    client = dask.distributed.Client(cluster)

    # X and y must be Dask dataframes or arrays
    num_obs = 1e5
    num_features = 20
    X = da.random.random(size=(num_obs, num_features), chunks=(1000, num_features))
    y = da.random.random(size=(num_obs, 1), chunks=(1000, 1))
    print(f"> shape(X): {X.shape}, shape(y): {y.shape}")

    dtrain = xgb.dask.DaskDMatrix(client, X, y)
    # or
    # dtrain = xgb.dask.DaskQuantileDMatrix(client, X, y)

    output = xgb.dask.train(
        client,
        {"verbosity": 2, "tree_method": "hist", "objective": "reg:squarederror"},
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
    )

    return


def demo(): 

    # Cross validation
    # demo_cv() 

    # Distributed XGboost 
    demo_distributed_xgboost()


def test(): 
    pass

if __name__ == "__main__":
    demo()
    # test()