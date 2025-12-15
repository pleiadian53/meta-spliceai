



def example_existing_dataset(): 
    """
    Use an existing dataset to generate a new dataset with both 
    continous/numeric and categorical features. 

    """
    from sklearn.datasets import load_breast_cancer
    import pandas as pd

    # Load the breast cancer dataset (binary classification problem)
    data = load_breast_cancer()
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = pd.Series(data['target'], name='label')

    # Create a synthetic categorical feature
    X['synthetic_category'] = pd.qcut(X['mean radius'], 3, labels=["Small", "Medium", "Large"])
    # NOTE: The pd.qcut() function is used to quantile-based discretize variables. 
    #       In simpler terms, it takes a continuous variable and converts it into 
    #       a categorical variable by placing the data into discrete bins based on quantiles.

    # Combine the features and label into a single DataFrame
    df = pd.concat([X, y], axis=1)

    # Show the first few rows of the DataFrame
    print(df.head())

    return df

def example_synthetic_dataset(): 
    """
    Synthesize a dataset with both continous/numeric and categorical features. 
    
    """

    from sklearn.datasets import make_classification
    import pandas as pd
    import numpy as np

    # Create a synthetic dataset with a mixture of numerical and categorical variables
    X, y = make_classification(
        n_samples=1000, 
        n_features=4, 
        n_informative=3, 
        n_redundant=0,
        n_classes=2,
        random_state=42
    )

    # Convert to DataFrame for better visualization and manipulation
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, X.shape[1] + 1)])

    # Add a categorical feature
    df['feature_cat'] = np.random.choice(['A', 'B', 'C'], size=df.shape[0])

    # Add binary class labels
    df['label'] = y

    print(df.head())

    return df

def test(): 

    example_existing_dataset()

    return    

if __name__ == "__main__": 
    test()



