from pprint import pprint
from tabulate import tabulate
import pandas as pd

def describe(df, name='', print_=True): 
    from utils_sys import highlight
    msg = ""
    dataset = name
    if not dataset:
        dataset = 'test'
    msg += f"[demo] {dataset}'s data shape: {df.shape}\n" 
    msg += f"        ... columns:\n{df.columns}\n"
    msg += highlight(df.describe())
    
    if print_: 
        pprint(msg)
        df.describe()
    return msg

def display(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))

def config_display(df): 
    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 2)
    display(df)

def dataframes_equal(df1, df2):
    # Basic checks

    assert isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame), \
        f"type(df1):{type(df1)} <> type(df2): {type(df2)}"

    if not df1.columns.equals(df2.columns):
        return False
    if df1.shape != df2.shape:
        return False

    for col in df1.columns:
        col1 = df1[col]
        col2 = df2[col]

        for idx in df1.index:
            val1 = col1.at[idx]
            val2 = col2.at[idx]

            if isinstance(val1, pd.DataFrame) or isinstance(val2, pd.DataFrame):
                if not dataframes_equal(val1, val2):  # recursive check
                    return False
            else:
                if pd.isna(val1) and pd.isna(val2):
                    continue
                if val1 != val2:
                    return False

    return True

def project_to_dict(df, cols=[]): 
    """
    Convert a dataframe with two columns into a dictionary. 
    """
    if len(cols) > 0: 
        assert set(cols) <= set(df.columns), f"Unrecognized columns: {cols}"
    else: 
        assert df.shape[1] >= 2
        cols = df.columns[:2]
    
    # Convert the DataFrame to a dictionary
    return df.set_index(cols[0])[cols[1]].to_dict()

def demo_saving_nested_dataframe(): 

    # Create a main DataFrame
    df_main = pd.DataFrame({
        'model': ['Model1', 'Model2'],
        'accuracy': [0.9, 0.85]
    })

    # Create DataFrames to nest within the main DataFrame
    df1 = pd.DataFrame({
        'feature': ['a', 'b'],
        'importance': [0.1, 0.2]
    })

    df2 = pd.DataFrame({
        'feature': ['c', 'd'],
        'importance': [0.3, 0.4]
    })

    # Convert these DataFrames to dictionaries
    df1_dict = df1.set_index('feature')['importance'].to_dict()
    df2_dict = df2.set_index('feature')['importance'].to_dict()

    # Add these dictionaries to the main DataFrame
    df_main['top_features'] = [df1_dict, df2_dict]

    # Save to CSV
    df_main.to_csv('model_performance.csv', index=False)

    # Read from CSV
    df_read = pd.read_csv('model_performance.csv')

    # Convert the string back to a dictionary and then to a DataFrame
    df_read['top_features'] = df_read['top_features'].apply(eval)
    df_read['top_features'] = df_read['top_features'].apply(lambda x: pd.DataFrame(list(x.items()), columns=['feature', 'importance']))

    # print(dataframes_equal(df_main, df_read))    
    print(df_read['top_features'][1])

    print(df_main['top_features'][1])

    return


def demo_nested_df_equal(): 
    # Example:
    df1 = pd.DataFrame({
        'a': [1, 2, 3],
        'nested_df': [pd.DataFrame({'x': [1, 2], 'y': [3, 4]}),
                    pd.DataFrame({'x': [5, 6], 'y': [7, 8]}),
                    None]
    })

    df2 = pd.DataFrame({
        'a': [1, 2, 3],
        'nested_df': [pd.DataFrame({'x': [1, 2], 'y': [3, 4]}),
                    pd.DataFrame({'x': [5, 6], 'y': [7, 8]}),
                    None]
    })

    print(dataframe_equal(df1, df2))  # This should print True

    return

def test(): 

    # demo_nested_df_equal()

    # Create a sample DataFrame
    df = pd.DataFrame({
        'feature': ['a', 'b', 'c'],
        'importance': [0.1, 0.2, 0.3]
    })

    print(project_to_dict(df))

    demo_saving_nested_dataframe()

    return

if __name__ == "__main__": 
    test()


    