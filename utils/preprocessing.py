import pandas as pd
import pickle

def preprocess(df: pd.DataFrame):
    """Apply preprocessing steps on the pandas dataframe.
    
    Arguments:
        df {pd.DataFrame} -- Catalog dataframe to preprocess
    
    Returns:
        pd.DataFrame -- Preprocessed dataframe
    """
    df = drop_unavailable_index(df)
    df = df.dropna() 
    df = normalize_ghi(df)
    # TODO : Shuffle dataframe while keeping days together
    return df

def normalize_ghi(df: pd.DataFrame):
    """Standardize the GHI values using the mean and standard deviation
    of the observed GHI values.
    
    Arguments:
        df {pd.DataFrame} -- Catalog dataframe to standardize.
    
    Returns:
        pd.DataFrame -- Dataframe with standardized GHI values
    """
    df_observed_ghi = df.filter(regex=("^..._GHI")) # Select only observed GHI columns
    mean = df_observed_ghi.stack().mean()
    std = df_observed_ghi.stack().std()
    df_ghi = df.filter(regex=("_GHI")) # Select all GHI columns
    normalized_df=(df_ghi-mean)/std

    pd.options.mode.chained_assignment = None # Disable chained_assignment warning for the update operation
    df.update(normalized_df) # Replace normalized columns in the original dataframe
    pd.options.mode.chained_assignment = 'warn' # Turn warning back on

    # TODO : Save mean and std to inverse normalization of predictions later
    
    return df

def drop_unavailable_index(df: pd.DataFrame):
    """Drops rows where file information is unavailable
    """
    available_col = pickle.load(open('data/available_col.pkl', 'rb'))

    pd.options.mode.chained_assignment = None # Disable chained_assignment warning for the assignement operation
    df.drop(available_col[available_col==0].index, inplace=True)
    pd.options.mode.chained_assignment = 'warn' # Turn warning back on
    
    return df