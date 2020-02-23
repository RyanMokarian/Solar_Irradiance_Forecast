import os
import numpy as np
import pandas as pd
import pickle
import random

from utils import data

def preprocess(df: pd.DataFrame, shuffle: bool = True, scale_label: bool = True):
    """Apply preprocessing steps on the pandas dataframe.
    
    Arguments:
        df {pd.DataFrame} -- Catalog dataframe to preprocess
    
    Returns:
        pd.DataFrame -- Preprocessed dataframe
    """
    df = df.replace('nan',np.NaN)
    if scale_label:
        df = normalize_ghi(df)
    if shuffle:
        df = shuffle_df(df)
    return df

def normalize_ghi(df: pd.DataFrame):
    """Standardize the GHI values using the mean and standard deviation
    of the observed GHI values.
    
    Arguments:
        df {pd.DataFrame} -- Catalog dataframe to standardize.
    
    Returns:
        pd.DataFrame -- Dataframe with standardized GHI values
    """
    df_ghi = df.filter(regex=("_GHI")) # Select all GHI columns
    normalized_df=(df_ghi-data.GHI_MEAN)/data.GHI_STD

    pd.options.mode.chained_assignment = None # Disable chained_assignment warning for the update operation
    df.update(normalized_df) # Replace normalized columns in the original dataframe
    pd.options.mode.chained_assignment = 'warn' # Turn warning back on

    # TODO : Save mean and std to inverse normalization of predictions later
    
    return df

def unnormalize_ghi(ghis: np.ndarray):
    """Unstandardize the GHI values using the mean and standard deviation
    of the observed GHI values.
    
    Arguments:
        ghis {np.ndarray} -- Array of GHI values to unstandardize.
    
    Returns:
        np.ndarray -- Array of GHI values with unstandardized GHI values
    """
    return ghis * data.GHI_STD + data.GHI_MEAN

def shuffle_df(df: pd.DataFrame):
    """Shuffle the dataframe while keeping days together
    
    Arguments:
        df {pd.DataFrame} -- Catalog dataframe to standardize.
        
    Returns:
        pd.DataFrame -- Shuffled dataframe 
    
    """
    df['just_date'] = df.index.date
    groups = [df for _, df in df.groupby('just_date')]
    np.random.shuffle(groups)
    df = pd.concat(groups).reset_index(drop=False)
    df = df.drop('just_date', axis=1).set_index('iso-datetime')
    return df

