import numpy as np
import pandas as pd


def train_valid_df(df:pd.DataFrame,dates:list=[201301,201406,201208,201103,201010]):
    """A Year has approx 35000 rows and a month has approx 3000 rows.
       Default fetches roughly 14500 rows """
    valid_idx = []
    for ele in dates:
        ele = str(ele)
        if len(ele) == 4:
            for idx in df.index:
                if idx.year == int(ele):
                    valid_idx.append(idx)
        elif len(ele) == 6:
            for idx in df.index:
                if idx.year == int(ele[:4]) and idx.month == int(ele[-2:]):
                    valid_idx.append(idx)
        else:
            print(f"{ele} Not recognised")
        
        valid_df = df.loc[valid_idx]
        train_df = df.drop(valid_idx)
    return train_df, valid_df



