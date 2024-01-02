import pandas as pd
import numpy as np


def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    """
    testing possibility predictions function
    :param df:
    :return: processed df
    """
    df1 = df.copy()
    df1["gmv"] = np.where(df.stock * df.price < df.gmv,
                          df.stock * df.price, np.floor(df.gmv / df.price) *
                          df.price)
    return df1
