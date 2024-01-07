import pandas as pd
import numpy as np


def fillna_with_mean(
        df: pd.DataFrame, target: str, group: str
) -> pd.DataFrame:
    """
    Fill NaN for group mean
    :param df:
    :param target:
    :param group:
    :return:
    """
    df = df.copy()
    df[f'{target}'] = np.floor(df[f'{target}']
                               .fillna(df.groupby(f'{group}')[f'{target}']
                                       .transform('mean')))
    return df
