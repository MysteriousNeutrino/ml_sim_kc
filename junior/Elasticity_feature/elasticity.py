import pandas as pd
import numpy as np
from scipy.stats import linregress


def elasticity_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    count elasticity for every sku through linreg and r^2
    :param df:
    :return:
    """
    df = df.copy()
    df.qty = np.log(df.qty + 1)
    r_squares = {}
    for sku in df.sku.unique():
        df_for_one_sku = df.query(f'sku == {sku}')
        # print(df_for_one_sku)
        model = linregress(df_for_one_sku.price, df_for_one_sku.qty)
        r_squares[sku] = model.rvalue ** 2
    df = pd.DataFrame(list(r_squares.items()), columns=['sku', 'elasticity'])

    return df
