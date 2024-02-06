import pandas as pd


def apply_agg_function(group):
    """

    :param group:
    :return:
    """
    agg_result = group['comp_price'].agg(group['agg_function'].iloc[0])
    return pd.Series({
        'agg': group['agg'].iloc[0],
         'base_price': group['base_price'].iloc[0],
         'comp_price': agg_result})


def new_price(x):
    """

    :param x:
    :return:
    """
    if x['comp_price'] is None:
        return x['base_price']
    if abs(x['comp_price'] - x['base_price']) < x['base_price'] * 0.2:
        return x['comp_price']
    else:
        return x['base_price']


def agg_comp_price(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    df = df.copy()
    df['agg_function'] = df['agg'].map({
        'avg': 'mean',
        'med': 'median',
        'min': 'min',
        'max': 'max',
        'rnk': 'min'
    })

    result = df.groupby('sku').apply(apply_agg_function).reset_index()

    result['new_price'] = result.apply(new_price, axis=1)
    return result

#
# data = {
#     'sku': [0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 6, 7, 8, 9],
#     'agg': ['max', 'med', 'med', 'avg', 'avg', 'rnk', 'rnk', 'max',
#     'max', 'max', 'max', 'min', 'min', 'min', 'med', 'rnk', 'rnk'],
#     'rank': [-1, 0, 1, 0, 1, 0, 2, 0, 0, 1, 2, 0, 1, 2, 0, -1, 0],
#     'base_price': [33.0, 17.7, 17.7, 76.7, 76.7, 39.7, 39.7, 18.0,
#     84.8, 84.8, 84.8, 73.6, 73.6, 73.6, 58.6, 35.2, 87.0],
#     'comp_price': [np.nan, 16.4, 21.8, 77.0, 73.9, 37.4, 47.9,
#     22.4, 106.0, 104.7, 80.9, 31.7, 33.6, 74.7, 71.3, np.nan, 88.2]
# }
# data = pd.DataFrame(data)
# print(agg_comp_price(data))
