import numpy as np
import pandas as pd


def pivot_count_costs_single(df: pd.DataFrame):
    df = df.query('gmv > 0')
    final_table = pd.pivot_table(df,
                                 values='gmv',
                                 index=['week', 'user_id'],
                                 columns=['channel'],
                                 aggfunc='sum',
                                 fill_value=0,
                                 margins_name='user_id')
    final_table = final_table.reset_index()
    final_table.columns.names = [None]
    final_table['total_gmv'] = final_table.iloc[:, 2:].sum(axis=1)

    return final_table


def pivot_count_costs_multi(df: pd.DataFrame, indexes_groups):
    df = df.query('gmv > 0')
    final_table = pd.pivot_table(df,
                                 values='gmv',
                                 index=['week', 'user_id', df.index],
                                 columns=['channel'],
                                 aggfunc='sum',
                                 fill_value=0,
                                 margins_name='user_id')

    final_table['group'] = final_table.index.get_level_values(2).map(indexes_groups)

    final_table = final_table.reset_index().rename_axis(None, axis=1)
    # final_table.columns.names = [None]
    #
    # final_table.set_index('level_2', inplace=True)

    user_week_final_table = df.query('is_purchased==1').loc[:, ["week", 'user_id']]

    final_table = final_table.iloc[:, -5:].groupby('group').agg("sum").loc[::-1]

    final_table.set_index(user_week_final_table.index, inplace=True)

    final_table = pd.merge(user_week_final_table, final_table, left_index=True, right_index=True)

    final_table['total_gmv'] = final_table.iloc[:, 2:].sum(axis=1)

    final_table.index = np.arange(len(final_table))

    return final_table


def last_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate last touch attribution"""
    final_table = pivot_count_costs_single(events)
    return final_table


def group_by_actions(df: pd.DataFrame):
    current_index = {}  # user_id:curent_index
    current_user = df.iloc[-1].iloc[1]
    indexes = {}  # user:{}
    for index, row in reversed(list(df.iterrows())):
        if row.iloc[1] not in current_index.keys():
            current_index[row.iloc[1]] = 9999
            indexes[row.iloc[1]] = {9999: []}
        if row.iloc[3] == 1:
            current_index[row.iloc[1]] = index
            indexes[row.iloc[1]][current_index[row.iloc[1]]] = [index]
        else:
            indexes[row.iloc[1]][current_index[row.iloc[1]]].append(index)

    list_of_lists = [value for values in indexes.values() for key, value in values.items() if key != 9999]

    indexes_groups = {}
    for num, group in enumerate(list_of_lists):
        for index in group:
            indexes_groups[index] = num

    return indexes_groups


def first_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate first touch attribution"""
    df = events.copy()
    indexes_groups = group_by_actions(df)
    df['group'] = df.index.map(indexes_groups)

    for key, subgroup in df.groupby('group'):
        df.loc[df['group'] == key, 'gmv'] = subgroup['gmv'].values[::-1]
        df.loc[df['group'] == key, 'week'] = subgroup['week'].values[::-1]

    final_table = pivot_count_costs_single(df)

    return final_table


def linear_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate linear attribution"""
    df = events.copy()
    indexes_groups = group_by_actions(df)
    df['group'] = df.index.map(indexes_groups)

    df['gmv'] = df.groupby('group')['gmv'].transform(lambda x: round(x[::-1].mean()))

    final_table = pivot_count_costs_multi(df, indexes_groups)

    return final_table.round(2)


def u_shaped_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate U-Shaped attribution"""
    df = events.copy()
    indexes_groups = group_by_actions(df)
    df['group'] = df.index.map(indexes_groups)

    for key, subgroup in df.groupby('group'):
        if len(subgroup) == 1:
            continue
        elif len(subgroup) == 2:
            weight = [0.5, 0.5]
        else:
            weight = [0.4, *[0.2 / (len(subgroup) - 2)] * (len(subgroup) - 2), 0.4]

        df.loc[df['group'] == key, 'gmv'] = [df.loc[df['group'] == key, 'gmv'].sum()] * len(subgroup)

        df.loc[df['group'] == key, 'gmv'] = df.loc[df['group'] == key, 'gmv'].round(2) * weight

    final_table = pivot_count_costs_multi(df, indexes_groups)

    return final_table.round(2)


def roi(attribution: pd.DataFrame, ad_costs: pd.DataFrame) -> pd.DataFrame:
    """Calculate ROI"""
    # YOUR CODE HERE
    attribution_sum = pd.DataFrame(attribution.sum(axis=0)[2:-1]).rename(columns={0: "gmv"})
    attribution_sum = attribution_sum.reset_index()
    attribution_sum.rename(columns={"index": "channel"}, inplace=True)
    roi = attribution_sum.merge(ad_costs, on='channel')
    roi['roi%'] = round((roi['gmv'] - roi['costs']) / roi['costs'] * 100)
    return roi

#
# df = pd.read_csv('events.csv')
# df = df.query('user_id == 1')
#
# print(last_touch_attribution(df))
# print(first_touch_attribution(df))
#
# print(linear_attribution(df))
# print(u_shaped_attribution(df))
