"""Baseline for Kaggle AB."""

from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split, RepeatedKFold
from tqdm import tqdm


def prepare_dataset(DATA_PATH: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset.
    Load data, split into X and y, one-hot encode categorical

    Parameters
    ----------
    DATA_PATH: str :
        path to the dataset

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] :
        X and y
    """
    df = pd.read_csv(DATA_PATH)
    df = df.drop(["ID"], axis=1)
    y = df.pop("y").values

    # select only numeric columns
    X_num = df.select_dtypes(include="number")

    # select only categorical columns and one-hot encode them
    X_cat = df.select_dtypes(exclude="number")
    X_cat = pd.get_dummies(X_cat)

    # combine numeric and categorical
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.fillna(0).values

    return X, y


def cross_val_score(
        model: Callable,
        X: np.ndarray,
        y: np.ndarray,
        cv: Union[int, Tuple[int, int]],
        params_list: List[Dict],
        scoring: Callable,
        random_state: int = 42,
        show_progress: bool = False,
) -> np.ndarray:
    """
    Cross-validation score.

    Parameters
    ----------
    model: Callable :
        model to train (e.g. RandomForestRegressor)
    X: np.ndarray :

    y: np.ndarray :

    cv :
        number of folds fo cross-validation

    params_list: List[Dict] :
        list of model parameters

    scoring: Callable :
        scoring function (e.g. r2_score)

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    np.ndarray :
        cross-validation scores [n_models x n_folds]

    """
    if isinstance(cv, int):
        kf = KFold(n_splits=cv,shuffle=True, random_state=random_state)
    if isinstance(cv, tuple):
        kf = RepeatedKFold(n_splits=cv[0], n_repeats=cv[1], random_state=random_state)
    all_scores = []
    for n,params in tqdm(enumerate(params_list)):
        # fit
        model.set_params(**params)
        model_scores = []
        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, np.log1p(y_train))
        # predict
            y_pred = np.expm1(model.predict(X_test))
        #evaluate
            score = scoring(y_test, y_pred)
            model_scores.append(score)
        all_scores.append(model_scores)
    return np.array(all_scores)


def compare_models(
        cv: Union[int, Tuple[int, int]],
        model: Callable,
        params_list: List[Dict],
        X: np.ndarray,
        y: np.ndarray,
        random_state: int = 42,
        alpha: float = 0.05,
        show_progress: bool = False,
) -> List[Dict]:
    """Compare models with Cross val.

    Parameters
    ----------
    cv: int :
        number of folds fo cross-validation

    model: Callable :
        model to train (e.g. RandomForestRegressor)

    params_list: List[Dict] :
        list of model parameters

    X: np.ndarray :

    y: np.ndarray :

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    List[Dict] :
        list of dicts with model comparison results
        {
            model_index,
            avg_score,
            effect_sign
        }
    """
    all_scores = cross_val_score(model,
                                 X=X,
                                 y=y,
                                 cv=cv,
                                 params_list=params_list,
                                 random_state=random_state,
                                 scoring=r2_score)

    all_scores_mean = []
    if isinstance(cv, tuple):
        for j in range(cv[0]):
            indixes = [i for i in (range(cv[0] * cv[1])) if i % cv[0] == j]
            all_scores_mean.append(all_scores[:, indixes].mean(axis=1))
        all_scores_mean = np.array(all_scores_mean).T
    if isinstance(cv, int):
        all_scores_mean = all_scores.copy()

    compare_list = []
    for i, *(value) in enumerate(all_scores_mean[1:]):
        mean = np.array(value).mean()
        baseline_mean = np.array(all_scores_mean[0]).mean()
        _, p_value = ttest_rel(all_scores_mean[0], *value)
        compare_dict = {}
        compare_dict['model_index'] = i + 1
        compare_dict['avg_score'] = mean
        compare_dict['p_value'] = p_value
        compare_dict['effect_sign'] = (1 if mean > baseline_mean else -1) if p_value < alpha else 0
        compare_list.append(compare_dict)
    return sorted(compare_list, key=lambda x: x['avg_score'], reverse=True)


def run() -> None:
    """Run."""

    data_path = "train.csv.zip"
    random_state = 42
    cv = (5,2)
    params_list = [
        {"max_depth": 10},  # baseline
        {"max_depth": 2},
        {"max_depth": 3},
        # {"max_depth": 4},
        # {"max_depth": 5},
        # {"max_depth": 9},
        # {"max_depth": 11},
        {"max_depth": 12},
        {"max_depth": 15},
    ]

    X, y = prepare_dataset(data_path)
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state)

    result = compare_models(
        cv=cv,
        model=model,
        params_list=params_list,
        X=X,
        y=y,
        random_state=random_state,
        show_progress=True,
    )
    print("KFold")
    print(pd.DataFrame(result))


if __name__ == "__main__":
    run()
