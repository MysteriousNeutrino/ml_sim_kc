import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200)
params_list = [
    {"max_depth": 10},  # baseline
    {"max_depth": 2},
    {"max_depth": 3},
    {"max_depth": 4},
    {"max_depth": 5},
    {"max_depth": 9},
    {"max_depth": 11},
    {"max_depth": 12},
    {"max_depth": 15},
]

for params in params_list:
    # fit
    model.set_params(**params)
    model.fit(X_train, np.log1p(y_train))

    # predict
    y_pred = np.expm1(model.predict(X_test))

    #evaluate
    score = r2_score(y_test, y_pred)
