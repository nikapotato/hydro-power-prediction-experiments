# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Add description here
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://docs.ploomber.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# %% tags=["parameters"]
# If this task has dependencies, declare them in the YAML spec and leave this
# as None
upstream = None

# This is a placeholder, leave it as None
product = None


# %% tags=["injected-parameters"]
# Parameters
random_seed = 1
valid_from = "2021-11-01"
upstream = {"load-data": {"nb": "path/to/load_data.ipynb", "train": "path/to/train.csv", "val": "path/to/test.csv"}}
product = {"nb": "path/to/hp_optimize_lgbm.ipynb"}


# %%
import pandas as pd
from hydro_timeseries.util import *
from hydro_timeseries.plotting import *
from sklearn.model_selection import KFold, train_test_split
from lightgbm import LGBMRegressor

# %%
'''
Load train, val
'''

train = load_timeseries_csv(upstream['load-data']['train'])
val = load_timeseries_csv(upstream['load-data']['val'])

y = train.Value
del train['Value']
X = train

X.shape

# %%
'''
GBDT - tradition. grad boosted tree
DART - gradient boosting which is a method that uses  dropout, standard in Neural Networks, to improve model regularization
GOSS - overfitting when dataset is small
'''

from sklearn.model_selection import KFold
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import mean_squared_error
import optuna

import warnings
warnings.filterwarnings("ignore")


def objective(trial, X, y):
    # param_grid = {
    #     # "device_type": trial.suggest_categorical("device_type", ['gpu']),
    #     "n_estimators": trial.suggest_categorical("n_estimators", [500, 1000, 5000]),
    #     "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
    #     "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
    #     "max_depth": trial.suggest_int("max_depth", 3, 12),
    #     "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 3000, step=100),
    #     "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
    #     "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
    #     "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
    #     "bagging_fraction": trial.suggest_float(
    #         "bagging_fraction", 0.01, 0.95, step=0.2
    #     ),
    #     "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
    #     "feature_fraction": trial.suggest_float(
    #         "feature_fraction", 0.01, 0.95, step=0.2
    #     ),
    # }

    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "boosting": trial.suggest_categorical("boosting", ['gbdt', 'dart', 'goss']),
        "n_estimators": trial.suggest_int("n_estimators", 10, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 20, step=5),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 3000, step=100),
        "lambda_l1": trial.suggest_float("lambda_l1", 3.0, 50, step=0.5),
        "lambda_l2": trial.suggest_float("lambda_l2", 3.0, 50, step=0.5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        # "bagging_fraction": trial.suggest_float(
        #     "bagging_fraction", 0.01, 0.95, step=0.2
        # ),
        # "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        # "feature_fraction": trial.suggest_float(
        #     "feature_fraction", 0.01, 0.95, step=0.2
        # ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LGBMRegressor(objective="regression", **param_grid)

        model.set_params(**{'objective': 'rmse'})
        model.set_params(**{'random_state': random_seed})
        # model.set_params(**{'verbose_eval': -1})

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="rmse",
            early_stopping_rounds=10,
            callbacks=[
                LightGBMPruningCallback(trial, "rmse")
            ],# Add a pruning callback
            verbose = -1,
        )
        preds = model.predict(X_test)
        cv_scores[idx] = np.sqrt(mean_squared_error(y_test, preds))

    return np.mean(cv_scores)

study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
func = lambda trial: objective(trial, X, y)
study.optimize(func, n_trials=10000, show_progress_bar=True, n_jobs=1)
