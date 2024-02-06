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
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.feature_selection import RFE

from sklearn.metrics import mean_squared_error
import pandas as pd
from hydro_timeseries.util import *
from hydro_timeseries.plotting import *
from sklearn.model_selection import KFold, train_test_split, RepeatedKFold
from lightgbm import LGBMRegressor
import lightgbm
import numpy as np

np.set_printoptions(precision=3)

upstream = None
product = None


# %% tags=["injected-parameters"]
# Parameters
random_seed = 1
valid_from = "2021-11-01"
objective = "mae"
upstream = {"run-tests": {"nb": "path/to/run_tests.ipynb", "data": "path/to/feature_manual.csv"}}
product = {"nb": "path/to/lgbm_var_selector.ipynb"}


# %%
'''
Load train, val
'''

data = load_timeseries_csv(upstream['run-tests']['data'])
feat_manual = load_timeseries_csv(upstream['feature-manual']['data'])

train_until = pd.to_datetime(valid_from) - timedelta(minutes=15)
train = data[:train_until]
train_feat = feat_manual[:train_until]
test = data[valid_from:]
test_feat = feat_manual[valid_from:]

# %%
'''
select all vars
'''
vars = train_feat.columns.difference(['Value', 'Date'] + Variables.meteo).to_list()
# %%
# vars = Variables.manual_selected
train_vars = train_feat[vars + ['Value']]
train_vars = train_vars.dropna(axis=0)

y = train_vars.Value
# %%
# kfold = KFold(n_splits = 5, random_state = random_seed, shuffle = True)
kfold = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)

# Create test array to store predictions
oof_predictions = np.zeros(train_vars.shape[0])

gbm_params = {'boosting': 'gbdt',
              'n_estimators': 500,
              # 'num_leaves': 500,
              # 'max_depth': 3,
              # 'learning_rate': 0.05,
              # 'min_data_in_leaf': 300,
              'lambda_l1': 55,
              'lambda_l2': 5,
              # 'n_jobs': -1,
              'n_jobs': -1,
              'early_stopping_rounds': 10,
              'eval_metric':'mae'
              }

train_vars = train_vars[vars]
model = LGBMRegressor(**gbm_params)

'''
mae = |y_true - y_pred|  
mse = (y_true - y_pred)^2 - Variance kills me 
'''

model.set_params(**{'objective': 'mae'})

for fold, (trn_ind, val_ind) in enumerate(kfold.split(train_vars)):
    print(f'Training fold #{fold + 1}')
    X_train, X_val = train_vars.iloc[trn_ind], train_vars.iloc[val_ind]
    y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
    print(val_ind.shape)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    oof_predictions[val_ind] = model.predict(X_val)
# %%
'''
Feature importances
'''
fea_imp = get_fea_imp_gbm(model, train_vars)
print(fea_imp)
lightgbm.plot_importance(model)
plt.tight_layout()
plt.show()

# %%
fea_imp = fea_imp.sort_values('fea_imp', ascending=False).reset_index(drop=True)
print("LGBM 20 most important variables")
print(fea_imp.head(20))

# %%
top_20_list = fea_imp.head(20).cols.values
top_20_list
# print(top_20_list)
# %%
# plot_fea_imp_gbm(model, train_vars, file_name='gbm2')

# %%
# '''
# Recursive Feature Elimination(RFE) top 20
# '''
#
# fea_rank = rfe_var_eliminate(model, 10, train_vars, y)
# print(fea_rank)
#
# print("Scikit RFE selected top 20")
# rfe_selected = fea_rank[fea_rank.fea_rank == 1].cols.values
# print(rfe_selected)

# %%
rmse = np.sqrt(mean_squared_error(y_true= y.values, y_pred=oof_predictions))
plot_residuals(y.values - oof_predictions, title=f'OOF: lgbm meteo means plus arima features, rmse = {rmse}')

oof_series = pd.Series(oof_predictions)
oof_series.index = y.index
# tsplot_pred(y_true=y, y_pred=oof_series, label='LGBM, raw meteo features, oof training set')


test_vars = test_feat[vars + ['Value']]
test_vars = test_vars.dropna(axis=0)
y_test = test_vars.Value
test_vars = test_vars[vars]

y_hat = model.predict(test_vars)
y_hat = pd.Series(y_hat)
y_hat.index = y_test.index

#TODO check this
tsplot_pred(y_true=y_val, y_pred=y_hat, label='gbm valid set')
tsplot_pred(y_true=y_test, y_pred=y_hat, label='gbm valid set')

mae_test = np.mean(np.abs(y_test - y_hat))
rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_hat))

print(f"Test dataset mae={mae_test} rmse={rmse_test}")
