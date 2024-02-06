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
from sklearn.model_selection import KFold, train_test_split
from lightgbm import LGBMRegressor
import lightgbm

np.set_printoptions(precision=3)

upstream = None
product = None
objective = None


# %% tags=["injected-parameters"]
# Parameters
random_seed = 1
valid_from = "2021-11-01"
objective = "rmse"
upstream = {"run-tests": {"nb": "path/to/run_tests.ipynb", "data": "path/to/feature_manual.csv"}}
product = {"nb": "path/to/lgbm_manual.ipynb"}


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
vars = Variables.manual_top10
train_vars = train_feat[vars + ['Value']]
train_vars = train_vars.dropna(axis=0)

y = train_vars.Value
# %%
kfold = KFold(n_splits = 5, random_state = random_seed, shuffle = True)

# Create test array to store predictions
oof_predictions = np.zeros(train_vars.shape[0])

# gbm_params = {'boosting': 'gbdt', 'n_estimators': 1310, 'learning_rate': 0.4999491976376299,
#                'num_leaves': 1420, 'max_depth': 18, 'min_data_in_leaf': 50, 'lambda_l1': 45.5,
#                'lambda_l2': 24.5, 'min_gain_to_split': 6.660113425150756,
#               'n_jobs': -1, 'early_stopping_rounds': 10
#               }

gbm_params = {'boosting': 'gbdt',
              'n_estimators': 20,
              # 'num_leaves': 500,
              'max_depth': 2,
              # 'learning_rate': 0.05,
              # 'min_data_in_leaf': 200,
              'lambda_l1': 15,
              # 'lambda_l2': 2,
              'n_jobs': -1,
              'early_stopping_rounds': 10
              }

# gbm_params = {'boosting': 'gbdt', 'n_estimators': 910, 'max_depth': 13, 'min_data_in_leaf': 10,
#               'lambda_l1': 3.0, 'lambda_l2': 3.0,
#               'n_jobs': -1,
#               # 'early_stopping_rounds': 10
#               }

train_vars = train_vars[vars]
model = LGBMRegressor(**gbm_params)
model.set_params(**{'objective': 'mae'})

for fold, (trn_ind, val_ind) in enumerate(kfold.split(train_vars)):
    print(f'Training fold #{fold + 1}')
    X_train, X_val = train_vars.iloc[trn_ind], train_vars.iloc[val_ind]
    y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    oof_predictions[val_ind] = model.predict(X_val)

# %%
'''
Plot feat importance again
'''
lightgbm.plot_importance(model)
plt.tight_layout()
plt.show()

# %%
'''
save model
'''
# gbm.booster_.save_model('path/to/lgbm_meteo_raw.txt')
# model = lightgbm.Booster(model_file='path/to/lgbm_meteo_raw.txt')


# %%
rmse = np.sqrt(mean_squared_error(y_true= y.values, y_pred=oof_predictions))
plot_residuals(y.values - oof_predictions, title=f'LGBM raw meteo features, rmse = {rmse}')

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

tsplot_pred(y_true=y_test, y_pred=y_hat, label='gbm valid set')

mae_test = np.mean(np.abs(y_test - y_hat))
rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_hat))

print(f"prediction is consistently overshooting by abt {y_hat.mean() - y_test.mean()}")
plot_residuals(y_test - y_hat, title='Test: residuals lgbm - meteo, arima, cyclical')

print(f"Test dataset mae={mae_test} rmse={rmse_test}")
