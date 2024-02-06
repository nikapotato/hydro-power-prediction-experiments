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
import lightgbm
from lightgbm import LGBMRegressor
# from darts.metrics import smape
from pmdarima.metrics import smape
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from statsmodels.tools.eval_measures import rmse
from tsfresh import select_features
import lightgbm as lgb

from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt

from darts import TimeSeries
from datetime import timedelta
import time
import pandas as pd

from hydro_timeseries.darts_utils import plot_backtest
from hydro_timeseries.plotting import plot_residuals, tsplot
from hydro_timeseries.util import load_timeseries_csv, add_mean_vars, select_kbest_features, get_fea_imp_gbm
from hydro_timeseries.variables import Variables
from sklearn.model_selection import TimeSeriesSplit
import torch


# from bokbokbok.eval_metrics.regression import LogCoshMetric
# from bokbokbok.loss_functions.regression import LogCoshLoss
from bokbokbok.eval_metrics.regression import RMSPEMetric
from bokbokbok.loss_functions.regression import SPELoss

upstream = None

# This is a placeholder, leave it as None
product = None


# %% tags=["injected-parameters"]
# Parameters
valid_from = "2021-09-01"
simulate_from = "2021-11-01"
random_seed = 1
upstream = {"run-tests": {"nb": "../reports/run_tests.ipynb", "data": "../data/feature_manual.csv"}}
product = {"nb": "../reports/arima_lgbm_baseline.ipynb"}


# %%
'''
Feature selection - focused on the hourly mean 
- detrend hourly by daily arima and try to fit the residuals using meteo + cyclicals
'''
data = load_timeseries_csv(upstream['run-tests']['data'])
data_ts = TimeSeries.from_dataframe(data)

simulate_from = pd.to_datetime(simulate_from)
valid_from = pd.to_datetime(valid_from)

# %%
'''
Splits
'''
# Repeated cross val
N_SPLITS = 3
N_REPEATS = 3

plt.figure(100, figsize=(25, 5))
data_ts[:valid_from]['Value'].plot(label="Training")
data_ts[valid_from:simulate_from]['Value'].plot(label="Validation")
data_ts[simulate_from:]['Value'].plot(label="Simulation")
plt.show()

# %%
features_all = load_timeseries_csv(upstream['feature-manual']['data'])
features = features_all.dropna()
assert np.all(features[['arima_current','arima_prev']] > 0), "Arima trend variables are <= 0"

# %%
'''
Arima used to detrend - good but some residual trend seem to remain.
'''
features.loc[:,'new_target'] = features.loc[:,'Value'] - features.loc[:,'arima_current']
tsplot(features['new_target'])

# %%
'''
one more diff
'''
features['target_diff'] = features['new_target'].diff(1)
features = features.dropna()

# %%
tsplot(features['target_diff'])
# %%
'''
split train test
'''
train = features[:simulate_from - timedelta(minutes=15)]
test = features[simulate_from:]

# %%
'''
no split
'''
train = features
test = features[simulate_from:]

# %%
var_hourly = features.filter(regex='.*hourly.*').columns.to_list()
var_minute = features.filter(regex='.*minute.*').columns.to_list()
# var_time = [features.filter(like='sin').columns] + [features.filter(like='cos').columns]
feature_cols = features.columns.difference(['Value', 'arima_current', 'target_diff', 'new_target'])
features_selected = select_features(X = train[feature_cols], y = train['new_target'], ml_task='regression')
dropped_features = train.columns.difference(features_selected.columns).difference(['Value', 'arima_current', 'target_diff', 'new_target'])
print(f"Remaining features n={len(features_selected.columns)}, Dropped n={len(dropped_features)} features")
# print("Dropped features")
# print(dropped_features)

# %%
X = features_selected
# X = train[feature_cols]
y = train['new_target']


# %%
'''
root mean squared percentage error
'''
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def custom_rmspe_eval(y_true, y_pred):
    return "rmspe", rmspe(y_true, y_pred), False

# fh = list(range(2, 5952))
# # kfold = ExpandingWindowSplitter(initial_window=5952, fh=fh, step_length=3000)
# %%
N_SPLITS = 4

# kfold = RepeatedKFold(n_splits = N_SPLITS, n_repeats=N_REPEATS, random_state = random_seed)
# kfold = KFold(n_splits = N_SPLITS, shuffle=False)
kfold = TimeSeriesSplit(n_splits=4, test_size=6623)

fh = np.arange(1, 193)
initial_window = 300 * 96
cv = ExpandingWindowSplitter(initial_window=initial_window, fh=fh)
n_splits = cv.get_n_splits(y)


# Create test array to store predictions
oof_predictions = np.zeros(X.shape[0])

gbm_params = {'boosting': 'gbdt',
              'n_estimators': 10000,
              # 'num_leaves': 500,
              # 'max_depth': 30,
              # 'learning_rate': 0.05,
              # 'min_data_in_leaf': 10,
              # 'lambda_l1': 5,
              # 'learning_rate':0.1,
              # 'lambda_l2': 2,
              'n_jobs': -1,
              # 'early_stopping_rounds': 50
              }

model = LGBMRegressor(**gbm_params)
model.set_params(**{'objective': 'rmse'})

# %%
# scores = np.empty(N_SPLITS * N_REPEATS)
# smapes = np.empty(N_SPLITS * N_REPEATS)

scores = np.empty(N_SPLITS)
smapes = np.empty(N_SPLITS)

for fold, (trn_ind, val_ind) in enumerate(kfold.split(X)):
    print("=" * 12 + f"Training fold {fold+1}" + 12 * "=")
    start = time.time()

    X_train, X_val = X.iloc[trn_ind], X.iloc[val_ind]
    y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

    print(f"Train length={len(y_train)}, val_length={len(y_val)}")
    # print(f"Train ends at={X_train.iloc[-1].index} and val starts at={X_val.iloc[-1].index}")

    # train_weights = (1 / np.square(y_train)).values
    # val_weights = (1 / np.square(y_val)).values

    model = LGBMRegressor(**gbm_params)
    model.set_params(**{'objective': 'rmse'})
    model.fit(
        X_train,
        y_train,
        # sample_weight=train_weights,
        # eval_sample_weight=[train_weights, val_weights],
        # categorical_feature=cat_idx,
        eval_set=[(X_val, y_val)],

        eval_metric='rmse',
        verbose=True,
    )
    oof_predictions[val_ind] = model.predict(X_val)
    preds = model.predict(X_val)
    preds_value = features.loc[y_val.index.values]['arima_current'] + preds
    value = features.loc[y_val.index.values]['Value']

    smape_val = smape(value, preds_value)
    smapes[fold] = smape_val
    rmse_val = rmse(y_val, preds)
    scores[fold] = rmse_val
    print(f"Fold {fold+1} finished with rmse: {rmse_val:.5f}kw against resid \n")

# %%
print(f"Mean rmse across folds={np.mean(scores)}")
print(f"Mean smape across folds={np.mean(smapes)}")
# %%
gbm_params = {'boosting': 'gbdt',
              'n_estimators': 10000,
              # 'num_leaves': 500,
              # 'max_depth': 30,
              # 'learning_rate': 0.05,
              # 'min_data_in_leaf': 100,
              # 'lambda_l1': 50,
              # # 'learning_rate':0.1,
              # 'lambda_l2': 50,
              'n_jobs': -1,
              # 'early_stopping_rounds': 1000
              }

model = LGBMRegressor(**gbm_params)
model.set_params(**{'objective': 'rmse'})

X_test = test[features_selected.columns.difference(['new_target'])]
y_test = test['new_target']

# weights = 1 / np.square(y)
model.fit(
        X,
        y,
        # sample_weight=weights,
        # eval_sample_weight=[weights],
        # categorical_feature=cat_idx,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        verbose=True,
)
lgb.plot_metric(model,'rmse')
plt.show()
# %%
# eval_results = model.evals_result_

# %%
'''
Feature importances
'''
fea_imp = get_fea_imp_gbm(model, X)
print(fea_imp[0:30])
lgb.plot_importance(model, max_num_features=30)
plt.tight_layout()
plt.show()

# %%
np.all((features[simulate_from:]['Value'] - features[simulate_from:]['arima_current']).values == test['new_target'])
# %%
y_test_pred = model.predict(X_test)
print(f"Test rmse={rmse(y_test, y_test_pred)}")

# %%
trend = features['arima_current'][simulate_from:]
value_true = features['Value'][simulate_from:]
value_pred = y_test_pred + trend
# %%
mae = np.mean(np.abs(value_true - value_pred))
mae
# %%
print(f"Arima + lgbm rmse={rmse(y_test, y_test_pred)}")
print(rmse(value_true, value_pred))
# %%
from pmdarima.metrics import smape
smape_test = smape(value_true, value_pred)
smape_test

# %%
plot_backtest(model, value_true, value_pred)

# %%
# from darts.models import ARIMA
#RMSE=356.87 MAE=253.67
#arima = ARIMA(p=5, d=1, q=0, trend=None)>.fit(TimeSeries.from_series(train_daily['Value'].dropna()))

#RMSE=353.14 MAE=253.41
# arima = ARIMA(p=2, d=1, q=1, trend=None)

# arima = ARIMA(p=5, d=1, q=0, trend=None).fit(TimeSeries.from_series(train_daily['Value'].dropna()))
# arima = ARIMA(p=2, d=1, q=1, trend=None).fit(TimeSeries.from_series(train_daily['Value'].dropna()))


# # %%
# '''
# hourly
# '''
# data_hourly = data.resample('h').mean().dropna(axis=0)
# features_hourly = features.resample('h').mean().dropna(axis=0)
#
# # %%
# minute_vars = list(features_hourly.filter(regex='minute'))
# specific_meteo_vars = Variables.meteo
# selector = features_hourly.columns.difference(minute_vars + specific_meteo_vars)
#
# hourly = TimeSeries.from_dataframe(data_hourly)
# features = TimeSeries.from_dataframe(features_hourly[selector])
# feature_cols = features.columns.difference(['Value']).to_list()
# # %%
# '''
# Scaling
# '''
# target_scaler = Scaler()
# covs_scaler = Scaler()
#
# target = target_scaler.fit_transform(features['Value'])
# covs = covs_scaler.fit_transform(features[feature_cols])
# # %%
# '''
# Split training/simulation
# '''
#
# train, sim = target.split_before(simulate_from)
# train_covs, sim_covs = covs.split_before(simulate_from)
#
#
# train_df = train.pd_dataframe()
# train_covs_df = train_covs.pd_dataframe()
#
# sim_df = sim.pd_dataframe()
# sim_covs_df = sim_covs.pd_dataframe()
# # %%
# import statsmodels.api as sm
# mod = sm.OLS(train_df,train_covs_df)
# fii = mod.fit()
# fii.summary2()
# # %%
# '''
# 'soil_moisture_index_104', 'soil_moisture_index_81',
# '''
# fs, selected = select_kbest_features(train_covs_df, train_df, k=30)
# print(selected.columns)
# # %%
# fs, selected = select_kbest_features(train_covs_df, train_df, score_func=mutual_info_regression, k=30)
# print(selected.columns)
#
# # %%
# cv = RepeatedKFold(n_splits=7, n_repeats=3, random_state=1)
# # define the pipeline to evaluate
# model = LinearRegression()
# fs = SelectKBest(score_func=mutual_info_regression)
# pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
# # define the grid
# grid = dict()
# num_features = list(range(50, len(train_covs_df.columns)))
#
# grid['sel__k'] = num_features
# # define the grid search
# search = GridSearchCV(pipeline, grid, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=cv, verbose=3)
# # perform the search
# results = search.fit(train_covs_df, train_df.values.ravel())
# # summarize best
# print('Best MAE: %.3f' % results.best_score_)
# print('Best Config: %s' % results.best_params_)
# # summarize all
# means = results.cv_results_['mean_test_score']
# params = results.cv_results_['params']
# for mean, param in zip(means, params):
#     print(">%.3f with: %r" % (mean, param))
#
# cv_res = results.cv_results_
# best_params_id = np.argmax(cv_res['mean_test_score'])
# lowest_variance_id = np.argmin(cv_res['std_test_score'])
# # %%
# # selected_num_vars = num_features[best_params_id]
# fs, selected = select_kbest_features(train_covs_df, train_df.values.ravel(), score_func=mutual_info_regression, k=108)
# print(selected.columns)
# # %%
# # pca = PCA(n_components=100)
# vars = list(selected.columns)
# # %%
# # scaler_back = MinMaxScaler()
# # vals = scaler_back.fit_transform(features_hourly['Value'].values.reshape(1,-1))
# # %%
# # pipeline = Pipeline(steps=[('pca',pca), ('lr', model)])
# pipeline = Pipeline(steps=[('lr', model)])
# scores = cross_val_score(pipeline, train_covs_df[vars], train_df.values.ravel(), scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, verbose=3)
#
# # %%
# mod = sm.OLS(train_df.values.ravel(),train_covs_df[vars])
# fii = mod.fit()
# fii.summary2()
# # %%
# mmscaler = MinMaxScaler(feature_range=(0, 1))
# mmscaler.fit_transform(data_hourly['Value'].values.ravel().reshape(-1,1))
#
# # %%
# regr = RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=0)
#
# # %%
# regr.fit(train_covs_df[vars], train_df.values.ravel())
#
# # %%
# y_pred = regr.predict(sim_covs_df[vars])
# y_pred = mmscaler.inverse_transform(y_pred.reshape(-1,1))
# y_true = mmscaler.inverse_transform(sim.values())
# np.sqrt(np.mean((y_true - y_pred)**2))
# # %%
# mmscaler.inverse_transform(y_pred.reshape(-1,1))
#
# # %%
# lr = LinearRegression()
# lr.fit(train_covs_df[vars], train_df.values.ravel())
#
# y_pred = lr.predict(sim_covs_df[vars])
# y_true = sim.values()
#
# # %%
# scores = mmscaler.inverse_transform(scores.reshape(-1,1))
# y_pred = mmscaler.inverse_transform(y_pred.reshape(-1,1))
# y_true = mmscaler.inverse_transform(y_true.reshape(-1,1))
# # %%
# plot_residuals(mmscaler.inverse_transform(y_true - y_pred))
#
# # %%
# brr = RegressionModel(lags=[], lags_future_covariates=[0], model=LinearRegression())
#
# brr.fit(
#     train, future_covariates=train_covs[vars]
# )
#
# # %%
# y_pred_ts = brr.predict(768, series=train, future_covariates=sim_covs[vars])
# y_pred_ts = target_scaler.inverse_transform(y_pred_ts)
# plot model performance for comparison
# pyplot.boxplot(results, labels=num_features, showmeans=True)
# pyplot.show()

