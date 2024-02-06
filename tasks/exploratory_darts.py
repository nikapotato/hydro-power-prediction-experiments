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
from datetime import timedelta

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.statistics import plot_residuals_analysis
from darts.models import AutoARIMA, ARIMA, VARIMA

from hydro_timeseries.plotting import plot_residuals, tsplot
from hydro_timeseries.util import load_timeseries_csv, add_mean_vars
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from hydro_timeseries.variables import Variables

upstream = None

# This is a placeholder, leave it as None
product = None

# %% tags=["injected-parameters"]
# Parameters
random_seed = 1
valid_from = "2021-11-01"
upstream = {"run-tests": {"nb": "/home/m/repo/hydro-power-prediction/reports/run_tests.ipynb", "data": "/home/m/repo/hydro-power-prediction/products/data/data_raw_tested.csv"}}
product = {"nb": "/home/m/repo/hydro-power-prediction/reports/exploratory_darts.ipynb"}


# %%
data = load_timeseries_csv(upstream['run-tests']['data'])
data = add_mean_vars(data)
data = data.dropna(axis = 0)
# data = data.dropna(axis=0)

# %%
data_daily = data.resample('D').mean()
# %%
ts_minute = TimeSeries.from_dataframe(data)
ts = TimeSeries.from_dataframe(data_daily)

# %%
# train, val = ts.split_before(pd.to_datetime(valid_from))
#
# power_train = train['Value']
# covariates_train = train[Variables.meteo_means_i]

power = ts['Value']
covariates = ts[Variables.meteo_means_i]

plt.figure(100, figsize=(25, 5))
ts['Value'].plot(label="data")
# val['Value'].plot(label="validation")
plt.show()

# %%
# plt.figure(100, figsize=(25, 5))
# train['Value'].plot(label="training")
# val['Value'].plot(label="validation")
# plt.show()

# %%
from darts.metrics import rmse

# # We first set aside the first 80% as training series:
# flow_train, _ = flow.split_before(0.8)


def eval_model(model, ts=None, past_covariates=None, future_covariates=None, scaler = None):
    # Past and future covariates are optional because they won't always be used in our tests

    # We backtest the model on the last 20% of the flow series, with a horizon of 10 steps:
    backtest = model.historical_forecasts(series=ts,
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=0.7,
                                          retrain=False,
                                          verbose=True,
                                          forecast_horizon=2,
                                          stride=1
                                          )
    if scaler:
        ts = scaler.inverse_transform(ts)
        backtest = scaler.inverse_transform(backtest)

    ts[-len(backtest) - 100:].plot()
    # plt.figure()
    backtest.plot(label='backtest (n=10)')
    plt.show()

    print('Backtest RMSE = {}'.format(rmse(ts, backtest)))

    return backtest

# %%
# from darts.dataprocessing.transformers import Scaler
# scaler = Scaler()
#
# power_scaled = scaler.fit_transform(power)

# %%
# from darts.models import BlockRNNModel
#
# brnn_no_cov = BlockRNNModel(input_chunk_length=50,
#                             output_chunk_length=2,
#                             n_rnn_layers=1)
#
# brnn_no_cov.fit(power_scaled,
#                 epochs=500,
#                 verbose=True)
#
# backtest_ts = eval_model(brnn_no_cov, ts=power_scaled, scaler=scaler)
# %%
scaler = Scaler()

covariates_daily = scaler.fit_transform(ts[Variables.meteo_means_i])

# %%

arima = AutoARIMA(
    information_criterion = 'bic',
    scoring = 'mae',
    trace = True,
    stepwise = True,
    out_of_sample_size=60,
)

# arima = ARIMA(p=2, d=1, q=1, trend=None, random_state=random_seed)
arima.fit(ts[:pd.to_datetime(valid_from)]['Value'],
          future_covariates=covariates_daily[:pd.to_datetime(valid_from)]
          )
pmd_model = arima.__getattribute__('model')
pmd_model.summary()

# %%
print(ts['Value'][:pd.to_datetime(valid_from) - timedelta(days=1)].head().pd_series())

print(ts['Value'][:pd.to_datetime(valid_from) - timedelta(days=1)].tail().pd_series())

# %%
arima = ARIMA(p=2, d=1, q=1, trend=None, random_state=random_seed).fit(
    series=ts['Value'][:pd.to_datetime(valid_from) - timedelta(days=1)],
    future_covariates=covariates_daily[:pd.to_datetime(valid_from)][['precip_mean', 'evapotranspiration_mean']]
    )

arima.__getattribute__('model').summary()

# %%
forecast_horizon = 2
backtest = arima.historical_forecasts(series=ts['Value'],
                                      future_covariates=covariates_daily[['precip_mean', 'evapotranspiration_mean']],
                                      start=pd.to_datetime(valid_from),
                                      retrain=True,
                                      verbose=True,
                                      forecast_horizon=forecast_horizon,
                                      stride=1
                                    )

backtest.time_index[0]

# %%
# sel = pd.to_datetime(valid_from) + timedelta(days = forecast_horizon - 1)
# %%

ts_eval_daily = ts[backtest.time_index]['Value']
resid = (ts_eval_daily - backtest)
tsplot(resid.pd_series())
plt.show()
# plot_residuals_analysis(resid)

# %%
bt_df = backtest.pd_dataframe()
data_eval = data.copy()

data_eval.loc[bt_df.index, 'Value_pred'] = bt_df['Value']
data_eval.loc[:,'Value_pred'] = data_eval['Value_pred'].ffill()

# %%
ts_eval = TimeSeries.from_dataframe(data_eval[['Value','Value_pred']])
ts_eval = ts_eval[backtest.time_index[0]:]

# %%
ts_eval['Value'].plot()
# plt.figure()z

ts_eval['Value_pred'].plot(label='backtest (n=2)')
plt.show()

print('Backtest RMSE = {}'.format(rmse(ts_eval['Value'], ts_eval['Value_pred'])))
# %%
plot_residuals_analysis(ts_eval['Value'] - ts_eval['Value_pred'])
plt.show()

# %%
resid_series = (ts_eval['Value'] - ts_eval['Value_pred']).pd_series()

# %%
plot_residuals(resid_series)


# %%
tsplot(resid_series, label='Hold out - day ahead arima residuals: $y_{t+1} - \hat{y}_{t+1}$')
# %%
#TODO baseline arima in darts
#TODO try FFILL


# model = AutoARIMA()
# model.fit(train)
#
#
# plt.show()
# %%

