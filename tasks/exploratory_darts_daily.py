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
from darts.utils.utils import ModelMode, SeasonalityMode
from datetime import timedelta

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.statistics import plot_residuals_analysis
from darts.models import AutoARIMA, ARIMA, VARIMA, ExponentialSmoothing
from darts.metrics import rmse, mae

from hydro_timeseries.darts_utils import backtest_minute_data, exploratory_arima
from hydro_timeseries.plotting import plot_residuals, tsplot
from hydro_timeseries.pytorch_utils import pl_trainer_kwargs
from hydro_timeseries.util import load_timeseries_csv, add_mean_vars
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from darts.models import BlockRNNModel, RNNModel

from hydro_timeseries.variables import Variables

upstream = None

# This is a placeholder, leave it as None
product = None

# %% tags=["injected-parameters"]
# Parameters
random_seed = 1
valid_from = "2021-10-01"
simulate_from = "2021-12-01"
upstream = {"run-tests": {"nb": "../run_tests.ipynb", "data": "../data_raw_tested.csv"}}
product = {"nb": "../exploratory_darts_daily.ipynb"}


# %%
# your code here...
data = load_timeseries_csv(upstream['run-tests']['data'])
data = add_mean_vars(data)
data = data.dropna(axis = 0)
# %%
data_daily = data.resample('D').mean()
# %%
simulate_from = pd.to_datetime(simulate_from)
valid_from = pd.to_datetime(valid_from)

ts_minute = TimeSeries.from_dataframe(data)
ts = TimeSeries.from_dataframe(data_daily)
covariates = ts[Variables.meteo_means_i]
train_until = valid_from - timedelta(days=1)
# %%
plt.figure(100, figsize=(25, 5))
ts_minute[:valid_from]['Value'].plot(label="training")
ts_minute[valid_from:simulate_from]['Value'].plot(label="validation")
ts_minute[simulate_from:]['Value'].plot(label="simulation")
plt.show()

# %%
'''
ARIMA(2,1,1) on daily mean to observe covariates
- 1 no covariates
- 2 means across stations
- 3 all precip covariates

- Conclusion = most information in the target? 
'''
print("No covariates")
exploratory_arima(ts[:train_until]['Value'], covariates=None)

# %%
print("Mean covariate")
exploratory_arima(ts[:train_until]['Value'], covariates=ts[:train_until][Variables.meteo_means_i])

# %%
print("Precip covariate")
exploratory_arima(ts[:train_until]['Value'], covariates=ts[:train_until][Variables.precip])


# %%
cov_scaler = Scaler()
value_scaler = Scaler()
covariates_daily = ts[Variables.meteo_means_i]
covariates_daily_scaled = cov_scaler.fit_transform(ts[Variables.meteo_means_i])
value_daily_scaled = value_scaler.fit_transform(ts['Value'])

train_scaled = value_daily_scaled[:train_until]

# %%
train, val = value_daily_scaled.split_before(valid_from)
train_covs, val_covs = covariates_daily_scaled.split_before(valid_from)

# %%
val, sim = val.split_before(simulate_from)
val_covs, sim_covs = val_covs.split_before(simulate_from)

# %%
'''
Day ahead backtest on hold out
'''
arima = ARIMA(p=2, d=1, q=1, trend=None)

backtest, ts_eval, rmse_val, mae_val, smape_val = backtest_minute_data(arima, ts['Value'],
                                                   data_df=data, valid_from=simulate_from,
                                                   future_covariates=covariates_daily_scaled['precip_mean'],
                                                   forecast_horizon=2, scaler=None
                                                   )

# %%
'''
Exponential smoothing
'''
exp_smooth = ExponentialSmoothing(trend=ModelMode.ADDITIVE)

backtest, ts_eval, rmse_val, mae_val, smape_val = backtest_minute_data(exp_smooth, ts['Value'],
                     data_df=data, valid_from=simulate_from,
                     forecast_horizon=2, scaler=None
                    )

# %%
'''
Simple RNN
'''
value_scaler = Scaler()
value_daily_scaled = value_scaler.fit_transform(ts['Value'])

# %%
'''
Simple RNN for daily = approx ARIMA
'''
brnn_no_cov = BlockRNNModel(input_chunk_length=30,
                            output_chunk_length=2,
                            n_rnn_layers=1)

brnn_no_cov.fit(train,
                epochs=100,
                verbose=True,
                val_series = val
                )

backtest, ts_eval, rmse_val, mae_val, smape_val = backtest_minute_data(brnn_no_cov, value_daily_scaled, scaler = value_scaler,
                                   data_df=data, valid_from=simulate_from, retrain=False)

# %%
'''
RNN with past covariates
'''
soil_moisture = covariates_daily_scaled[['soil_moisture_mean']]
# rain = covariates_daily['precip_mean']

brnn_past = BlockRNNModel(input_chunk_length=50,
                         output_chunk_length=2,
                         n_rnn_layers=3)

brnn_past.fit(train,
             past_covariates=train_covs['soil_moisture_mean'],
             val_series=val,
             val_past_covariates=val_covs['soil_moisture_mean'],
             epochs=200,
             verbose=True)
# %%
backtest, ts_eval, rmse_val, mae_val, smape_val = backtest_minute_data(brnn_past, value_daily_scaled,
                                data_df=data, valid_from=simulate_from,
                                past_covariates=soil_moisture,
                                scaler = value_scaler, retrain=False
                                )

# %%
'''
GRU + future covariates 330 RMSE
'''

future_cov = covariates_daily_scaled[['precip_mean', 'pressure_mean']]

'''
Generally speaking, `training_length` should have a higher value than `input_chunk_length`
because otherwise during training the RNN is never run for as many iterations as it will during
training.
'''

rnn_rain = RNNModel(input_chunk_length=3,
                    training_length=5,
                    model='GRU',
                    n_rnn_layers=2,
                    dropout=0.5
                    )

rnn_rain.fit(train,
             future_covariates=train_covs[['precip_mean', 'temperature_mean']],
             val_series=val,
             val_future_covariates=val_covs[['precip_mean', 'temperature_mean']],
             epochs=100,
             verbose=True)

# %%

backtest, ts_eval, rmse_val, mae_val, smape_val = backtest_minute_data(rnn_rain, value_daily_scaled,
                                   data_df=data, valid_from=simulate_from,
                                   # past_covariates=soil_moisture,
                                   future_covariates=covariates_daily_scaled[['precip_mean', 'temperature_mean']],
                                   scaler = value_scaler, retrain=False
                             )

# %%
'''
RegressionModel

- Past covariates are time series whose past values are known at prediction time. Those series often contain values that have to be observed to be known.
- Future covariates are time series whose future values are known at prediction time. More precisely, for a prediction made at time t for a forecast horizon n, the values at times t+1, …, t+n are known. Often, the past values (for times t-k, t-k+1, …, t for some lookback window k) of future covariates are known as well. Future covariates series contain for instance calendar informations or weather forecasts.

The lags of the target and past covariates have to be strictly negative (in the past), 
whereas the lags of the future covariates can also be positive (in the future). 
For instance, a lag value of -5 means that the value at time t-5 is used to predict the target at time t; and a lag of 0 means that the future covariate value at time t is used to predict the target at time t. 
In the code below, we specify past covariate lags as [-5, -4, -3, -2, -1] which means that the model will look at the last 5 past_covariates values
Similarly, we specify the future covariate lags as [-4, -3, -2, -1, 0] which means that the model will look at the last 4 historic values (lags -4 to -1) and the current value (lag 0) of the future_covariates.

'''
rain = train_covs['precip_mean']
soil = train_covs['soil_moisture_mean']

future_covs_regr = train_covs[['precip_mean','soil_moisture_mean']]

from darts.models import RegressionModel

regr_model = RegressionModel(lags=[-5,-4,-3,-2],
                             # lags_past_covariates=[-3, -2, -1],
                             lags_future_covariates=[ -2, -1, 0])

regr_model.fit(train,
               # past_covariates=soil,
               future_covariates=future_covs_regr
               )
# %%
backtest, ts_eval, rmse_val, mae_val, smape_val = backtest_minute_data(regr_model, value_daily_scaled,
                                   # past_covariates=covariates_daily_scaled['soil_moisture_mean'],
                                   future_covariates=covariates_daily_scaled[['precip_mean','soil_moisture_mean']],
                                   data_df=data, valid_from=simulate_from,
                                   scaler = value_scaler, retrain=False
                             )



