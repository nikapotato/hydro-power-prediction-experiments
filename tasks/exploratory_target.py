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
##NOTE necessary to ignore warnings for usable report.
import warnings
from pathlib import Path

from darts import TimeSeries

from darts.dataprocessing import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, RepeatedKFold, GridSearchCV

from hydro_timeseries.darts_utils import backtest_minute_data

warnings.filterwarnings("ignore")

import numpy as np
np.set_printoptions(precision=3)

import pandas as pd
from hydro_timeseries.plotting import *
from hydro_timeseries.util import *
import statsmodels.api as sm
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

# If this task has dependencies, declare them in the YAML spec and leave this
# as None
upstream = None

# This is a placeholder, leave it as None
product = None


# %% tags=["injected-parameters"]
# Parameters
random_seed = 1
valid_from = "2021-11-01"
plots_path = "/home/m/repo/hydro-power-prediction/plots"
upstream = {"run-tests": {"nb": "/home/m/repo/hydro-power-prediction/reports/run_tests.ipynb", "data": "/home/m/repo/hydro-power-prediction/products/data/data_raw_tested.csv"}}
product = {"nb": "/home/m/repo/hydro-power-prediction/reports/exploratory_target.ipynb"}


# %%
'''
Load data and split according to valid_from
Drop last two days
'''

data = load_timeseries_csv(upstream['run-tests']['data'])
data_daily = data.resample('D').mean()
data_hourly = data.resample('h').mean()

plots_path = Path(plots_path)
tsplot(data['2021-12-31':'2022-01-01']['Value'], do_plot_acf=False, filepath=plots_path / 'task_explanation.png')
# %%
tsplot(data['Value'], do_plot_acf=False, filepath=plots_path / 'value.png')

# %%
# target is loaded daily.
train_until = pd.to_datetime(valid_from) - timedelta(days=1)
train = data[:train_until]
val = data[valid_from:]

# NOTE drop NaN Values for the last two days
val = val.dropna(axis = 0)
# %%
'''
Plot time series.
15 minute - too much variance
daily mean - doable.
'''

tsplot(train.Value, label = 'Value, 15 min')

# %%
train_daily = data_daily[:train_until]
val_daily = data_daily[valid_from:]

tsplot(train_daily.Value, label = 'Value daily')
# %%
'''
Single day curve for max, min. 
True mean as a predictor
'''

train_daily_nonzero = train_daily[train_daily.Value != 0]
max_day = train_daily_nonzero.Value.idxmax().date()
min_day = train_daily_nonzero.Value.idxmin().date()
median_mean = train_daily_nonzero.Value.median()

max_day_pred = train[max_day.isoformat()].Value.copy()
max_day_mean = np.mean(train[max_day.isoformat()].Value)
max_day_pred[:] = max_day_mean

min_day_pred = train[min_day.isoformat()].Value.copy()
min_day_mean = np.mean(train[min_day.isoformat()].Value)
min_day_pred[:] = min_day_mean

tsplot(train[max_day.isoformat()].Value, y_pred=max_day_pred, label=f'Day curve for max daily mean, {max_day.isoformat()}, green predicted val')
tsplot(train[min_day.isoformat()].Value, y_pred=min_day_pred, label=f'Day curve for min daily mean, {min_day.isoformat()}, green predicted val')
# %%
'''
True Daily mean as a predictor 
- not bad 
- 288 kw error on average. 
'''

train['daily_true_mean'] = train.Value.resample('D').mean().reindex(train.index, method='ffill')
resid_true_mean = (train.Value - train.daily_true_mean).values

rmse_true_mean = np.sqrt(mean_squared_error(y_true=train.Value.values, y_pred=train.daily_true_mean.values))
plot_residuals(resid_true_mean, title='Residuals: $y_{15}-y_{daily}$, predictor: true daily mean')
print(f'rmse_true_mean_daily = {rmse_true_mean}')

# %%
'''
True hourly mean: 
- rmse = 122 kw error
'''

train['hourly_true_mean'] = train.Value.resample('h').mean().reindex(train.index, method='ffill')
resid_true_mean = (train.Value - train.hourly_true_mean).values

rmse_true_mean = np.sqrt(mean_squared_error(y_true=train.Value.values, y_pred=train.hourly_true_mean.values))
plot_residuals(resid_true_mean, title='Residuals: $y_{15}-y_{hourly}$, predictor: true hourly mean')
print(f'rmse_true_mean_hourly = {rmse_true_mean}')

# %%
'''
# Simple differencing d=1, check for stationarity


## STATIONARITY 
A stationary time series is one whose properties do not depend on the time at which the series is observed.

Some cases can be confusing — a time series with cyclic behaviour (but with no trend or seasonality) is stationary. This is because the cycles are not of a fixed length, so before we observe the series we cannot be sure where the peaks and troughs of the cycles will be.

In general, a stationary time series will have no predictable patterns in the long-term. Time plots will show the series to be roughly horizontal (although some cyclic behaviour is possible), with constant variance.

As well as looking at the time plot of the data, the ACF plot is also useful for identifying non-stationary time series. 
For a stationary time series, the ACF will drop to zero relatively quickly, while the ACF of non-stationary data decreases slowly. 
'''
daily_1diff = train_daily.Value.diff(1).dropna()
tsplot(daily_1diff, lags=15, label='Value diff = 1, "Daily changes"')

# 7 day window
adfuller_test(daily_1diff, window = 7)

print(f"Daily Value summary, mean={train_daily.Value.describe()['mean']}")
print(train_daily.Value.describe())

print("Daily Value changes summary")
# mean not in 0, slightly shifted.
print(daily_1diff.describe())
# %%
'''
Auto arma
'''
sm.tsa.arma_order_select_ic(daily_1diff.values, ic='bic', max_ar=3, max_ma=2)

# %%
'''
debug
'''
print(train_daily.Value.head())

print(train_daily.Value.tail())

# %%
'''
arima (2,1,1)
AIC, BIC - the lower the better
'''


model = sm.tsa.arima.ARIMA(train_daily.Value.values, order=(2,1,1), trend=None).fit(start_params=[0, 0, 0, 1])
print(model.summary())

tsplot(model.resid, label = 'Residuals $y_t - \hat{y}_t$', filepath=plots_path / 'arima_daily_1step.png')


model.plot_diagnostics(figsize=(15,5))
plt.show()

# %%
# '''
# Auto arima confirms (2,1,1) for daily.
# '''
#


# %%
'''
ARIMA (2,1,1) rolling day ahead forecast
'''
from datetime import timedelta
from datetime import date

valid_from_dt = date.fromisoformat(valid_from)
all_means = data_daily.dropna(axis=0)
all_means['Date'] = all_means.index.date

val['arima_daily'] = -10000
val['Date'] = pd.to_datetime(val.index.date)
training_dates = [valid_from_dt - timedelta(days=1)]
for idx, val_date in enumerate(sorted(set(val.Date))):
    date_ahead = val_date + timedelta(days = 1)

    print(f'current day={val_date.date()} predicting day={date_ahead.date()} with last training date={training_dates[-1]}')
    day_ahead = val[val.Date == date_ahead]

    # for training only use D-1, D-2, D-3...
    training_df = all_means[all_means.Date <= training_dates[-1]]
    arima_daily = sm.tsa.arima.ARIMA(training_df.Value.values, order=(2, 1, 1), trend=None).fit()
    pred = arima_daily.forecast(steps = 2)
    day_ahead_arima = pred[1]

    # fill dataframe with day ahead arima mean prediction
    val.loc[val.Date == date_ahead, 'arima_daily'] = day_ahead_arima

    # make a one day step
    training_dates.append(val_date.date())

baseline_eval = val[val.arima_daily != -10000]
print("Value should not be -10000")
print(baseline_eval.iloc[0])

# %%
'''
Evaluation of baseline for two step prediction

rmse = 353 kw on average
'''
baseline_eval[['Value', 'arima_daily']].plot(title='Hold out - arima day ahead mean')
plt.show()
baseline_eval.loc[:, 'resid'] = baseline_eval.Value - baseline_eval.arima_daily
tsplot(baseline_eval.resid, label='Hold out - day ahead arima residuals: $y_{t+1} - \hat{y}_{t+1}$', filepath=plots_path / 'arima_daily_2step.png')
baseline_mse = np.sqrt(np.mean(baseline_eval.resid.values**2))

smape_hold_out = smape(baseline_eval.Value, baseline_eval.arima_daily)

plot_residuals(baseline_eval.resid.values, title=f'Hold out - Residuals for day ahead ARIMA rmse={baseline_mse}')
print(f"Baseline: Day ahead ARIMA(2,1,1) for two step prediction rmse={baseline_mse} & smape={smape_hold_out} on hold out")
# %%
'''
Detrending and then explaining the seasonal + residual components using 
RF with cyclical variables + lagged residuals 
(NOTE: try longer trend: period = 30)
'''
# trend, seasonal, resid = seasonal_decompose_plot(data_daily.Value.dropna(axis=0), period=15)
trend, seasonal, resid = seasonal_decompose_plot(data_daily.Value.dropna(axis=0), period=7)
selector = ~trend.isna()
trend = trend[selector]
seasonal = seasonal[selector]
resid = resid[selector]
# %%
'''
(5,1,1) for trend period=30
(2,1,1) for trend if period not specified
'''

trend_arima = auto_arima(trend,
                          information_criterion='bic',
                          trace = True,
                          stepwise = True,
                          random_state=random_seed,
                          n_fits = 1000)

# %%
trend_arima = sm.tsa.arima.ARIMA(trend, order=(2,1,1), trend=None).fit()
print(trend_arima.summary())
trend_arima.plot_diagnostics(figsize=(15,5))
plt.show()

# %%
from darts.models import ARIMA
#RMSE=356.87 MAE=253.67
#arima = ARIMA(p=5, d=1, q=0, trend=None)>.fit(TimeSeries.from_series(train_daily['Value'].dropna()))

#RMSE=353.14 MAE=253.41
# arima = ARIMA(p=2, d=1, q=1, trend=None)

# arima = ARIMA(p=5, d=1, q=0, trend=None).fit(TimeSeries.from_series(train_daily['Value'].dropna()))
arima = ARIMA(p=2, d=1, q=1, trend=None).fit(TimeSeries.from_series(train_daily['Value'].dropna()))

# %%
backtest, ts_eval, rmse, mae, mape = backtest_minute_data(arima,
                   TimeSeries.from_series(data_daily['Value'].dropna()),
                   data_df=data.dropna(), valid_from='2021-01-20',
                   forecast_horizon=2, retrain=True,
                   scaler=None)

bt_df = backtest.pd_dataframe()

arima_true_resid = (data_daily['Value'] - bt_df['Value']).dropna()
tsplot(arima_true_resid, label="new_target = Seasonal + resid")
# %%
new_daily = arima_true_resid.to_frame('target')
new_daily['resid_lag2']=arima_true_resid.shift(2)
new_daily['arima'] = bt_df['Value']
new_daily['Value'] = data_daily['Value']

# %%
new_daily = (
        new_daily
        # .assign(hour = data.index.hour)
        # .assign(day_of_month = data.index.day)
        .assign(month = new_daily.index.month)
        # .assign(day_of_week = data.index.dayofweek)
        .assign(day_of_year = new_daily.index.day_of_year)
        .assign(week_of_year = new_daily.index.week)
        )

new_daily = generate_cyclical_features(new_daily, 'month', 12, 1)
new_daily = generate_cyclical_features(new_daily, 'week_of_year', 53, 1)
new_daily = generate_cyclical_features(new_daily, 'day_of_year', 366, 1)
# %%
new_daily = new_daily.dropna()
print(new_daily.head())
# %%

features = ['resid_lag2', 'sin_month', 'cos_month', 'sin_week_of_year',
       'cos_week_of_year', 'sin_day_of_year', 'cos_day_of_year']

X = new_daily[features][:pd.to_datetime(valid_from) - timedelta(days=1)]
y = new_daily['target'][:pd.to_datetime(valid_from) - timedelta(days=1)].values.ravel()

new_daily_test = new_daily[pd.to_datetime(valid_from):]
# %%
cv = RepeatedKFold(n_splits=4,n_repeats=2, random_state=random_seed)
rf = RandomForestRegressor(random_state=random_seed)
# %%
param_grid = {
    'n_estimators': list(range(50, 200, 20)),
    'max_features': [None, 'sqrt'],
    'max_depth' : [6, 7, 8, 10, 20, 30],
    'min_samples_split' : [4, 8, 16],
    'min_samples_leaf' : [4, 5, 6],
}
# %%
# CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, verbose=3, error_score="raise")
# CV_rf.fit(X, y)
# print(CV_rf.best_params_)
# print(CV_rf.best_score_)
#
# chosen_params = CV_rf.best_params_
# %%
chosen_params =\
    {'max_depth': 20,
     'max_features': 'sqrt',
     'min_samples_leaf': 5,
     'min_samples_split': 4,
     'n_estimators': 150}

# %%
rf = RandomForestRegressor(**chosen_params)
rf.fit(X, y)

# %%
y_test_pred = rf.predict(new_daily_test[features]) + new_daily_test['arima']
new_daily_test['Value_pred'] = y_test_pred
arimarf_rmse_daily = np.mean(np.sqrt((new_daily_test['Value'] - new_daily_test['Value_pred'])**2))
arima_rmse_daily = np.mean(np.sqrt((new_daily_test['Value'] - new_daily_test['arima'])**2))


print(f'ARIMA + RF on DAILY RMSE={arimarf_rmse_daily}')
print(f'ARIMA on DAILY RMSE={arima_rmse_daily}')
# %%
'''
IDEA: Hourly detrending instead of daily? - trend is almost daily mean.
Long ARIMA (5,1,0) ?
RNN to fit the hourly, 15 minute trend? 
'''
trend, seasonals, residuals = stl_decompose_plot(data_hourly.Value.dropna(), period=24)

# %%
'''
Best hourly arima (5,1,0) if max p = 5, huge ps when allowed higher. 
Model trend with some non linear models: RNN, TCN?
'''
hourly_trend_arima = auto_arima(trend.dropna(),
                          information_criterion='bic',
                          trace = True,
                          stepwise = True,
                          random_state=random_seed,
                          max_p=4,
                          max_d=2,
                          max_q=4,
                          n_fits = 100)

# %%
# from statsmodels.tsa.seasonal import STL
#
# period = 96
# res = STL(train.Value, period=period*7).fit()
#
# # %%
# sns.set_palette(sns.color_palette("bright", 6))
# res.plot()
# plt.show()

# %%
'''
BIC (5,2,2)
'''

# minute_trend_arima = auto_arima(res.trend,
#                           information_criterion='aic',
#                           trace = True,
#                           stepwise = True,
#                           random_state=random_seed,
#                           max_p=5,
#                           max_d=2,
#                           max_q=5,
#                           n_fits = 1000)
