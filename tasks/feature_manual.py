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

from datetime import timedelta

# %% tags=["parameters"]
# If this task has dependencies, declare them in the YAML spec and leave this
# as None
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.trend import STLForecaster
from sktime.transformations.series.detrend import STLTransformer

from hydro_timeseries.plotting import get_windows, tsplot, trend_eval_plot
from hydro_timeseries.util import load_timeseries_csv, add_mean_vars, generate_cyclical_features, cv_evaluate, smape, \
    rmse, mae
from hydro_timeseries.variables import Variables

upstream = None

# This is a placeholder, leave it as None
product = None
arima_init_steps = None
n_lags = None
import warnings
warnings.filterwarnings("ignore")


# %% tags=["injected-parameters"]
# Parameters
random_seed = 1
valid_from = "2021-11-01"
n_lags = 7
arima_init_steps = 7
plots_path = "../plots"
upstream = {"run-tests": {"nb": "../run_tests.ipynb", "data": "../data_raw_tested.csv"}}
product = {"nb": "../feature_manual.ipynb", "data": "../feature_manual.csv"}


# %%
data = load_timeseries_csv(upstream['run-tests']['data'])
data = add_mean_vars(data)

plots_path = Path(plots_path)
# %%
'''
Cyclical datetime features
'''
data = (
        data
        # .assign(hour = data.index.hour)
        # .assign(day_of_month = data.index.day)
        .assign(month = data.index.month)
        # .assign(day_of_week = data.index.dayofweek)
        .assign(day_of_year = data.index.day_of_year)
        .assign(week_of_year = data.index.week)
        )
# %%
data = generate_cyclical_features(data, 'month', 12, 1)
data = generate_cyclical_features(data, 'week_of_year', 53, 1)
data = generate_cyclical_features(data, 'day_of_year', 366, 1)

# %%
'''
Get daily means 
'''
data_daily = data.resample('D').mean()
value_daily = data.Value.resample('D').mean()


# %%
def generate_lagged_vars(df, n_lags, colname='Value', start_from=2):
    # df_n = df.copy()
    new_vars = []
    for n in range(start_from, n_lags + 1):
        new_var_name = f"daily_mean_{colname}_lag{n}"
        df[new_var_name] = df[colname].shift(n)
        new_vars.append(new_var_name)
    return df, new_vars

def generate_ewms(df, colname='Value', span_list=[3, 7, 14, 30], shift=2, var_suffix='daily'):
    # df_n = df.copy()
    new_vars = []
    for span in span_list:
        new_var_name = f"{colname}_{var_suffix}_ewm{span}"
        df[new_var_name] = df[colname].ewm(span=span).mean().shift(shift)
        new_vars.append(new_var_name)
    return df, new_vars

# %%
'''
Lagged Value means
lag has to be at least 2, because only the prev day is known when day ahead is being predicted
meteo can lag by 1. 
'''
data_daily, value_lag_vars = generate_lagged_vars(data_daily, colname='Value', n_lags=n_lags, start_from=2)
data_daily, precip_mean_vars = generate_lagged_vars(data_daily, colname='precip_mean', n_lags=n_lags, start_from=1)
data_daily, soil_moisture_mean_vars = generate_lagged_vars(data_daily, colname='soil_moisture_mean', n_lags=n_lags, start_from=1)
data_daily, snow_mean_vars = generate_lagged_vars(data_daily, colname='snow_mean', n_lags=n_lags, start_from=1)

lagged_selector = []
for var in Variables.meteo_i:
    data_daily, meteo_lagged_vars = generate_lagged_vars(data_daily, colname=var, n_lags=n_lags, start_from=1)
    lagged_selector = lagged_selector + meteo_lagged_vars

print("First two should be NaN, last two should be fine")
print(data_daily[['Value', 'daily_mean_Value_lag2']].head())
print(data_daily[['Value', 'daily_mean_Value_lag2']].tail())

# %%
selector = lagged_selector + value_lag_vars + precip_mean_vars + soil_moisture_mean_vars + snow_mean_vars
data = data.join(data_daily[selector], how='left')
# data_joined.loc[:, value_lag_vars] = data[value_lag_vars].ffill()
data.loc[:, selector] = data[selector].ffill()

# %%
print("Value lag2, Value and daily_mean_Value_lag2 should be equal")
print(data_daily.loc['2022-01-01'].Value)
print(data.loc['2022-01-03'][['Value', 'daily_mean_Value_lag2']].head())

# %%
print("precip_mean lag1, precip_mean and daily_mean_precip_mean_lag1 should be equal")
print(data_daily.loc['2022-01-02'].precip_mean)
print(data.loc['2022-01-03'][['precip_mean', 'daily_mean_precip_mean_lag1']].head())

# %%
'''
Exp Weighted averages - target + meteo
'''
'''
Daily - long timeframe week, month, 3 days, well suited for target - target known only day backwards.
#TODO - minute steps until something
'''
daily_spans = [3, 7, 14, 30, 60]
data_daily, daily_value_ewms = generate_ewms(data_daily, span_list=daily_spans, colname='Value', shift=2)

selector_ewm = daily_value_ewms

data = data.join(data_daily[selector_ewm], how='left')
data.loc[:, selector_ewm] = data[selector_ewm].ffill()
print(data[daily_value_ewms].head())
print(data[daily_value_ewms].tail())
# %%
'''
Hourly
'precip_mean',
 'snow_mean',
 'pressure_mean',
 'temperature_mean',
 'soil_moisture_mean',
 'evapotranspiration_mean'
'''
spans_hourly = [6, 9, 16, 32, 96, 192, 768, 1536]
data_hourly = data.resample('H').mean()

hourly_var_names = []
for var in Variables.meteo_means_i + Variables.meteo_i:
    print(var)
    data_hourly, hourly_var_ewms = generate_ewms(data_hourly, colname=var, span_list=spans_hourly, var_suffix='hourly', shift=0)
    hourly_var_names += hourly_var_ewms

data = data.join(data_hourly[hourly_var_names], how='left')
data.loc[:, hourly_var_names] = data[hourly_var_names].ffill()
print(data.filter(regex='precip.*hourly').head())
print(data.filter(regex='precip.*hourly').tail())
# %%
'''
Minute steps - short timeframes
well suited for weather, weather covariates are known all the way to the specific time step
'''
spans_minute = [4, 8, 16, 32, 64, 128, 256]

minute_var_names = []
for var in Variables.meteo_means_i + Variables.meteo_i:
    print(var)
    data, minute_var_ewms = generate_ewms(data, colname=var, span_list=spans_minute, var_suffix='minute', shift=0)
    minute_var_names += minute_var_ewms

print(data.filter(regex='precip.*minute').head())
print(data.filter(regex='precip.*minute').tail())

# %%
y = data.Value.asfreq('15min').dropna()
step = 96
fh = np.arange(97, 193)
cv = ExpandingWindowSplitter(initial_window=step*(arima_init_steps+1), fh=fh, step_length=step)

n_splits = cv.get_n_splits(y)
print(f"Number of Folds = {n_splits}")

train_windows, test_windows = get_windows(y, cv)
# plot_windows(y, train_windows, test_windows)

print(y[test_windows[-1]].index)


# %%
y_pred, y_test, smape_test, rmse_test, mae_test, df = cv_evaluate(forecaster=AutoETS(trend='add', damped_trend=True), y=y, cv=cv, X=None)

print(f"AutoETS as detrender n_val={len(y_pred)} SMAPE={smape_test:.4f} RMSE={rmse_test:.2f} MAE={mae_test:.2f}")
# %%
fh_2days = np.arange(1, 193)
ets_trend = AutoETS(trend='add', damped_trend=True)
ets_trend.fit(y, fh = fh_2days)
y_pred_2days = ets_trend.predict(fh = fh_2days)
data = data.assign(ets=y_pred)
data.loc[y_pred_2days.index, 'ets'] = y_pred_2days

# %%

# ExponentialSmoothing(trend="add", damped_trend=True, remove_bias=True)
y_pred, y_test, smape_test, rmse_test, mae_test, df = cv_evaluate(forecaster=ExponentialSmoothing(trend="add", damped_trend=True, remove_bias=True), y=y, cv=cv, X=None)
print(f"ExponentialSmoothing as detrender n_val={len(y_pred)} SMAPE={smape_test:.4f} RMSE={rmse_test:.2f} MAE={mae_test:.2f}")

data = data.assign(exp=y_pred)
# %%
fh_2days = np.arange(1, 193)
exp_trend = ExponentialSmoothing(trend="add", damped_trend=True, remove_bias=True)
exp_trend.fit(y, fh = fh_2days)
y_pred_2days = exp_trend.predict(fh = fh_2days)
data.loc[y_pred_2days.index, 'exp'] = y_pred_2days


# %%
'''
STL daily forecaster
'''
y = data.Value.resample('D').mean().dropna()
step = 1
fh = [2]
cv = ExpandingWindowSplitter(initial_window=step*(arima_init_steps+1), fh=fh, step_length=step)

n_splits = cv.get_n_splits(y)
print(f"Number of Folds = {n_splits}")

train_windows, test_windows = get_windows(y, cv)
# plot_windows(y, train_windows, test_windows)

print(y[test_windows[-1]].index)

# %%
y_pred, y_test, smape_test, rmse_test, mae_test, df = cv_evaluate(forecaster=STLForecaster(sp = 7, robust=True), y=y, cv=cv, X=None)

# %%
fh_2days = [1,2]
stl_trend = STLForecaster(sp = 7, robust=True)
stl_trend.fit(y, fh = fh_2days)
y_pred_2days = stl_trend.predict(fh = fh_2days)

data = data.assign(stl=y_pred)
data.loc[y_pred_2days.index, 'stl'] = y_pred_2days
data['stl'] = data['stl'].ffill()
# %%
eval_df = data[['stl', 'Value']].dropna()
y_true = eval_df['Value']
y_pred = eval_df['stl']
# %%
smape_stl = smape(y_true, y_pred)
rmse_stl = rmse(y_true, y_pred)
mae_stl = mae(y_true, y_pred)

print(f"STL(daily, period=7) as a detrender n_val={len(y_pred)} SMAPE={smape_stl:.4f} RMSE={rmse_stl:.2f} MAE={mae_stl:.2f}")

# %%
'''
ARIMA (2,1,1) 
STL(sp=7) for daily
'''
data['arima_prev'] = None # 1 step arima
data['arima_current'] = None # 2nd step arima
data['stl_trend_lag2'] = None
data['stl_seasonal_lag2'] = None
data['stl_resid_lag2'] = None
train_until = value_daily.index[arima_init_steps]
start_from = train_until
stl_transf = STLTransformer(sp = 7, robust=True, return_components = True)

for idx, date in enumerate(sorted(value_daily[(arima_init_steps+1):].index)):
        date_ahead = date + timedelta(days=1)
        date_before = date - timedelta(days=1)
        print(f'Current day={date.date()} predicting day={date_ahead.date()} with last training date={train_until.date()}')
        arima_daily = sm.tsa.arima.ARIMA(value_daily[:train_until], order=(2, 1, 1), trend=None).fit()
        steps = arima_daily.forecast(steps = 2)

        stl_df = STLTransformer(sp = 7, robust=True, return_components = True).fit_transform(value_daily[:train_until])

        print(f'- STL lag2 trend={stl_df.loc[train_until].trend} seasonal={stl_df.loc[train_until].seasonal}')

        if date_ahead <= value_daily.index[-1]:
                print(f'-inserting arima forecasts, stl components, into {date_ahead.date()} trained on data until {train_until.date()}\n')
                data.loc[date_ahead, 'arima_prev'] = steps[0]
                data.loc[date_ahead, 'arima_current'] = steps[1]

                data.loc[date_ahead, 'stl_trend_lag2'] = stl_df.loc[train_until]['trend']
                data.loc[date_ahead, 'stl_seasonal_lag2'] = stl_df.loc[train_until]['seasonal']
                data.loc[date_ahead, 'stl_resid_lag2'] = stl_df.loc[train_until]['resid']

                train_until = date
        else:
                print(f"Last forward filled day at date={date.date()}")
# %%
'''
NOTE: 2 step prediction
data at start_from should have arima none
data at start_from + 1 should have arima none as well 
'''
assert data.loc[start_from].arima_current is None, "Arima should be two step predicted"
assert data.loc[start_from + timedelta(days = 1)].arima_current is None, "Arima should be two step predicted"

assert data.loc[start_from + timedelta(days = 2)].arima_current is not None, "Arima should be calculated here"

# %%
'''
Forward fill both arimas
'''
data.loc[:, ['arima_current', 'arima_prev']] = data[['arima_current', 'arima_prev']].ffill()

# %%
'''
Forward fill daily stl lagged vars

'''
data.loc[:, ['stl_trend_lag2', 'stl_seasonal_lag2', 'stl_resid_lag2']] = data[['stl_trend_lag2', 'stl_seasonal_lag2', 'stl_resid_lag2']].ffill()

print(data[['stl_trend_lag2', 'stl_seasonal_lag2', 'stl_resid_lag2']].loc[start_from + timedelta(days=2)].head())
# %%
'''
Sanity checks
'''

print("Values arima_current and arima_prev should be empty")
print(data.loc[(start_from + timedelta(days = 1)).date().isoformat()][['arima_current', 'arima_prev']])

print("Values arima_current and arima_prev should be forward filled")
print(data.loc[(start_from + timedelta(days = 2)).date().isoformat()][['arima_current', 'arima_prev']])

print(f"STL values should start from {start_from + timedelta(days=2)}")
print(data[['stl_trend_lag2', 'stl_seasonal_lag2', 'stl_resid_lag2']].loc[(start_from + timedelta(days=2)).date().isoformat()])
print(data[['stl_trend_lag2', 'stl_seasonal_lag2', 'stl_resid_lag2']].loc[(start_from + timedelta(days=10)).date().isoformat()])


# %%
'''
True daily plot
'''
tsplot(data_daily.Value.dropna(), label = "True daily", filepath=plots_path / 'daily.png')

# %%
'''
Trend forecasts comparison
- daily forecasts are forward filled to 15 min steps.
'''
data['Value_daily'] = value_daily
data['Value_daily'] = data['Value_daily'].ffill()

trend_eval_plot(data['Value'], data['Value_daily'], label="True daily  - trend forecast for 15 min steps", filepath=plots_path / 'true_daily_detr.png')
del data['Value_daily'] #NOTE IMPORTANT


trend_eval_plot(data['Value'], data['arima_current'], label="Arima(daily, 2,1,1)  - trend forecast", filepath=plots_path / 'arima_detr.png')
trend_eval_plot(data['Value'], data['ets'], label="AutoETS(15min) - trend forecast", filepath=plots_path / 'ets_detr.png')
trend_eval_plot(data['Value'], data['exp'], label="ExponentialSmoothing(15min) - trend forecast", filepath=plots_path / 'exp_detr.png')
trend_eval_plot(data['Value'], data['stl'], label="STL(daily, period=7) - trend forecast", filepath=plots_path / 'stl_detr.png')

# %%
trend_eval_plot(data['Value'], data['arima_current'], label="Arima(daily, 2,1,1)  - trend forecast", compare=True,
                filepath=plots_path / 'arima_detr_comparison.png'
                )

# %%
data.to_csv(product['data'])
