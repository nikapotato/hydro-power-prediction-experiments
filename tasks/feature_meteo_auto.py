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

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from hydro_timeseries.util import load_timeseries_csv, add_mean_vars, generate_cyclical_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction.settings import TimeBasedFCParameters, ComprehensiveFCParameters
from hydro_timeseries.variables import Variables
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
from tsfresh import extract_features, select_features
import numpy as np
from datetime import timedelta
from pathlib import Path

upstream = None
file_path = None
random_seed = None
valid_from = None
product = None
minute_timeshift = None
daily_timeshift = None
features_daily_value = None

# %% tags=["injected-parameters"]
# Parameters
file_path = "/home/m/repo/hydro-power-prediction/data/data_mve_production_forecast_test.csv"
random_seed = 1
valid_from = "2021-11-01"
minute_timeshift = 96
daily_timeshift = 14
done = True
features_daily = "backup/critical/features_daily.csv"
features_minute = "backup/critical/features_minute.csv"
features_test = "backup/critical/features_test.csv"
upstream = {"run-tests": {"nb": "/home/m/repo/hydro-power-prediction/reports/run_tests.ipynb", "data": "/home/m/repo/hydro-power-prediction/products/data/data_raw_tested.csv"}}
product = {"nb": "/home/m/repo/hydro-power-prediction/reports/feature_meteo_auto.ipynb", "features-daily": "/home/m/repo/hydro-power-prediction/products/data/features_daily.csv", "features-minute": "/home/m/repo/hydro-power-prediction/products/data/features_minute.csv", "features-hand": "/home/m/repo/hydro-power-prediction/products/data/features_hand.csv", "features-test": "/home/m/repo/hydro-power-prediction/products/data/features_test.csv"}


# %%
'''
Load all data
train section is loaded for feature selection of generated features. 
'''


data = load_timeseries_csv(file_path)

# get train_until index
train_until = pd.to_datetime(valid_from) + timedelta(days=-1)
train = data.loc[:train_until]
# data = pd.read_csv(file_path, parse_dates=['Date_Time', 'Date'], index_col='Date_Time')

# %%
'''
Daily value features
'''
features_daily_value = "backup/critical/features_daily_value.csv"
data_daily = data[['Value', 'Date']].resample('D').mean()
data_daily['val_sh2'] = data_daily['Value'].shift(2)
data_daily.loc[:, 'Date'] = data_daily.index.date
data_daily['idx'] = 1
data_generate = data_daily[['idx', 'Date', 'val_sh2']].dropna()

data_rolled_daily = roll_time_series(data_generate, column_id="idx", column_sort="Date",
                             min_timeshift=7, max_timeshift=60, rolling_direction=1)

print(data_rolled_daily[data_rolled_daily.id == (1, pd.to_datetime('2022-01-03'))])
# %%
target_features_lagged = extract_features(data_rolled_daily,
                         column_id="id",
                         column_sort="Date", column_value='val_sh2', show_warnings=True,
                        default_fc_parameters=ComprehensiveFCParameters())

target_features_lagged.set_index(target_features_lagged.index.map(lambda x: x[1]).rename('index'), drop=True, inplace=True)
target_features_lagged.index = pd.to_datetime(target_features_lagged.index)
target_features_dropped = target_features_lagged.dropna(axis=1)
target_features_dropped.loc[:,'Value'] = data_daily.loc[:,'Value']
target_features_dropped = target_features_dropped.dropna(axis=0)


target_features_selected = select_features(X = target_features_dropped[target_features_dropped.columns.difference(['Value'])], y = target_features_dropped['Value'])


daily_target_vars = pd.concat([data.Value, target_features_lagged[target_features_selected.columns]], axis = 1)
daily_target_vars.loc[:, daily_target_vars.columns.difference(['Value'])] = daily_target_vars.loc[:, daily_target_vars.columns.difference(['Value'])].ffill()

daily_target_vars.to_csv(features_daily_value)
# %%
'''
Add means across different meteo station
'''
data = add_mean_vars(data)
# %%
# start with daily split, weekly windows

data_generate = data[Variables.meteo_means_i]
data_daily = data_generate.resample('D').mean()
data_daily.loc[:, 'timestamp'] = data_daily.index
data_daily['idx'] = 1

# %%
'''
Roll weekly
'''
data_rolled_daily = roll_time_series(data_daily, column_id="idx", column_sort="timestamp",
                             max_timeshift=daily_timeshift, min_timeshift=daily_timeshift, rolling_direction=1)


print('Daily rolled sample of a single var')
print(data_rolled_daily[data_rolled_daily.id == (1, pd.to_datetime('2020-12-08 00:00:00+00:00'))][Variables.meteo_means_i[0]])

data_rolled_daily.set_index(data_rolled_daily.timestamp.rename('index'), inplace=True)

# %% Test one meteo var
print("Test variable extraction for one var with warnings, log(0) warnings are fine")

var0 = Variables.meteo_means_i[0]
var_df = extract_features(data_rolled_daily,
                         column_id="id",
                         column_sort="timestamp", column_value=var0, show_warnings=True)
var_df.set_index(var_df.index.map(lambda x: x[1]).rename('index'), drop=True, inplace=True)

# %%
'''
forward fill the AVGS
'''
tst = pd.concat([data.Value, var_df], axis = 1)

# select 1st calculated vars index
tst = tst.loc[var_df.index[0]:]
tst.loc[:,tst.columns.difference(['Value'])] = tst.loc[:,tst.columns.difference(['Value'])].ffill()

print("Value should NOT be forward filled")
print(tst.tail())

# %%
'''
Do feature selection on training dataset
'''

tst_train = tst[:train_until]
tst_train = tst_train.dropna(axis = 1)

X_train_selected = select_features(X = tst_train[tst_train.columns.difference(['Value'])], y = tst_train.Value)

selected_features = X_train_selected.columns

tst = tst[['Value'] + selected_features.to_list()]

print("Test dataframe should have target - NaN alues at the end not forward filled, all else filled")
print(tst.tail())
# %%
tst.to_csv(product['features-test'])

# %%
'''
Generate values for all meteo means
'''
print("Generating weekly features for meteo means")
daily_dfs = []
for var in Variables.meteo_means_i:
    print(var)
    var_df = extract_features(data_rolled_daily,
                         column_id="id",
                         column_sort="timestamp", column_value=var, show_warnings=False)

    var_df.set_index(var_df.index.map(lambda x: x[1]).rename('index'), drop=True, inplace=True)

    print(f'Calculated variables for {var}, number of vars {len(var_df.columns)}, number of nan containing vars {np.sum(var_df.isna().any())}')
    daily_dfs.append(var_df)

# %%
daily_df = pd.concat(daily_dfs, axis = 1)
# daily_df = daily_df.set_index(daily_df.index.map(lambda x: x[1]).rename('index'), drop=True)


# %%
'''
Feature selection on all meteo
1. Forward fill the daily vars to all the 15 minute steps
2. Feature select agains the 15 minute Value
'''
daily_vars = pd.concat([data.Value, daily_df], axis = 1)

# select 1st calculated index and select all the following steps.
daily_vars = daily_vars.loc[daily_df.index[0]:]

#forward fill on all non Value vars
daily_vars.loc[:, daily_vars.columns.difference(['Value'])] = daily_vars.loc[:, daily_vars.columns.difference(['Value'])].ffill()

print("Value should not be FW filled, all else yes")
print("Data should have 15 minute steps")
print(daily_vars.tail())

# %%
'''
Do feature selection against 15 min Value on training data only
'''
train_daily = daily_vars[:train_until]
train_daily = train_daily.dropna(axis = 1)

X_train_selected_all_meteo = select_features(X = train_daily[train_daily.columns.difference(['Value'])], y = train_daily.Value)

selected_features_all_meteo = X_train_selected_all_meteo.columns

daily_vars_out = daily_vars[['Value'] + selected_features_all_meteo.to_list()]

print("Value should not be FW filled, all else yes")
print("Data should have 15 minute steps")
print(daily_vars_out.tail())

# %%
print(f"Storing daily data to {product['features-daily']}")
daily_vars_out.to_csv(product['features-daily'])

# %%
# clean ram
del var_df
del daily_df
del daily_vars_out
del data_rolled_daily
# %%
'''
15 minute split, 192x15 daily avg
Prepare for rolling
'''
data_minute = data[Variables.meteo_means_i]
data_minute.loc[:, 'timestamp'] = data_minute.index
data_minute.loc[:, 'idx'] = 1

print("Data should not include value")
print(data_minute.tail())
assert 'Value' not in data_minute.columns, "Value should not be in rolling dataset"
# %%
data_rolled_min = roll_time_series(data_minute, column_id="idx", column_sort="timestamp",
                             max_timeshift=minute_timeshift, min_timeshift=minute_timeshift)

# %%
print('Daily rolled minute sample of a single var')
print(data_rolled_min.head())
print("Tail")
print(data_rolled_min.tail())
print("Sample")
print(data_rolled_min[data_rolled_min.id == (1, pd.to_datetime('2020-12-08 00:00:00+00:00'))][Variables.meteo_means_i[0]])

data_rolled_min.set_index(data_rolled_min.timestamp.rename('index'), inplace=True)
# %%
minute_dfs = []
for var in Variables.meteo_means_i:
    var_df = extract_features(data_rolled_min,
                         column_id="id", column_sort="timestamp", column_value=var, show_warnings=False)

    print(f'Calculated variables for {var}, number of vars {len(var_df.columns)}, number of nan containing vars {np.sum(var_df.isna().any())}')

    var_df.set_index(var_df.index.map(lambda x: x[1]).rename('index'), drop=True, inplace=True)
    minute_dfs.append(var_df)

# %%
minute_df = pd.concat(minute_dfs, axis = 1)

# %%
'''
Feature selection on all meteo, 15 minute steps. 
1. NO NEED TO FORWARD FILL
2. Feature select against the 15 minute Value
'''
print("PRINT PRINT data Value")
print(data.head().Value)
print('---------------------')
minute_vars = pd.concat([data.Value, minute_df], axis = 1)
print(minute_vars.head())

# select 1st calculated index and select all the following steps.
minute_vars = minute_vars.loc[minute_df.index[0]:]

print("Data should have 15 minute steps and NaNs in the last 2 days Value")
print(minute_vars.tail())

print(minute_vars.tail().Value)

# %%
'''
Do feature selection against 15 min Value of minute_vars on training data only
'''
train_until = pd.to_datetime(valid_from) + timedelta(days=-1)
train_minute = minute_vars[:train_until]
print(train_minute.tail())

## NOTE! careful to not drop the Value column due to NaNs in the last two days.
train_minute = train_minute.dropna(axis = 1)
print(train_minute.tail())

# print(list(train_minute.columns))
X_train_selected_all_meteo_min = select_features(X = train_minute[train_minute.columns.difference(['Value'])], y = train_minute.Value)

selected_features_all_meteo_min = X_train_selected_all_meteo_min.columns

minute_vars_out = minute_vars[['Value'] + selected_features_all_meteo_min.to_list()]

print("Value should not be FW filled")
print("Data should have 15 minute steps")
print(minute_vars_out.tail())

# %%
print(f"Storing minute data to {product['features-minute']}")
minute_vars_out.to_csv(product['features-minute'])



